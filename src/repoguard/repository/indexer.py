"""Index orchestration: scan -> parse -> call graph -> passages -> (embeddings) -> JSON.

Public API::

    build_index(repo_path, output_dir, config=None) -> IndexArtifacts
    load_index(index_dir) -> IndexArtifacts

Outputs written to ``output_dir`` (see docs/index_schema.md):

    symbols.json      files, symbols (functions/methods/classes), imports, errors
    callgraph.json    static call edges
    passages.jsonl    retrieval passages, one JSON object per line
    manifest.json     repository provenance, effective config, counts, artefact hashes
    passages.faiss / passages.npy + vector_meta.json   (only when embeddings are enabled)

All outputs are deterministic for a given repository snapshot and configuration:
paths and records are sorted, no timestamps or absolute paths are written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from repoguard.exceptions import IndexingError, IndexNotFoundError
from repoguard.repository.callgraph import build_call_graph
from repoguard.repository.parser import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_GLOBS,
    ParsedFile,
    detect_source_roots,
    parse_file,
    resolve_repo_root,
    scan_repository,
)
from repoguard.repository.passage_builder import PassageConfig, build_passages
from repoguard.schemas import (
    COMPATIBLE_INDEX_VERSIONS,
    INDEX_SCHEMA_VERSION,
    CallEdge,
    FileMetadata,
    FreshnessReport,
    ImportInfo,
    IndexArtifacts,
    IndexIssue,
    Passage,
    RepositoryMetadata,
)

logger = logging.getLogger(__name__)

SYMBOLS_FILE = "symbols.json"
CALLGRAPH_FILE = "callgraph.json"
PASSAGES_FILE = "passages.jsonl"
MANIFEST_FILE = "manifest.json"
BUILD_INFO_FILE = "build_info.json"
"""Non-deterministic build metadata (timestamp, duration). Excluded from artefact hashes."""

INDEXER_VERSION = "2.0"
"""Bump when extraction, call-graph or passage logic changes output for the same input."""

DEFAULT_CONFIG: dict[str, Any] = {
    "exclude_dirs": [],  # added to the built-in exclusions
    "exclude_globs": [],  # fnmatch patterns on repo-relative paths
    "replace_default_excludes": False,
    "include_hidden_dirs": False,  # index .github/, .devcontainer/ ... (CI helpers, not library code)
    "include_files": None,  # explicit list of repo-relative files; None = scan
    "source_roots": "auto",  # import roots stripped from module names; "auto" detects src/
    "passages": {
        "max_lines": 80,
        "overlap_lines": 10,
        "include_module_passages": True,
        "include_class_passages": True,
    },
    "embeddings": {
        "enabled": False,
        "backend": "auto",  # auto | sentence-transformers | hashing
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "hashing_dim": 512,
        "vector_store": "auto",  # auto (faiss if importable) | faiss | numpy (portable)
    },
    "repository": {  # provenance; auto-detected from git when omitted
        "repository_id": None,
        "name": None,
        "source_url": None,
        "commit": None,
        "licence": None,
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in override.items():
            if k not in base:
                raise IndexingError(f"Unknown indexer config key: {k!r}")
            out[k] = merge(base[k], v) if isinstance(base[k], dict) and isinstance(v, dict) else v
        return out

    return merge(DEFAULT_CONFIG, config or {})


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "repository"


def _git(root: Path, *args: str) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )  # fmt: skip
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


_LICENCE_PATTERNS = [
    ("MIT", r"\bMIT License\b|Permission is hereby granted, free of charge"),
    ("Apache-2.0", r"Apache License,?\s+Version 2\.0"),
    ("BSD-3-Clause", r"Neither the name of .* nor the names of its\s+contributors"),
    ("BSD-2-Clause", r"Redistribution and use in source and binary forms"),
    ("GPL-3.0", r"GNU GENERAL PUBLIC LICENSE\s+Version 3"),
    ("GPL-2.0", r"GNU GENERAL PUBLIC LICENSE\s+Version 2"),
    ("LGPL", r"GNU LESSER GENERAL PUBLIC LICENSE"),
    ("MPL-2.0", r"Mozilla Public License,? (v\. |Version )2\.0"),
]


def detect_licence(root: Path) -> str | None:
    """Best-effort SPDX guess from a LICENSE/COPYING file. Always confirm manually."""
    for cand in sorted(root.iterdir()):
        if cand.is_file() and re.match(r"(?i)^(licen[cs]e|copying)", cand.name):
            try:
                text = cand.read_text(encoding="utf-8", errors="replace")[:5000]
            except OSError:
                continue
            for spdx, pattern in _LICENCE_PATTERNS:
                if re.search(pattern, text):
                    return spdx
            return f"unknown ({cand.name})"
    return None


def _dump_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_index_id(repository_id: str, commit: str | None, cfg: dict[str, Any]) -> str:
    """``<repository_id>@<commit12>-<config8>``.

    The config hash covers everything that changes index content (exclusions, passage
    and embedding settings, schema/indexer versions) but not provenance overrides.
    """
    relevant = {k: v for k, v in cfg.items() if k != "repository"}
    payload = json.dumps(
        {"config": relevant, "schema": INDEX_SCHEMA_VERSION, "indexer": INDEXER_VERSION},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{repository_id}@{(commit or 'nocommit')[:12]}-{digest}"


def tool_versions() -> dict[str, str]:
    """Versions that can change index content; recorded in the manifest."""
    import numpy

    from repoguard import __version__

    versions = {
        "repoguard": __version__,
        "index_schema": INDEX_SCHEMA_VERSION,
        "indexer": INDEXER_VERSION,
        "python": platform.python_version(),
        "parser": f"ast ({platform.python_implementation()} {platform.python_version()})",
        "pydantic": _pkg_version("pydantic"),
        "numpy": numpy.__version__,
    }
    return versions


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_index(
    repo_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: dict[str, Any] | None = None,
) -> IndexArtifacts:
    """Index a Python repository without executing any of its code.

    Raises :class:`RepositoryNotFoundError` if ``repo_path`` is not a directory.
    Problems with individual files are recorded in ``IndexArtifacts.errors`` and the
    file is marked ``parse_status="error"``; the rest of the repository is still indexed.
    """
    started = time.perf_counter()
    cfg = merge_config(config)
    root = resolve_repo_root(repo_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exclude_dirs = set() if cfg["replace_default_excludes"] else set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs |= set(cfg["exclude_dirs"])
    exclude_globs = [
        *([] if cfg["replace_default_excludes"] else DEFAULT_EXCLUDE_GLOBS),
        *cfg["exclude_globs"],
    ]

    # 1. scan
    scan_stats: dict[str, int] = {}
    rel_paths, issues = scan_repository(
        root,
        frozenset(exclude_dirs),
        exclude_globs,
        include_files=cfg["include_files"],
        stats=scan_stats,
        include_hidden_dirs=cfg["include_hidden_dirs"],
    )
    logger.info("Found %d Python files under %s", len(rel_paths), root)

    # 2. parse each file independently
    source_roots = (
        detect_source_roots(root) if cfg["source_roots"] == "auto" else list(cfg["source_roots"])
    )
    cfg["source_roots"] = source_roots  # record the effective value in the manifest
    parsed: list[ParsedFile] = []
    for rel in rel_paths:
        pf = parse_file(root, rel, source_roots)
        issues.extend(pf.issues)
        parsed.append(pf)

    # 3. call graph (needs the global symbol table)
    edges, cg_issues = build_call_graph(parsed)
    issues.extend(cg_issues)

    # 4. passages
    pcfg = PassageConfig(**cfg["passages"])
    passages: list[Passage] = []
    for pf in parsed:
        try:
            passages.extend(build_passages(pf, pcfg))
        except Exception as exc:  # never let one file abort the index
            issues.append(
                IndexIssue(
                    file=pf.metadata.path, stage="passages",
                    error_type=type(exc).__name__, message=str(exc),
                )
            )  # fmt: skip

    # 5. assemble + sort
    files: list[FileMetadata] = sorted((pf.metadata for pf in parsed), key=lambda f: f.path)
    symbols = sorted(
        [s for pf in parsed for s in (*pf.functions, *pf.classes)],
        key=lambda s: (s.file, s.line, s.col, s.qualified_name, s.symbol_id),
    )
    imports: list[ImportInfo] = sorted(
        (i for pf in parsed for i in pf.imports),
        key=lambda i: (i.file, i.line, i.module or "", i.name or "", i.alias or ""),
    )
    issues.sort(key=lambda e: (e.file, e.stage, e.line or 0, e.error_type, e.message))
    passages.sort(key=lambda p: (p.file_path, p.start_line, p.passage_type, p.passage_id))
    for p in passages:
        p.content_sha256 = text_sha256(p.content)

    failed = sum(1 for f in files if f.parse_status == "error")
    repo_cfg = cfg["repository"]
    name = repo_cfg["name"] or root.name
    repository = RepositoryMetadata(
        repository_id=repo_cfg["repository_id"] or slugify(name),
        name=name,
        repo_path=str(repo_path),
        source_url=repo_cfg["source_url"] or _git(root, "config", "--get", "remote.origin.url"),
        commit=repo_cfg["commit"] or _git(root, "rev-parse", "HEAD"),
        licence=repo_cfg["licence"] or detect_licence(root),
        status="failed" if files and failed == len(files) else "ready",
        total_files=len(files),
        indexed_files=len(files) - failed,
        failed_files=failed,
        total_symbols=len(symbols),
    )
    index_id = compute_index_id(repository.repository_id, repository.commit, cfg)
    versions = tool_versions()

    # 6. write JSON artefacts
    symbols_path = out / SYMBOLS_FILE
    callgraph_path = out / CALLGRAPH_FILE
    passages_path = out / PASSAGES_FILE
    manifest_path = out / MANIFEST_FILE

    _dump_json(
        symbols_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_id": index_id,
            "repository": repository.model_dump(mode="json"),
            "files": [f.model_dump(mode="json") for f in files],
            "symbols": [s.model_dump(mode="json") for s in symbols],
            "imports": [i.model_dump(mode="json") for i in imports],
            "errors": [e.model_dump(mode="json") for e in issues],
        },
    )
    _dump_json(
        callgraph_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_id": index_id,
            "repository_id": repository.repository_id,
            "edges": [e.model_dump(mode="json") for e in edges],
        },
    )
    _atomic_write(
        passages_path,
        "".join(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) + "\n" for p in passages),
    )

    artifacts = IndexArtifacts(
        index_id=index_id,
        tool_versions=versions,
        output_dir=str(out),
        symbols_path=str(symbols_path),
        callgraph_path=str(callgraph_path),
        passages_path=str(passages_path),
        manifest_path=str(manifest_path),
        repository=repository,
        files=files,
        symbols=symbols,
        imports=imports,
        call_edges=edges,
        passages=passages,
        errors=issues,
    )

    # 7. optional embeddings + vector index
    embedding_info: dict[str, Any] | None = None
    from repoguard.retrieval.search import VECTOR_FILES, build_vector_index

    if cfg["embeddings"]["enabled"]:
        embedding_info = build_vector_index(artifacts, cfg["embeddings"])
    else:
        for stale in VECTOR_FILES:  # never leave vectors from an older run next to new passages
            (out / stale).unlink(missing_ok=True)

    # 8. manifest
    resolution_counts = Counter(e.resolution.value for e in edges)
    artefact_hashes = {p.name: _sha256(p) for p in (symbols_path, callgraph_path, passages_path)}
    if embedding_info:
        for fname in embedding_info["files"]:
            artefact_hashes[fname] = _sha256(out / fname)
        versions = {**versions, **embedding_info.get("versions", {})}
        artifacts.tool_versions = versions
    _dump_json(
        manifest_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_id": index_id,
            "repository": repository.model_dump(mode="json"),
            "tool_versions": versions,
            "config": cfg,
            "counts": {
                "files": len(files),
                "files_failed": failed,
                "symbols": len(symbols),
                "functions": sum(1 for s in symbols if s.type == "function"),
                "methods": sum(1 for s in symbols if s.type == "method"),
                "classes": sum(1 for s in symbols if s.type == "class"),
                "imports": len(imports),
                "call_edges": len(edges),
                "call_edges_by_resolution": dict(sorted(resolution_counts.items())),
                "passages": len(passages),
                "errors": len(issues),
                "unsupported_files_skipped": scan_stats.get("unsupported_files", 0),
            },
            "embeddings": embedding_info,
            "artefacts": dict(sorted(artefact_hashes.items())),
        },
    )
    # build_info.json is the only non-deterministic output (timestamps)
    _dump_json(
        out / BUILD_INFO_FILE,
        {
            "index_id": index_id,
            "built_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "duration_s": round(time.perf_counter() - started, 3),
            "platform": platform.platform(terse=True),
        },
    )
    logger.info(
        "Indexed %s: %d files (%d failed), %d symbols, %d call edges, %d passages",
        repository.name, len(files), failed, len(symbols), len(edges), len(passages),
    )  # fmt: skip
    return artifacts


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def load_index(index_dir: str | os.PathLike[str]) -> IndexArtifacts:
    """Load a previously built index directory back into an :class:`IndexArtifacts`."""
    d = Path(index_dir)
    paths = {
        name: d / name for name in (SYMBOLS_FILE, CALLGRAPH_FILE, PASSAGES_FILE, MANIFEST_FILE)
    }
    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        raise IndexNotFoundError(f"Index at {d} is missing: {', '.join(missing)}")

    symbols_doc = json.loads(paths[SYMBOLS_FILE].read_text(encoding="utf-8"))
    callgraph_doc = json.loads(paths[CALLGRAPH_FILE].read_text(encoding="utf-8"))
    manifest = json.loads(paths[MANIFEST_FILE].read_text(encoding="utf-8"))
    version = symbols_doc.get("schema_version")
    if version not in COMPATIBLE_INDEX_VERSIONS:
        raise IndexNotFoundError(
            f"Index schema version {version!r} is not supported "
            f"(expected one of {COMPATIBLE_INDEX_VERSIONS}); rebuild the index."
        )
    index_id = manifest.get("index_id", "")
    for name, doc in ((SYMBOLS_FILE, symbols_doc), (CALLGRAPH_FILE, callgraph_doc)):
        if doc.get("index_id") != index_id:
            raise IndexNotFoundError(
                f"{name} belongs to index {doc.get('index_id')!r} but manifest is {index_id!r}; "
                "the index directory mixes two builds — rebuild it."
            )
    passages = [
        Passage.model_validate_json(line)
        for line in paths[PASSAGES_FILE].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    emb = manifest.get("embeddings") or {}
    return IndexArtifacts(
        index_id=index_id,
        tool_versions=manifest.get("tool_versions", {}),
        output_dir=str(d),
        symbols_path=str(paths[SYMBOLS_FILE]),
        callgraph_path=str(paths[CALLGRAPH_FILE]),
        passages_path=str(paths[PASSAGES_FILE]),
        manifest_path=str(paths[MANIFEST_FILE]),
        vector_index_path=str(d / emb["index_file"]) if emb.get("index_file") else None,
        vector_ids_path=str(d / emb["meta_file"]) if emb.get("meta_file") else None,
        repository=RepositoryMetadata.model_validate(symbols_doc["repository"]),
        files=symbols_doc["files"],
        symbols=symbols_doc["symbols"],
        imports=symbols_doc["imports"],
        call_edges=[CallEdge.model_validate(e) for e in callgraph_doc["edges"]],
        passages=passages,
        errors=symbols_doc["errors"],
    )


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


def check_freshness(
    index: IndexArtifacts, repo_path: str | os.PathLike[str] | None = None
) -> FreshnessReport:
    """Compare an index with a repository checkout (default: the path it was built from).

    Re-scans with the index's own config and compares file sets and sha256 hashes,
    plus the git commit when available. Nothing is parsed or executed.
    """
    manifest = json.loads(Path(index.manifest_path).read_text(encoding="utf-8"))
    cfg = merge_config(manifest.get("config"))
    root = resolve_repo_root(repo_path if repo_path is not None else index.repository.repo_path)
    exclude_dirs = set() if cfg["replace_default_excludes"] else set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs |= set(cfg["exclude_dirs"])
    exclude_globs = [
        *([] if cfg["replace_default_excludes"] else DEFAULT_EXCLUDE_GLOBS),
        *cfg["exclude_globs"],
    ]
    current, _ = scan_repository(
        root,
        frozenset(exclude_dirs),
        exclude_globs,
        include_files=cfg["include_files"],
        include_hidden_dirs=cfg["include_hidden_dirs"],
    )
    indexed = {f.path: f.sha256 for f in index.files}
    current_set = set(current)
    changed = sorted(
        p for p in current_set & indexed.keys() if indexed[p] and _sha256(root / p) != indexed[p]
    )
    added = sorted(current_set - indexed.keys())
    removed = sorted(indexed.keys() - current_set)
    head = _git(root, "rev-parse", "HEAD")
    commit_changed = bool(head and index.repository.commit and head != index.repository.commit)
    return FreshnessReport(
        index_id=index.index_id,
        stale=bool(changed or added or removed or commit_changed),
        indexed_commit=index.repository.commit,
        current_commit=head,
        commit_changed=commit_changed,
        changed_files=changed,
        added_files=added,
        removed_files=removed,
    )
