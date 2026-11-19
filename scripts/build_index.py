#!/usr/bin/env python
"""Index a Python repository.

Examples::

    python scripts/build_index.py --repo data/repos/schedule --out data/indexes/schedule
    python scripts/build_index.py --repo data/repos/schedule --out data/indexes/schedule \\
        --config configs/indexer_config.yaml --embedding-backend hashing
    python scripts/build_index.py --repo . --out /tmp/idx --no-embeddings --exclude docs

Exit codes: 0 = indexed (per-file errors are listed but tolerated),
            1 = every file failed or --strict and at least one file failed,
            2 = bad arguments / repository not found.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoguard.exceptions import RepoGuardError  # noqa: E402
from repoguard.repository import build_index  # noqa: E402


def load_yaml_config(path: Path) -> dict:
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return doc.get("indexer", doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", required=True, help="path to the repository checkout")
    ap.add_argument("--out", required=True, help="output index directory")
    ap.add_argument("--config", type=Path, help="YAML config (configs/indexer_config.yaml)")
    emb = ap.add_mutually_exclusive_group()
    emb.add_argument(
        "--embeddings",
        dest="embeddings",
        action="store_true",
        default=None,
        help="build vector index",
    )
    emb.add_argument(
        "--no-embeddings", dest="embeddings", action="store_false", help="skip vector index"
    )
    ap.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "hashing"])
    ap.add_argument(
        "--exclude", action="append", default=[], help="extra directory name to skip (repeatable)"
    )
    ap.add_argument("--name", help="repository display name")
    ap.add_argument("--source-url", help="repository URL for provenance")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any file failed to index")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config: dict = load_yaml_config(args.config) if args.config else {}
    if args.embeddings is not None or args.embedding_backend:
        config.setdefault("embeddings", {})
        if args.embeddings is not None:
            config["embeddings"]["enabled"] = args.embeddings
        if args.embedding_backend:
            config["embeddings"]["backend"] = args.embedding_backend
            config["embeddings"].setdefault("enabled", True)
    if args.exclude:
        config["exclude_dirs"] = [*config.get("exclude_dirs", []), *args.exclude]
    if args.name or args.source_url:
        config.setdefault("repository", {})
        if args.name:
            config["repository"]["name"] = args.name
        if args.source_url:
            config["repository"]["source_url"] = args.source_url

    try:
        art = build_index(args.repo, args.out, config)
    except RepoGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest = json.loads(Path(art.manifest_path).read_text(encoding="utf-8"))
    r = art.repository
    c = manifest["counts"]
    print(
        f"Repository : {r.name}  ({r.source_url or 'no remote'} @ {(r.commit or 'unknown')[:12]}, licence: {r.licence})"
    )
    print(f"Files      : {c['files']} ({c['files_failed']} failed)")
    print(
        f"Symbols    : {c['symbols']}  (functions {c['functions']}, methods {c['methods']}, classes {c['classes']})"
    )
    print(f"Imports    : {c['imports']}")
    print(f"Call edges : {c['call_edges']}  {c['call_edges_by_resolution']}")
    print(f"Passages   : {c['passages']}")
    emb_info = manifest.get("embeddings")
    if emb_info:
        print(
            f"Vectors    : {emb_info['backend']} / {emb_info['model']} (dim {emb_info['dim']}, store {emb_info['vector_store']})"
        )
    if art.errors:
        print(f"Issues     : {len(art.errors)}")
        for e in art.errors:
            loc = f":{e.line}" if e.line else ""
            print(f"  - [{e.stage}] {e.file}{loc}: {e.error_type}: {e.message}")
    print(f"Output     : {art.output_dir}")

    if r.status == "failed" or (args.strict and r.failed_files):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
