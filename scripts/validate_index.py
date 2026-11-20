#!/usr/bin/env python
"""Cross-check a built index against the repository source.

The checks deliberately use :mod:`tokenize` and raw text instead of :mod:`ast`, so they
are independent of the indexer's own extraction logic:

1. definitions  — every ``def``/``class`` token in the source is in symbols.json at the
                  same file+line, and every symbol corresponds to such a token
                  (no missing, no phantom definitions)
2. lines        — 1 <= line <= end_line <= file length; end_line is not blank
3. parameters   — every recorded parameter name appears as a NAME token in the signature
4. call edges   — the callee name appears at/after the recorded line; callers and
                  resolved callees are real symbols
5. passages     — function/method passages are exact source slices; ids are unique
6. freshness    — file hashes still match (index is not stale)

    python scripts/validate_index.py --repo data/repos/schedule --index data/indexes/schedule
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoguard.repository import load_index  # noqa: E402
from repoguard.schemas import MODULE_SCOPE  # noqa: E402

RESOLVED = {"local", "imported", "method", "module_attribute"}


def def_tokens(source: str) -> set[tuple[int, str]]:
    out = set()
    toks = [
        t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.NAME
    ]
    for a, b in zip(toks, toks[1:], strict=False):
        if a.string in ("def", "class"):
            out.add((a.start[0], b.string))
    return out


def signature_names(source_lines: list[str], line: int) -> set[str]:
    """NAME tokens from the `def` line up to the colon that ends the signature."""
    text = "\n".join(source_lines[line - 1 : line + 40])
    names: set[str] = set()
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.OP:
                if tok.string in "([{":
                    depth += 1
                elif tok.string in ")]}":
                    depth -= 1
                elif tok.string == ":" and depth == 0:
                    break
            elif tok.type == tokenize.NAME:
                names.add(tok.string)
    except (tokenize.TokenError, IndentationError):
        pass
    return names


def validate(repo: Path, index_dir: Path) -> dict:
    idx = load_index(index_dir)
    failures: dict[str, list[str]] = defaultdict(list)
    stats: dict[str, int] = defaultdict(int)

    sources: dict[str, str] = {}
    for f in idx.files:
        path = repo / f.path
        if not path.is_file():
            failures["freshness"].append(f"{f.path}: file no longer exists")
            continue
        raw = path.read_bytes()
        if f.sha256 and hashlib.sha256(raw).hexdigest() != f.sha256:
            failures["freshness"].append(f"{f.path}: sha256 changed since indexing")
        if f.parse_status == "ok":
            with tokenize.open(path) as fh:
                sources[f.path] = fh.read()

    symbol_qualnames = {s.qualified_name for s in idx.symbols}
    by_file: dict[str, list] = defaultdict(list)
    for s in idx.symbols:
        by_file[s.file].append(s)

    # 1 + 2 + 3
    for path, src in sources.items():
        lines = src.splitlines()
        expected = def_tokens(src)
        got = {(s.line, s.name) for s in by_file[path]}
        for line, name in sorted(expected - got):
            failures["definitions"].append(f"{path}:{line} `{name}` in source but not in index")
        for line, name in sorted(got - expected):
            failures["definitions"].append(
                f"{path}:{line} `{name}` in index but not in source (phantom)"
            )
        stats["definitions_checked"] += len(expected)

        for s in by_file[path]:
            if not (1 <= s.line <= s.end_line <= len(lines)):
                failures["lines"].append(
                    f"{s.symbol_id}: bad range {s.line}-{s.end_line} (file has {len(lines)})"
                )
            elif not lines[s.end_line - 1].strip():
                failures["lines"].append(f"{s.symbol_id}: end_line {s.end_line} is blank")
            stats["ranges_checked"] += 1
            if s.type in ("function", "method"):
                sig = signature_names(lines, s.line)
                for p in s.parameters:
                    stats["parameters_checked"] += 1
                    if p not in sig:
                        failures["parameters"].append(
                            f"{s.symbol_id}: parameter `{p}` not in signature"
                        )

    # 4
    for e in idx.call_edges:
        stats["edges_checked"] += 1
        src = sources.get(e.file)
        if src is None:
            failures["call_edges"].append(f"{e.file}:{e.line} edge from unparsed file")
            continue
        window = "\n".join(src.splitlines()[e.line - 1 : e.line + 10])
        if e.callee != "<dynamic>" and e.callee not in window:
            failures["call_edges"].append(
                f"{e.file}:{e.line} callee `{e.callee}` not found near line"
            )
        if e.caller_name != MODULE_SCOPE and e.caller not in symbol_qualnames:
            failures["call_edges"].append(f"{e.file}:{e.line} caller `{e.caller}` is not a symbol")
        if e.resolution.value in RESOLVED and e.callee_qualified not in symbol_qualnames:
            failures["call_edges"].append(
                f"{e.file}:{e.line} {e.resolution.value} callee `{e.callee_qualified}` is not a symbol"
            )

    # 5
    seen: set[str] = set()
    for p in idx.passages:
        stats["passages_checked"] += 1
        if p.passage_id in seen:
            failures["passages"].append(f"duplicate passage id {p.passage_id}")
        seen.add(p.passage_id)
        if p.passage_type in ("function", "method") and p.file_path in sources:
            lines = sources[p.file_path].splitlines()
            if p.content != "\n".join(lines[p.start_line - 1 : p.end_line]):
                failures["passages"].append(f"{p.passage_id}: content differs from source lines")

    return {
        "repository": idx.repository.name,
        "commit": idx.repository.commit,
        "index_errors": [e.model_dump(mode="json") for e in idx.errors],
        "stats": dict(stats),
        "failures": {k: v for k, v in failures.items() if v},
        "passed": not any(failures.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--report", type=Path, help="write the JSON report here")
    args = ap.parse_args(argv)

    report = validate(args.repo, args.index)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Index validation: {report['repository']} @ {(report['commit'] or '?')[:12]}")
    for k, v in report["stats"].items():
        print(f"  {k:22} {v}")
    print(f"  indexer issues         {len(report['index_errors'])}")
    if report["passed"]:
        print("PASS: no mismatches between index and source")
        return 0
    for check, items in report["failures"].items():
        print(f"FAIL [{check}] {len(items)} mismatches")
        for item in items[:20]:
            print(f"    {item}")
        if len(items) > 20:
            print(f"    ... {len(items) - 20} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
