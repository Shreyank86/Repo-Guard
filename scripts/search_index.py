#!/usr/bin/env python
"""Query a built index from the command line (manual retrieval checks).

python scripts/search_index.py --index data/indexes/schedule "how is the next run time computed"
python scripts/search_index.py --index data/indexes/schedule --symbol every
python scripts/search_index.py --index data/indexes/schedule --callers run_pending
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoguard.repository import load_index  # noqa: E402
from repoguard.retrieval import search  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("question", nargs="?", help="natural-language query for passage search")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--symbol", help="look up a symbol by short or qualified name")
    ap.add_argument("--callers", help="list call edges whose callee is this name")
    ap.add_argument("--callees", help="list call edges made by this caller")
    args = ap.parse_args(argv)
    idx = load_index(args.index)

    if args.symbol:
        hits = idx.find_symbols(args.symbol)
        if not hits:
            print(f"No symbol named {args.symbol!r}")
        for s in hits:
            params = f"({', '.join(s.parameters)})" if s.type != "class" else ""
            print(f"{s.type:8} {s.qualified_name}{params}  {s.file}:{s.line}-{s.end_line}")
    for label, name, edges in (
        ("callers of", args.callers, idx.callers_of(args.callers) if args.callers else []),
        ("callees of", args.callees, idx.callees_of(args.callees) if args.callees else []),
    ):
        if name:
            print(f"{label} {name}: {len(edges)}")
            for e in edges:
                print(
                    f"  {e.caller} -> {e.callee_expr}  [{e.resolution.value}: {e.callee_qualified}]  {e.file}:{e.line}"
                )
    if args.question:
        for ev in search(args.question, idx, k=args.k):
            print(
                f"#{ev.rank} {ev.similarity_score:.3f}  {ev.passage_id}  ({ev.file_path}:{ev.start_line}-{ev.end_line})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
