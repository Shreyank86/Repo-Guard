#!/usr/bin/env python
"""Query a built index from the command line (manual retrieval and signal checks).

python scripts/search_index.py --index data/indexes/schedule "how is the next run time computed"
python scripts/search_index.py --index data/indexes/schedule "cancel a job" --exclude-tests --type method -k 3
python scripts/search_index.py --index data/indexes/schedule --signals "Does `run_all_pending` exist?"
python scripts/search_index.py --index data/indexes/schedule --repo-root data/repos/schedule --inspect Scheduler
python scripts/search_index.py --index data/indexes/schedule --symbol every
python scripts/search_index.py --index data/indexes/schedule --callers run_pending
python scripts/search_index.py --index data/indexes/schedule --repo-root data/repos/schedule --freshness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoguard.exceptions import RepoGuardError  # noqa: E402
from repoguard.repository import load_index  # noqa: E402
from repoguard.retrieval import Retriever, SearchFilters, SignalExtractor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("question", nargs="?", help="natural-language query for passage search")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--repo-root",
        type=Path,
        help="repository checkout (needed for --inspect of classes/files, --freshness)",
    )
    ap.add_argument("--path-prefix")
    ap.add_argument("--glob", help="fnmatch pattern on file paths")
    ap.add_argument("--type", action="append", choices=["function", "method", "class", "module"])
    ap.add_argument("--exclude-tests", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument(
        "--dense-only", action="store_true", help="disable symbol boosts (pure vector ranking)"
    )
    ap.add_argument(
        "--signals", action="store_true", help="print the RL free signals for the question"
    )
    ap.add_argument("--inspect", help="evidence id, passage id, symbol or file path to expand")
    ap.add_argument("--symbol", help="look up a symbol by short or qualified name")
    ap.add_argument("--callers", help="list call edges whose callee is this name")
    ap.add_argument("--callees", help="list call edges made by this caller")
    ap.add_argument(
        "--freshness", action="store_true", help="check whether the index matches --repo-root"
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        idx = load_index(args.index)
        r = Retriever(idx, repo_root=args.repo_root)
        out: dict = {"index_id": idx.index_id}

        if args.freshness:
            from repoguard.repository.indexer import check_freshness

            rep = check_freshness(idx, args.repo_root)
            out["freshness"] = rep.model_dump()
            if not args.json:
                print(f"Index {idx.index_id}: {'STALE' if rep.stale else 'fresh'}")
                for label, files in (
                    ("changed", rep.changed_files),
                    ("added", rep.added_files),
                    ("removed", rep.removed_files),
                ):
                    for f in files:
                        print(f"  {label:8} {f}")
                if rep.commit_changed:
                    print(f"  commit   {rep.indexed_commit} -> {rep.current_commit}")
        if args.symbol:
            hits = idx.find_symbols(args.symbol)
            out["symbols"] = [s.model_dump(mode="json") for s in hits]
            if not args.json:
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
                out[label] = [e.model_dump(mode="json") for e in edges]
                if not args.json:
                    print(f"{label} {name}: {len(edges)}")
                    for e in edges:
                        print(
                            f"  {e.caller} -> {e.callee_expr}  [{e.resolution.value}: {e.callee_qualified}]  {e.file}:{e.line}"
                        )
        if args.inspect:
            ev = r.inspect(args.inspect)
            out["inspect"] = ev.model_dump()
            if not args.json:
                print(
                    f"{ev.evidence_id}  {ev.file_path}:{ev.start_line}-{ev.end_line}  ({ev.approx_tokens} tokens)"
                )
                print(ev.snippet)
        if args.question:
            filters = None
            if any([args.path_prefix, args.glob, args.type, args.exclude_tests]):
                filters = SearchFilters(
                    path_prefix=args.path_prefix, path_glob=args.glob,
                    passage_types=args.type, exclude_tests=args.exclude_tests,
                )  # fmt: skip
            boosts = {"definition_boost": 0.0, "mention_boost": 0.0} if args.dense_only else {}
            res = r.search(
                args.question, k=args.k, filters=filters, min_score=args.min_score, **boosts
            )
            out["search"] = res.model_dump()
            if not args.json:
                d = res.diagnostics
                print(f"{d.returned}/{d.k_requested} results from {d.candidates} candidates in {d.latency_ms} ms "
                      f"({d.backend}/{d.vector_store}, {d.retrieved_tokens} tokens, {d.duplicates_removed} duplicates removed)")  # fmt: skip
                for ev in res.evidence:
                    print(
                        f"#{ev.rank} {ev.similarity_score:.3f}  {ev.evidence_id}  {ev.passage_id}  ({ev.file_path}:{ev.start_line}-{ev.end_line})"
                    )
            if args.signals:
                sig = SignalExtractor(r, k=args.k).compute(args.question)
                out["signals"] = sig.model_dump()
                if not args.json:
                    print("free signals:")
                    for k, v in sig.feature_vector().items():
                        print(f"  {k:24} {v:.3f}")
                    print(
                        f"  named_symbols={sig.named_symbols} named_files={sig.named_files} nearest={sig.nearest_symbol} ({sig.nearest_symbol_file})"
                    )
        if args.json:
            print(json.dumps(out, indent=2))
    except RepoGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
