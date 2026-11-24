#!/usr/bin/env python
"""Build the retrieval benchmark (relevance labels derived from index facts).

Case types (per repository, deterministic sample):

* symbol_lookup -- "Where is `Class.method` defined?"            relevant: that symbol
* docstring     -- first docstring sentence, symbol name masked  relevant: that symbol
* callers       -- "Which functions call `name`?"                 relevant: its resolved callers
* baseline_q    -- Member 3's answerable baseline questions, with the target resolved
                   through the index (their hand-written line numbers are audited, not trusted)

Labels come from the static index, so they inherit its limits (e.g. unresolved dynamic
calls are missing from `callers`). ``split_hint`` follows configs/repos.yaml roles; Member 2
owns the final benchmark splits.

    python scripts/build_retrieval_benchmark.py                      # all indexes in data/indexes
    python scripts/build_retrieval_benchmark.py --per-type 40 --repos schedule requests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repoguard.repository import load_index  # noqa: E402
from repoguard.schemas import MODULE_SCOPE, IndexArtifacts  # noqa: E402

GENERATOR_VERSION = "1.0"
_ROLE_TO_SPLIT = {"primary": "dev", "train": "dev", "held_out": "test"}
_SPHINX = re.compile(r":(?:meth|func|class|attr|mod|obj|data|exc|ref):`~?([^`]+)`")


def _is_test(path: str) -> bool:
    parts = path.split("/")
    return (
        any(p in ("test", "tests", "testing") for p in parts[:-1])
        or parts[-1].startswith("test_")
        or parts[-1] == "conftest.py"
    )


def _display(qualified: str, module: str) -> str:
    """``schedule.Job.do`` -> ``Job.do``; ``schedule.every`` -> ``every``."""
    rest = qualified[len(module) + 1 :] if qualified.startswith(module + ".") else qualified
    return rest


def _first_sentence(doc: str) -> str:
    text = _SPHINX.sub(r"\1", doc)
    text = re.split(r"\n\s*\n", text.strip())[0]
    text = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text).strip()


def _rng(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def passage_symbol_ids(idx: IndexArtifacts) -> set[str]:
    return {p.symbol_id for p in idx.passages if p.symbol_id}


def build_cases(idx: IndexArtifacts, per_type: int, split: str) -> list[dict]:
    repo = idx.repository
    has_passage = passage_symbol_ids(idx)
    targets = [s for s in idx.symbols if s.symbol_id in has_passage and not _is_test(s.file)]
    by_display: dict[str, list[str]] = defaultdict(list)
    for s in idx.symbols:
        if s.symbol_id in has_passage:
            by_display[_display(s.qualified_name, s.module)].append(s.symbol_id)

    def case(ctype: str, n: int, query: str, units: list[str], meta: dict) -> dict:
        return {
            "case_id": f"{repo.repository_id}-{ctype}-{n:03d}",
            "repository_id": repo.repository_id,
            "commit": repo.commit,
            "index_id": idx.index_id,
            "case_type": ctype,
            "query": query,
            "relevant_units": sorted(units),
            "split_hint": split,
            "source": "generated-from-index",
            "generator_version": GENERATOR_VERSION,
            **meta,
        }

    cases: list[dict] = []

    # symbol lookup
    pool = sorted(targets, key=lambda s: s.symbol_id)
    _rng(repo.repository_id, "symbol").shuffle(pool)
    for n, s in enumerate(pool[:per_type], start=1):
        disp = _display(s.qualified_name, s.module)
        kind = "class" if s.type == "class" else "method" if s.type == "method" else "function"
        cases.append(
            case(
                "symbol_lookup",
                n,
                f"Where is the {kind} `{disp}` defined?",
                by_display[disp],
                {"target": s.qualified_name},
            )
        )

    # docstring (name masked)
    seen: set[str] = set()
    pool = sorted((s for s in targets if s.docstring), key=lambda s: s.symbol_id)
    _rng(repo.repository_id, "doc").shuffle(pool)
    n = 0
    for s in pool:
        sent = _first_sentence(s.docstring or "")
        masked = re.sub(rf"`?\b{re.escape(s.name)}\b(\(\))?`?", "this", sent)
        words = masked.split()
        if not 5 <= len(words) <= 40 or masked.lower() in seen:
            continue
        seen.add(masked.lower())
        n += 1
        cases.append(case("docstring", n, masked, [s.symbol_id], {"target": s.qualified_name}))
        if n >= per_type:
            break

    # callers
    callers_of: dict[str, set[str]] = defaultdict(set)
    sym_by_q = {s.qualified_name: s for s in idx.symbols}
    for e in idx.call_edges:
        if e.callee_qualified in sym_by_q and e.caller_name != MODULE_SCOPE:
            caller = sym_by_q.get(e.caller)
            if (
                caller is not None
                and caller.symbol_id in has_passage
                and caller.qualified_name != e.callee_qualified
            ):
                callers_of[e.callee_qualified].add(caller.symbol_id)
    pool = sorted(
        (
            sym_by_q[q]
            for q, c in callers_of.items()
            if 1 <= len(c) <= 5 and not _is_test(sym_by_q[q].file)
        ),
        key=lambda s: s.symbol_id,
    )
    _rng(repo.repository_id, "callers").shuffle(pool)
    for n, s in enumerate(pool[:per_type], start=1):
        disp = _display(s.qualified_name, s.module)
        cases.append(
            case(
                "callers",
                n,
                f"Which functions call `{disp}`?",
                sorted(callers_of[s.qualified_name]),
                {"target": s.qualified_name},
            )
        )
    return cases


def resolve_target(idx: IndexArtifacts, target: str | None, target_file: str | None) -> list:
    if not target:
        return []
    hits = [
        s for s in idx.symbols
        if s.name == target or s.qualified_name == target or s.qualified_name.endswith("." + target)
    ]  # fmt: skip
    if target_file:
        in_file = [s for s in hits if s.file == target_file]
        hits = in_file or hits
    return hits


def convert_baseline(
    idx: IndexArtifacts, questions: list[dict], split: str
) -> tuple[list[dict], list[dict]]:
    """Member 3's answerable questions -> cases; plus a grounding audit row per question."""
    has_passage = passage_symbol_ids(idx)
    module_units = {
        p.file_path: p.passage_id.rsplit("@", 1)[0]
        for p in idx.passages
        if p.passage_type == "module"
    }
    cases, audit = [], []
    for q in questions:
        if q.get("is_trap") or q.get("repository", "").split("/")[-1] != idx.repository.name:
            continue
        hits = resolve_target(idx, q.get("target_symbol"), q.get("target_file"))
        units = sorted({h.symbol_id for h in hits if h.symbol_id in has_passage})
        note = ""
        if not units and q.get("target_file") in module_units:
            units = [module_units[q["target_file"]]]  # import questions -> module overview
            note = "target is not a definition; using the module overview passage"
        rl = q.get("relevant_lines") or []
        row = {
            "question_id": q["question_id"],
            "question_type": q["question_type"],
            "target_symbol": q.get("target_symbol"),
            "target_file": q.get("target_file"),
            "relevant_lines": rl,
            "resolved": [f"{h.qualified_name} ({h.file}:{h.line}-{h.end_line})" for h in hits],
            "lines_overlap": None,
            "note": note,
        }
        if hits and len(rl) == 2:
            row["lines_overlap"] = any(rl[0] <= h.end_line and h.line <= rl[1] for h in hits)
        if not hits and not units:
            row["note"] = "target not found in index (e.g. instance attribute or property)"
        audit.append(row)
        if units:
            cases.append({
                "case_id": f"{idx.repository.repository_id}-baseline_q-{q['question_id']}",
                "repository_id": idx.repository.repository_id,
                "commit": idx.repository.commit,
                "index_id": idx.index_id,
                "case_type": "baseline_q",
                "query": q["question"],
                "relevant_units": units,
                "split_hint": split,
                "source": f"data/questions/baseline_questions.jsonl:{q['question_id']}",
                "generator_version": GENERATOR_VERSION,
                "target": q.get("target_symbol"),
            })  # fmt: skip
    return cases, audit


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--indexes", type=Path, default=ROOT / "data" / "indexes")
    ap.add_argument("--repos", nargs="*", help="index directory names (default: all)")
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument(
        "--baseline-questions",
        type=Path,
        default=ROOT / "data" / "questions" / "baseline_questions.jsonl",
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "data" / "retrieval" / "retrieval_benchmark_v1.jsonl"
    )
    ap.add_argument(
        "--audit-out",
        type=Path,
        default=ROOT / "reports" / "retrieval" / "baseline_question_grounding.json",
    )
    args = ap.parse_args(argv)

    roles = {
        e["name"]: e.get("role")
        for e in yaml.safe_load((ROOT / "configs" / "repos.yaml").read_text())["repositories"]
    }
    names = args.repos or sorted(
        p.name for p in args.indexes.iterdir() if (p / "manifest.json").exists()
    )
    questions = (
        [
            json.loads(line)
            for line in args.baseline_questions.read_text().splitlines()
            if line.strip()
        ]
        if args.baseline_questions.exists()
        else []
    )

    all_cases, all_audit = [], []
    for name in names:
        idx = load_index(args.indexes / name)
        split = _ROLE_TO_SPLIT.get(roles.get(name, ""), "dev")
        cases = build_cases(idx, args.per_type, split)
        bq, audit = convert_baseline(idx, questions, split)
        all_cases += cases + bq
        all_audit += audit
        counts = defaultdict(int)
        for c in cases + bq:
            counts[c["case_type"]] += 1
        print(f"[{name}] split={split} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_cases), encoding="utf-8"
    )
    print(f"Wrote {len(all_cases)} cases -> {args.out}")
    if all_audit:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(all_audit, indent=2) + "\n", encoding="utf-8")
        bad = [a for a in all_audit if a["lines_overlap"] is False]
        unresolved = [a for a in all_audit if not a["resolved"]]
        print(
            f"Baseline grounding audit: {len(all_audit)} answerable questions, "
            f"{len(bad)} with relevant_lines not overlapping the real definition, "
            f"{len(unresolved)} targets not resolvable -> {args.audit_out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
