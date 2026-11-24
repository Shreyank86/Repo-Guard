"""Run paired direct, retrieval, and verification baselines for Phase 2."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repoguard.evaluation.benchmark import load_benchmark, validate_indexes  # noqa: E402
from repoguard.evaluation.benchmark_scoring import GenerationSettings, Prediction  # noqa: E402
from repoguard.exceptions import RepoGuardError  # noqa: E402
from repoguard.retrieval import Retriever  # noqa: E402
from repoguard.retrieval.metrics import ranked_units  # noqa: E402
from repoguard.schemas import Evidence  # noqa: E402
from repoguard.verification import extract_claims, verify  # noqa: E402

PROMPT_RULES = (
    "Answer the developer's question about a pinned Python repository. "
    "If you cannot ground the requested answer, write ABSTAIN on the first line and briefly explain. "
    "Use exact identifiers in backticks. Use explicit sentences such as "
    "'The class `X` is defined in `pkg/file.py`.', "
    "'Function `X` accepts parameter `y`.', "
    "'Function `X` calls `Y`.', "
    "'File `pkg/file.py` exists.', or '`pkg/file.py` imports `pkg.mod.X`.'"
)


def _prompt(repository_id: str, question: str, evidence: list[Evidence] | None) -> str:
    sections = [PROMPT_RULES, f"Repository: {repository_id}"]
    if evidence is not None:
        context = "\n\n".join(
            f"{item.file_path}:{item.start_line}-{item.end_line}\n{item.snippet}"
            for item in evidence
        )
        sections.append(
            "Use the following repository passages as code evidence.\n"
            + (context or "No matching code passages.")
        )
    sections.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "data/questions/phase2_benchmark_v1.jsonl"
    )
    parser.add_argument("--indexes", type=Path, default=ROOT / "data/indexes")
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true")
    mode.add_argument(
        "--snapshot", type=Path, help="Local Hugging Face snapshots/<revision> directory."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve() if args.snapshot else None
    if snapshot is not None and not (snapshot / "config.json").is_file():
        parser.error("--snapshot must be a local model snapshot containing config.json.")

    try:
        generation = GenerationSettings(
            mode="mock" if args.mock else "live",
            model="repoguard-mock" if args.mock else args.model,
            revision=snapshot.name if snapshot else None,
            max_new_tokens=128,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.05,
            seed=42,
        )
        cases = [
            case for case in load_benchmark(args.benchmark) if args.split in ("all", case.split)
        ]
        indexes = validate_indexes(cases, args.indexes)
    except (OSError, RepoGuardError, ValueError) as exc:
        parser.error(str(exc))

    import torch

    from repoguard.llm import LLMService

    llm = LLMService(
        model_name=str(snapshot) if snapshot else generation.model,
        mock_mode=args.mock,
        device="cpu" if args.mock else args.device,
    )
    # Load weights and warm up once so cold start is excluded from paired latencies.
    llm.generate("Return the word ready.", max_new_tokens=1, temperature=0.0)
    torch.manual_seed(42)
    retrievers = {name: Retriever(index) for name, index in indexes.items()}
    settings = generation.model_dump(
        include={"max_new_tokens", "temperature", "top_p", "repetition_penalty"}
    )
    rows = []
    for case in cases:
        index = indexes[case.repository_id]
        for system in ("llm_only", "retrieval_llm"):
            started = time.perf_counter()
            evidence = None
            if system == "retrieval_llm":
                evidence = retrievers[case.repository_id].search(case.question, k=5).evidence
            answer = llm.generate(_prompt(case.repository_id, case.question, evidence), **settings)
            text = answer.text.strip()
            first_line = text.split("\n", 1)[0].strip().rstrip(".:").casefold()
            prediction = Prediction(
                system=system,
                case_id=case.case_id,
                commit=case.commit,
                generation=generation,
                answer=text,
                abstained=first_line == "abstain",
                index_id=index.index_id if evidence is not None else None,
                retrieved_units=ranked_units(evidence) if evidence is not None else None,
                input_tokens=answer.prompt_tokens,
                output_tokens=answer.completion_tokens,
                steps=1 + int(evidence is not None),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            rows.append(prediction)
            if system == "retrieval_llm":
                changes: dict = {"system": "fixed_verifier"}
                if not prediction.abstained:
                    result = verify(extract_claims(text), index)
                    changes["steps"] = 3
                    changes["latency_ms"] = (time.perf_counter() - started) * 1000
                    if result.verdict != "verified":
                        changes["abstained"] = True
                        changes["answer"] = "ABSTAIN\nThe verifier could not support the answer."
                rows.append(prediction.model_copy(update=changes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    print(
        f"Wrote {len(rows)} {generation.mode} predictions for {len(cases)} cases to {args.output}"
    )


if __name__ == "__main__":
    main()
