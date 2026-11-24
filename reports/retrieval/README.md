# Phase 2 Retrieval Report (Member 1)

Retrieval quality, RL free-signal validation, and a grounding audit of the baseline questions.
All numbers below come from `retrieval_eval_hashing.json` and `baseline_question_grounding.json` in this folder.

> **Backend caveat.** Everything here was measured with the deterministic **hashing** embedder (lexical, not semantic), because the all-MiniLM-L6-v2 weights cannot be downloaded in the build sandbox. Rebuild the indexes with `embeddings.backend: sentence-transformers` and re-run the commands below before reporting final numbers. The hashing results are a lower bound for semantic queries (docstring paraphrases) and a fair reference for name-based queries.

## Reproduce

```bash
make index                                           # fetch pinned repos + build indexes (configs/repos.yaml)
python scripts/build_retrieval_benchmark.py          # -> data/retrieval/retrieval_benchmark_v1.jsonl
python scripts/evaluate_retrieval.py                 # -> reports/retrieval/retrieval_eval_<backend>.{json,md}
```

## 1. Benchmark (`data/retrieval/retrieval_benchmark_v1.jsonl`, 757 cases)

Relevance labels are derived from **index facts**, not hand-written line numbers. Relevance unit = symbol (all parts of one long function count once).

| Case type | What it tests | Relevant passages |
|---|---|---|
| `symbol_lookup` (300) | "Where is the method `Job.do` defined?" | every definition with that display name |
| `docstring` (179) | first docstring sentence with the symbol's name masked | the documented symbol |
| `callers` (228) | "Which functions call `X`?" | callers found by the static call graph (1–5 per case) |
| `baseline_q` (50) | Member 3's answerable baseline questions | their target resolved through the index |

Repositories: `schedule`, `python-dotenv`, `cachetools`, `python-tabulate` (**dev**) and `requests` (**test**, held out, never used for tuning), all at the commits pinned in `configs/repos.yaml`. Up to 60 cases per type per repository; test files are never targets. Member 2 owns the final benchmark splits, so `split_hint` is a suggestion.

## 2. Retrieval results

**Hybrid ranking (default):** cosine + 0.30 if the passage defines a named symbol, + 0.10 if it mentions one. The boosts were picked by a grid search on the **dev split only**.

| Split | Case type | n | Hit@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| dev | symbol_lookup | 240 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| dev | docstring | 119 | 0.76 | 0.95 | 0.96 | 0.85 | 0.87 |
| dev | callers | 168 | 0.05 | 0.73 | 0.89 | 0.38 | 0.51 |
| dev | baseline_q | 50 | 0.86 | 0.86 | 0.88 | 0.87 | 0.86 |
| **test** | symbol_lookup | 60 | **0.97** | **1.00** | 1.00 | **0.98** | 0.99 |
| **test** | docstring | 60 | 0.77 | 0.92 | 0.97 | 0.84 | 0.87 |
| **test** | callers | 60 | 0.02 | **0.70** | 0.93 | 0.37 | 0.52 |
| all | all | 757 | 0.64 | **0.89** | **0.95** | **0.76** | 0.81 |

**Ablation: dense only (no boosts).**

| Split | Case type | Hit@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| dev | symbol_lookup | 0.57 | 0.77 | 0.85 | 0.66 |
| dev | callers | 0.14 | 0.48 | 0.66 | 0.33 |
| dev | baseline_q | 0.40 | 0.55 | 0.60 | 0.49 |
| test | symbol_lookup | 0.65 | 0.85 | 0.88 | 0.74 |
| test | callers | 0.12 | 0.52 | 0.65 | 0.35 |
| all | all | 0.48 | 0.72 | 0.80 | 0.60 |

What this shows:

* **The symbol boost generalises to the held-out repository.** `requests` symbol-lookup Hit@1 rises from 0.65 to 0.97 and callers Recall@5 from 0.52 to 0.70. Docstring queries are unaffected, because they name no symbol.
* **Trade-off on caller questions.** For "who calls X", the definition of X now ranks first, so callers **Hit@1 falls** (dev 0.14 → 0.05) while **Recall@5/10 rise** (0.48 → 0.73 / 0.66 → 0.89). For an evidence-gathering agent, recall at k matters more than rank 1, but this is a design choice to confirm with Member 3.
* Caller questions remain the hardest: answering them reliably needs the call graph (verifier / INSPECT), not similarity search alone.

**Ablation: `exclude_tests` filter.** No change in quality (Recall@5 0.893 → 0.895, MRR 0.764 → 0.764), but evidence comes only from library code. Useful when the agent should cite implementation rather than tests.

**Cost:** median latency 0.9 ms and p95 1.8 ms per query (CPU, exact search, at most 840 passages); about 870 approximate tokens for the top-5 evidence. Empty-result rate 0.

## 3. RL free signals (details in `docs/state_signals.md`)

AUROC for separating answerable questions from each trap type on the 100 baseline questions:

| Signal | nonexistent | near-miss | wrong parameter | off-topic |
|---|---|---|---|---|
| `symbol_found` | 0.98 | 0.98 | 0.98 | 0.93 |
| `name_similarity` | 0.98 | 0.98 | 0.98 | 0.94 |
| `retr_top1` | 0.85 | 0.79 | 0.72 | 0.89 |
| `stack_match` | — | — | — | 0.90* |

\* The foreign-technology list was extended after looking at these off-topic questions, so 0.90 is optimistic. Re-measure on new off-topic questions in the test split.

## 4. Grounding audit of `data/questions/baseline_questions.jsonl` (for Members 2 and 3)

Each answerable question's `target_symbol` was resolved through the index and compared with its hand-written `relevant_lines` (full list in `baseline_question_grounding.json`):

* **44 of the 50** answerable questions have `relevant_lines` that do **not overlap** the real definition at the pinned commit.
* 6 targets are not definitions (`default_scheduler`, and five import questions); for those the module overview passage was used.

| Question | Target | `relevant_lines` | Actual definition |
|---|---|---|---|
| q_ans_01 | `Job` | 28–160 | `schedule.Job`, lines 209–830 |
| q_ans_02 | `Scheduler` | 620–850 | `schedule.Scheduler`, lines 79–206 |
| q_ans_03 | `every` | 860–875 | `Scheduler.every` 162–170 (module function `every` 843–847) |
| q_ans_04 | `run_pending` | 878–885 | `Scheduler.run_pending` 89–101 |
| q_ans_08 | `next_run` | 918–925 | `schedule.next_run` 885–889 |

**Impact:** any metric that scores answers or evidence by these line numbers (citation support, evidence overlap) would be wrong. Recommendation: regenerate `relevant_lines` from the index (`idx.find_symbols(target)` gives exact `line`/`end_line`), or drop the field and score by symbol. The retrieval benchmark above already uses index-resolved targets.

## 5. Limitations

* Hashing backend only (see the caveat at the top). Re-run with sentence-transformers.
* Labels inherit the static index's limits: callers reached only through dynamic dispatch are missing from `callers` relevance sets, so caller recall is slightly understated.
* Docstring queries keep the docstring's wording, so they are easier than real paraphrases; they test content retrieval, not semantic generalisation.
* One query per case; no human relevance judgements yet. Member 2's audited benchmark should add some.
