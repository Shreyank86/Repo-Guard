# Phase 2 benchmark and evaluation handoff

## Benchmark v1

[Benchmark v1](../data/questions/phase2_benchmark_v1.jsonl) contains 60
source-audited questions at five exact repository commits. Each JSONL row
records the version, repository URL, licence, commit, split, question type,
answerability, expected structural facts, reference answer, and source ranges.
All 74 ranges and 40 affirmative facts were checked against the pinned source.
Question wording and labels were written independently of model answers.

| Split | Repositories | Answerable | Abstention expected | Total |
| --- | --- | ---: | ---: | ---: |
| Train | schedule, python-dotenv, cachetools | 21 | 15 | 36 |
| Validation | python-tabulate | 7 | 5 | 12 |
| Held-out test | requests | 7 | 5 | 12 |
| Total | 5 repositories | 35 | 25 | 60 |

The loader rejects duplicate normalized questions, repeated IDs, repository
aliases, and reused commits across repository groups. The validation and test
repositories have distinct source snapshots. Similar question forms across
splits ask about different repository facts. The three train repositories are
available for training. Use validation for calibration and the test split
for a frozen comparison.

Answerability is judged against the question as written. An abstention case
can receive a well-grounded correction of a false premise or a clarification
request. A positive multi-hop case requires both annotated facts. Evidence
ranges are inclusive and one-based. An enclosing function can be the
retrieval unit for a nested call whose exact source line appears in that passage.
Module overviews are the designated retrieval units for file and import cases.

## Verifier and reward handoff

The [contract audit](phase2_verifier_contract_audit.md) checked 33 cases across
two development repositories. The 30 cases with known truth yielded 15 true
positives, zero false positives, three false negatives, and 12 true negatives.
Two missed facts involved files outside the Python index. One involved an
unresolved dynamic call target. The typed `status` and `reason_code` distinguish
these gaps from contradictions. An empty claim set carries no positive evidence.

The evaluator records incorrect abstention, answered unanswerable questions,
missing facts, and specific claim verdicts separately. Member 3 can use these
records to check reward behavior. Keep gold facts and answerability out of
agent observations. Select reward weights using the validation split.

## Retrieval ablation

Both variants used the same hashing embeddings, exact FAISS search, source
labels, and five-result depth. The hybrid variant used Member 1's existing
definition and mention boosts. The second variant used hashing cosine alone.
Only answerable cases have positive retrieval labels.

| Split | Cases | Hybrid Recall@5 | Cosine Recall@5 | Hybrid MRR@5 | Cosine MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 21 | 0.667 | 0.429 | 0.676 | 0.344 |
| Validation | 7 | 0.714 | 0.286 | 0.714 | 0.286 |
| Held-out test | 7 | 0.714 | 0.071 | 0.643 | 0.143 |
| All | 35 | 0.686 | 0.329 | 0.677 | 0.292 |

These measurements describe retrieval on this small benchmark. For example,
hybrid search missed the two expected module overviews on the held-out split.
On `cachetools-ttl-decorator-chain`, it found the public decorator definition
and missed the enclosing helper. Search quality on these cases matters even
when the verifier can check each individual fact.

## Answer comparison and error analysis

The identical-case comparison pipeline produced 180 mock predictions from three
systems. The retrieval and fixed-verifier systems shared each generated candidate
and its retrieved passages, so their difference isolates the gate. This run
exercises the pipeline using RepoGuard's canned CPU responses.

| Mock system | Cases | Answered | Correct abstentions | False abstentions |
| --- | ---: | ---: | ---: | ---: |
| LLM only | 60 | 60 | 0 | 0 |
| Retrieval + LLM | 60 | 60 | 0 | 0 |
| Retrieval + LLM + fixed verifier | 60 | 0 | 25 | 35 |

The mock candidates yielded no structural claims for the answerable questions.
The gate therefore abstained on all 60 cases. Its perfect mock abstention recall
came with zero coverage and a 100% false-abstention rate. Score both properties
to expose the always-abstain failure mode.

Representative errors from the separate [saved Phase 1 outputs](../data/baseline/raw_outputs.jsonl):

| Case | Observation | Source-grounded assessment |
| --- | --- | --- |
| `q_ans_01` | Direct answer placed `Job` in `job.py`. | `Job` is in `schedule/__init__.py:209`. A name-only score gave false credit. |
| `q_ans_16` | Direct answer invented constructor arguments including `job_id` and `status`. | The signature at line 227 declares `self`, `interval`, and `scheduler`. |
| `q_ans_31` | Retrieval answer identified `_run_job` calling `Job.run`. | Line 173 has the call on an argument annotated as `Job`. Static resolution is insufficient, so the verifier reports unknown. |
| `q_trap_16` | Retrieval answer corrected the premise that `sync_all_jobs` exists while the trace marked it as answered. | Score the decision flag and independent answer judgment separately. |

The archived rows are examples for error analysis. The Phase 2 report has
separate slots for a live pinned-model comparison, Member 3's trained policy,
and the risk-feature and reward-term ablations. Those runs require the frozen
model snapshot, the trained policy and config, and the controller's prediction
exports. The shared API response tests also depend on the query API integration.

## Reproduce and hand off

```sh
python scripts/fetch_repos.py --index
python scripts/evaluate_phase2.py --validate-only
python scripts/evaluate_phase2.py --retrieval-only --out-dir experiments/results/phase2/retrieval
python scripts/run_phase2_baselines.py --mock --split all --output experiments/results/phase2/mock_predictions.jsonl
python scripts/evaluate_phase2.py --predictions experiments/results/phase2/mock_predictions.jsonl --out-dir experiments/results/phase2/mock
```

The retrieval report above used the `hashing` backend. With MiniLM installed,
fetch the repositories first and run `scripts/build_index.py` for each checkout
with `--config configs/indexer_config.yaml --embedding-backend hashing` to force
the same backend. Reuse the same generated indexes for prediction and scoring.
The report includes the backend and index IDs. The mock baseline runner uses
the existing PyTorch-backed adapter in CPU mock mode. For live generation,
replace `--mock` with `--snapshot` followed by a local Hugging Face
`snapshots/<40-character-revision>` directory that contains the team's frozen
model. Keep the model and generation settings identical across measured systems.

For policy comparisons, export one `Prediction` JSONL row per system and case.
Use distinct `system` IDs for `rl_agent` and ablation variants. The fields are
defined in [benchmark_scoring.py](../src/repoguard/evaluation/benchmark_scoring.py).
Include the commit, model revision, settings, answer, abstention decision,
step count, input/output tokens, latency, and ranked retrieval units plus their
index ID where applicable. Pass all prediction files to `--predictions` for
one split. The harness rejects incompatible runs and incomplete case coverage.
For independently reviewed answer accuracy, `--reviews` accepts JSONL rows
with `system`, `case_id`, `correct`, and `rationale`.

The structural answer metric gives credit for all required verified facts or
a justified abstention. Claim precision and recall use the annotated fact
set. Unknown and contradicted claims are reported separately. Retrieval
metrics collapse multiple parts of one symbol into one unit. Cost and latency
are summarized from measured records. Missing measurements remain null.
The per-case JSONL preserves ranked retrieval units, verdicts, costs, and the
error category.
