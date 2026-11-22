# Phase 1 Member 2 verifier audit

The audit uses the pinned `dbader/schedule` commit
`82a43db1b938d8fdf60103bd41f329e06c8d3651`. The regenerated index passed the
source cross-check for 170 definitions, 195 parameters, 1,065 call sites, and
172 passages, with zero index errors.

## Manually labelled structural claims

The [16 labelled checks](../tests/fixtures/verifier_audit.json) cover all six
implemented claim types and include a written source check for each label.
A positive result means the verifier returned `verified`.

| Result | Count |
| --- | ---: |
| True positives | 7 |
| False positives | 0 |
| False negatives | 2 |
| True negatives | 7 |
| Verification precision | 7/7 (100%) |
| Verification recall | 7/9 (77.8%) |

Both misses were explicitly marked `unverifiable`. The repository contains
`README.rst` outside the Python-only index. `Scheduler._run_job` calls
`job.run()` on a parameter annotated as `Job`, and the static graph leaves
that call target unresolved. These results describe the hand-checked sample.

## Saved baseline replay

The verifier processed all 300 saved runs. The extractor found at least one
structural claim in 31 of 213 answered responses (14.6%).

| Baseline | Answers | With extracted claims | With all extracted claims verified | With a refuted claim | With an unverifiable claim | Abstention decision accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_answer` | 100 | 19 | 0 | 6 | 13 | 50% |
| `always_evidence` | 100 | 10 | 3 | 2 | 5 | 50% |
| `threshold_gate` | 13 | 2 | 0 | 0 | 2 | 63% |

Abstention decision accuracy is the fraction of all runs that answered an
answerable question or abstained on a trap. The audit's structural hallucination
rate counts responses with an explicitly refuted claim, divided by all runs.
Unverifiable claims and responses with no extractable claims remain visible in
the counts. The low extraction coverage warrants manual review before drawing
answer-level accuracy or hallucination conclusions.

The regression test uses the saved `q_ans_01` answer, which places `Job` in
`job.py`. The verifier gives that claim no credit and
verifies the correct definition at `schedule/__init__.py:209`.

Reproduction commands are in the [verifier README](../src/repoguard/verification/README.md).
