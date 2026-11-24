# Phase 2 verifier contract and audit

The proposed contract adds a machine-readable `reason_code` to each verdict.
The Phase 1 status values remain `verified` (supported), `refuted`
(contradicted in the indexed scope), and `unverifiable` (insufficient evidence).
Existing saved verdicts load with an empty reason code. Structured semantic
claims stay unverifiable, and a function-or-class claim checks either kind.

For the RL handoff, use the per-claim statuses and codes alongside independent
answer labels. An empty extraction, an unresolved call, and a refuted claim
carry different evidence. Use the pinned index and the source references in
the audit to review disagreements before setting reward weights.

## Source-checked claims

| Repository | Pinned commit | Cases | True positives | False positives | False negatives | True negatives | Unknown truth |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dbader/schedule` | `82a43db1b938d8fdf60103bd41f329e06c8d3651` | 16 | 7 | 0 | 2 | 7 | 0 |
| `theskumar/python-dotenv` | `4fbc6ff07215367ece731006855e732d1405992c` | 17 | 8 | 0 | 1 | 5 | 3 |
| Total | | 33 | 15 | 0 | 3 | 12 | 3 |

A positive prediction means the verifier returned `verified`. On this sample,
precision was 15/15 and recall was 15/18 (83.3%). The three known supporting
facts that remained unverifiable were two README files outside the Python
index and `schedule.Scheduler._run_job` calling `job.run()` through an
unresolved receiver.

The [new python-dotenv cases](../tests/fixtures/phase2_verifier_audit_dotenv.json)
include source checks and expected statuses and reason codes. The ambiguous
constructor, non-canonical path, and semantic guarantee have no binary truth
label in this audit. All 17 expectations matched. The two repositories are
development inputs. The held-out `requests` repository was reserved for evaluation
of the verifier.

## Reproduce

```sh
python scripts/fetch_repos.py schedule python-dotenv --index
python scripts/audit_verifier.py --index data/indexes/schedule
python scripts/audit_verifier.py --index data/indexes/python-dotenv --labels tests/fixtures/phase2_verifier_audit_dotenv.json --labels-only
```

The schedule audit also replays the 300 saved baseline responses. A verified
fact in an answer establishes only that extracted fact. Measuring answer
correctness requires an independently audited benchmark with exact expected
facts and source locations. The Phase 2 benchmark should replace inherited
line labels with checked evidence and use separate development and held-out
repositories.
