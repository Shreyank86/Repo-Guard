# Verification

Member 2's verifier checks structured claims against Member 1's Python index. It
uses exact paths and names and reports supporting source locations. The
Phase 2 contract proposal lives in `verification/models.py` for team review.

```python
from repoguard.repository import load_index
from repoguard.verification import Claim, extract_claims, verify

index = load_index("data/indexes/schedule")
text = "The `Job` class is defined in `schedule/__init__.py`."
print(verify(extract_claims(text), index).model_dump(mode="json"))

claims = [{"type": "parameter_exists", "symbol": "Job.do", "parameter": "job_func"}]
print(verify(claims, index).model_dump(mode="json"))
```

| Claim type | Fields to supply | Evidence |
| --- | --- | --- |
| `file_exists` | `file` | A successfully parsed Python file at that path |
| `function_exists` | `symbol`, optional `file` | A matching function or method definition |
| `class_exists` | `symbol`, optional `file` | A matching class definition |
| `symbol_exists` | `symbol`, optional `file` | A matching function, method, or class when the claim leaves the kind open |
| `parameter_exists` | `symbol`, `parameter`, optional `file` | One identified signature that declares the parameter |
| `calls` | `caller`, `callee`, optional caller `file` | A resolved static edge in that direction |
| `imports` | `file`, `module` | An explicit import of the module or qualified symbol |
| `semantic` | `statement`, optional `file` | A behavioral assertion recorded as unverifiable by this index |

`Claim` also accepts these fields as keyword arguments. Its default type is
`class_exists`, so the original `Claim(symbol="Job", file="schedule/__init__.py")`
form still works. A file is a canonical repository-relative POSIX path. A symbol can be
given as `schedule.Job.do`, `Job.do`, or a short name. Parameter and call checks
require enough qualification to resolve ambiguity.

| Per-claim status | Meaning for consumers | Reason codes |
| --- | --- | --- |
| `verified` | Supported by facts in the index | `file_indexed`, `definition_found`, `parameter_declared`, `call_resolved`, `import_found` |
| `refuted` | Contradicted within the indexed scope | `definition_absent`, `parameter_absent`, `call_absent`, `import_absent` |
| `unverifiable` | Insufficient evidence or a claim outside the verifier's scope | `file_unindexed`, `file_parse_error`, `incomplete_index`, `ambiguous_symbol`, `unresolved_call`, `invalid_claim`, `invalid_path`, `semantic_claim` |

The aggregate is `verified` when all claims pass, `partially_verified` when only
some pass, and `unsupported` when none pass or the input is empty. Consume the
per-claim `status` and `reason_code` for reward or evaluation. An aggregate
`unsupported` can come from missing evidence, contradictions, or an empty
extraction. The human-readable `reason` explains the individual verdict.
Phase 1 status values and call signatures still work. A legacy saved verdict
loads with `reason_code=None`.

The extractor recognizes explicit positive assertions, including common forms
for definitions, parameter lists, calls, and imports. It skips fenced examples,
quoted passages, and the appended repository context. Semantic explanations,
negation, qualified runtime behavior, and loosely phrased claims need a more
capable extractor or human review.

## Reproduce the audit

```sh
python scripts/fetch_repos.py schedule
python scripts/build_index.py --repo data/repos/schedule --out data/indexes/schedule --config configs/indexer_config.yaml --embedding-backend hashing --strict
python scripts/validate_index.py --repo data/repos/schedule --index data/indexes/schedule
python -m pytest tests/unit/test_verifier.py tests/unit/test_claim_extractor.py tests/unit/test_verification_metrics.py
python scripts/audit_verifier.py --index data/indexes/schedule
python scripts/fetch_repos.py python-dotenv --index
python scripts/audit_verifier.py --index data/indexes/python-dotenv --labels tests/fixtures/phase2_verifier_audit_dotenv.json --labels-only
```

The audit produces JSON on stdout. Use `--output` to save it. It checks manual
labels against the pinned index and replays the saved baseline answers without
loading a model. See the [Phase 2 contract audit](../../../reports/phase2_verifier_contract_audit.md)
for the reason-code checks, measured results, and limitations.
