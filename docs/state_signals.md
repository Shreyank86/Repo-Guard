# RL State Signals from Retrieval — Feature Dictionary (v1.0)

**Owner:** Member 1. **Consumer:** Member 3 (RL environment, `StateVector`). **Code:** `src/repoguard/retrieval/signals.py` (`SIGNALS_VERSION = "1.0"`).

These are the report's **free signals**: they cost **zero LLM tokens** and are available at **step 0** of every episode. They use only the question text, the index and one retrieval probe — **never ground-truth labels or future information**.

```python
from repoguard.retrieval import Retriever, SignalExtractor
from repoguard.schemas import StateVector

extractor = SignalExtractor(Retriever.from_dir("data/indexes/schedule"))   # build once per repo
sig = extractor.compute(question)                                            # FreeSignals
state = StateVector(**sig.state_fields(), risk_lambda=lam)                  # first 6 of the 18 dims
sig.feature_vector()   # all 16 numeric signals (6 state + 10 extended), fixed order
```

Retrieval probe settings: `k = 5`, hybrid ranking with the default boosts (see `docs/index_schema.md` §6), no filters. `SignalExtractor(retriever, k=..., min_evidence_score=0.30, token_budget_ref=2000, stack_overlap_threshold=0.5)`.

---

## 1. StateVector "free" group (6 dims)

All values are in **[0, 1]**. "Missing" means the signal cannot be computed for this question; it is then **0**, and the availability flags in §2 (`has_named_symbol`, `has_named_file`) tell the policy why.

| Feature | Meaning | Computation | Range / normalisation | Missing-value rule |
|---|---|---|---|---|
| `symbol_found` | Every code name in the question exists in the repository | 1 if **all** names from `extract_named_symbols(question)` are in the repo name set, else 0. The name set holds symbol short names, qualified names and every dotted suffix (`Job.do`), parameter names, class variables, module names, imported names and call targets as written (`default_scheduler.every`) | binary {0, 1} | 0 when the question names no symbol (`has_named_symbol = 0`) |
| `name_similarity` | How close the **least similar** named symbol is to a real name (catches near-misses) | per name: 1.0 if it exists, else max `difflib` ratio (case-insensitive) against repo names of the same shape (dotted vs. plain). Dotted names also compare their last component, scaled ×0.95. Feature = **min** over named symbols | [0, 1] | 0 when no symbol is named |
| `retr_top1` | Best retrieval similarity | max raw cosine similarity over the top-k evidence, clipped to [0, 1] | [0, 1]; scale depends on the embedding backend (see §4) | 0 for an empty index or blank question |
| `retr_margin` | Margin of the best over the runner-up | `top1 − second` (by raw cosine) among the top-k; `top1` if there is only one result | [0, 1] | 0 when no evidence |
| `passage_mentions_symbol` | The top passage contains a named symbol | 1 if the rank-1 passage (symbol name + content) contains the last component of any named symbol as a whole identifier | binary | 0 when no symbol is named or no evidence |
| `stack_match` | The question is about this repository / technology stack | 0 if the question mentions a foreign technology absent from the repo vocabulary (e.g. Django, React, Kafka); else 1 if it names code (symbol or `.py` file) **or** `vocab_overlap ≥ 0.5`; else 0 | binary | 0 for a blank question |

## 2. Extended signals (optional, all in [0, 1])

Exposed in `FreeSignals` for the environment, logging or a larger state. Not in the 18-dim vector unless Member 3 decides to add them.

| Feature | Meaning | Missing-value rule |
|---|---|---|
| `has_named_symbol` | Availability flag: the question names ≥ 1 code symbol | — |
| `has_named_file` | Availability flag: the question names ≥ 1 `.py` file | — |
| `file_found` | Every named file exists in the index (exact path or path suffix) | 0 when no file is named |
| `symbol_in_named_file` | A named, existing definition is defined in a named file (**wrong-path traps**: "Is `load_config` in `missing.py`?") | 0 when no file is named |
| `evidence_count` | Results with cosine ≥ `min_evidence_score` (0.30), divided by k | 0 when no evidence |
| `source_diversity` | Distinct files / number of returned evidence items | 0 when no evidence |
| `coverage` | Share of named *definitions* (functions/methods/classes) that appear in, or are mentioned by, the top-k evidence | 0 when no named definition or no evidence |
| `missing_evidence` | 1 if `evidence_count = 0` or a named definition is not covered | **1** for blank questions / empty index |
| `vocab_overlap` | Share of the question's content words that occur in the repo vocabulary (identifier parts, paths, docstrings). Question words ("where", "defined", "parameter") and the repo's own name are excluded | 0 when there are no content words |
| `retrieved_tokens` | Approximate tokens in the top-k evidence (chars / 4) / `token_budget_ref` (2000, as in `configs/rl_config.yaml`), clipped to 1 | 0 when no evidence |

Context (non-numeric) fields: `named_symbols`, `named_files`, `nearest_symbol` + `nearest_symbol_file` (useful for the abstention message: "Closest real match: `run_pending` in `schedule/__init__.py`"), and `evidence_ids` (stable ids of the probe's evidence).

## 3. How a "named symbol" is detected (`extract_named_symbols`)

In order of appearance, deduplicated:

1. Text in backticks: always kept, even if it is an English word (`` `every` ``). `name=value` → `name`; `f()` → `f`.
2. Calls: `sync_user()`.
3. Questions without backticks only: "function/method/class/parameter … `<name>`" and "`<Name>` class/function". A lowercase word before a noun ("base class") is ignored.
4. Identifier-shaped words: snake_case, camelCase, dotted `a.b`, dunder.

Paths ending in `.py` are files, not symbols. Plain English words are never symbols unless backticked.

## 4. Measured behaviour (Phase 2 validation)

Measured on Member 3's 100 baseline questions for `dbader/schedule` with the **hashing** embedder (full tables: `reports/retrieval/retrieval_eval_hashing.md`).

AUROC for separating the 50 answerable questions from each trap type (0.5 = no separation, 1.0 = perfect):

| Signal | vs. nonexistent | vs. near-miss | vs. wrong parameter | vs. off-topic |
|---|---|---|---|---|
| `symbol_found` | 0.98 | 0.98 | 0.98 | 0.93 |
| `name_similarity` | 0.98 | 0.98 | 0.98 | 0.94 |
| `retr_top1` | 0.85 | 0.79 | 0.72 | 0.89 |
| `retr_margin` | 0.75 | 0.79 | 0.74 | 0.78 |
| `passage_mentions_symbol` | 0.96 | 0.96 | 0.46 | 0.86 |
| `stack_match` | 0.53 | 0.50 | 0.50 | 0.90* |

\* **Caveat — not an unbiased estimate.** I extended the foreign-technology word list (Flask, Celery, FastAPI, pandas, …) *after* looking at these 10 off-topic questions. `stack_match` must be evaluated on **new** off-topic questions in the held-out test split (Member 2's benchmark). All other signals were defined before this check and are not tuned on these questions.

Reading the table:

* `symbol_found` / `name_similarity` are the strongest trap detectors (nonexistent, near-miss, wrong-parameter).
* `stack_match` is meant only for off-topic questions; near-miss and nonexistent traps are on-topic, so 0.5 there is the intended behaviour.
* Wrong-parameter traps name a real function, so `passage_mentions_symbol` and `coverage` stay high; `symbol_found` catches them because the made-up parameter is not a known name.
* Answerable `symbol_found` is < 1 only where the target cannot be confirmed statically: `default_scheduler.next_run` (a property on an instance; similarity 0.95) and import questions that name only a file.

## 5. Known limitations

* **Embedding dependence.** `retr_top1` / `retr_margin` were measured with the lexical hashing embedder (all-MiniLM-L6-v2 cannot be downloaded in the CI sandbox). Their scale will change with the sentence-transformers model; re-measure and normalise before training. Rebuild indexes with `embeddings.backend: sentence-transformers`.
* **Static name set.** Instance attributes and properties reached through variables (`default_scheduler.next_run`) are only partly visible; such names get high similarity but `symbol_found = 0`.
* **`symbol_found` requires all named symbols.** A question that names a real function *and* a nonexistent parameter scores 0. This is intended for trap detection, but Member 3 may prefer the extended fields for finer control.
* **English heuristics.** Symbol detection is tuned for English questions with backticked code (the format of the current benchmark).
* `stack_match` is a heuristic (vocabulary overlap + foreign-technology list), not a classifier.

## 6. Change control

Changing a definition here changes the policy's observation space. Bump `SIGNALS_VERSION`, update this table and `tests/unit/test_signals.py`, and get Member 3's review before merging.
