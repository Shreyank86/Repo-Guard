# Repository Intelligence Module (`src/repoguard/repository/`)

**Owner: Member 1 (Repository Intelligence Engineer)** · **Branch: `feature/indexer`**

Converts a Python repository into a structured, searchable index (symbols, imports, static call graph, passages, vectors) that the verifier, baseline and RL agent consume. Repository code is **parsed, never imported or executed**.

Output format and field meanings: **[`docs/index_schema.md`](../../../docs/index_schema.md)**.

## Files

| File | Responsibility |
|---|---|
| `parser.py` | Recursive scan (skips `.git`, virtualenvs incl. any dir with `pyvenv.cfg`, `build`, `dist`, caches, `*.egg-info`, configured globs), path-traversal and symlink protection, encoding-aware reading, AST extraction of functions / async functions / methods / classes (qualified names, lines, parameters with kinds/defaults/annotations, return annotation, decorators, docstrings, bases, class variables) and imports (relative imports resolved) |
| `callgraph.py` | Two-pass static call graph: builds a repo-wide symbol table, then records every call site with caller, callee, location and a `resolution` label (`local`, `imported`, `method`, `module_attribute`, `external`, `builtin`, `unresolved`) |
| `passage_builder.py` | Function/method passages (exact source), class and module overview passages (bodies elided), long passages split with overlap |
| `indexer.py` | `build_index()` orchestration, deterministic JSON/JSONL serialization, manifest with provenance and hashes; `load_index()` |
| `../retrieval/embedder.py` | `SentenceTransformerEmbedder` (all-MiniLM-L6-v2) and deterministic `HashingEmbedder` fallback |
| `../retrieval/search.py` | FAISS `IndexFlatIP` (NumPy fallback) build/load, `search(question, index, k) -> list[Evidence]` |

## Usage

```bash
# 1. fetch pinned snapshots (configs/repos.yaml) and index them into data/indexes/<name>
python scripts/fetch_repos.py --index            # or: python scripts/fetch_repos.py schedule --index

# 2. or index any checkout directly
python scripts/build_index.py --repo data/repos/schedule --out data/indexes/schedule \
    --config configs/indexer_config.yaml          # add --embedding-backend hashing when offline

# 3. cross-check the index against the source (independent tokenize-based checks)
python scripts/validate_index.py --repo data/repos/schedule --index data/indexes/schedule \
    --report reports/index_validation/schedule.json

# 4. inspect
python scripts/search_index.py --index data/indexes/schedule --symbol every
python scripts/search_index.py --index data/indexes/schedule --callers run_pending
python scripts/search_index.py --index data/indexes/schedule "when does a job run next"
```

```python
from repoguard.repository import build_index, load_index
from repoguard.retrieval import search

art = build_index("data/repos/schedule", "data/indexes/schedule", {"embeddings": {"enabled": True}})
idx = load_index("data/indexes/schedule")
idx.find_symbols("every")  # functions/classes by short or qualified name
idx.callers_of("run_pending")  # call edges into a function
search("cancel a scheduled job", idx, k=5)
```

`build_index(repo_path, output_dir, config=None)` raises `RepositoryNotFoundError` only if the repository directory is missing; every per-file problem (syntax error, undecodable file, missing file in `include_files`, symlink escaping the repo) is recorded in `errors` and indexing continues. Unknown config keys raise `IndexingError`. Config keys and defaults: `configs/indexer_config.yaml`.

## Tests

```bash
pytest tests/unit/test_parser.py tests/unit/test_callgraph.py tests/unit/test_indexer.py tests/unit/test_retrieval.py
```

Fixture: `tests/fixtures/indexer_repo/` (hand-written package with known definitions, all parameter kinds, inheritance, `super()`, nested functions, async, relative and aliased imports, cross-module calls). Broken/missing/excluded files are generated per test in a temporary copy.

| Phase 1 cross-check (assignment §3) | Tests |
|---|---|
| Function/class extraction — all found, no phantoms | `test_extracts_exactly_the_expected_definitions`, `test_reported_line_really_holds_the_definition` |
| Parameters and lines | `test_parameters_all_kinds_in_order`, `test_parameters_annotations_defaults_and_return`, `test_types_and_line_ranges_match_source` |
| Call graph — direct, nested, cross-module | `test_cross_module_calls_are_resolved`, `test_nested_calls_each_produce_an_edge`, `test_method_calls_via_self_inheritance_super_and_class`, … |
| Missing / syntax-error files | `test_malformed_and_missing_files_are_recorded_and_others_indexed`, `test_syntax_error_is_recorded_not_raised`, … |
| Repeatability | `test_two_runs_produce_byte_identical_output`, `test_rebuild_in_place_is_stable_and_removes_stale_vectors` |

## Known limitations

Static call graph only — dynamic dispatch, reflection, monkey-patching and runtime-generated calls are recorded as `unresolved`, never guessed. Full list: `docs/index_schema.md` §8.
