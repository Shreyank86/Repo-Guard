# Repository Index & Evidence Contract (schema v1.1)

**Owner:** Member 1 (Repository Intelligence). **Consumers:** Member 2 (verifier), Member 3 (baseline, RL environment).
**Pydantic models:** `src/repoguard/schemas.py`, section 1. **Producer:** `build_index()` in `src/repoguard/repository/indexer.py`.

Any change to a field name or meaning bumps `INDEX_SCHEMA_VERSION` and is agreed with the team first. `load_index()` refuses indexes whose schema version is not in `COMPATIBLE_INDEX_VERSIONS`.

**Changes in v1.1 (Phase 2):** `index_id` and `tool_versions` in every artefact/manifest; non-deterministic build metadata moved to `build_info.json`; `content_sha256` on passages; the evidence contract (§6) with stable evidence ids and provenance; `Retriever` API with filters, diagnostics and `inspect`; staleness checks; hidden directories (`.github/`) skipped by default. **v1.0 indexes must be rebuilt** (`make index`).

---

## 1. Output directory

```
data/indexes/<repo>/
  symbols.json        files, symbols, imports, errors
  callgraph.json      static call edges
  passages.jsonl      retrieval passages (one JSON object per line)
  manifest.json       index_id, provenance, tool versions, effective config, counts, artefact sha256 hashes
  build_info.json     build timestamp, duration, platform  (the only non-deterministic file)
  passages.faiss      FAISS IndexFlatIP over normalised vectors   } one of these, chosen by
  passages.npy        the same vectors as a NumPy matrix          } embeddings.vector_store
  vector_meta.json    index_id, embedding backend/model/dim and the passage-id order of the vectors
```

The vector files exist only when embeddings are enabled. `vector_store: numpy` gives an index that loads on any machine (e.g. Windows where FAISS is blocked); `auto` uses FAISS when importable. Load everything with:

```python
from repoguard.repository import load_index
idx = load_index("data/indexes/schedule")      # -> IndexArtifacts
```

**Conventions**

* Paths are repository-relative POSIX paths (`schedule/__init__.py`), never absolute.
* Lines are 1-based and inclusive; `col` is the 0-based AST column offset.
* Records are sorted (files by path; symbols by file, line, column; edges by file, line, column, caller). Re-running on the same commit with the same config gives **byte-identical** files; only `build_info.json` (timestamps) differs, and it is excluded from the artefact hashes.
* **`index_id`** = `<repository_id>@<commit[:12]>-<config hash[:8]>`, e.g. `schedule@82a43db1b938-66f82ca2`. It changes when the snapshot or anything that affects index content (exclusions, source roots, passage/embedding settings, schema/indexer version) changes; provenance overrides (name, URL, licence) do not change it. Every artefact carries it, and `load_index()` rejects a directory whose files come from different builds.
* Hidden directories (`.github/`, `.devcontainer/` …) are skipped unless `include_hidden_dirs: true`; non-`.py` files are counted in `counts.unsupported_files_skipped`.
* Module names strip a detected source root: `src/requests/api.py` → `requests.api` (`source_roots` in the manifest config shows what was used).

---

## 2. `symbols.json`

```json
{
  "schema_version": "1.0",
  "repository": { ...RepositoryMetadata... },
  "files":   [ ...FileMetadata... ],
  "symbols": [ ...FunctionMetadata | ClassMetadata... ],
  "imports": [ ...ImportInfo... ],
  "errors":  [ ...IndexIssue... ]
}
```

### 2.1 Function / method record (real example from `dbader/schedule`)

```json
{
  "symbol_id": "schedule/__init__.py::schedule.Scheduler.every",
  "name": "every",
  "qualified_name": "schedule.Scheduler.every",
  "type": "method",
  "file": "schedule/__init__.py",
  "module": "schedule",
  "line": 162,
  "end_line": 170,
  "col": 4,
  "decorator_line": null,
  "parent": "schedule.Scheduler",
  "decorators": [],
  "docstring": "Schedule a new periodic job. ...",
  "parameters": ["self", "interval"],
  "parameter_details": [
    {"name": "self", "kind": "positional_or_keyword", "annotation": null, "default": null, "has_default": false},
    {"name": "interval", "kind": "positional_or_keyword", "annotation": "int", "default": "1", "has_default": true}
  ],
  "return_annotation": "'Job'",
  "is_async": false,
  "parent_class": "Scheduler",
  "is_nested": false
}
```

| Field | Meaning |
|---|---|
| `symbol_id` | `<file>::<qualified_name>`, unique. A second definition with the same qualified name in the same file (e.g. inside `if/else`) gets `#2`, `#3`, … |
| `type` | `function` (module-level or nested in a function), `method` (defined directly in a class body), `class` |
| `qualified_name` | `<module>.<Class>.<method>`; nested definitions append to their parent: `pkg.app.App.fetch.inner` |
| `line` / `end_line` | line of the `def`/`class` keyword (decorators excluded) to last line of the body |
| `decorator_line` | first decorator line, or `null` |
| `parent` | qualified name of the enclosing class/function, `null` at module level |
| `parameters` | names in declaration order, **including `self`/`cls`**, without `*`/`**` prefixes. This is the list to use for `parameter_exists` |
| `parameter_details[].kind` | `positional_only`, `positional_or_keyword`, `var_positional` (`*args`), `keyword_only`, `var_keyword` (`**kwargs`) |
| `annotation`, `default`, `return_annotation` | source text via `ast.unparse` (string annotations keep their quotes: `'Job'`) |
| `is_async` | `async def` |
| `parent_class` | short name of the class for methods, else `null` |
| `is_nested` | defined inside another function |

### 2.2 Class record

Same common fields plus `bases` (source text, e.g. `["BaseService"]`, keyword arguments as `metaclass=ABCMeta`), `methods` (short names of methods directly in the body, in order) and `class_variables` (names assigned directly in the class body).

### 2.3 File record

`path`, `module`, `is_package` (`__init__.py`), `line_count`, `size_bytes`, `sha256` (of the raw bytes), `parse_status` (`ok` | `error`), `functions` and `classes` (qualified names), `import_count`.

### 2.4 Import record

```json
{"file": "pkg/app.py", "line": 6, "kind": "from", "module": null, "name": "utils",
 "alias": null, "level": 1, "resolved_module": "pkg", "scope": "<module>"}
```

`kind` is `import` (`import a.b as c` → `module="a.b"`, `alias="c"`) or `from` (`from x import y as z` → `module="x"`, `name="y"`, `alias="z"`). `level` counts leading dots; `resolved_module` is the absolute module after resolving relative imports (`null` if it goes above the top-level package). `scope` is `<module>` or the qualified name of the enclosing function/class for local imports. `from x import *` is recorded with `name="*"`.

### 2.5 Error record (`IndexIssue`)

```json
{"file": "pkg/broken.py", "stage": "parse", "error_type": "SyntaxError", "message": "...", "line": 4}
```

`stage`: `scan` (missing file, broken symlink, path escapes the repo), `read` (I/O or decoding), `parse` (syntax error, null bytes), `extract`, `callgraph`, `passages`. A file with an error has `parse_status: "error"` and contributes no symbols, edges or passages; the rest of the repository is indexed normally.

> **For the verifier:** a claim about a file whose `parse_status` is `error` should be `unverifiable`, not `refuted`; the index has no facts about it.

---

## 3. `callgraph.json`

```json
{
  "schema_version": "1.0",
  "repository_id": "schedule",
  "edges": [
    {"caller": "schedule.every", "caller_name": "every", "callee": "every",
     "callee_expr": "default_scheduler.every", "callee_qualified": "schedule.Scheduler.every",
     "resolution": "method", "file": "schedule/__init__.py", "line": 847, "col": 11}
  ]
}
```

One edge per `ast.Call` node (two calls on one line = two edges; `f(g(x))` = two edges).

| Field | Meaning |
|---|---|
| `caller` | qualified name of the enclosing function/method/class body, or `<module>.<module>` for top-level code |
| `caller_name` | short name (`<module>` for top-level code) |
| `callee` | last name component as written (`do` for `self.job.do()`); `<dynamic>` when the called object is a call result, subscript or lambda |
| `callee_expr` | full call target as written (`ast.unparse`) |
| `callee_qualified` | resolved target, or `null` when `resolution` is `unresolved` |
| `resolution` | how the target was determined, see below |

| `resolution` | Rule | `callee_qualified` |
|---|---|---|
| `local` | name defined in the same module or in an enclosing function | repo symbol |
| `imported` | name bound by `import`/`from … import`, following re-exports (e.g. through `__init__.py`) | repo symbol |
| `method` | `self.m()`, `cls.m()`, `super().m()`, `Class.m()`, or `var.m()` where `var` is a module-level variable assigned exactly once from a repo class constructor; inherited methods resolved through in-repo base classes | repo symbol |
| `module_attribute` | `module.func()` where `module` is an indexed module | repo symbol |
| `external` | target is outside the repository (stdlib or third-party), including re-exports that lead outside | dotted external name, e.g. `urllib.parse.urlparse` |
| `builtin` | Python builtin (`len`, `print`, `ValueError`) | `builtins.<name>` |
| `unresolved` | anything not statically determinable: methods on local variables or attributes (`self.jobs.remove()`), call results, dynamic dispatch | `null` |

Decorators, default values and annotations are attributed to the **enclosing** scope (that is where Python evaluates them).

> **For the verifier:** "A calls B" is supported when an edge has `caller` or `caller_name` equal to A and `callee_qualified` or `callee` equal to B (`IndexArtifacts.callers_of()` / `callees_of()`). An `unresolved` edge with a matching `callee` short name is weaker evidence than a resolved one.

---

## 4. `passages.jsonl`

```json
{"passage_id": "schedule/__init__.py::schedule.Scheduler.every", "file_path": "schedule/__init__.py",
 "start_line": 162, "end_line": 170, "passage_type": "method", "symbol": "schedule.Scheduler.every",
 "symbol_id": "schedule/__init__.py::schedule.Scheduler.every", "part": 1, "total_parts": 1,
 "header": "# schedule/__init__.py | method schedule.Scheduler.every | def every(self, interval: int = 1) -> 'Job' | lines 162-170",
 "content": "    def every(self, interval: int = 1) -> \"Job\":\n ..."}
```

| `passage_type` | Content | `passage_id` |
|---|---|---|
| `function` / `method` | exact source lines `start_line..end_line` (decorators included). Nested functions stay inside their parent's passage | the symbol's `symbol_id` |
| `class` | class overview: header, docstring, class attributes and method signatures, with each method body replaced by `...` | the class `symbol_id` |
| `module` | module overview: docstring, imports, top-level statements, signatures of top-level defs with bodies replaced by `...` | `<file>::<module>` |

Passages longer than `passages.max_lines` (default 80) are split into windows overlapping by `overlap_lines` (default 10); ids get `@1`, `@2`, … and `part`/`total_parts` are set. For `class`/`module` passages the content is elided, so it is **not** a contiguous slice of `start_line..end_line`. `header` is prepended to `content` when embedding (`Passage.embedding_text()`). `content_sha256` is the sha256 of `content` (UTF-8).

---

## 5. `manifest.json`

`index_id`; `repository` (id, name, source URL, commit, licence, status, counts); `tool_versions` (`repoguard`, `index_schema`, `indexer`, `python`, `parser` = `ast (CPython x.y.z)`, `pydantic`, `numpy`, and with embeddings `embedding_backend`, `embedding_model`, `sentence_transformers`, `faiss`); the full effective `config`; `counts` (files, failures, symbols by type, imports, edges by resolution, passages, errors), `embeddings` (backend, model, dim, vector store, files) and `artefacts` (sha256 of each output file). Use the manifest to state exactly which snapshot and settings produced any reported result.

---

## 6. Retrieval and the evidence contract

Other modules use `Retriever` (or the Phase 1 `search()` wrapper). FAISS/NumPy details are internal.

```python
from repoguard.retrieval import Retriever, SearchFilters

r = Retriever.from_dir("data/indexes/schedule", repo_root="data/repos/schedule")   # repo_root optional
res = r.search("how is the next run computed", k=5,
               filters=SearchFilters(path_prefix="schedule/", exclude_tests=True), min_score=0.0)
res.evidence      # list[Evidence], ranked
res.diagnostics   # RetrievalDiagnostics
r.get_evidence(evidence_id)            # resolve a stable id back to its passage
r.inspect("Scheduler")                 # INSPECT / OPEN: full source of a symbol, evidence id or file

from repoguard.retrieval import search   # Phase 1 API, unchanged: list[Evidence]
search("how is the next run computed", idx, k=5)
```

### 6.1 `Evidence`

```json
{"evidence_id": "ev_15849d01508468e3", "passage_id": "schedule/__init__.py::schedule.Scheduler.idle_seconds",
 "file_path": "schedule/__init__.py", "start_line": 197, "end_line": 206,
 "snippet": "    @property\n    def idle_seconds(self) -> Optional[float]: ...",
 "similarity_score": 0.31, "rank_score": 0.61, "associated_symbol": "schedule.Scheduler.idle_seconds",
 "rank": 1, "passage_type": "method", "part": 1, "total_parts": 1,
 "content_sha256": "…", "approx_tokens": 90,
 "index_id": "schedule@82a43db1b938-66f82ca2", "repository_id": "schedule",
 "commit": "82a43db1b938d8fdf60103bd41f329e06c8d3651",
 "source_url": "https://github.com/dbader/schedule/blob/82a43db…/schedule/__init__.py#L197-L206"}
```

| Field | Contract |
|---|---|
| `evidence_id` | `ev_` + first 16 hex of sha256(`index_id` \| `passage_id` \| `content_sha256`). **Stable** for the same passage of the same index across runs, reloads and identical rebuilds; different when the index id or content differs |
| `file_path`, `start_line`, `end_line` | repository-relative POSIX path; 1-based inclusive lines at the pinned commit |
| `snippet` | the passage content; for function/method passages an exact source slice |
| `associated_symbol` | qualified symbol (`None` for module overviews) |
| `similarity_score` | **raw cosine** clipped to [0, 1] (unchanged by boosts; use this for thresholds and RL signals) |
| `rank_score` | the ordering score: cosine + boosts (§6.3) |
| `content_sha256`, `approx_tokens` | integrity hash of `snippet`; token estimate = ⌈chars / 4⌉ (model-agnostic) |
| `index_id`, `repository_id`, `commit`, `source_url` | provenance; `source_url` is a GitHub permalink to the exact lines when the source is on GitHub, else `null` |

### 6.2 `SearchFilters` and `RetrievalDiagnostics`

Filters (all optional, combined with AND, applied before top-k): `path_prefix`, `path_glob` (fnmatch), `passage_types` (`function`/`method`/`class`/`module`), `symbol` (short, qualified or dotted suffix), `exclude_tests` (`test`/`tests` directories, `test_*.py`, `*_test.py`, `conftest.py`).

Diagnostics per search: `query_chars`, `k_requested`, `candidates` (passages after filtering), `returned`, `duplicates_removed` (identical content), `below_min_score`, `empty`, `top1_score` (max cosine), `retrieved_tokens`, `latency_ms`, `backend`, `vector_store`.

### 6.3 Ranking

Search is **exact**: every passage is scored, filtered, de-duplicated by `content_sha256`, then the top-k by `rank_score` are returned (ties: index order), so results are deterministic.

`rank_score = cosine + 0.30 if the passage defines a symbol named in the query, or + 0.10 if it mentions one as an identifier.` Named symbols are detected as in `docs/state_signals.md` §3. Pass `definition_boost=0, mention_boost=0` for pure vector search. The defaults were chosen on the development repositories only; the held-out `requests` results are in `reports/retrieval/`.

### 6.4 `inspect` (for the INSPECT / OPEN action)

`inspect(target, max_lines=200)` takes an evidence id, passage id, symbol id, qualified/short symbol name or an indexed file path:

* function / method → the whole definition, with all passage parts stitched together (no `repo_root` needed)
* class / module / file → the real source lines; needs `repo_root`. Without it, a class falls back to its overview passage
* output is capped at `max_lines`; `similarity_score = 0`; `passage_id = inspect:<target>:<start>-<end>`

### 6.5 Errors and robustness

| Situation | Behaviour |
|---|---|
| blank query | empty `SearchResult` (`diagnostics.empty = true`) |
| non-string query, `k` outside 1..200 (or bool/float), `min_score` outside [0, 1] | `InvalidQueryError` (a `ValueError`) |
| query longer than 4000 characters | truncated; `query_chars` shows the length used |
| empty repository / no passages | empty results; signals all 0 with `missing_evidence = 1` |
| `inspect` on unknown target, path outside the index, `../..` | `InvalidQueryError` |
| file changed or deleted since indexing (`inspect`) | `StaleIndexError` |
| `vector_meta.json` from another build or reordered ids | `StaleIndexError` on load — never silently wrong evidence |
| FAISS file but FAISS not importable | `EmbeddingBackendError` telling you to rebuild with `vector_store: numpy` |
| index files from different builds | `IndexNotFoundError` on `load_index` |

Whole-repository staleness: `check_freshness(idx, repo_path)` (or `scripts/search_index.py --freshness`) re-scans with the index's own config and reports changed / added / removed files and a changed git commit.

### 6.6 Embedding backends

| Backend | When | Notes |
|---|---|---|
| `sentence-transformers` | `all-MiniLM-L6-v2`, needs `pip install -e .[ml]` and the model download | the agreed semantic embedder |
| `hashing` | no ML stack / offline / tests | deterministic signed feature hashing over code-aware tokens; lexical, **not semantic** |

`backend: auto` tries sentence-transformers and falls back to hashing with a warning. Always report the backend (`manifest.json → tool_versions.embedding_backend`) alongside retrieval results.

### 6.7 Fixture index for other modules

`tests/fixtures/indexes/indexer_repo/` is a committed, prebuilt index (hashing embeddings, NumPy vectors) of `tests/fixtures/indexer_repo/`. Use it in other modules' tests and mocks instead of building indexes. `test_committed_fixture_index_is_current` fails if it drifts; rebuild it from the repo root with:

```bash
python -c "from repoguard.repository import build_index; build_index('tests/fixtures/indexer_repo', 'tests/fixtures/indexes/indexer_repo', {'embeddings': {'enabled': True, 'backend': 'hashing', 'vector_store': 'numpy'}})"
```

---

## 7. Verifier lookup recipes

| Claim | Lookup |
|---|---|
| `file_exists` | `path in idx.file_paths()` |
| `function_exists` / `class_exists` | `idx.find_symbols(name, file=..., type=...)` (matches short or qualified name) |
| `parameter_exists` | `param in symbol.parameters` |
| `calls` | `idx.callees_of(caller)` / `idx.callers_of(callee)` |
| `imports` | filter `idx.imports` by `file` and `module` / `name` / `resolved_module` |

---

## 8. Known limitations

* **The call graph is a static approximation.** It does not resolve dynamic dispatch on local variables or instance attributes (`self.jobs.remove()`), `getattr`/reflection, monkey-patching, callbacks passed as values, decorators that replace functions, or runtime-generated code. These edges are recorded as `unresolved` rather than guessed. It is not complete for every Python runtime behaviour.
* Module-level instance inference (`x = Class()`) is used only when the variable is bound exactly once at module level and not shadowed by a local parameter or import.
* Only `.py` files are indexed; stubs (`.pyi`), notebooks and C extensions are not.
* Code that differs by Python version or platform is indexed as written: both branches of an `if sys.version_info …` definition appear (with `#2` ids).
* Parsing uses the Python running the indexer. Syntax newer than that interpreter appears as a `parse` error.
* Passage ids are stable for a given commit; editing a file shifts line numbers but symbol-based ids stay the same unless the symbol is renamed or moved.
* Hidden directories are skipped by default, so CI helpers such as `cachetools/.github/copilot-bench.py` are not indexed.
* `approx_tokens` is a character-based estimate, not the LLM tokenizer's count; Member 3's adapter measures real tokens.
