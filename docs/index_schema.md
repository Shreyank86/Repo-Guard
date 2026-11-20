# Repository Index Schema (v1.0)

**Owner:** Member 1 (Repository Intelligence). **Consumers:** Member 2 (verifier), Member 3 (baseline, RL environment).
**Pydantic models:** `src/repoguard/schemas.py`, section 1. **Producer:** `build_index()` in `src/repoguard/repository/indexer.py`.

Any change to a field name or meaning bumps `INDEX_SCHEMA_VERSION` and is agreed with the team first. `load_index()` refuses indexes written with a different schema version.

---

## 1. Output directory

```
data/indexes/<repo>/
  symbols.json        files, symbols, imports, errors
  callgraph.json      static call edges
  passages.jsonl      retrieval passages (one JSON object per line)
  manifest.json       provenance, effective config, counts, artefact sha256 hashes
  passages.faiss      FAISS IndexFlatIP over normalised vectors (or passages.npy without faiss)
  vector_meta.json    embedding backend/model/dim and the passage-id order of the vectors
```

The last two exist only when embeddings are enabled. Load everything with:

```python
from repoguard.repository import load_index
idx = load_index("data/indexes/schedule")      # -> IndexArtifacts
```

**Conventions**

* Paths are repository-relative POSIX paths (`schedule/__init__.py`), never absolute.
* Lines are 1-based and inclusive; `col` is the 0-based AST column offset.
* Records are sorted (files by path; symbols by file, line, column; edges by file, line, column, caller). Re-running on the same commit with the same config gives **byte-identical** files. No timestamps are written.
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

Passages longer than `passages.max_lines` (default 80) are split into windows overlapping by `overlap_lines` (default 10); ids get `@1`, `@2`, … and `part`/`total_parts` are set. For `class`/`module` passages the content is elided, so it is **not** a contiguous slice of `start_line..end_line`. `header` is prepended to `content` when embedding (`Passage.embedding_text()`).

---

## 5. `manifest.json`

`repository` (id, name, source URL, commit, licence, status, counts), the full effective `config`, `counts` (files, failures, symbols by type, imports, edges by resolution, passages, errors), `embeddings` (backend, model, dim, vector store, files) and `artefacts` (sha256 of each output file). Use the manifest to state exactly which snapshot and settings produced any reported result.

---

## 6. Retrieval

```python
from repoguard.retrieval import search
evidence = search("how is the next run computed", idx, k=5)   # -> list[Evidence]
```

`Evidence`: `evidence_id`, `passage_id`, `file_path`, `start_line`, `end_line`, `snippet` (passage content), `similarity_score` (cosine clipped to [0, 1]), `associated_symbol`, `rank`. Query embeddings use the same backend recorded in `vector_meta.json`. If an index was built without embeddings, `search` builds an in-memory lexical (hashing) index instead.

| Backend | When | Notes |
|---|---|---|
| `sentence-transformers` | `all-MiniLM-L6-v2`, needs `pip install -e .[ml]` and the model download | the agreed semantic embedder |
| `hashing` | no ML stack / offline / tests | deterministic signed feature hashing over code-aware tokens; lexical, **not semantic** |

`backend: auto` tries sentence-transformers and falls back to hashing with a warning. Always report the backend in `manifest.json → embeddings` alongside retrieval results.

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
