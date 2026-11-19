# Repository Intelligence Module (`src/repoguard/repository/`)

**Owner: Member 1 (Repository Intelligence Engineer)**
**Branch: `feature/indexer`**

---

## Purpose
This package is responsible for safely scanning Python source repositories, performing Abstract Syntax Tree (AST) analysis, extracting code entities and signatures, building static call graphs, and partitioning files into retrieval passages.

---

## Files to Implement

### 1. `parser.py`
- **Owner**: Member 1
- **Responsibility**:
  - Recursively locate all `.py` files in the repository.
  - Implement security checks to prevent directory traversal outside the repository root.
  - Parse files using Python standard `ast` module.
  - Extract functions: name, parameters (with type annotations and defaults), return types, docstrings, start and end line numbers.
  - Extract classes: name, base classes, methods, class variables, line numbers.
  - Extract imports: module names and imported symbols.
- **Output**: List of `FileMetadata`, `FunctionMetadata`, `ClassMetadata`.

### 2. `callgraph.py`
- **Owner**: Member 1
- **Responsibility**:
  - Analyze AST call nodes (`ast.Call`) inside functions and methods.
  - Resolve caller symbol to callee function name where statically determinable.
  - Construct a directed graph representing function-to-function invocation.
- **Output**: List of `CallEdge` objects, serialized into `callgraph.json`.

### 3. `passage_builder.py`
- **Owner**: Member 1
- **Responsibility**:
  - Convert parsed functions, classes, and module-level code into structured `Passage` chunks.
  - Attach file path, start line, end line, and symbol identifiers to every passage.
- **Output**: List of `Passage` objects, serialized to `passages.jsonl`.

### 4. `indexer.py`
- **Owner**: Member 1
- **Responsibility**:
  - Orchestrate parsing, call graph generation, and passage creation.
  - Save outputs into `symbols.json`, `callgraph.json`, and `passages.jsonl` in target index directory.
- **Primary Function Contract**:
  ```python
  def build_index(repo_path: str, output_dir: str, config: dict | None = None) -> IndexArtifacts:
      """Scan repository, extract metadata, and produce index artifacts."""
  ```

---

## Associated Tests
Implement and pass tests in `tests/unit/test_parser.py`, `tests/unit/test_callgraph.py`, and `tests/unit/test_indexer.py`.

