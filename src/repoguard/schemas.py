"""Shared data contracts for RepoGuard.

Each section is owned by one team member. Change another member's section only after
agreeing the interface change with them (see docs/team_responsibilities.md).

Section 1 (repository intelligence & retrieval) is owned by Member 1 and is what the
indexer writes to ``symbols.json``, ``callgraph.json`` and ``passages.jsonl``.
Its on-disk format is documented in ``docs/index_schema.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# 1. Repository intelligence & retrieval  (owner: Member 1)
# =============================================================================

INDEX_SCHEMA_VERSION = "1.0"

MODULE_SCOPE = "<module>"
"""Pseudo symbol name used as the caller for calls made at module top level."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class ParameterKind(str, Enum):
    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"  # *args
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"  # **kwargs


class ParameterInfo(_Model):
    name: str
    kind: ParameterKind
    annotation: str | None = None
    default: str | None = None
    """Source text of the default value; ``None`` when there is no default."""
    has_default: bool = False


class SymbolMetadata(_Model):
    """Fields common to every extracted definition."""

    symbol_id: str
    """Stable id: ``<file>::<qualified_name>``; a ``#2``, ``#3``... suffix is added when the
    same qualified name is defined more than once in a file (e.g. conditional definitions)."""
    name: str
    qualified_name: str
    """Dotted path: ``<module>.<Class>.<method>``; nested defs use their enclosing names."""
    type: str
    file: str
    """Repository-relative POSIX path."""
    module: str
    line: int
    """1-based line of the ``def``/``class`` keyword (decorators excluded)."""
    end_line: int
    col: int = 0
    decorator_line: int | None = None
    """First decorator line, when decorated."""
    parent: str | None = None
    """Qualified name of the enclosing class or function, ``None`` at module level."""
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None


class FunctionMetadata(SymbolMetadata):
    type: Literal["function", "method"]
    parameters: list[str] = Field(default_factory=list)
    """Parameter names in declaration order (``self``/``cls`` included, no ``*`` prefixes)."""
    parameter_details: list[ParameterInfo] = Field(default_factory=list)
    return_annotation: str | None = None
    is_async: bool = False
    parent_class: str | None = None
    """Short name of the directly enclosing class for methods."""
    is_nested: bool = False
    """True when defined inside another function."""


class ClassMetadata(SymbolMetadata):
    type: Literal["class"]
    bases: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    """Short names of methods defined directly in the class body."""
    class_variables: list[str] = Field(default_factory=list)


Symbol = Annotated[FunctionMetadata | ClassMetadata, Field(discriminator="type")]


class ImportInfo(_Model):
    file: str
    line: int
    kind: Literal["import", "from"]
    module: str | None
    """Module as written (``None`` for ``from . import x``)."""
    name: str | None = None
    """Imported name for ``from`` imports, ``None`` for plain ``import m``."""
    alias: str | None = None
    level: int = 0
    """Relative-import level (number of leading dots)."""
    resolved_module: str | None = None
    """Absolute dotted module after resolving relative imports."""
    scope: str = MODULE_SCOPE
    """Qualified name of the enclosing function/class, or ``<module>``."""


class CallResolution(str, Enum):
    LOCAL = "local"  # function/class defined in the same module
    IMPORTED = "imported"  # name imported from another indexed module
    METHOD = "method"  # self.x() / cls.x() / Class.x() resolved within the repo
    MODULE_ATTRIBUTE = "module_attribute"  # mod.func() where mod is an indexed module
    EXTERNAL = "external"  # imported from a module that is not part of the repo
    BUILTIN = "builtin"  # Python builtin such as len() or print()
    UNRESOLVED = "unresolved"  # could not be determined statically


class CallEdge(_Model):
    caller: str
    """Qualified name of the calling function/class, or ``<module-qualname>.<module>``."""
    caller_name: str
    """Short name of the caller (``<module>`` for top-level code)."""
    callee: str
    """Short name of what is called (last component, e.g. ``do`` for ``self.job.do()``);
    ``<dynamic>`` when the called object is a call result, subscript or lambda."""
    callee_expr: str
    """Full call target as written (e.g. ``self.job.do``)."""
    callee_qualified: str | None = None
    """Qualified name of the callee when resolved to a repo symbol or import."""
    resolution: CallResolution
    file: str
    line: int
    col: int = 0


class IndexIssue(_Model):
    """A recorded, non-fatal problem encountered while indexing."""

    file: str
    stage: Literal["scan", "read", "parse", "extract", "callgraph", "passages"]
    error_type: str
    message: str
    line: int | None = None


class FileMetadata(_Model):
    path: str
    module: str
    is_package: bool = False
    line_count: int = 0
    size_bytes: int = 0
    sha256: str = ""
    parse_status: Literal["ok", "error"] = "ok"
    functions: list[str] = Field(default_factory=list)
    """Qualified names of functions and methods defined in the file."""
    classes: list[str] = Field(default_factory=list)
    import_count: int = 0


class Passage(_Model):
    passage_id: str
    file_path: str
    start_line: int
    end_line: int
    passage_type: Literal["function", "method", "class", "module"]
    symbol: str | None = None
    """Qualified name of the symbol the passage belongs to (``None`` for module passages)."""
    symbol_id: str | None = None
    part: int = 1
    total_parts: int = 1
    header: str = ""
    """One-line context (path, symbol, signature) prepended for embedding."""
    content: str

    def embedding_text(self) -> str:
        return f"{self.header}\n{self.content}" if self.header else self.content


class Evidence(_Model):
    evidence_id: str
    passage_id: str
    file_path: str
    start_line: int
    end_line: int
    snippet: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    associated_symbol: str | None = None
    rank: int = 1


class RepositoryMetadata(_Model):
    repository_id: str
    name: str
    repo_path: str
    source_url: str | None = None
    commit: str | None = None
    licence: str | None = None
    status: Literal["pending", "indexing", "ready", "failed"] = "pending"
    total_files: int = 0
    indexed_files: int = 0
    failed_files: int = 0
    total_symbols: int = 0


class IndexArtifacts(_Model):
    """Paths of the written artefacts plus the loaded index content."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = INDEX_SCHEMA_VERSION
    output_dir: str
    symbols_path: str
    callgraph_path: str
    passages_path: str
    manifest_path: str
    vector_index_path: str | None = None
    vector_ids_path: str | None = None
    repository: RepositoryMetadata
    files: list[FileMetadata] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportInfo] = Field(default_factory=list)
    call_edges: list[CallEdge] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    errors: list[IndexIssue] = Field(default_factory=list)

    # ---- convenience lookups (read-only helpers for the verifier / agent) ----

    def file_paths(self) -> set[str]:
        return {f.path for f in self.files}

    def find_symbols(
        self, name: str, file: str | None = None, type: str | None = None
    ) -> list[FunctionMetadata | ClassMetadata]:
        """Match by short name or qualified name, optionally filtered by file/type."""
        out = []
        for s in self.symbols:
            if name not in (s.name, s.qualified_name):
                continue
            if file is not None and s.file != file:
                continue
            if type is not None and s.type != type:
                continue
            out.append(s)
        return out

    def callers_of(self, callee: str) -> list[CallEdge]:
        """Edges whose callee short name or resolved qualified name equals ``callee``."""
        return [e for e in self.call_edges if callee in (e.callee, e.callee_qualified)]

    def callees_of(self, caller: str) -> list[CallEdge]:
        """Edges whose caller short name or qualified name equals ``caller``."""
        return [e for e in self.call_edges if caller in (e.caller_name, e.caller)]


# =============================================================================
# 3. LLM, Baselines & Benchmark Datasets (owner: Member 3)
# =============================================================================

class GenerationResult(_Model):
    """Result of running inference with the frozen code LLM."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    entropy: float = 0.0
    min_token_prob: float = 1.0
    finish_reason: str = "stop"
    is_mock: bool = False
    model_name: str = ""


class RLAction(str, Enum):
    """Canonical 6-action space for the RepoGuard RL controller."""

    SEARCH = "search"
    INSPECT = "inspect"
    VERIFY = "verify"
    ANSWER = "answer"
    ANSWER_WITH_EVIDENCE = "answer_with_evidence"
    ABSTAIN = "abstain"


class StateVector(_Model):
    """Standard 18-dimensional normalized state vector observation for RepoGuard policy."""

    # Free signals (6)
    symbol_found: float = 0.0
    name_similarity: float = 0.0
    retr_top1: float = 0.0
    retr_margin: float = 0.0
    passage_mentions_symbol: float = 0.0
    stack_match: float = 1.0

    # Self-check probabilities (4)
    p_exists: float = 0.0
    p_no_code_needed: float = 0.0
    p_in_scope: float = 0.0
    p_missing_details: float = 0.0

    # Generation uncertainty (3)
    entropy: float = 0.0
    min_token_prob: float = 0.0
    sample_consistency: float = 0.0

    # Progress flags (3)
    did_search: float = 0.0
    did_inspect: float = 0.0
    did_verify: float = 0.0

    # Resource budget & Risk slider (2)
    tokens_spent: float = 0.0
    risk_lambda: float = 1.0

    def to_tensor(self) -> list[float]:
        return list(self.model_dump().values())


class BaselineType(str, Enum):
    ALWAYS_ANSWER = "always_answer"
    ALWAYS_EVIDENCE = "always_evidence"
    THRESHOLD_GATE = "threshold_gate"


class BenchmarkQuestion(_Model):
    """A repository-grounded evaluation question with ground truth annotations."""

    question_id: str
    repository: str
    commit: str
    question_type: str
    question: str
    expected_action: Literal["answer", "answer_with_evidence", "abstain"]
    is_trap: bool
    ground_truth_answer: str
    target_symbol: str | None = None
    target_file: str | None = None
    relevant_lines: list[int] = Field(default_factory=list)


class BaselineRunOutput(_Model):
    """Recorded trace of executing a baseline against a single question."""

    question_id: str
    baseline: BaselineType
    model_name: str
    question: str
    answer: str
    abstained: bool
    evidence_ids: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    is_mock: bool = False

