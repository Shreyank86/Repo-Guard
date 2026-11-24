"""Exception hierarchy shared across RepoGuard modules."""

from __future__ import annotations


class RepoGuardError(Exception):
    """Base class for all RepoGuard errors."""


# --- Repository intelligence (Member 1) -------------------------------------


class RepositoryNotFoundError(RepoGuardError):
    """The repository root does not exist or is not a directory."""


class PathTraversalError(RepoGuardError):
    """A path resolved to a location outside the repository root."""


class IndexingError(RepoGuardError):
    """Indexing failed as a whole (individual file failures are recorded, not raised)."""


class IndexNotFoundError(RepoGuardError):
    """An index directory or one of its required artefacts is missing."""


class EmbeddingBackendError(RepoGuardError):
    """The requested embedding / vector-search backend is unavailable."""


# --- Verification (Member 2) ------------------------------------------------


class VerificationError(RepoGuardError):
    """Verification against the repository index failed."""


# --- LLM, baselines and RL (Member 3) ---------------------------------------


class ModelInferenceError(RepoGuardError):
    """LLM model loading, tokenization or generation failed."""


class BaselineError(RepoGuardError):
    """Baseline execution or evaluation failed."""


class AgentError(RepoGuardError):
    """RL environment transition or policy execution failed."""


# --- Retrieval (Member 1, Phase 2) ------------------------------------------


class InvalidQueryError(RepoGuardError, ValueError):
    """A search/inspect request had an invalid query, k, filter or target."""


class StaleIndexError(RepoGuardError):
    """The repository on disk no longer matches the index (or vectors no longer match passages)."""
