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
