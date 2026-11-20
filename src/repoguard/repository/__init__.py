"""Repository intelligence: scanning, AST parsing, call graph, passages, indexing."""

from repoguard.repository.indexer import build_index, load_index

__all__ = ["build_index", "load_index"]
