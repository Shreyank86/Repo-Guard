"""Retrieval: embeddings, vector search, evidence packaging and RL free signals.

Stable public interface (FAISS/NumPy details stay internal):

    Retriever, SearchFilters, SearchResult, Evidence   -- search / inspect
    SignalExtractor, compute_free_signals, FreeSignals -- RL state signals
    search(question, index, k)                         -- Phase 1 functional API
"""

from repoguard.retrieval.retriever import Retriever, get_retriever, make_evidence_id
from repoguard.retrieval.search import search
from repoguard.retrieval.signals import (
    SignalExtractor,
    compute_free_signals,
    extract_named_files,
    extract_named_symbols,
)
from repoguard.schemas import Evidence, FreeSignals, SearchFilters, SearchResult

__all__ = [
    "Evidence",
    "FreeSignals",
    "Retriever",
    "SearchFilters",
    "SearchResult",
    "SignalExtractor",
    "compute_free_signals",
    "extract_named_files",
    "extract_named_symbols",
    "get_retriever",
    "make_evidence_id",
    "search",
]
