"""Embeddings, vector index and search."""

from __future__ import annotations

import numpy as np
import pytest

from repoguard.exceptions import EmbeddingBackendError
from repoguard.repository import build_index, load_index
from repoguard.retrieval import search
from repoguard.retrieval.embedder import HashingEmbedder, code_tokens, get_embedder
from repoguard.retrieval.search import VectorIndex, faiss


def test_code_tokens_split_identifiers():
    toks = code_tokens("def loadConfig(file_path): return HTTPAdapter")
    assert {
        "loadconfig",
        "load",
        "config",
        "file_path",
        "file",
        "path",
        "httpadapter",
        "http",
        "adapter",
    } <= set(toks)
    assert "def" not in toks and "return" not in toks


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a = emb.encode(["load the config file", ""])
    b = HashingEmbedder(dim=64).encode(["load the config file", ""])
    assert a.dtype == np.float32 and a.shape == (2, 64)
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0) and np.linalg.norm(a[1]) == 0.0


def test_get_embedder_backends():
    assert get_embedder({"backend": "hashing", "hashing_dim": 32}).dim == 32
    with pytest.raises(EmbeddingBackendError):
        get_embedder({"backend": "nope"})


@pytest.mark.parametrize(
    "use_faiss",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(faiss is None, reason="faiss not installed")),
    ],
)
def test_vector_index_exact_cosine(use_faiss):
    vecs = np.eye(3, dtype=np.float32)
    vi = VectorIndex(vecs, ["a", "b", "c"], use_faiss=use_faiss)
    res = vi.query(np.array([0.1, 0.9, 0.0], dtype=np.float32), k=5)
    assert [r[0] for r in res] == ["b", "a", "c"]
    assert vi.query(vecs[0], k=0) == []


@pytest.fixture
def index_with_vectors(fixture_repo, tmp_path):
    return build_index(
        fixture_repo, tmp_path / "idx", {"embeddings": {"enabled": True, "backend": "hashing"}}
    )


def test_search_returns_evidence_for_relevant_passage(index_with_vectors):
    results = search("load a JSON config file", index_with_vectors, k=3)
    assert len(results) == 3
    assert results[0].associated_symbol == "pkg.config.load_config"
    assert results[0].file_path == "pkg/config.py" and (
        results[0].start_line,
        results[0].end_line,
    ) == (9, 14)
    assert [r.rank for r in results] == [1, 2, 3]
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)


def test_search_after_reload_uses_persisted_vectors(index_with_vectors):
    loaded = load_index(index_with_vectors.output_dir)
    assert loaded.vector_index_path and loaded.vector_ids_path
    before = [r.passage_id for r in search("helper value", index_with_vectors, k=4)]
    after = [r.passage_id for r in search("helper value", loaded, k=4)]
    assert before == after and after[0] == "pkg/utils.py::pkg.utils.helper"


def test_search_without_vectors_falls_back_to_lexical(fixture_repo, tmp_path):
    art = build_index(fixture_repo, tmp_path / "plain")
    assert art.vector_index_path is None
    (top,) = search("initialize app", art, k=1)
    assert top.associated_symbol == "pkg.app.App.initialize_app"


def test_search_edge_cases(index_with_vectors):
    assert search("   ", index_with_vectors) == []
    assert len(search("config", index_with_vectors, k=1000)) == len(index_with_vectors.passages)
    with pytest.raises(ValueError):
        search("config", index_with_vectors, embedder=HashingEmbedder(dim=32))
