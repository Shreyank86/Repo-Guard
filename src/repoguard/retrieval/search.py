"""Vector store and the backward-compatible ``search()`` entry point.

Other modules should use :class:`repoguard.retrieval.Retriever` (or ``search()``); FAISS
and NumPy details stay inside this file.

* FAISS ``IndexFlatIP`` over L2-normalised vectors = exact cosine search.
* NumPy matrix with the same semantics when FAISS is unavailable or
  ``embeddings.vector_store: numpy`` is configured (portable: loads on any machine).
* On load, the stored passage-id order and index id must match the index exactly;
  otherwise :class:`StaleIndexError` is raised instead of returning wrong evidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from repoguard.exceptions import EmbeddingBackendError, IndexNotFoundError, StaleIndexError
from repoguard.retrieval.embedder import Embedder, get_embedder
from repoguard.schemas import Evidence, IndexArtifacts

logger = logging.getLogger(__name__)

FAISS_FILE = "passages.faiss"
NUMPY_FILE = "passages.npy"
META_FILE = "vector_meta.json"
VECTOR_FILES = (FAISS_FILE, NUMPY_FILE, META_FILE)

try:  # optional dependency
    import faiss  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ImportError, or a DLL blocked by OS policy
    faiss = None


def faiss_available() -> bool:
    return faiss is not None


class VectorIndex:
    """Exact inner-product index over normalised vectors."""

    def __init__(self, vectors: np.ndarray, ids: list[str], use_faiss: bool | None = None) -> None:
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise ValueError("vectors must be 2-D with one row per id")
        self.ids = ids
        self.dim = int(vectors.shape[1])
        self.use_faiss = (faiss is not None) if use_faiss is None else use_faiss
        if self.use_faiss and faiss is None:
            raise EmbeddingBackendError(
                "vector_store 'faiss' requested but faiss is not importable"
            )
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        if self.use_faiss and len(ids):
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self._matrix)

    @property
    def backend(self) -> str:
        return "faiss" if self.use_faiss else "numpy"

    def __len__(self) -> int:
        return len(self.ids)

    def scores(self, vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of ``vec`` to every stored vector, aligned with ``ids``."""
        if not self.ids:
            return np.zeros(0, dtype=np.float32)
        q = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            dists, idx = self._faiss.search(q, len(self.ids))
            out = np.empty(len(self.ids), dtype=np.float32)
            out[idx[0]] = dists[0]
            return out
        return self._matrix @ q[0]

    def query(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (id, score), highest first; ties broken by insertion order."""
        if not self.ids or k <= 0:
            return []
        sims = self.scores(vec)
        order = np.lexsort((np.arange(len(sims)), -sims))[: min(k, len(sims))]
        return [(self.ids[i], float(sims[i])) for i in order]

    def save(self, out_dir: Path) -> str:
        if self.use_faiss:
            if self._faiss is not None:
                faiss.write_index(self._faiss, str(out_dir / FAISS_FILE))
            else:  # empty index: faiss cannot persist an empty flat index usefully
                np.save(out_dir / NUMPY_FILE, self._matrix)
                (out_dir / FAISS_FILE).unlink(missing_ok=True)
                return NUMPY_FILE
            (out_dir / NUMPY_FILE).unlink(missing_ok=True)
            return FAISS_FILE
        np.save(out_dir / NUMPY_FILE, self._matrix)
        (out_dir / FAISS_FILE).unlink(missing_ok=True)
        return NUMPY_FILE

    @classmethod
    def load(cls, index_file: Path, ids: list[str]) -> VectorIndex:
        if not index_file.is_file():
            raise IndexNotFoundError(f"Vector file missing: {index_file}")
        if index_file.suffix == ".faiss":
            if faiss is None:
                raise EmbeddingBackendError(
                    "This index stores vectors in FAISS format but faiss cannot be imported "
                    "here. Rebuild with embeddings.vector_store: numpy for a portable index."
                )
            fidx = faiss.read_index(str(index_file))
            vectors = (
                fidx.reconstruct_n(0, fidx.ntotal)
                if fidx.ntotal
                else np.zeros((0, fidx.d), np.float32)
            )
            return cls(vectors, ids, use_faiss=True)
        return cls(np.load(index_file), ids, use_faiss=False)


def _resolve_store(choice: str) -> bool:
    if choice == "numpy":
        return False
    if choice == "faiss":
        if faiss is None:
            raise EmbeddingBackendError(
                "embeddings.vector_store is 'faiss' but faiss is not importable"
            )
        return True
    if choice == "auto":
        return faiss is not None
    raise EmbeddingBackendError(f"Unknown vector_store {choice!r} (auto | faiss | numpy)")


def _embedding_versions(embedder: Embedder) -> dict[str, str]:
    from repoguard.repository.indexer import _pkg_version

    return {
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model_name,
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "faiss": getattr(faiss, "__version__", "not available") if faiss else "not available",
    }


def build_vector_index(artifacts: IndexArtifacts, emb_cfg: dict[str, Any]) -> dict[str, Any]:
    """Embed all passages, write the vector store + meta file, and update ``artifacts``."""
    out = Path(artifacts.output_dir)
    use_faiss = _resolve_store(emb_cfg.get("vector_store", "auto"))
    embedder = get_embedder(emb_cfg)
    passages = artifacts.passages
    texts = [p.embedding_text() for p in passages]
    vectors = embedder.encode(texts) if texts else np.zeros((0, embedder.dim), np.float32)
    index = VectorIndex(vectors, [p.passage_id for p in passages], use_faiss=use_faiss)
    index_file = index.save(out)
    meta = {
        "index_id": artifacts.index_id,
        "backend": embedder.backend,
        "model": embedder.model_name,
        "device": emb_cfg.get("device", "cpu"),
        "dim": embedder.dim,
        "count": len(passages),
        "vector_store": "faiss" if index_file == FAISS_FILE else "numpy",
        "ids": index.ids,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    artifacts.vector_index_path = str(out / index_file)
    artifacts.vector_ids_path = str(out / META_FILE)
    return {
        "backend": embedder.backend,
        "model": embedder.model_name,
        "dim": embedder.dim,
        "vector_store": meta["vector_store"],
        "index_file": index_file,
        "meta_file": META_FILE,
        "files": [index_file, META_FILE],
        "versions": _embedding_versions(embedder),
    }


def load_vector_store(index: IndexArtifacts) -> tuple[VectorIndex, dict[str, Any]] | None:
    """Load persisted vectors, checking they belong to exactly this index's passages."""
    if not (index.vector_index_path and index.vector_ids_path):
        return None
    meta = json.loads(Path(index.vector_ids_path).read_text(encoding="utf-8"))
    expected = [p.passage_id for p in index.passages]
    if meta.get("index_id") != index.index_id or meta.get("ids") != expected:
        raise StaleIndexError(
            f"Vectors in {index.vector_ids_path} do not match the passages of index "
            f"{index.index_id!r}; rebuild the index."
        )
    return VectorIndex.load(Path(index.vector_index_path), meta["ids"]), meta


# ---------------------------------------------------------------------------
# backward-compatible functional API
# ---------------------------------------------------------------------------


def search(
    question: str, index: IndexArtifacts, k: int = 5, embedder: Embedder | None = None
) -> list[Evidence]:
    """Return the ``k`` passages most similar to ``question`` as :class:`Evidence`.

    Thin wrapper over :class:`repoguard.retrieval.Retriever` kept for Phase 1 callers.
    A blank question or ``k <= 0`` returns ``[]``; ``k`` above ``MAX_K`` is clamped.
    """
    from repoguard.retrieval.retriever import get_retriever

    if not isinstance(question, str) or not question.strip() or k <= 0:
        return []
    from repoguard.retrieval.retriever import MAX_K

    return get_retriever(index, embedder=embedder).search(question, k=min(k, MAX_K)).evidence
