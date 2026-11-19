"""Vector index construction and passage search.

Public API::

    build_vector_index(artifacts, embeddings_cfg) -> dict      # used by build_index
    search(question, index, k=5, embedder=None) -> list[Evidence]

FAISS (``IndexFlatIP`` over L2-normalised vectors = exact cosine search) is used when
``faiss`` is importable; otherwise an equivalent NumPy matrix is stored and searched.
If an index has no vectors at all, :func:`search` embeds the passages on the fly with
the hashing embedder so retrieval still works for small repositories and tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from repoguard.exceptions import IndexNotFoundError
from repoguard.retrieval.embedder import Embedder, HashingEmbedder, embedder_from_meta, get_embedder
from repoguard.schemas import Evidence, IndexArtifacts, Passage

logger = logging.getLogger(__name__)

FAISS_FILE = "passages.faiss"
NUMPY_FILE = "passages.npy"
META_FILE = "vector_meta.json"
VECTOR_FILES = (FAISS_FILE, NUMPY_FILE, META_FILE)

try:  # optional dependency
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


class VectorIndex:
    """Exact inner-product index over normalised vectors."""

    def __init__(self, vectors: np.ndarray, ids: list[str], use_faiss: bool | None = None) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        self.ids = ids
        self.dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        self.use_faiss = (faiss is not None) if use_faiss is None else use_faiss
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        if self.use_faiss and len(ids):
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self._matrix)

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def query(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids or k <= 0:
            return []
        k = min(k, len(self.ids))
        q = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist(), strict=True))
        else:
            sims = self._matrix @ q[0]
            # stable: highest score first, ties broken by insertion order
            order = np.lexsort((np.arange(len(sims)), -sims))[:k]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [(self.ids[i], s) for i, s in pairs if i >= 0]

    def save(self, out_dir: Path) -> str:
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(out_dir / FAISS_FILE))
            (out_dir / NUMPY_FILE).unlink(missing_ok=True)
            return FAISS_FILE
        np.save(out_dir / NUMPY_FILE, self._matrix)
        (out_dir / FAISS_FILE).unlink(missing_ok=True)
        return NUMPY_FILE

    @classmethod
    def load(cls, index_file: Path, ids: list[str]) -> VectorIndex:
        if index_file.suffix == ".faiss":
            if faiss is None:
                raise IndexNotFoundError("Index was built with FAISS but faiss is not installed")
            fidx = faiss.read_index(str(index_file))
            vectors = (
                fidx.reconstruct_n(0, fidx.ntotal)
                if fidx.ntotal
                else np.zeros((0, fidx.d), np.float32)
            )
            return cls(vectors, ids, use_faiss=True)
        return cls(np.load(index_file), ids, use_faiss=False)


def build_vector_index(artifacts: IndexArtifacts, emb_cfg: dict[str, Any]) -> dict[str, Any]:
    """Embed all passages, write the vector index + meta file, and update ``artifacts``."""
    out = Path(artifacts.output_dir)
    embedder = get_embedder(emb_cfg)
    passages = artifacts.passages
    texts = [p.embedding_text() for p in passages]
    vectors = embedder.encode(texts) if texts else np.zeros((0, embedder.dim), np.float32)
    index = VectorIndex(vectors, [p.passage_id for p in passages])
    index_file = index.save(out)
    meta = {
        "backend": embedder.backend,
        "model": embedder.model_name,
        "device": emb_cfg.get("device", "cpu"),
        "dim": embedder.dim,
        "count": len(passages),
        "vector_store": index.backend,
        "ids": index.ids,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    artifacts.vector_index_path = str(out / index_file)
    artifacts.vector_ids_path = str(out / META_FILE)
    _CACHE.pop(str(out / index_file), None)
    return {
        "backend": embedder.backend,
        "model": embedder.model_name,
        "dim": embedder.dim,
        "vector_store": index.backend,
        "index_file": index_file,
        "meta_file": META_FILE,
        "files": [index_file, META_FILE],
    }


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[VectorIndex, Embedder]] = {}


def _load_vectors(index: IndexArtifacts) -> tuple[VectorIndex, Embedder]:
    if index.vector_index_path and index.vector_ids_path:
        key = index.vector_index_path
        if key not in _CACHE:
            meta = json.loads(Path(index.vector_ids_path).read_text(encoding="utf-8"))
            vi = VectorIndex.load(Path(index.vector_index_path), meta["ids"])
            _CACHE[key] = (vi, embedder_from_meta(meta))
        return _CACHE[key]
    # no persisted vectors: lexical fallback built in memory
    key = f"mem:{index.passages_path}:{len(index.passages)}"
    if key not in _CACHE:
        emb = HashingEmbedder()
        texts = [p.embedding_text() for p in index.passages]
        vecs = emb.encode(texts) if texts else np.zeros((0, emb.dim), np.float32)
        _CACHE[key] = (VectorIndex(vecs, [p.passage_id for p in index.passages]), emb)
    return _CACHE[key]


def search(
    question: str, index: IndexArtifacts, k: int = 5, embedder: Embedder | None = None
) -> list[Evidence]:
    """Return the ``k`` passages most similar to ``question`` as :class:`Evidence`.

    ``similarity_score`` is the cosine similarity clipped to ``[0, 1]``.
    """
    if not question.strip() or not index.passages:
        return []
    vi, stored_embedder = _load_vectors(index)
    emb = embedder or stored_embedder
    if emb.dim != vi.dim:
        raise ValueError(f"Embedder dim {emb.dim} does not match index dim {vi.dim}")
    qvec = emb.encode([question])[0]
    by_id: dict[str, Passage] = {p.passage_id: p for p in index.passages}
    results: list[Evidence] = []
    for rank, (pid, score) in enumerate(vi.query(qvec, k), start=1):
        p = by_id.get(pid)
        if p is None:
            logger.warning("Vector index references unknown passage %s; rebuild the index", pid)
            continue
        results.append(
            Evidence(
                evidence_id=f"ev-{rank}",
                passage_id=p.passage_id,
                file_path=p.file_path,
                start_line=p.start_line,
                end_line=p.end_line,
                snippet=p.content,
                similarity_score=float(min(1.0, max(0.0, score))),
                associated_symbol=p.symbol,
                rank=rank,
            )
        )
    return results
