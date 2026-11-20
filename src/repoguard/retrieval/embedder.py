"""Text embedders for passage retrieval.

* :class:`SentenceTransformerEmbedder` — the agreed model
  (``sentence-transformers/all-MiniLM-L6-v2``). Requires the ``ml`` extra and the
  model weights (downloaded once by sentence-transformers / Hugging Face).
* :class:`HashingEmbedder` — deterministic, dependency-free lexical fallback
  (code-aware tokens + bigrams, signed feature hashing). Used for tests, CI and
  machines without the ML stack. It is *not* semantic; report which backend produced
  any retrieval numbers (the manifest records it).

All embedders return L2-normalised ``float32`` arrays so inner product == cosine.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Protocol

import numpy as np

from repoguard.exceptions import EmbeddingBackendError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    backend: str
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Hashing fallback
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "and",
        "or",
        "is",
        "are",
        "be",
        "for",
        "on",
        "with",
        "by",
        "as",
        "it",
        "this",
        "that",
        "self",
        "cls",
        "def",
        "return",
        "none",
        "true",
        "false",
        "if",
        "else",
        "elif",
        "not",
        "from",
        "import",
        "class",
    ]
)


def code_tokens(text: str) -> list[str]:
    """Lower-cased tokens; identifiers are kept whole *and* split on snake/camel case."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        pieces = [p.lower() for part in word.split("_") if part for p in _CAMEL.findall(part)]
        if lower not in _STOP:
            tokens.append(lower)
        if len(pieces) > 1:
            tokens.extend(p for p in pieces if p not in _STOP and len(p) > 1)
    return tokens


class HashingEmbedder:
    backend = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim < 16:
            raise ValueError("hashing dim must be >= 16")
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(h, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = code_tokens(text)
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for a, b in zip(toks, toks[1:], strict=False):
                key = f"{a} {b}"
                counts[key] = counts.get(key, 0) + 1
            for feat, c in counts.items():
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * (1.0 + math.log(c))
        return _normalise(out)


# ---------------------------------------------------------------------------
# sentence-transformers
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 32
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbeddingBackendError(
                "sentence-transformers is not installed; run `pip install -e .[ml]`"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # pragma: no cover - network / weights issues
            raise EmbeddingBackendError(
                f"Could not load embedding model {model_name!r}: {exc}"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        self.dim = int(dim) if dim else int(self._model.encode(["x"]).shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )  # fmt: skip
        return _normalise(vecs)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def get_embedder(cfg: dict[str, Any] | None = None) -> Embedder:
    """Create an embedder from an ``embeddings`` config block.

    ``backend``: ``"sentence-transformers"`` (fail if unavailable), ``"hashing"``, or
    ``"auto"`` (try sentence-transformers, fall back to hashing with a warning).
    """
    cfg = cfg or {}
    backend = cfg.get("backend", "auto")
    if backend == "hashing":
        return HashingEmbedder(dim=int(cfg.get("hashing_dim", 512)))
    if backend not in ("auto", "sentence-transformers"):
        raise EmbeddingBackendError(f"Unknown embedding backend: {backend!r}")
    try:
        return SentenceTransformerEmbedder(
            model_name=cfg.get("model", DEFAULT_MODEL),
            device=cfg.get("device", "cpu"),
            batch_size=int(cfg.get("batch_size", 32)),
        )
    except EmbeddingBackendError:
        if backend == "sentence-transformers":
            raise
        logger.warning("sentence-transformers unavailable; falling back to HashingEmbedder")
        return HashingEmbedder(dim=int(cfg.get("hashing_dim", 512)))


def embedder_from_meta(meta: dict[str, Any]) -> Embedder:
    """Recreate exactly the embedder that built a vector index (from vector_meta.json)."""
    if meta["backend"] == "hashing":
        return HashingEmbedder(dim=int(meta["dim"]))
    return SentenceTransformerEmbedder(model_name=meta["model"], device=meta.get("device", "cpu"))
