"""Retriever: the stable search / inspect interface for the rest of RepoGuard.

    from repoguard.retrieval import Retriever, SearchFilters

    r = Retriever.from_dir("data/indexes/schedule", repo_root="data/repos/schedule")
    res = r.search("how is the next run computed", k=5,
                   filters=SearchFilters(path_prefix="schedule/", exclude_tests=True))
    res.evidence          # list[Evidence], stable evidence_id + provenance
    res.diagnostics       # latency, candidates, duplicates, empty, tokens ...
    r.inspect(res.evidence[0].evidence_id)   # full source of that symbol (INSPECT/OPEN)

Search is exact: every passage is scored, filters are applied, identical passages are
removed, then the top-k are returned. Ordering is deterministic (score, then index order).
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import re
import time
from pathlib import Path, PurePosixPath

import numpy as np

from repoguard.exceptions import InvalidQueryError, StaleIndexError
from repoguard.retrieval.embedder import Embedder, HashingEmbedder, embedder_from_meta
from repoguard.retrieval.query import extract_named_symbols
from repoguard.retrieval.search import VectorIndex, load_vector_store
from repoguard.schemas import (
    Evidence,
    IndexArtifacts,
    Passage,
    RetrievalDiagnostics,
    SearchFilters,
    SearchResult,
)

MAX_K = 200
MAX_QUERY_CHARS = 4000
DEFAULT_INSPECT_MAX_LINES = 200

# Hybrid re-rank (tuned on dev repositories only; see reports/retrieval/).
DEFAULT_DEFINITION_BOOST = 0.30
"""Added to the ranking score of a passage that *defines* a symbol named in the query."""
DEFAULT_MENTION_BOOST = 0.10
"""Added when the passage *mentions* a named symbol as an identifier (e.g. callers)."""
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_GITHUB = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)([^/\s]+)/([^/\s]+?)(?:\.git)?/?$"
)


def approx_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (characters / 4, rounded up)."""
    return math.ceil(len(text) / 4)


def make_evidence_id(index_id: str, passage_id: str, content_sha256: str) -> str:
    digest = hashlib.sha256(f"{index_id}|{passage_id}|{content_sha256}".encode()).hexdigest()
    return f"ev_{digest[:16]}"


def github_permalink(
    source_url: str | None, commit: str | None, path: str, start: int, end: int
) -> str | None:
    if not source_url or not commit:
        return None
    m = _GITHUB.match(source_url.strip())
    if not m:
        return None
    owner, repo = m.groups()
    lines = f"#L{start}" if start == end else f"#L{start}-L{end}"
    return f"https://github.com/{owner}/{repo}/blob/{commit}/{path}{lines}"


def _is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    return (
        any(p in ("test", "tests", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _symbol_matches(passage_symbol: str | None, wanted: str) -> bool:
    if not passage_symbol:
        return False
    return passage_symbol == wanted or passage_symbol.endswith("." + wanted)


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Retriever:
    """Search and inspect over one loaded index."""

    def __init__(
        self,
        index: IndexArtifacts,
        embedder: Embedder | None = None,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.index = index
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._passages: list[Passage] = list(index.passages)
        self._pos = {p.passage_id: i for i, p in enumerate(self._passages)}
        stored = load_vector_store(index)
        if stored is not None:
            self._vectors, meta = stored
            self._stored_embedder: Embedder | None = None
            self._meta = meta
            self.backend = meta["backend"]
        else:  # no persisted vectors: lexical hashing index built in memory
            self._stored_embedder = HashingEmbedder()
            texts = [p.embedding_text() for p in self._passages]
            vecs = (
                self._stored_embedder.encode(texts)
                if texts
                else np.zeros((0, self._stored_embedder.dim), np.float32)
            )
            self._vectors = VectorIndex(
                vecs, [p.passage_id for p in self._passages], use_faiss=False
            )
            self._meta = {"backend": "hashing", "model": self._stored_embedder.model_name}
            self.backend = "hashing (in-memory)"
        self._embedder = embedder
        self._ev_to_pid: dict[str, str] | None = None
        self._idents: list[frozenset[str]] | None = None

    def _passage_idents(self) -> list[frozenset[str]]:
        if self._idents is None:
            self._idents = [frozenset(_IDENT_RE.findall(p.content)) for p in self._passages]
        return self._idents

    @classmethod
    def from_dir(
        cls, index_dir: str | os.PathLike[str], embedder: Embedder | None = None,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> Retriever:  # fmt: skip
        from repoguard.repository import load_index

        return cls(load_index(index_dir), embedder=embedder, repo_root=repo_root)

    # ------------------------------------------------------------------ utils
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._stored_embedder or embedder_from_meta(self._meta)
        if self._embedder.dim != self._vectors.dim and len(self._vectors):
            raise InvalidQueryError(
                f"Embedder dim {self._embedder.dim} does not match index dim {self._vectors.dim}"
            )
        return self._embedder

    def evidence_for(
        self, p: Passage, score: float = 0.0, rank: int = 1, rank_score: float | None = None
    ) -> Evidence:
        repo = self.index.repository
        sha = p.content_sha256 or _text_sha(p.content)
        return Evidence(
            evidence_id=make_evidence_id(self.index.index_id, p.passage_id, sha),
            passage_id=p.passage_id,
            file_path=p.file_path,
            start_line=p.start_line,
            end_line=p.end_line,
            snippet=p.content,
            similarity_score=float(min(1.0, max(0.0, score))),
            associated_symbol=p.symbol,
            rank=rank,
            passage_type=p.passage_type,
            part=p.part,
            total_parts=p.total_parts,
            content_sha256=sha,
            approx_tokens=approx_tokens(p.content),
            index_id=self.index.index_id,
            repository_id=repo.repository_id,
            commit=repo.commit,
            source_url=github_permalink(
                repo.source_url, repo.commit, p.file_path, p.start_line, p.end_line
            ),
            rank_score=None if rank_score is None else round(rank_score, 6),
        )

    def _passes(self, p: Passage, f: SearchFilters) -> bool:
        if f.path_prefix is not None and not p.file_path.startswith(f.path_prefix):
            return False
        if f.path_glob is not None and not fnmatch.fnmatch(p.file_path, f.path_glob):
            return False
        if f.passage_types is not None and p.passage_type not in f.passage_types:
            return False
        if f.symbol is not None and not _symbol_matches(p.symbol, f.symbol):
            return False
        return not (f.exclude_tests and _is_test_path(p.file_path))

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
        dedupe: bool = True,
        definition_boost: float = DEFAULT_DEFINITION_BOOST,
        mention_boost: float = DEFAULT_MENTION_BOOST,
    ) -> SearchResult:
        """Top-k evidence for ``query``. Blank queries return an empty result.

        Ranking = cosine similarity + ``definition_boost`` if the passage defines a symbol
        named in the query + ``mention_boost`` if it mentions one. Set both to 0 for pure
        vector search. ``similarity_score`` is always the raw cosine; ``min_score`` applies
        to it.

        Raises :class:`InvalidQueryError` for a non-string query, ``k`` outside
        ``1..MAX_K`` or ``min_score`` outside ``[0, 1]``.
        """
        t0 = time.perf_counter()
        if not isinstance(query, str):
            raise InvalidQueryError("query must be a string")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise InvalidQueryError(f"k must be an integer in 1..{MAX_K}, got {k!r}")
        if not 0.0 <= min_score <= 1.0:
            raise InvalidQueryError("min_score must be in [0, 1]")
        query = query.strip()[:MAX_QUERY_CHARS]
        diag = RetrievalDiagnostics(
            query_chars=len(query), k_requested=k, backend=self.backend,
            vector_store=self._vectors.backend,
        )  # fmt: skip
        result = SearchResult(query=query, filters=filters, diagnostics=diag)
        if not query or not self._passages:
            diag.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return result

        scores = self._vectors.scores(self.embedder.encode([query])[0])
        candidates = [
            i for i, p in enumerate(self._passages) if filters is None or self._passes(p, filters)
        ]
        diag.candidates = len(candidates)
        ranking = {i: float(scores[i]) for i in candidates}
        named = extract_named_symbols(query) if (definition_boost or mention_boost) else []
        if named:
            tails = {n.split(".")[-1] for n in named}
            idents = self._passage_idents()
            for i in candidates:
                p = self._passages[i]
                if definition_boost and any(_symbol_matches(p.symbol, n) for n in named):
                    ranking[i] += definition_boost
                elif mention_boost and tails & idents[i]:
                    ranking[i] += mention_boost
        order = sorted(candidates, key=lambda i: (-ranking[i], i))

        seen: set[str] = set()
        for i in order:
            p = self._passages[i]
            score = float(scores[i])
            if dedupe:
                key = p.content_sha256 or _text_sha(p.content)
                if key in seen:
                    diag.duplicates_removed += 1
                    continue
                seen.add(key)
            if score < min_score:
                diag.below_min_score += 1
                continue
            result.evidence.append(
                self.evidence_for(p, score, rank=len(result.evidence) + 1, rank_score=ranking[i])
            )
            if len(result.evidence) == k:
                break

        diag.returned = len(result.evidence)
        diag.empty = diag.returned == 0
        diag.top1_score = max((e.similarity_score for e in result.evidence), default=0.0)
        diag.retrieved_tokens = sum(e.approx_tokens for e in result.evidence)
        diag.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return result

    # ---------------------------------------------------------------- lookup
    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Resolve a stable evidence id back to its passage (score 0, rank 1)."""
        if self._ev_to_pid is None:
            self._ev_to_pid = {
                make_evidence_id(
                    self.index.index_id, p.passage_id, p.content_sha256 or _text_sha(p.content)
                ): p.passage_id
                for p in self._passages
            }
        pid = self._ev_to_pid.get(evidence_id)
        return self.evidence_for(self._passages[self._pos[pid]]) if pid else None

    def _resolve_passage(self, target: str) -> Passage | None:
        if target in self._pos:
            return self._passages[self._pos[target]]
        if target.startswith("ev_"):
            ev = self.get_evidence(target)
            return self._passages[self._pos[ev.passage_id]] if ev else None
        # symbol id / qualified name / dotted suffix / short name; prefer full passages (part 1)
        matches = [
            p for p in self._passages
            if p.symbol_id == target or _symbol_matches(p.symbol, target)
        ]  # fmt: skip
        if not matches:
            return None
        matches.sort(key=lambda p: (p.passage_type == "class", p.file_path, p.start_line, p.part))
        return matches[0]

    # --------------------------------------------------------------- inspect
    def inspect(self, target: str, max_lines: int = DEFAULT_INSPECT_MAX_LINES) -> Evidence:
        """Expand an evidence id, passage id, symbol or file path to its full source.

        * function / method: the whole definition (all passage parts stitched together)
        * class / module / file path: the full source lines; needs ``repo_root`` and a
          repository whose file hash still matches the index (else :class:`StaleIndexError`)

        Output is capped at ``max_lines``. ``similarity_score`` is 0 (not a search hit).
        """
        if not isinstance(target, str) or not target.strip():
            raise InvalidQueryError("inspect target must be a non-empty string")
        if not 1 <= max_lines <= 5000:
            raise InvalidQueryError("max_lines must be in 1..5000")
        target = target.strip()

        file_meta = {f.path: f for f in self.index.files}
        if target in file_meta:  # whole file
            lines = self._read_lines(target)
            return self._span_evidence(target, 1, len(lines), lines, None, "module", max_lines)

        p = self._resolve_passage(target)
        if p is None:
            raise InvalidQueryError(f"Nothing in index {self.index.index_id!r} matches {target!r}")

        symbol = next((s for s in self.index.symbols if s.symbol_id == p.symbol_id), None)
        if p.passage_type in ("function", "method") and symbol is not None:
            start = symbol.decorator_line or symbol.line
            parts = [q for q in self._passages if q.symbol_id == p.symbol_id]
            by_line: dict[int, str] = {}
            for q in parts:
                for off, text in enumerate(q.content.split("\n")):
                    by_line[q.start_line + off] = text
            lines_span = [by_line.get(n, "") for n in range(start, symbol.end_line + 1)]
            return self._span_evidence(
                p.file_path, start, symbol.end_line, lines_span, p.symbol, p.passage_type,
                max_lines, offset_lines=True, passage_id=p.symbol_id or p.passage_id,
            )  # fmt: skip

        # class or module overview -> real source from the repository
        if self.repo_root is None:
            return self.evidence_for(p)  # best available without the checkout
        lines = self._read_lines(p.file_path)
        if p.passage_type == "class" and symbol is not None:
            start = symbol.decorator_line or symbol.line
            return self._span_evidence(
                p.file_path, start, symbol.end_line, lines[start - 1 : symbol.end_line],
                p.symbol, "class", max_lines, offset_lines=True, passage_id=p.symbol_id or p.passage_id,
            )  # fmt: skip
        return self._span_evidence(p.file_path, 1, len(lines), lines, None, "module", max_lines)

    def _read_lines(self, rel_path: str) -> list[str]:
        from repoguard.repository.parser import read_source, safe_join

        if self.repo_root is None:
            raise InvalidQueryError("inspecting whole files needs Retriever(repo_root=...)")
        meta = next((f for f in self.index.files if f.path == rel_path), None)
        if meta is None:
            raise InvalidQueryError(f"{rel_path!r} is not an indexed file")
        path = safe_join(self.repo_root, rel_path)
        if not path.is_file():
            raise StaleIndexError(f"{rel_path} no longer exists under {self.repo_root}")
        if meta.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != meta.sha256:
            raise StaleIndexError(f"{rel_path} changed since index {self.index.index_id} was built")
        return read_source(path).splitlines()

    def _span_evidence(
        self, path: str, start: int, end: int, lines: list[str], symbol: str | None,
        passage_type: str, max_lines: int, offset_lines: bool = False, passage_id: str | None = None,
    ) -> Evidence:  # fmt: skip
        span = lines if offset_lines else lines[start - 1 : end]
        if len(span) > max_lines:
            span = span[:max_lines]
            end = start + max_lines - 1
        text = "\n".join(span)
        sha = _text_sha(text)
        pid = f"inspect:{passage_id or path}:{start}-{end}"
        repo = self.index.repository
        return Evidence(
            evidence_id=make_evidence_id(self.index.index_id, pid, sha),
            passage_id=pid, file_path=path, start_line=start, end_line=end, snippet=text,
            similarity_score=0.0, associated_symbol=symbol, rank=1, passage_type=passage_type,
            content_sha256=sha, approx_tokens=approx_tokens(text), index_id=self.index.index_id,
            repository_id=repo.repository_id, commit=repo.commit,
            source_url=github_permalink(repo.source_url, repo.commit, path, start, end),
        )  # fmt: skip


# ---------------------------------------------------------------------------
# shared retriever cache for the functional API
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[str, str, int], Retriever] = {}


def get_retriever(index: IndexArtifacts, embedder: Embedder | None = None) -> Retriever:
    """Cached :class:`Retriever` per (index id, passages file, passage count)."""
    if embedder is not None:
        return Retriever(index, embedder=embedder)
    key = (index.index_id, index.passages_path, len(index.passages))
    if key not in _CACHE or _CACHE[key].index is not index:
        _CACHE[key] = Retriever(index)
    return _CACHE[key]
