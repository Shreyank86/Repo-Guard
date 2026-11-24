"""Retrieval-derived RL state signals ("free signals": computed with zero LLM tokens).

    from repoguard.retrieval import Retriever, SignalExtractor
    sig = SignalExtractor(Retriever.from_dir("data/indexes/schedule")).compute(question)
    state = StateVector(**sig.state_fields(), risk_lambda=lam)   # Member 3's 18-dim state

Every numeric signal is in [0, 1]. Definitions, normalisation and missing-value rules are
documented in docs/state_signals.md; keep that file in sync with this module.
No ground-truth labels are used: only the question text, the index and retrieval results.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import cached_property

from repoguard.retrieval.embedder import code_tokens
from repoguard.retrieval.query import (
    _DOTTED,
    _ENGLISH,
    extract_named_files,
    extract_named_symbols,
)
from repoguard.retrieval.retriever import Retriever
from repoguard.schemas import FreeSignals

SIGNALS_VERSION = "1.0"

DEFAULT_K = 5
DEFAULT_MIN_EVIDENCE_SCORE = 0.30
DEFAULT_TOKEN_BUDGET_REF = 2000  # matches configs/rl_config.yaml rewards.token_budget_ref
DEFAULT_STACK_OVERLAP = 0.5

# Content words that say nothing about the repository's topic.
_QUESTION_WORDS = _ENGLISH | frozenset(
    ["where", "what", "which", "does", "how", "why", "when", "who", "whom", "whose", "is", "are", "do", "did", "defined", "define", "definition", "located", "location", "implement", "implemented", "implementation", "exist", "exists", "existing", "named", "name", "names", "type", "types", "default", "defaults", "value", "values", "kind", "role", "purpose", "responsible", "inherit", "inherits", "inheritance", "base", "subclass", "param", "params", "arg", "args", "kwarg", "kwargs", "signature", "line", "lines", "codebase", "source", "python", "py", "please", "explain", "show", "tell", "describe", "find", "list", "give", "me", "my", "our", "we", "you", "your", "i"]
)

# Technologies whose mention (when absent from the repo) signals a different stack.
_FOREIGN_STACK = frozenset(
    ["javascript", "typescript", "react", "angular", "vue", "svelte", "nodejs", "npm", "yarn", "java", "kotlin", "scala", "spring", "maven", "gradle", "rust", "cargo", "golang", "csharp", "dotnet", "php", "laravel", "ruby", "rails", "swift", "objective-c", "flutter", "dart", "kubernetes", "terraform", "html", "css", "jquery", "django", "flask", "fastapi", "celery", "redis", "sqlalchemy", "pandas", "dataframe", "tensorflow", "pytorch", "torch", "keras", "kafka", "rabbitmq", "postgres", "postgresql", "mysql", "mongodb", "boto3", "aws", "gcp", "azure", "graphql", "oauth", "jwt", "blueprint", "orm"]
)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class _RepoNames:
    exact: set[str] = field(default_factory=set)
    """Every name that exists in the repository (short, qualified, dotted suffixes,
    parameters, imports, call targets, module names)."""
    short: list[str] = field(default_factory=list)
    dotted: list[str] = field(default_factory=list)
    definitions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    """name or dotted suffix -> [(qualified_name, file)] for functions/methods/classes."""
    vocabulary: set[str] = field(default_factory=set)


class SignalExtractor:
    """Computes :class:`FreeSignals` for questions against one index."""

    def __init__(
        self,
        retriever: Retriever,
        k: int = DEFAULT_K,
        min_evidence_score: float = DEFAULT_MIN_EVIDENCE_SCORE,
        token_budget_ref: int = DEFAULT_TOKEN_BUDGET_REF,
        stack_overlap_threshold: float = DEFAULT_STACK_OVERLAP,
    ) -> None:
        self.retriever = retriever
        self.index = retriever.index
        self.k = k
        self.min_evidence_score = min_evidence_score
        self.token_budget_ref = token_budget_ref
        self.stack_overlap_threshold = stack_overlap_threshold

    @cached_property
    def names(self) -> _RepoNames:
        n = _RepoNames()
        for s in self.index.symbols:
            parts = s.qualified_name.split(".")
            n.exact.add(s.name)
            # every dotted suffix: Job.do, schedule.Job.do ...
            for i in range(len(parts) - 1):
                suffix = ".".join(parts[i:])
                n.exact.add(suffix)
                n.definitions.setdefault(suffix, []).append((s.qualified_name, s.file))
            n.definitions.setdefault(s.name, []).append((s.qualified_name, s.file))
            for p in getattr(s, "parameters", []) or []:
                n.exact.add(p)
            for cv in getattr(s, "class_variables", []) or []:
                n.exact.add(cv)
        for f in self.index.files:
            n.exact.update({f.module, f.module.split(".")[-1]})
        for imp in self.index.imports:
            for v in (imp.module, imp.name, imp.alias, imp.resolved_module):
                if v and v != "*":
                    n.exact.add(v)
                    n.exact.add(v.split(".")[0])
        for e in self.index.call_edges:
            if e.callee_expr and re.fullmatch(_DOTTED, e.callee_expr):
                n.exact.add(e.callee_expr)
                n.exact.add(e.callee_expr.split(".")[0])
        n.short = sorted({x for x in n.exact if "." not in x})
        n.dotted = sorted({x for x in n.exact if "." in x})

        vocab: set[str] = set()
        for name in n.exact:
            vocab.update(code_tokens(name))
        for f in self.index.files:
            vocab.update(code_tokens(f.path))
        for s in self.index.symbols:
            if s.docstring:
                vocab.update(code_tokens(s.docstring))
        n.vocabulary = vocab
        return n

    # ------------------------------------------------------------------
    def _best_match(self, name: str) -> tuple[float, str | None]:
        if name in self.names.exact:
            return 1.0, name
        pool = self.names.dotted if "." in name else self.names.short
        best, best_name = 0.0, None
        for cand in difflib.get_close_matches(name, pool, n=5, cutoff=0.0) if pool else []:
            sim = _similarity(name, cand)
            if sim > best:
                best, best_name = sim, cand
        if "." in name:  # also compare the last component against short names
            tail = name.split(".")[-1]
            sim, cand = self._best_match(tail) if tail != name else (0.0, None)
            sim *= 0.95  # a matching tail alone is slightly weaker evidence
            if sim > best:
                best, best_name = sim, cand
        return round(best, 4), best_name

    def _file_exists(self, named: str) -> list[str]:
        paths = [f.path for f in self.index.files]
        return [p for p in paths if p == named or p.endswith("/" + named)]

    def compute(self, question: str) -> FreeSignals:
        sig = FreeSignals()
        if not isinstance(question, str) or not question.strip():
            return sig  # all zeros, missing_evidence = 1

        named = extract_named_symbols(question)
        files = extract_named_files(question)
        sig.named_symbols, sig.named_files = named, files
        sig.has_named_symbol = 1.0 if named else 0.0
        sig.has_named_file = 1.0 if files else 0.0

        # --- symbol existence and similarity
        if named:
            matches = {n: self._best_match(n) for n in named}
            sig.symbol_found = 1.0 if all(m[0] == 1.0 for m in matches.values()) else 0.0
            worst = min(named, key=lambda n: (matches[n][0], named.index(n)))
            sig.name_similarity = matches[worst][0]
            nearest = matches[worst][1]
            if nearest is not None:
                sig.nearest_symbol = nearest
                defs = self.names.definitions.get(nearest)
                sig.nearest_symbol_file = defs[0][1] if defs else None

        # --- files
        matched_files: list[str] = []
        if files:
            hits = [self._file_exists(f) for f in files]
            sig.file_found = 1.0 if all(hits) else 0.0
            matched_files = [p for h in hits for p in h]
            def_files = {file for n in named for _, file in self.names.definitions.get(n, [])}
            sig.symbol_in_named_file = 1.0 if def_files & set(matched_files) else 0.0

        # --- retrieval probe
        res = self.retriever.search(question, k=self.k)
        ev = res.evidence
        sig.evidence_ids = [e.evidence_id for e in ev]
        if ev:
            sims = sorted((e.similarity_score for e in ev), reverse=True)
            sig.retr_top1 = round(sims[0], 4)
            sig.retr_margin = round(max(0.0, sims[0] - (sims[1] if len(sims) > 1 else 0.0)), 4)
            strong = [e for e in ev if e.similarity_score >= self.min_evidence_score]
            sig.evidence_count = round(len(strong) / self.k, 4)
            sig.source_diversity = round(len({e.file_path for e in ev}) / len(ev), 4)
            sig.retrieved_tokens = round(
                min(1.0, sum(e.approx_tokens for e in ev) / self.token_budget_ref), 4
            )
            if named:
                top_text = f"{ev[0].associated_symbol or ''}\n{ev[0].snippet}"
                sig.passage_mentions_symbol = (
                    1.0 if any(_mentions(top_text, n) for n in named) else 0.0
                )

        # --- coverage of found definitions by the evidence
        found_defs = [n for n in named if n in self.names.definitions]
        if found_defs and ev:
            covered = sum(
                1 for n in found_defs
                if any(_symbol_hit(e.associated_symbol, n) or _mentions(e.snippet, n) for e in ev)
            )  # fmt: skip
            sig.coverage = round(covered / len(found_defs), 4)
        sig.missing_evidence = (
            1.0 if sig.evidence_count == 0.0 or (found_defs and sig.coverage < 1.0) else 0.0
        )

        # --- topic / stack match
        repo_words = set(code_tokens(self.index.repository.name)) | set(
            code_tokens(self.index.repository.repository_id)
        )
        content = [
            t
            for t in code_tokens(question)
            if t not in _QUESTION_WORDS and t not in repo_words and len(t) > 1
        ]
        if content:
            sig.vocab_overlap = round(
                sum(1 for t in content if t in self.names.vocabulary) / len(content), 4
            )
        q_words = set(re.findall(r"[a-z][a-z0-9#+.-]*", question.lower()))
        foreign = any(t in q_words and t not in self.names.vocabulary for t in _FOREIGN_STACK)
        code_ref = bool(named or files)
        sig.stack_match = (
            1.0
            if not foreign and (code_ref or sig.vocab_overlap >= self.stack_overlap_threshold)
            else 0.0
        )
        return sig


def _mentions(text: str, name: str) -> bool:
    tail = name.split(".")[-1]
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(tail)}(?![A-Za-z0-9_])", text) is not None


def _symbol_hit(symbol: str | None, name: str) -> bool:
    return bool(symbol) and (symbol == name or symbol.endswith("." + name))


def compute_free_signals(question: str, retriever: Retriever, **kwargs) -> FreeSignals:
    """Convenience wrapper: ``SignalExtractor(retriever, **kwargs).compute(question)``."""
    return SignalExtractor(retriever, **kwargs).compute(question)
