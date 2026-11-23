"""Phase 2 retrieval contract: evidence provenance, filters, diagnostics, inspect, robustness."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from repoguard.exceptions import InvalidQueryError, StaleIndexError
from repoguard.repository import build_index, load_index
from repoguard.retrieval import Retriever, SearchFilters, make_evidence_id, search
from repoguard.retrieval.retriever import MAX_K, approx_tokens, github_permalink
from repoguard.retrieval.search import faiss_available

HASHING = {"embeddings": {"enabled": True, "backend": "hashing", "vector_store": "numpy"}}


@pytest.fixture
def idx(fixture_repo, tmp_path):
    return build_index(fixture_repo, tmp_path / "idx", HASHING)


@pytest.fixture
def retriever(idx, fixture_repo):
    return Retriever(idx, repo_root=fixture_repo)


# ---- evidence contract ------------------------------------------------------


def test_evidence_carries_full_provenance(retriever, fixture_repo):
    (ev,) = retriever.search("load a JSON config file", k=1).evidence
    assert (ev.file_path, ev.start_line, ev.end_line) == ("pkg/config.py", 9, 14)
    assert ev.associated_symbol == "pkg.config.load_config" and ev.passage_type == "function"
    lines = (fixture_repo / ev.file_path).read_text().splitlines()
    assert ev.snippet == "\n".join(lines[ev.start_line - 1 : ev.end_line])
    assert ev.content_sha256 == hashlib.sha256(ev.snippet.encode()).hexdigest()
    assert ev.approx_tokens == approx_tokens(ev.snippet) > 0
    assert ev.index_id == retriever.index.index_id and ev.repository_id == "indexer-repo"
    assert ev.evidence_id == make_evidence_id(ev.index_id, ev.passage_id, ev.content_sha256)
    assert ev.evidence_id.startswith("ev_") and len(ev.evidence_id) == 19
    assert 0.0 <= ev.similarity_score <= 1.0 and ev.rank == 1


def test_evidence_ids_stable_across_reload_and_rebuild(idx, fixture_repo, tmp_path):
    first = [e.evidence_id for e in Retriever(idx).search("helper value", k=5).evidence]
    reloaded = [
        e.evidence_id
        for e in Retriever(load_index(idx.output_dir)).search("helper value", k=5).evidence
    ]
    rebuilt = build_index(fixture_repo, tmp_path / "again", HASHING)
    again = [e.evidence_id for e in Retriever(rebuilt).search("helper value", k=5).evidence]
    assert first == reloaded == again
    other = build_index(
        fixture_repo, tmp_path / "other", {**HASHING, "passages": {"max_lines": 50}}
    )
    assert other.index_id != idx.index_id
    assert Retriever(other).search("helper value", k=1).evidence[0].evidence_id not in first


def test_get_evidence_round_trip(retriever):
    ev = retriever.search("initialize app", k=3).evidence[2]
    back = retriever.get_evidence(ev.evidence_id)
    assert back is not None and (back.passage_id, back.content_sha256) == (
        ev.passage_id,
        ev.content_sha256,
    )
    assert retriever.get_evidence("ev_0000000000000000") is None


def test_github_permalinks():
    url = github_permalink(
        "https://github.com/dbader/schedule.git", "abc123", "schedule/__init__.py", 10, 20
    )
    assert url == "https://github.com/dbader/schedule/blob/abc123/schedule/__init__.py#L10-L20"
    assert github_permalink("git@github.com:psf/requests.git", "c0", "a.py", 3, 3) == (
        "https://github.com/psf/requests/blob/c0/a.py#L3"
    )
    assert github_permalink("https://gitlab.com/x/y", "c", "a.py", 1, 2) is None
    assert github_permalink("https://github.com/x/y", None, "a.py", 1, 2) is None


def test_permalink_on_evidence(fixture_repo, tmp_path):
    cfg = {**HASHING, "repository": {"source_url": "https://github.com/o/r", "commit": "deadbeef"}}
    ev = Retriever(build_index(fixture_repo, tmp_path / "i", cfg)).search("helper", k=1).evidence[0]
    assert (
        ev.source_url
        == f"https://github.com/o/r/blob/deadbeef/{ev.file_path}#L{ev.start_line}-L{ev.end_line}"
    )
    assert ev.commit == "deadbeef"


# ---- search behaviour -------------------------------------------------------------


def test_definition_boost_ranks_named_symbol_first_without_changing_similarity(retriever):
    q = "Where is `validate` defined?"
    boosted = retriever.search(q, k=5)
    dense = retriever.search(q, k=5, definition_boost=0.0, mention_boost=0.0)
    assert boosted.evidence[0].associated_symbol == "pkg.config.validate"
    raw = {e.passage_id: e.similarity_score for e in dense.evidence}
    for e in boosted.evidence:
        if e.passage_id in raw:
            assert e.similarity_score == raw[e.passage_id]
        assert e.rank_score is not None and e.rank_score >= e.similarity_score - 1e-6
    scores = [e.similarity_score for e in dense.evidence]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(retriever):
    a = retriever.search("config strict validation", k=10)
    b = retriever.search("config strict validation", k=10)
    assert [e.model_dump() for e in a.evidence] == [e.model_dump() for e in b.evidence]


def test_filters(retriever):
    def files(**kw):
        return {
            e.file_path
            for e in retriever.search(
                "app config helper", k=50, filters=SearchFilters(**kw)
            ).evidence
        }

    assert files(path_prefix="pkg/config") == {"pkg/config.py"}
    assert files(path_glob="*/u*.py") == {"pkg/utils.py"}
    types = {
        e.passage_type
        for e in retriever.search(
            "app", k=50, filters=SearchFilters(passage_types=["class"])
        ).evidence
    }
    assert types == {"class"}
    syms = {
        e.associated_symbol
        for e in retriever.search("x", k=50, filters=SearchFilters(symbol="App.fetch")).evidence
    }
    assert syms == {"pkg.app.App.fetch"}
    assert retriever.search(
        "app", k=5, filters=SearchFilters(path_prefix="nope/")
    ).diagnostics.empty


def test_exclude_tests_filter(tmp_repo, tmp_path):
    (tmp_repo / "tests").mkdir()
    (tmp_repo / "tests" / "test_app.py").write_text(
        "def test_helper():\n    assert helper(1) == 1\n"
    )
    r = Retriever(build_index(tmp_repo, tmp_path / "i", HASHING))
    all_files = {e.file_path for e in r.search("helper", k=50).evidence}
    no_tests = {
        e.file_path
        for e in r.search("helper", k=50, filters=SearchFilters(exclude_tests=True)).evidence
    }
    assert "tests/test_app.py" in all_files and "tests/test_app.py" not in no_tests


def test_min_score_and_diagnostics(retriever):
    res = retriever.search("load config", k=5)
    d = res.diagnostics
    assert d.returned == len(res.evidence) == 5 and not d.empty
    assert d.candidates == len(retriever.index.passages) and d.k_requested == 5
    assert d.retrieved_tokens == sum(e.approx_tokens for e in res.evidence)
    assert d.top1_score == max(e.similarity_score for e in res.evidence)
    assert d.latency_ms >= 0 and d.backend == "hashing" and d.vector_store == "numpy"
    strict = retriever.search("load config", k=5, min_score=0.99)
    assert (
        strict.evidence == []
        and strict.diagnostics.below_min_score > 0
        and strict.diagnostics.empty
    )


def test_duplicate_passages_are_removed(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    body = "def same(x):\n    return x * 2\n"
    (repo / "a.py").write_text(body)
    (repo / "b.py").write_text("# copy\n" + body)  # same function text, different module
    r = Retriever(build_index(repo, tmp_path / "i", HASHING))
    # identical *content* of the function passages -> one kept
    res = r.search("same x times two", k=10, filters=SearchFilters(passage_types=["function"]))
    assert len(res.evidence) == 1 and res.diagnostics.duplicates_removed == 1
    raw = r.search(
        "same x times two", k=10, filters=SearchFilters(passage_types=["function"]), dedupe=False
    )
    assert len(raw.evidence) == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"query": None}, {"query": "x", "k": 0}, {"query": "x", "k": MAX_K + 1}, {"query": "x", "k": True},
     {"query": "x", "k": 2.5}, {"query": "x", "min_score": 1.5}],
)  # fmt: skip
def test_invalid_search_inputs_raise(retriever, kwargs):
    with pytest.raises(InvalidQueryError):
        retriever.search(**kwargs)


def test_blank_and_huge_queries(retriever):
    res = retriever.search("   \n ", k=3)
    assert res.evidence == [] and res.diagnostics.empty
    huge = retriever.search("config " * 5000, k=1)
    assert huge.diagnostics.query_chars == 4000 and len(huge.evidence) == 1


def test_empty_index_search_and_inspect(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    (repo / "README.md").write_text("# nothing here\n")
    art = build_index(repo, tmp_path / "i", HASHING)
    r = Retriever(art)
    res = r.search("anything", k=5)
    assert res.evidence == [] and res.diagnostics.empty and res.diagnostics.candidates == 0
    with pytest.raises(InvalidQueryError):
        r.inspect("anything")
    manifest = json.loads(Path(art.manifest_path).read_text())
    assert manifest["counts"]["unsupported_files_skipped"] == 1


def test_parser_failures_do_not_break_retrieval(tmp_repo, tmp_path):
    (tmp_repo / "pkg" / "broken.py").write_text("def broken(:\n")
    r = Retriever(build_index(tmp_repo, tmp_path / "i", HASHING))
    assert all(e.file_path != "pkg/broken.py" for e in r.search("broken", k=50).evidence)


# ---- vector store integrity ------------------------------------------------------------


def test_stale_vectors_are_rejected(idx):
    meta_path = Path(idx.vector_ids_path)
    meta = json.loads(meta_path.read_text())
    meta["ids"] = list(reversed(meta["ids"]))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(StaleIndexError):
        Retriever(load_index(idx.output_dir))


def test_vectors_from_another_build_are_rejected(idx, fixture_repo, tmp_path):
    other = build_index(
        fixture_repo,
        tmp_path / "o",
        {**HASHING, "embeddings": {**HASHING["embeddings"], "hashing_dim": 64}},
    )
    shutil.copy(other.vector_ids_path, idx.vector_ids_path)
    with pytest.raises(StaleIndexError):
        Retriever(load_index(idx.output_dir))


@pytest.mark.skipif(not faiss_available(), reason="faiss not importable")
def test_faiss_and_numpy_stores_rank_identically(fixture_repo, tmp_path):
    f = build_index(
        fixture_repo,
        tmp_path / "f",
        {"embeddings": {"enabled": True, "backend": "hashing", "vector_store": "faiss"}},
    )
    n = build_index(fixture_repo, tmp_path / "n", HASHING)
    assert (
        Path(f.vector_index_path).suffix == ".faiss" and Path(n.vector_index_path).suffix == ".npy"
    )
    for q in ("load config", "app initialize", "helper value"):
        a = [
            (e.passage_id, round(e.similarity_score, 5))
            for e in Retriever(load_index(f.output_dir)).search(q, k=8).evidence
        ]
        b = [
            (e.passage_id, round(e.similarity_score, 5))
            for e in Retriever(load_index(n.output_dir)).search(q, k=8).evidence
        ]
        assert a == b


def test_index_without_vectors_still_searchable(fixture_repo, tmp_path):
    art = build_index(fixture_repo, tmp_path / "plain")
    r = Retriever(art)
    assert r.backend.startswith("hashing") and r.search("initialize app", k=1).evidence


# ---- inspect -------------------------------------------------------------------------


def test_inspect_function_stitches_all_parts(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    body = "\n".join(f"    x{i} = {i}" for i in range(30))
    (repo / "big.py").write_text(f"def big():\n{body}\n    return x0\n")
    r = Retriever(
        build_index(
            repo, tmp_path / "i", {**HASHING, "passages": {"max_lines": 10, "overlap_lines": 2}}
        )
    )
    ev = r.inspect("big")
    source = (repo / "big.py").read_text().splitlines()
    assert ev.snippet == "\n".join(source) and (ev.start_line, ev.end_line) == (1, 32)
    assert ev.similarity_score == 0.0 and ev.passage_id.startswith("inspect:")
    capped = r.inspect("big", max_lines=5)
    assert capped.end_line == 5 and len(capped.snippet.split("\n")) == 5


def test_inspect_class_and_file_need_repo_root(retriever, idx, fixture_repo):
    ev = retriever.inspect("pkg.app.App")
    lines = (fixture_repo / "pkg/app.py").read_text().splitlines()
    assert ev.snippet == "\n".join(lines[16:44]) and "load_config(self.path)" in ev.snippet
    whole = retriever.inspect("pkg/utils.py")
    assert whole.snippet == (fixture_repo / "pkg/utils.py").read_text().rstrip("\n")
    overview = Retriever(idx).inspect("pkg.app.App")  # no repo_root -> class overview passage
    assert overview.passage_type == "class" and "load_config(self.path)" not in overview.snippet
    with pytest.raises(InvalidQueryError):
        Retriever(idx).inspect("pkg/utils.py")


def test_inspect_by_evidence_id(retriever):
    ev = retriever.search("Where is `helper` defined?", k=1).evidence[0]
    assert retriever.inspect(ev.evidence_id).associated_symbol == "pkg.utils.helper"


def test_inspect_detects_stale_files_and_bad_targets(tmp_repo, tmp_path):
    art = build_index(tmp_repo, tmp_path / "i", HASHING)
    r = Retriever(art, repo_root=tmp_repo)
    (tmp_repo / "pkg" / "utils.py").write_text("def helper(value):\n    return value + 1\n")
    with pytest.raises(StaleIndexError):
        r.inspect("pkg/utils.py")
    for bad in ("", "   ", "no_such_symbol", "../../etc/passwd"):
        with pytest.raises(InvalidQueryError):
            r.inspect(bad)


# ---- backward compatibility ---------------------------------------------------------


def test_functional_search_api_is_backward_compatible(idx):
    ev = search("load a JSON config file", idx, k=3)
    assert len(ev) == 3 and ev[0].associated_symbol == "pkg.config.load_config"
    assert search("", idx) == [] and search("x", idx, k=0) == []
    assert len(search("config", idx, k=10_000)) == len(idx.passages)
