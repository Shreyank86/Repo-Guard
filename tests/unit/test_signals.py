"""RL free signals (Member 1 -> Member 3 contract)."""

from __future__ import annotations

from pathlib import Path

import pytest

from repoguard.repository import build_index
from repoguard.retrieval import (
    FreeSignals,
    Retriever,
    SignalExtractor,
    extract_named_files,
    extract_named_symbols,
)
from repoguard.schemas import StateVector

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "indexer_repo"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where is `load_config` defined?", ["load_config"]),
        ("Does `App.fetch` accept `timeout`?", ["App.fetch", "timeout"]),
        ("Does `Job.tag` accept a `unique=True` argument?", ["Job.tag", "unique"]),
        ("How does sync_all_users() work?", ["sync_all_users"]),
        ("what does initializeApp return", ["initializeApp"]),
        ("Where is the function compute defined?", ["compute"]),
        ("Where is the App class?", ["App"]),
        ("What is the base class of `CancelJob`?", ["CancelJob"]),
        ("Where is the module-level function `every` defined?", ["every"]),
        ("What is a good pasta recipe?", []),
        ("Is load_config in `pkg/missing.py`?", ["load_config"]),
    ],
)
def test_extract_named_symbols(question, expected):
    assert extract_named_symbols(question) == expected


def test_extract_named_files():
    assert extract_named_files("Is it in missing.py or `pkg/config.py`?") == [
        "missing.py",
        "pkg/config.py",
    ]
    assert extract_named_files("no files here") == []


@pytest.fixture(scope="module")
def extractor(tmp_path_factory):
    out = tmp_path_factory.mktemp("sig")
    art = build_index(
        FIXTURE_REPO,
        out / "i",
        {"embeddings": {"enabled": True, "backend": "hashing", "vector_store": "numpy"}},
    )
    return SignalExtractor(Retriever(art))


def test_existing_symbol(extractor):
    s = extractor.compute("Where is `load_config` defined and what does it return?")
    assert (s.symbol_found, s.name_similarity, s.passage_mentions_symbol, s.stack_match) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert s.has_named_symbol == 1.0 and s.coverage == 1.0 and s.missing_evidence == 0.0
    assert s.retr_top1 > 0 and 0 <= s.retr_margin <= s.retr_top1
    assert s.nearest_symbol == "load_config" and s.nearest_symbol_file == "pkg/config.py"


def test_near_miss_symbol(extractor):
    s = extractor.compute("Does `load_confg` exist?")
    assert s.symbol_found == 0.0 and 0.8 < s.name_similarity < 1.0
    assert (s.nearest_symbol, s.nearest_symbol_file) == ("load_config", "pkg/config.py")
    assert s.stack_match == 1.0  # still a question about this code


def test_nonexistent_symbol_is_dissimilar(extractor):
    s = extractor.compute("How does `cancel_all_async_jobs()` work?")
    assert s.symbol_found == 0.0 and s.name_similarity < 0.7


def test_wrong_parameter(extractor):
    ok = extractor.compute("Does `load_config` accept `strict`?")
    bad = extractor.compute("Does `load_config` accept `max_attempts`?")
    assert ok.symbol_found == 1.0 and bad.symbol_found == 0.0
    assert bad.name_similarity < ok.name_similarity


def test_wrong_and_right_path(extractor):
    wrong = extractor.compute("Is `load_config` defined in missing.py?")
    right = extractor.compute("Is `load_config` defined in config.py?")
    assert (wrong.has_named_file, wrong.file_found, wrong.symbol_in_named_file) == (1.0, 0.0, 0.0)
    assert (right.file_found, right.symbol_in_named_file) == (1.0, 1.0)
    other = extractor.compute("Is `load_config` defined in `pkg/app.py`?")
    assert other.file_found == 1.0 and other.symbol_in_named_file == 0.0


def test_off_topic_and_foreign_stack(extractor):
    pasta = extractor.compute("What is a good pasta recipe?")
    assert pasta.stack_match == 0.0 and pasta.has_named_symbol == 0.0 and pasta.symbol_found == 0.0
    django = extractor.compute("How does the Django ORM save models here?")
    assert django.stack_match == 0.0


def test_empty_question_gives_zeros(extractor):
    s = extractor.compute("   ")
    assert s.state_fields() == dict.fromkeys(s.state_fields(), 0.0) and s.missing_evidence == 1.0


@pytest.mark.parametrize(
    "q",
    ["Where is `load_config` defined?", "Does `load_confg` exist?", "pasta?", "Which functions call `helper`?",
     "Is `App` in `pkg/app.py`?", "`a.b.c.d` `x_y` `z`"],
)  # fmt: skip
def test_all_numeric_signals_in_unit_range(extractor, q):
    for name, v in extractor.compute(q).feature_vector().items():
        assert 0.0 <= v <= 1.0, name


def test_state_fields_plug_into_state_vector(extractor):
    s = extractor.compute("Where is `helper` defined?")
    fields = s.state_fields()
    assert list(fields) == [
        "symbol_found",
        "name_similarity",
        "retr_top1",
        "retr_margin",
        "passage_mentions_symbol",
        "stack_match",
    ]
    vec = StateVector(**fields, risk_lambda=1.0)
    assert vec.to_tensor()[:6] == list(fields.values()) and len(vec.to_tensor()) == 18


def test_signals_are_deterministic(extractor):
    q = "Which functions call `validate`?"
    assert extractor.compute(q) == extractor.compute(q)


def test_signals_on_empty_index(tmp_path):
    repo = tmp_path / "e"
    repo.mkdir()
    s = SignalExtractor(Retriever(build_index(repo, tmp_path / "i"))).compute(
        "Where is `x_y` defined?"
    )
    assert (
        isinstance(s, FreeSignals)
        and s.symbol_found == 0.0
        and s.retr_top1 == 0.0
        and s.missing_evidence == 1.0
    )
