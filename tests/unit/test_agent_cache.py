"""Unit tests for OperationCache in src/repoguard/agent/cache.py."""

import tempfile
from pathlib import Path
import pytest

from repoguard.agent.cache import OperationCache


@pytest.fixture
def temp_cache():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_cache.db"
        yield OperationCache(db_path=db_path)


def test_cache_miss_returns_none(temp_cache: OperationCache):
    res = temp_cache.get("search", "schedule", "commit123", {"query": "where is Job?"})
    assert res is None
    stats = temp_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["total_entries"] == 0


def test_cache_set_and_hit(temp_cache: OperationCache):
    inputs = {"query": "where is Job?", "k": 5}
    expected_output = [{"file": "schedule/__init__.py", "line": 209}]

    temp_cache.set("search", "schedule", "commit123", inputs, expected_output)

    assert temp_cache.has("search", "schedule", "commit123", inputs) is True
    cached = temp_cache.get("search", "schedule", "commit123", inputs)
    assert cached == expected_output

    stats = temp_cache.stats()
    assert stats["hits"] == 1
    assert stats["total_entries"] == 1
    assert stats["by_type"] == {"search": 1}
    assert stats["hit_ratio"] == 1.0


def test_cache_keys_are_canonical_and_order_independent(temp_cache: OperationCache):
    inputs_a = {"alpha": 1, "beta": 2, "gamma": [3, 4]}
    inputs_b = {"gamma": [3, 4], "beta": 2, "alpha": 1}

    key_a = temp_cache.compute_key("verify", "dotenv", "c1", inputs_a)
    key_b = temp_cache.compute_key("verify", "dotenv", "c1", inputs_b)

    assert key_a == key_b

    temp_cache.set("verify", "dotenv", "c1", inputs_a, {"verified": True})
    assert temp_cache.get("verify", "dotenv", "c1", inputs_b) == {"verified": True}


def test_cache_clear(temp_cache: OperationCache):
    temp_cache.set("inspect", "repo1", "c1", {"line": 10}, "def test(): pass")
    assert temp_cache.stats()["total_entries"] == 1

    temp_cache.clear()
    assert temp_cache.stats()["total_entries"] == 0
    assert temp_cache.get("inspect", "repo1", "c1", {"line": 10}) is None
