from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
INDEXER_REPO = FIXTURES / "indexer_repo"


@pytest.fixture
def fixture_repo() -> Path:
    """Read-only path to the hand-written indexer fixture repository."""
    return INDEXER_REPO


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A writable copy of the fixture repository (for adding broken/excluded files)."""
    dest = tmp_path / "repo"
    shutil.copytree(INDEXER_REPO, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest
