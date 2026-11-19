"""End-to-end indexer: serialization, schema, error tolerance, repeatability (check 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repoguard.exceptions import IndexingError, IndexNotFoundError, RepositoryNotFoundError
from repoguard.repository import build_index, load_index
from repoguard.repository.indexer import CALLGRAPH_FILE, MANIFEST_FILE, PASSAGES_FILE, SYMBOLS_FILE
from repoguard.schemas import INDEX_SCHEMA_VERSION

ARTEFACTS = (SYMBOLS_FILE, CALLGRAPH_FILE, PASSAGES_FILE, MANIFEST_FILE)


@pytest.fixture
def built(fixture_repo, tmp_path):
    return build_index(fixture_repo, tmp_path / "index")


def test_writes_all_artefacts_with_documented_top_level_keys(built):
    out = Path(built.output_dir)
    for name in ARTEFACTS:
        assert (out / name).is_file()
    symbols = json.loads((out / SYMBOLS_FILE).read_text())
    assert list(symbols) == [
        "schema_version",
        "repository",
        "files",
        "symbols",
        "imports",
        "errors",
    ]
    assert symbols["schema_version"] == INDEX_SCHEMA_VERSION
    callgraph = json.loads((out / CALLGRAPH_FILE).read_text())
    assert list(callgraph) == ["schema_version", "repository_id", "edges"]
    assert set(callgraph["edges"][0]) == {
        "caller", "caller_name", "callee", "callee_expr", "callee_qualified", "resolution", "file", "line", "col",
    }  # fmt: skip
    first_passage = json.loads((out / PASSAGES_FILE).read_text().splitlines()[0])
    assert {"passage_id", "file_path", "start_line", "end_line", "content"} <= set(first_passage)


def test_symbol_record_matches_shared_contract(built):
    """The PDF's illustrative record shape: name/qualified_name/type/file/line/end_line/parameters/return_annotation."""
    doc = json.loads(Path(built.symbols_path).read_text())
    rec = next(s for s in doc["symbols"] if s["qualified_name"] == "pkg.config.load_config")
    assert {
        k: rec[k]
        for k in (
            "name",
            "qualified_name",
            "type",
            "file",
            "line",
            "end_line",
            "parameters",
            "return_annotation",
        )
    } == {
        "name": "load_config",
        "qualified_name": "pkg.config.load_config",
        "type": "function",
        "file": "pkg/config.py",
        "line": 9,
        "end_line": 14,
        "parameters": ["path", "strict", "encoding", "overrides"],
        "return_annotation": "dict",
    }


def test_counts_and_repository_metadata(built):
    repo = built.repository
    assert (repo.total_files, repo.indexed_files, repo.failed_files, repo.status) == (
        4,
        4,
        0,
        "ready",
    )
    assert repo.total_symbols == len(built.symbols) == 17
    assert repo.repository_id == "indexer-repo"
    manifest = json.loads(Path(built.manifest_path).read_text())
    assert manifest["counts"]["classes"] == 2 and manifest["counts"]["methods"] == 7
    assert manifest["counts"]["call_edges"] == len(built.call_edges)
    assert set(manifest["artefacts"]) == {SYMBOLS_FILE, CALLGRAPH_FILE, PASSAGES_FILE}
    assert built.errors == []


def test_file_metadata(built):
    files = {f.path: f for f in built.files}
    app = files["pkg/app.py"]
    assert app.module == "pkg.app" and not app.is_package and app.line_count == 62
    assert app.classes == ["pkg.app.BaseService", "pkg.app.App"] and app.import_count == 3
    assert files["pkg/__init__.py"].is_package and len(app.sha256) == 64


# ---- repeatability --------------------------------------------------------


def test_two_runs_produce_byte_identical_output(fixture_repo, tmp_path):
    cfg = {"embeddings": {"enabled": True, "backend": "hashing"}}
    a = build_index(fixture_repo, tmp_path / "a", cfg)
    b = build_index(fixture_repo, tmp_path / "b", cfg)
    names = sorted(p.name for p in Path(a.output_dir).iterdir())
    assert names == sorted(p.name for p in Path(b.output_dir).iterdir())
    for name in names:
        assert (Path(a.output_dir) / name).read_bytes() == (
            Path(b.output_dir) / name
        ).read_bytes(), name


def test_rebuild_in_place_is_stable_and_removes_stale_vectors(fixture_repo, tmp_path):
    out = tmp_path / "idx"
    build_index(fixture_repo, out, {"embeddings": {"enabled": True, "backend": "hashing"}})
    first = (out / SYMBOLS_FILE).read_bytes()
    build_index(fixture_repo, out)
    assert (out / SYMBOLS_FILE).read_bytes() == first
    assert not (out / "vector_meta.json").exists()
    assert not list(out.glob("*.tmp"))


# ---- error tolerance ------------------------------------------------------


def test_missing_repository_raises(tmp_path):
    with pytest.raises(RepositoryNotFoundError):
        build_index(tmp_path / "does-not-exist", tmp_path / "out")


def test_repo_path_that_is_a_file_raises(tmp_path):
    (tmp_path / "f.py").write_text("")
    with pytest.raises(RepositoryNotFoundError):
        build_index(tmp_path / "f.py", tmp_path / "out")


def test_malformed_and_missing_files_are_recorded_and_others_indexed(tmp_repo, tmp_path):
    (tmp_repo / "pkg" / "broken.py").write_text("def broken(:\n    pass\n")
    files = [
        "pkg/app.py",
        "pkg/broken.py",
        "pkg/missing.py",
        "pkg/config.py",
        "pkg/utils.py",
        "pkg/__init__.py",
    ]
    art = build_index(tmp_repo, tmp_path / "out", {"include_files": files})
    errors = {(e.file, e.stage, e.error_type) for e in art.errors}
    assert errors == {
        ("pkg/broken.py", "parse", "SyntaxError"),
        ("pkg/missing.py", "scan", "FileNotFoundError"),
    }
    assert art.repository.status == "ready" and art.repository.failed_files == 1
    assert {s.qualified_name for s in art.symbols} >= {"pkg.app.App", "pkg.config.load_config"}
    broken = next(f for f in art.files if f.path == "pkg/broken.py")
    assert broken.parse_status == "error" and broken.functions == []
    # errors are persisted for the verifier
    doc = json.loads(Path(art.symbols_path).read_text())
    assert len(doc["errors"]) == 2


def test_all_files_failing_marks_index_failed(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def (:\n")
    art = build_index(repo, tmp_path / "out")
    assert art.repository.status == "failed" and art.symbols == []


def test_empty_repository(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    art = build_index(repo, tmp_path / "out")
    assert art.repository.total_files == 0 and art.repository.status == "ready"
    assert Path(art.passages_path).read_text() == ""


def test_unknown_config_key_is_rejected(fixture_repo, tmp_path):
    with pytest.raises(IndexingError):
        build_index(fixture_repo, tmp_path / "out", {"embedings": {}})


def test_src_layout_is_detected(tmp_path):
    repo = tmp_path / "proj"
    (repo / "src" / "lib").mkdir(parents=True)
    (repo / "src" / "lib" / "__init__.py").write_text("def api():\n    pass\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_lib.py").write_text("from lib import api\napi()\n")
    art = build_index(repo, tmp_path / "out")
    assert {s.qualified_name for s in art.symbols} == {"lib.api"}
    (edge,) = art.call_edges
    assert (edge.callee_qualified, edge.resolution.value) == ("lib.api", "imported")


def test_provenance_overrides(fixture_repo, tmp_path):
    art = build_index(
        fixture_repo, tmp_path / "out",
        {"repository": {"name": "Fixture", "source_url": "https://example.org/x", "commit": "abc", "licence": "MIT"}},
    )  # fmt: skip
    r = art.repository
    assert (r.repository_id, r.source_url, r.commit, r.licence) == (
        "fixture",
        "https://example.org/x",
        "abc",
        "MIT",
    )


# ---- passages ----------------------------------------------------------------


def test_function_passages_are_exact_source_slices(fixture_repo, built):
    for p in built.passages:
        if p.passage_type in ("function", "method"):
            lines = (fixture_repo / p.file_path).read_text().splitlines()
            assert p.content == "\n".join(lines[p.start_line - 1 : p.end_line]), p.passage_id


def test_passage_ids_unique_and_linked_to_symbols(built):
    ids = [p.passage_id for p in built.passages]
    assert len(ids) == len(set(ids))
    symbol_ids = {s.symbol_id for s in built.symbols}
    for p in built.passages:
        assert p.symbol_id is None or p.symbol_id in symbol_ids


def test_passage_kinds(built):
    by_id = {p.passage_id: p for p in built.passages}
    build = by_id["pkg/app.py::pkg.app.App.build"]
    assert build.start_line == 33 and build.content.startswith("    @staticmethod")
    assert "def build(path)" in build.header
    # nested functions live inside the parent's passage, not their own
    assert "pkg/app.py::pkg.app.App.fetch.inner" not in by_id
    assert "def inner(x)" in by_id["pkg/app.py::pkg.app.App.fetch"].content
    # class overview elides method bodies but keeps signatures
    cls = by_id["pkg/app.py::pkg.app.App"]
    assert cls.passage_type == "class" and "def initialize_app(self):" in cls.content
    assert "load_config(self.path)" not in cls.content and 'name: str = "app"' in cls.content
    # module overview keeps imports / top-level code, elides bodies
    mod = by_id["pkg/app.py::<module>"]
    assert (
        "from pkg.config import load_config" in mod.content and "app_instance = App" in mod.content
    )
    assert "return self.ready()" not in mod.content
    # empty-ish modules still produce an overview; files with only defs too
    assert "pkg/utils.py::<module>" in by_id


def test_long_functions_are_split_with_overlap(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    body = "\n".join(f"    x{i} = {i}" for i in range(25))
    (repo / "big.py").write_text(f"def big():\n{body}\n")
    art = build_index(repo, tmp_path / "out", {"passages": {"max_lines": 10, "overlap_lines": 2}})
    parts = [p for p in art.passages if p.symbol == "big.big"]
    assert [p.passage_id for p in parts] == [f"big.py::big.big@{i}" for i in range(1, 4)]
    assert [(p.start_line, p.end_line) for p in parts] == [(1, 10), (9, 18), (17, 26)]
    assert all(p.total_parts == 3 for p in parts)


def test_invalid_passage_config(fixture_repo, tmp_path):
    with pytest.raises(ValueError):
        build_index(
            fixture_repo, tmp_path / "out", {"passages": {"max_lines": 5, "overlap_lines": 5}}
        )


# ---- load ------------------------------------------------------------------------


def test_load_index_round_trip(built):
    loaded = load_index(built.output_dir)
    for attr in ("repository", "files", "symbols", "imports", "call_edges", "passages", "errors"):
        assert getattr(loaded, attr) == getattr(built, attr), attr


def test_load_index_helpers(built):
    idx = load_index(built.output_dir)
    assert [s.file for s in idx.find_symbols("load_config")] == ["pkg/config.py"]
    assert idx.find_symbols("load_config", file="pkg/app.py") == []
    assert idx.find_symbols("pkg.app.App", type="class")[0].name == "App"
    assert idx.find_symbols("load_confg") == []
    callers = {e.caller for e in idx.callers_of("load_config")}
    assert callers == {"pkg.app.App.initialize_app"}
    assert {e.callee for e in idx.callees_of("main")} == {
        "build",
        "initialize_app",
        "print",
        "compute",
        "nested",
    }
    assert "pkg/missing.py" not in idx.file_paths()


def test_load_index_missing_dir(tmp_path):
    with pytest.raises(IndexNotFoundError):
        load_index(tmp_path / "nothing")


def test_load_index_rejects_other_schema_version(built):
    p = Path(built.symbols_path)
    doc = json.loads(p.read_text())
    doc["schema_version"] = "0.0"
    p.write_text(json.dumps(doc))
    with pytest.raises(IndexNotFoundError):
        load_index(built.output_dir)
