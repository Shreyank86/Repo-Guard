"""Scanning, symbol extraction, parameters, line ranges and imports (Member 1 checks 1, 2, 4)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from repoguard.exceptions import PathTraversalError
from repoguard.repository.parser import (
    module_name_for,
    parse_file,
    resolve_relative_module,
    safe_join,
    scan_repository,
)

# Every definition in tests/fixtures/indexer_repo, written out by hand:
# qualified name -> (type, line, end_line)
EXPECTED_SYMBOLS = {
    "pkg.config.load_config": ("function", 9, 14),
    "pkg.config.validate": ("function", 17, 20),
    "pkg.config.parse_args": ("function", 23, 24),
    "pkg.utils.helper": ("function", 1, 2),
    "pkg.app.BaseService": ("class", 9, 14),
    "pkg.app.BaseService.start": ("method", 10, 11),
    "pkg.app.BaseService.ready": ("method", 13, 14),
    "pkg.app.App": ("class", 17, 44),
    "pkg.app.App.__init__": ("method", 23, 25),
    "pkg.app.App.initialize_app": ("method", 27, 31),
    "pkg.app.App.build": ("method", 34, 35),
    "pkg.app.App.fetch": ("method", 37, 41),
    "pkg.app.App.fetch.inner": ("function", 38, 39),
    "pkg.app.App.restart": ("method", 43, 44),
    "pkg.app.main": ("function", 47, 50),
    "pkg.app.compute": ("function", 53, 54),
    "pkg.app.nested": ("function", 57, 58),
}

FILES = ["pkg/__init__.py", "pkg/app.py", "pkg/config.py", "pkg/utils.py"]


def _all_symbols(root: Path):
    out = {}
    for rel in FILES:
        pf = parse_file(root, rel)
        assert pf.ok, pf.issues
        for s in (*pf.functions, *pf.classes):
            assert s.qualified_name not in out, f"duplicate {s.qualified_name}"
            out[s.qualified_name] = s
    return out


# ---------------------------------------------------------------------------
# 1. function / class extraction: all expected found, nothing phantom
# ---------------------------------------------------------------------------


def test_extracts_exactly_the_expected_definitions(fixture_repo):
    symbols = _all_symbols(fixture_repo)
    assert set(symbols) == set(EXPECTED_SYMBOLS)


def test_types_and_line_ranges_match_source(fixture_repo):
    symbols = _all_symbols(fixture_repo)
    for qn, (typ, line, end) in EXPECTED_SYMBOLS.items():
        s = symbols[qn]
        assert (s.type, s.line, s.end_line) == (typ, line, end), qn


def test_reported_line_really_holds_the_definition(fixture_repo):
    """Cross-check against raw text: the `line` must contain `def name` / `class name`."""
    for qn, s in _all_symbols(fixture_repo).items():
        text = (fixture_repo / s.file).read_text().splitlines()[s.line - 1]
        keyword = "class" if s.type == "class" else "def"
        assert f"{keyword} {s.name}" in text, qn


def test_symbol_metadata_details(fixture_repo):
    s = _all_symbols(fixture_repo)
    fetch = s["pkg.app.App.fetch"]
    assert fetch.is_async and fetch.parent_class == "App" and fetch.parent == "pkg.app.App"
    inner = s["pkg.app.App.fetch.inner"]
    assert inner.is_nested and inner.parent == "pkg.app.App.fetch" and inner.parent_class is None
    build = s["pkg.app.App.build"]
    assert build.decorators == ["staticmethod"] and build.decorator_line == 33
    assert s["pkg.config.load_config"].docstring == "Load a JSON config file."
    assert s["pkg.app.main"].parent is None and not s["pkg.app.main"].is_async

    app = s["pkg.app.App"]
    assert app.bases == ["BaseService"]
    assert app.methods == ["__init__", "initialize_app", "build", "fetch", "restart"]
    assert app.class_variables == ["name", "retries"]
    assert app.docstring == "Main application."
    assert s["pkg.app.BaseService"].bases == []


def test_symbol_ids_are_unique_and_include_file(fixture_repo):
    s = _all_symbols(fixture_repo)
    ids = [x.symbol_id for x in s.values()]
    assert len(ids) == len(set(ids))
    assert s["pkg.config.validate"].symbol_id == "pkg/config.py::pkg.config.validate"


def test_duplicate_definitions_get_suffixed_ids(tmp_path):
    (tmp_path / "dup.py").write_text(
        "import sys\nif sys.version_info > (3,):\n    def f(): pass\nelse:\n    def f(): pass\n"
    )
    pf = parse_file(tmp_path, "dup.py")
    assert [f.symbol_id for f in pf.functions] == ["dup.py::dup.f", "dup.py::dup.f#2"]


# ---------------------------------------------------------------------------
# 2. parameters: names, order, kinds, defaults, annotations
# ---------------------------------------------------------------------------


def test_parameters_all_kinds_in_order(fixture_repo):
    fn = _all_symbols(fixture_repo)["pkg.config.parse_args"]
    assert fn.parameters == ["a", "b", "c", "args", "d", "e", "kwargs"]
    got = [(p.name, p.kind.value, p.default, p.annotation) for p in fn.parameter_details]
    assert got == [
        ("a", "positional_only", None, None),
        ("b", "positional_only", "1", None),
        ("c", "positional_or_keyword", "2", None),
        ("args", "var_positional", None, None),
        ("d", "keyword_only", None, None),
        ("e", "keyword_only", "3", "int"),
        ("kwargs", "var_keyword", None, None),
    ]
    assert fn.return_annotation == "None"


def test_parameters_annotations_defaults_and_return(fixture_repo):
    fn = _all_symbols(fixture_repo)["pkg.config.load_config"]
    assert fn.parameters == ["path", "strict", "encoding", "overrides"]
    details = {p.name: p for p in fn.parameter_details}
    assert details["path"].annotation == "str" and not details["path"].has_default
    assert details["strict"].default == "False" and details["strict"].has_default
    assert (
        details["encoding"].kind.value == "keyword_only"
        and details["encoding"].default == "'utf-8'"
    )
    assert details["overrides"].kind.value == "var_keyword"
    assert fn.return_annotation == "dict"
    assert "retries" not in fn.parameters  # the PDF's trap example


def test_method_parameters_include_self(fixture_repo):
    s = _all_symbols(fixture_repo)
    assert s["pkg.app.App.fetch"].parameters == ["self", "url", "timeout"]
    assert s["pkg.app.App.build"].parameters == ["path"]  # staticmethod


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------


def test_imports_preserve_module_name_alias_and_resolution(fixture_repo):
    pf = parse_file(fixture_repo, "pkg/app.py")
    got = {
        (i.kind, i.module, i.name, i.alias, i.level, i.resolved_module, i.line) for i in pf.imports
    }
    assert got == {
        ("from", "pkg", "config", "cfg", 0, "pkg", 3),
        ("from", "pkg.config", "load_config", None, 0, "pkg.config", 4),
        ("from", None, "utils", None, 1, "pkg", 6),
    }
    init = parse_file(fixture_repo, "pkg/__init__.py")
    assert [(i.module, i.name, i.resolved_module) for i in init.imports] == [
        ("config", "load_config", "pkg.config")
    ]
    cfg = parse_file(fixture_repo, "pkg/config.py")
    assert [(i.kind, i.module, i.name) for i in cfg.imports] == [
        ("import", "json", None),
        ("from", "pathlib", "Path"),
    ]


def test_function_scoped_import_records_scope(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    import os.path as p\n    return p\n")
    (imp,) = parse_file(tmp_path, "m.py").imports
    assert (imp.module, imp.alias, imp.scope) == ("os.path", "p", "m.f")


@pytest.mark.parametrize(
    ("module", "is_pkg", "mod", "level", "expected"),
    [
        ("pkg.app", False, "utils", 1, "pkg.utils"),
        ("pkg.app", False, None, 1, "pkg"),
        ("pkg", True, "config", 1, "pkg.config"),
        ("a.b.c", False, "x", 2, "a.x"),
        ("a.b.c", False, None, 3, None),
        ("top", False, "x", 2, None),
        ("pkg.app", False, "os", 0, "os"),
    ],
)
def test_resolve_relative_module(module, is_pkg, mod, level, expected):
    assert resolve_relative_module(module, is_pkg, mod, level) == expected


@pytest.mark.parametrize(
    ("rel", "roots", "expected"),
    [
        ("pkg/sub/mod.py", (), ("pkg.sub.mod", False)),
        ("pkg/__init__.py", (), ("pkg", True)),
        ("setup.py", (), ("setup", False)),
        ("src/pkg/mod.py", ("src",), ("pkg.mod", False)),
        ("src/pkg/__init__.py", ("src",), ("pkg", True)),
        ("tests/test_x.py", ("src",), ("tests.test_x", False)),
    ],
)
def test_module_name_for(rel, roots, expected):
    assert module_name_for(rel, roots) == expected


# ---------------------------------------------------------------------------
# scanning, exclusions, path safety
# ---------------------------------------------------------------------------


def test_scan_finds_python_files_sorted_with_relative_paths(fixture_repo):
    files, issues = scan_repository(fixture_repo)
    assert files == FILES and issues == []


def test_scan_skips_virtualenvs_builds_and_caches(tmp_repo):
    for rel in [
        ".venv/lib/x.py", "venv/y.py", "build/lib/z.py", "dist/d.py", "__pycache__/c.py",
        "pkg.egg-info/e.py", "node_modules/n.py", "custom_env/lib/v.py", ".git/hooks/h.py",
    ]:  # fmt: skip
        p = tmp_repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    (tmp_repo / "custom_env" / "pyvenv.cfg").write_text("home = /usr\n")
    (tmp_repo / "README.md").write_text("# not python\n")
    files, _ = scan_repository(tmp_repo)
    assert files == FILES


def test_scan_respects_configured_exclusions(tmp_repo):
    from repoguard.repository.parser import DEFAULT_EXCLUDE_DIRS

    files, _ = scan_repository(tmp_repo, DEFAULT_EXCLUDE_DIRS, exclude_globs=["pkg/u*.py"])
    assert "pkg/utils.py" not in files and "pkg/app.py" in files


def test_include_files_reports_missing_paths(fixture_repo):
    files, issues = scan_repository(
        fixture_repo, include_files=["pkg/app.py", "pkg/missing.py", "../outside.py"]
    )
    assert files == ["pkg/app.py"]
    by_file = {i.file: i.error_type for i in issues}
    assert by_file == {"pkg/missing.py": "FileNotFoundError", "../outside.py": "PathTraversalError"}


def test_safe_join_blocks_traversal(fixture_repo):
    root = fixture_repo.resolve()
    assert safe_join(root, "pkg/app.py") == root / "pkg" / "app.py"
    with pytest.raises(PathTraversalError):
        safe_join(root, "../../etc/passwd")
    with pytest.raises(PathTraversalError):
        safe_join(root, "/etc/passwd")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_scan_reports_symlinks_escaping_root_and_broken_links(tmp_repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def secret(): pass\n")
    os.symlink(outside, tmp_repo / "pkg" / "escape.py")
    os.symlink(tmp_repo / "pkg" / "nope.py", tmp_repo / "pkg" / "dangling.py")
    files, issues = scan_repository(tmp_repo.resolve())
    assert files == FILES
    assert {(i.file, i.error_type) for i in issues} == {
        ("pkg/escape.py", "PathTraversalError"),
        ("pkg/dangling.py", "FileNotFoundError"),
    }


# ---------------------------------------------------------------------------
# 4. malformed files never raise
# ---------------------------------------------------------------------------


def test_syntax_error_is_recorded_not_raised(tmp_path):
    (tmp_path / "bad.py").write_text("def ok():\n    return 1\n\ndef broken(:\n    pass\n")
    pf = parse_file(tmp_path, "bad.py")
    assert not pf.ok and pf.metadata.parse_status == "error"
    (issue,) = pf.issues
    assert (issue.stage, issue.error_type, issue.line) == ("parse", "SyntaxError", 4)
    assert pf.functions == []


def test_undecodable_file_is_recorded(tmp_path):
    (tmp_path / "latin.py").write_bytes(b"x = '\xff\xfe'\n")
    pf = parse_file(tmp_path, "latin.py")
    assert not pf.ok and pf.issues[0].stage == "read"


def test_encoding_cookie_is_honoured(tmp_path):
    (tmp_path / "enc.py").write_bytes(
        "# -*- coding: latin-1 -*-\ndef caf\xe9(): pass\n".encode("latin-1")
    )
    pf = parse_file(tmp_path, "enc.py")
    assert pf.ok and pf.functions[0].name == "café"


def test_null_bytes_are_recorded(tmp_path):
    (tmp_path / "nul.py").write_bytes(b"x = 1\x00\n")
    pf = parse_file(tmp_path, "nul.py")
    assert not pf.ok and pf.issues[0].stage == "parse"


def test_missing_file_is_recorded(tmp_path):
    pf = parse_file(tmp_path, "ghost.py")
    assert not pf.ok and pf.issues[0].error_type == "FileNotFoundError"


def test_parser_never_executes_repository_code(tmp_path):
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('x')\ndef f(): pass\n")
    pf = parse_file(tmp_path, "evil.py")
    assert pf.ok and not marker.exists()
