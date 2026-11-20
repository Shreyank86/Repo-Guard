"""Static call graph (Member 1 check 3): direct, nested and cross-module calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from repoguard.repository.callgraph import build_call_graph
from repoguard.repository.parser import parse_file, scan_repository


def _edges(root: Path):
    files, _ = scan_repository(root)
    edges, issues = build_call_graph([parse_file(root, f) for f in files])
    assert issues == []
    return edges


def _edge_set(edges):
    return {
        (e.caller, e.callee_expr, e.callee_qualified, e.resolution.value, e.file, e.line)
        for e in edges
    }


@pytest.fixture
def edges(fixture_repo):
    return _edges(fixture_repo)


def test_cross_module_calls_are_resolved(edges):
    got = _edge_set(edges)
    caller = "pkg.app.App.initialize_app"
    assert (caller, "load_config", "pkg.config.load_config", "imported", "pkg/app.py", 28) in got
    assert (
        caller,
        "cfg.validate",
        "pkg.config.validate",
        "module_attribute",
        "pkg/app.py",
        29,
    ) in got
    assert (caller, "utils.helper", "pkg.utils.helper", "module_attribute", "pkg/app.py", 31) in got
    assert (caller, "len", "builtins.len", "builtin", "pkg/app.py", 31) in got


def test_method_calls_via_self_inheritance_super_and_class(edges):
    got = _edge_set(edges)
    # self.start() is inherited from BaseService
    assert (
        "pkg.app.App.initialize_app",
        "self.start",
        "pkg.app.BaseService.start",
        "method",
        "pkg/app.py",
        30,
    ) in got
    assert (
        "pkg.app.BaseService.start",
        "self.ready",
        "pkg.app.BaseService.ready",
        "method",
        "pkg/app.py",
        11,
    ) in got
    assert (
        "pkg.app.App.restart",
        "super().start",
        "pkg.app.BaseService.start",
        "method",
        "pkg/app.py",
        44,
    ) in got
    assert ("pkg.app.main", "App.build", "pkg.app.App.build", "method", "pkg/app.py", 48) in got
    assert ("pkg.app.App.build", "App", "pkg.app.App", "local", "pkg/app.py", 35) in got


def test_nested_calls_each_produce_an_edge(edges):
    # print(compute(nested(1))) on line 50
    line50 = {
        (e.callee_expr, e.callee_qualified, e.resolution.value) for e in edges if e.line == 50
    }
    assert line50 == {
        ("print", "builtins.print", "builtin"),
        ("compute", "pkg.app.compute", "local"),
        ("nested", "pkg.app.nested", "local"),
    }


def test_nested_function_is_its_own_caller(edges):
    got = _edge_set(edges)
    assert (
        "pkg.app.App.fetch.inner",
        "utils.helper",
        "pkg.utils.helper",
        "module_attribute",
        "pkg/app.py",
        39,
    ) in got
    assert (
        "pkg.app.App.fetch",
        "inner",
        "pkg.app.App.fetch.inner",
        "local",
        "pkg/app.py",
        41,
    ) in got


def test_module_level_calls_and_module_instances(edges):
    got = _edge_set(edges)
    assert ("pkg.app.<module>", "App", "pkg.app.App", "local", "pkg/app.py", 61) in got
    # app_instance = App("x") is assigned once at module level -> method resolvable
    assert (
        "pkg.app.<module>",
        "app_instance.initialize_app",
        "pkg.app.App.initialize_app",
        "method",
        "pkg/app.py",
        62,
    ) in got
    module_edge = next(e for e in edges if e.line == 62)
    assert module_edge.caller_name == "<module>" and module_edge.callee == "initialize_app"


def test_external_and_unresolved_calls_are_labelled_not_guessed(edges):
    got = _edge_set(edges)
    assert (
        "pkg.config.load_config",
        "json.loads",
        "json.loads",
        "external",
        "pkg/config.py",
        11,
    ) in got
    assert (
        "pkg.config.load_config",
        "Path",
        "pathlib.Path",
        "external",
        "pkg/config.py",
        11,
    ) in got
    # `app` is a local variable: we must not invent a target for app.initialize_app()
    assert ("pkg.app.main", "app.initialize_app", None, "unresolved", "pkg/app.py", 49) in got
    assert ("pkg.config.load_config", "data.update", None, "unresolved", "pkg/config.py", 13) in got


def test_same_module_call_and_raise(edges):
    got = _edge_set(edges)
    assert (
        "pkg.config.load_config",
        "validate",
        "pkg.config.validate",
        "local",
        "pkg/config.py",
        12,
    ) in got
    assert (
        "pkg.config.validate",
        "ValueError",
        "builtins.ValueError",
        "builtin",
        "pkg/config.py",
        19,
    ) in got


def test_no_phantom_edges_and_every_edge_matches_source_text(fixture_repo, edges):
    for e in edges:
        line = (fixture_repo / e.file).read_text().splitlines()[e.line - 1]
        assert e.callee in line, e
    callers = {e.caller for e in edges}
    assert "pkg.app.compute" not in callers and "pkg.utils.helper" not in callers


def test_edges_are_sorted_deterministically(edges):
    keys = [(e.file, e.line, e.col, e.caller, e.callee_expr) for e in edges]
    assert keys == sorted(keys)


def test_decorators_and_defaults_belong_to_enclosing_scope(tmp_path):
    (tmp_path / "m.py").write_text(
        "def deco(f):\n    return f\n\ndef default():\n    return 1\n\n"
        "class C:\n    @deco\n    def m(self, x=default()):\n        pass\n"
    )
    edges = _edges(tmp_path)
    got = {(e.caller, e.callee_qualified) for e in edges}
    assert got == {("m.C", "m.default")}  # bare @deco is not a call; default() runs in class body


def test_shadowing_parameter_is_not_resolved_to_module_instance(tmp_path):
    (tmp_path / "m.py").write_text(
        "class S:\n    def go(self):\n        pass\n\ninst = S()\n\ndef f(inst):\n    inst.go()\n"
    )
    e = next(e for e in _edges(tmp_path) if e.callee_expr == "inst.go")
    assert e.resolution.value == "unresolved"


def test_reexport_through_package_init(tmp_path):
    pkg = tmp_path / "lib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .core import run\n")
    (pkg / "core.py").write_text("def run():\n    pass\n")
    (tmp_path / "main.py").write_text("import lib\nfrom lib import run\nrun()\nlib.run()\n")
    got = {(e.callee_expr, e.callee_qualified, e.resolution.value) for e in _edges(tmp_path)}
    assert got == {
        ("run", "lib.core.run", "imported"),
        ("lib.run", "lib.core.run", "module_attribute"),
    }


def test_reexport_of_external_name_is_external(tmp_path):
    pkg = tmp_path / "lib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "compat.py").write_text("from urllib.parse import urlparse\n")
    (pkg / "use.py").write_text("from .compat import urlparse\nurlparse('x')\n")
    (e,) = _edges(tmp_path)
    assert (e.callee_qualified, e.resolution.value) == ("urllib.parse.urlparse", "external")


def test_files_with_errors_are_skipped_but_others_processed(tmp_repo):
    (tmp_repo / "pkg" / "broken.py").write_text("def x(:\n")
    files, _ = scan_repository(tmp_repo)
    parsed = [parse_file(tmp_repo, f) for f in files]
    edges, issues = build_call_graph(parsed)
    assert issues == [] and edges and all(e.file != "pkg/broken.py" for e in edges)


def test_dynamic_callees_are_marked(tmp_path):
    (tmp_path / "m.py").write_text(
        "def make():\n    return print\n\nmake()('x')\nhandlers = {}\nhandlers['a']()\n"
    )
    dyn = [e for e in _edges(tmp_path) if e.callee == "<dynamic>"]
    assert {e.callee_expr for e in dyn} == {"make()", "handlers['a']"}
    assert all(e.resolution.value == "unresolved" and e.callee_qualified is None for e in dyn)
