"""Repository scanning and AST-based symbol / import extraction.

Nothing here imports or executes repository code: files are read as text and parsed
with :mod:`ast`. Per-file failures (unreadable, undecodable, syntax errors) are returned
as :class:`IndexIssue` records so one bad file never stops the scan.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import os
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from repoguard.exceptions import PathTraversalError, RepositoryNotFoundError
from repoguard.schemas import (
    MODULE_SCOPE,
    ClassMetadata,
    FileMetadata,
    FunctionMetadata,
    ImportInfo,
    IndexIssue,
    ParameterInfo,
    ParameterKind,
)

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn",
        ".venv", "venv", "env", ".env", "virtualenv",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".tox", ".nox", ".eggs",
        "build", "dist", "site-packages", "node_modules",
        ".idea", ".vscode",
    }
)  # fmt: skip
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = ("*.egg-info", "*.egg-info/*")


# ---------------------------------------------------------------------------
# Paths and scanning
# ---------------------------------------------------------------------------


def resolve_repo_root(repo_path: str | os.PathLike[str]) -> Path:
    root = Path(repo_path).expanduser()
    if not root.exists():
        raise RepositoryNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise RepositoryNotFoundError(f"Repository path is not a directory: {repo_path}")
    return root.resolve()


def safe_join(root: Path, rel_path: str) -> Path:
    """Join ``rel_path`` onto ``root`` and refuse anything that escapes the root
    (``..`` components, absolute paths or symlinks pointing outside)."""
    candidate = (root / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise PathTraversalError(f"Path escapes repository root: {rel_path}")
    return candidate


def _is_excluded(
    rel_posix: str, name: str, exclude_dirs: frozenset[str], globs: Sequence[str]
) -> bool:
    if name in exclude_dirs:
        return True
    return any(fnmatch.fnmatch(rel_posix, g) or fnmatch.fnmatch(name, g) for g in globs)


def scan_repository(
    root: Path,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
    exclude_globs: Sequence[str] = DEFAULT_EXCLUDE_GLOBS,
    include_files: Sequence[str] | None = None,
) -> tuple[list[str], list[IndexIssue]]:
    """Return sorted repository-relative POSIX paths of ``.py`` files plus scan issues.

    If ``include_files`` is given only those paths are considered; missing ones are
    reported as issues. Otherwise the tree is walked recursively without following
    directory symlinks, skipping excluded directories and any virtualenv
    (a directory containing ``pyvenv.cfg``).
    """
    exclude = frozenset(exclude_dirs)
    issues: list[IndexIssue] = []
    found: set[str] = set()

    if include_files is not None:
        for rel in include_files:
            rel_posix = PurePosixPath(Path(rel).as_posix()).as_posix()
            try:
                path = safe_join(root, rel_posix)
            except PathTraversalError as exc:
                issues.append(_issue(rel_posix, "scan", exc))
                continue
            if not path.is_file():
                issues.append(
                    IndexIssue(
                        file=rel_posix, stage="scan", error_type="FileNotFoundError",
                        message=f"File not found: {rel_posix}",
                    )
                )  # fmt: skip
                continue
            found.add(path.relative_to(root).as_posix())
        return sorted(found), issues

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        kept = []
        for d in dirnames:
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if _is_excluded(rel, d, exclude, exclude_globs):
                continue
            if (current / d / "pyvenv.cfg").exists():
                continue
            kept.append(d)
        dirnames[:] = sorted(kept)

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            rel = fname if rel_dir == "." else f"{rel_dir}/{fname}"
            if _is_excluded(rel, fname, exclude, exclude_globs):
                continue
            full = current / fname
            if full.is_symlink():
                try:
                    safe_join(root, rel)
                except PathTraversalError as exc:
                    issues.append(_issue(rel, "scan", exc))
                    continue
                if not full.exists():
                    issues.append(
                        IndexIssue(
                            file=rel, stage="scan", error_type="FileNotFoundError",
                            message=f"Broken symlink: {rel}",
                        )
                    )  # fmt: skip
                    continue
            found.add(rel)
    return sorted(found), issues


def detect_source_roots(root: Path) -> list[str]:
    """Import roots relative to the repo: ``src`` for a src-layout repo (a ``src`` directory
    that is not itself a package), plus the repository root itself."""
    roots = []
    src = root / "src"
    if src.is_dir() and not (src / "__init__.py").exists():
        roots.append("src")
    return roots


def module_name_for(rel_path: str, source_roots: Sequence[str] = ()) -> tuple[str, bool]:
    """``pkg/sub/mod.py`` -> (``pkg.sub.mod``, False); ``pkg/__init__.py`` -> (``pkg``, True).

    A leading source root (e.g. ``src/``) is stripped so ``src/pkg/mod.py`` -> ``pkg.mod``.
    """
    path = PurePosixPath(rel_path)
    for sr in sorted(source_roots, key=len, reverse=True):
        sr_parts = PurePosixPath(sr).parts
        if sr_parts and path.parts[: len(sr_parts)] == sr_parts and len(path.parts) > len(sr_parts):
            path = PurePosixPath(*path.parts[len(sr_parts) :])
            break
    parts = list(path.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return (".".join(parts) if parts else "__init__"), is_package


def read_source(path: Path) -> str:
    """Read Python source honouring PEP 263 encoding cookies / BOM."""
    with tokenize.open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is robust on parsed trees
        return None


def extract_parameters(args: ast.arguments) -> list[ParameterInfo]:
    params: list[ParameterInfo] = []
    positional = [*args.posonlyargs, *args.args]
    # defaults align with the *last* positional parameters
    pad = [None] * (len(positional) - len(args.defaults))
    defaults: list[ast.expr | None] = [*pad, *args.defaults]
    for i, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = (
            ParameterKind.POSITIONAL_ONLY
            if i < len(args.posonlyargs)
            else ParameterKind.POSITIONAL_OR_KEYWORD
        )
        params.append(_param(arg, kind, default))
    if args.vararg is not None:
        params.append(_param(args.vararg, ParameterKind.VAR_POSITIONAL, None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(_param(arg, ParameterKind.KEYWORD_ONLY, default))
    if args.kwarg is not None:
        params.append(_param(args.kwarg, ParameterKind.VAR_KEYWORD, None))
    return params


def _param(arg: ast.arg, kind: ParameterKind, default: ast.expr | None) -> ParameterInfo:
    return ParameterInfo(
        name=arg.arg,
        kind=kind,
        annotation=_unparse(arg.annotation),
        default=_unparse(default),
        has_default=default is not None,
    )


@dataclass
class _Scope:
    qualified_name: str
    kind: str  # "module" | "class" | "function"
    name: str


class SymbolExtractor(ast.NodeVisitor):
    """Collects functions, methods, classes and imports with their enclosing scopes."""

    def __init__(self, rel_path: str, module: str, is_package: bool) -> None:
        self.rel_path = rel_path
        self.module = module
        self.is_package = is_package
        self.functions: list[FunctionMetadata] = []
        self.classes: list[ClassMetadata] = []
        self.imports: list[ImportInfo] = []
        self._stack: list[_Scope] = [_Scope(module, "module", MODULE_SCOPE)]
        self._id_counts: dict[str, int] = {}

    # -- helpers --
    def _qualify(self, name: str) -> str:
        return f"{self._stack[-1].qualified_name}.{name}"

    def _parent(self) -> str | None:
        top = self._stack[-1]
        return None if top.kind == "module" else top.qualified_name

    def _symbol_id(self, qualified_name: str) -> str:
        n = self._id_counts.get(qualified_name, 0) + 1
        self._id_counts[qualified_name] = n
        base = f"{self.rel_path}::{qualified_name}"
        return base if n == 1 else f"{base}#{n}"

    def _scope_label(self) -> str:
        top = self._stack[-1]
        return MODULE_SCOPE if top.kind == "module" else top.qualified_name

    # -- definitions --
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        enclosing = self._stack[-1]
        qualified = self._qualify(node.name)
        params = extract_parameters(node.args)
        self.functions.append(
            FunctionMetadata(
                symbol_id=self._symbol_id(qualified),
                name=node.name,
                qualified_name=qualified,
                type="method" if enclosing.kind == "class" else "function",
                file=self.rel_path,
                module=self.module,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                col=node.col_offset,
                decorator_line=min((d.lineno for d in node.decorator_list), default=None),
                parent=self._parent(),
                decorators=[_unparse(d) or "" for d in node.decorator_list],
                docstring=ast.get_docstring(node),
                parameters=[p.name for p in params],
                parameter_details=params,
                return_annotation=_unparse(node.returns),
                is_async=is_async,
                parent_class=enclosing.name if enclosing.kind == "class" else None,
                is_nested=enclosing.kind == "function",
            )
        )
        self._stack.append(_Scope(qualified, "function", node.name))
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualify(node.name)
        methods = [
            n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        class_vars: list[str] = []
        for stmt in node.body:
            targets: list[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id not in class_vars:
                        class_vars.append(n.id)
        bases = [_unparse(b) or "" for b in node.bases]
        bases += [f"{k.arg}={_unparse(k.value)}" for k in node.keywords if k.arg]
        self.classes.append(
            ClassMetadata(
                symbol_id=self._symbol_id(qualified),
                name=node.name,
                qualified_name=qualified,
                type="class",
                file=self.rel_path,
                module=self.module,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                col=node.col_offset,
                decorator_line=min((d.lineno for d in node.decorator_list), default=None),
                parent=self._parent(),
                decorators=[_unparse(d) or "" for d in node.decorator_list],
                docstring=ast.get_docstring(node),
                bases=bases,
                methods=methods,
                class_variables=class_vars,
            )
        )
        self._stack.append(_Scope(qualified, "class", node.name))
        self.generic_visit(node)
        self._stack.pop()

    # -- imports --
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    file=self.rel_path, line=node.lineno, kind="import",
                    module=alias.name, name=None, alias=alias.asname, level=0,
                    resolved_module=alias.name, scope=self._scope_label(),
                )
            )  # fmt: skip

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        resolved = resolve_relative_module(self.module, self.is_package, node.module, node.level)
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    file=self.rel_path, line=node.lineno, kind="from",
                    module=node.module, name=alias.name, alias=alias.asname,
                    level=node.level, resolved_module=resolved, scope=self._scope_label(),
                )
            )  # fmt: skip


def resolve_relative_module(
    current_module: str, is_package: bool, module: str | None, level: int
) -> str | None:
    """Resolve ``from ..x import y`` to an absolute dotted module name."""
    if level == 0:
        return module
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    if level - 1 > len(package_parts):
        return None  # beyond top-level package
    base = package_parts[: len(package_parts) - (level - 1)]
    if module:
        base = [*base, *module.split(".")]
    return ".".join(base) if base else None


# ---------------------------------------------------------------------------
# Per-file parse
# ---------------------------------------------------------------------------


@dataclass
class ParsedFile:
    metadata: FileMetadata
    source: str = ""
    tree: ast.Module | None = None
    functions: list[FunctionMetadata] = field(default_factory=list)
    classes: list[ClassMetadata] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    issues: list[IndexIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tree is not None


def _issue(file: str, stage: str, exc: BaseException, line: int | None = None) -> IndexIssue:
    return IndexIssue(
        file=file, stage=stage, error_type=type(exc).__name__, message=str(exc), line=line
    )  # type: ignore[arg-type]


def parse_file(root: Path, rel_path: str, source_roots: Sequence[str] = ()) -> ParsedFile:
    """Read and parse one file. Never raises for per-file problems."""
    module, is_package = module_name_for(rel_path, source_roots)
    meta = FileMetadata(path=rel_path, module=module, is_package=is_package)
    result = ParsedFile(metadata=meta)

    try:
        path = safe_join(root, rel_path)
        raw = path.read_bytes()
        meta.size_bytes = len(raw)
        meta.sha256 = hashlib.sha256(raw).hexdigest()
        source = read_source(path)
    except (OSError, UnicodeDecodeError, SyntaxError, PathTraversalError) as exc:
        # tokenize.open raises SyntaxError for an invalid encoding cookie
        meta.parse_status = "error"
        result.issues.append(_issue(rel_path, "read", exc))
        return result

    result.source = source
    meta.line_count = len(source.splitlines())

    try:
        tree = ast.parse(source, filename=rel_path, type_comments=False)
    except SyntaxError as exc:
        meta.parse_status = "error"
        result.issues.append(_issue(rel_path, "parse", exc, line=exc.lineno))
        return result
    except (ValueError, RecursionError, MemoryError) as exc:  # e.g. null bytes, deep nesting
        meta.parse_status = "error"
        result.issues.append(_issue(rel_path, "parse", exc))
        return result

    extractor = SymbolExtractor(rel_path, module, is_package)
    try:
        extractor.visit(tree)
    except RecursionError as exc:
        meta.parse_status = "error"
        result.issues.append(_issue(rel_path, "extract", exc))
        return result

    result.tree = tree
    result.functions = extractor.functions
    result.classes = extractor.classes
    result.imports = extractor.imports
    meta.functions = [f.qualified_name for f in extractor.functions]
    meta.classes = [c.qualified_name for c in extractor.classes]
    meta.import_count = len(extractor.imports)
    return result
