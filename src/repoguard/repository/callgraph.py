"""Static call-graph extraction.

Every ``ast.Call`` node becomes one :class:`CallEdge` from the enclosing function,
method, class body or module to the called expression. Resolution to a repository
symbol is attempted for the statically visible cases:

* plain names defined in the same module or in an enclosing function   -> ``local``
* names bound by ``import`` / ``from ... import`` (incl. re-exports)    -> ``imported``
* ``self.m()`` / ``cls.m()`` / ``super().m()`` / ``Class.m()``          -> ``method``
* ``module.func()`` where ``module`` is an indexed module               -> ``module_attribute``
* names imported from modules outside the repository                    -> ``external``
* Python builtins                                                        -> ``builtin``

Anything else (calls on attributes of instances, call results, subscripts, dynamic
dispatch, ``getattr``, monkey-patching) is recorded with ``resolution="unresolved"``.
The graph is therefore a *static approximation*; see docs/index_schema.md.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field

from repoguard.repository.parser import ParsedFile
from repoguard.schemas import (
    MODULE_SCOPE,
    CallEdge,
    CallResolution,
    ClassMetadata,
    FunctionMetadata,
    ImportInfo,
    IndexIssue,
)

_BUILTINS = frozenset(dir(builtins))
DYNAMIC_CALLEE = "<dynamic>"
"""``callee`` value when the called object is not a name or attribute (call result, subscript, lambda)."""
_MODULE = "module"  # internal marker: a name bound to an indexed module (never stored on edges)
_MAX_ALIAS_DEPTH = 8


@dataclass
class _ModuleContext:
    module: str
    top_level: dict[str, str] = field(default_factory=dict)  # short name -> qualified
    module_bindings: dict[str, str] = field(default_factory=dict)  # name -> dotted target
    scoped_bindings: dict[str, dict[str, str]] = field(default_factory=dict)  # scope -> {...}
    instances: dict[str, str] = field(default_factory=dict)  # var -> constructor expr (dotted)


def _import_bindings(
    imports: Iterable[ImportInfo],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    module_level: dict[str, str] = {}
    scoped: dict[str, dict[str, str]] = {}
    for imp in imports:
        if imp.kind == "import":
            assert imp.module is not None
            if imp.alias:
                name, target = imp.alias, imp.module
            else:  # `import a.b.c` binds `a`
                name = target = imp.module.split(".")[0]
        else:
            if imp.name == "*" or imp.resolved_module is None or imp.name is None:
                continue
            name = imp.alias or imp.name
            target = f"{imp.resolved_module}.{imp.name}"
        bucket = module_level if imp.scope == MODULE_SCOPE else scoped.setdefault(imp.scope, {})
        bucket[name] = target
    return module_level, scoped


class RepoSymbolTable:
    """Global view of all parsed files used to resolve call targets."""

    def __init__(self, parsed_files: Iterable[ParsedFile]) -> None:
        self.modules: dict[str, _ModuleContext] = {}
        self.functions: dict[str, FunctionMetadata] = {}
        self.classes: dict[str, ClassMetadata] = {}
        for pf in parsed_files:
            if not pf.ok:
                continue
            mod = pf.metadata.module
            ctx = _ModuleContext(module=mod)
            for f in pf.functions:
                self.functions.setdefault(f.qualified_name, f)
                if f.parent is None:
                    ctx.top_level[f.name] = f.qualified_name
            for c in pf.classes:
                self.classes.setdefault(c.qualified_name, c)
                if c.parent is None:
                    ctx.top_level[c.name] = c.qualified_name
            ctx.module_bindings, ctx.scoped_bindings = _import_bindings(pf.imports)
            if pf.tree is not None:
                ctx.instances = _module_instances(pf.tree)
            self.modules[mod] = ctx

    def instance_class(self, ctx: _ModuleContext, name: str) -> str | None:
        """Class of a module-level ``name = SomeClass(...)`` variable, if it is a repo class."""
        ctor = ctx.instances.get(name)
        if ctor is None:
            return None
        target = self.resolve_name_in_module(ctx, ctor)
        return target if target in self.classes else None

    # ---- lookups ----
    def is_symbol(self, qualified: str) -> bool:
        return qualified in self.functions or qualified in self.classes

    def is_repo_module(self, dotted: str) -> bool:
        return dotted in self.modules

    def in_repo_namespace(self, dotted: str) -> bool:
        top = dotted.split(".")[0]
        return any(m == top or m.startswith(top + ".") for m in self.modules)

    def follow(self, dotted: str, depth: int = 0) -> str:
        """Chase import aliases / re-exports / inherited methods to a final dotted name.

        The result is a repo symbol, a repo module, or a name that could not be followed
        further (possibly outside the repository).
        """
        if depth > _MAX_ALIAS_DEPTH or self.is_symbol(dotted) or dotted in self.modules:
            return dotted
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):  # longest known prefix first
            prefix = ".".join(parts[:i])
            rest = parts[i:]
            if prefix in self.modules:
                ctx = self.modules[prefix]
                head = rest[0]
                if head in ctx.top_level:
                    target = ".".join([ctx.top_level[head], *rest[1:]])
                elif head in ctx.module_bindings:
                    target = ".".join([ctx.module_bindings[head], *rest[1:]])
                else:
                    return dotted
                return dotted if target == dotted else self.follow(target, depth + 1)
            if prefix in self.classes and len(rest) == 1:
                return self.lookup_method(prefix, rest[0]) or dotted
        return dotted

    def resolve_dotted(self, dotted: str) -> str | None:
        """Resolve ``a.b.c`` to a repository symbol, or ``None``."""
        final = self.follow(dotted)
        return final if self.is_symbol(final) else None

    def lookup_method(
        self, class_qualified: str, method: str, _seen: set[str] | None = None
    ) -> str | None:
        """Find ``method`` on a class or, breadth-first, on its in-repo base classes."""
        seen = _seen if _seen is not None else set()
        queue = [class_qualified]
        while queue:
            cq = queue.pop(0)
            if cq in seen:
                continue
            seen.add(cq)
            cls = self.classes.get(cq)
            if cls is None:
                continue
            if method in cls.methods:
                return f"{cq}.{method}"
            queue.extend(b for b in self.base_classes(cls) if b not in seen)
        return None

    def base_classes(self, cls: ClassMetadata) -> list[str]:
        out = []
        ctx = self.modules.get(cls.module)
        for base in cls.bases:
            if "=" in base or ctx is None:  # metaclass=..., etc.
                continue
            target = self.resolve_name_in_module(ctx, base.split("[")[0])
            if target and target in self.classes:
                out.append(target)
        return out

    def resolve_name_in_module(self, ctx: _ModuleContext, dotted: str) -> str | None:
        head, *rest = dotted.split(".")
        if head in ctx.top_level:
            base = ctx.top_level[head]
        elif head in ctx.module_bindings:
            base = ctx.module_bindings[head]
        else:
            return None
        return self.resolve_dotted(".".join([base, *rest]))


def _module_instances(tree: ast.Module) -> dict[str, str]:
    """Module-level variables bound exactly once, to a ``Name(...)``/``a.B(...)`` call."""
    assigned: dict[str, int] = {}
    ctor: dict[str, str] = {}
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets, value = [stmt.target], stmt.value
        elif isinstance(stmt, (ast.AugAssign, ast.For, ast.With, ast.Import, ast.ImportFrom)):
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    assigned[n.id] = assigned.get(n.id, 0) + 2
            continue
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    assigned[n.id] = assigned.get(n.id, 0) + 1
            if isinstance(t, ast.Name) and isinstance(value, ast.Call):
                dotted = _dotted(value.func)
                if dotted:
                    ctor[t.id] = dotted
    # also count `global x` rebinding inside functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                assigned[name] = assigned.get(name, 0) + 2
    return {k: v for k, v in ctor.items() if assigned.get(k) == 1}


@dataclass
class _Frame:
    caller: str
    caller_name: str
    kind: str  # "module" | "class" | "function"
    self_names: dict[str, str] = field(default_factory=dict)  # 'self'/'cls' -> class qualname
    current_class: str | None = None  # for super()


def _dotted(expr: ast.expr) -> str | None:
    """``a.b.c`` for pure Name/Attribute chains, else None."""
    parts: list[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _CallVisitor:
    def __init__(self, pf: ParsedFile, table: RepoSymbolTable) -> None:
        self.pf = pf
        self.table = table
        self.module = pf.metadata.module
        self.ctx = table.modules[self.module]
        self.edges: list[CallEdge] = []
        self._stack: list[_Frame] = [
            _Frame(caller=f"{self.module}.{MODULE_SCOPE}", caller_name=MODULE_SCOPE, kind="module")
        ]
        # qualified names of nested defs per enclosing function
        self._children: dict[str, dict[str, str]] = {}
        for f in pf.functions:
            if f.parent is not None:
                self._children.setdefault(f.parent, {})[f.name] = f.qualified_name
        for c in pf.classes:
            if c.parent is not None:
                self._children.setdefault(c.parent, {})[c.name] = c.qualified_name
        self._params: dict[str, set[str]] = {}
        for f in pf.functions:
            self._params.setdefault(f.qualified_name, set()).update(f.parameters)

    # ---- traversal ----
    def run(self, tree: ast.Module) -> list[CallEdge]:
        for stmt in tree.body:
            self._visit(stmt)
        return self.edges

    def _visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(node)
        elif isinstance(node, ast.ClassDef):
            self._visit_class(node)
        else:
            if isinstance(node, ast.Call):
                self._record(node)
            for child in ast.iter_child_nodes(node):
                self._visit(child)

    def _qualify(self, name: str) -> str:
        top = self._stack[-1]
        base = self.module if top.kind == "module" else top.caller
        return f"{base}.{name}"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # decorators, defaults and annotations are evaluated in the enclosing scope
        for expr in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self._visit(expr)
        all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for extra in (node.args.vararg, node.args.kwarg):
            if extra is not None:
                all_args.append(extra)
        for a in all_args:
            if a.annotation is not None:
                self._visit(a.annotation)
        if node.returns is not None:
            self._visit(node.returns)

        parent = self._stack[-1]
        qualified = self._qualify(node.name)
        self_names: dict[str, str] = {}
        current_class: str | None = None
        if parent.kind == "function":
            # nested function: closes over the enclosing method's self/cls,
            # unless one of its own parameters shadows the name
            self_names = {
                k: v for k, v in parent.self_names.items() if k not in {a.arg for a in all_args}
            }
            current_class = parent.current_class
        elif parent.kind == "class":
            current_class = parent.caller
            decorators = {_dotted(d) for d in node.decorator_list}
            positional = [*node.args.posonlyargs, *node.args.args]
            if positional and "staticmethod" not in decorators:
                self_names = {positional[0].arg: parent.caller}

        self._stack.append(
            _Frame(qualified, node.name, "function", self_names, current_class=current_class)
        )
        for stmt in node.body:
            self._visit(stmt)
        self._stack.pop()

    def _visit_class(self, node: ast.ClassDef) -> None:
        for expr in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
            self._visit(expr)
        qualified = self._qualify(node.name)
        self._stack.append(_Frame(qualified, node.name, "class", current_class=qualified))
        for stmt in node.body:
            self._visit(stmt)
        self._stack.pop()

    # ---- resolution ----
    def _lookup_name(self, name: str) -> tuple[str | None, CallResolution | str]:
        # enclosing function scopes, innermost first (class scopes are skipped, as in Python)
        for frame in reversed(self._stack):
            if frame.kind != "function":
                continue
            local = self._children.get(frame.caller, {}).get(name)
            if local:
                return local, CallResolution.LOCAL
            bound = self.ctx.scoped_bindings.get(frame.caller, {}).get(name)
            if bound:
                return self._classify_import(bound)
        if name in self.ctx.top_level:
            return self.ctx.top_level[name], CallResolution.LOCAL
        if name in self.ctx.module_bindings:
            return self._classify_import(self.ctx.module_bindings[name])
        if name in _BUILTINS:
            return f"builtins.{name}", CallResolution.BUILTIN
        return None, CallResolution.UNRESOLVED

    def _shadowed(self, name: str) -> bool:
        """True if ``name`` is a parameter or local import in an enclosing function."""
        for frame in self._stack:
            if frame.kind == "function" and name in self._params.get(frame.caller, set()):
                return True
            if name in self.ctx.scoped_bindings.get(frame.caller, {}):
                return True
        return False

    def _classify_import(self, target: str) -> tuple[str | None, CallResolution | str]:
        final = self.table.follow(target)
        if self.table.is_symbol(final):
            return final, CallResolution.IMPORTED
        if self.table.is_repo_module(final):
            return final, _MODULE
        if self.table.in_repo_namespace(final):
            return None, CallResolution.UNRESOLVED  # e.g. a re-exported variable
        return final, CallResolution.EXTERNAL

    def _resolve(self, func: ast.expr) -> tuple[str, str, str | None, CallResolution]:
        short, expr_text, qualified, res = self._resolve_raw(func)
        if res == _MODULE:  # calling a module object directly is not a resolvable call
            return short, expr_text, None, CallResolution.UNRESOLVED
        return short, expr_text, qualified, CallResolution(res)

    def _resolve_raw(self, func: ast.expr) -> tuple[str, str, str | None, CallResolution | str]:
        """Return (callee short name, callee_expr, callee_qualified, resolution)."""
        expr_text = ast.unparse(func)

        # super().method()
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            cls_q = self._stack[-1].current_class
            cls = self.table.classes.get(cls_q) if cls_q else None
            if cls is not None:
                for base in self.table.base_classes(cls):
                    hit = self.table.lookup_method(base, func.attr)
                    if hit:
                        return func.attr, expr_text, hit, CallResolution.METHOD
            return func.attr, expr_text, None, CallResolution.UNRESOLVED

        dotted = _dotted(func)
        if dotted is None:
            # e.g. factory()(), handlers[k](), (lambda: 1)(): no static name to report
            short = func.attr if isinstance(func, ast.Attribute) else DYNAMIC_CALLEE
            return short, expr_text, None, CallResolution.UNRESOLVED

        parts = dotted.split(".")
        short = parts[-1]
        if len(parts) == 1:
            q, res = self._lookup_name(short)
            return short, expr_text, q, res

        head, rest = parts[0], parts[1:]
        # self.method() / cls.method()
        self_cls = self._stack[-1].self_names.get(head)
        if self_cls is not None:
            if len(rest) == 1:
                hit = self.table.lookup_method(self_cls, rest[0])
                if hit:
                    return short, expr_text, hit, CallResolution.METHOD
            return short, expr_text, None, CallResolution.UNRESOLVED

        inst_cls = None if self._shadowed(head) else self.table.instance_class(self.ctx, head)
        if inst_cls is not None:
            if len(rest) == 1:
                hit = self.table.lookup_method(inst_cls, rest[0])
                if hit:
                    return short, expr_text, hit, CallResolution.METHOD
            return short, expr_text, None, CallResolution.UNRESOLVED

        base_q, base_res = self._lookup_name(head)
        if base_q is None or base_res in (CallResolution.BUILTIN, CallResolution.UNRESOLVED):
            return short, expr_text, None, CallResolution.UNRESOLVED
        full = ".".join([base_q, *rest])
        if base_res == CallResolution.EXTERNAL:
            return short, expr_text, full, CallResolution.EXTERNAL
        final = self.table.follow(full)
        if self.table.is_symbol(final):
            res = CallResolution.MODULE_ATTRIBUTE if base_res == _MODULE else CallResolution.METHOD
            return short, expr_text, final, res
        if not self.table.in_repo_namespace(final):  # e.g. compat.urlparse -> urllib.parse
            return short, expr_text, final, CallResolution.EXTERNAL
        return short, expr_text, None, CallResolution.UNRESOLVED

    def _record(self, node: ast.Call) -> None:
        frame = self._stack[-1]
        short, expr_text, qualified, resolution = self._resolve(node.func)
        self.edges.append(
            CallEdge(
                caller=frame.caller,
                caller_name=frame.caller_name,
                callee=short,
                callee_expr=expr_text,
                callee_qualified=qualified,
                resolution=resolution,
                file=self.pf.metadata.path,
                line=node.lineno,
                col=node.col_offset,
            )
        )


def edge_sort_key(e: CallEdge) -> tuple:
    return (e.file, e.line, e.col, e.caller, e.callee_expr)


def build_call_graph(parsed_files: list[ParsedFile]) -> tuple[list[CallEdge], list[IndexIssue]]:
    """Extract call edges for every successfully parsed file."""
    table = RepoSymbolTable(parsed_files)
    edges: list[CallEdge] = []
    issues: list[IndexIssue] = []
    for pf in parsed_files:
        if not pf.ok or pf.tree is None:
            continue
        try:
            edges.extend(_CallVisitor(pf, table).run(pf.tree))
        except RecursionError as exc:
            issues.append(
                IndexIssue(
                    file=pf.metadata.path, stage="callgraph",
                    error_type=type(exc).__name__, message=str(exc),
                )
            )  # fmt: skip
    edges.sort(key=edge_sort_key)
    return edges, issues
