"""Turn parsed files into retrieval passages.

Passage kinds (one file -> many passages):

* ``function`` / ``method`` — full source of every non-nested function or method,
  decorators included. Nested functions stay inside their parent's passage.
* ``class`` — class *overview*: header, docstring, class attributes and method
  signatures; method bodies are replaced by an indented ``...`` line because they
  already have their own passages.
* ``module`` — module *overview*: docstring, imports, top-level statements and the
  signatures of top-level functions/classes (bodies elided).

Passages longer than ``max_lines`` are split into overlapping windows
(``part``/``total_parts``; ids get an ``@<part>`` suffix).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from repoguard.repository.parser import ParsedFile
from repoguard.schemas import MODULE_SCOPE, ClassMetadata, FunctionMetadata, Passage

_ELLIPSIS = "..."


@dataclass(frozen=True)
class PassageConfig:
    max_lines: int = 80
    overlap_lines: int = 10
    include_module_passages: bool = True
    include_class_passages: bool = True

    def __post_init__(self) -> None:
        if self.max_lines < 1:
            raise ValueError("max_lines must be >= 1")
        if not 0 <= self.overlap_lines < self.max_lines:
            raise ValueError("overlap_lines must be in [0, max_lines)")


def format_signature(fn: FunctionMetadata) -> str:
    parts: list[str] = []
    seen_kwonly_marker = False
    details = fn.parameter_details
    for i, p in enumerate(details):
        kind = p.kind.value
        if kind == "keyword_only" and not seen_kwonly_marker:
            if not any(d.kind.value == "var_positional" for d in details[:i]):
                parts.append("*")
            seen_kwonly_marker = True
        prefix = "*" if kind == "var_positional" else "**" if kind == "var_keyword" else ""
        text = prefix + p.name
        if p.annotation:
            text += f": {p.annotation}"
        if p.has_default:
            text += f" = {p.default}" if p.annotation else f"={p.default}"
        parts.append(text)
        is_last_posonly = kind == "positional_only" and (
            i + 1 == len(details) or details[i + 1].kind.value != "positional_only"
        )
        if is_last_posonly:
            parts.append("/")
    sig = f"({', '.join(parts)})"
    if fn.return_annotation:
        sig += f" -> {fn.return_annotation}"
    return sig


def _def_nodes(tree: ast.Module) -> dict[tuple[int, int], ast.AST]:
    out: dict[tuple[int, int], ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[(node.lineno, node.col_offset)] = node
    return out


def _elide(lines: list[str], start: int, end: int, bodies: list[tuple[int, int, int]]) -> str:
    """Lines ``start..end`` (1-based, inclusive) with each body range replaced by '...'.

    ``bodies`` holds ``(body_start, body_end, indent_col)`` triples.
    """
    out: list[str] = []
    skip_until = 0
    by_start = {b[0]: b for b in bodies}
    for ln in range(start, end + 1):
        if ln <= skip_until:
            continue
        if ln in by_start:
            b_start, b_end, indent = by_start[ln]
            out.append(" " * indent + _ELLIPSIS)
            skip_until = b_end
            continue
        out.append(lines[ln - 1])
    return "\n".join(out)


def _body_range(node: ast.AST) -> tuple[int, int, int] | None:
    """Body line span of a def/class whose body starts on a later line than the header."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if first.lineno <= node.lineno:  # one-liner such as `def f(): pass`
        return None
    end = node.end_lineno or first.lineno
    return first.lineno, end, first.col_offset


def _split(
    base_id: str, lines: list[str], start_line: int, cfg: PassageConfig
) -> list[tuple[str, int, int, str, int, int]]:
    """Split contiguous source lines into windows. Returns (id, start, end, text, part, total)."""
    n = len(lines)
    if n <= cfg.max_lines:
        return [(base_id, start_line, start_line + n - 1, "\n".join(lines), 1, 1)]
    step = cfg.max_lines - cfg.overlap_lines
    windows = []
    offset = 0
    while True:
        chunk = lines[offset : offset + cfg.max_lines]
        windows.append((offset, chunk))
        if offset + cfg.max_lines >= n:
            break
        offset += step
    total = len(windows)
    return [
        (
            f"{base_id}@{i}",
            start_line + off,
            start_line + off + len(chunk) - 1,
            "\n".join(chunk),
            i,
            total,
        )
        for i, (off, chunk) in enumerate(windows, start=1)
    ]


def build_passages(pf: ParsedFile, config: PassageConfig | None = None) -> list[Passage]:
    cfg = config or PassageConfig()
    if not pf.ok or pf.tree is None:
        return []
    lines = pf.source.splitlines()
    path = pf.metadata.path
    nodes = _def_nodes(pf.tree)
    passages: list[Passage] = []

    # -- functions and methods (non-nested), full source --
    for fn in pf.functions:
        if fn.is_nested:
            continue
        start = fn.decorator_line or fn.line
        sig = format_signature(fn)
        prefix = "async def" if fn.is_async else "def"
        header = f"# {path} | {fn.type} {fn.qualified_name} | {prefix} {fn.name}{sig}"
        for pid, s, e, text, part, total in _split(
            fn.symbol_id, lines[start - 1 : fn.end_line], start, cfg
        ):
            passages.append(
                Passage(
                    passage_id=pid, file_path=path, start_line=s, end_line=e,
                    passage_type=fn.type, symbol=fn.qualified_name, symbol_id=fn.symbol_id,
                    part=part, total_parts=total, header=f"{header} | lines {s}-{e}", content=text,
                )
            )  # fmt: skip

    # -- class overviews --
    if cfg.include_class_passages:
        for cls in pf.classes:
            if _inside_function(cls, pf):
                continue
            passages.extend(_class_passages(cls, lines, nodes, path, cfg))

    # -- module overview --
    if cfg.include_module_passages and lines:
        bodies = []
        for node in pf.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rng = _body_range(node)
                if rng:
                    bodies.append(rng)
        text = _elide(lines, 1, len(lines), bodies)
        if text.strip():
            base = f"{path}::{MODULE_SCOPE}"
            header = f"# {path} | module {pf.metadata.module}"
            # elided text is not contiguous source, so split on its own lines
            elided_lines = text.split("\n")
            for pid, _s, _e, chunk, part, total in _split(base, elided_lines, 1, cfg):
                passages.append(
                    Passage(
                        passage_id=pid, file_path=path, start_line=1, end_line=len(lines),
                        passage_type="module", symbol=None, symbol_id=None,
                        part=part, total_parts=total, header=header, content=chunk,
                    )
                )  # fmt: skip
    return passages


def _inside_function(cls: ClassMetadata, pf: ParsedFile) -> bool:
    fn_names = {f.qualified_name for f in pf.functions}
    parent = cls.parent
    while parent is not None:
        if parent in fn_names:
            return True
        parent = parent.rsplit(".", 1)[0] if "." in parent else None
    return False


def _class_passages(
    cls: ClassMetadata,
    lines: list[str],
    nodes: dict[tuple[int, int], ast.AST],
    path: str,
    cfg: PassageConfig,
) -> list[Passage]:
    node = nodes.get((cls.line, cls.col))
    start = cls.decorator_line or cls.line
    bodies = []
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rng = _body_range(child)
                if rng:
                    bodies.append(rng)
    text = _elide(lines, start, cls.end_line, bodies)
    bases = f"({', '.join(cls.bases)})" if cls.bases else ""
    header = (
        f"# {path} | class {cls.qualified_name}{bases} | methods: {', '.join(cls.methods) or '-'}"
    )
    out = []
    for pid, _s, _e, chunk, part, total in _split(f"{cls.symbol_id}", text.split("\n"), start, cfg):
        out.append(
            Passage(
                passage_id=pid, file_path=path, start_line=start, end_line=cls.end_line,
                passage_type="class", symbol=cls.qualified_name, symbol_id=cls.symbol_id,
                part=part, total_parts=total, header=header, content=chunk,
            )
        )  # fmt: skip
    return out
