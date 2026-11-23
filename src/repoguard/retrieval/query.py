"""Parse code references (symbols, files) out of a natural-language question."""

from __future__ import annotations

import re

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DOTTED = rf"{_IDENT}(?:\.{_IDENT})*"
_BACKTICK = re.compile(r"`([^`\n]{1,200})`")
_FILE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.py)\b")
_CALLISH = re.compile(rf"({_DOTTED})\s*\(\s*\)")
_NOUNS = r"function|method|class|def|decorator|attribute|parameter|argument|variable|module|symbol"
_NOUN_THEN_NAME = re.compile(
    rf"\b(?:{_NOUNS})s?\s+({_DOTTED})", re.IGNORECASE
)  # "function load_config"
_NAME_THEN_NOUN = re.compile(
    rf"\b({_DOTTED})\s+(?:function|method|class|decorator|module)\b"
)  # "Job class"
_WORD = re.compile(_DOTTED)

# Words that look like identifiers but are ordinary English in questions.
_ENGLISH = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "has", "have", "had", "of", "to", "in", "on", "at", "for", "from", "by", "with", "as", "and", "or", "not", "no", "yes", "it", "its", "this", "that", "these", "those", "there", "here", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "can", "could", "should", "would", "will", "may", "might", "must", "shall", "any", "all", "each", "every", "some", "such", "than", "then", "so", "if", "else", "into", "onto", "out", "up", "down", "over", "under", "again", "further", "once", "only", "own", "same", "too", "very", "just", "also", "about", "across", "after", "before", "between", "through", "during", "without", "within", "function", "functions", "method", "methods", "class", "classes", "def", "decorator", "attribute", "parameter", "parameters", "argument", "arguments", "variable", "variables", "module", "modules", "symbol", "symbols", "file", "files", "code", "repository", "repo", "project", "package", "defined", "define", "defines", "exist", "exists", "call", "calls", "called", "invoke", "invokes", "return", "returns", "accept", "accepts", "take", "takes", "use", "uses", "used", "import", "imports", "work", "works"]
)


def extract_named_symbols(question: str) -> list[str]:
    """Code identifiers a question refers to, in order of appearance (deduplicated).

    Sources: `backticked` text (always kept, even if it is an English word), ``name()`` calls, ``function/method/class <name>`` phrases,
    and identifier-shaped words (snake_case, camelCase, dotted ``a.b``). File paths are
    excluded (see :func:`extract_named_files`). Plain English words are never symbols.
    """
    found: list[str] = []

    def add(raw: str, explicit: bool = False) -> None:
        name = raw.strip().split("=")[0].strip().strip("()").strip()
        name = re.sub(r"\(.*\)$", "", name).strip()
        if not name or name.endswith(".py") or "/" in name or " " in name:
            return
        if not re.fullmatch(_DOTTED, name):
            return
        if not explicit and name.lower() in _ENGLISH and "." not in name:
            return
        if name not in found:
            found.append(name)

    backticked = _BACKTICK.findall(question)
    for b in backticked:
        add(b, explicit=True)
    plain = _BACKTICK.sub(" ", question)
    for m in _CALLISH.finditer(plain):
        add(m.group(1))
    if not backticked:  # explicit backticks win; phrase heuristics only for plain questions
        for m in _NOUN_THEN_NAME.finditer(plain):
            add(m.group(1))
        for m in _NAME_THEN_NOUN.finditer(plain):
            if looks_code(m.group(1)) or m.group(1)[:1].isupper():
                add(m.group(1))
    for m in _WORD.finditer(plain):
        w = m.group(0)
        if w.endswith(".py") or _FILE.fullmatch(w):
            continue
        if looks_code(w):
            add(w)
    return found


def looks_code(w: str) -> bool:
    return bool(
        "_" in w.strip("_") or "." in w or re.search(r"[a-z][A-Z]", w) or w.startswith("__")
    )


def extract_named_files(question: str) -> list[str]:
    out: list[str] = []
    for m in _FILE.finditer(question):
        f = m.group(1).lstrip("./")
        if f not in out:
            out.append(f)
    return out
