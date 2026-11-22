"""Extract positive structural assertions expressed in a small set of prose patterns."""

import re

from repoguard.verification.models import Claim, ClaimType

_IDENTIFIER = r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*"
_PATH = r"(?:[\w.-]+/)*[\w-][\w.-]*\.[A-Za-z0-9]+"
_PARAMETER = r"\*{0,2}[A-Za-z_]\w*"
_ROLE = r"(?:module-level\s+|async\s+)?(?:function\s+or\s+class|class\s+or\s+function|function|method|class)"
_START = r"(?:^|(?<=[.!])\s+)[ \t]*(?:[-*•]|\d+[.)])?[ \t]*(?:the[ \t]+)?"
_END = r"(?=[.!](?:\s|$)|[ \t]*$)"
_FLAGS = re.IGNORECASE | re.MULTILINE


def _token(field: str, value: str = _IDENTIFIER) -> str:
    return rf"(?P<{field}>`{value}`|{value})"


_SYMBOL = _token("symbol")
_FILE = _token("file", _PATH)
_PARAM = _token("parameter", _PARAMETER)
_CALLER = _token("caller")
_CALLEE = _token("callee")
_MODULE = _token("module")
_FILE_LOCATION = rf"(?:defined|declared)\s+in\s+(?:the\s+)?(?:file\s+)?{_FILE}(?:\s+file)?"

_DEFINITIONS = [
    re.compile(
        rf"{_START}(?P<role>{_ROLE})\s+{_SYMBOL}\s+(?:is\s+{_FILE_LOCATION}|exists){_END}", _FLAGS
    ),
    re.compile(
        rf"{_START}{_SYMBOL}\s+(?P<role>{_ROLE})\s+(?:is\s+{_FILE_LOCATION}|exists){_END}", _FLAGS
    ),
]

_EXPRESSIONS = [
    (ClaimType.FILE_EXISTS, rf"(?:file\s+)?{_FILE}\s+exists"),
    (ClaimType.IMPORTS, rf"(?:file\s+)?{_FILE}\s+imports\s+{_MODULE}"),
    (ClaimType.CALLS, rf"(?:{_ROLE}\s+)?{_CALLER}(?:\s+{_ROLE})?\s+calls\s+{_CALLEE}"),
    (ClaimType.CALLS, rf"{_CALLEE}\s+is\s+called\s+by\s+(?:{_ROLE}\s+)?{_CALLER}"),
    (ClaimType.CALLS, rf"{_ROLE}\s+that\s+calls\s+{_CALLEE}\s+is\s+{_CALLER}"),
    (
        ClaimType.PARAMETER_EXISTS,
        rf"(?:{_ROLE}\s+)?{_SYMBOL}(?:\s+{_ROLE})?\s+(?:accepts|takes|has)\s+"
        rf"(?:(?:the\s+)?(?:parameter|argument)\s+{_PARAM}"
        # Python regex group names must be unique within a pattern.
        rf"|(?:an?\s+)?(?:required\s+|optional\s+)?{_token('named_parameter', _PARAMETER)}"
        rf"\s+(?:parameter|argument))",
    ),
]
_PATTERNS = [
    (kind, re.compile(rf"{_START}{pattern}{_END}", _FLAGS)) for kind, pattern in _EXPRESSIONS
]

_PARAMETER_LIST = re.compile(
    rf"{_START}(?:{_ROLE}\s+)?{_SYMBOL}(?:\s+{_ROLE})?\s+(?:accepts|takes|has)\s+"
    r"(?:the\s+following\s+)?(?:parameters|arguments):[ \t]*\n"
    r"(?P<items>(?:[ \t]*\n|[ \t]*(?:[-*]|\d+[.)])[ \t]+[^\n]*\n?)+)",
    _FLAGS,
)
_LIST_ITEM = re.compile(rf"^[ \t]*(?:[-*]|\d+[.)])[ \t]+{_PARAM}(?=[ \t]*[:–—]|[ \t]*$)", _FLAGS)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_CONTEXT = re.compile(r"\bRepository Context\s*:", re.IGNORECASE)


def _answer_prose(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        marker = _FENCE.match(line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token.startswith(fence):
                fence = None
            lines.append("")
        elif fence is not None or line.lstrip().startswith(">"):
            lines.append("")
        elif context := _CONTEXT.search(line):
            lines.append(line[: context.start()])
            break
        else:
            lines.append(line)
    return "\n".join(lines)


def _claim_from_match(kind: ClaimType, match: re.Match[str]) -> Claim:
    fields = {
        name: token.strip("`").lstrip("*")
        if name in {"parameter", "named_parameter"}
        else token.strip("`")
        for name, token in match.groupdict().items()
        if token is not None and name != "role"
    }
    if "named_parameter" in fields:
        fields["parameter"] = fields.pop("named_parameter")
    return Claim(type=kind, **fields)


def extract_claims(text: str) -> list[Claim]:
    """Read explicit positive claims from prose. Unrecognized statements yield no claim."""
    prose = _answer_prose(text)
    positioned = []
    for definition in _DEFINITIONS:
        for match in definition.finditer(prose):
            kind = (
                ClaimType.CLASS_EXISTS
                if match["role"].lower() == "class"
                else ClaimType.FUNCTION_EXISTS
            )
            positioned.append((match.start(), _claim_from_match(kind, match)))
    for kind, pattern in _PATTERNS:
        positioned.extend(
            (match.start(), _claim_from_match(kind, match)) for match in pattern.finditer(prose)
        )
    for match in _PARAMETER_LIST.finditer(prose):
        for item in _LIST_ITEM.finditer(match["items"]):
            claim = Claim(
                type=ClaimType.PARAMETER_EXISTS,
                symbol=match["symbol"].strip("`"),
                parameter=item["parameter"].strip("`").lstrip("*"),
            )
            positioned.append((match.start("items") + item.start(), claim))
    positioned.sort(key=lambda item: item[0])
    return list(dict.fromkeys(claim for _, claim in positioned))
