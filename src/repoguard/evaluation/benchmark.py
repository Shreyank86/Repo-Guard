"""Versioned benchmark rows and checks for pinned repository splits."""

from collections.abc import Iterable
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from repoguard.repository import load_index
from repoguard.schemas import FileMetadata, IndexArtifacts
from repoguard.verification import Claim, ClaimType, VerdictStatus, verify

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
RepositoryId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
Split = Literal["train", "validation", "test"]
QuestionType = Literal[
    "file",
    "definition",
    "parameter",
    "call",
    "import",
    "multi_hop",
    "near_miss",
    "wrong_path",
    "wrong_parameter",
    "ambiguous",
    "out_of_scope",
]


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: Text
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: Text | None = None

    @model_validator(mode="after")
    def check_range(self) -> "SourceReference":
        if self.start_line > self.end_line:
            raise ValueError("Source ranges must have start_line <= end_line.")
        if (
            "\\" in self.file
            or "\x00" in self.file
            or PureWindowsPath(self.file).drive
            or {"", ".", ".."}.intersection(self.file.split("/"))
        ):
            raise ValueError("Source references require canonical repository-relative paths.")
        return self

    @property
    def unit(self) -> str:
        return f"{self.file}::{self.symbol or '<module>'}"


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_version: Literal["1.0"]
    case_id: Text
    repository_id: RepositoryId
    repository_url: Text
    commit: Commit
    licence: Text
    split: Split
    question_type: QuestionType
    question: Text
    answerable: bool
    expected_facts: list[Claim]
    evidence: list[SourceReference]
    reference_answer: Text

    @model_validator(mode="after")
    def check_label(self) -> "BenchmarkCase":
        if self.answerable and not (self.expected_facts and self.evidence):
            raise ValueError("Answerable cases require facts and source evidence.")
        if not self.answerable and self.expected_facts:
            raise ValueError("Abstention cases cannot require affirmative facts.")
        if self.question_type == "multi_hop" and len(self.expected_facts) < 2:
            raise ValueError("Multi-hop cases require at least two facts.")
        if any(claim.type == ClaimType.SEMANTIC for claim in self.expected_facts):
            raise ValueError("Benchmark v1 scores structural facts only.")
        if len(set(self.expected_facts)) != len(self.expected_facts):
            raise ValueError("Expected facts must be distinct.")
        return self


_Row = TypeVar("_Row", bound=BaseModel)


def read_jsonl(path: Path, model: type[_Row]) -> list[_Row]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contains no records.")
    return rows


def _repository_key(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.path.strip("/"):
        raise ValueError(f"Expected a public HTTPS repository URL, got {url!r}.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"Use a canonical repository URL, got {url!r}.")
    return (parsed.hostname + parsed.path.rstrip("/").removesuffix(".git")).casefold()


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    cases = read_jsonl(path, BenchmarkCase)
    seen_ids = set()
    seen_questions = set()
    by_repo: dict[str, tuple[str, str, str, Split]] = {}
    by_url: dict[str, str] = {}
    by_commit: dict[str, str] = {}

    for case in cases:
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case ID {case.case_id!r}.")
        seen_ids.add(case.case_id)

        question = " ".join(case.question.casefold().replace("`", "").split()).rstrip("?.!")
        if question in seen_questions:
            raise ValueError(f"Duplicate benchmark question at {case.case_id!r}.")
        seen_questions.add(question)

        url = _repository_key(case.repository_url)
        snapshot = (url, case.commit, case.licence, case.split)
        if by_repo.setdefault(case.repository_id, snapshot) != snapshot:
            raise ValueError(f"{case.repository_id!r} spans different snapshots or splits.")
        if by_url.setdefault(url, case.repository_id) != case.repository_id:
            raise ValueError(f"Repository {case.repository_url!r} has multiple IDs.")
        if by_commit.setdefault(case.commit, case.repository_id) != case.repository_id:
            raise ValueError(f"Commit {case.commit} appears under multiple repository IDs.")

    if {case.split for case in cases} != {"train", "validation", "test"}:
        raise ValueError(
            "The benchmark requires train, validation, and held-out test repositories."
        )
    return cases


def validate_indexes(cases: Iterable[BenchmarkCase], directory: Path) -> dict[str, IndexArtifacts]:
    """Check exact snapshots, evidence ranges, and positive labels against the index."""
    indexes: dict[str, IndexArtifacts] = {}
    indexed_files: dict[str, dict[str, FileMetadata]] = {}
    retrieval_units: dict[str, set[str]] = {}
    for case in cases:
        name = case.repository_id
        if name not in indexes:
            index = load_index(directory / name)
            indexes[name] = index
            indexed_files[name] = {record.path: record for record in index.files}
            retrieval_units[name] = {p.passage_id.rsplit("@", 1)[0] for p in index.passages}
        index = indexes[name]
        repository = index.repository
        if (
            repository.repository_id != name
            or repository.commit != case.commit
            or repository.licence != case.licence
            or not repository.source_url
            or _repository_key(repository.source_url) != _repository_key(case.repository_url)
        ):
            raise ValueError(f"Index snapshot does not match {case.case_id}.")

        for reference in case.evidence:
            indexed = indexed_files[name].get(reference.file)
            if (
                indexed is None
                or indexed.parse_status != "ok"
                or reference.end_line > indexed.line_count
            ):
                raise ValueError(f"Invalid source range for {case.case_id}: {reference}.")
            if reference.symbol is not None and not index.find_symbols(
                reference.symbol, file=reference.file
            ):
                raise ValueError(f"Unknown evidence symbol in {case.case_id}: {reference.symbol}.")
            if case.answerable and reference.unit not in retrieval_units[name]:
                raise ValueError(f"No retrieval passage for {case.case_id}: {reference.unit}.")

        for fact, result in zip(
            case.expected_facts, verify(case.expected_facts, index).claims, strict=True
        ):
            if result.status != VerdictStatus.VERIFIED:
                raise ValueError(
                    f"Gold fact for {case.case_id} disagrees with the index: {result}."
                )
            if fact.symbol is not None and {item.qualified_name for item in result.evidence} != {
                fact.symbol
            }:
                raise ValueError(f"Use a qualified gold symbol in {case.case_id}: {fact.symbol}.")
            if fact.type == ClaimType.CALLS:
                assert fact.callee is not None
                if not any(
                    edge.caller == fact.caller and edge.callee_qualified == fact.callee
                    for edge in index.callers_of(fact.callee)
                ):
                    raise ValueError(f"Use a qualified gold caller and callee in {case.case_id}.")
            if not any(
                reference.file == evidence.file
                and (
                    evidence.line is None
                    or reference.start_line <= evidence.line <= reference.end_line
                )
                for reference in case.evidence
                for evidence in result.evidence
            ):
                raise ValueError(f"No source range supports a gold fact for {case.case_id}.")
    return indexes
