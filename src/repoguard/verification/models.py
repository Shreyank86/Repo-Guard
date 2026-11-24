"""Claim and verdict contracts for RepoGuard's source verifier."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ClaimType(str, Enum):
    FILE_EXISTS = "file_exists"
    FUNCTION_EXISTS = "function_exists"
    CLASS_EXISTS = "class_exists"
    SYMBOL_EXISTS = "symbol_exists"
    PARAMETER_EXISTS = "parameter_exists"
    CALLS = "calls"
    IMPORTS = "imports"
    SEMANTIC = "semantic"


class VerdictStatus(str, Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


class ReasonCode(str, Enum):
    """Stable verdict details for audit, integration, and reward consumers."""

    FILE_INDEXED = "file_indexed"
    DEFINITION_FOUND = "definition_found"
    PARAMETER_DECLARED = "parameter_declared"
    CALL_RESOLVED = "call_resolved"
    IMPORT_FOUND = "import_found"
    DEFINITION_ABSENT = "definition_absent"
    PARAMETER_ABSENT = "parameter_absent"
    CALL_ABSENT = "call_absent"
    IMPORT_ABSENT = "import_absent"
    INVALID_CLAIM = "invalid_claim"
    INVALID_PATH = "invalid_path"
    FILE_UNINDEXED = "file_unindexed"
    FILE_PARSE_ERROR = "file_parse_error"
    INCOMPLETE_INDEX = "incomplete_index"
    AMBIGUOUS_SYMBOL = "ambiguous_symbol"
    UNRESOLVED_CALL = "unresolved_call"
    SEMANTIC_CLAIM = "semantic_claim"


_REQUIRED_FIELDS = {
    ClaimType.FILE_EXISTS: ("file",),
    ClaimType.FUNCTION_EXISTS: ("symbol",),
    ClaimType.CLASS_EXISTS: ("symbol",),
    ClaimType.SYMBOL_EXISTS: ("symbol",),
    ClaimType.PARAMETER_EXISTS: ("symbol", "parameter"),
    ClaimType.CALLS: ("caller", "callee"),
    ClaimType.IMPORTS: ("file", "module"),
    ClaimType.SEMANTIC: ("statement",),
}


class Claim(BaseModel):
    """A structural assertion or an explicitly marked semantic statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ClaimType = ClaimType.CLASS_EXISTS
    symbol: NonEmptyText | None = None
    file: NonEmptyText | None = None
    parameter: NonEmptyText | None = None
    caller: NonEmptyText | None = None
    callee: NonEmptyText | None = None
    module: NonEmptyText | None = None
    statement: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "Claim":
        required = _REQUIRED_FIELDS[self.type]
        populated = {name for name in type(self).model_fields if getattr(self, name) is not None}
        missing = set(required) - populated
        extra = populated - {"type", "file", *required}
        if missing or extra:
            raise ValueError(
                f"{self.type.value} has missing fields {sorted(missing)} "
                f"and unexpected fields {sorted(extra)}"
            )
        return self


class ClaimEvidence(BaseModel):
    file: str
    line: int | None = None
    end_line: int | None = None
    qualified_name: str | None = None


class ClaimVerdict(BaseModel):
    claim: Claim | None
    status: VerdictStatus
    reason: str
    reason_code: ReasonCode | None = None
    """Absent or null when loading a verdict produced before Phase 2."""
    evidence: list[ClaimEvidence] = Field(default_factory=list)


class VerificationResult(BaseModel):
    verdict: Literal["verified", "partially_verified", "unsupported"]
    claims: list[ClaimVerdict]
    unsupported_names: list[str]
