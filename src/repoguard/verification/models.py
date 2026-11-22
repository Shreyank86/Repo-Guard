"""Draft Phase 1 contracts for structural verification."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ClaimType(str, Enum):
    FILE_EXISTS = "file_exists"
    FUNCTION_EXISTS = "function_exists"
    CLASS_EXISTS = "class_exists"
    PARAMETER_EXISTS = "parameter_exists"
    CALLS = "calls"
    IMPORTS = "imports"


class VerdictStatus(str, Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


_REQUIRED_FIELDS = {
    ClaimType.FILE_EXISTS: ("file",),
    ClaimType.FUNCTION_EXISTS: ("symbol",),
    ClaimType.CLASS_EXISTS: ("symbol",),
    ClaimType.PARAMETER_EXISTS: ("symbol", "parameter"),
    ClaimType.CALLS: ("caller", "callee"),
    ClaimType.IMPORTS: ("file", "module"),
}


class Claim(BaseModel):
    """A positive assertion about indexed source, optionally scoped to a file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ClaimType = ClaimType.CLASS_EXISTS
    symbol: NonEmptyText | None = None
    file: NonEmptyText | None = None
    parameter: NonEmptyText | None = None
    caller: NonEmptyText | None = None
    callee: NonEmptyText | None = None
    module: NonEmptyText | None = None

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
    evidence: list[ClaimEvidence] = Field(default_factory=list)


class VerificationResult(BaseModel):
    verdict: Literal["verified", "partially_verified", "unsupported"]
    claims: list[ClaimVerdict]
    unsupported_names: list[str]
