"""Deterministic checks against the successfully indexed Python source."""

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from repoguard.schemas import CallResolution, ClassMetadata, FunctionMetadata, IndexArtifacts
from repoguard.verification.models import (
    Claim,
    ClaimEvidence,
    ClaimType,
    ClaimVerdict,
    VerdictStatus,
    VerificationResult,
)

IndexedSymbol = ClassMetadata | FunctionMetadata


def _find_symbols(
    index: IndexArtifacts,
    name: str | None,
    file: str | None = None,
    symbol_type: type[IndexedSymbol] | None = None,
) -> list[IndexedSymbol]:
    parsed_files = {record.path for record in index.files if record.parse_status == "ok"}
    return [
        symbol
        for symbol in index.symbols
        if symbol.file in parsed_files
        and (file is None or symbol.file == file)
        and (symbol_type is None or isinstance(symbol, symbol_type))
        and name
        in (
            symbol.name,
            symbol.qualified_name,
            symbol.qualified_name.removeprefix(f"{symbol.module}."),
        )
    ]


def _definition_missing(claim: Claim, index: IndexArtifacts) -> ClaimVerdict:
    # A check scoped to a specific file reaches here only if that file parsed.
    if claim.file is not None or (
        index.files
        and index.repository.status == "ready"
        and not index.errors
        and all(record.parse_status == "ok" for record in index.files)
    ):
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.REFUTED,
            reason="No matching definition in the indexed source.",
        )
    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.UNVERIFIABLE,
        reason="No matching definition in the available index, which is incomplete or empty.",
    )


def _check_definition(claim: Claim, index: IndexArtifacts) -> ClaimVerdict:
    symbol_type = ClassMetadata if claim.type == ClaimType.CLASS_EXISTS else FunctionMetadata
    matches = _find_symbols(index, claim.symbol, claim.file, symbol_type)
    if not matches:
        return _definition_missing(claim, index)

    if claim.type == ClaimType.PARAMETER_EXISTS:
        if len(matches) != 1:
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.UNVERIFIABLE,
                reason="The parameter claim matches multiple definitions. Qualify the symbol.",
            )
        function = matches[0]
        if isinstance(function, FunctionMetadata) and claim.parameter not in function.parameters:
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.REFUTED,
                reason="The indexed signature does not declare that parameter.",
            )

    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.VERIFIED,
        reason="The indexed definition supports the claim.",
        evidence=[
            ClaimEvidence(
                file=symbol.file,
                line=symbol.line,
                end_line=symbol.end_line,
                qualified_name=symbol.qualified_name,
            )
            for symbol in matches
        ],
    )


def _check_call(claim: Claim, index: IndexArtifacts) -> ClaimVerdict:
    callers = _find_symbols(index, claim.caller, claim.file, FunctionMetadata)
    if not callers:
        return _definition_missing(claim, index)

    callees = _find_symbols(index, claim.callee)
    if len(callers) != 1 or len(callees) > 1:
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.UNVERIFIABLE,
            reason="The caller or callee matches multiple definitions. Use a qualified name.",
        )

    caller = callers[0]
    edges = [
        edge
        for edge in index.callees_of(caller.qualified_name)
        if edge.caller == caller.qualified_name and edge.file == caller.file
    ]
    target = callees[0].qualified_name if callees else claim.callee
    assert target is not None
    short_target = target.rsplit(".", 1)[-1]
    is_short_name = target == short_target
    matching_edges = [
        edge
        for edge in edges
        if edge.resolution != CallResolution.UNRESOLVED
        and edge.callee_qualified is not None
        and (
            edge.callee_qualified == target
            or (is_short_name and edge.callee_qualified.rsplit(".", 1)[-1] == target)
        )
    ]
    if len({edge.callee_qualified for edge in matching_edges}) > 1:
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.UNVERIFIABLE,
            reason="The name could refer to multiple resolved call targets.",
        )
    if matching_edges:
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.VERIFIED,
            reason="The call graph contains a resolved edge in the claimed direction.",
            evidence=[
                ClaimEvidence(file=edge.file, line=edge.line, qualified_name=edge.callee_qualified)
                for edge in matching_edges
            ],
        )
    if any(
        edge.resolution == CallResolution.UNRESOLVED and edge.callee in (short_target, "<dynamic>")
        for edge in edges
    ):
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.UNVERIFIABLE,
            reason="A possible call site has an unresolved target.",
        )
    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.REFUTED,
        reason="The indexed static call graph has no matching edge from the caller.",
    )


def _check_import(claim: Claim, index: IndexArtifacts) -> ClaimVerdict:
    evidence = []
    for statement in index.imports:
        if statement.file != claim.file:
            continue
        modules = {value for value in (statement.module, statement.resolved_module) if value}
        imported = modules | {f"{module}.{statement.name}" for module in modules if statement.name}
        if claim.module in imported:
            evidence.append(ClaimEvidence(file=statement.file, line=statement.line))
    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.VERIFIED if evidence else VerdictStatus.REFUTED,
        reason=(
            "An explicit import matches the claimed name."
            if evidence
            else "The indexed file has no matching explicit import."
        ),
        evidence=evidence,
    )


def _verify_claim(claim: Claim, index: IndexArtifacts) -> ClaimVerdict:
    if claim.file is not None:
        file = next((record for record in index.files if record.path == claim.file), None)
        if file is None or file.parse_status != "ok":
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.UNVERIFIABLE,
                reason="The claimed file has no successfully parsed entry in the index.",
            )
        if claim.type == ClaimType.FILE_EXISTS:
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.VERIFIED,
                reason="The file is present in the Python index.",
                evidence=[ClaimEvidence(file=file.path)],
            )
    if claim.type == ClaimType.CALLS:
        return _check_call(claim, index)
    if claim.type == ClaimType.IMPORTS:
        return _check_import(claim, index)
    return _check_definition(claim, index)


def _unsupported_name(verdict: ClaimVerdict) -> str | None:
    claim = verdict.claim
    if claim is None or verdict.status == VerdictStatus.VERIFIED:
        return None
    if verdict.status == VerdictStatus.UNVERIFIABLE and claim.file is not None:
        return claim.file
    return claim.parameter or claim.callee or claim.module or claim.symbol or claim.file


def verify(
    claims: Sequence[Claim | Mapping[str, object]], index: IndexArtifacts
) -> VerificationResult:
    """Evaluate claims in order. Invalid input gets an explicit unverifiable verdict."""
    verdicts = []
    for supplied in claims:
        try:
            claim = Claim.model_validate(supplied)
        except ValidationError:
            verdicts.append(
                ClaimVerdict(
                    claim=None,
                    status=VerdictStatus.UNVERIFIABLE,
                    reason="Malformed or unsupported claim.",
                )
            )
        else:
            verdicts.append(_verify_claim(claim, index))

    verified_count = sum(item.status == VerdictStatus.VERIFIED for item in verdicts)
    if verified_count == 0:
        overall = "unsupported"
    elif verified_count == len(verdicts):
        overall = "verified"
    else:
        overall = "partially_verified"

    names = dict.fromkeys(
        name for item in verdicts if (name := _unsupported_name(item)) is not None
    )
    return VerificationResult(verdict=overall, claims=verdicts, unsupported_names=list(names))
