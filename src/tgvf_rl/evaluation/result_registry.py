"""Typed, fail-closed registry for policy-evaluation evidence.

The registry deliberately separates an evidence record (which includes the
weights and score artifact) from its comparison contract.  A score may be
reported for its own declared contract without being a valid causal
comparator.  Version 2 has no producer receipt binding score bytes to the
evaluation identity, trajectory set, weights and comparison contract, so it
mechanically rejects ``golden`` status and every numeric delta.  Those claims
remain blocked until a later schema verifies that missing provenance chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

# Several imports below are retained solely for the historical facade
# namespace; runtime use is not the complete compatibility contract.
from .result_registry_schema import (  # noqa: F401
    ComparisonContract,
    ComparisonDefinition,
    CoreDevScore,
    Enum,
    Mapping,
    PreregistrationEvidence,
    PurePosixPath,
    ResultRecord,
    ResultStatus,
    ResultTable,
    ScoreArtifact,
    WeightIdentity,
    WeightState,
    _require_matching_score,
    _validate_comparison_contract,
    math,
    sha256,
)
from .result_registry_support import (  # noqa: F401
    Any,
    COREDEV_COMPONENTS,
    INTERVENTION_AXES,
    INVARIANT_FIELDS,
    IncomparableResultsError,
    RESULT_REGISTRY_SCHEMA,
    RegistryValidationError,
    SecureFileReadError,
    _GOLDEN_PROMOTION_BLOCKED_REASON,
    _PREREGISTRATION_SCHEMA,
    _SCORE_MATCH_ABS_TOLERANCE,
    _SHA256_LENGTH,
    _canonical_sha256,
    _exact_keys,
    _finite_score,
    _object,
    _optional_positive_int,
    _positive_int,
    _read_repository_regular_file,
    _sha256,
    _text,
    read_regular_file_beneath_absolute_directory_nofollow,
)


@dataclass(frozen=True)
class ResultRegistry:
    comparison_definitions: tuple[ComparisonDefinition, ...]
    results: tuple[ResultRecord, ...]
    tables: tuple[ResultTable, ...]

    def __post_init__(self) -> None:
        result_ids = [record.result_id for record in self.results]
        if len(result_ids) != len(set(result_ids)):
            raise RegistryValidationError(
                "registry contains duplicate result_id values"
            )
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise RegistryValidationError("registry contains duplicate table_id values")
        known = set(result_ids)
        comparison_groups = [
            definition.comparison_group for definition in self.comparison_definitions
        ]
        if len(comparison_groups) != len(set(comparison_groups)):
            raise RegistryValidationError(
                "registry contains duplicate comparison_group definitions"
            )
        for definition in self.comparison_definitions:
            unknown_members = set(definition.member_result_ids) - known
            if unknown_members:
                raise RegistryValidationError(
                    f"comparison definition {definition.comparison_group} references "
                    f"unknown results: {sorted(unknown_members)}"
                )
        for record in self.results:
            if not record.delta_eligible:
                continue
            # V2 can check that a typed score JSON contains the registered
            # numbers, but it has no producer-authenticated receipt binding
            # those bytes to the evaluated trajectory set, evaluation
            # identity, weights, and comparison contract.  Preregistration is
            # orthogonal to that missing score provenance.  Reject the status
            # itself so neither parsing nor the table materializer can turn a
            # caller-authored score file into a golden claim.
            raise RegistryValidationError(_GOLDEN_PROMOTION_BLOCKED_REASON)
        for table in self.tables:
            missing = set(table.result_ids) - known
            if missing:
                raise RegistryValidationError(
                    f"table {table.table_id} references unknown results: {sorted(missing)}"
                )

    @classmethod
    def from_json(cls, value: object) -> "ResultRegistry":
        payload = _object(value, context="registry")
        _exact_keys(
            payload,
            required={
                "schema_version",
                "comparison_definitions",
                "results",
                "tables",
            },
            context="registry",
        )
        if payload["schema_version"] != RESULT_REGISTRY_SCHEMA:
            raise RegistryValidationError("unsupported result registry schema_version")
        raw_results = payload["results"]
        raw_tables = payload["tables"]
        raw_definitions = payload["comparison_definitions"]
        if not isinstance(raw_definitions, list):
            raise RegistryValidationError(
                "registry.comparison_definitions must be a list"
            )
        if not isinstance(raw_results, list) or not raw_results:
            raise RegistryValidationError("registry.results must be a non-empty list")
        if not isinstance(raw_tables, list) or not raw_tables:
            raise RegistryValidationError("registry.tables must be a non-empty list")
        return cls(
            comparison_definitions=tuple(
                ComparisonDefinition.from_json(item, index=index)
                for index, item in enumerate(raw_definitions)
            ),
            results=tuple(
                ResultRecord.from_json(item, index=index)
                for index, item in enumerate(raw_results)
            ),
            tables=tuple(
                ResultTable.from_json(item, index=index)
                for index, item in enumerate(raw_tables)
            ),
        )

    def result(self, result_id: str) -> ResultRecord:
        for record in self.results:
            if record.result_id == result_id:
                return record
        raise RegistryValidationError(f"unknown result_id: {result_id}")

    def table(self, table_id: str) -> ResultTable:
        for table in self.tables:
            if table.table_id == table_id:
                return table
        raise RegistryValidationError(f"unknown table_id: {table_id}")

    def comparison_definition(self, comparison_group: str) -> ComparisonDefinition:
        for definition in self.comparison_definitions:
            if definition.comparison_group == comparison_group:
                return definition
        raise IncomparableResultsError(
            f"comparison group is not preregistered: {comparison_group}"
        )

    def delta(self, *, result_id: str, baseline_result_id: str) -> float:
        # Keep the public API fail-closed even for a registry object assembled
        # outside ``from_json``.  V3 may replace this gate only after it verifies
        # a producer receipt linking each score to evaluation identity,
        # trajectory-set identity, weights and the full comparison contract.
        raise IncomparableResultsError(_GOLDEN_PROMOTION_BLOCKED_REASON)

    def verify_artifacts(self, repository_root: Path) -> None:
        root = repository_root.resolve()
        for definition in self.comparison_definitions:
            definition.preregistration_evidence.verify(
                root,
                comparison_group=definition.comparison_group,
                member_result_ids=definition.member_result_ids,
                intervention_axes=definition.intervention_axes,
            )
        for record in self.results:
            if record.score_artifact is not None and record.score is not None:
                record.score_artifact.verify(root, registered_score=record.score)

    def render_table(self, table_id: str) -> str:
        table = self.table(table_id)
        baseline = (
            None
            if table.baseline_result_id is None
            else self.result(table.baseline_result_id)
        )
        lines = [
            f"### {table.title}",
            "",
            "| Result | Status | Train px | Declared eval px | Effective eval px | Macro* | Score evidence | Δ vs reference | Contract |",
            "|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
        for result_id in table.result_ids:
            record = self.result(result_id)
            train_pixels = (
                "—"
                if record.contract.training_image_max_pixels is None
                else f"{record.contract.training_image_max_pixels:,}"
            )
            score = (
                "—"
                if record.score is None
                else f"{record.score.macro_star_percent:.4f}"
            )
            if record.score_artifact is None:
                score_evidence = "—"
            elif record.score_artifact.kind == "score_summary":
                score_evidence = "content-checked"
            else:
                score_evidence = "digest-only"
            if baseline is None:
                delta = "—"
            elif record.result_id == baseline.result_id:
                delta = "reference"
            else:
                try:
                    delta_value = self.delta(
                        result_id=record.result_id,
                        baseline_result_id=baseline.result_id,
                    )
                except IncomparableResultsError:
                    delta = "— (contract differs)"
                else:
                    delta = f"{delta_value:+.4f} pp"
            lines.append(
                "| "
                + " | ".join(
                    (
                        record.label.replace("|", "\\|"),
                        record.status.value,
                        train_pixels,
                        f"{record.contract.declared_evaluation_image_max_pixels:,}",
                        f"{record.contract.effective_evaluation_image_max_pixels:,}",
                        score,
                        score_evidence,
                        delta,
                        f"`{record.contract.identity_sha256[:12]}`",
                    )
                )
                + " |"
            )
        lines.extend(
            (
                "",
                "`Score evidence` is `content-checked` only when the registered Macro* and all seven components are parsed from and matched to a typed score summary; `digest-only` binds provenance bytes but does not validate the registered score content.",
                "",
                "Registry v2 cannot promote `golden` rows or emit numeric `Δ`: its score artifacts are not mechanically bound to evaluation identity, trajectory-set identity, weights and the full comparison contract. `contract differs` is the deliberate fail-closed display until a later receipt schema closes that provenance chain.",
            )
        )
        return "\n".join(lines)

    def render_markdown(self, table_ids: Sequence[str] | None = None) -> str:
        selected = (
            [table.table_id for table in self.tables]
            if table_ids is None
            else list(table_ids)
        )
        sections = [
            "<!-- Generated by tools/materialize_policy_result_table.py; edit the registry, not this table. -->",
            "",
        ]
        for index, table_id in enumerate(selected):
            if index:
                sections.extend(("", ""))
            sections.append(self.render_table(table_id))
        return "\n".join(sections) + "\n"


def load_result_registry(path: str | Path) -> ResultRegistry:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryValidationError(
            f"cannot load result registry: {source}"
        ) from error
    return ResultRegistry.from_json(payload)
