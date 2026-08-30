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
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


RESULT_REGISTRY_SCHEMA = "tgvf.policy-result-registry.v2"
COREDEV_COMPONENTS = (
    "vstar",
    "hr_average_all",
    "blink_single_180",
    "ocr_mean",
    "mmmu_single_269",
    "mathvista",
    "mathverse_five_version_macro",
)
INTERVENTION_AXES = (
    "method",
    "optimizer_step",
    "weights",
    "training_contract_identity",
    "training_image_max_pixels",
    "declared_evaluation_image_max_pixels",
    "effective_evaluation_image_max_pixels",
    "runtime_identity",
    "parser_identity",
    "action_boundary_identity",
    "observation_identity",
    "prompt_identity",
    "generation_identity",
)
INVARIANT_FIELDS = (
    "task_manifest_sha256",
    "rng_identity",
    "scorer_identity",
    "inference_sample_count",
    "scored_sample_count",
    "slice_count",
)
_SHA256_LENGTH = 64
_SCORE_MATCH_ABS_TOLERANCE = 5e-5
_PREREGISTRATION_SCHEMA = "tgvf.comparison-preregistration.v1"
_GOLDEN_PROMOTION_BLOCKED_REASON = (
    "result registry v2 cannot promote golden results: score artifacts are not "
    "mechanically bound to the evaluation identity, trajectory-set identity, "
    "weights, and comparison contract; use a future provenance-receipt schema"
)


class RegistryValidationError(ValueError):
    """Raised when registry evidence is incomplete or internally inconsistent."""


class IncomparableResultsError(RegistryValidationError):
    """Raised when a requested delta crosses a comparison contract."""


class ResultStatus(str, Enum):
    GOLDEN = "golden"
    STANDALONE = "standalone"
    CONFOUNDED = "confounded"
    INVALID = "invalid"
    PENDING = "pending"


class WeightState(str, Enum):
    PRESENT = "present"
    INVALID = "invalid"
    EXPECTED = "expected"


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing:
        raise RegistryValidationError(
            f"{context} is missing required keys: {sorted(missing)}"
        )
    if unknown:
        raise RegistryValidationError(
            f"{context} contains unknown keys: {sorted(unknown)}"
        )


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{context} must be non-empty text")
    return value


def _sha256(value: object, *, context: str) -> str:
    text = _text(value, context=context)
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RegistryValidationError(f"{context} must be lowercase SHA-256")
    return text


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryValidationError(f"{context} must be a positive integer")
    return value


def _optional_positive_int(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, context=context)


def _finite_score(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryValidationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise RegistryValidationError(f"{context} must be finite and in [0, 100]")
    return number


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_repository_regular_file(
    repository_root: Path,
    relative_path: str,
    *,
    context: str,
) -> bytes:
    """Read one bound file without following a symlink in its path chain."""

    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise RegistryValidationError(
            f"artifact repository root cannot be resolved: {repository_root}"
        ) from error
    if not root.is_dir():
        raise RegistryValidationError(
            f"artifact repository root is not a directory: {root}"
        )
    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise RegistryValidationError(
            f"{context} escapes or cannot be resolved under repository root: {candidate}"
        ) from error

    parts = PurePosixPath(relative_path).parts
    if not parts:
        raise RegistryValidationError(f"{context} path does not name a file")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(root, directory_flags)
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags | no_follow,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            file_flags | no_follow,
            dir_fd=directory_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise RegistryValidationError(
                    f"{context} is not a regular file: {candidate}"
                )
            with os.fdopen(file_descriptor, "rb", closefd=True) as stream:
                file_descriptor = -1
                return stream.read()
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
    except RegistryValidationError:
        raise
    except OSError as error:
        raise RegistryValidationError(
            f"{context} cannot be opened without following symlinks: {candidate}"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


@dataclass(frozen=True)
class ScoreArtifact:
    """One immutable file carrying or curating the registered score."""

    path: str
    sha256: str
    kind: str

    @classmethod
    def from_json(cls, value: object, *, context: str) -> "ScoreArtifact":
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={"path", "sha256", "kind"},
            context=context,
        )
        path = _text(payload["path"], context=f"{context}.path")
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise RegistryValidationError(
                f"{context}.path must be repository-relative and cannot contain '..'"
            )
        kind = _text(payload["kind"], context=f"{context}.kind")
        if kind not in {
            "score_summary",
            "evaluation_run_summary_digest_only",
            "curated_report_digest_only",
        }:
            raise RegistryValidationError(
                f"{context}.kind is not a supported score artifact kind"
            )
        return cls(
            path=path,
            sha256=_sha256(payload["sha256"], context=f"{context}.sha256"),
            kind=kind,
        )

    def verify(
        self, repository_root: Path, *, registered_score: "CoreDevScore"
    ) -> None:
        candidate = repository_root / self.path
        raw = _read_repository_regular_file(
            repository_root,
            self.path,
            context="registered score artifact",
        )
        digest = sha256(raw).hexdigest()
        if digest != self.sha256:
            raise RegistryValidationError(
                f"registered score artifact SHA-256 differs: {candidate}"
            )
        if self.kind != "score_summary":
            # These explicit artifact kinds bind provenance bytes only.  They
            # must never be described as content verification of the score.
            return
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistryValidationError(
                f"registered score_summary is not valid UTF-8 JSON: {candidate}"
            ) from error
        root = _object(payload, context=f"score_summary {candidate}")
        headline = _object(
            root.get("headline"), context=f"score_summary {candidate}.headline"
        )
        if headline.get("schema_version") != "tgvf.coredev-2511-macro-star.v1":
            raise RegistryValidationError(
                f"registered score_summary has unsupported headline schema: {candidate}"
            )
        artifact_score = CoreDevScore.from_json(
            {
                "macro_star_percent": headline.get("macro_star_percent"),
                "components_percent": headline.get("components_percent"),
            },
            context=f"score_summary {candidate}.headline",
        )
        _require_matching_score(
            registered_score,
            artifact_score,
            context=f"registered score differs from score_summary: {candidate}",
        )


@dataclass(frozen=True)
class PreregistrationEvidence:
    """Immutable binding to an independent comparison preregistration."""

    path: str
    sha256: str
    kind: str

    @classmethod
    def from_json(cls, value: object, *, context: str) -> "PreregistrationEvidence":
        payload = _object(value, context=context)
        _exact_keys(payload, required={"path", "sha256", "kind"}, context=context)
        path = _text(payload["path"], context=f"{context}.path")
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise RegistryValidationError(
                f"{context}.path must be repository-relative and cannot contain '..'"
            )
        kind = _text(payload["kind"], context=f"{context}.kind")
        if kind != "preregistration_json_v1":
            raise RegistryValidationError(
                f"{context}.kind must be preregistration_json_v1"
            )
        return cls(
            path=path,
            sha256=_sha256(payload["sha256"], context=f"{context}.sha256"),
            kind=kind,
        )

    def verify(
        self,
        repository_root: Path,
        *,
        comparison_group: str,
        member_result_ids: tuple[str, ...],
        intervention_axes: tuple[str, ...],
    ) -> None:
        candidate = repository_root / self.path
        raw = _read_repository_regular_file(
            repository_root,
            self.path,
            context="comparison preregistration",
        )
        if sha256(raw).hexdigest() != self.sha256:
            raise RegistryValidationError(
                f"comparison preregistration SHA-256 differs: {candidate}"
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistryValidationError(
                f"comparison preregistration is not valid UTF-8 JSON: {candidate}"
            ) from error
        preregistration = _object(
            payload, context=f"comparison preregistration {candidate}"
        )
        _exact_keys(
            preregistration,
            required={
                "schema_version",
                "comparison_group",
                "member_result_ids",
                "intervention_axes",
            },
            context=f"comparison preregistration {candidate}",
        )
        expected = {
            "schema_version": _PREREGISTRATION_SCHEMA,
            "comparison_group": comparison_group,
            "member_result_ids": list(member_result_ids),
            "intervention_axes": list(intervention_axes),
        }
        if preregistration != expected:
            raise RegistryValidationError(
                f"comparison preregistration content differs from definition: {candidate}"
            )


@dataclass(frozen=True)
class WeightIdentity:
    state: WeightState
    identity: str
    sha256: str | None

    @classmethod
    def from_json(cls, value: object, *, context: str) -> "WeightIdentity":
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={"state", "identity", "sha256"},
            context=context,
        )
        try:
            state = WeightState(payload["state"])
        except (TypeError, ValueError) as error:
            raise RegistryValidationError(
                f"{context}.state is not a supported weight state"
            ) from error
        digest_value = payload["sha256"]
        digest = (
            None
            if digest_value is None
            else _sha256(digest_value, context=f"{context}.sha256")
        )
        if state is WeightState.PRESENT and digest is None:
            raise RegistryValidationError(
                f"{context}.sha256 is required for present weights"
            )
        if state is WeightState.EXPECTED and digest is not None:
            raise RegistryValidationError(
                f"{context}.sha256 must be null for weights not yet produced"
            )
        return cls(
            state=state,
            identity=_text(payload["identity"], context=f"{context}.identity"),
            sha256=digest,
        )


@dataclass(frozen=True)
class ComparisonContract:
    """Evaluation evidence split into immutable and treatment-capable fields."""

    task_manifest_sha256: str
    training_contract_identity: str
    training_image_max_pixels: int | None
    declared_evaluation_image_max_pixels: int
    effective_evaluation_image_max_pixels: int
    runtime_identity: str
    parser_identity: str
    action_boundary_identity: str
    observation_identity: str
    prompt_identity: str
    rng_identity: str
    generation_identity: str
    scorer_identity: str
    inference_sample_count: int
    scored_sample_count: int
    slice_count: int

    @classmethod
    def from_json(cls, value: object, *, context: str) -> "ComparisonContract":
        payload = _object(value, context=context)
        required = {
            "task_manifest_sha256",
            "training_contract_identity",
            "training_image_max_pixels",
            "declared_evaluation_image_max_pixels",
            "effective_evaluation_image_max_pixels",
            "runtime_identity",
            "parser_identity",
            "action_boundary_identity",
            "observation_identity",
            "prompt_identity",
            "rng_identity",
            "generation_identity",
            "scorer_identity",
            "inference_sample_count",
            "scored_sample_count",
            "slice_count",
        }
        _exact_keys(payload, required=required, context=context)
        return cls(
            task_manifest_sha256=_sha256(
                payload["task_manifest_sha256"],
                context=f"{context}.task_manifest_sha256",
            ),
            training_contract_identity=_text(
                payload["training_contract_identity"],
                context=f"{context}.training_contract_identity",
            ),
            training_image_max_pixels=_optional_positive_int(
                payload["training_image_max_pixels"],
                context=f"{context}.training_image_max_pixels",
            ),
            declared_evaluation_image_max_pixels=_positive_int(
                payload["declared_evaluation_image_max_pixels"],
                context=f"{context}.declared_evaluation_image_max_pixels",
            ),
            effective_evaluation_image_max_pixels=_positive_int(
                payload["effective_evaluation_image_max_pixels"],
                context=f"{context}.effective_evaluation_image_max_pixels",
            ),
            runtime_identity=_text(
                payload["runtime_identity"], context=f"{context}.runtime_identity"
            ),
            parser_identity=_text(
                payload["parser_identity"], context=f"{context}.parser_identity"
            ),
            action_boundary_identity=_text(
                payload["action_boundary_identity"],
                context=f"{context}.action_boundary_identity",
            ),
            observation_identity=_text(
                payload["observation_identity"],
                context=f"{context}.observation_identity",
            ),
            prompt_identity=_text(
                payload["prompt_identity"], context=f"{context}.prompt_identity"
            ),
            rng_identity=_text(
                payload["rng_identity"], context=f"{context}.rng_identity"
            ),
            generation_identity=_text(
                payload["generation_identity"],
                context=f"{context}.generation_identity",
            ),
            scorer_identity=_text(
                payload["scorer_identity"], context=f"{context}.scorer_identity"
            ),
            inference_sample_count=_positive_int(
                payload["inference_sample_count"],
                context=f"{context}.inference_sample_count",
            ),
            scored_sample_count=_positive_int(
                payload["scored_sample_count"],
                context=f"{context}.scored_sample_count",
            ),
            slice_count=_positive_int(
                payload["slice_count"], context=f"{context}.slice_count"
            ),
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "task_manifest_sha256": self.task_manifest_sha256,
            "training_contract_identity": self.training_contract_identity,
            "training_image_max_pixels": self.training_image_max_pixels,
            "declared_evaluation_image_max_pixels": (
                self.declared_evaluation_image_max_pixels
            ),
            "effective_evaluation_image_max_pixels": (
                self.effective_evaluation_image_max_pixels
            ),
            "runtime_identity": self.runtime_identity,
            "parser_identity": self.parser_identity,
            "action_boundary_identity": self.action_boundary_identity,
            "observation_identity": self.observation_identity,
            "prompt_identity": self.prompt_identity,
            "rng_identity": self.rng_identity,
            "generation_identity": self.generation_identity,
            "scorer_identity": self.scorer_identity,
            "inference_sample_count": self.inference_sample_count,
            "scored_sample_count": self.scored_sample_count,
            "slice_count": self.slice_count,
        }

    def differing_fields(self, other: "ComparisonContract") -> tuple[str, ...]:
        own = self.as_dict()
        theirs = other.as_dict()
        return tuple(key for key in own if own[key] != theirs[key])

    def invariant_values(self) -> dict[str, object]:
        payload = self.as_dict()
        return {field: payload[field] for field in INVARIANT_FIELDS}

    def treatment_values(self) -> dict[str, object]:
        payload = self.as_dict()
        contract_axes = set(INTERVENTION_AXES) - {
            "method",
            "optimizer_step",
            "weights",
        }
        return {
            field: payload[field]
            for field in INTERVENTION_AXES
            if field in contract_axes
        }


@dataclass(frozen=True)
class CoreDevScore:
    macro_star_percent: float
    components_percent: Mapping[str, float]

    @classmethod
    def from_json(cls, value: object, *, context: str) -> "CoreDevScore":
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={"macro_star_percent", "components_percent"},
            context=context,
        )
        raw_components = _object(
            payload["components_percent"], context=f"{context}.components_percent"
        )
        _exact_keys(
            raw_components,
            required=set(COREDEV_COMPONENTS),
            context=f"{context}.components_percent",
        )
        components = {
            name: _finite_score(
                raw_components[name], context=f"{context}.components_percent.{name}"
            )
            for name in COREDEV_COMPONENTS
        }
        macro = _finite_score(
            payload["macro_star_percent"], context=f"{context}.macro_star_percent"
        )
        recomputed = sum(components.values()) / len(components)
        if not math.isclose(macro, recomputed, abs_tol=5e-3, rel_tol=0.0):
            raise RegistryValidationError(
                f"{context}.macro_star_percent does not equal the seven-component mean"
            )
        return cls(macro_star_percent=macro, components_percent=components)


def _require_matching_score(
    registered: CoreDevScore,
    artifact: CoreDevScore,
    *,
    context: str,
) -> None:
    mismatches: list[str] = []
    if not math.isclose(
        registered.macro_star_percent,
        artifact.macro_star_percent,
        abs_tol=_SCORE_MATCH_ABS_TOLERANCE,
        rel_tol=0.0,
    ):
        mismatches.append("macro_star_percent")
    for component in COREDEV_COMPONENTS:
        if not math.isclose(
            registered.components_percent[component],
            artifact.components_percent[component],
            abs_tol=_SCORE_MATCH_ABS_TOLERANCE,
            rel_tol=0.0,
        ):
            mismatches.append(f"components_percent.{component}")
    if mismatches:
        raise RegistryValidationError(f"{context}: {', '.join(mismatches)}")


@dataclass(frozen=True)
class ComparisonDefinition:
    """A preregistered comparison group independent from its result rows."""

    comparison_group: str
    member_result_ids: tuple[str, ...]
    intervention_axes: tuple[str, ...]
    preregistration_evidence: PreregistrationEvidence

    @classmethod
    def from_json(cls, value: object, *, index: int) -> "ComparisonDefinition":
        context = f"comparison_definitions[{index}]"
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={
                "comparison_group",
                "member_result_ids",
                "intervention_axes",
                "preregistration_evidence",
            },
            context=context,
        )
        raw_members = payload["member_result_ids"]
        if not isinstance(raw_members, list) or len(raw_members) < 2:
            raise RegistryValidationError(
                f"{context}.member_result_ids must contain at least two results"
            )
        members = tuple(
            _text(member, context=f"{context}.member_result_ids[{member_index}]")
            for member_index, member in enumerate(raw_members)
        )
        if len(members) != len(set(members)):
            raise RegistryValidationError(
                f"{context}.member_result_ids contains duplicates"
            )
        if members != tuple(sorted(members)):
            raise RegistryValidationError(f"{context}.member_result_ids must be sorted")
        raw_axes = payload["intervention_axes"]
        if not isinstance(raw_axes, list) or not raw_axes:
            raise RegistryValidationError(
                f"{context}.intervention_axes must be a non-empty list"
            )
        axes = tuple(
            _text(axis, context=f"{context}.intervention_axes[{axis_index}]")
            for axis_index, axis in enumerate(raw_axes)
        )
        if len(axes) != len(set(axes)):
            raise RegistryValidationError(
                f"{context}.intervention_axes contains duplicates"
            )
        unknown_axes = set(axes) - set(INTERVENTION_AXES)
        if unknown_axes:
            raise RegistryValidationError(
                f"{context}.intervention_axes contains unsupported axes: "
                f"{sorted(unknown_axes)}"
            )
        if axes != tuple(sorted(axes)):
            raise RegistryValidationError(f"{context}.intervention_axes must be sorted")
        return cls(
            comparison_group=_text(
                payload["comparison_group"], context=f"{context}.comparison_group"
            ),
            member_result_ids=members,
            intervention_axes=axes,
            preregistration_evidence=PreregistrationEvidence.from_json(
                payload["preregistration_evidence"],
                context=f"{context}.preregistration_evidence",
            ),
        )


@dataclass(frozen=True)
class ResultRecord:
    result_id: str
    label: str
    method: str
    optimizer_step: int | None
    status: ResultStatus
    status_reason: str
    comparison_group: str
    declared_intervention_axes: tuple[str, ...]
    weights: WeightIdentity
    contract: ComparisonContract
    score: CoreDevScore | None
    score_artifact: ScoreArtifact | None

    @classmethod
    def from_json(cls, value: object, *, index: int) -> "ResultRecord":
        context = f"results[{index}]"
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={
                "result_id",
                "label",
                "method",
                "optimizer_step",
                "status",
                "status_reason",
                "comparison_group",
                "declared_intervention_axes",
                "weights",
                "contract",
                "score",
                "score_artifact",
            },
            context=context,
        )
        try:
            status = ResultStatus(payload["status"])
        except (TypeError, ValueError) as error:
            raise RegistryValidationError(
                f"{context}.status is not a supported result status"
            ) from error
        if status is ResultStatus.GOLDEN:
            # Reject at the row parser as well as the registry aggregate.  A
            # caller must not be able to materialize a v2 ``ResultRecord`` and
            # present it as a provenance-verified result outside the normal
            # registry loader.
            raise RegistryValidationError(_GOLDEN_PROMOTION_BLOCKED_REASON)
        step_value = payload["optimizer_step"]
        if step_value is not None and (
            isinstance(step_value, bool)
            or not isinstance(step_value, int)
            or step_value < 0
        ):
            raise RegistryValidationError(
                f"{context}.optimizer_step must be null or a non-negative integer"
            )
        weights = WeightIdentity.from_json(
            payload["weights"], context=f"{context}.weights"
        )
        score = (
            None
            if payload["score"] is None
            else CoreDevScore.from_json(payload["score"], context=f"{context}.score")
        )
        artifact = (
            None
            if payload["score_artifact"] is None
            else ScoreArtifact.from_json(
                payload["score_artifact"], context=f"{context}.score_artifact"
            )
        )
        if (score is None) != (artifact is None):
            raise RegistryValidationError(
                f"{context} must bind score and score_artifact together"
            )
        if status in {
            ResultStatus.GOLDEN,
            ResultStatus.STANDALONE,
            ResultStatus.CONFOUNDED,
        }:
            if score is None or weights.state is not WeightState.PRESENT:
                raise RegistryValidationError(
                    f"{context} status {status.value} requires measured score and present weights"
                )
        if status is ResultStatus.PENDING:
            if score is not None or weights.state is not WeightState.EXPECTED:
                raise RegistryValidationError(
                    f"{context} pending results cannot carry scores or produced weights"
                )
        if status is ResultStatus.INVALID and weights.state is WeightState.EXPECTED:
            raise RegistryValidationError(
                f"{context} invalid results cannot use expected weights"
            )
        raw_axes = payload["declared_intervention_axes"]
        if not isinstance(raw_axes, list):
            raise RegistryValidationError(
                f"{context}.declared_intervention_axes must be a list"
            )
        axes = tuple(
            _text(axis, context=f"{context}.declared_intervention_axes[{axis_index}]")
            for axis_index, axis in enumerate(raw_axes)
        )
        if len(axes) != len(set(axes)):
            raise RegistryValidationError(
                f"{context}.declared_intervention_axes contains duplicates"
            )
        unknown_axes = set(axes) - set(INTERVENTION_AXES)
        if unknown_axes:
            raise RegistryValidationError(
                f"{context}.declared_intervention_axes contains unsupported axes: "
                f"{sorted(unknown_axes)}"
            )
        if axes != tuple(sorted(axes)):
            raise RegistryValidationError(
                f"{context}.declared_intervention_axes must be sorted"
            )
        return cls(
            result_id=_text(payload["result_id"], context=f"{context}.result_id"),
            label=_text(payload["label"], context=f"{context}.label"),
            method=_text(payload["method"], context=f"{context}.method"),
            optimizer_step=step_value,
            status=status,
            status_reason=_text(
                payload["status_reason"], context=f"{context}.status_reason"
            ),
            comparison_group=_text(
                payload["comparison_group"], context=f"{context}.comparison_group"
            ),
            declared_intervention_axes=axes,
            weights=weights,
            contract=ComparisonContract.from_json(
                payload["contract"], context=f"{context}.contract"
            ),
            score=score,
            score_artifact=artifact,
        )

    @property
    def delta_eligible(self) -> bool:
        # ``standalone`` means exactly that the measurement is valid only for
        # its own declared contract and is not a causal comparator.  Requiring
        # GOLDEN here prevents a comparison-group label from silently
        # promoting two standalone rows into a paper delta.
        return self.status is ResultStatus.GOLDEN

    def treatment_values(self) -> dict[str, object]:
        values = {
            "method": self.method,
            "optimizer_step": self.optimizer_step,
            "weights": {
                "state": self.weights.state.value,
                "identity": self.weights.identity,
                "sha256": self.weights.sha256,
            },
        }
        values.update(self.contract.treatment_values())
        return values


def _validate_comparison_contract(
    *,
    result: ResultRecord,
    baseline: ResultRecord,
    definition: ComparisonDefinition,
) -> None:
    """Validate comparison algebra without authorizing a result claim.

    This deliberately does not inspect or subtract scores.  It preserves the
    invariant/intervention logic needed by a future receipt-verifying schema,
    while registry v2 keeps both golden promotion and numeric deltas closed.
    """

    if result.comparison_group != baseline.comparison_group:
        raise IncomparableResultsError("comparison_group differs")
    if result.comparison_group != definition.comparison_group:
        raise IncomparableResultsError(
            "result comparison_group differs from comparison definition"
        )
    if (
        result.result_id not in definition.member_result_ids
        or baseline.result_id not in definition.member_result_ids
    ):
        raise IncomparableResultsError(
            "result is not a member of the preregistered comparison group"
        )
    if (
        result.declared_intervention_axes != definition.intervention_axes
        or baseline.declared_intervention_axes != definition.intervention_axes
    ):
        raise IncomparableResultsError(
            "result intervention axes differ from the preregistered definition"
        )
    result_invariants = result.contract.invariant_values()
    baseline_invariants = baseline.contract.invariant_values()
    invariant_differences = tuple(
        field
        for field in INVARIANT_FIELDS
        if result_invariants[field] != baseline_invariants[field]
    )
    if invariant_differences:
        raise IncomparableResultsError(
            "comparison invariant differs: " + ", ".join(invariant_differences)
        )
    result_treatments = result.treatment_values()
    baseline_treatments = baseline.treatment_values()
    actual_axes = {
        axis
        for axis in INTERVENTION_AXES
        if result_treatments[axis] != baseline_treatments[axis]
    }
    declared_axes = set(definition.intervention_axes)
    undeclared_axes = actual_axes - declared_axes
    missing_axes = declared_axes - actual_axes
    if undeclared_axes:
        raise IncomparableResultsError(
            "undeclared intervention differences: " + ", ".join(sorted(undeclared_axes))
        )
    if missing_axes:
        raise IncomparableResultsError(
            "declared intervention axes did not differ: "
            + ", ".join(sorted(missing_axes))
        )


@dataclass(frozen=True)
class ResultTable:
    table_id: str
    title: str
    result_ids: tuple[str, ...]
    baseline_result_id: str | None

    @classmethod
    def from_json(cls, value: object, *, index: int) -> "ResultTable":
        context = f"tables[{index}]"
        payload = _object(value, context=context)
        _exact_keys(
            payload,
            required={"table_id", "title", "result_ids", "baseline_result_id"},
            context=context,
        )
        raw_ids = payload["result_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise RegistryValidationError(f"{context}.result_ids must be non-empty")
        result_ids = tuple(
            _text(item, context=f"{context}.result_ids[{item_index}]")
            for item_index, item in enumerate(raw_ids)
        )
        if len(set(result_ids)) != len(result_ids):
            raise RegistryValidationError(f"{context}.result_ids contains duplicates")
        baseline_value = payload["baseline_result_id"]
        baseline = (
            None
            if baseline_value is None
            else _text(baseline_value, context=f"{context}.baseline_result_id")
        )
        if baseline is not None and baseline not in result_ids:
            raise RegistryValidationError(
                f"{context}.baseline_result_id must appear in result_ids"
            )
        return cls(
            table_id=_text(payload["table_id"], context=f"{context}.table_id"),
            title=_text(payload["title"], context=f"{context}.title"),
            result_ids=result_ids,
            baseline_result_id=baseline,
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
