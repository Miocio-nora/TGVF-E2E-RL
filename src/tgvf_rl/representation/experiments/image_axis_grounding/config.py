"""Strict identity sidecar for the removable image-axis grounding experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib
from typing import Any

from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.objective import RepresentationObjectiveKind

from .objective import ImageAxisGroundingObjectiveConfig


IMAGE_AXIS_GROUNDING_CONFIG_SCHEMA_VERSION = "image-axis-grounding-config-v1"
IMAGE_AXIS_GROUNDING_SCOPE = "isolated_representation_image_axis_grounding"
_HEX = frozenset("0123456789abcdef")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "run_id",
        "base_training_config_path",
        "base_training_config_sha256",
        "treatment_training_config_path",
        "treatment_training_config_sha256",
        "donor_manifest_path",
        "donor_manifest_sha256",
        "objective",
    }
)
_OBJECTIVE_FIELDS = frozenset(
    {
        "image_axis_matrix_weight",
        "image_axis_temperature",
        "negative_count",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageAxisGroundingExperimentConfig:
    """Outer contract binding one treatment to RP66 and one donor manifest."""

    schema_version: str
    scope: str
    run_id: str
    base_training_config_path: Path
    base_training_config_sha256: str
    treatment_training_config_path: Path
    treatment_training_config_sha256: str
    donor_manifest_path: Path
    donor_manifest_sha256: str
    objective: ImageAxisGroundingObjectiveConfig
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str
    base_training: RepresentationTrainingConfig
    treatment_training: RepresentationTrainingConfig

    def __post_init__(self) -> None:
        if self.schema_version != IMAGE_AXIS_GROUNDING_CONFIG_SCHEMA_VERSION:
            raise ValueError("image-axis grounding config schema mismatch")
        if self.scope != IMAGE_AXIS_GROUNDING_SCOPE:
            raise ValueError("image-axis grounding config scope mismatch")
        _text(self.run_id, name="run_id")
        for path, name in (
            (self.base_training_config_path, "base training config path"),
            (self.treatment_training_config_path, "treatment training config path"),
            (self.donor_manifest_path, "donor manifest path"),
            (self.source_path, "experiment config path"),
        ):
            _absolute(path, name=name)
        for digest, name in (
            (self.base_training_config_sha256, "base training config SHA256"),
            (
                self.treatment_training_config_sha256,
                "treatment training config SHA256",
            ),
            (self.donor_manifest_sha256, "donor manifest SHA256"),
            (self.source_toml_sha256, "experiment TOML SHA256"),
            (self.canonical_config_sha256, "canonical experiment SHA256"),
        ):
            _sha(digest, name=name)
        if not isinstance(self.objective, ImageAxisGroundingObjectiveConfig):
            raise TypeError("image-axis objective must be typed")
        if not isinstance(self.base_training, RepresentationTrainingConfig) or not isinstance(
            self.treatment_training, RepresentationTrainingConfig
        ):
            raise TypeError("base and treatment training configs must be typed")

    @property
    def expected_treatment_objective_identity(self) -> str:
        return image_axis_treatment_objective_identity(
            base_training_config_sha256=self.base_training_config_sha256,
            donor_manifest_sha256=self.donor_manifest_sha256,
            objective=self.objective,
            base_objective_kind=self.base_training.objective.objective.kind,
        )

    def validation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "run_id": self.run_id,
            "source_path": str(self.source_path),
            "source_toml_sha256": self.source_toml_sha256,
            "canonical_config_sha256": self.canonical_config_sha256,
            "base_training_config_path": str(self.base_training_config_path),
            "base_training_config_sha256": self.base_training_config_sha256,
            "treatment_training_config_path": str(
                self.treatment_training_config_path
            ),
            "treatment_training_config_sha256": (
                self.treatment_training_config_sha256
            ),
            "donor_manifest_path": str(self.donor_manifest_path),
            "donor_manifest_sha256": self.donor_manifest_sha256,
            "image_axis_matrix_weight": self.objective.image_axis_matrix_weight,
            "image_axis_temperature": self.objective.image_axis_temperature,
            "negative_count": self.objective.negative_count,
            "expected_treatment_objective_identity": (
                self.expected_treatment_objective_identity
            ),
            "treatment_run_id": self.treatment_training.run_id,
            "treatment_target_optimizer_steps": (
                self.treatment_training.training.target_optimizer_steps
            ),
            "control_planned_target_optimizer_steps": (
                self.base_training.training.target_optimizer_steps
            ),
            "treatment_stop_optimizer_steps": (
                self.treatment_training.training.target_optimizer_steps
            ),
            "scheduler_horizon_optimizer_steps": (
                self.treatment_training.scheduler.total_steps
            ),
            "gpu_work_launched": False,
        }


def load_image_axis_grounding_experiment_config(
    path: str | Path,
    *,
    verify_bound_files: bool = True,
) -> ImageAxisGroundingExperimentConfig:
    """Load and cross-check the outer, base, treatment, and donor identities."""

    if type(verify_bound_files) is not bool:
        raise TypeError("verify_bound_files must be a bool")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"image-axis config does not exist: {source}")
    raw = source.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid image-axis TOML: {error}") from error
    _fields(payload, _ROOT_FIELDS, name="root")
    objective_payload = _table(
        payload,
        "objective",
        expected=_OBJECTIVE_FIELDS,
    )
    objective = ImageAxisGroundingObjectiveConfig(
        image_axis_matrix_weight=_float(
            objective_payload,
            "image_axis_matrix_weight",
            table="objective",
        ),
        image_axis_temperature=_float(
            objective_payload,
            "image_axis_temperature",
            table="objective",
        ),
        negative_count=_int(objective_payload, "negative_count", table="objective"),
    )
    base_path = Path(_string(payload, "base_training_config_path", table="root"))
    treatment_path = Path(
        _string(payload, "treatment_training_config_path", table="root")
    )
    donor_path = Path(_string(payload, "donor_manifest_path", table="root"))
    base_sha = _string(payload, "base_training_config_sha256", table="root")
    treatment_sha = _string(
        payload,
        "treatment_training_config_sha256",
        table="root",
    )
    donor_sha = _string(payload, "donor_manifest_sha256", table="root")
    for candidate, name in (
        (base_path, "base training config"),
        (treatment_path, "treatment training config"),
        (donor_path, "donor manifest"),
    ):
        _absolute(candidate, name=f"{name} path")
    for digest, name in (
        (base_sha, "base training config SHA256"),
        (treatment_sha, "treatment training config SHA256"),
        (donor_sha, "donor manifest SHA256"),
    ):
        _sha(digest, name=name)
    if verify_bound_files:
        _verify_file(base_path, base_sha, name="base training config")
        _verify_file(treatment_path, treatment_sha, name="treatment training config")
        _verify_file(donor_path, donor_sha, name="donor manifest")
    # Training-config parsing is deliberately unconditional.  A caller may bypass
    # byte verification only for fixture construction, never the strict schema.
    base = load_representation_training_config(base_path)
    treatment = load_representation_training_config(treatment_path)
    config = ImageAxisGroundingExperimentConfig(
        schema_version=_string(payload, "schema_version", table="root"),
        scope=_string(payload, "scope", table="root"),
        run_id=_string(payload, "run_id", table="root"),
        base_training_config_path=base_path,
        base_training_config_sha256=base_sha,
        treatment_training_config_path=treatment_path,
        treatment_training_config_sha256=treatment_sha,
        donor_manifest_path=donor_path,
        donor_manifest_sha256=donor_sha,
        objective=objective,
        source_path=source,
        source_toml_sha256=sha256(raw).hexdigest(),
        canonical_config_sha256=_canonical_sha(payload),
        base_training=base,
        treatment_training=treatment,
    )
    _validate_treatment_parity(config)
    return config


def image_axis_treatment_objective_identity(
    *,
    base_training_config_sha256: str,
    donor_manifest_sha256: str,
    objective: ImageAxisGroundingObjectiveConfig,
    base_objective_kind: RepresentationObjectiveKind = (
        RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM
    ),
) -> str:
    """Identity embedded in the inner core run to bind all outer loss inputs."""

    _sha(base_training_config_sha256, name="base training config SHA256")
    _sha(donor_manifest_sha256, name="donor manifest SHA256")
    if not isinstance(objective, ImageAxisGroundingObjectiveConfig):
        raise TypeError("image-axis objective must be typed")
    if not isinstance(base_objective_kind, RepresentationObjectiveKind):
        raise TypeError("base objective kind must be explicit")
    prefix = "balanced-matrix-ce-l-gen-norm-plus-image-axis-v1"
    if base_objective_kind is (
        RepresentationObjectiveKind.L_GEN_AND_NORM_NO_MATRIX_CE_ABLATION
    ):
        prefix = "l-gen-norm-no-matrix-ce-plus-image-axis-ablation-v1"
    return (
        f"{prefix}:"
        f"base={base_training_config_sha256}:"
        f"donor={donor_manifest_sha256}:"
        f"weight={objective.image_axis_matrix_weight.hex()}:"
        f"temperature={objective.image_axis_temperature.hex()}:"
        f"negatives={objective.negative_count}"
    )


def _validate_treatment_parity(config: ImageAxisGroundingExperimentConfig) -> None:
    base = config.base_training
    treatment = config.treatment_training
    if treatment.run_id != config.run_id:
        raise ValueError("outer run_id must equal treatment run_id")
    for field in (
        "model",
        "adapter_variant",
        "provider",
        "data",
        "prompt",
        "optimizer",
        "scheduler",
        "execution",
        "initialization",
        "fsdp2",
    ):
        if getattr(treatment, field) != getattr(base, field):
            raise ValueError(f"treatment changes RP66 field: {field}")

    base_objective = base.objective
    treatment_objective = treatment.objective
    if type(treatment_objective) is not type(base_objective):
        raise TypeError("treatment changes the legacy objective schema")
    base_terms = base_objective.objective
    treatment_terms = treatment_objective.objective
    comparable_base = _legacy_objective_payload(base_terms)
    comparable_treatment = _legacy_objective_payload(treatment_terms)
    if comparable_treatment != comparable_base:
        raise ValueError("treatment changes an RP66 legacy objective term")
    if (
        treatment_terms.identity
        != config.expected_treatment_objective_identity
    ):
        raise ValueError(
            "treatment objective identity does not bind base/donor/image-axis loss"
        )
    if (
        treatment_objective.manifold_enabled != base_objective.manifold_enabled
        or treatment_objective.manifold_weight != base_objective.manifold_weight
    ):
        raise ValueError("treatment changes RP66 manifold settings")

    bt = base.training
    tt = treatment.training
    for field in (
        "gradient_accumulation_steps",
        "groups_per_rank_per_optimizer_step",
        "log_every_optimizer_steps",
    ):
        if getattr(tt, field) != getattr(bt, field):
            raise ValueError(f"treatment changes RP66 training geometry: {field}")
    accepted_targets = {500, bt.target_optimizer_steps}
    if tt.target_optimizer_steps not in accepted_targets:
        raise ValueError(
            "formal image-axis treatment must target either the 500-step probe "
            "or the control's full optimizer horizon"
        )
    if tt.validation_every_optimizer_steps != base.scheduler.total_steps:
        raise ValueError(
            "isolated treatment must retain the control validation boundary"
        )
    if treatment.checkpoint.save_every_optimizer_steps != 500:
        raise ValueError("formal image-axis treatment must checkpoint at step 500")
    if treatment.resume.enabled:
        checkpoint_path = treatment.resume.checkpoint_path
        if checkpoint_path is None or treatment.checkpoint.directory not in (
            checkpoint_path,
            *checkpoint_path.parents,
        ):
            raise ValueError("resume checkpoint must belong to the treatment output")
    internal_evaluation = treatment.post_training_internal_evaluation
    if tt.target_optimizer_steps == 500:
        if internal_evaluation is None or internal_evaluation.enabled:
            raise ValueError(
                "the 500-step image-axis probe must disable post-training evaluation"
            )
    elif internal_evaluation is None or not internal_evaluation.enabled:
        raise ValueError(
            "the full-horizon image-axis treatment must run internal diagnostics"
        )
    if treatment.output == base.output or treatment.checkpoint.directory == (
        base.checkpoint.directory
    ):
        raise ValueError("treatment outputs must be isolated from RP66")


def _legacy_objective_payload(value: Any) -> tuple[tuple[str, object], ...]:
    fields = (
        "kind",
        "matrix_ce_weight",
        "l_gen_weight",
        "norm_weight",
        "matrix_ce_mode",
        "matrix_ce_temperature",
    )
    return tuple((field, getattr(value, field, None)) for field in fields)


def _table(
    payload: Mapping[str, Any],
    key: str,
    *,
    expected: set[str] | frozenset[str],
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"[{key}] must be a TOML table")
    _fields(value, expected, name=key)
    return value


def _fields(
    payload: Mapping[str, Any], expected: set[str] | frozenset[str], *, name: str
) -> None:
    if set(payload) != set(expected):
        missing = sorted(set(expected) - set(payload))
        unknown = sorted(set(payload) - set(expected))
        raise ValueError(f"[{name}] fields differ: missing={missing} unknown={unknown}")


def _string(payload: Mapping[str, Any], key: str, *, table: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{table}.{key} must be non-empty text")
    return value


def _float(payload: Mapping[str, Any], key: str, *, table: str) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
    ):
        raise TypeError(f"{table}.{key} must be an explicit finite float")
    return value


def _int(payload: Mapping[str, Any], key: str, *, table: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{table}.{key} must be an integer")
    return value


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify_file(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {observed}")


def _absolute(value: object, *, name: str) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute Path")


def _text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _sha(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "IMAGE_AXIS_GROUNDING_CONFIG_SCHEMA_VERSION",
    "IMAGE_AXIS_GROUNDING_SCOPE",
    "ImageAxisGroundingExperimentConfig",
    "ImageAxisGroundingObjectiveConfig",
    "image_axis_treatment_objective_identity",
    "load_image_axis_grounding_experiment_config",
]
