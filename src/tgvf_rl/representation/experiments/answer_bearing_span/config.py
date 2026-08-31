"""Strict outer identity for the removable RP70 span-supervision experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tgvf_rl.representation.adapter import TGVFAdapterVariant
from tgvf_rl.representation.experiments.image_axis_grounding.trainer import (
    ImageAxisGroundingObjectiveConfig,
)
from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.losses import MatrixCEScoreMode
from tgvf_rl.representation.training.objective import RepresentationObjectiveKind

from .data import (
    ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION,
    ANSWER_BEARING_SPAN_MATCH_POLICY,
)
from .supervision import (
    ANSWER_BEARING_SPAN_SUPERVISION_POLICY,
    ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION,
)


ANSWER_BEARING_SPAN_CONFIG_SCHEMA_VERSION = "answer-bearing-span-config-v1"
ANSWER_BEARING_SPAN_SCOPE = "isolated_representation_answer_bearing_span"
ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION = (
    f"{ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION}+"
    f"{ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION}"
)
ANSWER_BEARING_SPAN_POLICY = (
    f"{ANSWER_BEARING_SPAN_MATCH_POLICY}:"
    f"{ANSWER_BEARING_SPAN_SUPERVISION_POLICY}:"
    "semantic-bound-explicit-no-unadjudicated"
)

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
        "train_span_sidecar_path",
        "train_span_sidecar_sha256",
        "test_span_sidecar_path",
        "test_span_sidecar_sha256",
        "span",
        "objective",
    }
)
_SPAN_FIELDS = frozenset({"schema_version", "policy"})
_OBJECTIVE_FIELDS = frozenset(
    {"image_axis_matrix_weight", "image_axis_temperature", "negative_count"}
)


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanPolicyConfig:
    """Frozen token-ownership policy for the first RP70 experiment."""

    schema_version: str = ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION
    policy: str = ANSWER_BEARING_SPAN_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION:
            raise ValueError("answer-bearing span policy schema mismatch")
        if self.policy != ANSWER_BEARING_SPAN_POLICY:
            raise ValueError("RP70 v1 span policy is fixed")

    @property
    def identity(self) -> str:
        return f"{self.schema_version}:{self.policy}"

    def validation_payload(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "policy": self.policy}


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerBearingSpanExperimentConfig:
    """Bind RP70 to RP66, RP67 donors, both data splits, and one span policy."""

    schema_version: str
    scope: str
    run_id: str
    base_training_config_path: Path
    base_training_config_sha256: str
    treatment_training_config_path: Path
    treatment_training_config_sha256: str
    donor_manifest_path: Path
    donor_manifest_sha256: str
    train_span_sidecar_path: Path
    train_span_sidecar_sha256: str
    test_span_sidecar_path: Path
    test_span_sidecar_sha256: str
    span: AnswerBearingSpanPolicyConfig
    objective: ImageAxisGroundingObjectiveConfig
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str
    base_training: RepresentationTrainingConfig
    treatment_training: RepresentationTrainingConfig

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_BEARING_SPAN_CONFIG_SCHEMA_VERSION:
            raise ValueError("answer-bearing span config schema mismatch")
        if self.scope != ANSWER_BEARING_SPAN_SCOPE:
            raise ValueError("answer-bearing span config scope mismatch")
        _text(self.run_id, name="run_id")
        for path, name in (
            (self.base_training_config_path, "base training config path"),
            (self.treatment_training_config_path, "treatment training config path"),
            (self.donor_manifest_path, "donor manifest path"),
            (self.train_span_sidecar_path, "train span sidecar path"),
            (self.test_span_sidecar_path, "test span sidecar path"),
            (self.source_path, "experiment config path"),
        ):
            _absolute(path, name=name)
        for digest, name in (
            (self.base_training_config_sha256, "base training config SHA256"),
            (self.treatment_training_config_sha256, "treatment training config SHA256"),
            (self.donor_manifest_sha256, "donor manifest SHA256"),
            (self.train_span_sidecar_sha256, "train span sidecar SHA256"),
            (self.test_span_sidecar_sha256, "test span sidecar SHA256"),
            (self.source_toml_sha256, "experiment TOML SHA256"),
            (self.canonical_config_sha256, "canonical experiment SHA256"),
        ):
            _sha(digest, name=name)
        if not isinstance(self.span, AnswerBearingSpanPolicyConfig):
            raise TypeError("span policy must be typed")
        if not isinstance(self.objective, ImageAxisGroundingObjectiveConfig):
            raise TypeError("image-axis objective must be typed")
        if not isinstance(
            self.base_training, RepresentationTrainingConfig
        ) or not isinstance(self.treatment_training, RepresentationTrainingConfig):
            raise TypeError("base and treatment training configs must be typed")

    @property
    def train_source_sha256(self) -> str:
        return self.treatment_training.data.train.source_sha256

    @property
    def test_source_sha256(self) -> str:
        return self.treatment_training.data.validation.source_sha256

    @property
    def expected_treatment_objective_identity(self) -> str:
        return answer_bearing_span_treatment_objective_identity(
            base_training_config_sha256=self.base_training_config_sha256,
            donor_manifest_sha256=self.donor_manifest_sha256,
            train_source_sha256=self.train_source_sha256,
            test_source_sha256=self.test_source_sha256,
            train_span_sidecar_sha256=self.train_span_sidecar_sha256,
            test_span_sidecar_sha256=self.test_span_sidecar_sha256,
            span=self.span,
            objective=self.objective,
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
            "treatment_training_config_path": str(self.treatment_training_config_path),
            "treatment_training_config_sha256": (self.treatment_training_config_sha256),
            "donor_manifest_path": str(self.donor_manifest_path),
            "donor_manifest_sha256": self.donor_manifest_sha256,
            "train_span_sidecar_path": str(self.train_span_sidecar_path),
            "train_span_sidecar_sha256": self.train_span_sidecar_sha256,
            "test_span_sidecar_path": str(self.test_span_sidecar_path),
            "test_span_sidecar_sha256": self.test_span_sidecar_sha256,
            "train_source_sha256": self.train_source_sha256,
            "test_source_sha256": self.test_source_sha256,
            "span": self.span.validation_payload(),
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
            "scheduler_horizon_optimizer_steps": (
                self.treatment_training.scheduler.total_steps
            ),
            "gpu_work_launched": False,
        }


def load_answer_bearing_span_experiment_config(
    path: str | Path,
    *,
    verify_bound_files: bool = True,
) -> AnswerBearingSpanExperimentConfig:
    """Load and cross-check the RP66/RP67/RP70 identities without CUDA."""

    if type(verify_bound_files) is not bool:
        raise TypeError("verify_bound_files must be a bool")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"answer-bearing span config does not exist: {source}")
    raw = source.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid answer-bearing span TOML: {error}") from error
    _fields(payload, _ROOT_FIELDS, name="root")
    span_payload = _table(payload, "span", expected=_SPAN_FIELDS)
    span = AnswerBearingSpanPolicyConfig(
        schema_version=_string(span_payload, "schema_version", table="span"),
        policy=_string(span_payload, "policy", table="span"),
    )
    objective_payload = _table(payload, "objective", expected=_OBJECTIVE_FIELDS)
    objective = ImageAxisGroundingObjectiveConfig(
        image_axis_matrix_weight=_float(
            objective_payload, "image_axis_matrix_weight", table="objective"
        ),
        image_axis_temperature=_float(
            objective_payload, "image_axis_temperature", table="objective"
        ),
        negative_count=_int(objective_payload, "negative_count", table="objective"),
    )
    base_path = Path(_string(payload, "base_training_config_path", table="root"))
    treatment_path = Path(
        _string(payload, "treatment_training_config_path", table="root")
    )
    donor_path = Path(_string(payload, "donor_manifest_path", table="root"))
    train_span_sidecar_path = Path(
        _string(payload, "train_span_sidecar_path", table="root")
    )
    test_span_sidecar_path = Path(
        _string(payload, "test_span_sidecar_path", table="root")
    )
    base_sha = _string(payload, "base_training_config_sha256", table="root")
    treatment_sha = _string(payload, "treatment_training_config_sha256", table="root")
    donor_sha = _string(payload, "donor_manifest_sha256", table="root")
    train_span_sidecar_sha = _string(payload, "train_span_sidecar_sha256", table="root")
    test_span_sidecar_sha = _string(payload, "test_span_sidecar_sha256", table="root")
    for candidate, name in (
        (base_path, "base training config"),
        (treatment_path, "treatment training config"),
        (donor_path, "donor manifest"),
        (train_span_sidecar_path, "train span sidecar"),
        (test_span_sidecar_path, "test span sidecar"),
    ):
        _absolute(candidate, name=f"{name} path")
    for digest, name in (
        (base_sha, "base training config SHA256"),
        (treatment_sha, "treatment training config SHA256"),
        (donor_sha, "donor manifest SHA256"),
        (train_span_sidecar_sha, "train span sidecar SHA256"),
        (test_span_sidecar_sha, "test span sidecar SHA256"),
    ):
        _sha(digest, name=name)
    if verify_bound_files:
        _verify_file(base_path, base_sha, name="base training config")
        _verify_file(treatment_path, treatment_sha, name="treatment training config")
        _verify_file(donor_path, donor_sha, name="donor manifest")
        _verify_file(
            train_span_sidecar_path,
            train_span_sidecar_sha,
            name="train span sidecar",
        )
        _verify_file(
            test_span_sidecar_path,
            test_span_sidecar_sha,
            name="test span sidecar",
        )
    base = load_representation_training_config(base_path)
    treatment = load_representation_training_config(treatment_path)
    config = AnswerBearingSpanExperimentConfig(
        schema_version=_string(payload, "schema_version", table="root"),
        scope=_string(payload, "scope", table="root"),
        run_id=_string(payload, "run_id", table="root"),
        base_training_config_path=base_path,
        base_training_config_sha256=base_sha,
        treatment_training_config_path=treatment_path,
        treatment_training_config_sha256=treatment_sha,
        donor_manifest_path=donor_path,
        donor_manifest_sha256=donor_sha,
        train_span_sidecar_path=train_span_sidecar_path,
        train_span_sidecar_sha256=train_span_sidecar_sha,
        test_span_sidecar_path=test_span_sidecar_path,
        test_span_sidecar_sha256=test_span_sidecar_sha,
        span=span,
        objective=objective,
        source_path=source,
        source_toml_sha256=sha256(raw).hexdigest(),
        canonical_config_sha256=_canonical_sha(payload),
        base_training=base,
        treatment_training=treatment,
    )
    _validate_treatment_parity(config)
    return config


def answer_bearing_span_treatment_objective_identity(
    *,
    base_training_config_sha256: str,
    donor_manifest_sha256: str,
    train_source_sha256: str,
    test_source_sha256: str,
    train_span_sidecar_sha256: str,
    test_span_sidecar_sha256: str,
    span: AnswerBearingSpanPolicyConfig,
    objective: ImageAxisGroundingObjectiveConfig,
) -> str:
    """Content-bind every source that can change RP70's supervised tokens."""

    for digest, name in (
        (base_training_config_sha256, "base training config SHA256"),
        (donor_manifest_sha256, "donor manifest SHA256"),
        (train_source_sha256, "train source SHA256"),
        (test_source_sha256, "test source SHA256"),
        (train_span_sidecar_sha256, "train span sidecar SHA256"),
        (test_span_sidecar_sha256, "test span sidecar SHA256"),
    ):
        _sha(digest, name=name)
    if not isinstance(span, AnswerBearingSpanPolicyConfig):
        raise TypeError("span policy must be explicit")
    if not isinstance(objective, ImageAxisGroundingObjectiveConfig):
        raise TypeError("image-axis objective must be explicit")
    return (
        "answer-bearing-span-balanced-matrix-ce-l-gen-norm-plus-image-axis-v1:"
        f"base={base_training_config_sha256}:"
        f"donor={donor_manifest_sha256}:"
        f"train={train_source_sha256}:"
        f"test={test_source_sha256}:"
        f"train_spans={train_span_sidecar_sha256}:"
        f"test_spans={test_span_sidecar_sha256}:"
        f"span={span.identity}:"
        f"weight={objective.image_axis_matrix_weight.hex()}:"
        f"temperature={objective.image_axis_temperature.hex()}:"
        f"negatives={objective.negative_count}:"
        "driver=answer-bearing-span-runner-v1"
    )


def _validate_treatment_parity(config: AnswerBearingSpanExperimentConfig) -> None:
    base = config.base_training
    treatment = config.treatment_training
    if treatment.run_id != config.run_id:
        raise ValueError("outer run_id must equal treatment run_id")
    if base.adapter_variant is not TGVFAdapterVariant.FULL_D_DEEPSTACK or (
        treatment.adapter_variant is not TGVFAdapterVariant.FULL_D_DEEPSTACK
    ):
        raise ValueError("RP70 must retain the historical RP66 Adapter structure")
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
    for name, execution in (
        ("base", base_objective),
        ("treatment", treatment_objective),
    ):
        terms = execution.objective
        expected = (
            ("kind", RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM),
            ("matrix_ce_weight", 1.0),
            ("l_gen_weight", 1.0),
            ("norm_weight", 0.1),
            ("matrix_ce_mode", MatrixCEScoreMode.BALANCED),
            ("matrix_ce_temperature", 1.0),
        )
        for field, value in expected:
            if getattr(terms, field, None) != value:
                raise ValueError(f"{name} objective changes fixed RP70 term: {field}")
        if execution.manifold_enabled or execution.manifold_weight != 0.0:
            raise ValueError(f"{name} objective enables an unplanned manifold term")
    if treatment_objective.objective.identity != (
        config.expected_treatment_objective_identity
    ):
        raise ValueError(
            "treatment objective identity does not bind RP70 data/span/image inputs"
        )

    bt = base.training
    tt = treatment.training
    for field in (
        "gradient_accumulation_steps",
        "groups_per_rank_per_optimizer_step",
        "log_every_optimizer_steps",
    ):
        if getattr(tt, field) != getattr(bt, field):
            raise ValueError(f"treatment changes RP66 training geometry: {field}")
    if tt.target_optimizer_steps not in {500, bt.target_optimizer_steps}:
        raise ValueError(
            "RP70 must target either the 500-step probe or RP66's full horizon"
        )
    if tt.validation_every_optimizer_steps != base.scheduler.total_steps:
        raise ValueError("RP70 must retain the RP66 validation boundary")
    if treatment.checkpoint.save_every_optimizer_steps != 500:
        raise ValueError("RP70 must checkpoint at step 500")
    if treatment.resume.enabled:
        checkpoint_path = treatment.resume.checkpoint_path
        if checkpoint_path is None or treatment.checkpoint.directory not in (
            checkpoint_path,
            *checkpoint_path.parents,
        ):
            raise ValueError("resume checkpoint must belong to the RP70 output")
    internal_evaluation = treatment.post_training_internal_evaluation
    if tt.target_optimizer_steps == 500:
        if internal_evaluation is None or internal_evaluation.enabled:
            raise ValueError("the 500-step RP70 probe must disable internal evaluation")
    elif internal_evaluation is None or not internal_evaluation.enabled:
        raise ValueError("full-horizon RP70 must run internal diagnostics")
    if treatment.output == base.output or treatment.checkpoint.directory == (
        base.checkpoint.directory
    ):
        raise ValueError("RP70 outputs must be isolated from RP66")


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
    "ANSWER_BEARING_SPAN_CONFIG_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_POLICY",
    "ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_SCOPE",
    "AnswerBearingSpanExperimentConfig",
    "AnswerBearingSpanPolicyConfig",
    "answer_bearing_span_treatment_objective_identity",
    "load_answer_bearing_span_experiment_config",
]
