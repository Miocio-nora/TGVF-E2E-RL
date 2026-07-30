"""Strict sidecar identity for the isolated answer-utility experiment.

The sidecar selects one frozen E0--E4 topology and byte-binds the unchanged
representation-training TOML that supplies model/data/execution settings.
It deliberately does not import or mutate the production training config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - local legacy test environment
    import tomli as tomllib
from typing import Any

from .objective import (
    ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION,
    AnswerUtilityObjectiveConfig,
)


ANSWER_UTILITY_EXPERIMENT_CONFIG_SCHEMA_VERSION = "answer-utility-experiment-config-v1"
ANSWER_UTILITY_EXPERIMENT_SCOPE = "isolated_representation_answer_utility"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "run_id",
        "variant",
        "base_training_config_path",
        "base_training_config_sha256",
        "objective",
    }
)
_OBJECTIVE_FIELDS = frozenset(
    {
        "schema_version",
        "answer_weight",
        "correct_vs_zero_weight",
        "correct_vs_wrong_weight",
        "existing_evidence_weight",
        "existing_matrix_weight",
        "norm_weight",
        "comparison_margin",
        "comparison_temperature",
    }
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class AnswerSupervisionView(str, Enum):
    """Where answer labels are attached for one experiment variant."""

    NONE = "none"
    GOLD_EVIDENCE = "gold_evidence"
    CLEAN_D_ONLY = "clean_d_only"


class AnswerUtilityExperimentVariant(str, Enum):
    """Frozen experiment grid discussed for the initial utility diagnosis."""

    E0 = "e0"
    E0_CONTINUATION = "e0_continuation"
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"


@dataclass(frozen=True, slots=True)
class AnswerUtilityExperimentProfile:
    """Exact topology and loss-weight contract for one E0--E4 variant."""

    variant: AnswerUtilityExperimentVariant
    answer_supervision_view: AnswerSupervisionView
    train_adapter: bool
    requires_zero_control: bool
    requires_wrong_control: bool
    expected_loss_weights: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.variant, AnswerUtilityExperimentVariant):
            raise TypeError("variant must be AnswerUtilityExperimentVariant")
        if not isinstance(self.answer_supervision_view, AnswerSupervisionView):
            raise TypeError("answer_supervision_view must be AnswerSupervisionView")
        for name in (
            "train_adapter",
            "requires_zero_control",
            "requires_wrong_control",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if len(self.expected_loss_weights) != 6 or any(
            isinstance(value, bool)
            or not isinstance(value, float)
            or not math.isfinite(value)
            or value < 0.0
            for value in self.expected_loss_weights
        ):
            raise ValueError(
                "expected_loss_weights must contain six finite non-negative floats"
            )
        answer, zero, wrong, evidence, matrix, norm = self.expected_loss_weights
        if self.answer_supervision_view is AnswerSupervisionView.NONE:
            if answer != 0.0 or zero != 0.0 or wrong != 0.0:
                raise ValueError("the no-answer view cannot carry answer loss weights")
        elif answer <= 0.0:
            raise ValueError("an answer-supervision view requires answer weight")
        if (zero > 0.0) is not self.requires_zero_control:
            raise ValueError("zero-control requirement must match its loss weight")
        if (wrong > 0.0) is not self.requires_wrong_control:
            raise ValueError("wrong-control requirement must match its loss weight")
        if evidence <= 0.0 or matrix <= 0.0 or norm <= 0.0:
            raise ValueError("all frozen profiles retain the three legacy auxiliaries")


# Tuple order: answer, correct-vs-zero, correct-vs-wrong, existing evidence,
# existing Matrix-CE, historical norm.  E0 is the unchanged RP66 objective and
# is evaluation-only.  E0_CONTINUATION is the separately named, trainable,
# matched-budget continuation needed for formal ablations.  E1--E4 use the
# agreed auxiliary weights for the smoke
# experiment; changing one requires a new named profile/schema rather than a
# silent sidecar override.
ANSWER_UTILITY_EXPERIMENT_PROFILES: Mapping[
    AnswerUtilityExperimentVariant, AnswerUtilityExperimentProfile
] = MappingProxyType(
    {
        AnswerUtilityExperimentVariant.E0: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E0,
            answer_supervision_view=AnswerSupervisionView.NONE,
            train_adapter=False,
            requires_zero_control=False,
            requires_wrong_control=False,
            expected_loss_weights=(0.0, 0.0, 0.0, 1.0, 1.0, 0.1),
        ),
        AnswerUtilityExperimentVariant.E0_CONTINUATION: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E0_CONTINUATION,
            answer_supervision_view=AnswerSupervisionView.NONE,
            train_adapter=True,
            requires_zero_control=False,
            requires_wrong_control=False,
            expected_loss_weights=(0.0, 0.0, 0.0, 1.0, 1.0, 0.1),
        ),
        AnswerUtilityExperimentVariant.E1: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E1,
            answer_supervision_view=AnswerSupervisionView.GOLD_EVIDENCE,
            train_adapter=True,
            requires_zero_control=False,
            requires_wrong_control=False,
            expected_loss_weights=(1.0, 0.0, 0.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E2: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E2,
            answer_supervision_view=AnswerSupervisionView.CLEAN_D_ONLY,
            train_adapter=True,
            requires_zero_control=False,
            requires_wrong_control=False,
            expected_loss_weights=(1.0, 0.0, 0.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E3: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E3,
            answer_supervision_view=AnswerSupervisionView.GOLD_EVIDENCE,
            train_adapter=True,
            requires_zero_control=True,
            requires_wrong_control=True,
            expected_loss_weights=(1.0, 1.0, 1.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E4: AnswerUtilityExperimentProfile(
            variant=AnswerUtilityExperimentVariant.E4,
            answer_supervision_view=AnswerSupervisionView.CLEAN_D_ONLY,
            train_adapter=True,
            requires_zero_control=True,
            requires_wrong_control=True,
            expected_loss_weights=(1.0, 1.0, 1.0, 0.25, 0.25, 0.1),
        ),
    }
)


def answer_utility_experiment_profile(
    variant: AnswerUtilityExperimentVariant,
) -> AnswerUtilityExperimentProfile:
    """Resolve one frozen profile without accepting an untyped string."""

    if not isinstance(variant, AnswerUtilityExperimentVariant):
        raise TypeError("variant must be AnswerUtilityExperimentVariant")
    return ANSWER_UTILITY_EXPERIMENT_PROFILES[variant]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerUtilityExperimentConfig:
    """Loaded sidecar plus its byte-level source identities."""

    schema_version: str
    scope: str
    run_id: str
    variant: AnswerUtilityExperimentVariant
    base_training_config_path: Path
    base_training_config_sha256: str
    objective: AnswerUtilityObjectiveConfig
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_UTILITY_EXPERIMENT_CONFIG_SCHEMA_VERSION:
            raise ValueError("answer-utility experiment config schema mismatch")
        if self.scope != ANSWER_UTILITY_EXPERIMENT_SCOPE:
            raise ValueError("answer-utility experiment scope mismatch")
        _non_empty_string(self.run_id, field_name="run_id")
        if not isinstance(self.variant, AnswerUtilityExperimentVariant):
            raise TypeError("variant must be AnswerUtilityExperimentVariant")
        _absolute_path(
            self.base_training_config_path,
            field_name="base_training_config_path",
        )
        _absolute_path(self.source_path, field_name="source_path")
        _sha256(self.base_training_config_sha256, field_name="base config SHA256")
        _sha256(self.source_toml_sha256, field_name="source TOML SHA256")
        _sha256(self.canonical_config_sha256, field_name="canonical config SHA256")
        if not isinstance(self.objective, AnswerUtilityObjectiveConfig):
            raise TypeError("objective must be AnswerUtilityObjectiveConfig")
        profile = answer_utility_experiment_profile(self.variant)
        if self.objective.loss_weights != profile.expected_loss_weights:
            raise ValueError(
                f"objective weights do not match frozen {self.variant.value} profile"
            )
        comparisons_active = (
            profile.requires_zero_control or profile.requires_wrong_control
        )
        if comparisons_active and self.objective.comparison_margin <= 0.0:
            raise ValueError("counterfactual profile requires a positive margin")
        if not comparisons_active and self.objective.comparison_margin != 0.0:
            raise ValueError("non-counterfactual profile requires margin 0.0")

    @property
    def profile(self) -> AnswerUtilityExperimentProfile:
        return answer_utility_experiment_profile(self.variant)

    def validation_payload(self) -> dict[str, object]:
        """Small JSON-safe identity for validation-only tools."""

        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "run_id": self.run_id,
            "variant": self.variant.value,
            "answer_supervision_view": self.profile.answer_supervision_view.value,
            "train_adapter": self.profile.train_adapter,
            "requires_zero_control": self.profile.requires_zero_control,
            "requires_wrong_control": self.profile.requires_wrong_control,
            "base_training_config_path": str(self.base_training_config_path),
            "base_training_config_sha256": self.base_training_config_sha256,
            "source_path": str(self.source_path),
            "source_toml_sha256": self.source_toml_sha256,
            "canonical_config_sha256": self.canonical_config_sha256,
            "objective_schema_version": self.objective.schema_version,
            "objective_loss_weights": list(self.objective.loss_weights),
            "comparison_margin": self.objective.comparison_margin,
            "comparison_temperature": self.objective.comparison_temperature,
            "gpu_work_launched": False,
        }


def load_answer_utility_experiment_config(
    path: str | Path,
    *,
    verify_base_training_config: bool = True,
) -> AnswerUtilityExperimentConfig:
    """Load a strict sidecar and optionally verify its bound base TOML bytes."""

    if type(verify_base_training_config) is not bool:
        raise TypeError("verify_base_training_config must be a bool")
    source_path = _source_path(path)
    raw = source_path.read_bytes()
    source_sha256 = sha256(raw).hexdigest()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid answer-utility sidecar TOML: {error}") from error
    if not isinstance(value, dict):  # tomllib currently always returns dict
        raise TypeError("answer-utility sidecar TOML root must be a table")
    _exact_fields(value, _ROOT_FIELDS, table="root")

    raw_variant = _string(value, "variant", table="root")
    try:
        variant = AnswerUtilityExperimentVariant(raw_variant)
    except ValueError as error:
        raise ValueError(f"root.variant is unsupported: {raw_variant!r}") from error
    objective = _parse_objective(_table(value, "objective", table="root"))
    config = AnswerUtilityExperimentConfig(
        schema_version=_string(value, "schema_version", table="root"),
        scope=_string(value, "scope", table="root"),
        run_id=_string(value, "run_id", table="root"),
        variant=variant,
        base_training_config_path=_path(
            value, "base_training_config_path", table="root"
        ),
        base_training_config_sha256=_string(
            value, "base_training_config_sha256", table="root"
        ),
        objective=objective,
        source_path=source_path,
        source_toml_sha256=source_sha256,
        canonical_config_sha256=_canonical_mapping_sha256(value),
    )
    if verify_base_training_config:
        _verify_base_training_config(config)
    return config


def _parse_objective(value: Mapping[str, Any]) -> AnswerUtilityObjectiveConfig:
    _exact_fields(value, _OBJECTIVE_FIELDS, table="objective")
    return AnswerUtilityObjectiveConfig(
        schema_version=_string(value, "schema_version", table="objective"),
        answer_weight=_float(value, "answer_weight", table="objective"),
        correct_vs_zero_weight=_float(
            value, "correct_vs_zero_weight", table="objective"
        ),
        correct_vs_wrong_weight=_float(
            value, "correct_vs_wrong_weight", table="objective"
        ),
        existing_evidence_weight=_float(
            value, "existing_evidence_weight", table="objective"
        ),
        existing_matrix_weight=_float(
            value, "existing_matrix_weight", table="objective"
        ),
        norm_weight=_float(value, "norm_weight", table="objective"),
        comparison_margin=_float(value, "comparison_margin", table="objective"),
        comparison_temperature=_float(
            value, "comparison_temperature", table="objective"
        ),
    )


def _verify_base_training_config(config: AnswerUtilityExperimentConfig) -> None:
    path = config.base_training_config_path
    if not path.is_file():
        raise FileNotFoundError(f"base training config does not exist: {path}")
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != config.base_training_config_sha256:
        raise ValueError(
            "base training config SHA256 mismatch: "
            f"expected {config.base_training_config_sha256}, got {actual}"
        )


def _source_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("answer-utility config path must be str or Path")
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"answer-utility config does not exist: {candidate}")
    return candidate


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, table: str
) -> None:
    actual = set(value)
    missing = sorted(set(expected) - actual)
    unknown = sorted(actual - set(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"[{table}] fields do not match schema: {' '.join(details)}")


def _table(value: Mapping[str, Any], key: str, *, table: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(f"[{table}.{key}] must be a TOML table")
    return item


def _string(value: Mapping[str, Any], key: str, *, table: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{table}.{key} must be a non-empty string")
    return item


def _path(value: Mapping[str, Any], key: str, *, table: str) -> Path:
    return Path(_string(value, key, table=table))


def _float(value: Mapping[str, Any], key: str, *, table: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, float) or not math.isfinite(item):
        raise TypeError(f"{table}.{key} must be an explicit finite TOML float")
    return item


def _non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")


def _absolute_path(value: object, *, field_name: str) -> None:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be absolute")


def _sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")


__all__ = [
    "ANSWER_UTILITY_EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "ANSWER_UTILITY_EXPERIMENT_PROFILES",
    "ANSWER_UTILITY_EXPERIMENT_SCOPE",
    "ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION",
    "AnswerSupervisionView",
    "AnswerUtilityExperimentConfig",
    "AnswerUtilityExperimentProfile",
    "AnswerUtilityExperimentVariant",
    "AnswerUtilityObjectiveConfig",
    "answer_utility_experiment_profile",
    "load_answer_utility_experiment_config",
]
