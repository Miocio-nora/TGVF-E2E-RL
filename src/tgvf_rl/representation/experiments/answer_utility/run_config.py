"""Strict single-GPU execution sidecar for the isolated utility experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - local legacy test environment
    import tomli as tomllib
from typing import Any


ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION = "answer-utility-run-config-v1"
ANSWER_UTILITY_RUN_SCOPE = "isolated_representation_answer_utility_single_gpu"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class AnswerUtilitySourceArtifactConfig:
    path: Path
    file_sha256: str
    manifest_sha256: str
    expected_run_identity_sha256: str
    expected_global_step: int

    def __post_init__(self) -> None:
        _absolute(self.path, name="source artifact path")
        for name in (
            "file_sha256",
            "manifest_sha256",
            "expected_run_identity_sha256",
        ):
            _sha(getattr(self, name), name=name)
        _positive_int(self.expected_global_step, name="expected_global_step")


@dataclass(frozen=True, slots=True)
class AnswerUtilityRunConfig:
    run_id: str
    experiment_config_path: Path
    experiment_config_sha256: str
    source_artifact: AnswerUtilitySourceArtifactConfig
    physical_gpu_id: int
    seed: int
    learning_rate: float
    target_optimizer_steps: int
    checkpoint_every_optimizer_steps: int
    log_every_optimizer_steps: int
    output_directory: Path
    resume_checkpoint_path: Path | None
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str
    schema_version: str = ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION
    scope: str = ANSWER_UTILITY_RUN_SCOPE

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError("answer-utility run schema mismatch")
        if self.scope != ANSWER_UTILITY_RUN_SCOPE:
            raise ValueError("answer-utility run scope mismatch")
        _text(self.run_id, name="run_id")
        for path, name in (
            (self.experiment_config_path, "experiment config path"),
            (self.output_directory, "output directory"),
            (self.source_path, "run config source path"),
        ):
            _absolute(path, name=name)
        if self.resume_checkpoint_path is not None:
            _absolute(self.resume_checkpoint_path, name="resume checkpoint path")
        for value, name in (
            (self.experiment_config_sha256, "experiment config SHA256"),
            (self.source_toml_sha256, "run TOML SHA256"),
            (self.canonical_config_sha256, "canonical run config SHA256"),
        ):
            _sha(value, name=name)
        if not isinstance(self.source_artifact, AnswerUtilitySourceArtifactConfig):
            raise TypeError("source_artifact must be typed")
        _nonnegative_int(self.physical_gpu_id, name="physical_gpu_id")
        _integer(self.seed, name="seed")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, float)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be an explicit positive float")
        for value, name in (
            (self.target_optimizer_steps, "target_optimizer_steps"),
            (
                self.checkpoint_every_optimizer_steps,
                "checkpoint_every_optimizer_steps",
            ),
            (self.log_every_optimizer_steps, "log_every_optimizer_steps"),
        ):
            _positive_int(value, name=name)
        if self.checkpoint_every_optimizer_steps > self.target_optimizer_steps:
            raise ValueError("checkpoint interval cannot exceed target steps")
        if self.log_every_optimizer_steps > self.target_optimizer_steps:
            raise ValueError("log interval cannot exceed target steps")
        if self.checkpoint_every_optimizer_steps % self.log_every_optimizer_steps:
            raise ValueError("every checkpoint step must also be a logged step")

    @property
    def checkpoint_directory(self) -> Path:
        return self.output_directory / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        return self.output_directory / "metrics.jsonl"

    @property
    def final_artifact_path(self) -> Path:
        return self.output_directory / "answer_utility_adapter.pt"

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "run_id": self.run_id,
            "experiment_config_path": str(self.experiment_config_path),
            "experiment_config_sha256": self.experiment_config_sha256,
            "source_artifact": {
                "path": str(self.source_artifact.path),
                "file_sha256": self.source_artifact.file_sha256,
                "manifest_sha256": self.source_artifact.manifest_sha256,
                "expected_run_identity_sha256": (
                    self.source_artifact.expected_run_identity_sha256
                ),
                "expected_global_step": self.source_artifact.expected_global_step,
            },
            "physical_gpu_id": self.physical_gpu_id,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "target_optimizer_steps": self.target_optimizer_steps,
            "checkpoint_every_optimizer_steps": (self.checkpoint_every_optimizer_steps),
            "log_every_optimizer_steps": self.log_every_optimizer_steps,
            "output_directory": str(self.output_directory),
            "resume_checkpoint_path": (
                None
                if self.resume_checkpoint_path is None
                else str(self.resume_checkpoint_path)
            ),
            "source_path": str(self.source_path),
            "source_toml_sha256": self.source_toml_sha256,
            "canonical_config_sha256": self.canonical_config_sha256,
        }


def load_answer_utility_run_config(path: str | Path) -> AnswerUtilityRunConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"answer-utility run config does not exist: {source}")
    raw = source.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid answer-utility run TOML: {error}") from error
    _fields(
        payload,
        {
            "schema_version",
            "scope",
            "run_id",
            "experiment",
            "source_artifact",
            "execution",
            "optimizer",
            "training",
            "output",
            "resume",
        },
        name="root",
    )
    experiment = _table(payload, "experiment", {"config_path", "config_sha256"})
    artifact = _table(
        payload,
        "source_artifact",
        {
            "path",
            "file_sha256",
            "manifest_sha256",
            "expected_run_identity_sha256",
            "expected_global_step",
        },
    )
    execution = _table(payload, "execution", {"physical_gpu_id", "seed"})
    optimizer = _table(payload, "optimizer", {"learning_rate"})
    training = _table(
        payload,
        "training",
        {
            "target_optimizer_steps",
            "checkpoint_every_optimizer_steps",
            "log_every_optimizer_steps",
        },
    )
    output = _table(payload, "output", {"directory"})
    resume = _table(payload, "resume", {"enabled", "checkpoint_path"})
    resume_enabled = _bool(resume, "enabled", table="resume")
    raw_resume_path = _string(resume, "checkpoint_path", table="resume")
    resume_path = None if raw_resume_path == "none" else Path(raw_resume_path)
    if resume_enabled != (resume_path is not None):
        raise ValueError("resume.enabled and checkpoint_path presence must agree")
    config = AnswerUtilityRunConfig(
        schema_version=_string(payload, "schema_version", table="root"),
        scope=_string(payload, "scope", table="root"),
        run_id=_string(payload, "run_id", table="root"),
        experiment_config_path=Path(
            _string(experiment, "config_path", table="experiment")
        ),
        experiment_config_sha256=_string(
            experiment, "config_sha256", table="experiment"
        ),
        source_artifact=AnswerUtilitySourceArtifactConfig(
            path=Path(_string(artifact, "path", table="source_artifact")),
            file_sha256=_string(artifact, "file_sha256", table="source_artifact"),
            manifest_sha256=_string(
                artifact, "manifest_sha256", table="source_artifact"
            ),
            expected_run_identity_sha256=_string(
                artifact,
                "expected_run_identity_sha256",
                table="source_artifact",
            ),
            expected_global_step=_int(
                artifact, "expected_global_step", table="source_artifact"
            ),
        ),
        physical_gpu_id=_int(execution, "physical_gpu_id", table="execution"),
        seed=_int(execution, "seed", table="execution"),
        learning_rate=_float(optimizer, "learning_rate", table="optimizer"),
        target_optimizer_steps=_int(
            training, "target_optimizer_steps", table="training"
        ),
        checkpoint_every_optimizer_steps=_int(
            training,
            "checkpoint_every_optimizer_steps",
            table="training",
        ),
        log_every_optimizer_steps=_int(
            training, "log_every_optimizer_steps", table="training"
        ),
        output_directory=Path(_string(output, "directory", table="output")),
        resume_checkpoint_path=resume_path,
        source_path=source,
        source_toml_sha256=sha256(raw).hexdigest(),
        canonical_config_sha256=_canonical_sha(payload),
    )
    _verify_bound_file(
        config.experiment_config_path,
        config.experiment_config_sha256,
        name="experiment config",
    )
    _verify_bound_file(
        config.source_artifact.path,
        config.source_artifact.file_sha256,
        name="source Adapter artifact",
    )
    return config


def _table(
    payload: Mapping[str, Any], key: str, expected: set[str]
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"[{key}] must be a table")
    _fields(value, expected, name=key)
    return value


def _fields(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"[{name}] fields differ: missing={missing} unknown={unknown}")


def _string(payload: Mapping[str, Any], key: str, *, table: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{table}.{key} must be non-empty text")
    return value


def _int(payload: Mapping[str, Any], key: str, *, table: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{table}.{key} must be an integer")
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


def _bool(payload: Mapping[str, Any], key: str, *, table: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise TypeError(f"{table}.{key} must be a boolean")
    return value


def _verify_bound_file(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {observed}")


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


def _integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _nonnegative_int(value: object, *, name: str) -> None:
    _integer(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _positive_int(value: object, *, name: str) -> None:
    _integer(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION",
    "ANSWER_UTILITY_RUN_SCOPE",
    "AnswerUtilityRunConfig",
    "AnswerUtilitySourceArtifactConfig",
    "load_answer_utility_run_config",
]
