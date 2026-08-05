"""Explicit, fail-closed horizon extension for one immutable Policy run.

The base run config remains byte-for-byte unchanged and therefore keeps the
same checkpoint/run identity.  A separately committed manifest may authorize
only a larger trainer stopping boundary and additional checkpoint boundaries;
it cannot change the actor scheduler or any scientific run setting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .run_config import PolicyE2ESmokeRunConfig


POLICY_HORIZON_EXTENSION_SCHEMA = "policy-horizon-extension-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def _commit(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError("extension code_commit must be a full Git commit")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("extension code_commit must be hexadecimal") from error
    return value.lower()


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path.resolve()


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checkpoint_steps(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("effective_checkpoint_steps must be an integer list")
    steps = tuple(value)
    if not steps or steps[0] != 0:
        raise ValueError("effective_checkpoint_steps must start at zero")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("effective_checkpoint_steps must increase strictly")
    return steps


def _read_json(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{owner} is not readable canonical JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} must contain one JSON object")
    return payload


def _verify_integrity(payload: Mapping[str, Any], *, owner: str) -> None:
    expected = _sha256(payload.get("integrity_sha256"), name=f"{owner} integrity")
    content = dict(payload)
    del content["integrity_sha256"]
    observed = hashlib.sha256(_canonical_json(content)).hexdigest()
    if observed != expected:
        raise ValueError(f"{owner} integrity differs")


@dataclass(frozen=True, slots=True)
class PolicyHorizonExtension:
    """One audited trainer-horizon extension of an immutable base config."""

    source_path: Path
    source_sha256: str
    extension_id: str
    run_id: str
    base_config_path: Path
    base_config_source_sha256: str
    base_run_identity_sha256: str
    output_root: Path
    source_optimizer_step: int
    target_optimizer_step: int
    scheduler_total_steps: int
    effective_checkpoint_steps: tuple[int, ...]
    metrics_prefix_sha256: str
    checkpoint_pair_file_sha256: str
    project_state_file_sha256: str
    latest_lora_pointer_file_sha256: str
    source_weights_sha256: str
    code_commit: str
    integrity_sha256: str

    def validate_for_config(self, config: PolicyE2ESmokeRunConfig) -> None:
        if not isinstance(config, PolicyE2ESmokeRunConfig):
            raise TypeError("config must be PolicyE2ESmokeRunConfig")
        if self.run_id != config.run_id:
            raise ValueError("horizon extension run_id differs from base config")
        if self.base_config_path != config.source_path.resolve():
            raise ValueError("horizon extension base config path differs")
        if self.base_config_source_sha256 != config.source_sha256:
            raise ValueError("horizon extension base config bytes differ")
        if self.base_run_identity_sha256 != config.identity_sha256:
            raise ValueError("horizon extension base run identity differs")
        if self.output_root != config.output.root.resolve():
            raise ValueError("horizon extension output root differs")
        if config.training.resume_mode != "auto":
            raise ValueError("horizon extension requires base resume_mode=auto")
        if self.source_optimizer_step != config.training.maximum_optimizer_steps:
            raise ValueError("horizon extension source is not the base final step")
        if self.target_optimizer_step <= self.source_optimizer_step:
            raise ValueError("horizon extension target must exceed its source")
        if self.scheduler_total_steps != config.scheduler.total_steps:
            raise ValueError("horizon extension changes the actor scheduler horizon")
        if self.target_optimizer_step != self.scheduler_total_steps:
            raise ValueError("horizon extension must end at the existing scheduler horizon")
        if self.effective_checkpoint_steps[-1] != self.target_optimizer_step:
            raise ValueError("horizon extension checkpoint plan lacks its target")
        base_steps = config.training.checkpoint_steps
        if self.effective_checkpoint_steps[: len(base_steps)] != base_steps:
            raise ValueError("horizon extension changes base checkpoint boundaries")
        added = self.effective_checkpoint_steps[len(base_steps) :]
        if not added or any(step <= self.source_optimizer_step for step in added):
            raise ValueError("horizon extension adds a checkpoint at/before its source")

    @property
    def environment(self) -> dict[str, str]:
        return {
            "TGVF_POLICY_HORIZON_EXTENSION_PATH": str(self.source_path),
            "TGVF_POLICY_HORIZON_EXTENSION_SHA256": self.source_sha256,
        }


def load_policy_horizon_extension(
    path: str | Path,
    config: PolicyE2ESmokeRunConfig,
    *,
    validate_artifacts: bool = True,
) -> PolicyHorizonExtension:
    """Load an integrity-bound extension and optionally gate its resume state."""

    source = Path(path).resolve()
    payload = _read_json(source, owner="Policy horizon extension")
    if payload.get("schema_version") != POLICY_HORIZON_EXTENSION_SCHEMA:
        raise ValueError("Policy horizon extension schema differs")
    _verify_integrity(payload, owner="Policy horizon extension")
    allowed = {
        "schema_version",
        "extension_id",
        "run_id",
        "base_config_path",
        "base_config_source_sha256",
        "base_run_identity_sha256",
        "output_root",
        "source_optimizer_step",
        "target_optimizer_step",
        "scheduler_total_steps",
        "effective_checkpoint_steps",
        "metrics_prefix_sha256",
        "checkpoint_pair_file_sha256",
        "project_state_file_sha256",
        "latest_lora_pointer_file_sha256",
        "source_weights_sha256",
        "code_commit",
        "integrity_sha256",
    }
    if set(payload) != allowed:
        raise ValueError(
            "Policy horizon extension fields differ: "
            f"missing={sorted(allowed.difference(payload))}, "
            f"extra={sorted(set(payload).difference(allowed))}"
        )
    extension_id = payload["extension_id"]
    run_id = payload["run_id"]
    if not isinstance(extension_id, str) or not extension_id.strip():
        raise ValueError("extension_id must be non-empty")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("extension run_id must be non-empty")
    extension = PolicyHorizonExtension(
        source_path=source,
        source_sha256=_file_sha256(source),
        extension_id=extension_id,
        run_id=run_id,
        base_config_path=_absolute_path(payload["base_config_path"], name="base_config_path"),
        base_config_source_sha256=_sha256(
            payload["base_config_source_sha256"], name="base config source"
        ),
        base_run_identity_sha256=_sha256(
            payload["base_run_identity_sha256"], name="base run identity"
        ),
        output_root=_absolute_path(payload["output_root"], name="output_root"),
        source_optimizer_step=_positive_int(
            payload["source_optimizer_step"], name="source_optimizer_step"
        ),
        target_optimizer_step=_positive_int(
            payload["target_optimizer_step"], name="target_optimizer_step"
        ),
        scheduler_total_steps=_positive_int(
            payload["scheduler_total_steps"], name="scheduler_total_steps"
        ),
        effective_checkpoint_steps=_checkpoint_steps(
            payload["effective_checkpoint_steps"]
        ),
        metrics_prefix_sha256=_sha256(
            payload["metrics_prefix_sha256"], name="metrics prefix"
        ),
        checkpoint_pair_file_sha256=_sha256(
            payload["checkpoint_pair_file_sha256"], name="checkpoint pair file"
        ),
        project_state_file_sha256=_sha256(
            payload["project_state_file_sha256"], name="project state file"
        ),
        latest_lora_pointer_file_sha256=_sha256(
            payload["latest_lora_pointer_file_sha256"], name="latest LoRA pointer file"
        ),
        source_weights_sha256=_sha256(
            payload["source_weights_sha256"], name="source weights"
        ),
        code_commit=_commit(payload["code_commit"]),
        integrity_sha256=_sha256(
            payload["integrity_sha256"], name="extension integrity"
        ),
    )
    extension.validate_for_config(config)
    if validate_artifacts:
        validate_policy_horizon_extension_resume(extension, config)
    return extension


def policy_horizon_extension_from_environment(
    config: PolicyE2ESmokeRunConfig,
) -> PolicyHorizonExtension | None:
    """Read the optional extension only from a path+file-hash environment pair."""

    raw_path = os.environ.get("TGVF_POLICY_HORIZON_EXTENSION_PATH")
    raw_sha = os.environ.get("TGVF_POLICY_HORIZON_EXTENSION_SHA256")
    if raw_path is None and raw_sha is None:
        return None
    if not raw_path or not raw_sha:
        raise RuntimeError("Policy horizon extension environment pair is incomplete")
    path = Path(raw_path).resolve()
    expected = _sha256(raw_sha, name="extension environment file")
    if _file_sha256(path) != expected:
        raise RuntimeError("Policy horizon extension environment file differs")
    return load_policy_horizon_extension(path, config, validate_artifacts=True)


def validate_policy_horizon_extension_resume(
    extension: PolicyHorizonExtension,
    config: PolicyE2ESmokeRunConfig,
) -> int:
    """Validate the current exact recovery boundary and return its step."""

    extension.validate_for_config(config)
    output = extension.output_root
    checkpoint_root = output / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    try:
        current_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise ValueError("horizon extension checkpoint tracker is unavailable") from error
    if not extension.source_optimizer_step <= current_step <= extension.target_optimizer_step:
        raise ValueError("horizon extension current checkpoint is outside its interval")

    metrics_path = config.output.metrics_path
    try:
        metric_lines = [line for line in metrics_path.read_bytes().splitlines() if line]
        metrics = [json.loads(line) for line in metric_lines]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("horizon extension metrics are unreadable") from error
    steps = [row.get("optimizer_step") for row in metrics]
    if steps != list(range(1, current_step + 1)):
        raise ValueError("horizon extension metrics are not the exact contiguous prefix")
    prefix = b"\n".join(metric_lines[: extension.source_optimizer_step]) + b"\n"
    if hashlib.sha256(prefix).hexdigest() != extension.metrics_prefix_sha256:
        raise ValueError("horizon extension original metrics prefix differs")

    source_actor = (
        checkpoint_root / f"global_step_{extension.source_optimizer_step}" / "actor"
    )
    source_pair = source_actor / "tgvf_policy_checkpoint_pair.json"
    source_project = source_actor / "tgvf_policy_project_state.json"
    if _file_sha256(source_pair) != extension.checkpoint_pair_file_sha256:
        raise ValueError("horizon extension source checkpoint pair differs")
    if _file_sha256(source_project) != extension.project_state_file_sha256:
        raise ValueError("horizon extension source project state differs")

    actor = checkpoint_root / f"global_step_{current_step}" / "actor"
    for stem in ("model", "optim", "extra_state"):
        shards = sorted(actor.glob(f"{stem}_world_size_4_rank_*.pt"))
        if len(shards) != 4 or any(not path.is_file() or path.stat().st_size == 0 for path in shards):
            raise ValueError(f"horizon extension current {stem} shard set is incomplete")
    pair = _read_json(actor / "tgvf_policy_checkpoint_pair.json", owner="checkpoint pair")
    project = _read_json(
        actor / "tgvf_policy_project_state.json", owner="project checkpoint"
    )
    _verify_integrity(pair, owner="checkpoint pair")
    _verify_integrity(project, owner="project checkpoint")
    if pair.get("run_id") != config.run_id or pair.get("optimizer_step") != current_step:
        raise ValueError("horizon extension current checkpoint pair identity differs")
    if pair.get("project_state_sha256") != project.get("integrity_sha256"):
        raise ValueError("horizon extension checkpoint pair/project digest differs")
    run_identity = project.get("run_identity")
    if not isinstance(run_identity, dict) or run_identity.get("run_id") != config.run_id:
        raise ValueError("horizon extension project run identity differs")
    hashes = dict(run_identity.get("hashes", []))
    if hashes.get("run_config") != config.identity_sha256:
        raise ValueError("horizon extension project config identity differs")
    if hashes.get("run_config_file") != config.source_sha256:
        raise ValueError("horizon extension project config source differs")
    policy = project.get("policy_version")
    if not isinstance(policy, dict) or policy.get("optimizer_step") != current_step:
        raise ValueError("horizon extension project policy step differs")
    weights = _sha256(policy.get("weights_sha256"), name="project policy weights")
    if current_step == extension.source_optimizer_step and weights != extension.source_weights_sha256:
        raise ValueError("horizon extension source policy weights differ")

    state_root = output / "runtime-policy-state"
    pointer_path = state_root / "latest-lora-snapshot.json"
    if current_step == extension.source_optimizer_step and (
        _file_sha256(pointer_path) != extension.latest_lora_pointer_file_sha256
    ):
        raise ValueError("horizon extension source latest LoRA pointer differs")
    pointer = _read_json(pointer_path, owner="latest LoRA pointer")
    _verify_integrity(pointer, owner="latest LoRA pointer")
    if (
        pointer.get("run_id") != config.run_id
        or pointer.get("optimizer_step") != current_step
        or pointer.get("weights_sha256") != weights
    ):
        raise ValueError("horizon extension latest LoRA identity differs")
    manifest_path = state_root / str(pointer.get("manifest_file", ""))
    manifest = _read_json(manifest_path, owner="LoRA manifest")
    _verify_integrity(manifest, owner="LoRA manifest")
    if (
        manifest.get("run_id") != config.run_id
        or manifest.get("optimizer_step") != current_step
        or manifest.get("weights_sha256") != weights
    ):
        raise ValueError("horizon extension LoRA manifest identity differs")
    tensor = state_root / str(manifest.get("tensor_file", ""))
    if not tensor.is_file() or _file_sha256(tensor) != manifest.get("tensor_file_sha256"):
        raise ValueError("horizon extension LoRA tensor differs")
    return current_step


__all__ = [
    "POLICY_HORIZON_EXTENSION_SCHEMA",
    "PolicyHorizonExtension",
    "load_policy_horizon_extension",
    "policy_horizon_extension_from_environment",
    "validate_policy_horizon_extension_resume",
]
