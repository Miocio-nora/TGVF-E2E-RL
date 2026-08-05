from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tgvf_rl.policy.horizon_extension import (
    POLICY_HORIZON_EXTENSION_SCHEMA,
    PolicyHorizonExtension,
    load_policy_horizon_extension,
    validate_policy_horizon_extension_resume,
)
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig


SHA = "a" * 64
COMMIT = "b" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _integrity(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["integrity_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return result


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) + b"\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> MagicMock:
    config = MagicMock(spec=PolicyE2ESmokeRunConfig)
    config.run_id = "policy-run"
    config.source_path = (tmp_path / "base.toml").resolve()
    config.source_path.write_text("immutable", encoding="utf-8")
    config.source_sha256 = "c" * 64
    config.identity_sha256 = "d" * 64
    config.output.root = (tmp_path / "output").resolve()
    config.output.metrics_path = config.output.root / "metrics.jsonl"
    config.training.resume_mode = "auto"
    config.training.maximum_optimizer_steps = 20
    config.training.checkpoint_steps = (0, 1, 5, 10, 20)
    config.scheduler.total_steps = 80
    return config


def _extension(config: MagicMock, tmp_path: Path, **changes) -> PolicyHorizonExtension:
    values = {
        "source_path": (tmp_path / "extension.json").resolve(),
        "source_sha256": "e" * 64,
        "extension_id": "step20-to80",
        "run_id": config.run_id,
        "base_config_path": config.source_path,
        "base_config_source_sha256": config.source_sha256,
        "base_run_identity_sha256": config.identity_sha256,
        "output_root": config.output.root,
        "source_optimizer_step": 20,
        "target_optimizer_step": 80,
        "scheduler_total_steps": 80,
        "effective_checkpoint_steps": (0, 1, 5, 10, 20, 30, 40, 60, 80),
        "metrics_prefix_sha256": SHA,
        "checkpoint_pair_file_sha256": SHA,
        "project_state_file_sha256": SHA,
        "latest_lora_pointer_file_sha256": SHA,
        "source_weights_sha256": SHA,
        "code_commit": COMMIT,
        "integrity_sha256": SHA,
    }
    values.update(changes)
    return PolicyHorizonExtension(**values)


def test_extension_static_contract_rejects_non_horizon_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _extension(config, tmp_path).validate_for_config(config)

    with pytest.raises(ValueError, match="base checkpoint"):
        _extension(
            config,
            tmp_path,
            effective_checkpoint_steps=(0, 1, 10, 20, 40, 80),
        ).validate_for_config(config)


def test_manifest_integrity_is_checked_before_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = {
        "schema_version": POLICY_HORIZON_EXTENSION_SCHEMA,
        "extension_id": "step20-to80",
        "run_id": config.run_id,
        "base_config_path": str(config.source_path),
        "base_config_source_sha256": config.source_sha256,
        "base_run_identity_sha256": config.identity_sha256,
        "output_root": str(config.output.root),
        "source_optimizer_step": 20,
        "target_optimizer_step": 80,
        "scheduler_total_steps": 80,
        "effective_checkpoint_steps": [0, 1, 5, 10, 20, 30, 40, 60, 80],
        "metrics_prefix_sha256": SHA,
        "checkpoint_pair_file_sha256": SHA,
        "project_state_file_sha256": SHA,
        "latest_lora_pointer_file_sha256": SHA,
        "source_weights_sha256": SHA,
        "code_commit": COMMIT,
    }
    path = tmp_path / "extension.json"
    _write_json(path, _integrity(payload))
    loaded = load_policy_horizon_extension(path, config, validate_artifacts=False)
    assert loaded.target_optimizer_step == 80

    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["target_optimizer_step"] = 79
    _write_json(path, broken)
    with pytest.raises(ValueError, match="integrity differs"):
        load_policy_horizon_extension(path, config, validate_artifacts=False)


@pytest.mark.parametrize("current_step", [20, 21, 80])
def test_resume_gate_accepts_intermediate_and_completed_checkpoint(
    tmp_path: Path, current_step: int
) -> None:
    config = _config(tmp_path)
    output = config.output.root
    checkpoint_root = output / "checkpoints"
    state_root = output / "runtime-policy-state"
    checkpoint_root.mkdir(parents=True)
    state_root.mkdir(parents=True)

    metric_lines = [
        _canonical({"optimizer_step": step}) for step in range(1, current_step + 1)
    ]
    config.output.metrics_path.write_bytes(b"\n".join(metric_lines) + b"\n")
    prefix_sha = hashlib.sha256(b"\n".join(metric_lines[:20]) + b"\n").hexdigest()
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        str(current_step), encoding="utf-8"
    )

    weights = SHA if current_step == 20 else "2" * 64

    def write_checkpoint(step: int, policy_weights: str) -> tuple[Path, Path]:
        actor = checkpoint_root / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        for stem in ("model", "optim", "extra_state"):
            for rank in range(4):
                (actor / f"{stem}_world_size_4_rank_{rank}.pt").write_bytes(b"state")
        project = _integrity(
            {
                "run_identity": {
                    "run_id": config.run_id,
                    "hashes": [
                        ["run_config", config.identity_sha256],
                        ["run_config_file", config.source_sha256],
                    ],
                },
                "policy_version": {
                    "run_id": config.run_id,
                    "optimizer_step": step,
                    "weights_sha256": policy_weights,
                },
            }
        )
        project_path = actor / "tgvf_policy_project_state.json"
        _write_json(project_path, project)
        pair = _integrity(
            {
                "run_id": config.run_id,
                "optimizer_step": step,
                "project_state_sha256": project["integrity_sha256"],
            }
        )
        pair_path = actor / "tgvf_policy_checkpoint_pair.json"
        _write_json(pair_path, pair)
        return pair_path, project_path

    source_pair, source_project = write_checkpoint(20, SHA)
    if current_step != 20:
        write_checkpoint(current_step, weights)

    tensor = state_root / "lora-snapshots/current.safetensors"
    tensor.parent.mkdir()
    tensor.write_bytes(b"lora")
    manifest = _integrity(
        {
            "run_id": config.run_id,
            "optimizer_step": current_step,
            "weights_sha256": weights,
            "tensor_file": "lora-snapshots/current.safetensors",
            "tensor_file_sha256": _sha(tensor),
        }
    )
    manifest_path = state_root / "lora-manifests/current.json"
    _write_json(manifest_path, manifest)
    pointer = _integrity(
        {
            "run_id": config.run_id,
            "optimizer_step": current_step,
            "weights_sha256": weights,
            "manifest_file": "lora-manifests/current.json",
        }
    )
    pointer_path = state_root / "latest-lora-snapshot.json"
    _write_json(pointer_path, pointer)
    extension = _extension(
        config,
        tmp_path,
        metrics_prefix_sha256=prefix_sha,
        checkpoint_pair_file_sha256=_sha(source_pair),
        project_state_file_sha256=_sha(source_project),
        latest_lora_pointer_file_sha256=(
            _sha(pointer_path) if current_step == 20 else SHA
        ),
    )

    assert validate_policy_horizon_extension_resume(extension, config) == current_step
