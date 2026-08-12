#!/usr/bin/env python3
"""Materialize an integrity-bound continuation from a completed Policy step.

The generated manifest changes only the trainer stopping/checkpoint horizon.
All scientific settings remain owned by the immutable base run config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tgvf_rl.policy.horizon_extension import load_policy_horizon_extension
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, owner: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{owner} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} is not a JSON object: {path}")
    return value


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40:
        raise RuntimeError("Git HEAD is not a full commit identity")
    return commit


def _metrics_prefix_sha256(path: Path, *, source_step: int) -> str:
    lines = [line for line in path.read_bytes().splitlines() if line]
    if len(lines) != source_step:
        raise RuntimeError(
            "continuation requires metrics to end exactly at its source step"
        )
    steps: list[object] = []
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("metrics row is not a JSON object")
        steps.append(value.get("optimizer_step"))
    if steps != list(range(1, source_step + 1)):
        raise RuntimeError("metrics are not the exact contiguous source prefix")
    return hashlib.sha256(b"\n".join(lines) + b"\n").hexdigest()


def materialize(
    *,
    run_config_path: Path,
    output_path: Path,
    extension_id: str,
    target_step: int,
    repository: Path,
) -> dict[str, object]:
    run_config_path = run_config_path.resolve()
    output_path = output_path.resolve()
    repository = repository.resolve()
    config = load_policy_e2e_smoke_run_config(
        run_config_path, allow_external_agent_loop_config=True
    )
    source_step = config.training.maximum_optimizer_steps
    if target_step <= source_step:
        raise ValueError("continuation target must exceed the base final step")

    if output_path.is_file():
        loaded = load_policy_horizon_extension(
            output_path, config, validate_artifacts=True
        )
        if loaded.extension_id != extension_id or loaded.target_optimizer_step != target_step:
            raise RuntimeError("existing continuation manifest identity differs")
        return _read_json(output_path, owner="existing continuation manifest")

    tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
    try:
        observed_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("base checkpoint tracker is unavailable") from error
    if observed_step != source_step:
        raise RuntimeError("base run has not stopped exactly at its source step")

    actor = (
        config.output.checkpoint_directory
        / f"global_step_{source_step}"
        / "actor"
    )
    pair_path = actor / "tgvf_policy_checkpoint_pair.json"
    project_path = actor / "tgvf_policy_project_state.json"
    pointer_path = config.output.root / "runtime-policy-state/latest-lora-snapshot.json"
    pair = _read_json(pair_path, owner="source checkpoint pair")
    project = _read_json(project_path, owner="source project state")
    pointer = _read_json(pointer_path, owner="source Adapter pointer")
    if pair.get("optimizer_step") != source_step:
        raise RuntimeError("source checkpoint pair step differs")
    policy = project.get("policy_version")
    if not isinstance(policy, dict) or policy.get("optimizer_step") != source_step:
        raise RuntimeError("source project policy step differs")
    if pointer.get("optimizer_step") != source_step:
        raise RuntimeError("source Adapter pointer step differs")
    weights_sha256 = policy.get("weights_sha256")
    if not isinstance(weights_sha256, str) or len(weights_sha256) != 64:
        raise RuntimeError("source policy weights identity is malformed")
    if pointer.get("weights_sha256") != weights_sha256:
        raise RuntimeError("source Qwen/Adapter checkpoint pair is not closed")

    base_steps = tuple(config.training.checkpoint_steps)
    effective_steps = (*base_steps, *range(source_step + 1, target_step + 1))
    content: dict[str, object] = {
        "schema_version": "policy-horizon-extension-v1",
        "extension_id": extension_id,
        "run_id": config.run_id,
        "base_config_path": str(run_config_path),
        "base_config_source_sha256": config.source_sha256,
        "base_run_identity_sha256": config.identity_sha256,
        "output_root": str(config.output.root.resolve()),
        "source_optimizer_step": source_step,
        "target_optimizer_step": target_step,
        "scheduler_total_steps": config.scheduler.total_steps,
        "effective_checkpoint_steps": list(effective_steps),
        "metrics_prefix_sha256": _metrics_prefix_sha256(
            config.output.metrics_path, source_step=source_step
        ),
        "checkpoint_pair_file_sha256": _sha256_file(pair_path),
        "project_state_file_sha256": _sha256_file(project_path),
        "latest_lora_pointer_file_sha256": _sha256_file(pointer_path),
        "source_weights_sha256": weights_sha256,
        "code_commit": _git_commit(repository),
    }
    payload = {
        **content,
        "integrity_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    temporary.replace(output_path)
    load_policy_horizon_extension(output_path, config, validate_artifacts=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-id", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = materialize(
        run_config_path=args.run_config,
        output_path=args.output,
        extension_id=args.extension_id,
        target_step=args.target_step,
        repository=args.repository,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
