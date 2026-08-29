#!/usr/bin/env python3
"""Fail-closed artifact and resource checks for the PRL-26 A-to-B handoff."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


_SOURCE_RECEIPT_SCHEMA = "tgvf.prl15-permanent-checkpoint-receipt.v1"
_AUTHORIZATION_SCHEMA = "tgvf.prl26-train512-crop-fresh-authorization.v1"
_METRICS_SCHEMA = "policy-pilot-v1-metrics-event-v1"
_EXPECTED_GPU_IDS = tuple(range(8))


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _read_json(path: Path, owner: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{owner} is unreadable: {error}") from error


def _read_json_lines(path: Path, owner: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"{owner} is unreadable: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        _fail(f"{owner} is empty or contains an empty record")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(
                f"{owner} line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            _fail(f"{owner} line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def _require_finite_numbers(value: object, owner: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{owner} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_numbers(item, f"{owner}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{owner}[{index}]")


def _validate_metrics_rows(
    rows: Sequence[Mapping[str, object]], *, expected_step: int, owner: str
) -> None:
    if len(rows) != expected_step:
        _fail(f"{owner} must contain exactly {expected_step} records")
    observed_steps: list[int] = []
    for index, row in enumerate(rows, start=1):
        _require_finite_numbers(row, f"{owner}[{index}]")
        step = row.get("optimizer_step")
        if type(step) is not int:
            _fail(f"{owner} record {index} has a malformed optimizer step")
        observed_steps.append(step)
        if row.get("schema_version") != _METRICS_SCHEMA:
            _fail(f"{owner} record {index} has an unexpected schema")
        for scope in ("step", "cumulative", "timing"):
            if not isinstance(row.get(scope), Mapping):
                _fail(f"{owner} record {index} omits {scope}")
        cumulative = row["cumulative"]
        assert isinstance(cumulative, Mapping)
        if cumulative.get("optimizer_steps") != step:
            _fail(f"{owner} cumulative optimizer step differs at record {index}")
    if observed_steps != list(range(1, expected_step + 1)):
        _fail(f"{owner} optimizer steps are not exact and contiguous")


def _validate_supervisor_events(
    rows: Sequence[Mapping[str, object]],
    *,
    event_directory: Path,
    target_step: int,
) -> dict[str, object]:
    if not rows:
        _fail("NoTool supervisor event stream is empty")
    paired_attempts: list[
        tuple[int, int, Mapping[str, object], Mapping[str, object]]
    ] = []
    active: tuple[int, Mapping[str, object]] | None = None
    used_logs: set[Path] = set()
    for record_number, row in enumerate(rows, start=1):
        event = row.get("event")
        attempt = row.get("attempt")
        if type(attempt) is not int or attempt <= 0:
            _fail(f"NoTool supervisor event {record_number} has a bad attempt")
        if event == "attempt_started":
            if active is not None:
                _fail("NoTool supervisor has an unfinished attempt")
            before = row.get("checkpoint_step")
            if type(before) is not int or before < 0 or before > target_step:
                _fail("NoTool supervisor attempt starts from an ambiguous step")
            active = (record_number, row)
            continue
        if event != "attempt_finished" or active is None:
            _fail(f"NoTool supervisor event {record_number} is out of sequence")
        start_record, started = active
        if attempt != started.get("attempt"):
            _fail("NoTool supervisor finish attempt identity differs")
        before = row.get("checkpoint_step_before")
        after = row.get("checkpoint_step_after")
        return_code = row.get("return_code")
        if (
            type(before) is not int
            or type(after) is not int
            or type(return_code) is not int
            or before != started.get("checkpoint_step")
            or after < before
            or after > target_step
        ):
            _fail("NoTool supervisor finish boundary is malformed")
        start_log = started.get("log_path")
        finish_log = row.get("log_path")
        if start_log != finish_log or not isinstance(finish_log, str):
            _fail("NoTool supervisor attempt log identity differs")
        attempt_log = Path(finish_log)
        try:
            attempt_log.relative_to(event_directory)
        except ValueError as error:
            raise RuntimeError("NoTool attempt log escaped its control root") from error
        if (
            attempt_log.is_symlink()
            or not attempt_log.is_file()
            or attempt_log.stat().st_size <= 0
        ):
            _fail("NoTool supervisor attempt log is absent or empty")
        resolved_log = attempt_log.resolve()
        if resolved_log in used_logs:
            _fail("NoTool supervisor reused an attempt log")
        used_logs.add(resolved_log)
        decision = row.get("decision")
        retry_decisions = {
            "retry_weight_wake_oom",
            "retry_judge_transient_429",
        }
        if decision not in {"complete", *retry_decisions, "fail"}:
            _fail("NoTool supervisor decision is malformed")
        if decision in retry_decisions and return_code == 0:
            _fail("NoTool supervisor retry has an impossible zero return code")
        if decision == "fail" and return_code == 0:
            _fail("NoTool supervisor failure has an impossible zero return code")
        if decision == "complete" and (return_code != 0 or after != target_step):
            _fail("NoTool supervisor completion boundary is malformed")
        if decision != "complete" and after >= target_step:
            _fail("NoTool supervisor non-completion reached the target step")
        paired_attempts.append((start_record, record_number, started, row))
        active = None
    if active is not None:
        _fail("NoTool supervisor has an unfinished attempt")
    if not paired_attempts:
        _fail("NoTool supervisor event stream has no finished attempt")

    # The supervisor schema predates an explicit invocation ID. Its attempt
    # counter is process-local, so a reset to attempt 1 after a closed pair is
    # the only auditable invocation boundary; increasing attempts stay within
    # one invocation and are legal only after the classified retry decision.
    invocations: list[dict[str, object]] = []
    previous_attempt: int | None = None
    previous_finish: Mapping[str, object] | None = None
    current_invocation: dict[str, object] | None = None
    for start_record, finish_record, started, finished in paired_attempts:
        attempt = started["attempt"]
        before = started["checkpoint_step"]
        after = finished["checkpoint_step_after"]
        assert type(attempt) is int
        assert type(before) is int
        assert type(after) is int
        if previous_finish is None:
            if attempt != 1 or before != 0:
                _fail(
                    "NoTool supervisor first invocation must start at attempt 1, step 0"
                )
            current_invocation = {
                "invocation": 1,
                "event_record_start": start_record,
                "checkpoint_step_before": before,
                "attempts": 0,
            }
            invocations.append(current_invocation)
        else:
            previous_after = previous_finish["checkpoint_step_after"]
            previous_decision = previous_finish["decision"]
            if before != previous_after:
                _fail("NoTool supervisor checkpoint chain has a gap or overlap")
            if previous_decision == "complete":
                _fail("NoTool supervisor continued after a successful invocation")
            if attempt == 1:
                if previous_decision != "fail":
                    _fail(
                        "NoTool supervisor began a new invocation without a prior "
                        "terminal failure"
                    )
                current_invocation = {
                    "invocation": len(invocations) + 1,
                    "event_record_start": start_record,
                    "checkpoint_step_before": before,
                    "attempts": 0,
                }
                invocations.append(current_invocation)
            else:
                if (
                    current_invocation is None
                    or previous_attempt is None
                    or attempt != previous_attempt + 1
                ):
                    _fail("NoTool supervisor attempts are not sequential")
                if previous_decision not in {
                    "retry_weight_wake_oom",
                    "retry_judge_transient_429",
                }:
                    _fail(
                        "NoTool supervisor continued an invocation after a terminal decision"
                    )
        assert current_invocation is not None
        current_invocation["attempts"] = int(current_invocation["attempts"]) + 1
        current_invocation["event_record_end"] = finish_record
        current_invocation["checkpoint_step_after"] = after
        current_invocation["terminal_return_code"] = finished["return_code"]
        current_invocation["terminal_decision"] = finished["decision"]
        previous_attempt = attempt
        previous_finish = finished

    final = paired_attempts[-1][3]
    if (
        final.get("decision") != "complete"
        or final.get("return_code") != 0
        or final.get("checkpoint_step_after") != target_step
    ):
        _fail("NoTool supervisor did not finish S32 with return code zero")
    return {
        "invocation_count": len(invocations),
        "attempts": len(paired_attempts),
        "invocations": invocations,
        "final_return_code": 0,
        "final_decision": "complete",
        "final_checkpoint_step": target_step,
    }


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _required_checkpoint_files(actor: Path, generation: Path, world_size: int) -> list[Path]:
    required = [generation / "data.pt", actor / "fsdp_config.json"]
    for prefix in ("model", "optim", "extra_state"):
        required.extend(
            actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        )
    return required


def _validate_generation(
    config: object,
    generation: Path,
    *,
    optimizer_step: int,
) -> tuple[object, object, list[Path]]:
    from tgvf_rl.framework.verl.checkpoint_bridge import (
        read_committed_policy_checkpoint_pair,
    )
    from tgvf_rl.framework.verl.compatibility import FSDP2BridgeConfig
    from tgvf_rl.framework.verl.policy_task_runner import _run_identity

    world_size = config.distributed.world_size
    actor = generation / "actor"
    state, pair = read_committed_policy_checkpoint_pair(
        actor,
        fsdp2=FSDP2BridgeConfig(world_size=world_size, fsdp_size=world_size),
    )
    if state.run_identity != _run_identity(config):
        _fail(f"{generation} belongs to a different run identity")
    if (
        state.progress.optimizer_step != optimizer_step
        or pair.optimizer_step != optimizer_step
        or state.metrics_state.optimizer_steps != optimizer_step
    ):
        _fail(f"{generation} optimizer-step owners differ")
    fsdp_path = actor / "fsdp_config.json"
    if _read_json(fsdp_path, "FSDP config") != {
        "FSDP_version": 2,
        "world_size": world_size,
    }:
        _fail(f"{generation} FSDP contract differs")
    required = _required_checkpoint_files(actor, generation, world_size)
    for path in required:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            _fail(f"checkpoint component is absent, empty, or a symlink: {path}")
    return state, pair, required


def _load_config(path: Path) -> object:
    from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

    return load_policy_e2e_smoke_run_config(path.resolve())


def _canonical_tracker_step(path: Path, owner: str) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        step = int(raw)
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"{owner} tracker is unreadable") from error
    if step <= 0 or raw != str(step):
        _fail(f"{owner} tracker is not a canonical positive integer")
    return step


def validate_source_completion(
    *, config_path: Path, events_path: Path, target_step: int
) -> dict[str, object]:
    config = _load_config(config_path)
    tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
    if _canonical_tracker_step(tracker, "NoTool") != target_step:
        _fail("NoTool tracker is not exactly S32")
    event_rows = _read_json_lines(events_path, "NoTool supervisor events")
    event_audit = _validate_supervisor_events(
        event_rows,
        event_directory=events_path.parent.resolve(),
        target_step=target_step,
    )
    metrics_rows = _read_json_lines(config.output.metrics_path, "NoTool metrics")
    _validate_metrics_rows(
        metrics_rows, expected_step=target_step, owner="NoTool metrics"
    )
    rolling = config.output.checkpoint_directory / f"global_step_{target_step}"
    permanent = config.output.root / "permanent-checkpoints" / f"global_step_{target_step}"
    rolling_state, rolling_pair, rolling_files = _validate_generation(
        config, rolling, optimizer_step=target_step
    )
    permanent_state, permanent_pair, permanent_files = _validate_generation(
        config, permanent, optimizer_step=target_step
    )
    if rolling_state != permanent_state or rolling_pair != permanent_pair:
        _fail("NoTool permanent and tracker-selected checkpoint owners differ")
    for rolling_file, permanent_file in zip(rolling_files, permanent_files, strict=True):
        if not os.path.samefile(rolling_file, permanent_file):
            _fail("NoTool permanent checkpoint is not the retained rolling generation")
    receipt_path = permanent / "tgvf_permanent_checkpoint_receipt.json"
    receipt = _read_json(receipt_path, "NoTool permanent receipt")
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "optimizer_step",
        "run_identity_sha256",
        "project_state_sha256",
        "pair_integrity_sha256",
    }:
        _fail("NoTool permanent receipt fields differ")
    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(rolling_state.run_identity.to_checkpoint_mapping())
    ).hexdigest()
    expected_receipt = {
        "schema_version": _SOURCE_RECEIPT_SCHEMA,
        "optimizer_step": target_step,
        "run_identity_sha256": identity_sha256,
        "project_state_sha256": rolling_state.integrity_sha256,
        "pair_integrity_sha256": rolling_pair.integrity_sha256,
    }
    if receipt != expected_receipt:
        _fail("NoTool permanent receipt integrity differs")
    return {
        "schema_version": "tgvf.prl26-notool-source-completion-audit.v1",
        "status": "accepted",
        "run_id": config.run_id,
        "run_config_identity_sha256": config.identity_sha256,
        "run_config_file_sha256": config.source_sha256,
        "optimizer_step": target_step,
        "metrics_records": len(metrics_rows),
        "project_state_sha256": rolling_state.integrity_sha256,
        "pair_integrity_sha256": rolling_pair.integrity_sha256,
        "permanent_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "supervisor": event_audit,
    }


def validate_contracts(
    *, training_repository: Path, notool_config_path: Path, crop_config_path: Path
) -> dict[str, object]:
    notool = _load_config(notool_config_path)
    crop = _load_config(crop_config_path)
    expected_notool_root = Path(
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-26-A-train512-s32-parity-notool-qwen3-instruct-bs16-n16-"
        "teacher25-ws8"
    )
    expected_crop_root = Path(
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-26-B-train512-s32-parity-crop-qwen3-instruct-bs16-n16-"
        "teacher25-ws8"
    )
    for arm, config, root in (
        ("NoTool", notool, expected_notool_root),
        ("Crop", crop, expected_crop_root),
    ):
        if (
            config.output.root != root
            or config.output.checkpoint_directory != root / "checkpoints"
            or config.output.metrics_path != root / "metrics.jsonl"
            or config.policy.image_max_pixels != 512 * 512
            or config.distributed.world_size != 8
            or config.distributed.physical_gpu_ids != _EXPECTED_GPU_IDS
            or config.training.maximum_optimizer_steps != 32
            or config.training.checkpoint_steps != (0, 8, 16, 24, 32)
            or config.training.permanent_checkpoint_steps != (8, 16, 24, 32)
            or config.training.resume_mode != "auto"
            or config.training.resume_from_path is not None
        ):
            _fail(f"PRL-26 {arm} formal contract differs")
        try:
            root.relative_to(training_repository.resolve())
        except ValueError:
            pass
        else:
            _fail(f"PRL-26 {arm} output is inside the clean training worktree")
    if notool.protocol.enabled_tool_names or notool.policy.sampling.stop_strings:
        _fail("PRL-26 NoTool protocol differs")
    if (
        crop.protocol.enabled_tool_names != ("image_zoom_in_tool",)
        or crop.protocol.maximum_tool_calls != 6
        or crop.policy.sampling.stop_strings != ("</tool_call>",)
        or crop.policy.sampling.include_stop_str_in_output is not True
    ):
        _fail("PRL-26 Crop fixed action-boundary contract differs")
    return {
        "schema_version": "tgvf.prl26-train512-handoff-contract-audit.v1",
        "status": "accepted",
        "notool_run_id": notool.run_id,
        "notool_run_config_identity_sha256": notool.identity_sha256,
        "crop_run_id": crop.run_id,
        "crop_run_config_identity_sha256": crop.identity_sha256,
        "crop_output_root": str(crop.output.root),
    }


def _authorization_payload(config: object, *, training_head: str) -> dict[str, object]:
    return {
        "schema_version": _AUTHORIZATION_SCHEMA,
        "status": "fresh-root-authorized",
        "run_id": config.run_id,
        "run_config_identity_sha256": config.identity_sha256,
        "run_config_file_sha256": config.source_sha256,
        "output_root": str(config.output.root),
        "training_repository_head": training_head,
        "root_absent_at_authorization": True,
    }


def _validate_authorization(
    value: object, config: object, *, training_head: str
) -> None:
    if not isinstance(value, dict):
        _fail("Crop fresh-root authorization is malformed")
    authorized_at = value.get("authorized_at_utc")
    content = {key: item for key, item in value.items() if key != "authorized_at_utc"}
    if not isinstance(authorized_at, str) or content != _authorization_payload(
        config, training_head=training_head
    ):
        _fail("Crop fresh-root authorization identity differs")


def target_launch_ready(
    *, config_path: Path, authorization_path: Path, training_head: str
) -> dict[str, object]:
    config = _load_config(config_path)
    root = config.output.root
    root_exists = os.path.lexists(root)
    authorization_exists = os.path.lexists(authorization_path)
    if root_exists:
        if root.is_symlink() or not root.is_dir():
            _fail("Crop output root is not a real directory")
        if not authorization_exists:
            _fail("Crop root exists without this handoff's fresh-root authorization")
        authorization = _read_json(authorization_path, "Crop fresh-root authorization")
        _validate_authorization(authorization, config, training_head=training_head)
        tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
        step = _canonical_tracker_step(tracker, "Crop")
        if step > config.training.maximum_optimizer_steps:
            _fail("Crop tracker exceeds its formal horizon")
        _validate_generation(
            config,
            config.output.checkpoint_directory / f"global_step_{step}",
            optimizer_step=step,
        )
        metrics_rows = _read_json_lines(config.output.metrics_path, "Crop metrics")
        _validate_metrics_rows(metrics_rows, expected_step=step, owner="Crop metrics")
        mode = "resume"
    else:
        if authorization_exists:
            authorization = _read_json(
                authorization_path, "Crop fresh-root authorization"
            )
            _validate_authorization(authorization, config, training_head=training_head)
        else:
            authorization_path.parent.mkdir(parents=True, exist_ok=True)
            authorization = _authorization_payload(config, training_head=training_head)
            authorization["authorized_at_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                with authorization_path.open("x", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            authorization,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as error:
                raise RuntimeError(
                    "Crop fresh-root authorization raced another handoff"
                ) from error
        if os.path.lexists(root):
            _fail("Crop output root appeared during fresh-root authorization")
        step = 0
        mode = "fresh"
    return {
        "schema_version": "tgvf.prl26-crop-launch-readiness.v1",
        "status": "accepted",
        "launch_mode": mode,
        "run_id": config.run_id,
        "run_config_identity_sha256": config.identity_sha256,
        "output_root": str(root),
        "checkpointed_step": step,
        "authorization_path": str(authorization_path),
    }


def _ray_processes() -> list[dict[str, object]]:
    markers = (
        "/ray/core/src/ray/gcs/gcs_server",
        "/ray/core/src/ray/raylet/raylet",
        "/ray/_private/",
        "/ray/dashboard/",
        "ray::",
    )
    found: list[dict[str, object]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if any(marker in command for marker in markers):
            found.append({"pid": int(entry.name), "command": command[:300]})
    return sorted(found, key=lambda item: int(item["pid"]))


def resource_release_audit(*, memory_threshold_mib: int) -> dict[str, object]:
    if memory_threshold_mib < 0:
        _fail("GPU memory threshold must be non-negative")
    try:
        memory_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        apps_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"nvidia-smi resource audit failed: {error}") from error
    memory: dict[int, int] = {}
    for line in memory_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            _fail("nvidia-smi memory output is malformed")
        try:
            gpu_id, used_mib = (int(field) for field in fields)
        except ValueError as error:
            raise RuntimeError("nvidia-smi memory output is malformed") from error
        if gpu_id in memory:
            _fail("nvidia-smi repeated a GPU")
        memory[gpu_id] = used_mib
    if tuple(sorted(memory)) != _EXPECTED_GPU_IDS:
        _fail("nvidia-smi did not report exactly GPUs 0-7")
    active_pids: list[int] = []
    for line in apps_result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            active_pids.append(int(value))
        except ValueError as error:
            raise RuntimeError("nvidia-smi compute-app output is malformed") from error
    ray_processes = _ray_processes()
    released = (
        not active_pids
        and not ray_processes
        and all(value <= memory_threshold_mib for value in memory.values())
    )
    result = {
        "schema_version": "tgvf.prl26-gpu-ray-release-audit.v1",
        "released": released,
        "memory_used_mib": {str(key): memory[key] for key in sorted(memory)},
        "memory_threshold_mib": memory_threshold_mib,
        "active_compute_pids": active_pids,
        "ray_processes": ray_processes,
    }
    if not released:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
    return result


def _write_result(value: Mapping[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contracts = subparsers.add_parser("contracts")
    contracts.add_argument("--training-repository", required=True, type=Path)
    contracts.add_argument("--notool-config", required=True, type=Path)
    contracts.add_argument("--crop-config", required=True, type=Path)

    source = subparsers.add_parser("source-complete")
    source.add_argument("--config", required=True, type=Path)
    source.add_argument("--events", required=True, type=Path)
    source.add_argument("--target-step", required=True, type=int)

    target = subparsers.add_parser("target-ready")
    target.add_argument("--config", required=True, type=Path)
    target.add_argument("--authorization", required=True, type=Path)
    target.add_argument("--training-head", required=True)

    resources = subparsers.add_parser("resources-free")
    resources.add_argument("--memory-threshold-mib", type=int, default=32)

    args = parser.parse_args(argv)
    try:
        if args.command == "contracts":
            result = validate_contracts(
                training_repository=args.training_repository,
                notool_config_path=args.notool_config,
                crop_config_path=args.crop_config,
            )
        elif args.command == "source-complete":
            if args.target_step != 32:
                _fail("PRL-26 handoff source target must be exactly S32")
            result = validate_source_completion(
                config_path=args.config,
                events_path=args.events,
                target_step=args.target_step,
            )
        elif args.command == "target-ready":
            if len(args.training_head) != 40:
                _fail("training repository HEAD is malformed")
            result = target_launch_ready(
                config_path=args.config,
                authorization_path=args.authorization,
                training_head=args.training_head,
            )
        else:
            result = resource_release_audit(
                memory_threshold_mib=args.memory_threshold_mib
            )
    except RuntimeError as error:
        print(f"PRL-26 handoff rejected: {error}", file=sys.stderr)
        return 1
    _write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "resource_release_audit",
    "target_launch_ready",
    "validate_contracts",
    "validate_source_completion",
]
