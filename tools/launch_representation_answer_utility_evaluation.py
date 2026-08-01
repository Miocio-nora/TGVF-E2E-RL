#!/usr/bin/env python3
"""Launch fail-closed multi-GPU answer-utility evaluation shards.

This is an orchestration layer over
``run_representation_answer_utility_evaluation.py``.  It deliberately keeps
each evaluator single-GPU and gives every worker an independent, resumable
output root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    REPOSITORY_ROOT / "tools/run_representation_answer_utility_evaluation.py"
)
PLAN_SCHEMA_VERSION = "answer_utility_multi_worker_launch_plan_v1"
RESULT_SCHEMA_VERSION = "answer_utility_multi_worker_launch_result_v1"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_EVALUATOR_LAUNCH_MARKER = "TGVF_ANSWER_UTILITY_EVALUATION_LAUNCH_READY"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one answer-utility shard per worker, with multiple workers "
            "optionally sharing each explicitly selected physical GPU."
        )
    )
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--run-config", type=Path)
    candidate.add_argument("--production-source", action="store_true")
    parser.add_argument("--source-evaluation-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--physical-gpu-id",
        dest="physical_gpu_ids",
        type=int,
        action="append",
        required=True,
        help="Physical GPU ID; repeat to use more GPUs.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--eos-token-id", type=int, action="append")
    parser.add_argument(
        "--decode-mode", choices=("cached", "no_cache"), default="cached"
    )
    parser.add_argument(
        "--arm-batch-size",
        type=int,
        default=1,
        help="Arm decode batch size inside each evaluator worker.",
    )
    parser.add_argument("--arm", action="append")
    parser.add_argument("--include-direct-replacement", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    gpu_ids = tuple(args.physical_gpu_ids)
    if not gpu_ids:
        raise ValueError("at least one --physical-gpu-id is required")
    if any(isinstance(value, bool) or value < 0 for value in gpu_ids):
        raise ValueError("physical GPU IDs must be non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("physical GPU IDs must not be repeated")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be positive")
    if args.group_start < 0:
        raise ValueError("--group-start must be non-negative")
    if args.group_limit is not None and args.group_limit <= 0:
        raise ValueError("--group-limit must be positive")
    run_config = (
        None if args.run_config is None else args.run_config.expanduser().resolve()
    )
    source_config = args.source_evaluation_config.expanduser().resolve()
    lexical_output_root = Path(os.path.abspath(args.output_root.expanduser()))
    if lexical_output_root.is_symlink():
        raise ValueError("--output-root must not be a symlink")
    output_root = lexical_output_root
    total_workers = len(gpu_ids) * args.workers_per_gpu
    if args.group_limit is not None and total_workers > 1:
        raise ValueError(
            "--group-limit cannot be combined with multiple workers because the "
            "core evaluator applies the limit independently after sharding"
        )
    if args.arm_batch_size <= 0:
        raise ValueError("--arm-batch-size must be positive")
    return {
        "run_config": run_config,
        "production_source": args.production_source,
        "source_config": source_config,
        "output_root": output_root,
        "gpu_ids": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "poll_interval_seconds": args.poll_interval_seconds,
    }


def _common_evaluator_args(
    args: argparse.Namespace, resolved: Mapping[str, Any]
) -> list[str]:
    values = ["--source-evaluation-config", str(resolved["source_config"])]
    if resolved["production_source"]:
        values.append("--production-source")
    else:
        values.extend(("--run-config", str(resolved["run_config"])))
    values.extend(("--group-start", str(args.group_start)))
    if args.group_limit is not None:
        values.extend(("--group-limit", str(args.group_limit)))
    if args.max_new_tokens is not None:
        values.extend(("--max-new-tokens", str(args.max_new_tokens)))
    for token_id in args.eos_token_id or ():
        values.extend(("--eos-token-id", str(token_id)))
    values.extend(("--decode-mode", args.decode_mode))
    values.extend(("--arm-batch-size", str(args.arm_batch_size)))
    for arm in args.arm or ():
        values.extend(("--arm", arm))
    if args.include_direct_replacement:
        values.append("--include-direct-replacement")
    return values


def _preflight_command(
    common_args: Sequence[str], *, shard_index: int, shard_count: int
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR_PATH),
        *common_args,
        "--validate-only",
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]


def _worker_command(
    common_args: Sequence[str],
    *,
    output_root: Path,
    gpu_id: int,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR_PATH),
        *common_args,
        "--output-root",
        str(output_root),
        "--physical-gpu-id",
        str(gpu_id),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]


def _decode_preflight(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"answer-utility preflight failed with {completed.returncode}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "answer-utility preflight did not return one JSON object"
        ) from error
    if not isinstance(payload, dict) or payload.get("status") != "validated":
        raise RuntimeError("answer-utility preflight did not report validated status")
    return payload


def _run_preflight(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env={**os.environ, **THREAD_ENVIRONMENT},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return _decode_preflight(completed)


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_preflight_partition(
    whole: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> None:
    whole_count = _require_int(
        whole.get("selected_sample_count"),
        name="whole selected_sample_count",
        minimum=1,
    )
    arms = whole.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or len(set(arms)) != len(arms)
    ):
        raise RuntimeError("whole preflight arms are invalid or repeated")
    whole_ordinals = whole.get("selected_group_ordinals")
    if not isinstance(whole_ordinals, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in whole_ordinals
    ):
        raise RuntimeError("whole preflight group ordinals are invalid")
    if len(set(whole_ordinals)) != len(whole_ordinals):
        raise RuntimeError("whole preflight group ordinals are repeated")
    observed_ordinals: list[int] = []
    observed_samples = 0
    selection_fields = {
        "answer_safe_wrong_mapping_count",
        "same_target_wrong_image_mapping_count",
        "selected_group_count",
        "selected_group_ordinals",
        "selected_sample_count",
        "shard_count",
        "shard_index",
        "wrong_image_match_tiers",
    }
    whole_invariants = {
        key: value for key, value in whole.items() if key not in selection_fields
    }
    for expected_index, payload in enumerate(shards):
        if payload.get("status") != "validated":
            raise RuntimeError(f"shard {expected_index} preflight is not validated")
        if payload.get("arms") != arms:
            raise RuntimeError(
                f"shard {expected_index} arms differ from whole preflight"
            )
        if payload.get("shard_index") != expected_index:
            raise RuntimeError(f"shard {expected_index} preflight index mismatch")
        if payload.get("shard_count") != len(shards):
            raise RuntimeError(f"shard {expected_index} preflight count mismatch")
        shard_invariants = {
            key: value for key, value in payload.items() if key not in selection_fields
        }
        if shard_invariants != whole_invariants:
            raise RuntimeError(
                f"shard {expected_index} candidate/config binding differs from whole preflight"
            )
        sample_count = _require_int(
            payload.get("selected_sample_count"),
            name=f"shard {expected_index} selected_sample_count",
            minimum=1,
        )
        ordinals = payload.get("selected_group_ordinals")
        if not isinstance(ordinals, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in ordinals
        ):
            raise RuntimeError(f"shard {expected_index} group ordinals are invalid")
        observed_samples += sample_count
        observed_ordinals.extend(ordinals)
    if observed_samples != whole_count:
        raise RuntimeError(
            "shard preflight sample counts do not cover the whole selection"
        )
    if len(set(observed_ordinals)) != len(observed_ordinals):
        raise RuntimeError("shard preflights contain repeated group ordinals")
    if set(observed_ordinals) != set(whole_ordinals):
        raise RuntimeError("shard preflights do not cover the whole group selection")


def _shard_name(index: int, count: int) -> str:
    width = max(4, len(str(count - 1)))
    return f"shard-{index:0{width}d}-of-{count:0{width}d}"


def _build_plan(
    resolved: Mapping[str, Any],
    common_args: Sequence[str],
) -> dict[str, Any]:
    gpu_ids: tuple[int, ...] = resolved["gpu_ids"]
    shard_count = len(gpu_ids) * resolved["workers_per_gpu"]
    commands = [
        _preflight_command(common_args, shard_index=0, shard_count=1),
        *(
            _preflight_command(
                common_args, shard_index=shard_index, shard_count=shard_count
            )
            for shard_index in range(shard_count)
        ),
    ]
    with ThreadPoolExecutor(max_workers=min(4, len(commands))) as executor:
        preflights = list(executor.map(_run_preflight, commands))
    whole, *shard_preflights = preflights
    _validate_preflight_partition(whole, shard_preflights)
    assignments = []
    for shard_index, preflight in enumerate(shard_preflights):
        gpu_id = gpu_ids[shard_index % len(gpu_ids)]
        name = _shard_name(shard_index, shard_count)
        shard_root = resolved["output_root"] / name
        assignments.append(
            {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "physical_gpu_id": gpu_id,
                "worker_slot_on_gpu": shard_index // len(gpu_ids),
                "output_root": str(shard_root),
                "log_path": str(resolved["output_root"] / "logs" / f"{name}.log"),
                "preflight": preflight,
                "command": _worker_command(
                    common_args,
                    output_root=shard_root,
                    gpu_id=gpu_id,
                    shard_index=shard_index,
                    shard_count=shard_count,
                ),
            }
        )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "output_root": str(resolved["output_root"]),
        "physical_gpu_ids": list(gpu_ids),
        "workers_per_gpu": resolved["workers_per_gpu"],
        "thread_environment": THREAD_ENVIRONMENT,
        "whole_preflight": whole,
        "assignments": assignments,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


@contextmanager
def _exclusive_launch_lock(output_root: Path) -> Iterator[None]:
    import fcntl

    if output_root.is_symlink():
        raise RuntimeError("output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".launch.lock"
    if lock_path.is_symlink():
        raise RuntimeError("launcher lock path must not be a symlink")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"evaluation launcher is already active: {output_root}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _prepare_output(plan: Mapping[str, Any]) -> None:
    output_root = Path(plan["output_root"])
    if output_root.is_symlink():
        raise RuntimeError("output root must not be a symlink")
    plan_path = output_root / "launch-plan.json"
    declared = _canonical_bytes(plan) + b"\n"
    if plan_path.exists():
        if plan_path.is_symlink() or plan_path.read_bytes() != declared:
            raise RuntimeError("existing launch plan conflicts with this invocation")
    else:
        preexisting = sorted(
            path.name for path in output_root.iterdir() if path.name != ".launch.lock"
        )
        if preexisting:
            raise RuntimeError(
                f"non-empty output root has no matching launch plan: {preexisting}"
            )
        _write_bytes_exclusive(plan_path, declared)
    assignments = plan["assignments"]
    allowed = {
        ".launch.lock",
        "launch-plan.json",
        "launch-summary.json",
        "logs",
        *(Path(assignment["output_root"]).name for assignment in assignments),
    }
    unexpected = sorted(
        path.name for path in output_root.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise RuntimeError(f"output root contains unowned entries: {unexpected}")
    logs = output_root / "logs"
    if logs.exists() and (logs.is_symlink() or not logs.is_dir()):
        raise RuntimeError("launcher logs path is not an owned directory")
    logs.mkdir(exist_ok=True)
    allowed_logs = {Path(assignment["log_path"]).name for assignment in assignments}
    unexpected_logs = sorted(
        path.name for path in logs.iterdir() if path.name not in allowed_logs
    )
    if unexpected_logs:
        raise RuntimeError(f"launcher logs contain unowned entries: {unexpected_logs}")
    for assignment in assignments:
        log_path = Path(assignment["log_path"])
        if log_path.exists() and (log_path.is_symlink() or not log_path.is_file()):
            raise RuntimeError(f"worker log path is not a regular file: {log_path}")
        shard_root = Path(assignment["output_root"])
        if shard_root.exists() and (shard_root.is_symlink() or not shard_root.is_dir()):
            raise RuntimeError(f"shard output root is not a directory: {shard_root}")


def _worker_environment(gpu_id: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    environment.pop(_EVALUATOR_LAUNCH_MARKER, None)
    return environment


def _signal_process(process: subprocess.Popen[bytes], signum: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _stop_and_reap(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        _signal_process(process, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _signal_process(process, signal.SIGKILL)
    for process in processes:
        if process.poll() is None:
            process.wait()


def _launch_and_wait(
    plan: Mapping[str, Any], *, poll_interval_seconds: float
) -> list[int]:
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[BinaryIO] = []
    try:
        for assignment in plan["assignments"]:
            log_path = Path(assignment["log_path"])
            log = log_path.open("ab")
            logs.append(log)
            process = subprocess.Popen(
                assignment["command"],
                cwd=REPOSITORY_ROOT,
                env=_worker_environment(assignment["physical_gpu_id"]),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        pending = set(range(len(processes)))
        return_codes: list[int | None] = [None] * len(processes)
        first_failure: tuple[int, int] | None = None
        while pending:
            for index in tuple(pending):
                returncode = processes[index].poll()
                if returncode is None:
                    continue
                pending.remove(index)
                return_codes[index] = returncode
                if returncode != 0 and first_failure is None:
                    first_failure = (index, returncode)
            if first_failure is not None:
                _stop_and_reap([processes[index] for index in sorted(pending)])
                for index in tuple(pending):
                    return_codes[index] = processes[index].returncode
                    pending.remove(index)
                failed_index, failed_code = first_failure
                failed = plan["assignments"][failed_index]
                raise RuntimeError(
                    "answer-utility worker failed: "
                    f"shard={failed_index}, returncode={failed_code}, "
                    f"log={failed['log_path']}"
                )
            if pending:
                time.sleep(poll_interval_seconds)
        return [int(value) for value in return_codes]
    except BaseException:
        _stop_and_reap(processes)
        raise
    finally:
        for log in logs:
            log.close()


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"JSON artifact must not be a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"required valid JSON artifact is missing: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def _read_records(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink():
        raise RuntimeError(f"records JSONL must not be a symlink: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(f"records JSONL is missing: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise RuntimeError(f"records JSONL has a blank line: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"records JSONL has invalid JSON: {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(f"record is not an object: {path}:{line_number}")
        records.append(value)
    return tuple(records)


def _expected_keys_from_identity(
    identity_document: Mapping[str, Any], *, assignment: Mapping[str, Any]
) -> tuple[str, set[tuple[str, str]]]:
    identity = identity_document.get("identity")
    identity_sha = identity_document.get("identity_sha256")
    if not isinstance(identity, dict) or not isinstance(identity_sha, str):
        raise RuntimeError("evaluation identity document is malformed")
    if _canonical_sha256(identity) != identity_sha:
        raise RuntimeError("evaluation identity digest mismatch")
    if identity.get("shard_index") != assignment["shard_index"]:
        raise RuntimeError("evaluation identity shard index mismatch")
    if identity.get("shard_count") != assignment["shard_count"]:
        raise RuntimeError("evaluation identity shard count mismatch")
    arms = identity.get("arms")
    samples = identity.get("ordered_selected_samples")
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or len(set(arms)) != len(arms)
        or not isinstance(samples, list)
    ):
        raise RuntimeError("evaluation identity samples or arms are malformed")
    sample_ids = []
    for sample in samples:
        sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError("evaluation identity contains an invalid sample ID")
        sample_ids.append(sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("evaluation identity contains repeated sample IDs")
    preflight = assignment["preflight"]
    if arms != preflight.get("arms") or len(sample_ids) != preflight.get(
        "selected_sample_count"
    ):
        raise RuntimeError("evaluation identity differs from its validated preflight")
    identity_preflight_bindings = {
        "arm_batch_size": "arm_batch_size",
        "candidate_adapter_state_sha256": "candidate_adapter_state_sha256",
        "candidate_artifact_file_sha256": "candidate_artifact_file_sha256",
        "candidate_global_step": "candidate_global_step",
        "candidate_id": "candidate_id",
        "candidate_kind": "candidate_kind",
        "candidate_training_run_identity_sha256": (
            "candidate_training_run_identity_sha256"
        ),
        "data_manifest_sha256": "evaluation_data_manifest_sha256",
        "decode_mode": "decode_mode",
        "eos_token_ids": "eos_token_ids",
        "max_new_tokens": "max_new_tokens",
        "ordered_group_manifest_identity": "ordered_group_manifest_identity",
    }
    for identity_name, preflight_name in identity_preflight_bindings.items():
        if identity.get(identity_name) != preflight.get(preflight_name):
            raise RuntimeError(
                f"evaluation identity differs from preflight binding: {identity_name}"
            )
    return identity_sha, {(sample_id, arm) for sample_id in sample_ids for arm in arms}


def _validate_completed_shards(plan: Mapping[str, Any]) -> dict[str, Any]:
    union_expected: set[tuple[str, str]] = set()
    union_observed: set[tuple[str, str]] = set()
    shard_results = []
    for assignment in plan["assignments"]:
        shard_root = Path(assignment["output_root"])
        identity_document = _read_json_object(shard_root / "identity.json")
        identity_sha, expected = _expected_keys_from_identity(
            identity_document, assignment=assignment
        )
        if union_expected.intersection(expected):
            raise RuntimeError(
                "evaluation shard identities contain duplicate expected keys"
            )
        union_expected.update(expected)
        summary = _read_json_object(shard_root / "summary.json")
        if summary.get("status") != "complete":
            raise RuntimeError(f"evaluation shard is incomplete: {shard_root}")
        if summary.get("run_identity_sha256") != identity_sha:
            raise RuntimeError("evaluation summary identity mismatch")
        records_path = shard_root / "records.jsonl"
        records = _read_records(records_path)
        records_payload = records_path.read_bytes()
        if summary.get("records_jsonl_sha256") != sha256(records_payload).hexdigest():
            raise RuntimeError("evaluation records digest differs from summary")
        observed: set[tuple[str, str]] = set()
        for record in records:
            sample_id = record.get("sample_id")
            arm = record.get("arm")
            if not isinstance(sample_id, str) or not isinstance(arm, str):
                raise RuntimeError("evaluation record key is malformed")
            key = (sample_id, arm)
            if key in observed or key in union_observed:
                raise RuntimeError(f"evaluation record key is duplicated: {key}")
            if record.get("run_identity_sha256") != identity_sha:
                raise RuntimeError("evaluation record identity mismatch")
            observed.add(key)
            union_observed.add(key)
        if observed != expected:
            missing = len(expected - observed)
            unexpected = len(observed - expected)
            raise RuntimeError(
                "evaluation shard record coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        sample_count = assignment["preflight"]["selected_sample_count"]
        if summary.get("sample_count") != sample_count:
            raise RuntimeError("evaluation summary sample count mismatch")
        if summary.get("record_count") != len(expected):
            raise RuntimeError("evaluation summary record count mismatch")
        shard_results.append(
            {
                "shard_index": assignment["shard_index"],
                "physical_gpu_id": assignment["physical_gpu_id"],
                "output_root": str(shard_root),
                "sample_count": sample_count,
                "record_count": len(records),
                "run_identity_sha256": identity_sha,
                "records_jsonl_sha256": sha256(records_payload).hexdigest(),
            }
        )
    expected_count = plan["whole_preflight"]["selected_sample_count"] * len(
        plan["whole_preflight"]["arms"]
    )
    if union_expected != union_observed or len(union_observed) != expected_count:
        raise RuntimeError("completed shard union has missing or unexpected records")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "shard_count": len(plan["assignments"]),
        "sample_count": plan["whole_preflight"]["selected_sample_count"],
        "record_count": len(union_observed),
        "arms": plan["whole_preflight"]["arms"],
        "shards": shard_results,
    }


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = _canonical_bytes(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    resolved = _resolve_inputs(args)
    common_args = _common_evaluator_args(args, resolved)
    plan = _build_plan(resolved, common_args)
    with _exclusive_launch_lock(resolved["output_root"]):
        _prepare_output(plan)
        return_codes = _launch_and_wait(
            plan, poll_interval_seconds=resolved["poll_interval_seconds"]
        )
        if any(returncode != 0 for returncode in return_codes):
            raise RuntimeError(f"answer-utility workers failed: {return_codes}")
        result = _validate_completed_shards(plan)
        _atomic_replace_json(resolved["output_root"] / "launch-summary.json", result)
        return result


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
