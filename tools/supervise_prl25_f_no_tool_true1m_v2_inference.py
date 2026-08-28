#!/usr/bin/env python3
"""Prepare and supervise corrected true1M PRL25-F No-Tool inference.

Two four-rank arms run concurrently on eight distinct physical GPUs.  The
next pair starts only after the first pair closes.  Each durable rank JSONL is
identity-validated by ``run_policy_benchmark.py`` on resume, while every vLLM,
Triton, and TorchInductor cache is isolated by arm and rank.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from prepare_prl25_f_no_tool_true1m_v2_arm import (  # noqa: E402
    MAIN_ROOT,
    PLAN_PATH,
    PYTHON_ENVIRONMENT_ROOT,
    PYTHON_HEADER_ROOT,
    STEPS,
    TOOLCHAIN_ENVIRONMENT,
    TRUE1M_MAX_PIXELS,
    WORLD_SIZE_PER_ARM,
    _write_immutable_json,
    arm_paths,
    load_true1m_v2_plan,
)
from tgvf_rl.evaluation.controlled_toolchain import (  # noqa: E402
    PURGED_ENVIRONMENT_PREFIXES as SHARED_PURGED_ENVIRONMENT_PREFIXES,
    PURGED_TOOLCHAIN_ENVIRONMENT,
    build_controlled_toolchain_environment,
    controlled_toolchain_contract,
    controlled_toolchain_verification,
    python312_toolchain_environment,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_bound_policy_benchmark_tasks,
    load_frozen_policy_evaluation_snapshot,
    load_policy_benchmark_results,
    load_policy_coredev_config,
    policy_evaluation_identity,
)


PYTHON_BIN = MAIN_ROOT / ".venv312/bin/python"
RUNNER = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"
PREPARER = REPOSITORY_ROOT / "tools/prepare_prl25_f_no_tool_true1m_v2_arm.py"
SUPERVISOR = Path(__file__).resolve()
RUNTIME_ENVIRONMENT_SCHEMA = "tgvf.prl25-f-runtime-environment-contract.v2"
WORKER_LAUNCH_SCHEMA = "tgvf.prl25-f-worker-launch-contract.v2"
INFERENCE_STATUS_SCHEMA = "tgvf.prl25-f-true1m-inference-status.v2"
INFERENCE_COMPLETION_SCHEMA = "tgvf.prl25-f-true1m-inference-completion.v4"
MATCHED_COMPLETION_SCHEMA = "tgvf.prl25-f-true1m-matched-inference-completion.v4"
RANK_TREE_SCHEMA = "tgvf.prl25-f-true1m-rank-tree.v1"
RUNTIME_ENVIRONMENT_FILENAME = "runtime-environment-contract.json"
WORKER_LAUNCH_FILENAME = "worker-launch-contract.json"
FIXED_RUNTIME_ENVIRONMENT = {
    **TOOLCHAIN_ENVIRONMENT,
    "PYTHONPATH": os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(MAIN_ROOT / ".deps/verl"))
    ),
    "TGVF_REPOSITORY_ROOT": str(REPOSITORY_ROOT),
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "42",
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_PLUGINS": "",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
}
PURGED_ONLY_TOOLCHAIN_ENVIRONMENT = tuple(
    name
    for name in PURGED_TOOLCHAIN_ENVIRONMENT
    if name not in FIXED_RUNTIME_ENVIRONMENT
)
PURGED_ENVIRONMENT_PREFIXES = SHARED_PURGED_ENVIRONMENT_PREFIXES


@dataclass(frozen=True, slots=True)
class WorkerLaunch:
    step: int
    rank: int
    gpu_id: int
    config_path: Path
    output_root: Path
    log_path: Path
    command: tuple[str, ...]
    environment: dict[str, str]


@dataclass(slots=True)
class ActiveWorker:
    launch: WorkerLaunch
    process: subprocess.Popen[bytes]
    log_handle: BinaryIO


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_identity(payload: dict[str, object]) -> dict[str, object]:
    if "identity_sha256" in payload:
        raise ValueError("identity payload already contains an identity")
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _validate_identity(
    payload: dict[str, Any], *, schema: str, name: str
) -> dict[str, Any]:
    identity = payload.get("identity_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    if (
        payload.get("schema_version") != schema
        or not isinstance(identity, str)
        or identity != _canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{name} identity differs")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"accepted JSON artifact is absent or a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"accepted JSON artifact is not an object: {path}")
    return payload


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required runtime file is absent: {path}")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _file_binding(path: Path, *, identity_sha256: str) -> dict[str, object]:
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": _sha256_file(path),
        "identity_sha256": identity_sha256,
    }


@contextmanager
def _completed_rank_locks(output_root: Path) -> Iterator[None]:
    """Hold all existing worker locks while hashing a completed rank tree."""

    handles: list[BinaryIO] = []
    try:
        for rank in range(WORLD_SIZE_PER_ARM):
            lock_path = output_root / f"runtime/locks/rank-{rank}.lock"
            if lock_path.is_symlink() or not lock_path.is_file():
                raise RuntimeError(f"rank {rank} completion lock is absent")
            handle = lock_path.open("r+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"rank {rank} is still active") from error
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _rank_tree_evidence(plan: dict[str, Any], *, step: int) -> dict[str, object]:
    """Revalidate and bind every durable row and rank-file byte."""

    paths = arm_paths(plan, step)
    with _completed_rank_locks(paths.output_root):
        config = load_policy_coredev_config(paths.config_path)
        snapshot = load_frozen_policy_evaluation_snapshot(config)
        expected_identity = policy_evaluation_identity(config, snapshot)
        identity_path = paths.output_root / "runtime/evaluation-identity.json"
        observed_identity = _load_json(identity_path)
        if observed_identity != expected_identity:
            raise RuntimeError(f"S{step} evaluation identity differs")
        tasks = load_bound_policy_benchmark_tasks(config)
        records = load_policy_benchmark_results(
            paths.output_root / "inference",
            tasks=tasks,
            evaluation_identity=observed_identity,
            require_complete=True,
        )
        if len(records) != 2240:
            raise RuntimeError(f"S{step} rank tree row count differs")

        files: list[dict[str, object]] = []
        global_identities: list[dict[str, object]] = []
        for rank in range(WORLD_SIZE_PER_ARM):
            rank_path = paths.output_root / f"inference/rank-{rank}.jsonl"
            if rank_path.is_symlink() or not rank_path.is_file():
                raise RuntimeError(f"S{step} rank{rank} JSONL is absent")
            ordinals = sorted(
                ordinal for ordinal in records if ordinal % WORLD_SIZE_PER_ARM == rank
            )
            with rank_path.open("rb") as handle:
                line_count = sum(bool(line.strip()) for line in handle)
            if line_count != len(ordinals):
                raise RuntimeError(f"S{step} rank{rank} line count differs")
            identities = [
                {
                    "ordinal": ordinal,
                    "result_identity_sha256": records[ordinal][
                        "result_identity_sha256"
                    ],
                }
                for ordinal in ordinals
            ]
            global_identities.extend(identities)
            files.append(
                {
                    **_file_record(rank_path),
                    "rank": rank,
                    "line_count": line_count,
                    "ordinal_sequence_sha256": _canonical_sha256(ordinals),
                    "result_identity_sequence_sha256": _canonical_sha256(identities),
                }
            )
        global_identities.sort(key=lambda item: int(item["ordinal"]))
        return _with_identity(
            {
                "schema_version": RANK_TREE_SCHEMA,
                "evaluation_id": paths.evaluation_id,
                "optimizer_step": step,
                "evaluation_identity_sha256": observed_identity["identity_sha256"],
                "task_manifest_sha256": config.task_manifest_sha256,
                "world_size": WORLD_SIZE_PER_ARM,
                "row_count": len(records),
                "files": files,
                "result_identity_sequence_sha256": _canonical_sha256(global_identities),
            }
        )


def _environment_purge_policy() -> dict[str, object]:
    return controlled_toolchain_contract(TOOLCHAIN_ENVIRONMENT)


def _runtime_environment_contract(plan: dict[str, Any]) -> dict[str, object]:
    if plan["execution"].get("toolchain_environment") != TOOLCHAIN_ENVIRONMENT:
        raise RuntimeError("planned NoTool toolchain environment differs")
    if TOOLCHAIN_ENVIRONMENT != python312_toolchain_environment(
        python_environment_root=PYTHON_ENVIRONMENT_ROOT,
        python_header_root=PYTHON_HEADER_ROOT,
    ):
        raise RuntimeError("NoTool shared toolchain environment differs")
    for directory in (
        PYTHON_ENVIRONMENT_ROOT / "lib",
        *(Path(item) for item in TOOLCHAIN_ENVIRONMENT["PATH"].split(os.pathsep)),
    ):
        if not directory.is_dir():
            raise RuntimeError(f"required runtime directory is absent: {directory}")
    artifacts = {
        "python": _file_record(PYTHON_BIN),
        "cc": _file_record(Path(TOOLCHAIN_ENVIRONMENT["CC"])),
        "cxx": _file_record(Path(TOOLCHAIN_ENVIRONMENT["CXX"])),
        "python_h": _file_record(PYTHON_HEADER_ROOT / "python3.12/Python.h"),
        "python_pyconfig_h": _file_record(PYTHON_HEADER_ROOT / "python3.12/pyconfig.h"),
        "platform_pyconfig_h": _file_record(
            PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h"
        ),
        "runner": _file_record(RUNNER),
        "preparer": _file_record(PREPARER),
        "supervisor": _file_record(SUPERVISOR),
        "plan": _file_record(PLAN_PATH),
    }
    controlled_environment = _base_environment()
    if any(
        controlled_environment.get(name) != value
        for name, value in FIXED_RUNTIME_ENVIRONMENT.items()
    ):
        raise RuntimeError("NoTool controlled runtime environment differs")
    return _with_identity(
        {
            "schema_version": RUNTIME_ENVIRONMENT_SCHEMA,
            "evaluation_id": plan["evaluation_id"],
            "repository_root": str(REPOSITORY_ROOT),
            "python_prefix": str(PYTHON_ENVIRONMENT_ROOT),
            "environment": dict(FIXED_RUNTIME_ENVIRONMENT),
            "environment_purge_policy": _environment_purge_policy(),
            "environment_purge_verification": controlled_toolchain_verification(
                controlled_environment,
                controlled=TOOLCHAIN_ENVIRONMENT,
            ),
            "artifacts": artifacts,
        }
    )


def _validate_gpu_ids(gpu_ids: tuple[int, ...]) -> None:
    if (
        len(gpu_ids) != 8
        or len(set(gpu_ids)) != 8
        or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
    ):
        raise ValueError("PRL25-F true1M V2 requires eight distinct GPU IDs")


def _gpu_groups(gpu_ids: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    _validate_gpu_ids(gpu_ids)
    return {"left": gpu_ids[:4], "right": gpu_ids[4:]}


def _base_environment() -> dict[str, str]:
    return build_controlled_toolchain_environment(
        controlled=TOOLCHAIN_ENVIRONMENT,
        overlay=FIXED_RUNTIME_ENVIRONMENT,
    )


def _worker_launches(
    plan: dict[str, Any],
    *,
    gpu_ids: tuple[int, ...],
    evaluation_root: Path | None = None,
    create_cache_directories: bool = True,
) -> tuple[WorkerLaunch, ...]:
    groups = _gpu_groups(gpu_ids)
    root = (
        Path(plan["evaluation_root"])
        if evaluation_root is None
        else evaluation_root.resolve()
    )
    base_environment = _base_environment()
    launches: list[WorkerLaunch] = []
    for arm in plan["arms"]:
        step = arm["optimizer_step"]
        output_root = root / f"matched/step{step}"
        config_path = output_root / "config.json"
        for rank, gpu_id in enumerate(groups[arm["gpu_group"]]):
            cache_root = output_root / "runtime/cache"
            caches = {
                "VLLM_CACHE_ROOT": cache_root / f"vllm/rank-{rank}",
                "TRITON_CACHE_DIR": cache_root / f"triton/rank-{rank}",
                "TORCHINDUCTOR_CACHE_DIR": (cache_root / f"torchinductor/rank-{rank}"),
            }
            if create_cache_directories:
                for cache in caches.values():
                    cache.mkdir(parents=True, exist_ok=True)
            environment = {
                **base_environment,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                **{name: str(path) for name, path in caches.items()},
            }
            launches.append(
                WorkerLaunch(
                    step=step,
                    rank=rank,
                    gpu_id=gpu_id,
                    config_path=config_path,
                    output_root=output_root,
                    log_path=root / f"logs/true1m-v2-s{step}-rank{rank}.log",
                    command=(
                        str(PYTHON_BIN),
                        str(RUNNER),
                        "--config",
                        str(config_path),
                        "--mode",
                        "worker",
                        "--rank",
                        str(rank),
                        "--world-size",
                        str(WORLD_SIZE_PER_ARM),
                    ),
                    environment=environment,
                )
            )
    return tuple(launches)


def _worker_environment_record(launch: WorkerLaunch) -> dict[str, object]:
    required = {
        **FIXED_RUNTIME_ENVIRONMENT,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(launch.gpu_id),
        "VLLM_CACHE_ROOT": str(
            launch.output_root / f"runtime/cache/vllm/rank-{launch.rank}"
        ),
        "TRITON_CACHE_DIR": str(
            launch.output_root / f"runtime/cache/triton/rank-{launch.rank}"
        ),
        "TORCHINDUCTOR_CACHE_DIR": str(
            launch.output_root / f"runtime/cache/torchinductor/rank-{launch.rank}"
        ),
    }
    if any(launch.environment.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            f"S{launch.step} rank{launch.rank} runtime environment differs"
        )
    try:
        purge_verification = controlled_toolchain_verification(
            launch.environment,
            controlled=TOOLCHAIN_ENVIRONMENT,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"S{launch.step} rank{launch.rank} retained forbidden environment"
        ) from error
    identity_payload = {
        "environment": required,
        "environment_purge_policy": _environment_purge_policy(),
        "environment_purge_verification": purge_verification,
    }
    return {
        **identity_payload,
        "environment_identity_sha256": _canonical_sha256(identity_payload),
    }


def _worker_launch_contract(
    plan: dict[str, Any],
    launches: tuple[WorkerLaunch, ...],
    *,
    runtime_environment_identity_sha256: str,
) -> dict[str, object]:
    gpu_ids: list[int] = []
    for arm in plan["arms"]:
        step_launches = [
            launch for launch in launches if launch.step == arm["optimizer_step"]
        ]
        if len(step_launches) != WORLD_SIZE_PER_ARM:
            raise RuntimeError("NoTool worker launch population differs")
        if arm["concurrency_round"] == 0:
            gpu_ids.extend(launch.gpu_id for launch in step_launches)
    unique_gpu_ids = tuple(dict.fromkeys(gpu_ids))
    _validate_gpu_ids(unique_gpu_ids)
    workers: list[dict[str, object]] = []
    for launch in launches:
        environment = _worker_environment_record(launch)
        workers.append(
            {
                "optimizer_step": launch.step,
                "rank": launch.rank,
                "gpu_id": launch.gpu_id,
                "config_path": str(launch.config_path),
                "output_root": str(launch.output_root),
                "log_path": str(launch.log_path),
                "command": list(launch.command),
                **environment,
            }
        )
    return _with_identity(
        {
            "schema_version": WORKER_LAUNCH_SCHEMA,
            "evaluation_id": plan["evaluation_id"],
            "gpu_ids": list(unique_gpu_ids),
            "runtime_environment_identity_sha256": (
                runtime_environment_identity_sha256
            ),
            "workers": workers,
        }
    )


def _pair_schedule(
    launches: tuple[WorkerLaunch, ...],
) -> tuple[tuple[WorkerLaunch, ...], ...]:
    by_step: dict[int, list[WorkerLaunch]] = {}
    for launch in launches:
        by_step.setdefault(launch.step, []).append(launch)
    schedule = (
        tuple(by_step[step] for step in (0, 8)),
        tuple(by_step[step] for step in (16, 32)),
    )
    return tuple(tuple(launch for arm in pair for launch in arm) for pair in schedule)


def _run_capture(
    command: tuple[str, ...],
    *,
    log_path: Path,
    environment: dict[str, str],
) -> str:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout
        + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {' '.join(command)}"
        )
    return completed.stdout


def _validate_processor_proof(
    validation: dict[str, object], plan: dict[str, Any], *, step: int
) -> dict[str, object]:
    proof = validation.get("no_tool_matched_processor_proof")
    expected = plan["image_preprocessing"]["prepare_validate_probe"]
    if not isinstance(proof, dict):
        raise RuntimeError(f"S{step} validation omitted the no-tool processor proof")
    represented = proof.get("synthetic_native_represented_pixel_area")
    if (
        validation.get("optimizer_step") != step
        or validation.get("gpu_or_api_used") is not False
        or validation.get("vllm_engine_constructed") is not False
        or proof.get("configured_image_max_pixels") != TRUE1M_MAX_PIXELS
        or proof.get("processor_image_size", {}).get("longest_edge")
        != plan["image_preprocessing"]["qwen_fast_processor_default_max_pixels"]
        or proof.get("effective_processor_image_size", {}).get("longest_edge")
        != TRUE1M_MAX_PIXELS
        or proof.get("synthetic_native_source_pixel_area")
        != expected["source_pixel_area"]
        or represented != expected["expected_represented_pixel_area"]
        or type(represented) is not int
        or represented > TRUE1M_MAX_PIXELS
        or proof.get("synthetic_native_visual_token_count")
        != expected["expected_visual_token_count"]
        or proof.get("runtime_mm_processor_kwargs")
        != {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": TRUE1M_MAX_PIXELS,
            }
        }
        or proof.get("runtime_override_path") != "mm_processor_kwargs.size.longest_edge"
        or proof.get("vllm_012_shallow_hashable") is not True
        or proof.get("nested_images_kwargs_present") is not False
        or proof.get("max_pixels_kwarg_present") is not False
        or proof.get("tool_schema_visible") is not False
        or proof.get("system_prompt_present") is not False
    ):
        raise RuntimeError(f"S{step} real-Qwen true1M processor proof differs")
    return {
        "schema_version": "tgvf.prl25-f-true1m-processor-acceptance.v2",
        "evaluation_id": validation["evaluation_id"],
        "optimizer_step": step,
        "gpu_or_api_used": False,
        "vllm_engine_constructed": False,
        "proof": proof,
    }


def _prepare_arm(plan: dict[str, Any], *, step: int, gpu_ids: tuple[int, ...]) -> None:
    paths = arm_paths(plan, step)
    log_root = Path(plan["evaluation_root"]) / "logs"
    environment = {**_base_environment(), "CUDA_VISIBLE_DEVICES": ""}
    _run_capture(
        (
            str(PYTHON_BIN),
            str(PREPARER),
            "--step",
            str(step),
            "--gpu-ids",
            *(str(gpu) for gpu in gpu_ids),
        ),
        log_path=log_root / f"true1m-v2-prepare-artifacts-s{step}.log",
        environment=environment,
    )
    _run_capture(
        (
            str(PYTHON_BIN),
            str(RUNNER),
            "--config",
            str(paths.config_path),
            "--mode",
            "prepare",
        ),
        log_path=log_root / f"true1m-v2-prepare-eval-s{step}.log",
        environment=environment,
    )
    stdout = _run_capture(
        (
            str(PYTHON_BIN),
            str(RUNNER),
            "--config",
            str(paths.config_path),
            "--mode",
            "validate",
            "--world-size",
            str(WORLD_SIZE_PER_ARM),
        ),
        log_path=log_root / f"true1m-v2-validate-s{step}.log",
        environment=environment,
    )
    accepted = _validate_processor_proof(json.loads(stdout), plan, step=step)
    _write_immutable_json(
        paths.output_root / "runtime/true1m-processor-proof.json", accepted
    )


def _gpu_has_compute_process(gpu_id: int) -> bool:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query GPU {gpu_id}")
    return bool(completed.stdout.strip())


def _await_idle_gpus(gpu_ids: tuple[int, ...], *, wait: bool) -> None:
    while True:
        busy = tuple(gpu for gpu in gpu_ids if _gpu_has_compute_process(gpu))
        if not busy:
            return
        if not wait:
            raise RuntimeError(f"requested GPUs are busy: {busy}")
        print(f"waiting for GPUs to become idle: {busy}", flush=True)
        time.sleep(10)


def _terminate_process_groups(workers: list[ActiveWorker]) -> None:
    for worker in workers:
        try:
            os.killpg(worker.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if all(worker.process.poll() is not None for worker in workers):
            break
        time.sleep(1)
    for worker in workers:
        if worker.process.poll() is None:
            try:
                os.killpg(worker.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _run_pair(
    plan: dict[str, Any],
    launches: tuple[WorkerLaunch, ...],
    *,
    wait_for_gpus: bool,
    runtime_contract: dict[str, object],
    launch_contract: dict[str, object],
) -> None:
    if len(launches) != 8 or len({item.gpu_id for item in launches}) != 8:
        raise RuntimeError("one PRL25-F inference round must occupy eight GPUs")
    _await_idle_gpus(tuple(item.gpu_id for item in launches), wait=wait_for_gpus)
    workers: list[ActiveWorker] = []
    try:
        for launch in launches:
            launch.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = launch.log_path.open("ab")
            process = subprocess.Popen(
                launch.command,
                cwd=REPOSITORY_ROOT,
                env=launch.environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            workers.append(ActiveWorker(launch, process, handle))
        remaining = set(range(len(workers)))
        while remaining:
            for index in tuple(remaining):
                status = workers[index].process.poll()
                if status is None:
                    continue
                remaining.remove(index)
                if status != 0:
                    launch = workers[index].launch
                    raise RuntimeError(
                        f"S{launch.step} rank{launch.rank} failed with status {status}"
                    )
            if remaining:
                time.sleep(1)
    except BaseException:
        _terminate_process_groups(workers)
        raise
    finally:
        for worker in workers:
            worker.log_handle.close()

    for step in sorted({item.step for item in launches}):
        paths = arm_paths(plan, step)
        status_stdout = _run_capture(
            (
                str(PYTHON_BIN),
                str(RUNNER),
                "--config",
                str(paths.config_path),
                "--mode",
                "status",
                "--world-size",
                str(WORLD_SIZE_PER_ARM),
            ),
            log_path=(
                Path(plan["evaluation_root"]) / f"logs/true1m-v2-status-s{step}.json"
            ),
            environment=_base_environment(),
        )
        status = json.loads(status_stdout)
        if (
            status.get("evaluation_id") != paths.evaluation_id
            or status.get("completed_single_image") != 2240
            or status.get("total_single_image") != 2240
            or status.get("remaining_single_image") != 0
            or status.get("multi_image_pending_protocol_decision") != 271
        ):
            raise RuntimeError(f"S{step} true1M inference completion differs")
        worker_identities = [
            worker["environment_identity_sha256"]
            for worker in launch_contract["workers"]  # type: ignore[index]
            if worker["optimizer_step"] == step
        ]
        if len(worker_identities) != WORLD_SIZE_PER_ARM:
            raise RuntimeError(f"S{step} worker environment population differs")
        rank_tree = _rank_tree_evidence(plan, step=step)
        status_receipt = _with_identity(
            {
                "schema_version": INFERENCE_STATUS_SCHEMA,
                "evaluation_id": paths.evaluation_id,
                "optimizer_step": step,
                "runner_status": status,
                "runtime_environment_identity_sha256": runtime_contract[
                    "identity_sha256"
                ],
                "worker_launch_identity_sha256": launch_contract["identity_sha256"],
                "worker_environment_identity_sha256": worker_identities,
                "rank_tree": rank_tree,
            }
        )
        status_path = paths.output_root / "runtime/inference-status.json"
        _write_immutable_json(status_path, status_receipt)
        completion = _with_identity(
            {
                "schema_version": INFERENCE_COMPLETION_SCHEMA,
                "evaluation_id": paths.evaluation_id,
                "optimizer_step": step,
                "completed_single_image": 2240,
                "unsupported_multi_image": 271,
                "runtime_environment_identity_sha256": runtime_contract[
                    "identity_sha256"
                ],
                "worker_launch_identity_sha256": launch_contract["identity_sha256"],
                "rank_tree_identity_sha256": rank_tree["identity_sha256"],
                "result_identity_sequence_sha256": rank_tree[
                    "result_identity_sequence_sha256"
                ],
                "status_receipt": _file_binding(
                    status_path,
                    identity_sha256=str(status_receipt["identity_sha256"]),
                ),
            }
        )
        _write_immutable_json(
            paths.output_root / "runtime/inference-complete.json", completion
        )


def _matched_completion_receipt(
    plan: dict[str, Any],
    *,
    runtime_contract_path: Path,
    runtime_contract: dict[str, Any],
    launch_contract_path: Path,
    launch_contract: dict[str, Any],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for step in STEPS:
        completion_path = arm_paths(plan, step).output_root / (
            "runtime/inference-complete.json"
        )
        completion = _validate_identity(
            _load_json(completion_path),
            schema=INFERENCE_COMPLETION_SCHEMA,
            name=f"S{step} inference completion",
        )
        arms[str(step)] = _file_binding(
            completion_path,
            identity_sha256=str(completion["identity_sha256"]),
        )
    return _with_identity(
        {
            "schema_version": MATCHED_COMPLETION_SCHEMA,
            "evaluation_id": plan["evaluation_id"],
            "optimizer_steps": list(STEPS),
            "completed_single_image_per_step": 2240,
            "unsupported_multi_image_per_step": 271,
            "runtime_environment_contract": _file_binding(
                runtime_contract_path,
                identity_sha256=str(runtime_contract["identity_sha256"]),
            ),
            "worker_launch_contract": _file_binding(
                launch_contract_path,
                identity_sha256=str(launch_contract["identity_sha256"]),
            ),
            "arms": arms,
        }
    )


def _validate_matched_inference_completion(
    plan: dict[str, Any],
) -> dict[str, Any]:
    evaluation_root = Path(plan["evaluation_root"])
    control_root = evaluation_root / "runtime/supervisor"
    runtime_contract_path = control_root / RUNTIME_ENVIRONMENT_FILENAME
    runtime_contract = _validate_identity(
        _load_json(runtime_contract_path),
        schema=RUNTIME_ENVIRONMENT_SCHEMA,
        name="NoTool runtime environment contract",
    )
    expected_runtime_contract = _runtime_environment_contract(plan)
    if runtime_contract != expected_runtime_contract:
        raise RuntimeError("NoTool runtime environment contract bytes differ")

    launch_contract_path = control_root / WORKER_LAUNCH_FILENAME
    launch_contract = _validate_identity(
        _load_json(launch_contract_path),
        schema=WORKER_LAUNCH_SCHEMA,
        name="NoTool worker launch contract",
    )
    raw_gpu_ids = launch_contract.get("gpu_ids")
    if not isinstance(raw_gpu_ids, list) or any(
        type(gpu_id) is not int for gpu_id in raw_gpu_ids
    ):
        raise RuntimeError("NoTool worker launch GPU identity differs")
    gpu_ids = tuple(raw_gpu_ids)
    _validate_gpu_ids(gpu_ids)
    expected_launches = _worker_launches(
        plan,
        gpu_ids=gpu_ids,
        create_cache_directories=False,
    )
    expected_launch_contract = _worker_launch_contract(
        plan,
        expected_launches,
        runtime_environment_identity_sha256=str(runtime_contract["identity_sha256"]),
    )
    if launch_contract != expected_launch_contract:
        raise RuntimeError("NoTool worker launch contract bytes differ")

    workers = launch_contract["workers"]
    if not isinstance(workers, list):
        raise RuntimeError("NoTool worker launch population is malformed")
    for step in STEPS:
        paths = arm_paths(plan, step)
        status_path = paths.output_root / "runtime/inference-status.json"
        status = _validate_identity(
            _load_json(status_path),
            schema=INFERENCE_STATUS_SCHEMA,
            name=f"S{step} inference status",
        )
        expected_worker_identities = [
            worker["environment_identity_sha256"]
            for worker in workers
            if worker["optimizer_step"] == step
        ]
        expected_rank_tree = _rank_tree_evidence(plan, step=step)
        if (
            status.get("evaluation_id") != paths.evaluation_id
            or status.get("optimizer_step") != step
            or status.get("runtime_environment_identity_sha256")
            != runtime_contract["identity_sha256"]
            or status.get("worker_launch_identity_sha256")
            != launch_contract["identity_sha256"]
            or status.get("worker_environment_identity_sha256")
            != expected_worker_identities
            or status.get("rank_tree") != expected_rank_tree
            or status.get("runner_status")
            != {
                "evaluation_id": paths.evaluation_id,
                "completed_single_image": 2240,
                "total_single_image": 2240,
                "remaining_single_image": 0,
                "multi_image_pending_protocol_decision": 271,
            }
        ):
            raise RuntimeError(f"S{step} formal inference status differs")
        completion_path = paths.output_root / "runtime/inference-complete.json"
        completion = _validate_identity(
            _load_json(completion_path),
            schema=INFERENCE_COMPLETION_SCHEMA,
            name=f"S{step} inference completion",
        )
        if (
            completion.get("evaluation_id") != paths.evaluation_id
            or completion.get("optimizer_step") != step
            or completion.get("completed_single_image") != 2240
            or completion.get("unsupported_multi_image") != 271
            or completion.get("runtime_environment_identity_sha256")
            != runtime_contract["identity_sha256"]
            or completion.get("worker_launch_identity_sha256")
            != launch_contract["identity_sha256"]
            or completion.get("rank_tree_identity_sha256")
            != expected_rank_tree["identity_sha256"]
            or completion.get("result_identity_sequence_sha256")
            != expected_rank_tree["result_identity_sequence_sha256"]
            or completion.get("status_receipt")
            != _file_binding(
                status_path,
                identity_sha256=str(status["identity_sha256"]),
            )
        ):
            raise RuntimeError(f"S{step} formal inference completion differs")

    expected_aggregate = _matched_completion_receipt(
        plan,
        runtime_contract_path=runtime_contract_path,
        runtime_contract=runtime_contract,
        launch_contract_path=launch_contract_path,
        launch_contract=launch_contract,
    )
    aggregate_path = control_root / "matched-inference-complete.json"
    aggregate = _validate_identity(
        _load_json(aggregate_path),
        schema=MATCHED_COMPLETION_SCHEMA,
        name="NoTool matched inference completion",
    )
    if aggregate != expected_aggregate:
        raise RuntimeError("NoTool matched inference completion bytes differ")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=int, nargs=8, default=tuple(range(8)))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-completion-only", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.verify_completion_only:
        raise ValueError("select only one NoTool supervisor validation mode")
    gpu_ids = tuple(args.gpu_ids)
    _validate_gpu_ids(gpu_ids)
    plan = load_true1m_v2_plan(PLAN_PATH)
    evaluation_root = Path(plan["evaluation_root"])
    if args.verify_completion_only:
        print(
            json.dumps(
                _validate_matched_inference_completion(plan),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    control_root = evaluation_root / "runtime/supervisor"
    control_root.mkdir(parents=True, exist_ok=True)
    lock_path = control_root / "supervisor.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "PRL25-F true1M V2 inference supervisor is already active"
            ) from error
        groups = _gpu_groups(gpu_ids)
        phase = "preparing"
        failure = control_root / "failed.json"
        runtime_contract: dict[str, object] | None = None
        # A fresh lock-owning attempt supersedes a stale failure from an older
        # attempt.  If this run fails, the exception path below writes a new,
        # phase-accurate marker.  Clearing it here also lets a concurrently
        # started scoring supervisor wait for this inference attempt instead
        # of failing on obsolete state.
        failure.unlink(missing_ok=True)
        try:
            runtime_contract = _runtime_environment_contract(plan)
            runtime_contract_path = control_root / RUNTIME_ENVIRONMENT_FILENAME
            _write_immutable_json(runtime_contract_path, runtime_contract)
            for arm in plan["arms"]:
                _prepare_arm(
                    plan,
                    step=arm["optimizer_step"],
                    gpu_ids=groups[arm["gpu_group"]],
                )
            if args.prepare_only:
                print(
                    "PRL25-F true1M V2 prepare/validate and runtime contract "
                    "complete; GPU unused"
                )
                return 0
            launches = _worker_launches(plan, gpu_ids=gpu_ids)
            launch_contract = _worker_launch_contract(
                plan,
                launches,
                runtime_environment_identity_sha256=str(
                    runtime_contract["identity_sha256"]
                ),
            )
            launch_contract_path = control_root / WORKER_LAUNCH_FILENAME
            _write_immutable_json(launch_contract_path, launch_contract)
            for round_index, pair in enumerate(_pair_schedule(launches)):
                phase = f"inference-round-{round_index}"
                _run_pair(
                    plan,
                    pair,
                    wait_for_gpus=args.wait_for_gpus,
                    runtime_contract=runtime_contract,
                    launch_contract=launch_contract,
                )
            phase = "complete"
            aggregate_path = control_root / "matched-inference-complete.json"
            _write_immutable_json(
                aggregate_path,
                _matched_completion_receipt(
                    plan,
                    runtime_contract_path=runtime_contract_path,
                    runtime_contract=runtime_contract,
                    launch_contract_path=launch_contract_path,
                    launch_contract=launch_contract,
                ),
            )
            _validate_matched_inference_completion(plan)
            # Compatibility marker for the established scoring-supervisor layout.
            (control_root / "matched-inference-complete").touch(exist_ok=True)
            failure.unlink(missing_ok=True)
            print("PRL25-F corrected true1M V2 inference complete")
            return 0
        except BaseException as error:
            failure.write_text(
                json.dumps(
                    {
                        "schema_version": "tgvf.prl25-f-true1m-failure.v2",
                        "phase": phase,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "durable_rank_rows_preserved_for_resume": True,
                        "runtime_environment_identity_sha256": (
                            None
                            if runtime_contract is None
                            else runtime_contract["identity_sha256"]
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
