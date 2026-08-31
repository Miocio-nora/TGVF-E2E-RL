#!/usr/bin/env python3
"""Wait for RP70 ACC generation and publish pinned semantic rescoring.

This supervisor never starts an answer-utility generator.  It accepts only the
two already materialized RP70 three-arm launch plans, waits for their immutable
``launch-summary.json`` publications, then owns a Qwen2.5-72B TP=2 judge on
physical GPUs 6 and 7 for the duration of semantic rescoring.  The judge is
stopped on success, failure, timeout, SIGINT, or SIGTERM.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPOSITORY_ROOT / ".venv312/bin/python"
CLEAN_RUNTIME = REPOSITORY_ROOT / ".eval-runtime-rp70-20260802"
CLEAN_RUNTIME_COMMIT = "2d61b07995b1d5b90c221fe1faf5090e8d985fef"
RESCORE_TOOL = (
    CLEAN_RUNTIME / "tools/run_representation_answer_utility_semantic_rescore.py"
)
PYTHON_HEADER_ROOT = REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"

EVALUATION_ROOT = REPOSITORY_ROOT / (
    "artifacts/evaluation/"
    "RP-70-qwen3-instruct-answer-bearing-span-step0500"
)
FIRST_GENERATION = EVALUATION_ROOT / "ACC-VAL-first200-generation-3arm"
FULL_GENERATION = EVALUATION_ROOT / "ACC-VAL-full867-generation-3arm"
FIRST_SEMANTIC = EVALUATION_ROOT / "ACC-VAL-first200-semantic-3arm"
FULL_SEMANTIC = EVALUATION_ROOT / "ACC-VAL-full867-semantic-3arm"
SUPERVISOR_ROOT = EVALUATION_ROOT / "semantic-rescore-supervisor"
EVENTS = SUPERVISOR_ROOT / "events.jsonl"
STATE = SUPERVISOR_ROOT / "state.json"
LOCK = SUPERVISOR_ROOT / "supervisor.lock"
OWNERSHIP = SUPERVISOR_ROOT / "judge-ownership.json"
JUDGE_LOG = SUPERVISOR_ROOT / "judge-server.log"
FIRST_RESCORE_LOG = SUPERVISOR_ROOT / "first200-semantic-rescore.log"
FULL_RESCORE_LOG = SUPERVISOR_ROOT / "full867-semantic-rescore.log"
COMPLETE_MARKER = SUPERVISOR_ROOT / "complete.json"

FIRST_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/answer_bearing_span/evaluation/"
    "rp70_step0500_first200_acc_gpu0.toml"
)
FULL_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/answer_bearing_span/evaluation/"
    "rp70_step0500_full867_gpu0.toml"
)
FIRST_CONFIG_SHA256 = "07c38ff05d549f00b7af487078beda514cf7990ffe8a14e4181dde80c8ec87f0"
FULL_CONFIG_SHA256 = "06795255773fbf0388b788430ddabc816c36e697d19870d92d6630732838926e"
JUDGE_CONFIG = (
    REPOSITORY_ROOT / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
)
JUDGE_CONFIG_SHA256 = "3737504858912a6392679d2c9720597cde58dd7d3218aa6f75b67ad00a769573"
JUDGE_MODEL = REPOSITORY_ROOT.parent / "models/hf/Qwen2.5-72B-Instruct"
# The repository and model roots are siblings only on some mounts.  Preserve
# the deployment path pinned by the accepted judge config instead.
JUDGE_MODEL = Path("/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct")
JUDGE_SERVED_NAME = "Qwen2.5-72B-Instruct"
JUDGE_PORT = 8013
JUDGE_GPUS = (6, 7)

ARMS = (
    "image_only",
    "image_correct_D",
    "image_same_target_wrong_image_D",
)
GENERATION_SCHEMA = "answer_utility_multi_worker_launch_result_v1"
PLAN_SCHEMA = "answer_utility_multi_worker_launch_plan_v1"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
MARKER_SCHEMA = "rp70-acc-val-semantic-rescore-complete-v1"
EXPECTED_GENERATIONS = (
    ("first200", FIRST_GENERATION, FIRST_CONFIG, 200, (1, 2), FIRST_SEMANTIC),
    ("full867", FULL_GENERATION, FULL_CONFIG, 867, (3, 4, 5, 6, 7), FULL_SEMANTIC),
)


class SupervisorError(RuntimeError):
    """Fail-closed RP70 semantic-supervisor error."""


class SupervisorInterrupted(SupervisorError):
    """Raised by an owned termination signal so children can be reaped."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise SupervisorError(f"{label} is not one regular file: {path}")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_event(event: str, **fields: object) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at": _utc_now(), "event": event, **fields}
    with EVENTS.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(payload) + b"\n")
        os.fsync(handle.fileno())


def _artifact_record(path: Path) -> dict[str, object]:
    _regular_file(path, label="completion artifact")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise SupervisorError(
            f"git {' '.join(arguments)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _assert_static_inputs() -> None:
    for path, expected, label in (
        (FIRST_CONFIG, FIRST_CONFIG_SHA256, "first200 source config"),
        (FULL_CONFIG, FULL_CONFIG_SHA256, "full867 source config"),
        (JUDGE_CONFIG, JUDGE_CONFIG_SHA256, "judge config"),
    ):
        _regular_file(path, label=label)
        if _file_sha256(path) != expected:
            raise SupervisorError(f"pinned {label} SHA256 differs")
    # A virtualenv's ``bin/python`` is intentionally a symlink; retain that
    # lexical path so Python discovers the virtualenv rather than the system
    # prefix.
    if not PYTHON.is_file():
        raise SupervisorError(f"Python runtime is missing: {PYTHON}")
    for path, label in (
        (RESCORE_TOOL, "clean semantic-rescore tool"),
        (JUDGE_MODEL / "config.json", "judge model config"),
    ):
        _regular_file(path, label=label)
    observed_commit = _git_output(CLEAN_RUNTIME, "rev-parse", "HEAD")
    if observed_commit != CLEAN_RUNTIME_COMMIT:
        raise SupervisorError("clean RP70 runtime commit differs")
    dirty = _git_output(
        CLEAN_RUNTIME,
        "status",
        "--porcelain",
        "--",
        "src/tgvf_rl",
        "pyproject.toml",
        "requirements/compatibility.lock",
        "requirements/compatibility-torch211-cu129.lock",
        "uv.lock",
    )
    if dirty:
        raise SupervisorError("clean RP70 runtime evaluation paths are dirty")
    judge = json.loads(JUDGE_CONFIG.read_text(encoding="utf-8"))
    if (
        judge.get("model", {}).get("local_path") != str(JUDGE_MODEL)
        or judge.get("model", {}).get("served_name") != JUDGE_SERVED_NAME
        or judge.get("service", {}).get("base_url")
        != f"http://127.0.0.1:{JUDGE_PORT}/v1"
        or tuple(judge.get("service", {}).get("integration_devices", ())) != (0, 1)
    ):
        raise SupervisorError("pinned judge deployment fields differ")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SupervisorError(f"{label} is malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise SupervisorError(f"{label} is not one JSON object: {path}")
    return value


def _validate_launch_plan(
    root: Path, *, samples: int, expected_gpus: Sequence[int]
) -> dict[str, Any]:
    plan = _load_json(root / "launch-plan.json", label="generation launch plan")
    whole = plan.get("whole_preflight")
    assignments = plan.get("assignments")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("output_root") != str(root.resolve())
        or tuple(plan.get("physical_gpu_ids", ())) != tuple(expected_gpus)
        or plan.get("workers_per_gpu") != 1
        or not isinstance(whole, dict)
        or whole.get("status") != "validated"
        or tuple(whole.get("arms", ())) != ARMS
        or whole.get("selected_sample_count") != samples
        or whole.get("training_model_name") != "Qwen3-VL-8B-Instruct"
        or whole.get("production_source_global_step") != 500
        or whole.get("production_source_artifact_sha256")
        != "8fdd20eb96334ff538c381fc95777028fafa0f7d86ccd684fd854c6ed8258b47"
        or not isinstance(assignments, list)
        or len(assignments) != len(expected_gpus)
    ):
        raise SupervisorError(f"generation launch plan identity differs: {root}")
    for index, (assignment, gpu) in enumerate(zip(assignments, expected_gpus)):
        expected_root = root / f"shard-{index:04d}-of-{len(expected_gpus):04d}"
        if (
            not isinstance(assignment, dict)
            or assignment.get("shard_index") != index
            or assignment.get("shard_count") != len(expected_gpus)
            or assignment.get("physical_gpu_id") != gpu
            or assignment.get("output_root") != str(expected_root.resolve())
        ):
            raise SupervisorError(f"generation shard assignment differs: {root}")
    return plan


def _load_complete_generation(
    root: Path, *, samples: int, expected_gpus: Sequence[int]
) -> dict[str, Any] | None:
    plan = _validate_launch_plan(root, samples=samples, expected_gpus=expected_gpus)
    summary_path = root / "launch-summary.json"
    if not summary_path.exists():
        return None
    summary = _load_json(summary_path, label="generation launch summary")
    shards = summary.get("shards")
    if (
        summary.get("schema_version") != GENERATION_SCHEMA
        or summary.get("status") != "complete"
        or tuple(summary.get("arms", ())) != ARMS
        or summary.get("sample_count") != samples
        or summary.get("record_count") != samples * len(ARMS)
        or not isinstance(shards, list)
        or len(shards) != len(expected_gpus)
    ):
        raise SupervisorError(f"generation launch summary differs: {summary_path}")
    assignments = plan["assignments"]
    if sum(int(shard.get("sample_count", -1)) for shard in shards) != samples or sum(
        int(shard.get("record_count", -1)) for shard in shards
    ) != samples * len(ARMS):
        raise SupervisorError(f"generation shard totals differ: {summary_path}")
    for index, (shard, assignment) in enumerate(zip(shards, assignments)):
        expected_root = Path(assignment["output_root"])
        records_path = expected_root / "records.jsonl"
        shard_summary_path = expected_root / "summary.json"
        if (
            not isinstance(shard, dict)
            or shard.get("shard_index") != index
            or shard.get("physical_gpu_id") != assignment["physical_gpu_id"]
            or shard.get("output_root") != str(expected_root)
            or shard.get("record_count") != shard.get("sample_count", -1) * len(ARMS)
        ):
            raise SupervisorError(f"generation shard receipt differs: {summary_path}")
        _regular_file(records_path, label="generation records")
        _regular_file(shard_summary_path, label="generation shard summary")
        if _file_sha256(records_path) != shard.get("records_jsonl_sha256"):
            raise SupervisorError(f"generation records SHA256 differs: {records_path}")
    return summary


def _semantic_complete(root: Path, *, samples: int) -> bool:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return False
    summary = _load_json(summary_path, label="semantic summary")
    manifest = _load_json(manifest_path, label="semantic manifest")
    by_arm = summary.get("by_arm")
    files = manifest.get("files")
    if (
        summary.get("schema_version") != SEMANTIC_SCHEMA
        or manifest.get("schema_version") != SEMANTIC_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("status") != "complete"
        or manifest.get("run_identity_sha256") != summary.get("run_identity_sha256")
        or not isinstance(by_arm, dict)
        or set(by_arm) != set(ARMS)
        or summary.get("overall", {}).get("total") != samples * len(ARMS)
        or any(by_arm.get(arm, {}).get("total") != samples for arm in ARMS)
        or not isinstance(files, dict)
        or files.get("summary", {}).get("sha256") != _file_sha256(summary_path)
        or files.get("overlay_records", {}).get("rows") != samples * len(ARMS)
    ):
        raise SupervisorError(f"semantic publication differs: {root}")
    return True


def _marker_complete() -> bool:
    if not COMPLETE_MARKER.exists():
        return False
    value = _load_json(COMPLETE_MARKER, label="supervisor completion marker")
    if (
        value.get("schema_version") != MARKER_SCHEMA
        or value.get("status") != "complete"
        or value.get("runtime_commit") != CLEAN_RUNTIME_COMMIT
        or value.get("judge_config_sha256") != JUDGE_CONFIG_SHA256
        or not _semantic_complete(FIRST_SEMANTIC, samples=200)
        or not _semantic_complete(FULL_SEMANTIC, samples=867)
    ):
        raise SupervisorError("existing supervisor completion marker drifted")
    return True


def _status() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": MARKER_SCHEMA,
        "status": "waiting",
        "first200_generation_complete": False,
        "full867_generation_complete": False,
        "first200_semantic_complete": False,
        "full867_semantic_complete": False,
        "complete_marker": str(COMPLETE_MARKER),
    }
    try:
        _assert_static_inputs()
        first = _load_complete_generation(
            FIRST_GENERATION, samples=200, expected_gpus=(1, 2)
        )
        full = _load_complete_generation(
            FULL_GENERATION, samples=867, expected_gpus=(3, 4, 5, 6, 7)
        )
        result["first200_generation_complete"] = first is not None
        result["full867_generation_complete"] = full is not None
        result["first200_semantic_complete"] = _semantic_complete(
            FIRST_SEMANTIC, samples=200
        )
        result["full867_semantic_complete"] = _semantic_complete(
            FULL_SEMANTIC, samples=867
        )
        if _marker_complete():
            result["status"] = "complete"
    except (SupervisorError, FileNotFoundError, ValueError) as error:
        result["status"] = "blocked"
        result["blocked_reason"] = str(error)
    return result


def _wait_generations(timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_heartbeat = 0.0
    while True:
        first = _load_complete_generation(
            FIRST_GENERATION, samples=200, expected_gpus=(1, 2)
        )
        full = _load_complete_generation(
            FULL_GENERATION, samples=867, expected_gpus=(3, 4, 5, 6, 7)
        )
        now = time.monotonic()
        if first is not None and full is not None:
            _append_event("generations_complete")
            return first, full
        if now >= deadline:
            raise SupervisorError("timed out waiting for both generation summaries")
        if now - last_heartbeat >= 30:
            _atomic_json(
                STATE,
                {
                    "schema_version": MARKER_SCHEMA,
                    "status": "waiting_for_generation",
                    "updated_at": _utc_now(),
                    "first200_complete": first is not None,
                    "full867_complete": full is not None,
                    "remaining_timeout_seconds": max(0, int(deadline - now)),
                },
            )
            last_heartbeat = now
        time.sleep(5)


def _gpu_compute_pids(indices: Sequence[int]) -> dict[int, tuple[int, ...]]:
    gpu_rows = subprocess.run(
        ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for index, uuid in (row.split(",", 1) for row in gpu_rows)
    }
    result: dict[int, list[int]] = {index: [] for index in indices}
    rows = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    for row in rows:
        if not row.strip():
            continue
        pid_text, uuid = (part.strip() for part in row.split(",", 1))
        index = uuid_to_index.get(uuid)
        if index in result:
            result[index].append(int(pid_text))
    return {index: tuple(sorted(pids)) for index, pids in result.items()}


def _wait_gpus_empty(timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_pids: dict[int, tuple[int, ...]] | None = None
    while True:
        pids = _gpu_compute_pids(JUDGE_GPUS)
        if not any(pids.values()):
            _append_event("judge_gpus_empty", gpu_ids=list(JUDGE_GPUS))
            return
        if pids != last_pids:
            _append_event("waiting_for_judge_gpus", gpu_ids=list(JUDGE_GPUS), pids=pids)
            last_pids = pids
        if time.monotonic() >= deadline:
            raise SupervisorError(f"timed out waiting for GPUs {JUDGE_GPUS}: {pids}")
        time.sleep(5)


def _endpoint_open() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{JUDGE_PORT}/v1/models", timeout=1) as response:
            response.read(1)
        return True
    except Exception:
        return False


def _python_header_cpath() -> str:
    python_headers = PYTHON_HEADER_ROOT / "python3.12"
    required = (
        python_headers / "Python.h",
        python_headers / "pyconfig.h",
        PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h",
    )
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise SupervisorError(f"Python development headers missing: {missing}")
    return os.pathsep.join((str(PYTHON_HEADER_ROOT), str(python_headers)))


def _judge_command() -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(JUDGE_MODEL),
        "--served-model-name",
        JUDGE_SERVED_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(JUDGE_PORT),
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "64",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
    ]


def _judge_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in JUDGE_GPUS),
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
            "PYTHONPATH": str(CLEAN_RUNTIME / "src"),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CPATH": _python_header_cpath(),
            "LIBRARY_PATH": str(REPOSITORY_ROOT / ".venv312/lib"),
            "TRITON_CACHE_DIR": str(SUPERVISOR_ROOT / "cache/judge-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(
                SUPERVISOR_ROOT / "cache/judge-torchinductor"
            ),
        }
    )
    Path(environment["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return environment


def _proc_starttime(pid: int) -> int:
    value = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    closing = value.rfind(")")
    return int(value[closing + 2 :].split()[19])


def _wait_judge_ready(process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SupervisorError(f"judge exited during startup with {process.returncode}")
        try:
            with urlopen(
                f"http://127.0.0.1:{JUDGE_PORT}/v1/models", timeout=2
            ) as response:
                value = json.loads(response.read())
            if {item.get("id") for item in value.get("data", [])} == {
                JUDGE_SERVED_NAME
            }:
                _append_event("judge_ready", pid=process.pid)
                return
        except Exception:
            pass
        time.sleep(2)
    raise SupervisorError("judge readiness timed out")


def _stop_owned_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def _start_judge() -> tuple[subprocess.Popen[bytes], Any]:
    if _endpoint_open():
        raise SupervisorError(f"judge port {JUDGE_PORT} is occupied by an unowned endpoint")
    JUDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = JUDGE_LOG.open("ab", buffering=0)
    process = subprocess.Popen(
        _judge_command(),
        cwd=CLEAN_RUNTIME,
        env=_judge_environment(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ownership = {
        "schema_version": "rp70-owned-judge-process-v1",
        "status": "starting",
        "started_at": _utc_now(),
        "pid": process.pid,
        "pgid": process.pid,
        "proc_starttime_ticks": _proc_starttime(process.pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        "gpu_ids": list(JUDGE_GPUS),
        "port": JUDGE_PORT,
        "command": _judge_command(),
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "judge_config_sha256": JUDGE_CONFIG_SHA256,
    }
    _atomic_json(OWNERSHIP, ownership)
    _append_event("judge_started", pid=process.pid, pgid=process.pid)
    try:
        _wait_judge_ready(process, 600)
    except BaseException:
        _stop_owned_group(process)
        log.close()
        raise
    ownership["status"] = "ready"
    ownership["ready_at"] = _utc_now()
    _atomic_json(OWNERSHIP, ownership)
    return process, log


def _rescore_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(CLEAN_RUNTIME / "src"),
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _run_rescore(
    *,
    name: str,
    generation: Mapping[str, Any],
    source_config: Path,
    output_root: Path,
    samples: int,
    log_path: Path,
) -> None:
    if _semantic_complete(output_root, samples=samples):
        _append_event("semantic_already_complete", split=name)
        return
    command = [str(PYTHON), str(RESCORE_TOOL)]
    shards = generation.get("shards")
    if not isinstance(shards, list) or not shards:
        raise SupervisorError(f"{name} generation has no shards")
    for shard in shards:
        command.extend(("--generation-output-root", str(shard["output_root"])))
    command.extend(
        (
            "--source-evaluation-config",
            str(source_config),
            "--judge-config",
            str(JUDGE_CONFIG),
            "--judge-config-sha256",
            JUDGE_CONFIG_SHA256,
            "--output-root",
            str(output_root),
            "--concurrency",
            "32",
        )
    )
    _append_event("semantic_started", split=name, command=command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=CLEAN_RUNTIME,
            env=_rescore_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=2 * 60 * 60)
        except BaseException:
            _stop_owned_group(process)
            raise
    if return_code != 0 or not _semantic_complete(output_root, samples=samples):
        raise SupervisorError(f"{name} semantic rescore failed; inspect {log_path}")
    _append_event("semantic_complete", split=name)


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    SUPERVISOR_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SupervisorError("another RP70 semantic supervisor is active") from error
        yield


def _install_signal_handlers() -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: object) -> None:
        raise SupervisorInterrupted(f"received signal {signal.Signals(signum).name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: Mapping[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _write_complete_marker(
    first_generation: Mapping[str, Any], full_generation: Mapping[str, Any]
) -> None:
    value = {
        "schema_version": MARKER_SCHEMA,
        "status": "complete",
        "completed_at": _utc_now(),
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "judge_config_sha256": JUDGE_CONFIG_SHA256,
        "judge_gpu_ids": list(JUDGE_GPUS),
        "arms": list(ARMS),
        "first200": {
            "samples": 200,
            "generation": _artifact_record(FIRST_GENERATION / "launch-summary.json"),
            "semantic_summary": _artifact_record(FIRST_SEMANTIC / "summary.json"),
            "semantic_manifest": _artifact_record(FIRST_SEMANTIC / "manifest.json"),
            "run_identity_sha256": json.loads(
                (FIRST_SEMANTIC / "summary.json").read_text(encoding="utf-8")
            )["run_identity_sha256"],
        },
        "full867": {
            "samples": 867,
            "generation": _artifact_record(FULL_GENERATION / "launch-summary.json"),
            "semantic_summary": _artifact_record(FULL_SEMANTIC / "summary.json"),
            "semantic_manifest": _artifact_record(FULL_SEMANTIC / "manifest.json"),
            "run_identity_sha256": json.loads(
                (FULL_SEMANTIC / "summary.json").read_text(encoding="utf-8")
            )["run_identity_sha256"],
        },
        "generation_record_counts": {
            "first200": first_generation["record_count"],
            "full867": full_generation["record_count"],
        },
    }
    _atomic_json(COMPLETE_MARKER, value)


def _execute(generation_timeout: float, gpu_timeout: float) -> dict[str, object]:
    _assert_static_inputs()
    with _exclusive_lock():
        if _marker_complete():
            return _status()
        _append_event("supervisor_started", pid=os.getpid())
        _atomic_json(
            STATE,
            {
                "schema_version": MARKER_SCHEMA,
                "status": "running",
                "started_at": _utc_now(),
                "pid": os.getpid(),
            },
        )
        judge: subprocess.Popen[bytes] | None = None
        judge_log: Any | None = None
        previous = _install_signal_handlers()
        try:
            first, full = _wait_generations(generation_timeout)
            pending = not _semantic_complete(
                FIRST_SEMANTIC, samples=200
            ) or not _semantic_complete(FULL_SEMANTIC, samples=867)
            if pending:
                _wait_gpus_empty(gpu_timeout)
                judge, judge_log = _start_judge()
                _run_rescore(
                    name="first200",
                    generation=first,
                    source_config=FIRST_CONFIG,
                    output_root=FIRST_SEMANTIC,
                    samples=200,
                    log_path=FIRST_RESCORE_LOG,
                )
                _run_rescore(
                    name="full867",
                    generation=full,
                    source_config=FULL_CONFIG,
                    output_root=FULL_SEMANTIC,
                    samples=867,
                    log_path=FULL_RESCORE_LOG,
                )
            if not _semantic_complete(
                FIRST_SEMANTIC, samples=200
            ) or not _semantic_complete(FULL_SEMANTIC, samples=867):
                raise SupervisorError("semantic publications are incomplete")
            _write_complete_marker(first, full)
            _atomic_json(
                STATE,
                {
                    "schema_version": MARKER_SCHEMA,
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "marker": str(COMPLETE_MARKER),
                },
            )
            _append_event("supervisor_complete", marker=str(COMPLETE_MARKER))
        except BaseException as error:
            _atomic_json(
                STATE,
                {
                    "schema_version": MARKER_SCHEMA,
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            _append_event(
                "supervisor_failed", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            if judge is not None:
                _stop_owned_group(judge)
                _append_event("judge_stopped", pid=judge.pid, returncode=judge.returncode)
                if OWNERSHIP.exists():
                    ownership = _load_json(OWNERSHIP, label="judge ownership")
                    ownership["status"] = "stopped"
                    ownership["stopped_at"] = _utc_now()
                    ownership["returncode"] = judge.returncode
                    _atomic_json(OWNERSHIP, ownership)
            if judge_log is not None:
                judge_log.close()
            _restore_signal_handlers(previous)
    return _status()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--generation-timeout-seconds", type=float, default=4 * 60 * 60)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=60 * 60)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.generation_timeout_seconds <= 0 or args.gpu_wait_timeout_seconds <= 0:
        raise SupervisorError("timeouts must be positive")
    if args.status:
        print(json.dumps(_status(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.preflight:
        _assert_static_inputs()
        result = _status()
        # Waiting generation is an accepted preflight state; only a blocked
        # identity is a refusal.
        if result["status"] == "blocked":
            raise SupervisorError(str(result.get("blocked_reason")))
        print(
            json.dumps(
                {
                    **result,
                    "preflight": "passed",
                    "runtime_commit": CLEAN_RUNTIME_COMMIT,
                    "judge_gpu_ids": list(JUDGE_GPUS),
                    "generator_launch_authority": "none",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = _execute(
        args.generation_timeout_seconds, args.gpu_wait_timeout_seconds
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SupervisorError, FileNotFoundError, ValueError) as error:
        print(f"RP70_SEMANTIC_SUPERVISOR_BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
