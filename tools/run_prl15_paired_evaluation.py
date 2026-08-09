#!/usr/bin/env python3
"""Wait for, run, resume, and officially score the PRL15 step0/step8 pair."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_paired_tgvf_policy_benchmark_config,
)
from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    check_qwen25_72b_judge,
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
)
from tgvf_rl.evaluation.policy_coredev_scoring import (  # noqa: E402
    DATASETS as COREDEV_DATASETS,
    MODEL_NAME as EVALUATED_MODEL,
    materialize_policy_coredev_scoring_views,
)
from tgvf_rl.evaluation.policy_paired_qwen_materialization import (  # noqa: E402
    materialize_qwen_only_policy_checkpoint,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    load_policy_e2e_smoke_run_config,
)


DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json"
)
RUNNER = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"
COREDEV_RUNNER = REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py"
PLAN_SCHEMA = "tgvf.prl15-paired-policy-benchmark-plan.v2"
PAIR_SUMMARY_SCHEMA = "tgvf.prl15-paired-coredev-summary.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("PRL15 paired evaluation plan schema differs")
    if [arm.get("optimizer_step") for arm in payload.get("arms", ())] != [0, 8]:
        raise ValueError("PRL15 paired evaluation plan must contain step0 and step8")
    if [arm.get("name") for arm in payload["arms"]] != ["step0", "step8"]:
        raise ValueError("PRL15 paired evaluation arm names differ")
    if (
        payload.get("expected_task_count") != 2511
        or payload.get("expected_single_image_count") != 2240
        or payload.get("unsupported_multi_image_count") != 271
    ):
        raise ValueError("PRL15 CoreDev coverage must remain 2,240 + 271")
    scoring = payload.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("PRL15 paired evaluation plan lacks official scoring")
    if tuple(scoring.get("datasets", ())) != COREDEV_DATASETS:
        raise ValueError("PRL15 official seven-suite order differs")
    if scoring.get("evaluated_model") != EVALUATED_MODEL:
        raise ValueError("PRL15 evaluated model name differs")
    if scoring.get("gpt_fallback") is not False:
        raise ValueError("PRL15 official scoring must forbid GPT fallback")
    if (
        not isinstance(scoring.get("run_id_prefix"), str)
        or not scoring["run_id_prefix"].startswith("T")
    ):
        raise ValueError("PRL15 VLMEvalKit run prefix is invalid")
    for field in ("judge_api_nproc", "judge_retry", "judge_timeout_seconds"):
        if type(scoring.get(field)) is not int or scoring[field] <= 0:
            raise ValueError(f"PRL15 scoring {field} must be positive")
    source_root = Path(scoring["source_root"])
    if not source_root.is_absolute() or not source_root.is_dir():
        raise RuntimeError("PRL15 pinned CoreDev source root is unavailable")
    for path_field, digest_field in (
        ("policy_config", "policy_config_sha256"),
        ("task_manifest_path", "task_manifest_sha256"),
    ):
        resolved = _resolve_repo_path(payload[path_field])
        if not resolved.is_file() or _sha256_file(resolved) != payload[digest_field]:
            raise RuntimeError(f"PRL15 plan {path_field} identity differs")
    judge_path = _resolve_repo_path(scoring["judge_config_path"])
    if (
        not judge_path.is_file()
        or _sha256_file(judge_path) != scoring["judge_config_sha256"]
    ):
        raise RuntimeError("PRL15 benchmark judge config identity differs")
    for path_field, digest_field in (
        ("pinned_artifacts_config_path", "pinned_artifacts_config_sha256"),
        ("vlmevalkit_deployment_config_path", "vlmevalkit_deployment_config_sha256"),
        ("mathverse_source_json", "mathverse_source_sha256"),
    ):
        resolved = _resolve_repo_path(scoring[path_field])
        if not resolved.is_file() or _sha256_file(resolved) != scoring[digest_field]:
            raise RuntimeError(f"PRL15 scoring {path_field} identity differs")
    return payload


def _validate_plan_run(plan: dict[str, Any], run: Any) -> None:
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("PRL15 plan protocol is missing")
    expected = {
        "evaluation_protocol": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_profile": run.protocol.tool_profile.value,
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "maximum_tool_calls": run.protocol.maximum_tool_calls,
        "sampling_source": "bound_policy_run_config",
        "same_tasks_and_rank_partition": True,
    }
    if protocol != expected:
        raise RuntimeError("PRL15 plan protocol differs from its policy run")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _step8_sources(run: Any) -> tuple[Path, Path]:
    checkpoint = run.output.checkpoint_directory / "global_step_8"
    pointer = run.output.root / "runtime-policy-state/latest-lora-snapshot.json"
    return checkpoint, pointer


def _wait_for_step8(run: Any, *, timeout_seconds: int, poll_seconds: int) -> None:
    checkpoint, pointer = _step8_sources(run)
    deadline = time.monotonic() + timeout_seconds
    while True:
        latest = run.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
        latest_step = None
        if latest.is_file():
            try:
                latest_step = int(latest.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                latest_step = None
        actor = checkpoint / "actor"
        required = (
            checkpoint / "data.pt",
            actor / "fsdp_config.json",
            actor / "huggingface/config.json",
            actor / "tgvf_policy_checkpoint_pair.json",
            actor / "tgvf_policy_project_state.json",
            *(
                actor
                / f"model_world_size_{run.distributed.world_size}_rank_{rank}.pt"
                for rank in range(run.distributed.world_size)
            ),
        )
        if (
            latest_step == 8
            and checkpoint.is_dir()
            and pointer.is_file()
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
        ):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the complete PRL15 step8 closure")
        time.sleep(poll_seconds)


def _gpu_memory_mib() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError("nvidia-smi GPU-memory output is malformed")
        index, memory = map(int, fields)
        result[index] = memory
    return result


def _wait_for_gpus(
    gpu_ids: tuple[int, ...],
    *,
    timeout_seconds: int,
    poll_seconds: int,
    free_threshold_mib: int,
    stable_polls: int = 2,
) -> None:
    """Wait until training has released every evaluation GPU."""

    if free_threshold_mib < 0 or stable_polls <= 0:
        raise ValueError("GPU availability thresholds are invalid")
    deadline = time.monotonic() + timeout_seconds
    consecutive = 0
    while True:
        memory = _gpu_memory_mib()
        missing = set(gpu_ids).difference(memory)
        if missing:
            raise RuntimeError(f"nvidia-smi omitted requested GPUs: {sorted(missing)}")
        if all(memory[gpu_id] <= free_threshold_mib for gpu_id in gpu_ids):
            consecutive += 1
            if consecutive >= stable_polls:
                return
        else:
            consecutive = 0
        if time.monotonic() >= deadline:
            occupied = {gpu_id: memory[gpu_id] for gpu_id in gpu_ids}
            raise TimeoutError(f"timed out waiting for evaluation GPUs: {occupied}")
        time.sleep(poll_seconds)


def _arm_paths(base: Path, arm: str) -> dict[str, Path]:
    root = base / arm
    return {
        "root": root,
        "config": root / "benchmark-config.json",
        "receipt": root / "runtime/source-paired-snapshot.json",
    }


def _materialize_arm(
    *,
    plan: dict[str, Any],
    run: Any,
    arm: str,
    step: int,
    output_base: Path,
    gpu_ids: tuple[int, int, int, int],
) -> Path:
    paths = _arm_paths(output_base, arm)
    if paths["config"].is_file() and paths["receipt"].is_file():
        config = load_policy_coredev_config(paths["config"])
        load_policy_evaluation_snapshot(config)
        return paths["config"]
    if step == 0:
        qwen_model = Path(run.model.revision_or_path)
        rp66_pointer = None
    else:
        checkpoint, rp66_pointer = _step8_sources(run)
        qwen_model = materialize_qwen_only_policy_checkpoint(
            policy_config_path=_resolve_repo_path(plan["policy_config"]),
            optimizer_step=step,
            checkpoint_path=checkpoint,
            rp66_pointer_path=rp66_pointer,
            bundle_path=paths["root"] / "runtime/qwen-only-bundle",
        )
    materialize_paired_tgvf_policy_benchmark_config(
        evaluation_id=f"{plan['evaluation_id']}-{arm.upper()}",
        policy_config_path=_resolve_repo_path(plan["policy_config"]),
        optimizer_step=step,
        qwen_model_path=qwen_model,
        rp66_pointer_path=rp66_pointer,
        paired_snapshot_receipt_path=paths["receipt"],
        task_manifest_path=plan["task_manifest_path"],
        expected_task_count=plan["expected_task_count"],
        expected_single_image_count=plan["expected_single_image_count"],
        output_root=paths["root"],
        config_path=paths["config"],
        gpu_ids=gpu_ids,
        inference_concurrency_per_gpu=8,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.9,
    )
    return paths["config"]


def _run_checked(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=environment)


def _prepare(config_path: Path) -> None:
    _run_checked([sys.executable, str(RUNNER), "--config", str(config_path), "--mode", "prepare"])


def _validate(config_path: Path) -> None:
    _run_checked(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--mode",
            "validate",
            "--world-size",
            "4",
        ]
    )


def _launch_workers(config_path: Path) -> list[subprocess.Popen[bytes]]:
    config = load_policy_coredev_config(config_path)
    processes: list[subprocess.Popen[bytes]] = []
    for rank, gpu_id in enumerate(config.gpu_ids):
        environment = dict(os.environ)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--mode",
            "worker",
            "--rank",
            str(rank),
            "--world-size",
            "4",
        ]
        processes.append(subprocess.Popen(command, env=environment))
    return processes


def _wait_workers(
    processes: list[subprocess.Popen[bytes]], *, owner: str = "paired evaluation"
) -> None:
    failure: tuple[int, int] | None = None
    while any(process.poll() is None for process in processes):
        for index, process in enumerate(processes):
            code = process.poll()
            if code not in {None, 0}:
                failure = (index, code)
                break
        if failure is not None:
            break
        time.sleep(5)
    if failure is not None:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise RuntimeError(
            f"{owner} worker {failure[0]} exited with {failure[1]}"
        )
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"{owner} workers failed: {codes}")


def _load_judge_config(plan: dict[str, Any]) -> dict[str, Any]:
    scoring = plan["scoring"]
    path = _resolve_repo_path(scoring["judge_config_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("benchmark judge config is not an object")
    model = payload.get("model")
    server = payload.get("server")
    devices = payload.get("devices")
    scope = payload.get("scope")
    if not all(isinstance(value, dict) for value in (model, server, devices, scope)):
        raise RuntimeError("benchmark judge config sections are malformed")
    assert isinstance(model, dict) and isinstance(server, dict)
    assert isinstance(devices, dict) and isinstance(scope, dict)
    if model.get("served_name") != "Qwen2.5-72B-Instruct":
        raise RuntimeError("benchmark judge served model differs")
    if not Path(str(model.get("local_path"))).is_dir():
        raise RuntimeError("benchmark judge local model is unavailable")
    if devices.get("tensor_parallel_size") != 2:
        raise RuntimeError("benchmark judge must remain tensor-parallel two")
    if scope.get("allows_vlmevalkit_benchmark_judging") is not True:
        raise RuntimeError("benchmark judge is not authorized for VLMEvalKit")
    if scope.get("allows_gpt_fallback") is not False:
        raise RuntimeError("GPT fallback must remain disabled")
    return payload


def _judge_command(judge: dict[str, Any]) -> list[str]:
    model = judge["model"]
    server = judge["server"]
    devices = judge["devices"]
    vllm = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm),
        "serve",
        str(model["local_path"]),
        "--served-model-name",
        str(model["served_name"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(server["dtype"]),
        "--tensor-parallel-size",
        str(devices["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--seed",
        str(server["seed"]),
        "--generation-config",
        str(server["generation_config"]),
    ]
    if server.get("prefix_caching") is True:
        command.append("--enable-prefix-caching")
    return command


def _judge_environment(judge: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(value) for value in judge["devices"]["physical"]
    )
    environment["VLLM_ATTENTION_BACKEND"] = str(
        judge["server"]["attention_backend"]
    )
    runtime = judge.get("runtime", {})
    for source, destination in (("cc", "CC"), ("cxx", "CXX"), ("cpath", "CPATH")):
        value = runtime.get(source)
        if isinstance(value, str) and value:
            environment[destination] = value
    return environment


def _wait_for_judge(
    process: subprocess.Popen[bytes],
    judge: dict[str, Any],
    *,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    base_url = str(judge["server"]["base_url"])
    served_name = str(judge["model"]["served_name"])
    last_error: Exception | None = None
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"benchmark judge exited during startup with {code}")
        try:
            check_qwen25_72b_judge(
                base_url=base_url,
                expected_model=served_name,
                timeout=min(30, max(1, poll_seconds)),
            )
            return
        except Exception as error:  # service is expected to reject until ready
            last_error = error
        if time.monotonic() >= deadline:
            raise TimeoutError("benchmark judge readiness timed out") from last_error
        time.sleep(poll_seconds)


@contextmanager
def _local_judge_service(
    judge: dict[str, Any],
    *,
    log_path: Path,
    timeout_seconds: int,
    poll_seconds: int,
):
    try:
        check_qwen25_72b_judge(
            base_url=str(judge["server"]["base_url"]),
            expected_model=str(judge["model"]["served_name"]),
            timeout=2,
        )
    except Exception:
        pass
    else:
        raise RuntimeError("benchmark judge endpoint is already occupied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            _judge_command(judge),
            env=_judge_environment(judge),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_for_judge(
            process,
            judge,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        yield
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _scoring_root(config_path: Path, plan: dict[str, Any]) -> Path:
    config = load_policy_coredev_config(config_path)
    return config.output_root / "scoring" / str(plan["scoring"]["view_name"])


def _materialize_official_scoring_view(
    config_path: Path, plan: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    config = load_policy_coredev_config(config_path)
    root = _scoring_root(config_path, plan)
    summary_path = root / "materialization-summary.json"
    run_id = f"{plan['scoring']['run_id_prefix']}-{arm.upper()}"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "evaluation_id": config.evaluation_id,
            "run_id": run_id,
            "observed_single_image_count": 2240,
            "unsupported_multi_image_count": 271,
            "official_row_count": 2511,
        }
        if not isinstance(result, dict) or any(
            result.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"existing {arm} scoring view identity differs")
        return result
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"partial immutable {arm} scoring view exists without its summary"
        )
    return materialize_policy_coredev_scoring_views(
        inference_root=config.output_root / "inference",
        tasks_path=plan["task_manifest_path"],
        source_root=plan["scoring"]["source_root"],
        output_root=root,
        evaluation_id=config.evaluation_id,
        run_id=run_id,
        mathverse_source_json=plan["scoring"]["mathverse_source_json"],
    )


def _official_score_command(
    *, dataset: str, scoring_root: Path, judge: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    scoring = plan["scoring"]
    return [
        sys.executable,
        str(COREDEV_RUNNER),
        "--data",
        dataset,
        "--model",
        EVALUATED_MODEL,
        "--work-dir",
        str(scoring_root / dataset),
        "--mode",
        "eval",
        "--reuse",
        "--judge",
        str(judge["model"]["served_name"]),
        "--judge-base-url",
        str(judge["server"]["base_url"]),
        "--judge-key",
        "EMPTY",
        "--judge-api-nproc",
        str(scoring["judge_api_nproc"]),
        "--judge-retry",
        str(scoring["judge_retry"]),
        "--judge-timeout",
        str(scoring["judge_timeout_seconds"]),
    ]


def _accepted_official_summary(
    scoring_root: Path, judge: dict[str, Any]
) -> dict[str, Any] | None:
    path = scoring_root / "coredev-2511-eval-summary.json"
    if not path.is_file():
        return None
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
    )
    if result.get("status") != "pass" or result.get("sample_count") != 2511:
        raise RuntimeError("existing official CoreDev summary is not accepted")
    return result


def _score_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    arm: str,
    log_root: Path,
) -> dict[str, Any]:
    scoring_root = _scoring_root(config_path, plan)
    accepted = _accepted_official_summary(scoring_root, judge)
    if accepted is not None:
        return accepted
    processes: list[subprocess.Popen[bytes]] = []
    for dataset in COREDEV_DATASETS:
        log_path = log_root / f"score-{arm}-{dataset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_handle:
            environment = dict(os.environ)
            environment["OPENAI_API_KEY"] = "EMPTY"
            processes.append(
                subprocess.Popen(
                    _official_score_command(
                        dataset=dataset,
                        scoring_root=scoring_root,
                        judge=judge,
                        plan=plan,
                    ),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            )
    _wait_workers(processes, owner=f"{arm} official scorer")
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
    )
    write_json_atomic(scoring_root / "coredev-2511-eval-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--mode", choices=("prepare", "run", "status"), default="run")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--wait-for-step8", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1024)
    parser.add_argument("--judge-startup-timeout-seconds", type=int, default=15 * 60)
    args = parser.parse_args()
    if min(
        args.wait_timeout_seconds,
        args.poll_seconds,
        args.judge_startup_timeout_seconds,
    ) <= 0:
        raise ValueError("evaluation wait durations must be positive")
    if len(args.gpu_ids) not in {4, 8} or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("paired evaluator requires four or eight distinct GPU IDs")
    plan = _load_plan(args.plan.resolve())
    policy_config = _resolve_repo_path(plan["policy_config"])
    run = load_policy_e2e_smoke_run_config(
        policy_config, allow_external_agent_loop_config=True
    )
    _validate_plan_run(plan, run)
    judge = _load_judge_config(plan)
    judge_gpus = tuple(judge["devices"]["physical"])
    if (
        len(judge_gpus) != 2
        or len(set(judge_gpus)) != 2
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in judge_gpus)
    ):
        raise RuntimeError("pinned benchmark judge GPU binding is malformed")
    if any(gpu_id not in args.gpu_ids for gpu_id in judge_gpus):
        raise ValueError("evaluation GPU set must include pinned judge GPUs")
    if args.wait_for_step8:
        _wait_for_step8(
            run,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    if args.wait_for_gpus:
        _wait_for_gpus(
            tuple(args.gpu_ids),
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            free_threshold_mib=args.gpu_free_threshold_mib,
        )
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else run.output.root / "evaluation" / plan["evaluation_id"]
    )
    first_gpus = tuple(args.gpu_ids[:4])
    second_gpus = tuple(args.gpu_ids[-4:])
    step0 = _materialize_arm(
        plan=plan, run=run, arm="step0", step=0, output_base=output_base, gpu_ids=first_gpus
    )
    step8 = _materialize_arm(
        plan=plan, run=run, arm="step8", step=8, output_base=output_base, gpu_ids=second_gpus
    )
    if args.mode == "status":
        for config in (step0, step8):
            _run_checked([sys.executable, str(RUNNER), "--config", str(config), "--mode", "status"])
        return 0
    _prepare(step0)
    _prepare(step8)
    _validate(step0)
    _validate(step8)
    if args.mode == "prepare":
        return 0
    if len(args.gpu_ids) == 8:
        processes = _launch_workers(step0) + _launch_workers(step8)
        _wait_workers(processes)
    else:
        _wait_workers(_launch_workers(step0))
        _wait_workers(_launch_workers(step8))
    for config in (step0, step8):
        _run_checked(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(config),
                "--mode",
                "status",
                "--world-size",
                "4",
            ]
        )
    materialization = {
        "step0": _materialize_official_scoring_view(step0, plan, arm="step0"),
        "step8": _materialize_official_scoring_view(step8, plan, arm="step8"),
    }
    log_root = output_base / "logs"
    existing = {
        "step0": _accepted_official_summary(_scoring_root(step0, plan), judge),
        "step8": _accepted_official_summary(_scoring_root(step8, plan), judge),
    }
    if any(value is None for value in existing.values()):
        with _local_judge_service(
            judge,
            log_path=log_root / "judge-qwen25-72b.log",
            timeout_seconds=args.judge_startup_timeout_seconds,
            poll_seconds=max(5, min(args.poll_seconds, 30)),
        ):
            for arm, config in (("step0", step0), ("step8", step8)):
                if existing[arm] is None:
                    existing[arm] = _score_arm(
                        config,
                        plan,
                        judge,
                        arm=arm,
                        log_root=log_root,
                    )
    report = {
        "schema_version": PAIR_SUMMARY_SCHEMA,
        "evaluation_id": plan["evaluation_id"],
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "multi_image_policy": "unsupported_explicit_hold",
        },
        "materialization": materialization,
        "step0": existing["step0"],
        "step8": existing["step8"],
    }
    report_path = output_base / "paired-summary.json"
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
