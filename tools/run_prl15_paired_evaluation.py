#!/usr/bin/env python3
"""Prepare, run, resume, and score the PRL15 step0/step8 paired benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_paired_tgvf_policy_benchmark_config,
)
from tgvf_rl.evaluation.policy_benchmark_scoring import (  # noqa: E402
    materialize_policy_benchmark_mcq_scoring,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    load_policy_e2e_smoke_run_config,
)


DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/evaluation/prl15_rp66_step0_step8_coredev2511_plan.json"
)
RUNNER = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "tgvf.prl15-paired-policy-benchmark-plan.v1"
    ):
        raise ValueError("PRL15 paired evaluation plan schema differs")
    if [arm.get("optimizer_step") for arm in payload.get("arms", ())] != [0, 8]:
        raise ValueError("PRL15 paired evaluation plan must contain step0 and step8")
    return payload


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _step8_sources(run: Any) -> tuple[Path, Path]:
    checkpoint = run.output.checkpoint_directory / "global_step_8"
    model_path = checkpoint / "actor/huggingface"
    pointer = run.output.root / "runtime-policy-state/latest-lora-snapshot.json"
    return model_path, pointer


def _wait_for_step8(run: Any, *, timeout_seconds: int, poll_seconds: int) -> None:
    model_path, pointer = _step8_sources(run)
    deadline = time.monotonic() + timeout_seconds
    while True:
        latest = run.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
        latest_step = None
        if latest.is_file():
            try:
                latest_step = int(latest.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                latest_step = None
        if latest_step == 8 and model_path.is_dir() and pointer.is_file():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the complete PRL15 step8 closure")
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
        qwen_model, rp66_pointer = _step8_sources(run)
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


def _wait_workers(processes: list[subprocess.Popen[bytes]]) -> None:
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
            f"paired evaluation worker {failure[0]} exited with {failure[1]}"
        )
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"paired evaluation workers failed: {codes}")


def _score(config_path: Path, plan: dict[str, Any]) -> dict[str, object]:
    config = load_policy_coredev_config(config_path)
    identity = config.output_root / "runtime/evaluation-identity.json"
    return materialize_policy_benchmark_mcq_scoring(
        inference_root=config.output_root / "inference",
        tasks_path=plan["task_manifest_path"],
        tasks_sha256=plan["task_manifest_sha256"],
        evaluation_identity_path=identity,
        evaluation_identity_file_sha256=_sha256_file(identity),
        output_root=config.output_root / "scoring",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--mode", choices=("prepare", "run", "status"), default="run")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--wait-for-step8", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if len(args.gpu_ids) not in {4, 8} or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("paired evaluator requires four or eight distinct GPU IDs")
    plan = _load_plan(args.plan.resolve())
    policy_config = _resolve_repo_path(plan["policy_config"])
    run = load_policy_e2e_smoke_run_config(
        policy_config, allow_external_agent_loop_config=True
    )
    if args.wait_for_step8:
        _wait_for_step8(
            run,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
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
    if args.mode == "prepare":
        return 0
    if len(args.gpu_ids) == 8:
        processes = _launch_workers(step0) + _launch_workers(step8)
        _wait_workers(processes)
    else:
        _wait_workers(_launch_workers(step0))
        _wait_workers(_launch_workers(step8))
    report = {"step0": _score(step0, plan), "step8": _score(step8, plan)}
    report_path = output_base / "paired-summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
