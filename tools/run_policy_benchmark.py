#!/usr/bin/env python3
"""Run one completed visual-tool policy LoRA on a bound task manifest."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.contracts.errors import PolicyOutputContractError  # noqa: E402
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    CoreDevTask,
    PolicyCoreDevConfig,
    PolicyCoreDevEvaluator,
    PolicyEvaluationSnapshot,
    build_standalone_manager,
    freeze_policy_evaluation_snapshot,
    load_bound_policy_benchmark_tasks,
    load_frozen_policy_evaluation_snapshot,
    load_policy_benchmark_results,
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
    materialize_vllm_lora_adapter,
    policy_output_contract_failure_audit_payload,
    prepare_policy_benchmark_tasks,
    trajectory_audit_payload,
    validate_policy_benchmark_runtime_interfaces,
    write_policy_evaluation_identity,
)
from tgvf_rl.evaluation.policy_official_visible import (  # noqa: E402
    OfficialVisiblePolicyEvaluator,
    official_visible_trajectory_audit_payload,
    validate_official_visible_processor,
)
from tgvf_rl.evaluation.policy_no_tool_matched import (  # noqa: E402
    NoToolMatchedPolicyEvaluator,
)
from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    FullModelEvaluationSnapshot,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("prepare", "validate", "worker", "status"), required=True
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--max-tasks", type=int, default=-1)
    return parser


def _world_size(config: PolicyCoreDevConfig, requested: int | None) -> int:
    expected = len(config.gpu_ids)
    if requested is not None and requested != expected:
        raise ValueError("world size must equal the number of configured GPUs")
    return expected


def _assert_worker_cuda_binding(config: PolicyCoreDevConfig, rank: int) -> None:
    expected = str(config.gpu_ids[rank])
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("worker requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError(
            f"rank {rank} requires CUDA_VISIBLE_DEVICES={expected} exactly"
        )


def _append_durable(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - operating-system failure guard
                raise OSError("short write while appending policy benchmark result")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _rank_lock(output_root: Path, rank: int):
    lock_path = output_root / "runtime" / "locks" / f"rank-{rank}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"policy benchmark rank {rank} is already active"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def _worker(args: argparse.Namespace, config: PolicyCoreDevConfig) -> int:
    world_size = _world_size(config, args.world_size)
    if args.rank is None or not 0 <= args.rank < world_size:
        raise ValueError("worker mode requires --rank in [0, world-size)")
    _assert_worker_cuda_binding(config, args.rank)
    with _rank_lock(config.output_root, args.rank):
        tasks = load_bound_policy_benchmark_tasks(config)
        snapshot = load_frozen_policy_evaluation_snapshot(config)
        evaluation_identity = write_policy_evaluation_identity(config, snapshot)
        records = load_policy_benchmark_results(
            config.output_root / "inference",
            tasks=tasks,
            evaluation_identity=evaluation_identity,
        )
        result_path = config.output_root / "inference" / f"rank-{args.rank}.jsonl"
        selected = [
            task
            for task in tasks
            if task.single_image
            and task.ordinal % world_size == args.rank
            and task.ordinal not in records
        ]
        if args.max_tasks >= 0:
            selected = selected[: args.max_tasks]
        if not selected:
            print(json.dumps({"rank": args.rank, "remaining": 0}, sort_keys=True))
            return 0

        # Importing transformers after CUDA_VISIBLE_DEVICES is fixed keeps every
        # evaluator process on exactly one physical GPU.
        from transformers import AutoProcessor

        manager, engine, run = await build_standalone_manager(config, snapshot)
        processor = AutoProcessor.from_pretrained(
            run.model.revision_or_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        no_tool_full_model = (
            isinstance(snapshot, FullModelEvaluationSnapshot)
            and config.evaluation_protocol
            != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
            and run.schema_version
            == POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA
        )
        evaluator = (
            NoToolMatchedPolicyEvaluator(
                config=config,
                run=run,
                manager=manager,
                processor=processor,
                snapshot=snapshot,
                evaluation_identity=evaluation_identity,
            )
            if no_tool_full_model
            else OfficialVisiblePolicyEvaluator(
                config=config,
                run=run,
                manager=manager,
                processor=processor,
                snapshot=snapshot,
                evaluation_identity=evaluation_identity,
            )
            if config.evaluation_protocol
            == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
            else PolicyCoreDevEvaluator(
                config=config,
                run=run,
                manager=manager,
                processor=processor,
                snapshot=snapshot,
                evaluation_identity=evaluation_identity,
            )
        )
        started = time.time()
        jobs: list[asyncio.Task[None]] = []

        async def evaluate_one(local_index: int, task: CoreDevTask) -> None:
            row_started = time.time()
            try:
                trajectory = await evaluator.evaluate(task)
            except PolicyOutputContractError as error:
                if (
                    config.evaluation_protocol
                    == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                ):
                    raise
                trajectory = None
                payload = policy_output_contract_failure_audit_payload(
                    task,
                    error,
                    evaluation_identity=evaluation_identity,
                    rank=args.rank,
                    world_size=world_size,
                )
            else:
                payload = (
                    official_visible_trajectory_audit_payload(
                        task,
                        trajectory,
                        evaluation_identity=evaluation_identity,
                        rank=args.rank,
                        world_size=world_size,
                    )
                    if config.evaluation_protocol
                    == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                    else trajectory_audit_payload(
                        task,
                        trajectory,
                        evaluation_identity=evaluation_identity,
                        rank=args.rank,
                        world_size=world_size,
                    )
                )
            payload["wall_seconds"] = time.time() - row_started
            _append_durable(result_path, payload)
            print(
                json.dumps(
                    {
                        "rank": args.rank,
                        "done": local_index,
                        "selected": len(selected),
                        "ordinal": task.ordinal,
                        "dataset": task.dataset,
                        "tool_calls": (
                            len(trajectory.tool_calls) if trajectory is not None else 0
                        ),
                        "stop": (
                            getattr(trajectory.stop, "value", trajectory.stop)
                            if trajectory is not None
                            else "invalid_format"
                        ),
                        "result_kind": payload.get("result_kind", "trajectory"),
                        "wall_seconds": payload["wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        try:
            for batch_start in range(
                0, len(selected), config.inference_concurrency_per_gpu
            ):
                batch = selected[
                    batch_start : batch_start + config.inference_concurrency_per_gpu
                ]
                jobs = [
                    asyncio.create_task(evaluate_one(batch_start + offset + 1, task))
                    for offset, task in enumerate(batch)
                ]
                try:
                    await asyncio.gather(*jobs)
                except BaseException:
                    for job in jobs:
                        if not job.done():
                            job.cancel()
                    await asyncio.gather(*jobs, return_exceptions=True)
                    raise
                finally:
                    jobs = []
        finally:
            engine.shutdown()
            await asyncio.sleep(0)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "completed": len(selected),
                    "wall_seconds": time.time() - started,
                },
                sort_keys=True,
            )
        )
        return 0


def _status(config: PolicyCoreDevConfig, requested_world_size: int | None) -> int:
    tasks = load_bound_policy_benchmark_tasks(config)
    _world_size(config, requested_world_size)
    snapshot = load_frozen_policy_evaluation_snapshot(config)
    evaluation_identity = write_policy_evaluation_identity(config, snapshot)
    completed = set(
        load_policy_benchmark_results(
            config.output_root / "inference",
            tasks=tasks,
            evaluation_identity=evaluation_identity,
        )
    )
    single = {task.ordinal for task in tasks if task.single_image}
    print(
        json.dumps(
            {
                "evaluation_id": config.evaluation_id,
                "completed_single_image": len(completed & single),
                "total_single_image": len(single),
                "remaining_single_image": len(single - completed),
                "multi_image_pending_protocol_decision": len(tasks) - len(single),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate(config: PolicyCoreDevConfig, requested_world_size: int | None) -> int:
    """Static-only validation; this never constructs vLLM or touches CUDA."""

    tasks = load_bound_policy_benchmark_tasks(config)
    _world_size(config, requested_world_size)
    snapshot = load_frozen_policy_evaluation_snapshot(config)
    identity = write_policy_evaluation_identity(config, snapshot)
    result: dict[str, object] = {
        "evaluation_id": config.evaluation_id,
        "evaluation_identity_sha256": identity["identity_sha256"],
        "task_count": len(tasks),
        "single_image_count": sum(task.single_image for task in tasks),
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "policy_weights_sha256": snapshot.policy_version.weights_sha256,
        "evaluation_protocol": config.evaluation_protocol,
        "gpu_or_api_used": False,
        "vllm_engine_constructed": False,
    }
    if config.evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        result["runtime_interface_preflight"] = (
            validate_policy_benchmark_runtime_interfaces(snapshot.run)
        )
    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            snapshot.run.model.revision_or_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        result["official_visible_processor_proof"] = (
            validate_official_visible_processor(
                processor,
                tokenizer_length=snapshot.run.model.tokenizer_length,
                image_max_pixels=snapshot.run.policy.image_max_pixels,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    config = load_policy_coredev_config(args.config)
    if args.mode == "prepare":
        counts = prepare_policy_benchmark_tasks(config)
        source_snapshot = load_policy_evaluation_snapshot(config)
        snapshot = freeze_policy_evaluation_snapshot(config, source_snapshot)
        if isinstance(snapshot, PolicyEvaluationSnapshot):
            materialize_vllm_lora_adapter(config, snapshot)
        write_policy_evaluation_identity(config, snapshot)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0
    if args.mode == "status":
        return _status(config, args.world_size)
    if args.mode == "validate":
        return _validate(config, args.world_size)
    return asyncio.run(_worker(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
