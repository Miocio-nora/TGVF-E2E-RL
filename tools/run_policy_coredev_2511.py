#!/usr/bin/env python3
"""Run one completed policy LoRA through native tools on CoreDev-2511."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    PolicyCoreDevEvaluator,
    build_standalone_manager,
    load_coredev_tasks,
    load_policy_coredev_config,
    materialize_vllm_lora_adapter,
    trajectory_audit_payload,
    write_official_coredev_tasks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("prepare", "worker", "status"), required=True
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=-1)
    return parser


def _append_durable(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            payload = json.loads(line)
            ordinal = payload["ordinal"]
        except Exception as error:
            raise RuntimeError(f"invalid result line {line_number} in {path}") from error
        if type(ordinal) is not int or ordinal in completed:
            raise RuntimeError(f"duplicate/invalid result ordinal in {path}")
        completed.add(ordinal)
    return completed


async def _worker(args: argparse.Namespace) -> int:
    if args.rank is None or not 0 <= args.rank < args.world_size:
        raise ValueError("worker mode requires --rank in [0, world-size)")
    if args.world_size != 4:
        raise ValueError("formal policy CoreDev evaluation requires world size 4")
    config = load_policy_coredev_config(args.config)
    task_path = config.output_root / "runtime" / "coredev-official-tasks.jsonl"
    tasks = load_coredev_tasks(task_path)
    result_path = config.output_root / "inference" / f"rank-{args.rank}.jsonl"
    completed = _completed(result_path)
    selected = [
        task
        for task in tasks
        if task.single_image
        and task.ordinal % args.world_size == args.rank
        and task.ordinal not in completed
    ]
    if args.max_tasks >= 0:
        selected = selected[: args.max_tasks]
    if not selected:
        print(json.dumps({"rank": args.rank, "remaining": 0}, sort_keys=True))
        return 0

    # Importing transformers after CUDA_VISIBLE_DEVICES is fixed keeps every
    # evaluator process on exactly one physical B200.
    from transformers import AutoProcessor

    manager, engine, run = await build_standalone_manager(config)
    processor = AutoProcessor.from_pretrained(
        run.model.revision_or_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    evaluator = PolicyCoreDevEvaluator(
        config=config, run=run, manager=manager, processor=processor
    )
    started = time.time()
    try:
        for local_index, task in enumerate(selected, 1):
            row_started = time.time()
            trajectory = await evaluator.evaluate(task)
            payload = trajectory_audit_payload(task, trajectory)
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
                        "tool_calls": len(trajectory.tool_calls),
                        "stop": trajectory.stop.value,
                        "wall_seconds": payload["wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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


def _status(config_path: Path) -> int:
    config = load_policy_coredev_config(config_path)
    tasks = load_coredev_tasks(
        config.output_root / "runtime" / "coredev-official-tasks.jsonl"
    )
    completed: set[int] = set()
    for rank in range(4):
        completed.update(
            _completed(config.output_root / "inference" / f"rank-{rank}.jsonl")
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


def main() -> int:
    args = _parser().parse_args()
    config = load_policy_coredev_config(args.config)
    if args.mode == "prepare":
        materialize_vllm_lora_adapter(config)
        counts = write_official_coredev_tasks(
            config.output_root / "runtime" / "coredev-official-tasks.jsonl"
        )
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0
    if args.mode == "status":
        return _status(args.config)
    return asyncio.run(_worker(args))


if __name__ == "__main__":
    raise SystemExit(main())
