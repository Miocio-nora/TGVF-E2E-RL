#!/usr/bin/env python3
"""Bind an official-visible task suite to one exact full-model snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_full_model_policy_benchmark_config,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    TRAINING_RUN_EVALUATION_PROTOCOL,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-optimizer-step", type=int, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--expected-single-image-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--inference-concurrency-per-gpu", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--evaluation-protocol",
        choices=(
            DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
            TRAINING_RUN_EVALUATION_PROTOCOL,
        ),
        default=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs=4,
        metavar=("GPU0", "GPU1", "GPU2", "GPU3"),
        default=(0, 1, 2, 3),
        help="Four distinct physical GPU IDs used by this evaluation arm.",
    )
    args = parser.parse_args()
    payload = materialize_full_model_policy_benchmark_config(
        evaluation_id=args.evaluation_id,
        policy_config_path=args.policy_config,
        snapshot_manifest_path=args.snapshot_manifest,
        materialization_receipt_path=args.materialization_receipt,
        expected_optimizer_step=args.expected_optimizer_step,
        task_manifest_path=args.tasks,
        expected_task_count=args.expected_task_count,
        expected_single_image_count=args.expected_single_image_count,
        output_root=args.output_root,
        config_path=args.config_output,
        inference_concurrency_per_gpu=args.inference_concurrency_per_gpu,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_ids=tuple(args.gpu_ids),
        evaluation_protocol=args.evaluation_protocol,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
