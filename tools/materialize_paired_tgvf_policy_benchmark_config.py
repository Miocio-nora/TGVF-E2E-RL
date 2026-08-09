#!/usr/bin/env python3
"""Materialize one full-Qwen plus RP66 benchmark arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_paired_tgvf_policy_benchmark_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--rp66-pointer", type=Path)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--expected-single-image-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=4, default=(0, 1, 2, 3))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    args = parser.parse_args()
    payload = materialize_paired_tgvf_policy_benchmark_config(
        evaluation_id=args.evaluation_id,
        policy_config_path=args.policy_config,
        optimizer_step=args.optimizer_step,
        qwen_model_path=args.qwen_model,
        rp66_pointer_path=args.rp66_pointer,
        paired_snapshot_receipt_path=args.snapshot_receipt,
        task_manifest_path=args.task_manifest,
        expected_task_count=args.expected_task_count,
        expected_single_image_count=args.expected_single_image_count,
        output_root=args.output_root,
        config_path=args.config,
        inference_concurrency_per_gpu=args.concurrency,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_ids=tuple(args.gpu_ids),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
