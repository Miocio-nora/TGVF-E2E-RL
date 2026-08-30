#!/usr/bin/env python3
"""Prepare, execute, and inspect the accepted Qwen3 T1 canary."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.data.policy_selection_vllm import (  # noqa: E402
    prepare_output_root,
    run_t1_worker,
    t1_status,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_mode_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument(
        "--cuda-visible-device",
        type=int,
        help=(
            "physical CUDA device selected in CUDA_VISIBLE_DEVICES; defaults "
            "to the logical rank"
        ),
    )
    worker.add_argument("--budget-revision", type=int, choices=(0,), default=0)
    worker.add_argument("--max-chunks", type=int)
    worker.add_argument("--chunk-subshard-count", type=int, default=1)
    worker.add_argument("--chunk-subshard-index", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/run_policy_data_selection_t1.py",
        selected_mode=args.command,
        read_only_modes=("status",),
        blocked_modes=("prepare", "worker"),
    )
    if args.command == "prepare":
        result = prepare_output_root(args.config)
    elif args.command == "status":
        result = t1_status(args.config)
    else:
        result = asyncio.run(
            run_t1_worker(
                args.config,
                rank=args.rank,
                cuda_visible_device=args.cuda_visible_device,
                budget_revision=args.budget_revision,
                max_chunks=args.max_chunks,
                chunk_subshard_count=args.chunk_subshard_count,
                chunk_subshard_index=args.chunk_subshard_index,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
