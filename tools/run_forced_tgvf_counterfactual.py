#!/usr/bin/env python3
"""Validate, run, resume, or finalize forced-representation TGVF attempts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/answer_utility/evaluation/"
    "rp66_step2000_full867_gpu7.toml"
)
DEFAULT_JUDGE_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/judges/openrouter_qwen25_72b_formal_pilot_judge_v4.json"
)


def _common(parser: argparse.ArgumentParser, *, include_model: bool) -> None:
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--attempts-per-sample", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=4)
    if include_model:
        parser.add_argument(
            "--source-evaluation-config", type=Path, default=DEFAULT_SOURCE_CONFIG
        )
        parser.add_argument("--judge-config", type=Path, default=DEFAULT_JUDGE_CONFIG)
        parser.add_argument("--master-seed", type=int, default=42)
        parser.add_argument("--max-new-tokens", type=int, default=40_960)
        parser.add_argument("--eos-token-id", type=int, action="append")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate image+selected-representation-D counterfactual answer "
            "attempts over the exact TGVF-80 schedule."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="CPU-only identity preflight")
    _common(validate, include_model=True)

    run = commands.add_parser(
        "run-shard", help="run/resume one deterministic GPU shard"
    )
    _common(run, include_model=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--physical-gpu-id", type=int, required=True)

    finalize = commands.add_parser(
        "finalize", help="merge complete shard ledgers into attempts.jsonl"
    )
    _common(finalize, include_model=False)
    return parser


def _enter_gpu_environment(gpu_id: int) -> None:
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("physical GPU ID must be non-negative")
    required = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    marker = "TGVF_FORCED_COUNTERFACTUAL_LAUNCH_READY"
    if os.environ.get(marker) == "1":
        mismatches = {
            name: (os.environ.get(name), expected)
            for name, expected in required.items()
            if os.environ.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"forced-TGVF re-exec environment differs: {mismatches}")
        return
    conflicts = {
        name: (os.environ.get(name), expected)
        for name, expected in required.items()
        if os.environ.get(name) is not None and os.environ.get(name) != expected
    }
    if conflicts:
        raise ValueError(f"forced-TGVF launch environment conflicts: {conflicts}")
    environment = dict(os.environ)
    environment.update(required)
    environment[marker] = "1"
    os.execve(sys.executable, (sys.executable, *sys.argv), environment)
    raise RuntimeError("forced-TGVF launcher re-exec unexpectedly returned")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run-shard":
        _enter_gpu_environment(args.physical_gpu_id)
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from tgvf_rl.data.forced_tgvf_counterfactual import (
        DEFAULT_EOS_TOKEN_IDS,
        build_forced_tgvf_run_plan,
        finalize_forced_tgvf_attempts,
        run_forced_tgvf_shard,
    )

    if args.command == "finalize":
        result = finalize_forced_tgvf_attempts(
            args.schedule_root,
            args.output_root,
            run_id=args.run_id,
            sample_count=args.sample_count,
            attempts_per_sample=args.attempts_per_sample,
            shard_count=args.shard_count,
        )
    else:
        eos = (
            DEFAULT_EOS_TOKEN_IDS
            if args.eos_token_id is None
            else tuple(args.eos_token_id)
        )
        keywords = {
            "run_id": args.run_id,
            "sample_count": args.sample_count,
            "attempts_per_sample": args.attempts_per_sample,
            "shard_count": args.shard_count,
            "master_seed": args.master_seed,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_ids": eos,
        }
        if args.command == "validate":
            plan = build_forced_tgvf_run_plan(
                args.schedule_root,
                args.source_evaluation_config,
                args.judge_config,
                **keywords,
            )
            result = {"status": "validated", **plan.as_record()}
        else:
            result = run_forced_tgvf_shard(
                args.schedule_root,
                args.source_evaluation_config,
                args.judge_config,
                args.output_root,
                shard_index=args.shard_index,
                **keywords,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
