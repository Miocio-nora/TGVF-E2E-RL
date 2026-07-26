#!/usr/bin/env python3
"""Run or finalize the accepted local semantic judge for T1."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_t1_judge import (
    finalize_t1_scoring,
    run_t1_semantic_judge,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    finalize = subparsers.add_parser("finalize")
    for command in (run, finalize):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--judge-config", type=Path, required=True)
    run.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.command == "run":
        result = asyncio.run(
            run_t1_semantic_judge(
                args.config,
                judge_config_path=args.judge_config,
                concurrency=args.concurrency,
            )
        )
    else:
        result = finalize_t1_scoring(
            args.config, judge_config_path=args.judge_config
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
