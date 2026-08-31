#!/usr/bin/env python3
"""Run or finalize the accepted local semantic judge for T1."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_t1_judge import (
    finalize_t1_scoring,
    publish_t1_semantic_judge_manifest,
    run_t1_semantic_judge,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    publish = subparsers.add_parser(
        "publish",
        help=(
            "foreground index-only closure: validate canonical indices and "
            "referenced evidence paths without reading evidence payloads"
        ),
        description=(
            "Publish judge-v2 using bounded parallel index-only closure. This "
            "validates queue/index identities and requires every referenced "
            "evidence path to be a regular file; it does not read or hash "
            "evidence payload contents. A full evidence audit is separate and "
            "nonblocking."
        ),
    )
    finalize = subparsers.add_parser("finalize")
    for command in (run, publish, finalize):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--judge-config", type=Path, required=True)
    run.add_argument("--concurrency", type=int, default=32)
    publish.add_argument("--workers", type=int, default=32)
    publish.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    if args.command == "run":
        result = asyncio.run(
            run_t1_semantic_judge(
                args.config,
                judge_config_path=args.judge_config,
                concurrency=args.concurrency,
            )
        )
    elif args.command == "publish":
        result = publish_t1_semantic_judge_manifest(
            args.config,
            judge_config_path=args.judge_config,
            workers=args.workers,
            progress_every=args.progress_every,
        )
    else:
        result = finalize_t1_scoring(args.config, judge_config_path=args.judge_config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
