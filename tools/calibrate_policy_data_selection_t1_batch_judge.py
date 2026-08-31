#!/usr/bin/env python3
"""Calibrate candidate-batched T1 verdicts against completed strict indices."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_t1_batch_calibration import (
    run_candidate_batch_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-root", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--scan-limit", type=int, default=20_000)
    parser.add_argument("--minimum-items", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--protocol",
        choices=("rationale-v1", "compact-v2", "compact-v3", "compact-v4"),
        default="rationale-v1",
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_candidate_batch_calibration(
            scoring_root=args.scoring_root,
            strict_root=args.strict_root,
            judge_config_path=args.judge_config,
            output_root=args.output_root,
            candidate_count=args.candidate_count,
            scan_limit=args.scan_limit,
            minimum_items=args.minimum_items,
            concurrency=args.concurrency,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            protocol=args.protocol,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
