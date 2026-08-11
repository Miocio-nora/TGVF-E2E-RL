#!/usr/bin/env python3
"""Publish a strict completed prefix from a larger forced-TGVF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate full-run identity and shard ledgers, then publish a "
            "complete canonical training-index prefix."
        )
    )
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prefix-sample-count", type=int, required=True)
    parser.add_argument("--attempts-per-sample", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from tgvf_rl.data.forced_tgvf_prefix import finalize_forced_tgvf_prefix

    result = finalize_forced_tgvf_prefix(
        args.schedule_root,
        args.output_root,
        run_id=args.run_id,
        prefix_sample_count=args.prefix_sample_count,
        attempts_per_sample=args.attempts_per_sample,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
