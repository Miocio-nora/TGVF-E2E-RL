#!/usr/bin/env python3
"""Materialize every final T1 retain from V*, ArxivQA, and ThinkLite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_t1_mixed_rl_dataset import (
    T1_04_EXPECTED_SOURCE_COUNTS,
    materialize_policy_t1_mixed_retained_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--expected-vstar-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["vstar"],
    )
    parser.add_argument(
        "--expected-arxivqa-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["arxivqa"],
    )
    parser.add_argument(
        "--expected-thinklite-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["thinklite"],
    )
    args = parser.parse_args()
    result = materialize_policy_t1_mixed_retained_pool(
        args.candidates,
        args.final_manifest,
        args.output_root,
        shuffle_seed=args.shuffle_seed,
        expected_source_counts={
            "vstar": args.expected_vstar_count,
            "arxivqa": args.expected_arxivqa_count,
            "thinklite": args.expected_thinklite_count,
        },
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
