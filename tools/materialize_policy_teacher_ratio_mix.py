#!/usr/bin/env python3
"""Materialize an immutable Teacher25/50/100 policy schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgvf_rl.data.policy_teacher_ratio_mix import (  # noqa: E402
    POLICY_TEACHER_RATIO_MIX_DEFAULT_SCHEDULE_INDEX,
    POLICY_TEACHER_RATIO_MIX_DEFAULT_TEACHER_ROOT,
    POLICY_TEACHER_RATIO_MIX_SEED,
    POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES,
    materialize_policy_teacher_ratio_mix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--teacher-percentage",
        required=True,
        type=int,
        choices=sorted(POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES),
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=POLICY_TEACHER_RATIO_MIX_DEFAULT_TEACHER_ROOT,
    )
    parser.add_argument(
        "--schedule-index",
        type=Path,
        default=POLICY_TEACHER_RATIO_MIX_DEFAULT_SCHEDULE_INDEX,
    )
    parser.add_argument("--seed", type=int, default=POLICY_TEACHER_RATIO_MIX_SEED)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize_policy_teacher_ratio_mix(
        args.output_root,
        teacher_percentage=args.teacher_percentage,
        teacher_root=args.teacher_root,
        schedule_index_path=args.schedule_index,
        schedule_seed=args.seed,
    )
    print(json.dumps(result.as_record(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
