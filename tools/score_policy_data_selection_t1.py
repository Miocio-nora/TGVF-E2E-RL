#!/usr/bin/env python3
"""Materialize deterministic T1 scores and the local semantic-judge queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_t1_scoring import (
    materialize_t1_deterministic_scoring,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--quality-exclusions", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_t1_deterministic_scoring(
        args.config,
        judge_config_path=args.judge_config,
        quality_exclusions_path=args.quality_exclusions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
