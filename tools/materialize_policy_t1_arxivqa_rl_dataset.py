#!/usr/bin/env python3
"""Materialize prompt-free retained ArxivQA rows for the crop Policy pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data import (
    PolicyT1DecisionStage,
    materialize_policy_t1_arxivqa_rl_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--decision-stage",
        choices=tuple(stage.value for stage in PolicyT1DecisionStage),
        required=True,
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()
    result = materialize_policy_t1_arxivqa_rl_dataset(
        args.candidates,
        args.decisions,
        args.output_root,
        decision_stage=PolicyT1DecisionStage(args.decision_stage),
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
