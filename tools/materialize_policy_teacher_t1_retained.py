#!/usr/bin/env python3
"""Publish the independent teacher rows retained by final T1 scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_teacher_t1_retained import (
    materialize_teacher_t1_retained,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()
    result = materialize_teacher_t1_retained(
        args.config,
        args.final_manifest,
        args.output_root,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
