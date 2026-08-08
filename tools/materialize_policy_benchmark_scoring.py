#!/usr/bin/env python3
"""Materialize identity-bound MCQ scores for an arbitrary policy benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_scoring import (  # noqa: E402
    materialize_policy_benchmark_mcq_scoring,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-sha256", required=True)
    parser.add_argument("--evaluation-identity", type=Path, required=True)
    parser.add_argument("--evaluation-identity-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_policy_benchmark_mcq_scoring(
        inference_root=args.inference_root,
        tasks_path=args.tasks,
        tasks_sha256=args.tasks_sha256,
        evaluation_identity_path=args.evaluation_identity,
        evaluation_identity_file_sha256=args.evaluation_identity_file_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
