#!/usr/bin/env python3
"""Materialize VLMEvalKit scoring views for a completed policy CoreDev arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_coredev_scoring import (  # noqa: E402
    materialize_policy_coredev_scoring_views,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mathverse-source-json", type=Path, required=True)
    parser.add_argument("--force-invalid-index", action="append", default=[])
    args = parser.parse_args()
    result = materialize_policy_coredev_scoring_views(
        inference_root=args.inference_root,
        tasks_path=args.tasks,
        source_root=args.source_root,
        output_root=args.output_root,
        evaluation_id=args.evaluation_id,
        run_id=args.run_id,
        mathverse_source_json=args.mathverse_source_json,
        forced_invalid_indices=args.force_invalid_index,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
