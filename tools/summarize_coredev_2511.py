"""Validate and aggregate the seven independently scored CoreDev-2511 slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    summarize_coredev_results,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("infer", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-base-url")
    args = parser.parse_args()

    summary = summarize_coredev_results(
        work_dir=args.work_dir.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase=args.phase,
        expected_judge_base_url=args.judge_base_url,
    )
    write_json_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
