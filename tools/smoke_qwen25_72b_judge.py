"""Validate the pinned Qwen2.5-72B OpenAI-compatible judge service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.evaluation.coredev_results import (
    check_qwen25_72b_judge,
    write_json_atomic,
)


EXPECTED_MODEL = "Qwen2.5-72B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_qwen25_72b_judge(
        base_url=args.base_url,
        expected_model=EXPECTED_MODEL,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
