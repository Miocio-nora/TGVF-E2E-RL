#!/usr/bin/env python3
"""Run blind semantic rescoring over completed answer-utility generations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reapply the pinned deterministic scorer, judge only unresolved answers, "
            "and publish a separate immutable diagnostic overlay."
        )
    )
    parser.add_argument(
        "--generation-output-root",
        type=Path,
        action="append",
        required=True,
        help="Completed generation output root; repeat for multiple checkpoints.",
    )
    parser.add_argument("--source-evaluation-config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--judge-config-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from tgvf_rl.representation.experiments.answer_utility.evaluation.semantic_rescore import (  # noqa: E501
        run_semantic_rescore,
    )

    result = asyncio.run(
        run_semantic_rescore(
            args.generation_output_root,
            args.source_evaluation_config,
            args.judge_config,
            expected_judge_config_sha256=args.judge_config_sha256,
            output_root=args.output_root,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
