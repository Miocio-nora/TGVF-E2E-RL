#!/usr/bin/python3 -I
"""Run blind semantic rescoring over completed answer-utility generations."""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/run_representation_answer_utility_semantic_rescore.py",
        ),
    )

import argparse
import asyncio
import json
from pathlib import Path
import sys

from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


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
    assert_legacy_standalone_execution_quarantined(
        "tools/run_representation_answer_utility_semantic_rescore.py"
    )
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
