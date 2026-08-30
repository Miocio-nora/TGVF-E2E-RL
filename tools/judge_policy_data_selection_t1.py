#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Run or finalize the accepted local semantic judge for T1."""

from __future__ import annotations

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
            "tools/judge_policy_data_selection_t1.py",
        ),
    )

import argparse
import asyncio
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tgvf_rl.data.policy_selection_t1_judge import (
    finalize_t1_scoring,
    run_t1_semantic_judge,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


def main() -> None:
    assert_legacy_standalone_execution_quarantined(
        "tools/judge_policy_data_selection_t1.py"
    )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    finalize = subparsers.add_parser("finalize")
    for command in (run, finalize):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--judge-config", type=Path, required=True)
    run.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.command == "run":
        result = asyncio.run(
            run_t1_semantic_judge(
                args.config,
                judge_config_path=args.judge_config,
                concurrency=args.concurrency,
            )
        )
    else:
        result = finalize_t1_scoring(args.config, judge_config_path=args.judge_config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
