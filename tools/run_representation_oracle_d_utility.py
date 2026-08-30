#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Run or inspect the resumable Stage1 oracle-target D utility evaluation."""

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
            "tools/run_representation_oracle_d_utility.py",
        ),
    )

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.representation.training.oracle_d_utility import (
    DEFAULT_ORACLE_D_UTILITY_ARMS,
    DEFAULT_THINKING_EOS_TOKEN_IDS,
    OracleDUtilityArm,
    run_oracle_d_utility_evaluation,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-VL-8B-Thinking Stage1 D under an oracle trajectory "
            "target. One process owns one visible GPU and one image-group shard."
        )
    )
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        choices=tuple(arm.value for arm in OracleDUtilityArm),
        help="repeat to override the default five-arm evaluation",
    )
    parser.add_argument(
        "--eos-token-id",
        action="append",
        type=int,
        help=(
            "repeat to override Thinking EOS IDs; defaults to both 151645 and "
            "151643 and is checked against the local generation config"
        ),
    )
    parser.add_argument(
        "--decode-mode", choices=("cached", "no_cache"), default="cached"
    )
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main() -> None:
    assert_legacy_standalone_execution_quarantined(
        "tools/run_representation_oracle_d_utility.py"
    )
    args = _parser().parse_args()
    result = run_oracle_d_utility_evaluation(
        args.source_config,
        output_root=args.output_root,
        arms=(
            tuple(args.arm) if args.arm is not None else DEFAULT_ORACLE_D_UTILITY_ARMS
        ),
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=(
            tuple(args.eos_token_id)
            if args.eos_token_id is not None
            else DEFAULT_THINKING_EOS_TOKEN_IDS
        ),
        decode_mode=args.decode_mode,
        group_start=args.group_start,
        group_limit=args.group_limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
