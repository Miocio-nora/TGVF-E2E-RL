#!/usr/bin/env python3
# ruff: noqa: E402
"""Plan or execute the immutable T1 deterministic replay audit."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tgvf_rl.data.policy_selection_t1_replay_audit import (
    T1ReplayAuditFailure,
    plan_t1_replay_audit,
    run_t1_replay_audit,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_mode_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--rank", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/audit_policy_data_selection_t1_replay.py",
        selected_mode=args.command,
        read_only_modes=("plan",),
        blocked_modes=("run",),
    )
    if args.command == "plan":
        result = plan_t1_replay_audit(args.config, rank=args.rank).as_record()
    else:
        try:
            result = asyncio.run(run_t1_replay_audit(args.config, rank=args.rank))
        except T1ReplayAuditFailure as exc:
            print(json.dumps(exc.result, indent=2, sort_keys=True))
            raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
