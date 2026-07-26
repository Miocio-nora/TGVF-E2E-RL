#!/usr/bin/env python3
"""Plan or execute the immutable T1 deterministic replay audit."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_t1_replay_audit import (
    T1ReplayAuditFailure,
    plan_t1_replay_audit,
    run_t1_replay_audit,
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
    if args.command == "plan":
        result = plan_t1_replay_audit(args.config, rank=args.rank).as_record()
    else:
        try:
            result = asyncio.run(
                run_t1_replay_audit(args.config, rank=args.rank)
            )
        except T1ReplayAuditFailure as exc:
            print(json.dumps(exc.result, indent=2, sort_keys=True))
            raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
