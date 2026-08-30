#!/usr/bin/env python3
"""Plan or execute length-only budget retries for the accepted T1 run."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.data.policy_selection_vllm_retry import (  # noqa: E402
    run_t1_length_retry_worker,
    t1_length_retry_status,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_mode_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--budget-revision", type=int, required=True)
    plan.add_argument("--rank", type=int)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--budget-revision", type=int, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--physical-gpu", type=int, required=True)
    worker.add_argument("--expected-request-id", action="append", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/run_policy_data_selection_t1_retry.py",
        selected_mode=args.command,
        read_only_modes=("plan",),
        blocked_modes=("worker",),
    )
    if args.command == "plan":
        result = t1_length_retry_status(
            args.config, budget_revision=args.budget_revision, rank=args.rank
        )
    else:
        result = asyncio.run(
            run_t1_length_retry_worker(
                args.config,
                rank=args.rank,
                physical_gpu=args.physical_gpu,
                budget_revision=args.budget_revision,
                expected_request_ids=args.expected_request_id,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
