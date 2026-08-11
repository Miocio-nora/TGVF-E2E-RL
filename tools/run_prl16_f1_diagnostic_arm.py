#!/usr/bin/env python3
"""Prepare and run one resumable PRL16-F1 diagnostic CoreDev arm."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = REPOSITORY_ROOT / "tools/run_prl15_paired_evaluation.py"
DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/evaluation/prl16_f1_frozen_rp66_step0_step1_step2_coredev2511_plan.json"
)


def _load_executor():
    spec = importlib.util.spec_from_file_location("prl16_f1_evaluator", EXECUTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load PRL16-F1 evaluation executor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=4, required=True)
    args = parser.parse_args()

    executor = _load_executor()
    plan = executor._load_plan(args.plan.resolve())
    arm_steps = {arm["name"]: arm["optimizer_step"] for arm in plan["arms"]}
    if args.arm not in arm_steps:
        raise ValueError(f"unknown diagnostic arm: {args.arm}")
    if len(set(args.gpu_ids)) != 4:
        raise ValueError("diagnostic arm requires four distinct GPUs")
    policy_config = executor._resolve_repo_path(plan["policy_config"])
    run = executor.load_policy_e2e_smoke_run_config(
        policy_config, allow_external_agent_loop_config=True
    )
    executor._validate_plan_run(plan, run)
    output_base = run.output.root / "evaluation" / plan["evaluation_id"]
    config = executor._materialize_arm(
        plan=plan,
        run=run,
        arm=args.arm,
        step=arm_steps[args.arm],
        output_base=output_base,
        gpu_ids=tuple(args.gpu_ids),
    )
    executor._prepare_prevalidated_bound_manifest(config)
    executor._validate(config)
    executor._wait_workers(executor._launch_workers(config), owner=args.arm)
    executor._run_checked(
        [
            sys.executable,
            str(executor.RUNNER),
            "--config",
            str(config),
            "--mode",
            "status",
            "--world-size",
            "4",
        ]
    )
    executor._materialize_official_scoring_view(config, plan, arm=args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
