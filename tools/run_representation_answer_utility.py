#!/usr/bin/env python3
"""Validate, run, or exactly resume the isolated answer-utility experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - legacy local launcher only
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one removable RP66 answer-utility trainable cell on one "
            "visible GPU. Use --validate-only before allocating CUDA."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stop-after-global-step", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _physical_gpu_id(path: Path) -> int:
    payload = tomllib.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("answer-utility run sidecar has no [execution] table")
    value = execution.get("physical_gpu_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("execution.physical_gpu_id must be non-negative")
    return value


def _enter_launch_environment(physical_gpu_id: int) -> None:
    required = {
        "CUDA_VISIBLE_DEVICES": str(physical_gpu_id),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    mismatches = {
        name: (observed, expected)
        for name, expected in required.items()
        if (observed := os.environ.get(name)) is not None and observed != expected
    }
    if mismatches:
        raise ValueError(f"launch environment conflicts with sidecar: {mismatches}")
    marker = "TGVF_ANSWER_UTILITY_LAUNCH_READY"
    if os.environ.get(marker) == "1":
        if any(os.environ.get(name) != value for name, value in required.items()):
            raise ValueError("answer-utility re-exec environment is incomplete")
        return
    environment = dict(os.environ)
    environment.update(required)
    environment[marker] = "1"
    os.execve(
        sys.executable,
        (sys.executable, *sys.argv),
        environment,
    )
    raise RuntimeError("answer-utility launcher re-exec unexpectedly returned")


def main() -> None:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"answer-utility run config is missing: {config_path}")
    if not args.validate_only:
        _enter_launch_environment(_physical_gpu_id(config_path))

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from tgvf_rl.representation.experiments.answer_utility.runner import (  # noqa: E402
        run_answer_utility_experiment,
        validate_answer_utility_experiment,
    )

    resume = (
        None
        if args.resume_checkpoint is None
        else args.resume_checkpoint.expanduser().resolve()
    )
    if args.validate_only:
        if args.stop_after_global_step is not None:
            raise ValueError("--validate-only cannot carry a stop step")
        result = validate_answer_utility_experiment(
            config_path,
            resume_checkpoint_path=resume,
        )
    else:
        result = run_answer_utility_experiment(
            config_path,
            resume_checkpoint_path=resume,
            stop_after_global_step=args.stop_after_global_step,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
