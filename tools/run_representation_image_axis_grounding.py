#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate or launch the isolated RP66 image-axis grounding treatment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.representation.experiments.image_axis_grounding.runner import (
    run_image_axis_grounding_experiment,
    validate_image_axis_grounding_experiment,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_mode_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all CPU/data/identity checks without initializing CUDA.",
    )
    parser.add_argument(
        "--stop-after-global-step",
        type=int,
        default=None,
        help="Pause at an exact optimizer boundary (for smoke or interruption).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_legacy_standalone_mode_quarantined(
        "tools/run_representation_image_axis_grounding.py",
        selected_mode="validate" if args.validate_only else "execute",
        read_only_modes=("validate",),
        blocked_modes=("execute",),
    )
    if args.stop_after_global_step is not None and args.stop_after_global_step < 1:
        raise ValueError("--stop-after-global-step must be positive")
    if args.validate_only:
        if args.stop_after_global_step is not None:
            raise ValueError("--validate-only cannot be combined with a stop step")
        result = validate_image_axis_grounding_experiment(args.config)
    else:
        result = run_image_axis_grounding_experiment(
            args.config,
            stop_after_global_step=args.stop_after_global_step,
        )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and result is not None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
