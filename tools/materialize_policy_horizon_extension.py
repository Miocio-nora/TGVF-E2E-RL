#!/usr/bin/env python3
"""Bind one completed bounded Policy run to its next audited horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.policy.horizon_extension import (  # noqa: E402
    materialize_policy_horizon_extension,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    load_policy_e2e_smoke_run_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-id", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        action="append",
        required=True,
        help="repeat in strictly increasing order, including zero and target",
    )
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_policy_e2e_smoke_run_config(args.config)
    extension = materialize_policy_horizon_extension(
        config,
        output_path=args.output,
        extension_id=args.extension_id,
        target_optimizer_step=args.target_step,
        effective_checkpoint_steps=tuple(args.checkpoint_step),
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "extension_id": extension.extension_id,
                "run_id": extension.run_id,
                "source_optimizer_step": extension.source_optimizer_step,
                "target_optimizer_step": extension.target_optimizer_step,
                "source_sha256": extension.source_sha256,
                "gpu_work_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
