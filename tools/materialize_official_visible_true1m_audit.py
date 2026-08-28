#!/usr/bin/env python3
"""Create an immutable post-hoc true1M receipt for one completed Crop arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_true1m_audit import (  # noqa: E402
    materialize_official_visible_true1m_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--rng-reference-receipt", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize_official_visible_true1m_audit(
        config_path=args.config,
        plan_path=args.plan,
        arm_name=args.arm,
        rng_reference_receipt_path=args.rng_reference_receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
