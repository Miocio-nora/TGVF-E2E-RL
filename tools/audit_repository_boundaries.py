#!/usr/bin/env python3
"""Audit reusable code and canonical configuration repository boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.repository_boundaries import (  # noqa: E402
    REPOSITORY_BOUNDARY_AUDIT_SCHEMA,
    RepositoryBoundaryError,
    audit_repository_boundaries,
)


DEFAULT_POLICY_PATH = Path("configs/ops/repository_boundary_policy.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository tree to audit (default: the script's repository).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Boundary policy path, relative to --repository-root by default.",
    )
    parser.add_argument(
        "--baseline-policy",
        type=Path,
        help="Optional base policy whose module-size ceilings may not be relaxed.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_repository_boundaries(
            args.repository_root,
            args.policy,
            baseline_policy_path=args.baseline_policy,
        )
    except (OSError, RepositoryBoundaryError) as error:
        failure = {
            "schema_version": REPOSITORY_BOUNDARY_AUDIT_SCHEMA,
            "status": "error",
            "error": str(error),
        }
        print(
            json.dumps(
                failure,
                ensure_ascii=False,
                indent=None if args.compact else 2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            report.as_record(),
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
