"""Verify or materialize the pinned DeepEyes-47K snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.data import (  # noqa: E402
    materialize_deepeyes47k,
    verify_deepeyes47k_source_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or materialize the pinned DeepEyes-47K parquet snapshot"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="required integer run-identity input when materializing",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    if args.verify_only:
        if args.output_root is not None:
            parser.error("--output-root is not used with --verify-only")
        result = verify_deepeyes47k_source_files(source_root).as_record()
    else:
        if args.output_root is None:
            parser.error("--output-root is required unless --verify-only is set")
        if args.shuffle_seed is None:
            parser.error("--shuffle-seed must be explicitly provided")
        result = materialize_deepeyes47k(
            source_root,
            args.output_root.resolve(),
            shuffle_seed=args.shuffle_seed,
        ).as_record()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
