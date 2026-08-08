#!/usr/bin/env python3
"""Materialize the immutable V*/HR4K/HR8K DeepEyesDev591 suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.deepeyes_dev import materialize_deepeyes_dev591  # noqa: E402


DEFAULT_COREDEV_ROOT = Path(
    "/nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/"
    "DeepEyesDev591-seed20260625-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vstar-tsv", type=Path, default=DEFAULT_COREDEV_ROOT / "VStarBench.tsv"
    )
    parser.add_argument(
        "--hrbench4k-tsv", type=Path, default=DEFAULT_COREDEV_ROOT / "HRBench4K.tsv"
    )
    parser.add_argument(
        "--hrbench8k-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "sources" / "hr_bench_8k.parquet",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = materialize_deepeyes_dev591(
        vstar_tsv=args.vstar_tsv,
        hrbench4k_tsv=args.hrbench4k_tsv,
        hrbench8k_parquet=args.hrbench8k_parquet,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
