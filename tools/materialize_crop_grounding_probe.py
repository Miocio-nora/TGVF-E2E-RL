#!/usr/bin/env python3
"""Freeze an image-disjoint held-out VStar/SEAL Crop grounding probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.crop_grounding_probe import (  # noqa: E402
    materialize_crop_grounding_probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--training-samples", type=Path, required=True)
    parser.add_argument("--training-samples-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--per-stratum", type=int, default=50)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize_crop_grounding_probe(
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        training_samples_path=args.training_samples,
        training_samples_sha256=args.training_samples_sha256,
        output_root=args.output_root,
        seed=args.seed,
        per_stratum=args.per_stratum,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
