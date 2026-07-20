"""Create or verify the seven CoreDev-2511 VLMEvalKit TSV slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_materialize import (  # noqa: E402
    materialize_coredev_2511,
)


DEFAULT_MANIFEST = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/"
    "benchmark_manifests/core_balanced_dev_2511_seed20260625.json"
)
DEFAULT_BENCHMARK_ROOT = Path("/nvmesv/dredvpn009/datasets/benchmarks")
DEFAULT_OUTPUT_ROOT = DEFAULT_BENCHMARK_ROOT / "coredev_2511_vlmevalkit_7055d301_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = materialize_coredev_2511(
        manifest_path=args.manifest.resolve(),
        benchmark_root=args.benchmark_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
