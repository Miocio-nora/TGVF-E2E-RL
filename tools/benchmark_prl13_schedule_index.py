#!/usr/bin/env python3
"""Benchmark legacy PRL13 pool reconstruction against the compact index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

from tgvf_rl.data.deepeyes_official_schedule import (
    build_deepeyes_schedule,
    load_deepeyes_official_t1_pool,
)
from tgvf_rl.data.deepeyes_official_schedule_index import (
    load_deepeyes_schedule_index,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-file-sha256", required=True)
    parser.add_argument("--index-identity-sha256", required=True)
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Measure only indexed startup when a repeated all-image scan is undesirable.",
    )
    return parser.parse_args()


def _rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    args = _parse_args()
    report: dict[str, object] = {
        "schema_version": "tgvf.prl13-schedule-index-benchmark.v1",
        "index": str(args.index.resolve(strict=True)),
    }
    if not args.skip_legacy:
        started = perf_counter()
        samples = load_deepeyes_official_t1_pool()
        schedule = build_deepeyes_schedule(samples, mode="stratified")
        report["legacy"] = {
            "elapsed_seconds": perf_counter() - started,
            "population_rows": len(samples),
            "schedule_identity_sha256": schedule.identity_sha256,
            "maximum_rss_mib": _rss_mib(),
        }
        del schedule, samples
    started = perf_counter()
    index = load_deepeyes_schedule_index(
        args.index,
        expected_file_sha256=args.index_file_sha256,
        expected_identity_sha256=args.index_identity_sha256,
    )
    report["indexed"] = {
        "elapsed_seconds": perf_counter() - started,
        "indexed_rows": len(index.train) + len(index.probe) + len(index.smoke),
        "schedule_identity_sha256": index.schedule_identity_sha256,
        "maximum_rss_mib": _rss_mib(),
        "images_hashed_during_index_load": 0,
    }
    legacy = report.get("legacy")
    if isinstance(legacy, dict):
        if legacy["schedule_identity_sha256"] != index.schedule_identity_sha256:
            raise RuntimeError("legacy and indexed schedule identities differ")
        report["speedup"] = legacy["elapsed_seconds"] / report["indexed"]["elapsed_seconds"]
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
