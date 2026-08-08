#!/usr/bin/env python3
"""Materialize the SHA-bound compact PRL13 schedule index once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_T1_SAMPLE_COUNT,
    load_deepeyes_official_t1_pool,
)
from tgvf_rl.data.deepeyes_official_schedule_index import (
    DEEPEYES_SCHEDULE_INDEX_PATH,
    write_deepeyes_schedule_index,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Create-only output. Use a temporary path for review, then copy the "
            f"validated bytes to {DEEPEYES_SCHEDULE_INDEX_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = perf_counter()
    pool_started = perf_counter()
    samples = load_deepeyes_official_t1_pool()
    pool_seconds = perf_counter() - pool_started
    if len(samples) != DEEPEYES_T1_SAMPLE_COUNT:
        raise RuntimeError("validated parent pool sample count differs")
    index_started = perf_counter()
    file_sha256, identity_sha256, byte_count = write_deepeyes_schedule_index(
        samples, args.output
    )
    index_seconds = perf_counter() - index_started
    print(
        json.dumps(
            {
                "schema_version": "tgvf.prl13-schedule-index-materialization.v1",
                "temporary_output": str(args.output.resolve(strict=True)),
                "canonical_destination": str(DEEPEYES_SCHEDULE_INDEX_PATH),
                "parent_pool_rows_verified": len(samples),
                "indexed_rows": 20_480 + 256 + 4,
                "file_bytes": byte_count,
                "file_sha256": file_sha256,
                "identity_sha256": identity_sha256,
                "parent_validation_seconds": pool_seconds,
                "index_materialization_seconds": index_seconds,
                "total_seconds": perf_counter() - started,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
