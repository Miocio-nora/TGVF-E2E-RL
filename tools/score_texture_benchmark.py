#!/usr/bin/env python3
"""Score complete LAS&T/MMAD result JSONLs with the pinned local protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.texture_bench.io import write_json_idempotent  # noqa: E402
from tgvf_rl.evaluation.texture_bench.schema import file_sha256  # noqa: E402
from tgvf_rl.evaluation.texture_bench.scoring import (  # noqa: E402
    load_result_rows,
    score_texture_benchmark,
)
from tgvf_rl.evaluation.texture_bench.task import load_texture_tasks  # noqa: E402


SCORED_ARTIFACT_SCHEMA = "tgvf-texture-benchmark-scored-artifact-v1"


def score_paths(
    *,
    tasks_path: str | Path,
    result_paths: Sequence[str | Path],
    verify_images: bool = True,
) -> dict[str, object]:
    """Load an exact complete result set and return its provenance-bound score."""

    task_source = Path(tasks_path).expanduser().resolve(strict=True)
    results = tuple(
        Path(path).expanduser().resolve(strict=True) for path in result_paths
    )
    if not results:
        raise ValueError("at least one result JSONL is required")
    tasks = load_texture_tasks(task_source, verify_images=verify_images)
    records = load_result_rows(results)
    task_manifest_sha256 = file_sha256(task_source)
    score = score_texture_benchmark(
        tasks,
        records,
        task_manifest_sha256=task_manifest_sha256,
    )
    return {
        "schema_version": SCORED_ARTIFACT_SCHEMA,
        "task_manifest": {
            "path": str(task_source),
            "sha256": task_manifest_sha256,
            "task_count": len(tasks),
        },
        "results": [
            {"path": str(path), "sha256": file_sha256(path)} for path in results
        ],
        "score": score,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rehash and decode every bound task image before scoring.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_paths(
        tasks_path=args.tasks,
        result_paths=args.results,
        verify_images=args.verify_images,
    )
    if args.output is not None:
        write_json_idempotent(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
