#!/usr/bin/env python3
"""Prepare deterministic LAS&T and MMAD single-image benchmark manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.evaluation.texture_bench.io import (  # noqa: E402
    build_benchmark_identity,
    reindex_task_rows,
    write_json_idempotent,
    write_jsonl_idempotent,
)
from tgvf_rl.evaluation.texture_bench.last import (  # noqa: E402
    LAST_DATASET_NAME,
    LAST_DEFAULT_PROMPT_PROFILE,
    LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
    LAST_DEFAULT_SEED,
    LAST_EXPECTED_IDENTITY_COUNT,
    LAST_EXPECTED_IMAGES_PER_IDENTITY,
    LAST_EXPECTED_SOURCE_IMAGE_SIZE,
    LAST_PNG_COMPRESSION_LEVEL,
    LAST_PROMPT_PROFILES,
    materialize_last_rows,
)
from tgvf_rl.evaluation.texture_bench.mmad import (  # noqa: E402
    MMAD_OFFICIAL_QUESTION_COUNT,
    MMAD_PINNED_JSON_SHA256,
    MMAD_PNG_COMPRESSION_LEVEL,
    build_mmad_task_rows,
    canonical_mmad_manifest_bytes,
    validate_mmad_task_rows,
)
from tgvf_rl.evaluation.texture_bench.schema import file_sha256  # noqa: E402
from tgvf_rl.evaluation.texture_bench.task import (  # noqa: E402
    load_texture_tasks,
)


SUMMARY_SCHEMA = "tgvf-texture-benchmark-preparation-summary-v1"
DATASET_ROOT = Path("/nvmesv/dredvpn009/datasets/benchmarks")
DEFAULT_LAST_SOURCE = DATASET_ROOT / "las_t_2d_texture_retrieval" / "snapshot"
DEFAULT_MMAD_SOURCE = DATASET_ROOT / "mmad" / "snapshot"
DEFAULT_PREPARED_ROOT = DATASET_ROOT / "texture_evaluation_v1"
DEFAULT_LAST_OUTPUT = DEFAULT_PREPARED_ROOT / "last-paper400-neutral-v1"
DEFAULT_MMAD_ONE_SHOT_OUTPUT = DEFAULT_PREPARED_ROOT / "mmad-1shot-random"
DEFAULT_MMAD_ZERO_SHOT_OUTPUT = DEFAULT_PREPARED_ROOT / "mmad-0shot"
DEFAULT_SUITE_OUTPUT = DEFAULT_PREPARED_ROOT / "suite-last-paper400-mmad-1shot-random"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _resolved_output(
    requested: Path | None, *, default: Path, prefix: int | None
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if prefix is None:
        return default.resolve()
    return default.with_name(f"{default.name}-prefix-{prefix}").resolve()


def _source_binding(source_root: Path) -> dict[str, object]:
    binding: dict[str, object] = {"snapshot_root": str(source_root)}
    deployment_path = source_root.parent / "DEPLOYMENT.json"
    if deployment_path.is_file() and not deployment_path.is_symlink():
        binding["deployment"] = {
            "path": str(deployment_path.resolve(strict=True)),
            "sha256": file_sha256(deployment_path),
        }
    return binding


def _write_prepared_artifacts(
    *,
    command: str,
    benchmark_id: str,
    output_root: Path,
    rows: Sequence[Mapping[str, object]],
    components: Mapping[str, object],
    prefix: int | None,
) -> dict[str, object]:
    tasks_path = output_root / "tasks.jsonl"
    task_artifact = write_jsonl_idempotent(tasks_path, rows)
    identity = build_benchmark_identity(
        benchmark_id=benchmark_id,
        tasks_path=tasks_path,
        task_count=len(rows),
        components=components,
    )
    identity_artifact = write_json_idempotent(output_root / "identity.json", identity)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "command": command,
        "benchmark_id": benchmark_id,
        "output_root": str(output_root),
        "prefix": prefix,
        "task_count": len(rows),
        "tasks": task_artifact,
        "identity": {
            **identity_artifact,
            "identity_sha256": identity["identity_sha256"],
        },
    }


def _prepare_last(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source.expanduser().resolve(strict=True)
    output_root = _resolved_output(
        args.output,
        default=DEFAULT_LAST_OUTPUT,
        prefix=args.prefix,
    )
    rows = materialize_last_rows(
        source_root,
        output_root,
        prompt_profile=args.prompt_profile,
        quizzes_per_directory=args.quizzes_per_directory,
        seed=args.seed,
        max_samples=args.prefix,
        expected_identity_count=args.expected_identities,
        expected_images_per_identity=args.expected_images_per_identity,
        expected_image_size=(args.expected_width, args.expected_height),
    )
    full_count = 8 * args.quizzes_per_directory
    expected_count = full_count if args.prefix is None else min(args.prefix, full_count)
    if len(rows) != expected_count:
        raise RuntimeError("LAS&T materialized task count differs")
    rows = tuple(reindex_task_rows(rows))
    return _write_prepared_artifacts(
        command="last",
        benchmark_id=(
            f"last-paper-{args.quizzes_per_directory}-"
            f"{args.prompt_profile}-seed-{args.seed}"
        ),
        output_root=output_root,
        rows=rows,
        prefix=args.prefix,
        components={
            "dataset": "LAS&T - 2D Texture Retrieval",
            "source": _source_binding(source_root),
            "protocol": {
                "physical_test_directories": 8,
                "quizzes_per_directory": args.quizzes_per_directory,
                "prompt_profile": args.prompt_profile,
                "seed": args.seed,
                "training_directory_excluded": True,
                "composite_size": [1024, 1024],
                "lossless_png": True,
                "png_compression_level": LAST_PNG_COMPRESSION_LEVEL,
            },
            "source_contract": {
                "identities_per_directory": args.expected_identities,
                "images_per_identity": args.expected_images_per_identity,
                "native_image_size": [args.expected_width, args.expected_height],
            },
        },
    )


def _default_mmad_output(*, shot: int, template_kind: str) -> Path:
    if shot == 0:
        return DEFAULT_MMAD_ZERO_SHOT_OUTPUT
    if template_kind == "random":
        return DEFAULT_MMAD_ONE_SHOT_OUTPUT
    return DEFAULT_PREPARED_ROOT / f"mmad-1shot-{template_kind}"


def _prepare_mmad(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source.expanduser().resolve(strict=True)
    output_root = _resolved_output(
        args.output,
        default=_default_mmad_output(
            shot=args.shot,
            template_kind=args.template_kind,
        ),
        prefix=args.prefix,
    )
    rows = build_mmad_task_rows(
        snapshot_root=source_root,
        shot=args.shot,
        canvas_root=output_root / "canvases" if args.shot == 1 else None,
        template_kind=args.template_kind,
        stable_prefix=args.prefix,
        verify_official_source=not args.allow_unpinned_source,
    )
    expected_count = (
        MMAD_OFFICIAL_QUESTION_COUNT if args.prefix is None else args.prefix
    )
    if not args.allow_unpinned_source and len(rows) != expected_count:
        raise RuntimeError("MMAD materialized task count differs")
    validate_mmad_task_rows(rows, expected_count=len(rows), verify_images=False)
    rows = tuple(reindex_task_rows(rows))
    annotation_path = source_root / "mmad.json"
    annotation_sha256 = file_sha256(annotation_path)
    if not args.allow_unpinned_source and annotation_sha256 != MMAD_PINNED_JSON_SHA256:
        raise RuntimeError("MMAD annotation SHA256 differs after materialization")
    manifest_sha256 = hashlib.sha256(canonical_mmad_manifest_bytes(rows)).hexdigest()
    effective_template = args.template_kind if args.shot == 1 else "none"
    benchmark_id = (
        f"mmad-{args.shot}shot-{effective_template}" if args.shot == 1 else "mmad-0shot"
    )
    return _write_prepared_artifacts(
        command="mmad",
        benchmark_id=benchmark_id,
        output_root=output_root,
        rows=rows,
        prefix=args.prefix,
        components={
            "dataset": "MMAD",
            "source": {
                **_source_binding(source_root),
                "annotation_path": str(annotation_path.resolve(strict=True)),
                "annotation_sha256": annotation_sha256,
                "pinned_source_required": not args.allow_unpinned_source,
            },
            "protocol": {
                "shot": args.shot,
                "template_kind": effective_template,
                "question_count": len(rows),
                "single_image_per_question": True,
                "mask_supplied_to_model": False,
                "one_shot_layout": (
                    "normal_template_left__query_right"
                    if args.shot == 1
                    else "query_only"
                ),
                "one_shot_canvas_size": [1048, 560] if args.shot == 1 else None,
                "png_compression_level": (
                    MMAD_PNG_COMPRESSION_LEVEL if args.shot == 1 else None
                ),
                "manifest_sha256_before_write": manifest_sha256,
            },
        },
    )


def _read_manifest_rows(path: Path) -> tuple[dict[str, object], ...]:
    source = path.expanduser().resolve(strict=True)
    tasks = load_texture_tasks(source, verify_images=False)
    try:
        rows = tuple(
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"task manifest is unreadable: {source}") from error
    if len(rows) != len(tasks) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"task manifest rows are malformed: {source}")
    return rows


def _prepared_benchmark_id(manifest: Path, *, fallback: str) -> str:
    """Reuse a component's declared identity without trusting it implicitly."""

    identity_path = manifest.parent / "identity.json"
    if not identity_path.is_file() or identity_path.is_symlink():
        return f"{fallback}-{file_sha256(manifest)[:12]}"
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return f"{fallback}-{file_sha256(manifest)[:12]}"
    if not isinstance(payload, dict):
        return f"{fallback}-{file_sha256(manifest)[:12]}"
    binding = payload.get("task_manifest")
    benchmark_id = payload.get("benchmark_id")
    if (
        not isinstance(binding, dict)
        or binding.get("path") != str(manifest)
        or binding.get("sha256") != file_sha256(manifest)
        or not isinstance(benchmark_id, str)
        or not benchmark_id
    ):
        return f"{fallback}-{file_sha256(manifest)[:12]}"
    return benchmark_id


def _prepare_suite(args: argparse.Namespace) -> dict[str, object]:
    last_manifest = args.last_manifest.expanduser().resolve(strict=True)
    anomaly_manifest = args.mmad_manifest.expanduser().resolve(strict=True)
    last_rows = _read_manifest_rows(last_manifest)
    anomaly_rows = _read_manifest_rows(anomaly_manifest)
    if args.prefix is not None:
        last_rows = last_rows[: args.prefix]
        anomaly_rows = anomaly_rows[: args.prefix]
    if not last_rows or not anomaly_rows:
        raise ValueError("suite requires non-empty LAS&T and MMAD manifests")
    if {row["dataset"] for row in last_rows} != {LAST_DATASET_NAME}:
        raise ValueError("suite LAS&T manifest has an unexpected dataset name")
    if {row["dataset"] for row in anomaly_rows} != {"MMAD"}:
        raise ValueError("suite MMAD manifest has an unexpected dataset name")
    output_root = _resolved_output(
        args.output,
        default=DEFAULT_SUITE_OUTPUT,
        prefix=args.prefix,
    )
    rows = tuple(reindex_task_rows((*last_rows, *anomaly_rows)))
    last_benchmark_id = _prepared_benchmark_id(last_manifest, fallback="last")
    anomaly_benchmark_id = _prepared_benchmark_id(anomaly_manifest, fallback="mmad")
    return _write_prepared_artifacts(
        command="suite",
        benchmark_id=f"{last_benchmark_id}__{anomaly_benchmark_id}",
        output_root=output_root,
        rows=rows,
        prefix=args.prefix,
        components={
            "order": ["last", "mmad"],
            "reindexed": True,
            "prefix_per_component": args.prefix,
            "last": {
                "manifest_path": str(last_manifest),
                "manifest_sha256": file_sha256(last_manifest),
                "selected_task_count": len(last_rows),
            },
            "mmad": {
                "manifest_path": str(anomaly_manifest),
                "manifest_sha256": file_sha256(anomaly_manifest),
                "selected_task_count": len(anomaly_rows),
            },
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    last_parser = subparsers.add_parser(
        "last", help="Prepare the deterministic LAS&T 400-per-test derived manifest."
    )
    last_parser.add_argument("--source", type=Path, default=DEFAULT_LAST_SOURCE)
    last_parser.add_argument("--output", type=Path)
    last_parser.add_argument("--prefix", type=_positive_int)
    last_parser.add_argument(
        "--seed", type=_non_negative_int, default=LAST_DEFAULT_SEED
    )
    last_parser.add_argument(
        "--prompt-profile",
        choices=tuple(sorted(LAST_PROMPT_PROFILES)),
        default=LAST_DEFAULT_PROMPT_PROFILE,
    )
    last_parser.add_argument(
        "--quizzes-per-directory",
        type=_positive_int,
        default=LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
    )
    last_parser.add_argument(
        "--expected-identities",
        type=_positive_int,
        default=LAST_EXPECTED_IDENTITY_COUNT,
    )
    last_parser.add_argument(
        "--expected-images-per-identity",
        type=_positive_int,
        default=LAST_EXPECTED_IMAGES_PER_IDENTITY,
    )
    last_parser.add_argument(
        "--expected-width",
        type=_positive_int,
        default=LAST_EXPECTED_SOURCE_IMAGE_SIZE[0],
    )
    last_parser.add_argument(
        "--expected-height",
        type=_positive_int,
        default=LAST_EXPECTED_SOURCE_IMAGE_SIZE[1],
    )
    last_parser.set_defaults(handler=_prepare_last)

    anomaly_parser = subparsers.add_parser(
        "mmad", help="Prepare the pinned MMAD zero- or one-shot manifest."
    )
    anomaly_parser.add_argument("--source", type=Path, default=DEFAULT_MMAD_SOURCE)
    anomaly_parser.add_argument("--output", type=Path)
    anomaly_parser.add_argument("--prefix", type=_positive_int)
    anomaly_parser.add_argument("--shot", type=int, choices=(0, 1), default=1)
    anomaly_parser.add_argument(
        "--template-kind", choices=("random", "similar"), default="random"
    )
    anomaly_parser.add_argument(
        "--allow-unpinned-source",
        action="store_true",
        help="Development-only: accept a non-official annotation JSON.",
    )
    anomaly_parser.set_defaults(handler=_prepare_mmad)

    suite_parser = subparsers.add_parser(
        "suite", help="Merge and reindex prepared LAS&T and MMAD manifests."
    )
    suite_parser.add_argument(
        "--last-manifest",
        type=Path,
        default=DEFAULT_LAST_OUTPUT / "tasks.jsonl",
    )
    suite_parser.add_argument(
        "--mmad-manifest",
        type=Path,
        default=DEFAULT_MMAD_ONE_SHOT_OUTPUT / "tasks.jsonl",
    )
    suite_parser.add_argument("--output", type=Path)
    suite_parser.add_argument(
        "--prefix",
        type=_positive_int,
        help="Take this deterministic prefix from each component manifest.",
    )
    suite_parser.set_defaults(handler=_prepare_suite)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = args.handler(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
