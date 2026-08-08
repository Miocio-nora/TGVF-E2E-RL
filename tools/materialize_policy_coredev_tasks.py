#!/usr/bin/env python3
"""Materialize the image-bound CoreDev-2511 policy task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_benchmark_tasks,
    write_official_coredev_tasks,
)
from tgvf_rl.evaluation.vlmevalkit import (  # noqa: E402
    COREDEV_2511_MANIFEST_SHA256,
    COREDEV_2511_SAMPLE_COUNT,
    COREDEV_2511_SOURCE_FILE_SHA256,
    VLMEVALKIT_REVIEW_COMMIT,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/"
    "CoreDev2511-official-visible-v1"
)
EXPECTED_SINGLE_IMAGE_COUNT = 2240


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable CoreDev artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    tasks_path = output_root / "tasks.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_tasks_path = output_root / f".tasks.{uuid4().hex}.tmp"
    try:
        counts = write_official_coredev_tasks(candidate_tasks_path)
        _write_once(tasks_path, candidate_tasks_path.read_bytes())
    finally:
        candidate_tasks_path.unlink(missing_ok=True)
    if counts != {
        "total": COREDEV_2511_SAMPLE_COUNT,
        "single_image": EXPECTED_SINGLE_IMAGE_COUNT,
        "multi_image": COREDEV_2511_SAMPLE_COUNT - EXPECTED_SINGLE_IMAGE_COUNT,
    }:
        raise ValueError("CoreDev task counts differ from the frozen suite")
    tasks_sha256 = _sha256_file(tasks_path)
    load_benchmark_tasks(
        tasks_path,
        expected_task_count=COREDEV_2511_SAMPLE_COUNT,
        expected_single_image_count=EXPECTED_SINGLE_IMAGE_COUNT,
        expected_sha256=tasks_sha256,
        verify_image_contents=True,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    source_config = REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
    content = {
        "schema_version": "tgvf-policy-coredev-task-artifact-v1",
        "name": "CoreDev2511",
        "vlmevalkit_commit": VLMEVALKIT_REVIEW_COMMIT,
        "membership_manifest_sha256": COREDEV_2511_MANIFEST_SHA256,
        "membership_source_file_sha256": COREDEV_2511_SOURCE_FILE_SHA256,
        "source_config_path": str(source_config.resolve()),
        "source_config_sha256": _sha256_file(source_config),
        "task_manifest_path": str(tasks_path),
        "task_manifest_sha256": tasks_sha256,
        "task_count": counts["total"],
        "single_image_count": counts["single_image"],
        "multi_image_count": counts["multi_image"],
        "sample_id_contract": "sample_id_equals_official_index",
        "image_identity_contract": "sha256_and_width_height_per_image",
    }
    manifest = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_once(
        output_root / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
