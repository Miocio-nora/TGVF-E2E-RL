from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tgvf_rl.evaluation.crop_grounding_probe import (
    CROP_GROUNDING_PROBE_SCHEMA,
    DEFAULT_STRATA,
    file_sha256,
    materialize_crop_grounding_probe,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(
    tmp_path: Path,
    *,
    sample_id: str,
    source_file: str,
    row: int,
    image_number: int,
) -> dict[str, object]:
    image_path = tmp_path / "images" / f"{sample_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (8, 6),
        (image_number % 256, (image_number * 3) % 256, (image_number * 7) % 256),
    ).save(image_path)
    image_sha256 = file_sha256(image_path)
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": sample_id,
        "source": "vstar",
        "question": f"Where is the object for {sample_id}?",
        "ground_truth": "left",
        "image": {
            "path": str(image_path.resolve()),
            "sha256": image_sha256,
            "width": 8,
            "height": 6,
        },
        "gt_regions": [[1, 1, 4, 4]],
        "provenance": {
            "source_file": source_file,
            "source_row_index": row,
        },
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for stratum_index, stratum in enumerate(DEFAULT_STRATA):
        candidates.extend(
            [
                _candidate(
                    tmp_path,
                    sample_id=f"excluded-{stratum_index}",
                    source_file=stratum,
                    row=stratum_index * 10,
                    image_number=1,
                ),
                _candidate(
                    tmp_path,
                    sample_id=f"heldout-a-{stratum_index}",
                    source_file=stratum,
                    row=stratum_index * 10 + 1,
                    image_number=stratum_index + 2,
                ),
                _candidate(
                    tmp_path,
                    sample_id=f"heldout-b-{stratum_index}",
                    source_file=stratum,
                    row=stratum_index * 10 + 2,
                    image_number=stratum_index + 20,
                ),
            ]
        )
    candidate_dir = tmp_path / "candidates"
    candidate_path = candidate_dir / "candidates.jsonl"
    _jsonl(candidate_path, candidates)
    candidate_manifest = candidate_dir / "manifest.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "schema_version": "tgvf.policy-selection.source-manifest.v1",
                "source": "vstar",
                "candidate_rows": len(candidates),
                "candidates": {
                    "path": candidate_path.name,
                    "sha256": file_sha256(candidate_path),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    training_path = tmp_path / "training.jsonl"
    _jsonl(
        training_path,
        [
            {
                "schema_version": "tgvf.policy-t1-mixed-rl.sample.v2",
                "sample_id": "training-sample",
                "data_source": "vstar",
                "image": {"sha256": candidates[0]["image"]["sha256"]},
            }
        ],
    )
    return {
        "candidate_manifest_path": candidate_manifest,
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "training_samples_path": training_path,
        "training_samples_sha256": file_sha256(training_path),
        "output_root": tmp_path / "probe",
        "seed": 20260807,
        "per_stratum": 1,
    }


def test_materializes_balanced_image_disjoint_probe_idempotently(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)

    first = materialize_crop_grounding_probe(**inputs)
    second = materialize_crop_grounding_probe(**inputs)

    assert second == first
    assert first["sample_count"] == 4
    assert first["stratum_counts"] == {stratum: 1 for stratum in DEFAULT_STRATA}
    probe_path = Path(first["probe_manifest_path"])
    tasks_path = Path(first["task_manifest_path"])
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert probe["schema_version"] == CROP_GROUNDING_PROBE_SCHEMA
    assert probe["task_manifest"]["sha256"] == file_sha256(tasks_path)
    assert [task["ordinal"] for task in tasks] == list(range(4))
    assert [task["sample_id"] for task in tasks] == probe["ordered_sample_ids"]
    assert all(task["sample_id"] == task["index"] for task in tasks)
    assert not any(task["sample_id"].startswith("excluded-") for task in tasks)
    assert len({task["image_paths"][0] for task in tasks}) == 4
    assert all(len(task["image_sha256s"][0]) == 64 for task in tasks)
    assert all(task["image_dimensions"] == [[8, 6]] for task in tasks)


def test_bound_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["training_samples_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="training samples SHA256 mismatch"):
        materialize_crop_grounding_probe(**inputs)


def test_insufficient_image_disjoint_stratum_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["per_stratum"] = 3

    with pytest.raises(ValueError, match="unique held-out images"):
        materialize_crop_grounding_probe(**inputs)


def test_declared_candidate_image_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    manifest_path = inputs["candidate_manifest_path"]
    assert isinstance(manifest_path, Path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_path = manifest_path.parent / manifest["candidates"]["path"]
    rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    rows[1]["image"]["sha256"] = "f" * 64
    _jsonl(candidate_path, rows)
    manifest["candidates"]["sha256"] = file_sha256(candidate_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["candidate_manifest_sha256"] = file_sha256(manifest_path)

    with pytest.raises(ValueError, match="image SHA256 differs"):
        materialize_crop_grounding_probe(**inputs)
