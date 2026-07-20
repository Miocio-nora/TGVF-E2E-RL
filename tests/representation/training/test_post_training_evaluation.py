from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.post_training_evaluation import (
    REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_LEGACY_SCHEMA_VERSION,
    REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION,
    load_internal_evaluation_group_manifest,
    materialize_internal_evaluation_groups,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _sample(sample_id: str, image_id: str, target: str) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=sample_id,
        image=f"/images/{image_id}.png",
        image_id=image_id,
        question="What value is shown?",
        target=target,
        evidence_description=f"The value is {target}.",
        short_answer=target,
    )


def _write_manifest(
    path: Path,
    *,
    data_manifest_sha256: str,
    groups: tuple[tuple[RepresentationTrainingSample, ...], ...],
    schema_version: str = REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "identity": "ordered-same-image-groups-v1",
                "source_data_manifest_sha256": data_manifest_sha256,
                "groups": [
                    {
                        "image_group_key": group[0].image_group_key,
                        "samples": [
                            {
                                "sample_id": sample.sample_id,
                                "content_sha256": sample.content_sha256,
                            }
                            for sample in group
                        ],
                    }
                    for group in groups
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_group_manifest_preserves_exact_group_and_sample_order(tmp_path: Path) -> None:
    first = (_sample("a0", "image-a", "1"), _sample("a1", "image-a", "2"))
    second = (_sample("b0", "image-b", "3"), _sample("b1", "image-b", "4"))
    data_sha256 = "1" * 64
    manifest = load_internal_evaluation_group_manifest(
        _write_manifest(
            tmp_path / "groups.json",
            data_manifest_sha256=data_sha256,
            groups=(first, second),
        )
    )

    materialized = materialize_internal_evaluation_groups(
        manifest,
        data_manifest_sha256=data_sha256,
        samples=tuple(reversed(first + second)),
    )

    assert tuple(sample.sample_id for group in materialized for sample in group) == (
        "a0",
        "a1",
        "b0",
        "b1",
    )


def test_group_manifest_rejects_dataset_content_and_group_drift(tmp_path: Path) -> None:
    first = (_sample("a0", "image-a", "1"), _sample("a1", "image-a", "2"))
    second = (_sample("b0", "image-b", "3"), _sample("b1", "image-b", "4"))
    data_sha256 = "2" * 64
    manifest = load_internal_evaluation_group_manifest(
        _write_manifest(
            tmp_path / "groups.json",
            data_manifest_sha256=data_sha256,
            groups=(first, second),
        )
    )

    with pytest.raises(ValueError, match="another dataset"):
        materialize_internal_evaluation_groups(
            manifest,
            data_manifest_sha256="3" * 64,
            samples=first + second,
        )

    changed = _sample("a0", "image-a", "changed")
    with pytest.raises(ValueError, match="content SHA256 drifted"):
        materialize_internal_evaluation_groups(
            manifest,
            data_manifest_sha256=data_sha256,
            samples=(changed, first[1], *second),
        )


def test_legacy_group_manifest_requires_equal_sized_groups(tmp_path: Path) -> None:
    first = (_sample("a0", "image-a", "1"), _sample("a1", "image-a", "2"))
    second = (
        _sample("b0", "image-b", "3"),
        _sample("b1", "image-b", "4"),
        _sample("b2", "image-b", "5"),
    )
    with pytest.raises(ValueError, match="equal K"):
        load_internal_evaluation_group_manifest(
            _write_manifest(
                tmp_path / "groups.json",
                data_manifest_sha256="4" * 64,
                groups=(first, second),
                schema_version=(
                    REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_LEGACY_SCHEMA_VERSION
                ),
            )
        )


def test_v2_group_manifest_accepts_golden_variable_k(tmp_path: Path) -> None:
    first = (_sample("a0", "image-a", "1"), _sample("a1", "image-a", "2"))
    second = (
        _sample("b0", "image-b", "3"),
        _sample("b1", "image-b", "4"),
        _sample("b2", "image-b", "5"),
    )

    manifest = load_internal_evaluation_group_manifest(
        _write_manifest(
            tmp_path / "groups-v2.json",
            data_manifest_sha256="5" * 64,
            groups=(first, second),
        )
    )

    assert tuple(len(group.samples) for group in manifest.groups) == (2, 3)
