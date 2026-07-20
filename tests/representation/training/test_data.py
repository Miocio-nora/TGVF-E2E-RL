from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.data import (
    REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
    REPRESENTATION_DATA_TRANSFORM_VERSION,
    SPLIT_OVERLAP_REPORT_SCHEMA_VERSION,
    DuplicateKind,
    RepresentationDataError,
    RepresentationDataLeakageWarning,
    RowExclusionReason,
    SplitOverlapKind,
    SplitOverlapPolicy,
    load_retained_representation_jsonl,
    train_validation_group_overlap,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _focus_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "uid": "row-1",
        "image": "image.png",
        "question": "What is written on the sign?",
        "need_focus": True,
        "trajectory_type": "single_focus",
        "evidence_state": "need_local_visual_evidence",
        "target": "the red-sign beside the door",
        "evidence_description": "The red sign reads OPEN.",
        "short_answer": "OPEN",
    }
    row.update(overrides)
    return row


def _write_source(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for row in rows
    )
    path.write_bytes(payload + b"\n")
    return sha256(payload + b"\n").hexdigest()


def _load(path: Path, source_sha256: str, **kwargs: object):
    return load_retained_representation_jsonl(
        path, expected_source_sha256=source_sha256, **kwargs
    )


def test_loads_focus_rows_resolves_images_and_records_every_disposition(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-decoded-by-the-loader")
    rows = [
        _focus_row(
            image_id="image-1",
            stable_image_uid="stable-1",
            item_content_hash="item-1",
            source_dataset="source",
            short_answer="red/sign",
        ),
        _focus_row(uid="direct", need_focus=False),
        _focus_row(uid="multi", trajectory_type="multi_focus"),
        _focus_row(uid="no-local", evidence_state="answerable_directly"),
    ]
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, rows)

    with pytest.warns(RepresentationDataLeakageWarning, match="red, sign"):
        dataset = _load(source, source_sha256)

    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.sample_id == "row-1"
    assert sample.image == str(image.resolve())
    assert sample.image_group_key == "image-1"
    assert sample.short_answer == "red/sign"
    manifest = dataset.manifest
    assert manifest.schema_version == REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION
    assert manifest.transform_version == REPRESENTATION_DATA_TRANSFORM_VERSION
    assert manifest.source_sha256 == source_sha256
    assert len(manifest.accepted_rows) == 1
    assert len(manifest.excluded_rows) == 3
    assert all(
        row.reasons == (RowExclusionReason.NOT_FOCUS_ROW,)
        for row in manifest.excluded_rows
    )
    accepted = manifest.accepted_rows[0]
    assert accepted.source_line == 1
    assert accepted.source_image_reference == "image.png"
    assert accepted.resolved_image_path == str(image.resolve())
    assert accepted.sample == sample.identity
    assert manifest.leakage_records[0].overlapping_terms == ("red", "sign")
    assert len(manifest.manifest_sha256) == 64
    assert manifest.manifest_sha256 == manifest.manifest_sha256
    with pytest.raises(FrozenInstanceError):
        manifest.source_sha256 = "0" * 64  # type: ignore[misc]


def test_source_hash_is_checked_before_json_parsing(tmp_path: Path) -> None:
    source = tmp_path / "retained.jsonl"
    source.write_bytes(b"not-json\n")

    with pytest.raises(RepresentationDataError, match="SHA256 mismatch"):
        _load(source, "0" * 64)


def test_focus_metadata_is_strict_and_never_uses_truthy_or_missing_defaults(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    missing_need_focus = _focus_row(uid="missing")
    del missing_need_focus["need_focus"]
    rows = [
        _focus_row(),
        _focus_row(uid="integer", need_focus=1),
        missing_need_focus,
        _focus_row(uid="missing-state", evidence_state=None),
    ]
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, rows)

    dataset = _load(source, source_sha256)

    assert tuple(sample.sample_id for sample in dataset.samples) == ("row-1",)
    assert all(
        entry.reasons == (RowExclusionReason.INVALID_FOCUS_METADATA,)
        for entry in dataset.manifest.excluded_rows
    )


def test_invalid_fields_and_missing_images_are_excluded_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    rows = [
        _focus_row(),
        _focus_row(uid="no-target", target=" "),
        _focus_row(uid="bad-optional", image_id=4),
        _focus_row(uid="missing-image", image="missing.png"),
    ]
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, rows)

    dataset = _load(source, source_sha256)

    assert [entry.reasons for entry in dataset.manifest.excluded_rows] == [
        (RowExclusionReason.INVALID_REQUIRED_FIELD,),
        (RowExclusionReason.INVALID_OPTIONAL_FIELD,),
        (RowExclusionReason.IMAGE_NOT_FOUND,),
    ]


def test_contract_requires_and_preserves_short_answer(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    missing_answer = _focus_row(uid="missing-answer")
    del missing_answer["short_answer"]
    rows = [
        _focus_row(short_answer="OPEN"),
        missing_answer,
        _focus_row(uid="blank-answer", short_answer=" "),
    ]
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, rows)

    dataset = _load(source, source_sha256)

    assert len(dataset.samples) == 1
    assert dataset.samples[0].short_answer == "OPEN"
    assert dataset.manifest.transform_version == REPRESENTATION_DATA_TRANSFORM_VERSION
    assert [entry.reasons for entry in dataset.manifest.excluded_rows] == [
        (RowExclusionReason.INVALID_REQUIRED_FIELD,),
        (RowExclusionReason.INVALID_REQUIRED_FIELD,),
    ]


def test_choices_requires_a_new_transform_version(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, [_focus_row(choices=[])])

    with pytest.raises(RepresentationDataError, match="choices"):
        _load(source, source_sha256)


def test_duplicate_rows_ids_and_group_targets_are_recorded_and_excluded(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    direct = _focus_row(uid="direct", need_focus=False)
    rows = [
        _focus_row(uid="first", target="target A"),
        _focus_row(uid="first", target="target B"),
        _focus_row(uid="third", target="target A"),
        _focus_row(uid="fourth", target="target B"),
        direct,
        direct,
    ]
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(source, rows)

    dataset = _load(source, source_sha256)

    # The rejected duplicate-ID row must not reserve its otherwise-new target.
    assert tuple(sample.sample_id for sample in dataset.samples) == ("first", "fourth")
    duplicate_kinds = tuple(
        record.kind for record in dataset.manifest.duplicate_records
    )
    assert duplicate_kinds == (
        DuplicateKind.SAMPLE_ID,
        DuplicateKind.GROUP_TARGET,
        DuplicateKind.SOURCE_ROW,
    )
    assert dataset.manifest.excluded_rows[0].reasons == (
        RowExclusionReason.DUPLICATE_SAMPLE_ID,
    )
    assert dataset.manifest.excluded_rows[1].reasons == (
        RowExclusionReason.DUPLICATE_GROUP_TARGET,
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"uid":"one","uid":"two"}\n',
        b"[]\n",
        b"{}\n\n",
    ],
)
def test_noncanonical_jsonl_fails_closed(tmp_path: Path, payload: bytes) -> None:
    source = tmp_path / "retained.jsonl"
    source.write_bytes(payload)
    source_sha256 = sha256(payload).hexdigest()

    with pytest.raises(RepresentationDataError):
        _load(source, source_sha256)


def _sample(sample_id: str, **overrides: object) -> RepresentationTrainingSample:
    values: dict[str, object] = {
        "sample_id": sample_id,
        "image": f"/images/{sample_id}.png",
        "image_id": f"group-{sample_id}",
        "question": "Question?",
        "target": f"target {sample_id}",
        "evidence_description": "Evidence.",
        "short_answer": "Answer.",
        "stable_image_uid": f"stable-{sample_id}",
        "item_content_hash": f"content-{sample_id}",
    }
    values.update(overrides)
    return RepresentationTrainingSample(**values)  # type: ignore[arg-type]


def test_train_validation_overlap_reports_all_recorded_manifest_keys() -> None:
    train = (_sample("train"),)
    validation = (
        _sample("group", image_id="group-train"),
        _sample("path", image="/images/train.png"),
        _sample("stable", stable_image_uid="stable-train"),
        _sample("content", item_content_hash="content-train"),
    )

    report = train_validation_group_overlap(train, validation)

    assert not report.is_disjoint
    assert tuple(record.kind for record in report.records) == (
        SplitOverlapKind.IMAGE_GROUP_KEY,
        SplitOverlapKind.IMAGE_PATH,
        SplitOverlapKind.STABLE_IMAGE_UID,
        SplitOverlapKind.ITEM_CONTENT_HASH,
    )
    assert all(len(record.value_sha256) == 64 for record in report.records)
    assert report.canonical_payload()["schema_version"] == (
        SPLIT_OVERLAP_REPORT_SCHEMA_VERSION
    )
    assert len(report.identity_sha256) == 64
    with pytest.raises(RepresentationDataError, match="overlap"):
        report.require_disjoint()
    with pytest.raises(RepresentationDataError, match="cannot accept"):
        report.validate_policy(
            SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
            expected_report_sha256=report.identity_sha256,
        )

    disjoint = train_validation_group_overlap(train, (_sample("validation"),))
    assert disjoint.is_disjoint
    disjoint.require_disjoint()
    disjoint.validate_policy(
        SplitOverlapPolicy.REQUIRE_DISJOINT,
        expected_report_sha256=None,
    )


def test_recorded_image_path_overlap_policy_is_content_bound() -> None:
    train = (_sample("train"),)
    validation = (_sample("path", image="/images/train.png"),)
    report = train_validation_group_overlap(train, validation)

    assert tuple(record.kind for record in report.records) == (
        SplitOverlapKind.IMAGE_PATH,
    )
    report.validate_policy(
        SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
        expected_report_sha256=report.identity_sha256,
    )
    with pytest.raises(RepresentationDataError, match="differs"):
        report.validate_policy(
            SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
            expected_report_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="requires"):
        report.validate_policy(
            SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
            expected_report_sha256=None,
        )


def test_target_leakage_risk_is_metadata_not_the_legacy_overlap_signal(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    source = tmp_path / "retained.jsonl"
    source_sha256 = _write_source(
        source,
        [
            _focus_row(
                target="red sign",
                short_answer="it",
                target_leakage_risk="high",
            )
        ],
    )

    dataset = _load(source, source_sha256)

    assert dataset.samples[0].target_leakage_risk == "high"
    assert dataset.manifest.leakage_records == ()
