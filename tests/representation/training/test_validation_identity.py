from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.data import (
    RepresentationDataset,
    SplitOverlapKind,
    SplitOverlapPolicy,
    load_retained_representation_jsonl,
    train_validation_group_overlap,
)
from tgvf_rl.representation.training.validation_identity import (
    REPRESENTATION_IMAGE_RAW_BYTE_MANIFEST_SCHEMA_VERSION,
    REPRESENTATION_VALIDATION_DATA_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION,
    ImageRawByteManifest,
    ImageRawByteManifestEntry,
    RepresentationValidationDataAudit,
    RepresentationValidationIdentityError,
    build_image_raw_byte_manifest,
    build_representation_validation_data_audit,
    build_retained_image_raw_byte_manifest,
)


def _dataset(
    tmp_path: Path,
    *,
    split_name: str,
    image: Path,
    image_id: str,
    count: int = 4,
) -> RepresentationDataset:
    rows = [
        {
            "uid": f"{split_name}-{index}",
            "image": str(image.resolve()),
            "question": f"question {split_name} {index}",
            "need_focus": True,
            "trajectory_type": "single_focus",
            "evidence_state": "need_local_visual_evidence",
            "target": f"{split_name} target {index}",
            "evidence_description": f"{split_name} evidence {index}",
            "short_answer": f"{split_name} answer {index}",
            "image_id": image_id,
            "stable_image_uid": f"{split_name}-stable-{index}",
            "item_content_hash": f"{split_name}-content-{index}",
        }
        for index in range(count)
    ]
    payload = (
        b"\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            for row in rows
        )
        + b"\n"
    )
    source = tmp_path / f"{split_name}.jsonl"
    source.write_bytes(payload)
    return load_retained_representation_jsonl(
        source,
        expected_source_sha256=sha256(payload).hexdigest(),
        warn_on_leakage=False,
    )


def _audit(
    train: RepresentationDataset,
    validation: RepresentationDataset,
    *,
    policy: SplitOverlapPolicy,
) -> RepresentationValidationDataAudit:
    report = train_validation_group_overlap(train.samples, validation.samples)
    return build_representation_validation_data_audit(
        train_dataset=train,
        validation_dataset=validation,
        validation_batch_k=4,
        validation_sampler_seed=73,
        validation_every_optimizer_steps=25,
        evaluator_schema_version=REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION,
        overlap_policy=policy,
        expected_overlap_report_sha256=report.identity_sha256,
    )


def test_image_raw_byte_manifest_is_path_deduplicated_sorted_and_content_bound(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    same_bytes = tmp_path / "c.png"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    same_bytes.write_bytes(b"first-image")
    first = first.resolve()
    second = second.resolve()
    same_bytes = same_bytes.resolve()

    manifest = build_image_raw_byte_manifest((second, first, first, same_bytes))
    reordered = build_image_raw_byte_manifest((same_bytes, first, second))

    assert manifest.schema_version == (
        REPRESENTATION_IMAGE_RAW_BYTE_MANIFEST_SCHEMA_VERSION
    )
    assert tuple(entry.resolved_path for entry in manifest.entries) == tuple(
        sorted((str(first), str(second), str(same_bytes)))
    )
    assert manifest.file_count == 3
    assert manifest.total_size_bytes == len(b"first-image") * 2 + len(b"second-image")
    assert manifest.entries[0].sha256 == sha256(b"first-image").hexdigest()
    assert manifest.manifest_sha256 == reordered.manifest_sha256
    assert len(manifest.manifest_sha256) == 64
    assert manifest.canonical_payload()["file_count"] == 3

    first.write_bytes(b"changed-image")
    changed = build_image_raw_byte_manifest((same_bytes, first, second))
    assert changed.manifest_sha256 != manifest.manifest_sha256


def test_image_raw_byte_manifest_rejects_noncanonical_or_untyped_inputs(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    resolved = image.resolve()
    alias = tmp_path / "alias.png"
    alias.symlink_to(resolved)

    with pytest.raises(TypeError, match="non-string sequence"):
        build_image_raw_byte_manifest(str(resolved))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absolute normalized"):
        build_image_raw_byte_manifest((Path("relative.png"),))
    with pytest.raises(RepresentationValidationIdentityError, match="fully resolved"):
        build_image_raw_byte_manifest((alias.absolute(),))
    with pytest.raises(RepresentationValidationIdentityError, match="regular file"):
        build_image_raw_byte_manifest((tmp_path.resolve(),))
    with pytest.raises(ValueError, match="non-empty"):
        build_image_raw_byte_manifest(())

    entry = ImageRawByteManifestEntry(
        resolved_path=str(resolved), size_bytes=5, sha256=sha256(b"image").hexdigest()
    )
    with pytest.raises(ValueError, match="unique sorted"):
        ImageRawByteManifest(entries=(entry, entry))
    with pytest.raises(ValueError, match="schema mismatch"):
        ImageRawByteManifest(entries=(entry,), schema_version="wrong")


def test_build_audit_binds_validation_overlap_cadence_and_every_image_byte(
    tmp_path: Path,
) -> None:
    shared_image = tmp_path / "shared.png"
    shared_image.write_bytes(b"shared-image-v1")
    train = _dataset(
        tmp_path,
        split_name="train",
        image=shared_image,
        image_id="train-group",
    )
    validation = _dataset(
        tmp_path,
        split_name="validation",
        image=shared_image,
        image_id="validation-group",
    )

    audit = _audit(
        train,
        validation,
        policy=SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
    )
    identity = audit.identity

    assert identity.schema_version == (
        REPRESENTATION_VALIDATION_DATA_IDENTITY_SCHEMA_VERSION
    )
    assert identity.train_retained_manifest_sha256 == train.manifest.manifest_sha256
    assert identity.validation_retained_manifest_sha256 == (
        validation.manifest.manifest_sha256
    )
    assert identity.validation_batch_k == 4
    assert identity.validation_sampler_seed == 73
    assert identity.validation_every_optimizer_steps == 25
    assert identity.evaluator_schema_version == (
        REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION
    )
    assert identity.overlap_policy is SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH
    assert identity.overlap_report_sha256 == audit.overlap_report.identity_sha256
    assert identity.overlap_record_count == 1
    assert identity.overlap_kinds == (SplitOverlapKind.IMAGE_PATH,)
    assert identity.train_image_file_count == 1
    assert identity.validation_image_file_count == 1
    assert identity.train_image_total_size_bytes == len(b"shared-image-v1")
    assert identity.validation_image_total_size_bytes == len(b"shared-image-v1")
    assert len(identity.identity_sha256) == 64
    assert identity.identity_sha256 == identity.identity_sha256
    assert identity.canonical_payload()["overlap_policy"] == (
        SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH.value
    )
    assert replace(identity, validation_sampler_seed=74).identity_sha256 != (
        identity.identity_sha256
    )
    with pytest.raises(FrozenInstanceError):
        identity.validation_sampler_seed = 99  # type: ignore[misc]

    retained_hash = train.manifest.manifest_sha256
    shared_image.write_bytes(b"shared-image-v2")
    changed = _audit(
        train,
        validation,
        policy=SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
    )
    assert changed.identity.train_retained_manifest_sha256 == retained_hash
    assert changed.identity.train_image_manifest_sha256 != (
        identity.train_image_manifest_sha256
    )
    assert changed.identity.identity_sha256 != identity.identity_sha256


def test_disjoint_audit_binds_the_exact_empty_report(tmp_path: Path) -> None:
    train_image = tmp_path / "train.png"
    validation_image = tmp_path / "validation.png"
    train_image.write_bytes(b"train")
    validation_image.write_bytes(b"validation")
    train = _dataset(
        tmp_path,
        split_name="train",
        image=train_image,
        image_id="train-group",
    )
    validation = _dataset(
        tmp_path,
        split_name="validation",
        image=validation_image,
        image_id="validation-group",
    )

    audit = _audit(train, validation, policy=SplitOverlapPolicy.REQUIRE_DISJOINT)

    assert audit.overlap_report.is_disjoint
    assert audit.identity.overlap_report_sha256 == audit.overlap_report.identity_sha256
    assert audit.identity.overlap_record_count == 0
    assert audit.identity.overlap_kinds == ()
    assert build_retained_image_raw_byte_manifest(train.manifest) == (
        audit.train_image_manifest
    )


def test_audit_fails_closed_on_report_policy_schema_and_binding_mismatches(
    tmp_path: Path,
) -> None:
    shared_image = tmp_path / "shared.png"
    shared_image.write_bytes(b"image")
    train = _dataset(
        tmp_path,
        split_name="train",
        image=shared_image,
        image_id="same-group",
    )
    validation = _dataset(
        tmp_path,
        split_name="validation",
        image=shared_image,
        image_id="same-group",
    )
    report = train_validation_group_overlap(train.samples, validation.samples)
    kwargs = {
        "train_dataset": train,
        "validation_dataset": validation,
        "validation_batch_k": 4,
        "validation_sampler_seed": 73,
        "validation_every_optimizer_steps": 25,
        "evaluator_schema_version": (
            REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION
        ),
        "overlap_policy": SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
        "expected_overlap_report_sha256": report.identity_sha256,
    }

    with pytest.raises(RepresentationValidationIdentityError, match="differs"):
        build_representation_validation_data_audit(
            **{**kwargs, "expected_overlap_report_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="cannot accept"):
        build_representation_validation_data_audit(**kwargs)
    with pytest.raises(ValueError, match="evaluator schema"):
        build_representation_validation_data_audit(
            **{**kwargs, "evaluator_schema_version": "unknown"}
        )
    with pytest.raises(TypeError, match="validation_batch_k"):
        build_representation_validation_data_audit(
            **{**kwargs, "validation_batch_k": True}
        )

    validation_distinct_group = _dataset(
        tmp_path,
        split_name="validation-distinct",
        image=shared_image,
        image_id="validation-group",
    )
    valid_audit = _audit(
        train,
        validation_distinct_group,
        policy=SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
    )
    tampered_identity = replace(
        valid_audit.identity, validation_image_manifest_sha256="f" * 64
    )
    with pytest.raises(RepresentationValidationIdentityError, match="validation image"):
        RepresentationValidationDataAudit(
            identity=tampered_identity,
            overlap_report=valid_audit.overlap_report,
            train_image_manifest=valid_audit.train_image_manifest,
            validation_image_manifest=valid_audit.validation_image_manifest,
        )
    with pytest.raises(ValueError, match="disjoint policy"):
        replace(
            valid_audit.identity,
            overlap_policy=SplitOverlapPolicy.REQUIRE_DISJOINT,
        )
    with pytest.raises(ValueError, match="permits only"):
        replace(
            valid_audit.identity,
            overlap_kinds=(SplitOverlapKind.IMAGE_GROUP_KEY,),
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        replace(valid_audit.identity, schema_version="wrong")


def test_allow_recorded_policy_rejects_an_empty_report(tmp_path: Path) -> None:
    train_image = tmp_path / "train.png"
    validation_image = tmp_path / "validation.png"
    train_image.write_bytes(b"train")
    validation_image.write_bytes(b"validation")
    train = _dataset(
        tmp_path,
        split_name="train",
        image=train_image,
        image_id="train-group",
    )
    validation = _dataset(
        tmp_path,
        split_name="validation",
        image=validation_image,
        image_id="validation-group",
    )
    report = train_validation_group_overlap(train.samples, validation.samples)

    with pytest.raises(RepresentationValidationIdentityError, match="non-empty"):
        build_representation_validation_data_audit(
            train_dataset=train,
            validation_dataset=validation,
            validation_batch_k=4,
            validation_sampler_seed=73,
            validation_every_optimizer_steps=25,
            evaluator_schema_version=(
                REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION
            ),
            overlap_policy=SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH,
            expected_overlap_report_sha256=report.identity_sha256,
        )
