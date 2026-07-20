from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import tgvf_rl.representation.training as representation_training
from tgvf_rl.representation.training.schema import (
    REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION,
    RepresentationChoice,
    RepresentationSampleIdentity,
    RepresentationTrainingSample,
)


def test_representation_training_public_surface_exports_core_contracts() -> None:
    assert (
        representation_training.RepresentationTrainingSample
        is RepresentationTrainingSample
    )
    assert (
        representation_training.SameImageBatchSampler.__name__
        == "SameImageBatchSampler"
    )
    assert representation_training.same_image_matrix_ce_loss.__name__ == (
        "same_image_matrix_ce_loss"
    )


def _sample(**overrides: object) -> RepresentationTrainingSample:
    values: dict[str, object] = {
        "sample_id": "row-001",
        "image": "/images/shared.png",
        "question": "What is written on the sign?",
        "target": "the small sign beside the door",
        "evidence_description": "The sign reads OPEN.",
        "short_answer": "OPEN",
    }
    values.update(overrides)
    return RepresentationTrainingSample(**values)  # type: ignore[arg-type]


def test_sample_identity_is_immutable_and_uses_image_id_group_key() -> None:
    sample = _sample(image_id="image-42")

    assert sample.image_group_key == "image-42"
    assert sample.identity == RepresentationSampleIdentity(
        "row-001", "image-42", sample.content_sha256
    )
    assert (
        sample.identity.schema_version == REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION
    )
    with pytest.raises(FrozenInstanceError):
        sample.target = "another target"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sample.identity.sample_id = "another-row"  # type: ignore[misc]


def test_group_key_falls_back_to_image_reference() -> None:
    sample = _sample()

    assert sample.image_group_key == "/images/shared.png"
    assert sample.identity.image_group_key == sample.image


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_id", ""),
        ("image", "  "),
        ("question", ""),
        ("target", "\t"),
        ("evidence_description", ""),
        ("short_answer", " "),
        ("image_id", " "),
        ("evidence_type", ""),
        ("source_profile", "\t"),
    ],
)
def test_required_identity_and_training_text_fail_closed(
    field_name: str, value: str
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _sample(**{field_name: value})


def test_identity_rejects_invalid_direct_construction() -> None:
    with pytest.raises(ValueError, match="sample_id"):
        RepresentationSampleIdentity("", "image", "a" * 64)
    with pytest.raises(ValueError, match="image_group_key"):
        RepresentationSampleIdentity("row", "", "a" * 64)
    with pytest.raises(ValueError, match="content_sha256"):
        RepresentationSampleIdentity("row", "image", "not-a-sha")


def test_content_identity_changes_with_every_supervision_field() -> None:
    baseline = _sample()
    for field_name, value in (
        ("sample_id", "row-002"),
        ("image", "/images/other.png"),
        ("image_id", "image-2"),
        ("question", "Another question?"),
        ("target", "another target"),
        ("evidence_description", "Different evidence."),
        ("short_answer", "CLOSED"),
        ("stable_image_uid", "stable-image-2"),
        ("item_content_hash", "content-hash-2"),
        ("source_dataset", "another-dataset"),
        ("source_profile", "another-profile"),
        ("evidence_type", "another-evidence-type"),
        ("answer_type", "another-answer-type"),
        ("visual_difficulty", "hard"),
        ("target_leakage_risk", "high"),
        ("choices", (RepresentationChoice(label="A", text="OPEN"),)),
    ):
        assert _sample(**{field_name: value}).content_sha256 != baseline.content_sha256


def test_choices_are_immutable_ordered_and_uniquely_labeled() -> None:
    choices = (
        RepresentationChoice(label="A", text="OPEN"),
        RepresentationChoice(label="B", text="CLOSED"),
    )
    sample = _sample(choices=choices)

    assert sample.choices == choices
    assert _sample(choices=tuple(reversed(choices))).content_sha256 != (
        sample.content_sha256
    )
    with pytest.raises(TypeError, match="tuple"):
        _sample(choices=list(choices))
    with pytest.raises(ValueError, match="unique"):
        _sample(choices=(choices[0], choices[0]))
