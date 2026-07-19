"""Immutable inputs shared by representation-training data components."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION = "representation_sample_identity_v1"


@dataclass(frozen=True, slots=True)
class RepresentationSampleIdentity:
    """Stable row and same-image group identity for one training sample."""

    sample_id: str
    image_group_key: str
    content_sha256: str
    schema_version: str = REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.sample_id, field_name="sample_id")
        _require_non_empty_text(self.image_group_key, field_name="image_group_key")
        _require_sha256(self.content_sha256, field_name="content_sha256")
        if self.schema_version != REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("representation sample identity schema mismatch")


@dataclass(frozen=True, slots=True)
class RepresentationTrainingSample:
    """Protocol-neutral fields required by representation training.

    The native transcript and target-conditioning tensors are deliberately not
    stored here: they are derived by the selected Qwen-family/provider pipeline.
    """

    sample_id: str
    image: str
    question: str
    target: str
    evidence_description: str
    image_id: str | None = None
    stable_image_uid: str | None = None
    item_content_hash: str | None = None
    source_dataset: str | None = None
    source_profile: str | None = None
    evidence_type: str | None = None
    answer_type: str | None = None
    visual_difficulty: str | None = None
    target_leakage_risk: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "image",
            "question",
            "target",
            "evidence_description",
        ):
            _require_non_empty_text(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "image_id",
            "stable_image_uid",
            "item_content_hash",
            "source_dataset",
            "source_profile",
            "evidence_type",
            "answer_type",
            "visual_difficulty",
            "target_leakage_risk",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty_text(value, field_name=field_name)

    @property
    def image_group_key(self) -> str:
        """Legacy-compatible image key: explicit ID, then image reference."""

        return self.image_id or self.image

    @property
    def identity(self) -> RepresentationSampleIdentity:
        return RepresentationSampleIdentity(
            sample_id=self.sample_id,
            image_group_key=self.image_group_key,
            content_sha256=self.content_sha256,
        )

    @property
    def content_sha256(self) -> str:
        """Digest every field that can change grouping or supervision."""

        payload = {
            "schema_version": REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "image": self.image,
            "image_id": self.image_id,
            "question": self.question,
            "target": self.target,
            "evidence_description": self.evidence_description,
            "stable_image_uid": self.stable_image_uid,
            "item_content_hash": self.item_content_hash,
            "source_dataset": self.source_dataset,
            "source_profile": self.source_profile,
            "evidence_type": self.evidence_type,
            "answer_type": self.answer_type,
            "visual_difficulty": self.visual_difficulty,
            "target_leakage_risk": self.target_leakage_risk,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")
