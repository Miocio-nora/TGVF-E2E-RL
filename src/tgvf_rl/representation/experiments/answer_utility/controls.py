"""Atomic correct/zero/wrong focused-D controls for answer utility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import unicodedata

import torch

from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


class AnswerUtilityArm(str, Enum):
    CORRECT = "correct"
    ZERO = "zero"
    WRONG_SAME_IMAGE_TARGET = "wrong_same_image_target"


@dataclass(frozen=True, slots=True)
class AnswerUtilityControlRow:
    """One row-fixed answer target and three whole-observation interventions."""

    sample_id: str
    correct_source_sample_id: str
    wrong_source_sample_id: str | None
    correct: RepresentationVisualTensorBundle
    zero: RepresentationVisualTensorBundle | None
    wrong: RepresentationVisualTensorBundle | None

    def __post_init__(self) -> None:
        for name in ("sample_id", "correct_source_sample_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.sample_id != self.correct_source_sample_id:
            raise ValueError("correct D must come from the current sample")
        if (self.wrong_source_sample_id is None) != (self.wrong is None):
            raise ValueError("wrong D and its source identity must coexist")
        if self.wrong_source_sample_id is not None:
            if (
                not isinstance(self.wrong_source_sample_id, str)
                or not self.wrong_source_sample_id.strip()
            ):
                raise ValueError("wrong_source_sample_id must be non-empty")
            if self.wrong_source_sample_id == self.sample_id:
                raise ValueError("wrong D must come from another sample")
            assert self.wrong is not None
            _assert_same_visual_contract(self.wrong, self.correct, name="wrong D")
        if self.zero is not None:
            _assert_same_visual_contract(self.zero, self.correct, name="zero D")
            if any(
                tensor.requires_grad
                for tensor in (self.zero.main, *self.zero.deepstack)
            ):
                raise ValueError("zero D must not retain an Adapter graph")
            if any(
                not bool(torch.count_nonzero(tensor).eq(0).item())
                for tensor in (self.zero.main, *self.zero.deepstack)
            ):
                raise ValueError("zero D must atomically zero main and all DeepStack")

    def observation(self, arm: AnswerUtilityArm) -> RepresentationVisualTensorBundle:
        if arm is AnswerUtilityArm.CORRECT:
            return self.correct
        if arm is AnswerUtilityArm.ZERO:
            if self.zero is None:
                raise ValueError("zero-D arm is inactive for this experiment")
            return self.zero
        if arm is AnswerUtilityArm.WRONG_SAME_IMAGE_TARGET:
            if self.wrong is None:
                raise ValueError("wrong-D arm is inactive for this experiment")
            return self.wrong
        raise ValueError("unknown answer utility arm")


def build_same_image_answer_controls(
    samples: tuple[RepresentationTrainingSample, ...],
    group: SameImageReadoutGroup,
    *,
    requires_zero_control: bool,
    requires_wrong_control: bool,
) -> tuple[AnswerUtilityControlRow, ...]:
    """Build only selected arms and reject answer-preserving wrong-D pairs."""

    if not isinstance(samples, tuple) or len(samples) < 2:
        raise ValueError("same-image answer controls require K>=2 typed samples")
    if any(not isinstance(sample, RepresentationTrainingSample) for sample in samples):
        raise TypeError("answer controls samples must be representation samples")
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("answer controls require a same-image readout group")
    if (
        type(requires_zero_control) is not bool
        or type(requires_wrong_control) is not bool
    ):
        raise TypeError("answer-control requirements must be explicit booleans")
    sample_ids = tuple(sample.sample_id for sample in samples)
    candidate_ids = tuple(candidate.sample_id for candidate in group.candidates)
    if sample_ids != candidate_ids:
        raise ValueError("answer controls sample/candidate order differs")
    if requires_wrong_control and len({sample.target for sample in samples}) != len(
        samples
    ):
        raise ValueError("wrong-target control requires distinct targets")
    rows: list[AnswerUtilityControlRow] = []
    for index, candidate in enumerate(group.candidates):
        wrong_index = (
            _answer_safe_wrong_index(samples, index) if requires_wrong_control else None
        )
        wrong = None if wrong_index is None else group.candidates[wrong_index]
        if wrong is not None:
            _validate_candidate_pair(candidate, wrong)
        rows.append(
            AnswerUtilityControlRow(
                sample_id=candidate.sample_id,
                correct_source_sample_id=candidate.sample_id,
                wrong_source_sample_id=None if wrong is None else wrong.sample_id,
                correct=candidate.visual,
                zero=(
                    zero_visual_bundle(candidate.visual)
                    if requires_zero_control
                    else None
                ),
                wrong=None if wrong is None else wrong.visual,
            )
        )
    return tuple(rows)


def _answer_safe_wrong_index(
    samples: tuple[RepresentationTrainingSample, ...],
    row_index: int,
) -> int:
    answer = _normalized_answer_identity(samples[row_index].short_answer)
    for offset in range(1, len(samples)):
        candidate_index = (row_index + offset) % len(samples)
        if _normalized_answer_identity(samples[candidate_index].short_answer) != answer:
            return candidate_index
    raise ValueError(
        "same-image group has no answer-safe wrong target for "
        f"{samples[row_index].sample_id}"
    )


def _normalized_answer_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("short answer must be non-empty")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def zero_visual_bundle(
    reference: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    """Create a detached, shape-identical zero for every atomic D path."""

    if not isinstance(reference, RepresentationVisualTensorBundle):
        raise TypeError("zero D reference must be a visual bundle")
    return RepresentationVisualTensorBundle(
        main=torch.zeros_like(reference.main, requires_grad=False),
        deepstack=tuple(
            torch.zeros_like(branch, requires_grad=False)
            for branch in reference.deepstack
        ),
        branch_layers=reference.branch_layers,
        d_deepstack_active=reference.d_deepstack_active,
    )


def _validate_candidate_pair(
    correct: RepresentationCandidateObservation,
    wrong: RepresentationCandidateObservation,
) -> None:
    if correct.sample_id == wrong.sample_id:
        raise ValueError("wrong candidate must have another sample identity")
    if correct.image_group_key != wrong.image_group_key or (
        correct.source_visual_identity != wrong.source_visual_identity
    ):
        raise ValueError("same-image wrong D must retain exact source identity")
    if correct.target_conditioning_provider is not wrong.target_conditioning_provider:
        raise ValueError("correct/wrong D mix target-conditioning providers")
    if correct.projection_identities != wrong.projection_identities:
        raise ValueError("correct/wrong D mix projection identities")
    _assert_same_visual_contract(wrong.visual, correct.visual, name="wrong D")


def _assert_same_visual_contract(
    actual: RepresentationVisualTensorBundle,
    expected: RepresentationVisualTensorBundle,
    *,
    name: str,
) -> None:
    if not isinstance(actual, RepresentationVisualTensorBundle) or not isinstance(
        expected, RepresentationVisualTensorBundle
    ):
        raise TypeError(f"{name} must be a visual bundle")
    if (
        actual.branch_layers != expected.branch_layers
        or actual.d_deepstack_active != expected.d_deepstack_active
    ):
        raise ValueError(f"{name} changed DeepStack identity/activity")
    for actual_tensor, expected_tensor in zip(
        (actual.main, *actual.deepstack),
        (expected.main, *expected.deepstack),
        strict=True,
    ):
        if (
            actual_tensor.shape != expected_tensor.shape
            or actual_tensor.dtype != expected_tensor.dtype
            or actual_tensor.device != expected_tensor.device
        ):
            raise ValueError(f"{name} changed tensor shape/dtype/device")


__all__ = [
    "AnswerUtilityArm",
    "AnswerUtilityControlRow",
    "build_same_image_answer_controls",
    "zero_visual_bundle",
]
