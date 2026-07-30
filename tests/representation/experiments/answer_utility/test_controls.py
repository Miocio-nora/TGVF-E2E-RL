from __future__ import annotations

import pytest
import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.representation.experiments.answer_utility.controls import (
    AnswerUtilityArm,
    AnswerUtilityControlRow,
    build_same_image_answer_controls,
    zero_visual_bundle,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


def _bundle(
    value: float,
    *,
    requires_grad: bool,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 4), value, requires_grad=requires_grad),
        deepstack=tuple(
            torch.full(
                (1, 2, 4),
                value + float(index + 1) * 10.0,
                requires_grad=requires_grad,
            )
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token_id if position in evidence_positions else EVIDENCE_IGNORE_INDEX
            for position, token_id in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


def _samples_and_group() -> tuple[
    tuple[RepresentationTrainingSample, ...],
    SameImageReadoutGroup,
]:
    samples = tuple(
        RepresentationTrainingSample(
            sample_id=f"sample-{index}",
            image="/fixture/shared.png",
            image_id="shared-image",
            question=f"question {index}",
            target=f"target {index}",
            evidence_description=f"evidence {index}",
            short_answer=f"answer {index}",
        )
        for index in range(2)
    )
    rows: list[RepresentationReadoutRow] = []
    candidates: list[RepresentationCandidateObservation] = []
    for index, sample in enumerate(samples):
        token_ids = (1, 2, 2, 2, 2, 3, 5 + index, 7 + index)
        rows.append(
            RepresentationReadoutRow(
                sample_id=sample.sample_id,
                image_group_key=sample.image_group_key,
                source_visual_identity="shared-source-sha",
                supervision=_supervision(token_ids),
                input_ids=torch.tensor((token_ids,), dtype=torch.long),
                attention_mask=torch.ones((1, len(token_ids)), dtype=torch.bool),
                position_ids=torch.arange(len(token_ids)).view(1, len(token_ids)),
                source_positions=(1, 2),
                d_positions=(3, 4),
            )
        )
        candidates.append(
            RepresentationCandidateObservation(
                sample_id=sample.sample_id,
                image_group_key=sample.image_group_key,
                source_visual_identity="shared-source-sha",
                target_conditioning_provider=(
                    TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                ),
                projection_identities=(
                    "main",
                    "deepstack-8",
                    "deepstack-16",
                    "deepstack-24",
                ),
                visual=_bundle(float(index + 1), requires_grad=True),
            )
        )
    return samples, SameImageReadoutGroup(
        image_group_key="shared-image",
        source_visual_identity="shared-source-sha",
        source_visual=_bundle(0.25, requires_grad=False),
        rows=tuple(rows),
        candidates=tuple(candidates),
    )


def test_same_image_controls_swap_one_whole_main_plus_deepstack_bundle() -> None:
    samples, group = _samples_and_group()

    controls = build_same_image_answer_controls(
        samples,
        group,
        requires_zero_control=True,
        requires_wrong_control=True,
    )

    assert len(controls) == 2
    for index, row in enumerate(controls):
        correct = group.candidates[index].visual
        wrong = group.candidates[(index + 1) % 2].visual
        assert row.observation(AnswerUtilityArm.CORRECT) is correct
        assert row.observation(AnswerUtilityArm.WRONG_SAME_IMAGE_TARGET) is wrong
        assert row.wrong_source_sample_id == group.candidates[(index + 1) % 2].sample_id

        # Identity checks make a Frankenstein intervention (main from one D,
        # DeepStack from another D) visible at every accepted injection path.
        assert row.wrong is not None
        assert row.wrong.main is wrong.main
        assert len(row.wrong.deepstack) == 3
        assert all(
            actual is expected
            for actual, expected in zip(
                row.wrong.deepstack, wrong.deepstack, strict=True
            )
        )
        assert row.wrong.branch_layers == (8, 16, 24)

        zero = row.observation(AnswerUtilityArm.ZERO)
        assert zero is row.zero
        assert zero.branch_layers == correct.branch_layers
        assert zero.d_deepstack_active is correct.d_deepstack_active
        for zero_tensor, reference_tensor in zip(
            (zero.main, *zero.deepstack),
            (correct.main, *correct.deepstack),
            strict=True,
        ):
            assert zero_tensor.shape == reference_tensor.shape
            assert zero_tensor.dtype == reference_tensor.dtype
            assert zero_tensor.device == reference_tensor.device
            assert not zero_tensor.requires_grad
            assert torch.count_nonzero(zero_tensor).item() == 0


def test_zero_control_rejects_a_partially_zeroed_deepstack_intervention() -> None:
    reference = _bundle(1.0, requires_grad=True)
    wrong = _bundle(2.0, requires_grad=True)
    zero = zero_visual_bundle(reference)
    partially_zero = RepresentationVisualTensorBundle(
        main=zero.main,
        deepstack=(
            zero.deepstack[0],
            torch.ones_like(zero.deepstack[1]),
            zero.deepstack[2],
        ),
        branch_layers=zero.branch_layers,
    )

    with pytest.raises(ValueError, match="atomically zero main and all DeepStack"):
        AnswerUtilityControlRow(
            sample_id="sample-0",
            correct_source_sample_id="sample-0",
            wrong_source_sample_id="sample-1",
            correct=reference,
            zero=partially_zero,
            wrong=wrong,
        )


def test_control_builder_rejects_a_non_counterfactual_duplicate_target() -> None:
    samples, group = _samples_and_group()
    duplicate = RepresentationTrainingSample(
        sample_id=samples[1].sample_id,
        image=samples[1].image,
        image_id=samples[1].image_id,
        question=samples[1].question,
        target=samples[0].target,
        evidence_description=samples[1].evidence_description,
        short_answer=samples[1].short_answer,
    )

    with pytest.raises(ValueError, match="requires distinct targets"):
        build_same_image_answer_controls(
            (samples[0], duplicate),
            group,
            requires_zero_control=True,
            requires_wrong_control=True,
        )


def test_wrong_control_skips_a_target_with_the_same_normalized_answer() -> None:
    samples, group = _samples_and_group()
    third = RepresentationTrainingSample(
        sample_id="sample-2",
        image=samples[0].image,
        image_id=samples[0].image_id,
        question="question 2",
        target="target 2",
        evidence_description="evidence 2",
        short_answer="different answer",
    )
    duplicate_answer = RepresentationTrainingSample(
        sample_id=samples[1].sample_id,
        image=samples[1].image,
        image_id=samples[1].image_id,
        question=samples[1].question,
        target=samples[1].target,
        evidence_description=samples[1].evidence_description,
        short_answer="  ANSWER 0  ",
    )
    third_candidate = RepresentationCandidateObservation(
        sample_id=third.sample_id,
        image_group_key=third.image_group_key,
        source_visual_identity=group.source_visual_identity,
        target_conditioning_provider=(
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ),
        projection_identities=("main", "deepstack-8", "deepstack-16", "deepstack-24"),
        visual=_bundle(3.0, requires_grad=True),
    )
    third_row = RepresentationReadoutRow(
        sample_id=third.sample_id,
        image_group_key=third.image_group_key,
        source_visual_identity=group.source_visual_identity,
        supervision=group.rows[0].supervision,
        input_ids=group.rows[0].input_ids,
        attention_mask=group.rows[0].attention_mask,
        position_ids=group.rows[0].position_ids,
        source_positions=group.rows[0].source_positions,
        d_positions=group.rows[0].d_positions,
    )
    expanded_group = SameImageReadoutGroup(
        image_group_key=group.image_group_key,
        source_visual_identity=group.source_visual_identity,
        source_visual=group.source_visual,
        rows=(*group.rows, third_row),
        candidates=(*group.candidates, third_candidate),
    )

    controls = build_same_image_answer_controls(
        (samples[0], duplicate_answer, third),
        expanded_group,
        requires_zero_control=True,
        requires_wrong_control=True,
    )

    assert controls[0].wrong_source_sample_id == third.sample_id


def test_correct_only_profile_does_not_construct_or_validate_wrong_arms() -> None:
    samples, group = _samples_and_group()
    same_answers = tuple(
        RepresentationTrainingSample(
            sample_id=sample.sample_id,
            image=sample.image,
            image_id=sample.image_id,
            question=sample.question,
            target=sample.target,
            evidence_description=sample.evidence_description,
            short_answer="same",
        )
        for sample in samples
    )

    controls = build_same_image_answer_controls(
        same_answers,
        group,
        requires_zero_control=False,
        requires_wrong_control=False,
    )

    assert all(row.zero is None and row.wrong is None for row in controls)

    with pytest.raises(ValueError, match="no answer-safe wrong target"):
        build_same_image_answer_controls(
            same_answers,
            group,
            requires_zero_control=True,
            requires_wrong_control=True,
        )
