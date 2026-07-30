from __future__ import annotations

import math

import pytest
import torch

import tgvf_rl.representation.experiments.image_axis_grounding.trainer as trainer_module
from tgvf_rl.representation.experiments.image_axis_grounding.streaming import (
    ImageAxisStreamingMetrics,
)
from tgvf_rl.representation.experiments.image_axis_grounding.trainer import (
    IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION,
    ImageAxisGroundingObjectiveConfig,
    ImageAxisGroundingStepMetrics,
)
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
)
from tgvf_rl.representation.training.streaming import StreamingBackwardMetrics


def _legacy_metrics() -> StreamingBackwardMetrics:
    return StreamingBackwardMetrics(
        matrix_ce_numerator=torch.tensor(4.0),
        l_gen_numerator=torch.tensor(6.0),
        norm_numerator=torch.tensor(2.0),
        local_row_count=2,
        local_sample_count=2,
        weighted_local_mean=torch.tensor(5.1),
        weighted_norm_local_mean=torch.tensor(0.1),
        qwen_forward_batch_sizes=(4,),
    )


def test_v1_objective_is_named_and_has_no_silent_tuning_surface() -> None:
    objective = ImageAxisGroundingObjectiveConfig()

    assert objective.schema_version == IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION
    assert objective.loss_weights == (1.0,)
    assert objective.validation_payload() == {
        "schema_version": IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION,
        "image_axis_matrix_weight": 1.0,
        "image_axis_temperature": 1.0,
        "negative_count": 1,
    }
    with pytest.raises(ValueError, match="matrix_weight"):
        ImageAxisGroundingObjectiveConfig(image_axis_matrix_weight=0.5)
    with pytest.raises(ValueError, match="temperature"):
        ImageAxisGroundingObjectiveConfig(image_axis_temperature=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        ImageAxisGroundingObjectiveConfig(negative_count=2)


def test_metric_accumulation_keeps_image_axis_numerator_and_count_separate() -> None:
    totals = torch.zeros(7, dtype=torch.float64)
    metrics = ImageAxisStreamingMetrics(
        legacy=_legacy_metrics(),
        image_axis_numerator=torch.tensor(1.5),
        local_image_axis_row_count=2,
        correct_score_sum=torch.tensor(-2.0),
        wrong_score_sum=torch.tensor(-5.0),
        correct_top1_count=1,
        image_axis_qwen_forward_batch_sizes=(4,),
    )

    trainer_module._accumulate_metrics(
        totals,
        metrics,
        expected_sample_count=2,
        expected_image_count=2,
    )

    assert torch.equal(
        totals,
        torch.tensor((4.0, 6.0, 2.0, 1.5, -2.0, -5.0, 1.0), dtype=torch.float64),
    )
    with pytest.raises(RuntimeError, match="incorrect local counts"):
        trainer_module._accumulate_metrics(
            torch.zeros(7, dtype=torch.float64),
            metrics,
            expected_sample_count=2,
            expected_image_count=0,
        )


def test_zero_eligible_window_has_zero_loss_and_no_fabricated_diagnostics() -> None:
    metrics = ImageAxisGroundingStepMetrics(
        global_step=1,
        global_matrix_ce_loss=0.8,
        global_l_gen_loss=0.7,
        global_norm_loss=0.2,
        global_weighted_norm_loss=0.02,
        global_image_axis_loss=0.0,
        global_weighted_image_axis_loss=0.0,
        global_image_axis_score_gap=None,
        global_image_axis_correct_top1=None,
        global_total_loss=1.52,
        global_row_count=8,
        global_sample_count=8,
        global_image_axis_row_count=0,
        gradient_norm_before_clip=1.2,
        learning_rate=1e-6,
        local_sample_ids=("sample-0", "sample-1"),
        local_qwen_forward_batch_sizes=(4,),
        local_legacy_qwen_forward_batch_sizes=(4,),
        local_image_axis_qwen_forward_batch_sizes=(),
    )

    assert metrics.global_image_axis_row_count == 0
    assert metrics.global_image_axis_score_gap is None
    assert metrics.global_image_axis_correct_top1 is None
    assert math.isclose(metrics.global_total_loss, 1.52)
    with pytest.raises(ValueError, match="cannot have image diagnostics"):
        ImageAxisGroundingStepMetrics(
            **{
                **{
                    field: getattr(metrics, field)
                    for field in metrics.__dataclass_fields__
                },
                "global_image_axis_score_gap": 0.0,
            }
        )


def test_v1_fails_closed_on_direct_multi_group_microsteps() -> None:
    accumulation = RepresentationAccumulationIdentity(
        gradient_accumulation_steps=4,
        data_parallel_world_size=2,
    )
    assert trainer_module._execution_group_counts(accumulation) == (4, 1)

    direct = RepresentationAccumulationIdentityV2(
        gradient_accumulation_steps=1,
        data_parallel_world_size=2,
        groups_per_rank_per_optimizer_step=2,
    )
    with pytest.raises(ValueError, match="rejects direct multi-group"):
        trainer_module._execution_group_counts(direct)


@pytest.mark.parametrize(
    ("mask", "error"),
    [
        ((True,), "one value per row"),
        ((True, False), "group-homogeneous"),
        ((1, 1), "must be bool"),
    ],
)
def test_manifest_preflight_mask_fails_closed(
    mask: tuple[object, ...],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        trainer_module._validate_predicted_mask(mask, expected_size=2)
