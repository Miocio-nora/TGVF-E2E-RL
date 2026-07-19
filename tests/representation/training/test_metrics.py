from __future__ import annotations

import math

import pytest
import torch

from tgvf_rl.representation.training.metrics import (
    INDEX_STABLE_TIE_BREAK,
    NEAR_IDENTICAL_TOKEN_STD_THRESHOLD,
    NLL_LOWER_IS_BETTER,
    ReadoutNLLs,
    attention_diagnostics,
    grouped_query_row_metrics,
    grouped_query_score_metrics,
    grouped_readout_metrics,
    grouped_tensor_diagnostics,
    norm_comparison_diagnostics,
    query_score_matrix_metrics,
    readout_sample_metrics,
    summarize_attention_diagnostics,
    summarize_query_score_matrices,
    summarize_readout,
    summarize_representation_health,
    summarize_tensor_diagnostics,
    tensor_distribution_diagnostics,
)


def test_readout_reports_lower_nll_advantages_and_strict_win_rates() -> None:
    rows = (
        ReadoutNLLs(
            correct_d=1.0,
            target_only=2.0,
            random_d=3.0,
            wrong_same_image_d=4.0,
            wrong_different_image_d=0.5,
        ),
        ReadoutNLLs(
            correct_d=2.0,
            target_only=1.0,
            random_d=4.0,
            wrong_different_image_d=5.0,
        ),
        ReadoutNLLs(
            correct_d=3.0,
            target_only=4.0,
            random_d=5.0,
            wrong_same_image_d=4.0,
            wrong_different_image_d=5.0,
        ),
    )

    first = readout_sample_metrics(rows[0])
    assert first.advantage_vs_target_only == 1.0
    assert first.advantage_vs_random_d == 2.0
    assert first.advantage_vs_wrong_same_image_d == 3.0
    assert first.advantage_vs_wrong_different_image_d == -0.5
    assert not first.correct_beats_all_available_controls
    assert first.correct_beats_all_controls is False

    metrics = summarize_readout(rows)
    assert metrics.score_semantics == NLL_LOWER_IS_BETTER
    assert metrics.sample_count == 3
    assert metrics.mean_correct_d_nll == 2.0
    assert metrics.median_correct_d_nll == 2.0
    assert metrics.target_only.mean_nll == pytest.approx(7 / 3)
    assert metrics.target_only.mean_correct_advantage == pytest.approx(1 / 3)
    assert metrics.target_only.correct_win_rate == pytest.approx(2 / 3)
    assert metrics.random_d.mean_correct_advantage == 2.0
    assert metrics.random_d.correct_win_rate == 1.0
    assert metrics.wrong_same_image_d.available_count == 2
    assert metrics.wrong_same_image_d.mean_correct_advantage == 2.0
    assert metrics.wrong_same_image_d.correct_win_rate == 1.0
    assert metrics.wrong_different_image_d.mean_correct_advantage == 1.5
    assert metrics.wrong_different_image_d.correct_win_rate == pytest.approx(2 / 3)
    assert metrics.correct_beats_all_available_controls_rate == pytest.approx(1 / 3)
    assert metrics.complete_control_sample_count == 2
    assert metrics.correct_beats_all_controls_rate == 0.5


def test_readout_tie_is_not_a_win_and_missing_controls_stay_unavailable() -> None:
    row = ReadoutNLLs(correct_d=2, target_only=2, random_d=3)
    sample = readout_sample_metrics(row)
    metrics = summarize_readout((row,))

    assert sample.advantage_vs_target_only == 0.0
    assert not sample.correct_beats_all_available_controls
    assert sample.correct_beats_all_controls is None
    assert metrics.target_only.correct_win_rate == 0.0
    assert metrics.wrong_same_image_d.available_count == 0
    assert metrics.wrong_same_image_d.mean_nll is None
    assert metrics.wrong_same_image_d.correct_win_rate is None
    assert metrics.complete_control_sample_count == 0
    assert metrics.correct_beats_all_controls_rate is None


def test_readout_groups_are_optional_and_typed_nlls_fail_closed() -> None:
    rows = (
        ReadoutNLLs(1, 2, 3),
        ReadoutNLLs(2, 3, 4),
        ReadoutNLLs(3, 4, 5),
    )
    assert grouped_readout_metrics(rows) == {}
    grouped = grouped_readout_metrics(rows, ("hard", "easy", "hard"))
    assert tuple(grouped) == ("easy", "hard")
    assert grouped["easy"].sample_count == 1
    assert grouped["hard"].sample_count == 2

    with pytest.raises(ValueError, match="align one-to-one"):
        grouped_readout_metrics(rows, ("only-one",))
    with pytest.raises(ValueError, match="finite"):
        ReadoutNLLs(float("nan"), 1.0, 2.0)
    with pytest.raises(ValueError, match="finite"):
        ReadoutNLLs(1.0, 2.0, float("inf"))


def test_query_matrix_uses_nll_lower_is_better_and_sample_weighting() -> None:
    matrix = torch.tensor(
        [
            [1.0, 4.0, 3.0],
            [2.0, 1.0, 0.0],
            [4.0, 2.0, 1.0],
        ]
    )
    metrics = query_score_matrix_metrics(matrix)

    assert metrics.score_semantics == NLL_LOWER_IS_BETTER
    assert metrics.tie_break_semantics == INDEX_STABLE_TIE_BREAK
    assert metrics.nll_matrix == ((1.0, 4.0, 3.0), (2.0, 1.0, 0.0), (4.0, 2.0, 1.0))
    assert [row.diagonal_rank for row in metrics.rows] == [1, 2, 1]
    assert [row.diagonal_gap for row in metrics.rows] == [2.0, -1.0, 1.0]
    assert metrics.top1_accuracy == pytest.approx(2 / 3)
    assert metrics.top2_accuracy == 1.0
    assert metrics.mean_reciprocal_rank == pytest.approx(5 / 6)
    assert metrics.mean_diagonal_gap == pytest.approx(2 / 3)
    assert metrics.median_diagonal_gap == 1.0

    tied = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    tied_metrics = query_score_matrix_metrics(tied)
    # Exact legacy behavior: stable sorting favors column zero on equal NLL.
    assert [row.diagonal_rank for row in tied_metrics.rows] == [1, 2]
    assert tied_metrics.top1_accuracy == 0.5
    assert tied_metrics.top2_accuracy == 1.0
    assert tied_metrics.mean_reciprocal_rank == 0.75
    assert tied_metrics.mean_diagonal_gap == 0.0

    summary = summarize_query_score_matrices((matrix, tied))
    assert summary.group_count == 2
    assert summary.sample_count == 5
    assert summary.retrieval_top1 == pytest.approx(3 / 5)
    assert summary.retrieval_top2 == 1.0
    assert summary.mean_reciprocal_rank == pytest.approx(4 / 5)
    assert summary.mean_diagonal_gap == pytest.approx(2 / 5)
    assert summary.median_diagonal_gap == 0.0


def test_query_grouping_and_invalid_matrices_fail_closed() -> None:
    first = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    second = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    assert grouped_query_score_metrics((first, second)) == {}
    grouped = grouped_query_score_metrics((first, second), ("b", "a"))
    assert tuple(grouped) == ("a", "b")
    assert grouped["a"].sample_count == 2
    assert grouped["b"].retrieval_top1 == 1.0

    row_grouped = grouped_query_row_metrics(
        (first, second),
        (("evidence-a", "evidence-b"), ("evidence-b", "evidence-b")),
    )
    assert row_grouped["evidence-a"].sample_count == 1
    assert row_grouped["evidence-a"].group_count == 1
    assert row_grouped["evidence-a"].retrieval_top1 == 1.0
    assert row_grouped["evidence-b"].sample_count == 3
    assert row_grouped["evidence-b"].group_count == 2
    assert row_grouped["evidence-b"].retrieval_top1 == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="align one-to-one with matrices"):
        grouped_query_row_metrics((first, second), (("a", "b"),))
    with pytest.raises(ValueError, match="align one-to-one with metric inputs"):
        grouped_query_row_metrics((first, second), (("a",), ("b", "b")))

    with pytest.raises(ValueError, match="square"):
        query_score_matrix_metrics(torch.ones(2, 3))
    with pytest.raises(ValueError, match="at least two"):
        query_score_matrix_metrics(torch.ones(1, 1))
    with pytest.raises(ValueError, match="finite"):
        query_score_matrix_metrics(torch.tensor([[1.0, float("inf")], [2.0, 1.0]]))
    with pytest.raises(TypeError, match="floating"):
        query_score_matrix_metrics(torch.ones(2, 2, dtype=torch.long))


def test_tensor_health_reports_finite_collapse_and_norm_distributions_only() -> None:
    healthy = tensor_distribution_diagnostics(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    collapsed = tensor_distribution_diagnostics(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    nonfinite = tensor_distribution_diagnostics(
        torch.tensor([[1.0, float("nan")], [float("inf"), 2.0]])
    )

    assert healthy.shape == (2, 2)
    assert healthy.fully_finite
    assert healthy.finite_rate == 1.0
    assert healthy.element_values.mean == 1.75
    assert healthy.token_norms.mean == 2.5
    assert healthy.token_norms.population_std == 2.5
    assert healthy.near_identical_token_collapse is False
    assert healthy.collapse_definition_threshold == NEAR_IDENTICAL_TOKEN_STD_THRESHOLD

    assert collapsed.near_identical_token_collapse is True
    assert collapsed.token_cosine_to_mean_population_std == pytest.approx(0.0)

    assert not nonfinite.fully_finite
    assert nonfinite.element_values.finite_count == 2
    assert nonfinite.finite_rate == 0.5
    assert nonfinite.token_norms.finite_count == 0
    assert nonfinite.token_norms.mean is None
    assert nonfinite.near_identical_token_collapse is None

    summary = summarize_tensor_diagnostics((healthy, collapsed, nonfinite))
    assert summary.tensor_count == 3
    assert summary.fully_finite_tensor_rate == pytest.approx(2 / 3)
    assert summary.mean_element_finite_rate == pytest.approx(5 / 6)
    assert summary.collapse_evaluable_count == 2
    assert summary.near_identical_token_collapse_count == 1
    assert summary.near_identical_token_collapse_rate == 0.5
    assert summary.mean_token_norm_across_tensors.count == 2
    assert not hasattr(summary, "promotion_threshold")
    assert not hasattr(summary, "loss")
    assert not hasattr(summary, "collapse_warning")


def test_norm_comparison_is_a_diagnostic_and_group_reduction_is_optional() -> None:
    diagnostic = norm_comparison_diagnostics(
        torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
    )
    ratio = diagnostic.d_to_source_mean_token_norm_ratio
    assert ratio.count == 2
    assert ratio.mean == 1.5
    assert ratio.minimum == 1.0
    assert ratio.maximum == 2.0
    assert not hasattr(diagnostic, "loss")

    collapsed = tensor_distribution_diagnostics(torch.ones(2, 2))
    healthy = tensor_distribution_diagnostics(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    assert grouped_tensor_diagnostics((collapsed, healthy)) == {}
    grouped = grouped_tensor_diagnostics((collapsed, healthy), ("same", "same"))
    assert grouped["same"].tensor_count == 2
    assert grouped["same"].near_identical_token_collapse_rate == 0.5


def test_paired_representation_health_matches_historical_warning_rule() -> None:
    source = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    healthy = norm_comparison_diagnostics(
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]), source
    )
    collapsed = norm_comparison_diagnostics(
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]), source
    )
    nonfinite = norm_comparison_diagnostics(
        torch.tensor([[float("nan"), 1.0], [float("inf"), 2.0]]), source
    )
    partially_nonfinite = norm_comparison_diagnostics(
        torch.tensor([[1.0, 0.0], [float("nan"), 2.0]]), source
    )

    low_collapse_rate = summarize_representation_health((collapsed, *([healthy] * 19)))
    assert low_collapse_rate.sample_count == 20
    assert low_collapse_rate.joint_d_source_finite_rate == 1.0
    assert low_collapse_rate.d_near_identical_token_collapse_rate == 0.05
    assert not low_collapse_rate.collapse_warning

    unhealthy = summarize_representation_health((healthy, collapsed, nonfinite))
    assert unhealthy.joint_d_source_finite_rate == pytest.approx(2 / 3)
    assert unhealthy.d_near_identical_token_collapse_rate == pytest.approx(1 / 3)
    assert unhealthy.collapse_warning
    assert unhealthy.mean_d_token_norm.count == 2
    assert unhealthy.mean_source_visual_token_norm.count == 3
    assert unhealthy.d_to_source_mean_token_norm_ratio.count == 2

    partial = summarize_representation_health((healthy, partially_nonfinite))
    assert partial.mean_d_token_norm.count == 1
    assert partial.mean_source_visual_token_norm.count == 2
    assert partial.d_to_source_mean_token_norm_ratio.count == 1

    with pytest.raises(ValueError, match="at least one sample"):
        summarize_representation_health(())


def test_attention_diagnostics_reproduce_registered_rank_fold_and_reductions() -> None:
    sub_slot = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )
    diagnostic = attention_diagnostics(
        sub_slot_attention_weights=sub_slot,
        attention_weights=torch.tensor([[0.0, 1.0, 0.0]]),
        topk=2,
    )
    assert diagnostic is not None
    assert diagnostic.slot_count == 2
    assert diagnostic.visual_token_count == 3
    assert diagnostic.effective_topk == 2
    assert diagnostic.entropy_values[0] == pytest.approx(math.log(2), abs=1e-6)
    assert diagnostic.entropy_values[1] == pytest.approx(0.0, abs=1e-6)
    assert diagnostic.top1_mass_values == pytest.approx((0.5, 1.0))
    assert diagnostic.topk_mass_values == pytest.approx((1.0, 1.0))
    assert diagnostic.visual_token_coverage == 1.0

    # A valid sub-slot tensor has priority over the fallback attention tensor.
    priority = attention_diagnostics(
        sub_slot_attention_weights=torch.tensor([[1.0, 0.0]]),
        attention_weights=torch.tensor([[0.0, 1.0]]),
        topk=1,
    )
    assert priority is not None
    assert priority.top1_mass_values == (1.0,)
    assert priority.visual_token_coverage == 0.5

    same_k_priority = attention_diagnostics(
        sub_slot_attention_weights=torch.tensor([[1.0, 0.0]]),
        topk=2,
    )
    assert same_k_priority is not None
    summary = summarize_attention_diagnostics((diagnostic, same_k_priority))
    assert summary.observation_count == 2
    assert summary.slot_count == 3
    assert summary.effective_topk == 2
    assert summary.entropy.count == 3
    assert summary.top1_mass.mean == pytest.approx(5 / 6)
    assert summary.topk_mass.mean == 1.0
    assert summary.mean_visual_token_coverage == 1.0

    with pytest.raises(ValueError, match="different effective_topk"):
        summarize_attention_diagnostics((diagnostic, priority))


def test_attention_fallback_and_validation_do_not_guess_invalid_axes() -> None:
    fallback = attention_diagnostics(
        sub_slot_attention_weights=torch.ones(2),
        attention_weights=torch.tensor([[0.25, 0.75]]),
        topk=1,
    )
    assert fallback is not None
    assert fallback.top1_mass_values == (0.75,)
    assert attention_diagnostics(sub_slot_attention_weights=torch.ones(2)) is None

    with pytest.raises(ValueError, match="non-negative"):
        attention_diagnostics(attention_weights=torch.tensor([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="finite"):
        attention_diagnostics(attention_weights=torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="positive integer"):
        attention_diagnostics(attention_weights=torch.ones(1, 2), topk=0)
