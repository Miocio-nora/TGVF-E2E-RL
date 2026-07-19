from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tgvf_rl.representation.training.losses import (
    CausalEvidenceLosses,
    EVIDENCE_IGNORE_INDEX,
    MatrixCEScoreMode,
    _causal_evidence_losses_from_native_labels,
    _materialize_native_causal_evidence_labels,
    causal_evidence_losses,
    evidence_readability_loss_terms,
    historical_norm_loss_terms,
    historical_sample_norm_loss,
    historical_visual_token_norm_loss,
    matrix_ce_cell_scores,
    same_image_matrix_ce_loss,
    same_image_matrix_ce_loss_terms,
    same_image_matrix_ce_score_gradients,
)


def test_balanced_matrix_ce_equalizes_equal_mean_nll_across_lengths() -> None:
    summed_log_likelihood = torch.tensor([-2.0, -6.0])
    losses = CausalEvidenceLosses(
        per_sample_token_mean_nll=torch.tensor([2.0, 2.0]),
        per_sample_summed_log_likelihood=summed_log_likelihood,
        valid_token_counts=torch.tensor([1, 3]),
    )

    balanced = matrix_ce_cell_scores(
        losses,
        mode=MatrixCEScoreMode.BALANCED,
        temperature=1.0,
    )
    colder = matrix_ce_cell_scores(
        losses,
        mode=MatrixCEScoreMode.BALANCED,
        temperature=0.5,
    )
    legacy = matrix_ce_cell_scores(
        losses,
        mode=MatrixCEScoreMode.LEGACY_SUMMED_NLL,
    )

    assert torch.equal(balanced, torch.tensor([-2.0, -2.0]))
    assert torch.equal(colder, torch.tensor([-4.0, -4.0]))
    assert legacy is summed_log_likelihood
    with pytest.raises(ValueError, match="requires temperature 1.0"):
        matrix_ce_cell_scores(
            losses,
            mode=MatrixCEScoreMode.LEGACY_SUMMED_NLL,
            temperature=0.5,
        )


def test_historical_norm_formula_is_fp32_and_detaches_source() -> None:
    d_tokens = torch.tensor(
        [[3.0, 4.0], [5.0, 12.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    source_tokens = torch.tensor(
        [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
        requires_grad=True,
    )

    loss = historical_visual_token_norm_loss(d_tokens, source_tokens)
    expected = torch.log((torch.tensor([5.0, 13.0]) + 1e-6) / 4.0).square().mean()

    assert loss.dtype == torch.float32
    assert torch.allclose(loss, expected, atol=1e-7, rtol=0)
    loss.backward()
    assert d_tokens.grad is not None
    assert torch.count_nonzero(d_tokens.grad).item() == d_tokens.numel()
    assert source_tokens.grad is None


def test_checked_historical_norm_api_rejects_nonfinite_inputs() -> None:
    finite = torch.ones(1, 2)
    nonfinite = torch.tensor([[float("nan"), 1.0]])

    with pytest.raises(ValueError, match="D tokens must be finite"):
        historical_visual_token_norm_loss(nonfinite, finite)
    with pytest.raises(ValueError, match="source visual tokens must be finite"):
        historical_visual_token_norm_loss(finite, nonfinite)


def test_historical_sample_norm_reduces_branch_mean_then_main_equally() -> None:
    main_d = torch.tensor([[3.0, 4.0]], requires_grad=True)
    main_source = torch.tensor([[0.0, 2.0]])
    branch_d = tuple(
        torch.tensor([[value, 0.0]], requires_grad=True) for value in (2.0, 4.0, 8.0)
    )
    branch_source = tuple(torch.tensor([[value, 0.0]]) for value in (1.0, 2.0, 4.0))

    sample = historical_sample_norm_loss(
        main_d,
        main_source,
        branch_d,
        branch_source,
    )
    main = historical_visual_token_norm_loss(main_d, main_source)
    branches = torch.stack(
        tuple(
            historical_visual_token_norm_loss(d, source)
            for d, source in zip(branch_d, branch_source, strict=True)
        )
    )
    terms = historical_norm_loss_terms((sample, sample * 2))

    assert torch.equal(sample, (main + branches.mean()) / 2)
    assert terms.sample_count == 2
    assert torch.equal(terms.numerator, sample * 3)
    with pytest.raises(ValueError, match="exactly three"):
        historical_sample_norm_loss(
            main_d,
            main_source,
            branch_d[:2],
            branch_source[:2],
        )


def test_causal_evidence_losses_shift_mask_and_reduce_per_sample() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [-0.5, 1.5, 0.0],
                [9.0, -9.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            [
                [-1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0],
                [3.0, 1.0, -2.0],
                [7.0, 8.0, 9.0],
            ],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [EVIDENCE_IGNORE_INDEX, 0, 1, EVIDENCE_IGNORE_INDEX],
            [EVIDENCE_IGNORE_INDEX, 2, EVIDENCE_IGNORE_INDEX, 0],
        ]
    )

    result = causal_evidence_losses(logits, labels)

    row_0_nll = torch.stack(
        (
            -F.log_softmax(logits[0, 0], dim=-1)[0],
            -F.log_softmax(logits[0, 1], dim=-1)[1],
        )
    )
    row_1_nll = torch.stack(
        (
            -F.log_softmax(logits[1, 0], dim=-1)[2],
            -F.log_softmax(logits[1, 2], dim=-1)[0],
        )
    )
    expected_sums = torch.stack((row_0_nll.sum(), row_1_nll.sum()))

    assert torch.equal(result.valid_token_counts, torch.tensor([2, 2]))
    assert torch.allclose(result.per_sample_token_mean_nll, expected_sums / 2)
    assert torch.allclose(
        result.per_sample_summed_log_likelihood,
        -expected_sums,
    )

    result.per_sample_token_mean_nll.mean().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, -1]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() == 0
    assert torch.count_nonzero(logits.grad[1, 1]).item() == 0


def test_causal_evidence_losses_reject_sample_with_no_post_shift_labels() -> None:
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor(
        [
            [EVIDENCE_IGNORE_INDEX, 1, EVIDENCE_IGNORE_INDEX],
            [2, EVIDENCE_IGNORE_INDEX, EVIDENCE_IGNORE_INDEX],
        ]
    )

    # The second sample's only label is at position zero and therefore has no
    # causal predictor.  The historical clamp_min(1) path silently returned
    # zero for it; the native loss deliberately fails closed.
    with pytest.raises(ValueError, match="every sample.*causal shift"):
        causal_evidence_losses(logits, labels)


def test_causal_evidence_losses_reject_invalid_non_ignored_label() -> None:
    with pytest.raises(ValueError, match="valid vocabulary ids"):
        causal_evidence_losses(
            torch.zeros(1, 2, 3),
            torch.tensor([[EVIDENCE_IGNORE_INDEX, 3]]),
        )


def test_generic_causal_evidence_api_retains_two_content_host_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor(
        [
            [EVIDENCE_IGNORE_INDEX, 1, EVIDENCE_IGNORE_INDEX],
            [EVIDENCE_IGNORE_INDEX, 2, 3],
        ]
    )
    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(tensor: torch.Tensor, *args: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    causal_evidence_losses(logits, labels)

    assert item_calls == 2


def test_native_causal_label_proof_rejects_invalid_rows_and_mutation() -> None:
    with pytest.raises(ValueError, match="every sample.*causal shift"):
        _materialize_native_causal_evidence_labels(
            ((1, EVIDENCE_IGNORE_INDEX),),
            2,
            device=torch.device("cpu"),
            vocabulary_size=4,
        )
    with pytest.raises(ValueError, match="valid vocabulary ids"):
        _materialize_native_causal_evidence_labels(
            ((EVIDENCE_IGNORE_INDEX, 4),),
            2,
            device=torch.device("cpu"),
            vocabulary_size=4,
        )

    proven = _materialize_native_causal_evidence_labels(
        ((EVIDENCE_IGNORE_INDEX, 1),),
        2,
        device=torch.device("cpu"),
        vocabulary_size=4,
    )
    proven.values[0, 1] = 2
    with pytest.raises(ValueError, match="changed after construction"):
        _causal_evidence_losses_from_native_labels(torch.zeros(1, 2, 4), proven)


def test_evidence_readability_reduces_token_mean_per_sample_then_sample_mean() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [2.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor(
        [
            [EVIDENCE_IGNORE_INDEX, 0, EVIDENCE_IGNORE_INDEX, EVIDENCE_IGNORE_INDEX],
            [EVIDENCE_IGNORE_INDEX, 1, 0, 1],
        ]
    )

    losses = causal_evidence_losses(logits, labels)
    terms = evidence_readability_loss_terms(losses)
    global_token_mean = (
        -losses.per_sample_summed_log_likelihood.sum() / losses.valid_token_counts.sum()
    )

    assert torch.equal(losses.valid_token_counts, torch.tensor([1, 3]))
    assert terms.sample_count == 2
    assert torch.equal(terms.numerator, losses.per_sample_token_mean_nll.sum())
    assert torch.equal(terms.mean, losses.per_sample_token_mean_nll.mean())
    assert not torch.allclose(terms.mean, global_token_mean)


def test_matrix_ce_diagonal_preferred_value_oracle_has_no_temperature() -> None:
    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    loss = same_image_matrix_ce_loss((scores,))

    expected = torch.log1p(torch.exp(torch.tensor(-2.0)))
    assert torch.allclose(loss, expected)


def test_matrix_ce_uses_total_row_mean_across_different_group_sizes() -> None:
    first = torch.tensor([[1.2, -0.4], [0.1, 0.8]], dtype=torch.float64)
    second = torch.tensor(
        [[0.4, -0.1, 0.2], [-0.6, 1.1, 0.0], [0.3, -0.2, 0.7]],
        dtype=torch.float64,
    )

    loss = same_image_matrix_ce_loss((first, second))

    expected_sum = sum(
        torch.logsumexp(row, dim=-1) - row[row_index]
        for matrix in (first, second)
        for row_index, row in enumerate(matrix)
    )
    assert torch.allclose(loss, expected_sum / 5)

    mean_of_group_means = (
        F.cross_entropy(first, torch.arange(2))
        + F.cross_entropy(second, torch.arange(3))
    ) / 2
    assert not torch.allclose(loss, mean_of_group_means)


def test_matrix_ce_terms_support_global_rank_four_vs_five_row_reduction() -> None:
    rank_four = same_image_matrix_ce_loss_terms((torch.zeros(4, 4),))
    rank_five_scores = torch.eye(5) * 4.0
    rank_five = same_image_matrix_ce_loss_terms((rank_five_scores,))

    global_mean = (rank_four.numerator + rank_five.numerator) / (
        rank_four.valid_row_count + rank_five.valid_row_count
    )
    direct = same_image_matrix_ce_loss((torch.zeros(4, 4), rank_five_scores))
    incorrect_equal_rank_mean = (rank_four.mean + rank_five.mean) / 2

    assert rank_four.valid_row_count == 4
    assert rank_five.valid_row_count == 5
    assert torch.allclose(global_mean, direct)
    assert not torch.allclose(global_mean, incorrect_equal_rank_mean)


def test_explicit_matrix_ce_score_gradients_match_autograd() -> None:
    first = torch.tensor(
        [[0.4, -0.8], [1.2, 0.3]], dtype=torch.float32, requires_grad=True
    )
    second = torch.tensor(
        [[0.1, 0.6, -0.4], [-0.7, 1.4, 0.2], [0.5, -0.3, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, explicit_gradients = same_image_matrix_ce_score_gradients((first, second))
    autograd_gradients = torch.autograd.grad(loss, (first, second))

    assert torch.allclose(loss, same_image_matrix_ce_loss((first, second)))
    assert len(explicit_gradients) == 2
    for explicit, automatic in zip(explicit_gradients, autograd_gradients, strict=True):
        assert not explicit.requires_grad
        assert torch.allclose(explicit, automatic, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_explicit_matrix_ce_low_precision_uses_legacy_fp32_softmax(
    dtype: torch.dtype,
) -> None:
    scores = torch.tensor(
        [[0.125, -0.75], [1.5, 0.25]], dtype=dtype, requires_grad=True
    )

    _, (explicit_gradient,) = same_image_matrix_ce_score_gradients((scores,))
    expected = torch.softmax(scores.detach().float(), dim=-1).to(dtype=dtype)
    expected[torch.arange(2), torch.arange(2)] -= 1
    expected = expected / 2

    assert explicit_gradient.dtype == dtype
    assert torch.equal(explicit_gradient, expected)


def test_matrix_ce_zero_valid_groups_has_no_training_signal() -> None:
    empty = torch.empty((0, 0), requires_grad=True)

    absent_loss = same_image_matrix_ce_loss(())
    empty_loss, gradients = same_image_matrix_ce_score_gradients((empty,))

    assert absent_loss.shape == ()
    assert absent_loss.item() == 0
    assert not absent_loss.requires_grad
    assert empty_loss.shape == ()
    assert empty_loss.item() == 0
    assert not empty_loss.requires_grad
    assert len(gradients) == 1
    assert gradients[0].shape == (0, 0)
    assert not gradients[0].requires_grad


def test_matrix_ce_rejects_non_square_or_mixed_dtype_groups() -> None:
    with pytest.raises(ValueError, match="square"):
        same_image_matrix_ce_loss((torch.zeros(2, 3),))
    with pytest.raises(ValueError, match="share device and dtype"):
        same_image_matrix_ce_loss(
            (torch.zeros(2, 2, dtype=torch.float32), torch.zeros(2, 2).double())
        )
    with pytest.raises(ValueError, match="at least two targets"):
        same_image_matrix_ce_loss((torch.zeros(1, 1),))
    with pytest.raises(TypeError, match="require FP16, BF16, or FP32"):
        same_image_matrix_ce_score_gradients((torch.zeros(2, 2, dtype=torch.float64),))
