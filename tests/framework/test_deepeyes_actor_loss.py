from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from verl.trainer.ppo.core_algos import get_policy_loss_fn

from tgvf_rl.framework.verl.deepeyes_actor_loss import (
    DEEPEYES_OFFICIAL_POLICY_LOSS_MODE,
    PRL24_SINGLE_PASS_POLICY_LOSS_MODE,
    compute_deepeyes_official_micro_token_mean_loss,
    compute_deepeyes_single_pass_micro_token_mean_loss,
)


def _actor_config(
    *,
    dp_size: int = 2,
    global_batch_size: int = 16,
    micro_batch_size: int = 4,
    loss_mode: str = DEEPEYES_OFFICIAL_POLICY_LOSS_MODE,
) -> SimpleNamespace:
    return SimpleNamespace(
        policy_loss=SimpleNamespace(loss_mode=loss_mode),
        use_dynamic_bsz=False,
        ppo_epochs=1,
        entropy_coeff=0.0,
        use_kl_loss=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        ppo_micro_batch_size_per_gpu=micro_batch_size,
        global_batch_info={
            "dp_size": dp_size,
            "global_batch_size": global_batch_size,
            # Pinned veRL supplies this, but the official reduction must not
            # use it as a denominator.
            "batch_num_tokens": 123456,
        },
    )


def _response_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [
            [1 if position < length else 0 for position in range(width)]
            for length in lengths
        ],
        dtype=torch.bool,
    )


def _official_per_token_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    ratio = torch.exp(log_prob - old_log_prob)
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(min=0.8, max=1.2)
    clipped = torch.maximum(losses1, losses2)
    return torch.where(
        advantages < 0,
        torch.minimum(-advantages * 3.0, clipped),
        clipped,
    )


def test_registry_selects_only_the_project_owned_deepeyes_loss() -> None:
    assert get_policy_loss_fn(DEEPEYES_OFFICIAL_POLICY_LOSS_MODE) is (
        compute_deepeyes_official_micro_token_mean_loss
    )
    assert get_policy_loss_fn(PRL24_SINGLE_PASS_POLICY_LOSS_MODE) is (
        compute_deepeyes_single_pass_micro_token_mean_loss
    )


def test_single_pass_value_and_gradient_match_same_model_old_policy() -> None:
    shape = (4, 7)
    old_placeholder = torch.linspace(
        -4.0, 3.0, shape[0] * shape[1], dtype=torch.float64
    ).reshape(shape)
    advantages = torch.linspace(
        -1.5, 1.5, shape[0] * shape[1], dtype=torch.float64
    ).reshape(shape)
    mask = _response_mask([7, 5, 3, 1], shape[1])

    official_log_prob = torch.linspace(
        -0.2, 0.2, shape[0] * shape[1], dtype=torch.float64
    ).reshape(shape)
    official_log_prob.requires_grad_(True)
    official_loss, official_metrics = compute_deepeyes_official_micro_token_mean_loss(
        official_log_prob.detach().clone(),
        official_log_prob,
        advantages,
        mask,
        config=_actor_config(),
    )
    official_gradient = torch.autograd.grad(official_loss, official_log_prob)[0]

    single_log_prob = official_log_prob.detach().clone().requires_grad_(True)
    single_loss, single_metrics = compute_deepeyes_single_pass_micro_token_mean_loss(
        old_placeholder,
        single_log_prob,
        advantages,
        mask,
        config=_actor_config(loss_mode=PRL24_SINGLE_PASS_POLICY_LOSS_MODE),
    )
    single_gradient = torch.autograd.grad(single_loss, single_log_prob)[0]

    torch.testing.assert_close(single_loss, official_loss, rtol=0, atol=0)
    torch.testing.assert_close(single_gradient, official_gradient, rtol=0, atol=0)
    assert "actor/deepeyes_single_pass_self_anchor" not in official_metrics
    assert single_metrics["actor/deepeyes_single_pass_self_anchor"] == 1.0
    assert single_metrics["actor/ppo_kl"] == 0.0
    assert single_metrics["actor/pg_clipfrac"] == 0.0


def test_value_and_gradient_match_equal_micro_token_means_not_other_means() -> None:
    # Two simulated DP ranks, two fixed four-sequence micros per rank.  Token
    # counts differ both between sequences and between micros, which makes all
    # three candidate reductions observably different.
    lengths = [1, 1, 1, 1, 6, 5, 4, 3, 1, 2, 3, 4, 6, 6, 6, 6]
    width = 6
    response_mask = _response_mask(lengths, width)
    old_log_prob = torch.zeros((16, width), dtype=torch.float64)
    log_prob = torch.linspace(-0.08, 0.08, 16 * width, dtype=torch.float64).reshape(
        16, width
    )
    log_prob.requires_grad_(True)
    advantages = (
        torch.arange(1, 16 * width + 1, dtype=torch.float64).reshape(16, width) / 17.0
    )
    advantages[::3] *= -1
    config = _actor_config()

    micro_losses = []
    for start in range(0, 16, 4):
        loss, _ = compute_deepeyes_official_micro_token_mean_loss(
            old_log_prob[start : start + 4],
            log_prob[start : start + 4],
            advantages[start : start + 4],
            response_mask[start : start + 4],
            config=config,
        )
        micro_losses.append(loss)
    # Each returned loss already contains 1 / local_micro_count.  FSDP then
    # averages the two rank-local accumulated gradients.
    actual = torch.stack(micro_losses).sum() / 2

    per_token = _official_per_token_loss(old_log_prob, log_prob, advantages)
    official_oracle = torch.stack(
        [
            per_token[start : start + 4]
            .masked_select(response_mask[start : start + 4])
            .mean()
            for start in range(0, 16, 4)
        ]
    ).mean()
    global_token_mean = per_token.masked_select(response_mask).mean()
    sequence_mean = torch.stack(
        [
            per_token[index].masked_select(response_mask[index]).mean()
            for index in range(16)
        ]
    ).mean()

    actual_gradient = torch.autograd.grad(actual, log_prob, retain_graph=True)[0]
    official_gradient = torch.autograd.grad(
        official_oracle, log_prob, retain_graph=True
    )[0]
    global_token_gradient = torch.autograd.grad(
        global_token_mean, log_prob, retain_graph=True
    )[0]
    sequence_gradient = torch.autograd.grad(sequence_mean, log_prob)[0]

    torch.testing.assert_close(actual, official_oracle, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual_gradient, official_gradient, rtol=0, atol=1e-12)
    assert not torch.isclose(actual, global_token_mean, rtol=0, atol=1e-8)
    assert not torch.isclose(actual, sequence_mean, rtol=0, atol=1e-8)
    assert not torch.allclose(actual_gradient, global_token_gradient, rtol=0, atol=1e-8)
    assert not torch.allclose(actual_gradient, sequence_gradient, rtol=0, atol=1e-8)
    assert [
        int(response_mask[start : start + 4].sum()) for start in range(0, 16, 4)
    ] == [4, 18, 10, 24]
    assert actual.detach().item() == pytest.approx(-0.8464697448242664)
    assert global_token_mean.detach().item() == pytest.approx(-0.7438512345090551)
    assert sequence_mean.detach().item() == pytest.approx(-0.7028142003424558)
    assert (actual_gradient - global_token_gradient).abs().max().item() == (
        pytest.approx(0.04747654239317267)
    )
    assert (actual_gradient - sequence_gradient).abs().max().item() == (
        pytest.approx(0.10817929530161885)
    )


def test_formal_shape_scales_each_micro_by_one_over_256() -> None:
    config = _actor_config(
        dp_size=4,
        global_batch_size=4096,
        micro_batch_size=4,
    )
    zeros = torch.zeros((4, 3), dtype=torch.float64)
    advantages = torch.ones_like(zeros)
    mask = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
        dtype=torch.bool,
    )
    loss, metrics = compute_deepeyes_official_micro_token_mean_loss(
        zeros,
        zeros,
        advantages,
        mask,
        config=config,
    )
    torch.testing.assert_close(
        loss, torch.tensor(-1 / 256, dtype=torch.float64), rtol=0, atol=0
    )
    assert metrics["actor/deepeyes_micro_policy_tokens"] == 7.0
    assert metrics["actor/deepeyes_local_micro_batches"] == 256.0


def test_bs64_ga4_value_and_gradient_are_not_multiplied_by_four() -> None:
    # PRL24-A has 1,024 trajectories globally, DP8 and trajectory micro32.
    # One rank therefore accumulates exactly four local micros.  The sum of
    # their returned losses must be their mean, not four times that mean.
    config = _actor_config(
        dp_size=8,
        global_batch_size=1024,
        micro_batch_size=32,
    )
    width = 2
    old_log_prob = torch.zeros((128, width), dtype=torch.float64)
    log_prob = torch.linspace(-0.05, 0.05, 128 * width, dtype=torch.float64).reshape(
        128, width
    )
    log_prob.requires_grad_(True)
    advantages = torch.linspace(-1.0, 1.0, 128 * width, dtype=torch.float64).reshape(
        128, width
    )
    mask = torch.ones_like(log_prob, dtype=torch.bool)

    returned = []
    raw_micro_means = []
    per_token = _official_per_token_loss(old_log_prob, log_prob, advantages)
    for start in range(0, 128, 32):
        loss, metrics = compute_deepeyes_official_micro_token_mean_loss(
            old_log_prob[start : start + 32],
            log_prob[start : start + 32],
            advantages[start : start + 32],
            mask[start : start + 32],
            config=config,
        )
        returned.append(loss)
        raw_micro_means.append(per_token[start : start + 32].mean())
        assert metrics["actor/deepeyes_local_micro_batches"] == 4.0

    actual = torch.stack(returned).sum()
    oracle = torch.stack(raw_micro_means).mean()
    multiplied_by_four = torch.stack(raw_micro_means).sum()
    actual_gradient = torch.autograd.grad(actual, log_prob, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, log_prob)[0]

    torch.testing.assert_close(actual, oracle, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual_gradient, oracle_gradient, rtol=0, atol=1e-12)
    assert not torch.isclose(actual, multiplied_by_four, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("partial_micro", "partial or differently sized"),
        ("dynamic", "forbids dynamic batching"),
        ("wrong_mode", "registry identity differs"),
        ("zero_tokens", "must contain policy tokens"),
        ("rollout_weights", "disables rollout importance weights"),
    ),
)
def test_loss_fails_closed_on_reduction_drift(mutation: str, message: str) -> None:
    config = _actor_config()
    shape = (4, 3)
    if mutation == "partial_micro":
        shape = (3, 3)
    if mutation == "dynamic":
        config.use_dynamic_bsz = True
    if mutation == "wrong_mode":
        config.policy_loss.loss_mode = "vanilla"
    zeros = torch.zeros(shape, dtype=torch.float64)
    mask = torch.ones(shape, dtype=torch.bool)
    if mutation == "zero_tokens":
        mask.zero_()
    rollout_weights = torch.ones(shape) if mutation == "rollout_weights" else None

    with pytest.raises((ValueError, TypeError), match=message):
        compute_deepeyes_official_micro_token_mean_loss(
            zeros,
            zeros,
            torch.ones_like(zeros),
            mask,
            config=config,
            rollout_is_weights=rollout_weights,
        )
