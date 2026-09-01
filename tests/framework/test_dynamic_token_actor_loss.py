from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

_verl_core_algos = pytest.importorskip(
    "verl.trainer.ppo.core_algos",
    reason="dynamic actor-loss registration requires the optional pinned veRL",
)
get_policy_loss_fn = _verl_core_algos.get_policy_loss_fn

from tgvf_rl.framework.verl.dynamic_token_actor_loss import (  # noqa: E402
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    compute_dynamic_global_token_mean_loss,
)


def _actor_config(*, dp_size: int, batch_num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy_loss=SimpleNamespace(loss_mode=DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE),
        use_dynamic_bsz=True,
        ppo_micro_batch_size_per_gpu=None,
        ppo_epochs=1,
        entropy_coeff=0.0,
        use_kl_loss=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        global_batch_info={
            "dp_size": dp_size,
            "batch_num_tokens": batch_num_tokens,
            "global_batch_size": 8,
        },
    )


def _response_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[position < length for position in range(width)] for length in lengths],
        dtype=torch.bool,
    )


def _dual_clip_per_token(
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


def test_registry_uses_distinct_dynamic_global_token_identity() -> None:
    assert get_policy_loss_fn(DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE) is (
        compute_dynamic_global_token_mean_loss
    )


def test_variable_micros_match_global_token_mean_value_and_gradient() -> None:
    lengths = [1, 5, 2, 6, 3, 1, 4, 6]
    width = 6
    mask = _response_mask(lengths, width)
    old_log_prob = torch.zeros((8, width), dtype=torch.float64)
    log_prob = torch.linspace(
        -0.1,
        0.1,
        8 * width,
        dtype=torch.float64,
    ).reshape(8, width)
    log_prob.requires_grad_(True)
    advantages = (
        torch.arange(1, 8 * width + 1, dtype=torch.float64).reshape(8, width) / 13.0
    )
    advantages[::2] *= -1
    global_tokens = int(mask.sum().item())
    config = _actor_config(dp_size=2, batch_num_tokens=global_tokens)

    # Simulate two DP ranks whose dynamic packer creates different local
    # sequence counts and token counts per micro-batch.
    rank_micro_slices = (((0, 1), (1, 4)), ((4, 6), (6, 8)))
    rank_losses: list[torch.Tensor] = []
    for rank_slices in rank_micro_slices:
        local_losses: list[torch.Tensor] = []
        for start, stop in rank_slices:
            loss, _ = compute_dynamic_global_token_mean_loss(
                old_log_prob[start:stop],
                log_prob[start:stop],
                advantages[start:stop],
                mask[start:stop],
                config=config,
            )
            local_losses.append(loss)
        rank_losses.append(torch.stack(local_losses).sum())
    # FSDP averages the accumulated rank-local gradients.
    actual = torch.stack(rank_losses).mean()

    per_token = _dual_clip_per_token(old_log_prob, log_prob, advantages)
    oracle = per_token.masked_select(mask).mean()
    equal_micro_mean = torch.stack(
        [
            per_token[start:stop].masked_select(mask[start:stop]).mean()
            for rank_slices in rank_micro_slices
            for start, stop in rank_slices
        ]
    ).mean()
    actual_gradient = torch.autograd.grad(actual, log_prob, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, log_prob, retain_graph=True)[0]
    micro_gradient = torch.autograd.grad(equal_micro_mean, log_prob)[0]

    torch.testing.assert_close(actual, oracle, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual_gradient, oracle_gradient, rtol=0, atol=1e-12)
    assert not torch.isclose(actual, equal_micro_mean, rtol=0, atol=1e-8)
    assert not torch.allclose(actual_gradient, micro_gradient, rtol=0, atol=1e-8)
    assert [
        int(mask[start:stop].sum().item())
        for rank_slices in rank_micro_slices
        for start, stop in rank_slices
    ] == [1, 13, 4, 10]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("fixed", "requires dynamic batching"),
        ("fixed_micro", "fixed micro-batch size unset"),
        ("wrong_mode", "registry identity differs"),
        ("zero_global_tokens", "batch_num_tokens must be a positive integer"),
        ("zero_micro_tokens", "must contain policy tokens"),
        ("rollout_weights", "disables rollout importance weights"),
    ),
)
def test_dynamic_loss_fails_closed_on_contract_drift(
    mutation: str,
    message: str,
) -> None:
    config = _actor_config(dp_size=2, batch_num_tokens=8)
    if mutation == "fixed":
        config.use_dynamic_bsz = False
    elif mutation == "fixed_micro":
        config.ppo_micro_batch_size_per_gpu = 2
    elif mutation == "wrong_mode":
        config.policy_loss.loss_mode = "deepeyes_official_micro_token_mean"
    elif mutation == "zero_global_tokens":
        config.global_batch_info["batch_num_tokens"] = 0
    zeros = torch.zeros((2, 3), dtype=torch.float64)
    mask = torch.ones_like(zeros, dtype=torch.bool)
    if mutation == "zero_micro_tokens":
        mask.zero_()
    rollout_weights = torch.ones_like(zeros) if mutation == "rollout_weights" else None

    with pytest.raises((TypeError, ValueError), match=message):
        compute_dynamic_global_token_mean_loss(
            zeros,
            zeros,
            torch.ones_like(zeros),
            mask,
            config=config,
            rollout_is_weights=rollout_weights,
        )
