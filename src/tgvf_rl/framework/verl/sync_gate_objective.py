"""Infrastructure-only objective for the veRL actor-to-vLLM sync gate.

This module is imported explicitly through veRL's public ``external_lib``
model hook.  It deliberately does not implement GRPO, PPO, or SDPO.  The
registered advantage estimator writes an all-zero sentinel and the policy loss
ignores rewards and minimizes generated-token negative log likelihood.  That
is enough to create a real, deterministic actor optimizer update without
claiming any production reinforcement-learning mathematics.

The module is not imported from :mod:`tgvf_rl.framework.verl`; importing the
normal project package therefore continues to work when veRL is absent.
"""

from __future__ import annotations

from typing import Any

import torch

from verl.trainer.ppo.core_algos import (
    agg_loss,
    register_adv_est,
    register_policy_loss,
)


ADVANTAGE_ESTIMATOR_NAME = "tgvf_sync_gate_zero"
POLICY_LOSS_NAME = "tgvf_sync_gate_nll"


@register_adv_est(ADVANTAGE_ESTIMATOR_NAME)
def zero_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact zero sentinels; rewards cannot affect this smoke update."""

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("sync-gate reward and response-mask shapes differ")
    zeros = torch.zeros_like(token_level_rewards)
    return zeros, zeros.clone()


@register_policy_loss(POLICY_LOSS_NAME)
def generated_token_nll(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config: Any = None,
    rollout_is_weights: torch.Tensor | None = None,
    **_: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create a real actor update while rejecting RL-specific transformations."""

    if config is None:
        raise ValueError("sync-gate objective requires the veRL actor config")
    if rollout_is_weights is not None:
        raise ValueError("sync-gate objective forbids rollout correction weights")
    if old_log_prob.shape != log_prob.shape or log_prob.shape != advantages.shape:
        raise ValueError("sync-gate logprob and advantage shapes differ")
    if response_mask.shape != log_prob.shape:
        raise ValueError("sync-gate response mask shape differs from logprobs")
    if not torch.isfinite(old_log_prob).all() or not torch.isfinite(log_prob).all():
        raise ValueError("sync-gate logprobs must be finite")
    if torch.count_nonzero(advantages).item() != 0:
        raise ValueError("sync-gate zero-advantage sentinel was overwritten")

    selected = response_mask.to(dtype=torch.bool)
    if not selected.any():
        raise ValueError("sync-gate objective received no generated tokens")
    nll = agg_loss(
        loss_mat=-log_prob,
        loss_mask=selected,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )
    return nll, {
        "sync_gate_nll": nll.detach(),
        "sync_gate_selected_logprob_mean": log_prob[selected].mean().detach(),
        "sync_gate_zero_advantage_valid": torch.ones(
            (), dtype=log_prob.dtype, device=log_prob.device
        ),
    }


__all__ = [
    "ADVANTAGE_ESTIMATOR_NAME",
    "POLICY_LOSS_NAME",
    "generated_token_nll",
    "zero_advantage",
]
