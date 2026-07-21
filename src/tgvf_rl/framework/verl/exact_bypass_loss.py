"""Exact Policy Pilot v1 loss registered into veRL's bypass dataflow.

This module is imported inside every actor worker through veRL's public
``actor_rollout_ref.model.external_lib`` hook.  Importing it intentionally
replaces the pinned e003 registry entry named ``bypass_mode`` while leaving
the trainer-owned ``rollout_log_probs -> old_log_probs`` dataflow unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

from .exact_replay_engine import register_qwen3_exact_replay_fsdp2_engine


EXACT_BYPASS_LOSS_REGISTRY_NAME = "bypass_mode"
POLICY_PILOT_V1_EXACT_BYPASS_MODULE = (
    "tgvf_rl.framework.verl.exact_bypass_loss"
)


def _value(container: object, name: str, default: object = None) -> object:
    if isinstance(container, Mapping):
        return container.get(name, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(container, name, default)


def _require_pilot_config(config: object, loss_agg_mode: str) -> tuple[int, int]:
    if config is None:
        raise ValueError("Policy Pilot exact bypass loss requires actor config")
    if loss_agg_mode != "token-mean":
        raise ValueError("Policy Pilot exact bypass loss requires token-mean")

    expected_actor = {
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0,
    }
    mismatches = {
        name: (_value(config, name), expected)
        for name, expected in expected_actor.items()
        if _value(config, name) != expected
    }
    policy_loss = _value(config, "policy_loss")
    rollout_correction = _value(policy_loss, "rollout_correction")
    expected_correction = {
        "bypass_mode": True,
        "loss_type": "ppo_clip",
        "rollout_is": None,
        "rollout_rs": None,
        "rollout_is_batch_normalize": False,
    }
    mismatches.update(
        {
            f"rollout_correction.{name}": (
                _value(rollout_correction, name),
                expected,
            )
            for name, expected in expected_correction.items()
            if _value(rollout_correction, name) != expected
        }
    )
    if mismatches:
        raise ValueError(
            f"Policy Pilot exact bypass loss config differs: {mismatches!r}"
        )

    global_batch_info = _value(config, "global_batch_info")
    dp_size = _value(global_batch_info, "dp_size")
    batch_num_tokens = _value(global_batch_info, "batch_num_tokens")
    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("global_batch_info.dp_size must be a positive integer")
    if type(batch_num_tokens) is not int or batch_num_tokens <= 0:
        raise ValueError(
            "global_batch_info.batch_num_tokens must be a positive integer"
        )
    return dp_size, batch_num_tokens


def _validate_tensors(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    values = (old_log_prob, log_prob, advantages, response_mask)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Policy Pilot exact bypass loss inputs must be tensors")
    shape = log_prob.shape
    if log_prob.ndim != 2 or any(value.shape != shape for value in values):
        raise ValueError("exact bypass loss tensors must share [batch,response]")
    if any(value.device != log_prob.device for value in values):
        raise ValueError("exact bypass loss tensors must share one device")
    for name, value in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating dtype")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise ValueError(f"{name} must be finite")
    if not bool(((response_mask == 0) | (response_mask == 1)).all().item()):
        raise ValueError("response_mask must be binary")
    mask = response_mask.bool()
    if not bool(mask.any().item()):
        raise ValueError("response_mask must select at least one policy token")
    return mask


@register_policy_loss(EXACT_BYPASS_LOSS_REGISTRY_NAME)
def compute_policy_pilot_v1_exact_bypass_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: object = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the accepted unclamped behavior-ratio dual-clip objective."""

    if rollout_is_weights is not None:
        raise ValueError("Policy Pilot disables rollout importance weights")
    dp_size, global_policy_token_count = _require_pilot_config(
        config, loss_agg_mode
    )
    mask = _validate_tensors(
        old_log_prob, log_prob, advantages, response_mask
    )

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=0.8, max=1.2)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    surrogate = torch.where(
        advantages < 0,
        torch.maximum(surrogate, 3.0 * advantages),
        surrogate,
    )
    per_token_loss = torch.where(mask, -surrogate, torch.zeros_like(surrogate))
    if not bool(torch.isfinite(per_token_loss.detach()).all().item()):
        raise FloatingPointError(
            "Policy Pilot exact bypass loss produced non-finite token values"
        )
    loss = (
        per_token_loss.sum()
        / global_policy_token_count
        * dp_size
    )

    selected_log_ratio = log_ratio[mask]
    selected_log_ratio_fp32 = selected_log_ratio.to(torch.float32)
    selected_log_ratio_abs = selected_log_ratio_fp32.abs()
    selected_ratio = ratio[mask].to(torch.float32)
    raw_negative_surrogate = -advantages * ratio
    clipped_negative_surrogate = -advantages * clipped_ratio
    selected_negative_surrogate = torch.maximum(
        raw_negative_surrogate, clipped_negative_surrogate
    )
    dual_clip_bound = -advantages * 3.0
    clipped = clipped_negative_surrogate > raw_negative_surrogate
    lower_clipped = (selected_negative_surrogate > dual_clip_bound) & (
        advantages < 0
    )
    metrics = {
        "actor/pg_clipfrac": clipped[mask].to(torch.float32).mean().item(),
        "actor/ppo_kl": (-selected_log_ratio).mean().detach().item(),
        "actor/pg_clipfrac_lower": lower_clipped[mask]
        .to(torch.float32)
        .mean()
        .item(),
        "actor/behavior_current_log_ratio_abs_mean": selected_log_ratio_abs
        .mean()
        .item(),
        "actor/behavior_current_log_ratio_abs_p99": torch.quantile(
            selected_log_ratio_abs, 0.99
        ).item(),
        "actor/behavior_current_log_ratio_abs_max": selected_log_ratio_abs
        .max()
        .item(),
        "actor/behavior_current_ratio_outside_clip_fraction": (
            (selected_ratio < 0.8) | (selected_ratio > 1.2)
        )
        .to(torch.float32)
        .mean()
        .item(),
    }
    if not math.isfinite(float(loss.detach().item())):
        raise FloatingPointError("Policy Pilot exact bypass loss is non-finite")
    return loss, metrics


# ``HFModelConfig.__post_init__`` imports this module before TrainingWorker
# calls EngineRegistry.new().  Registering here makes the custom model type
# reachable in every Ray worker through the existing external_lib hook.
QWEN3_EXACT_REPLAY_ENGINE_CLASS = register_qwen3_exact_replay_fsdp2_engine()


__all__ = [
    "EXACT_BYPASS_LOSS_REGISTRY_NAME",
    "POLICY_PILOT_V1_EXACT_BYPASS_MODULE",
    "QWEN3_EXACT_REPLAY_ENGINE_CLASS",
    "compute_policy_pilot_v1_exact_bypass_loss",
]
