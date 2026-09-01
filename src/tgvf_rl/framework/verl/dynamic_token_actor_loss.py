"""Variable-micro dual-clip PPO with global policy-token normalization.

Dynamic token batching changes the number and shape of local micro-batches.
It therefore cannot use the historical DeepEyes equal-micro reduction.  This
module owns a separate registry identity whose only mathematical difference
from that control is the reduction: every local micro contributes its token
sum scaled by ``dp_size / global_policy_token_count``.  Summing local micros
and then applying FSDP's data-parallel gradient mean is exactly one token mean
over the complete global mini-batch, independent of the packing layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

from .dynamic_token_loss_contract import (
    DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE,
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODULE,
)


def _value(container: object, name: str, default: object = None) -> object:
    if isinstance(container, Mapping):
        return container.get(name, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(container, name, default)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_dynamic_global_token_config(
    config: object,
    *,
    loss_agg_mode: str,
) -> tuple[int, int]:
    if config is None:
        raise ValueError("dynamic global-token actor loss requires an actor config")
    if loss_agg_mode != DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE:
        raise ValueError("dynamic global-token actor loss requires token-mean")
    policy_loss = _value(config, "policy_loss")
    if _value(policy_loss, "loss_mode") != DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE:
        raise ValueError("dynamic global-token policy-loss registry identity differs")
    if _value(config, "use_dynamic_bsz") is not True:
        raise ValueError("dynamic global-token actor loss requires dynamic batching")
    if _value(config, "ppo_micro_batch_size_per_gpu") is not None:
        raise ValueError(
            "dynamic global-token actor loss requires fixed micro-batch size unset"
        )
    if _value(config, "ppo_epochs") != 1:
        raise ValueError("dynamic global-token control requires exactly one PPO epoch")
    if _value(config, "entropy_coeff") != 0.0:
        raise ValueError(
            "dynamic global-token control requires zero entropy coefficient"
        )
    if _value(config, "use_kl_loss") is not False:
        raise ValueError("dynamic global-token control disables actor KL loss")

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
    if mismatches:
        raise ValueError(
            f"dynamic global-token actor clipping config differs: {mismatches!r}"
        )

    global_batch_info = _value(config, "global_batch_info")
    dp_size = _positive_int(_value(global_batch_info, "dp_size"), "dp_size")
    batch_num_tokens = _positive_int(
        _value(global_batch_info, "batch_num_tokens"),
        "batch_num_tokens",
    )
    global_batch_size = _value(global_batch_info, "global_batch_size")
    if global_batch_size is not None:
        _positive_int(global_batch_size, "global_batch_size")
    return dp_size, batch_num_tokens


def _validate_inputs(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    values = (old_log_prob, log_prob, advantages, response_mask)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("dynamic global-token actor loss inputs must be tensors")
    if log_prob.ndim != 2 or any(value.shape != log_prob.shape for value in values):
        raise ValueError(
            "dynamic global-token actor loss tensors must share [batch,response]"
        )
    if any(value.device != log_prob.device for value in values):
        raise ValueError(
            "dynamic global-token actor loss tensors must share one device"
        )
    for name, value in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating dtype")
    return response_mask.bool()


@register_policy_loss(DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE)
def compute_dynamic_global_token_mean_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE,
    config: object = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DeepEyes-style dual-clip PPO over variable-sized micros."""

    if rollout_is_weights is not None:
        raise ValueError(
            "dynamic global-token control disables rollout importance weights"
        )
    mask = _validate_inputs(old_log_prob, log_prob, advantages, response_mask)
    dp_size, batch_num_tokens = _require_dynamic_global_token_config(
        config,
        loss_agg_mode=loss_agg_mode,
    )

    # Preserve the accepted DeepEyes dual-clip token math. In particular this
    # path does not add veRL vanilla's separate [-20, 20] log-ratio clamp.
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(min=0.8, max=1.2)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * 3.0
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    per_token_loss = torch.where(
        advantages < 0,
        clip_pg_losses2,
        clip_pg_losses1,
    )

    micro_policy_token_count = mask.sum()
    loss = per_token_loss.masked_select(mask).sum() / batch_num_tokens * dp_size

    checks = (
        torch.stack(
            (
                ((response_mask == 0) | (response_mask == 1)).all(),
                torch.isfinite(old_log_prob).all(),
                torch.isfinite(log_prob).all(),
                torch.isfinite(advantages).all(),
                mask.any(),
                torch.isfinite(ratio).all(),
                torch.isfinite(loss),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    if not checks[0]:
        raise ValueError("response_mask must be binary")
    for name, valid in zip(
        ("old_log_prob", "log_prob", "advantages"),
        checks[1:4],
        strict=True,
    ):
        if not valid:
            raise ValueError(f"{name} must be finite")
    if not checks[4]:
        raise ValueError("every dynamic micro-batch must contain policy tokens")
    if not checks[5]:
        raise FloatingPointError("dynamic actor probability ratio is non-finite")
    if not checks[6]:
        raise FloatingPointError("dynamic global-token actor loss is non-finite")

    selected_log_ratio = log_ratio.masked_select(mask)
    clipped = (pg_losses2 > pg_losses1).masked_select(mask)
    lower_clipped = ((clip_pg_losses1 > pg_losses3) & (advantages < 0)).masked_select(
        mask
    )
    metrics = {
        "actor/pg_clipfrac": clipped.to(torch.float32).mean().item(),
        "actor/ppo_kl": (-selected_log_ratio).mean().detach().item(),
        "actor/pg_clipfrac_lower": lower_clipped.to(torch.float32).mean().item(),
        "actor/dynamic_micro_policy_tokens": float(
            micro_policy_token_count.detach().item()
        ),
        "actor/dynamic_global_policy_tokens": float(batch_num_tokens),
    }
    return loss, metrics


__all__ = [
    "DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE",
    "DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE",
    "DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODULE",
    "compute_dynamic_global_token_mean_loss",
]
