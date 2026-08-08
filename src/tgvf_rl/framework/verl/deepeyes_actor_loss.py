"""Official DeepEyes fixed-micro actor-loss reduction for PRL13.

The released DeepEyes runner fixes ``ppo_micro_batch_size_per_gpu=4`` and its
actor computes ``token-mean`` independently inside every local micro-batch.
It then divides every micro loss by the local gradient-accumulation count
before backward; FSDP/DDP averages those accumulated gradients across data
parallel ranks.  Consequently the optimized scalar is an equal mean of
micro-batch token means, not a token mean over the complete global batch and
not a mean of per-sequence token means.

Pinned veRL e003 deliberately changed ``token-mean`` to normalize against the
complete global mini-batch.  PRL13 selects this project-owned loss through a
distinct registry name so no other run inherits the historical reduction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODULE,
)
from .qwen3_flex_attention_compat import (
    install_qwen3_vl_text_flex_attention_compat,
)
from .torch_bert_padding import install_prl13_torch_bert_padding


# ``ModelConfig.external_lib`` imports this module independently in every
# FSDP worker.  Install the CUDA-safe Torch padding primitives at that same
# boundary as well as registering the project-owned actor loss below.
install_prl13_torch_bert_padding()
install_qwen3_vl_text_flex_attention_compat()


DEEPEYES_OFFICIAL_POLICY_LOSS_MODE = NATIVE_DEEPEYES_POLICY_LOSS_MODE
DEEPEYES_OFFICIAL_POLICY_LOSS_MODULE = NATIVE_DEEPEYES_POLICY_LOSS_MODULE
DEEPEYES_OFFICIAL_LOSS_AGG_MODE = NATIVE_DEEPEYES_LOSS_AGG_MODE


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


def _require_deepeyes_config(
    config: object,
    *,
    loss_agg_mode: str,
    observed_micro_batch_size: int,
) -> int:
    """Return the number of fixed-size local micros accumulated per update."""

    if config is None:
        raise ValueError("DeepEyes actor loss requires an actor config")
    if loss_agg_mode != DEEPEYES_OFFICIAL_LOSS_AGG_MODE:
        raise ValueError("DeepEyes actor loss requires token-mean input semantics")
    policy_loss = _value(config, "policy_loss")
    if _value(policy_loss, "loss_mode") != DEEPEYES_OFFICIAL_POLICY_LOSS_MODE:
        raise ValueError("DeepEyes actor policy-loss registry identity differs")
    if _value(config, "use_dynamic_bsz") is not False:
        raise ValueError("DeepEyes equal-micro reduction forbids dynamic batching")
    if _value(config, "ppo_epochs") != 1:
        raise ValueError("DeepEyes control requires exactly one PPO epoch")
    if _value(config, "entropy_coeff") != 0.0:
        raise ValueError("DeepEyes control requires zero entropy coefficient")
    if _value(config, "use_kl_loss") is not False:
        raise ValueError("DeepEyes control disables actor KL loss")

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
        raise ValueError(f"DeepEyes actor clipping config differs: {mismatches!r}")

    configured_micro_batch_size = _positive_int(
        _value(config, "ppo_micro_batch_size_per_gpu"),
        "ppo_micro_batch_size_per_gpu",
    )
    if configured_micro_batch_size != observed_micro_batch_size:
        raise ValueError(
            "DeepEyes actor observed a partial or differently sized micro-batch"
        )

    global_batch_info = _value(config, "global_batch_info")
    dp_size = _positive_int(_value(global_batch_info, "dp_size"), "dp_size")
    global_batch_size = _positive_int(
        _value(global_batch_info, "global_batch_size"), "global_batch_size"
    )
    if global_batch_size % dp_size:
        raise ValueError("global_batch_size must divide evenly across DP ranks")
    local_batch_size = global_batch_size // dp_size
    if local_batch_size % configured_micro_batch_size:
        raise ValueError(
            "local mini-batch must contain complete fixed-size micro-batches"
        )
    local_micro_batch_count = local_batch_size // configured_micro_batch_size
    return _positive_int(local_micro_batch_count, "local_micro_batch_count")


def _validate_inputs(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    values = (old_log_prob, log_prob, advantages, response_mask)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("DeepEyes actor loss inputs must be tensors")
    if log_prob.ndim != 2 or any(value.shape != log_prob.shape for value in values):
        raise ValueError("DeepEyes actor loss tensors must share [batch,response]")
    if any(value.device != log_prob.device for value in values):
        raise ValueError("DeepEyes actor loss tensors must share one device")
    for name, value in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating dtype")
    return response_mask.bool()


@register_policy_loss(DEEPEYES_OFFICIAL_POLICY_LOSS_MODE)
def compute_deepeyes_official_micro_token_mean_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = DEEPEYES_OFFICIAL_LOSS_AGG_MODE,
    config: object = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DeepEyes dual-clip PPO with its released reduction order.

    This function is called once per local micro-batch by pinned veRL.  The
    returned scalar is ``masked_token_mean / local_micro_batch_count``.  After
    all local backwards and FSDP's data-parallel mean, the update therefore
    equals the official equal mean over rank/micro token means.
    """

    if rollout_is_weights is not None:
        raise ValueError("DeepEyes control disables rollout importance weights")
    mask = _validate_inputs(
        old_log_prob, log_prob, advantages, response_mask
    )
    local_micro_batch_count = _require_deepeyes_config(
        config,
        loss_agg_mode=loss_agg_mode,
        observed_micro_batch_size=int(log_prob.shape[0]),
    )

    # These lines intentionally mirror DeepEyes@11d20c6 core_algos.py.  In
    # particular, the published path does not clamp the log ratio before exp.
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(min=0.8, max=1.2)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * 3.0
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    per_token_loss = torch.where(
        advantages < 0, clip_pg_losses2, clip_pg_losses1
    )

    micro_policy_token_count = mask.sum()
    loss = (
        per_token_loss.masked_select(mask).sum()
        / micro_policy_token_count
        / local_micro_batch_count
    )

    # Keep numerical fail-closed checks without introducing a series of GPU
    # synchronization points in every micro-batch.  One compact host transfer
    # validates all tensor predicates before metrics trigger their usual sync.
    checks = torch.stack(
        (
            ((response_mask == 0) | (response_mask == 1)).all(),
            torch.isfinite(old_log_prob).all(),
            torch.isfinite(log_prob).all(),
            torch.isfinite(advantages).all(),
            mask.any(),
            torch.isfinite(ratio).all(),
            torch.isfinite(loss),
        )
    ).detach().cpu().tolist()
    if not checks[0]:
        raise ValueError("response_mask must be binary")
    for name, valid in zip(
        ("old_log_prob", "log_prob", "advantages"), checks[1:4], strict=True
    ):
        if not valid:
            raise ValueError(f"{name} must be finite")
    if not checks[4]:
        raise ValueError("every DeepEyes micro-batch must contain policy tokens")
    if not checks[5]:
        raise FloatingPointError("DeepEyes actor probability ratio is non-finite")
    if not checks[6]:
        raise FloatingPointError("DeepEyes actor loss is non-finite")

    selected_log_ratio = log_ratio.masked_select(mask)
    clipped = (pg_losses2 > pg_losses1).masked_select(mask)
    lower_clipped = (
        (clip_pg_losses1 > pg_losses3) & (advantages < 0)
    ).masked_select(mask)
    metrics = {
        "actor/pg_clipfrac": clipped.to(torch.float32).mean().item(),
        "actor/ppo_kl": (-selected_log_ratio).mean().detach().item(),
        "actor/pg_clipfrac_lower": lower_clipped.to(torch.float32).mean().item(),
        "actor/deepeyes_micro_policy_tokens": float(
            micro_policy_token_count.detach().item()
        ),
        "actor/deepeyes_local_micro_batches": float(local_micro_batch_count),
    }
    return loss, metrics


__all__ = [
    "DEEPEYES_OFFICIAL_LOSS_AGG_MODE",
    "DEEPEYES_OFFICIAL_POLICY_LOSS_MODE",
    "DEEPEYES_OFFICIAL_POLICY_LOSS_MODULE",
    "compute_deepeyes_official_micro_token_mean_loss",
]
