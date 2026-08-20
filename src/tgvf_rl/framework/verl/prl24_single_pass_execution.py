"""Strict, opt-in execution optimizations for PRL24 native-Crop training.

The PRL24-D batch is one PPO mini-batch and one PPO epoch.  The actor therefore
does not update between the old-policy forward and the policy-loss forward.
For a dropout-free Qwen3-VL model, the old-policy tensor is exactly the current
log-probability tensor treated as a constant.  This module provides the small
runtime patches needed to exploit that identity without changing older PRL13
runs or the PRL24 reward/sample contract.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import torch


PRL24_SINGLE_PASS_ENV = "TGVF_PRL24_SINGLE_PASS_EXECUTION"
PRL24_SINGLE_PASS_POLICY_LOSS_MODE = "deepeyes_single_pass_micro_token_mean"

_PATCH_MARKER = "_tgvf_prl24_single_pass_v1"
_ORIGINAL_QWEN3_TORCH_FORWARD: object | None = None


@lru_cache(maxsize=256)
def _cpu_bilinear_coefficients(
    num_grid_per_side: int,
    height: int,
    width: int,
    weight_dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Build the upstream bilinear coefficients once per image-grid shape."""

    if min(num_grid_per_side, height, width) <= 0:
        raise ValueError("Qwen3-VL position grids must be positive")
    h_idxs = torch.linspace(0, num_grid_per_side - 1, height)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, width)

    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
    w_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    base_h = h_floor * num_grid_per_side
    base_h_ceil = h_ceil * num_grid_per_side

    indices = (
        (base_h[None].T + w_floor[None]).flatten(),
        (base_h[None].T + w_ceil[None]).flatten(),
        (base_h_ceil[None].T + w_floor[None]).flatten(),
        (base_h_ceil[None].T + w_ceil[None]).flatten(),
    )
    weights = (
        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        ((1 - dh)[None].T * dw[None]).flatten(),
        (dh[None].T * (1 - dw)[None]).flatten(),
        (dh[None].T * dw[None]).flatten(),
    )
    return (
        tuple(value.to(dtype=torch.long) for value in indices),
        tuple(value.to(dtype=weight_dtype) for value in weights),
    )


def cached_fast_pos_embed_interpolate(self: object, grid_thw: torch.Tensor):
    """Bitwise-equivalent Qwen3-VL interpolation without large Python lists."""

    if not isinstance(grid_thw, torch.Tensor) or grid_thw.ndim != 2:
        raise ValueError("grid_thw must be a rank-2 tensor")
    grid = grid_thw.tolist()
    grid_ts = [row[0] for row in grid]
    grid_hs = [row[1] for row in grid]
    grid_ws = [row[2] for row in grid]
    device = grid_thw.device
    side = int(self.num_grid_per_side)
    weight_dtype = self.pos_embed.weight.dtype

    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for _t, height, width in grid:
        indices, weights = _cpu_bilinear_coefficients(
            side, int(height), int(width), weight_dtype
        )
        for corner in range(4):
            index_parts[corner].append(indices[corner])
            weight_parts[corner].append(weights[corner])

    idx_tensor = torch.stack([torch.cat(parts) for parts in index_parts]).to(
        device=device
    )
    weight_tensor = torch.stack([torch.cat(parts) for parts in weight_parts]).to(
        device=device
    )
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    # Preserve upstream's sequential addition order; ``sum(dim=0)`` is not
    # bitwise equivalent in BF16.
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split(
        [height * width for height, width in zip(grid_hs, grid_ws, strict=False)]
    )

    merge_size = int(self.config.spatial_merge_size)
    permuted = []
    for pos_embed, temporal, height, width in zip(
        patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False
    ):
        pos_embed = pos_embed.repeat(temporal, 1)
        pos_embed = (
            pos_embed.view(
                temporal,
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


def _target_log_probs_forward(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = (hidden_states @ vocab_weights.t()) / temperature
    logits = logits.to(torch.float32)
    return logits.log_softmax(dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)


class _TargetOnlyLinearForPPOFunction(torch.autograd.Function):
    """The pinned Torch PPO projection with its unused entropy branch removed."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        original_ndim = hidden_states.ndim
        if original_ndim not in (2, 3):
            raise ValueError("hidden_states must have rank 2 or 3")
        original_batch_size = -1
        if original_ndim == 3:
            if input_ids.ndim != 2:
                raise ValueError("input_ids shape does not match hidden_states")
            original_batch_size = hidden_states.shape[0]
            requires_grad = hidden_states.requires_grad
            hidden_states = hidden_states.flatten(0, 1)
            hidden_states.requires_grad_(requires_grad)
            input_ids = input_ids.flatten(0, 1)

        token_count = hidden_states.shape[0]
        output_requires_grad = (
            hidden_states.requires_grad or vocab_weights.requires_grad
        )
        log_probs = torch.zeros(
            token_count,
            device=hidden_states.device,
            dtype=torch.float32,
            requires_grad=output_requires_grad,
        )
        for start in range(0, token_count, chunk_size):
            stop = min(start + chunk_size, token_count)
            log_probs[start:stop] = _target_log_probs_forward(
                hidden_states[start:stop],
                vocab_weights,
                input_ids[start:stop],
                temperature,
            )

        if original_ndim == 3:
            log_probs = log_probs.view(original_batch_size, -1)
        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.original_batch_size = original_batch_size
        ctx.original_ndim = original_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        return log_probs

    @staticmethod
    def backward(ctx: Any, dlog_probs: torch.Tensor):
        from verl.utils.experimental.torch_functional import (
            _fused_linear_for_ppo_bwd,
        )

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        if ctx.original_ndim == 3:
            dlog_probs = dlog_probs.flatten()
        token_count = hidden_states.shape[0]
        dhidden = (
            torch.zeros_like(hidden_states) if hidden_states.requires_grad else None
        )
        dvocab = (
            torch.zeros_like(vocab_weights) if vocab_weights.requires_grad else None
        )
        for start in range(0, token_count, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, token_count)
            chunk_hidden, chunk_vocab = _fused_linear_for_ppo_bwd(
                dlog_probs=dlog_probs[start:stop],
                dentropy=None,
                hidden_states=hidden_states[start:stop],
                vocab_weights=vocab_weights,
                input_ids=input_ids[start:stop],
                temperature=ctx.temperature,
            )
            if dhidden is not None:
                dhidden[start:stop] += chunk_hidden
            if dvocab is not None:
                dvocab += chunk_vocab
        if ctx.original_ndim == 3 and dhidden is not None:
            dhidden = dhidden.view(ctx.original_batch_size, -1, hidden_states.shape[-1])
        return dhidden, dvocab, None, None, None


def target_only_linear_for_ppo(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    return _TargetOnlyLinearForPPOFunction.apply(
        hidden_states,
        vocab_weights,
        input_ids.to(torch.int64),
        temperature,
        chunk_size,
    )


def _single_pass_qwen3_torch_forward(
    self: object,
    input_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
    shift_labels: torch.Tensor | None = None,
    **kwargs: Any,
):
    """Use target-only projection in actor train mode; preserve eval behavior."""

    if not self.training:
        if not callable(_ORIGINAL_QWEN3_TORCH_FORWARD):
            raise RuntimeError("original Qwen3-VL Torch forward is unavailable")
        return _ORIGINAL_QWEN3_TORCH_FORWARD(
            self,
            input_ids=input_ids,
            labels=labels,
            temperature=temperature,
            shift_labels=shift_labels,
            **kwargs,
        )

    from torch.distributed.tensor import DTensor
    from verl.models.transformers.qwen3_vl import Qwen3VLCausalLMOutputForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("Qwen3-VL Torch forward requires labels or input_ids")

    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor().to(hidden_states.device)
    vocab_weights = vocab_weights.to(dtype=hidden_states.dtype)
    log_probs = target_only_linear_for_ppo(
        hidden_states,
        vocab_weights,
        rolled_labels,
        temperature,
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=None,
        hidden_states=outputs.hidden_states,
    )


def install_prl24_single_pass_model_optimizations() -> dict[str, bool]:
    """Patch only the veRL symbols consumed during later Qwen3 model setup."""

    global _ORIGINAL_QWEN3_TORCH_FORWARD
    from verl.models.transformers import qwen3_vl

    if getattr(qwen3_vl, _PATCH_MARKER, False):
        return {"cached_vision_positions": True, "target_only_lm_head": True}
    _ORIGINAL_QWEN3_TORCH_FORWARD = qwen3_vl.forward_with_torch_backend
    qwen3_vl.fast_pos_embed_interpolate = cached_fast_pos_embed_interpolate
    qwen3_vl.forward_with_torch_backend = _single_pass_qwen3_torch_forward
    setattr(qwen3_vl, _PATCH_MARKER, True)
    return {"cached_vision_positions": True, "target_only_lm_head": True}


def maybe_install_prl24_single_pass_model_optimizations() -> dict[str, bool]:
    if os.environ.get(PRL24_SINGLE_PASS_ENV) != "1":
        return {"cached_vision_positions": False, "target_only_lm_head": False}
    return install_prl24_single_pass_model_optimizations()


def install_prl24_single_pass_rollout_bypass() -> bool:
    """Skip old-logprob recomputation while retaining the project loss mode."""

    from omegaconf import open_dict
    from verl.trainer.ppo import rollout_corr_helper

    if getattr(rollout_corr_helper, _PATCH_MARKER, False):
        return True
    original = rollout_corr_helper.apply_bypass_mode

    def apply_bypass_mode(
        batch: object,
        rollout_corr_config: object = None,
        policy_loss_config: object = None,
    ) -> None:
        getter = getattr(policy_loss_config, "get", None)
        mode = (
            getter("loss_mode", None)
            if callable(getter)
            else getattr(policy_loss_config, "loss_mode", None)
        )
        if mode != PRL24_SINGLE_PASS_POLICY_LOSS_MODE:
            original(batch, rollout_corr_config, policy_loss_config)
            return
        correction_get = getattr(rollout_corr_config, "get", None)
        if not callable(correction_get) or not correction_get("bypass_mode", False):
            raise ValueError("PRL24 single-pass requires rollout bypass mode")
        if correction_get("rollout_is", None) is not None:
            raise ValueError("PRL24 single-pass forbids rollout IS")
        if correction_get("rollout_rs", None) is not None:
            raise ValueError("PRL24 single-pass forbids rollout rejection")
        if "rollout_log_probs" not in batch.batch:
            raise ValueError("PRL24 single-pass requires rollout_log_probs")
        # Shape-compatible placeholder only.  The registered loss validates it
        # but anchors the ratio to ``log_prob.detach()``.
        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        with open_dict(policy_loss_config):
            policy_loss_config["rollout_correction"] = rollout_corr_config

    rollout_corr_helper.apply_bypass_mode = apply_bypass_mode
    setattr(rollout_corr_helper, _PATCH_MARKER, True)
    return True


__all__ = [
    "PRL24_SINGLE_PASS_ENV",
    "PRL24_SINGLE_PASS_POLICY_LOSS_MODE",
    "cached_fast_pos_embed_interpolate",
    "install_prl24_single_pass_model_optimizations",
    "install_prl24_single_pass_rollout_bypass",
    "maybe_install_prl24_single_pass_model_optimizations",
    "target_only_linear_for_ppo",
]
