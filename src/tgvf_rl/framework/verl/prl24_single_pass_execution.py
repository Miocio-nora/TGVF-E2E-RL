"""Strict, opt-in execution optimizations for PRL24 native-Crop training.

The PRL24-D batch is one PPO mini-batch and one PPO epoch.  The actor therefore
does not update between the old-policy forward and the policy-loss forward.
For a dropout-free Qwen3-VL model, the old-policy tensor is exactly the current
log-probability tensor treated as a constant.  This module provides the small
runtime patches needed to exploit that identity without changing older PRL13
runs or the PRL24 reward/sample contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import torch


PRL24_SINGLE_PASS_ENV = "TGVF_PRL24_SINGLE_PASS_EXECUTION"
PRL24_CROP_AWARE_BATCHING_ENV = "TGVF_PRL24_CROP_AWARE_BATCHING"
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


@lru_cache(maxsize=8)
def _qwen3_training_cost_coefficients(
    config_path: str,
) -> tuple[int, int, int, int]:
    """Return exact FLOP-model coefficients used for deterministic scheduling."""

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    text = payload["text_config"]
    vision = payload["vision_config"]

    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    layers = int(text["num_hidden_layers"])
    head_dim = int(text.get("head_dim", hidden // heads))
    intermediate = int(text["intermediate_size"])
    vocab = int(text["vocab_size"])
    q_size = heads * head_dim
    k_size = kv_heads * head_dim
    v_size = kv_heads * head_dim
    text_dense_parameters = (
        (hidden * intermediate * 3) + hidden * (q_size + k_size + v_size + q_size)
    ) * layers + vocab * hidden * 2

    vision_hidden = int(vision["hidden_size"])
    vision_heads = int(vision["num_heads"])
    vision_depth = int(vision["depth"])
    vision_head_dim = vision_hidden // vision_heads
    merge = int(vision["spatial_merge_size"])
    patch_parameters = (
        vision_hidden
        * int(vision["in_channels"])
        * int(vision["temporal_patch_size"])
        * int(vision["patch_size"])
        * int(vision["patch_size"])
    )
    vision_mlp = vision_hidden * int(vision["intermediate_size"]) * 2
    vision_attention = vision_hidden * (4 * vision_hidden)
    merger = (int(vision["out_hidden_size"]) + vision_hidden * merge**2) * (
        vision_hidden * merge**2
    )
    deepstack_count = len(vision.get("deepstack_visual_indexes", []))
    vision_dense_parameters = (
        patch_parameters
        + (vision_mlp + vision_attention) * vision_depth
        + merger * (deepstack_count + 1)
    )
    return (
        6 * text_dense_parameters,
        6 * head_dim * heads * layers,
        6 * vision_dense_parameters,
        12 * vision_head_dim * vision_heads * vision_depth,
    )


def _image_sequence_lengths(multi_modal_input: object) -> tuple[int, ...]:
    if not isinstance(multi_modal_input, Mapping):
        return (16,)
    values = multi_modal_input.get("images_seqlens")
    if isinstance(values, torch.Tensor):
        result = tuple(int(value) for value in values.reshape(-1).tolist())
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        result = tuple(int(value) for value in values)
    else:
        result = ()
    # Qwen3-VL deliberately executes one 4x4 dummy image for text-only rows so
    # every FSDP rank touches the vision parameters.
    return result or (16,)


def estimate_qwen3_training_costs(
    sequence_lengths: Sequence[int],
    multi_modal_inputs: Sequence[object],
    *,
    model_config_path: str | Path,
) -> list[int]:
    """Estimate text+vision train FLOPs per sample using veRL's own formula."""

    if len(sequence_lengths) != len(multi_modal_inputs):
        raise ValueError("sequence lengths and multimodal inputs must align")
    coefficients = _qwen3_training_cost_coefficients(
        str(Path(model_config_path).resolve(strict=True))
    )
    text_dense, text_attention, vision_dense, vision_attention = coefficients
    costs = []
    for raw_length, multi_modal_input in zip(
        sequence_lengths, multi_modal_inputs, strict=True
    ):
        length = int(raw_length)
        if length <= 0:
            raise ValueError("training sequence lengths must be positive")
        image_lengths = _image_sequence_lengths(multi_modal_input)
        cost = text_dense * length + text_attention * length * length
        cost += vision_dense * sum(image_lengths)
        cost += vision_attention * sum(value * value for value in image_lengths)
        costs.append(cost)
    return costs


def crop_aware_micro_block_schedule(
    sample_costs: Sequence[int],
    *,
    dp_size: int,
    micro_batch_size: int,
) -> tuple[list[int], dict[str, float]]:
    """Schedule intact loss micros to reduce synchronous visual stragglers.

    The current order is first split into the exact contiguous actor micros that
    veRL would train.  Only whole micros move between ranks/waves, so the equal
    mean over fixed micro token-means—and therefore the optimized scalar—is
    unchanged.
    """

    if dp_size <= 0 or micro_batch_size <= 0:
        raise ValueError("DP and micro-batch sizes must be positive")
    sample_count = len(sample_costs)
    block_width = dp_size * micro_batch_size
    if sample_count == 0 or sample_count % block_width:
        raise ValueError("sample count must contain complete DP micro waves")
    local_samples = sample_count // dp_size
    if local_samples % micro_batch_size:
        raise ValueError("local sample count must contain complete micros")
    waves = local_samples // micro_batch_size
    block_count = dp_size * waves
    blocks = [
        list(range(index * micro_batch_size, (index + 1) * micro_batch_size))
        for index in range(block_count)
    ]
    block_costs = [sum(int(sample_costs[index]) for index in block) for block in blocks]

    old_wave_maxima = [
        max(block_costs[rank * waves + wave] for rank in range(dp_size))
        for wave in range(waves)
    ]
    sorted_blocks = sorted(
        range(block_count), key=lambda index: (-block_costs[index], index)
    )
    rank_blocks: list[list[int]] = [[] for _ in range(dp_size)]
    rank_totals = [0] * dp_size
    new_wave_maxima = []
    for wave in range(waves):
        wave_blocks = sorted_blocks[wave * dp_size : (wave + 1) * dp_size]
        new_wave_maxima.append(max(block_costs[index] for index in wave_blocks))
        rank_order = sorted(range(dp_size), key=lambda rank: (rank_totals[rank], rank))
        for rank, block_index in zip(rank_order, wave_blocks, strict=True):
            rank_blocks[rank].append(block_index)
            rank_totals[rank] += block_costs[block_index]

    global_indices = [
        sample_index
        for rank in range(dp_size)
        for block_index in rank_blocks[rank]
        for sample_index in blocks[block_index]
    ]
    old_critical = sum(old_wave_maxima)
    new_critical = sum(new_wave_maxima)
    return global_indices, {
        "actor/crop_aware_block_schedule_old_critical_cost": float(old_critical),
        "actor/crop_aware_block_schedule_new_critical_cost": float(new_critical),
        "actor/crop_aware_block_schedule_ratio": float(new_critical / old_critical),
        "actor/crop_aware_block_count": float(block_count),
    }


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


def install_prl24_crop_aware_batching() -> bool:
    """Patch the controller balancer with loss-micro-preserving scheduling."""

    from verl.trainer.ppo import ray_trainer

    trainer_class = ray_trainer.RayPPOTrainer
    marker = _PATCH_MARKER + "_crop_aware"
    if getattr(trainer_class, marker, False):
        return True
    original = trainer_class._balance_batch

    def _balance_batch(
        self: object,
        batch: object,
        metrics: dict[str, object],
        logging_prefix: str = "global_seqlen",
        keep_minibatch: bool = False,
    ) -> None:
        original(self, batch, metrics, logging_prefix, keep_minibatch)
        actor = self.config.actor_rollout_ref.actor
        loss_mode = actor.policy_loss.get("loss_mode")
        if loss_mode != PRL24_SINGLE_PASS_POLICY_LOSS_MODE:
            return
        attention_mask = batch.batch["attention_mask"]
        sequence_lengths = (
            attention_mask.view(attention_mask.shape[0], -1).sum(-1).tolist()
        )
        multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs")
        if multi_modal_inputs is None or len(multi_modal_inputs) != len(
            sequence_lengths
        ):
            raise ValueError(
                "PRL24 Crop-aware batching requires aligned multimodal inputs"
            )
        model_config_path = (
            Path(str(self.config.actor_rollout_ref.model.path)) / "config.json"
        )
        sample_costs = estimate_qwen3_training_costs(
            sequence_lengths,
            list(multi_modal_inputs),
            model_config_path=model_config_path,
        )
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        global_indices, schedule_metrics = crop_aware_micro_block_schedule(
            sample_costs,
            dp_size=dp_size,
            micro_batch_size=int(actor.ppo_micro_batch_size_per_gpu),
        )
        batch.reorder(torch.tensor(global_indices, dtype=torch.long))
        metrics.update(schedule_metrics)

    trainer_class._balance_batch = _balance_batch
    setattr(trainer_class, marker, True)
    return True


__all__ = [
    "PRL24_SINGLE_PASS_ENV",
    "PRL24_CROP_AWARE_BATCHING_ENV",
    "PRL24_SINGLE_PASS_POLICY_LOSS_MODE",
    "cached_fast_pos_embed_interpolate",
    "crop_aware_micro_block_schedule",
    "estimate_qwen3_training_costs",
    "install_prl24_crop_aware_batching",
    "install_prl24_single_pass_model_optimizations",
    "install_prl24_single_pass_rollout_bypass",
    "maybe_install_prl24_single_pass_model_optimizations",
    "target_only_linear_for_ppo",
]
