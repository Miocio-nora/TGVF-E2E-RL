"""veRL fused selected-token log probabilities for exact replay.

The standard veRL Qwen3-VL fused forward avoids materializing a full
``[tokens, vocabulary]`` logits tensor by combining the LM-head projection and
log-softmax.  Exact TGVF replay has already constructed its own visual
``inputs_embeds`` and therefore bypasses that top-level forward.  This module
applies the same fused primitive at the actual shared boundary: injected
decoder hidden states to policy-owned next-token log probabilities.
"""

from __future__ import annotations

from typing import Any

import torch

from tgvf_rl.contracts.tokens import SamplingIdentity


def fused_selected_next_token_logprobs(
    *,
    hidden_states: torch.Tensor,
    lm_head: Any,
    token_ids: torch.Tensor,
    sampled_positions: torch.Tensor,
    sampling: SamplingIdentity,
) -> torch.Tensor:
    """Return exact policy-token log probabilities without full logits.

    Selecting hidden states before the vocabulary projection is algebraically
    equivalent to projecting every sequence position and gathering afterward:
    a linear LM head has no interaction across token positions.  It also
    preserves the same gradients for Qwen, the vision path and RP66 because
    only the selected policy-token log probabilities contribute to the loss.
    """

    if not isinstance(sampling, SamplingIdentity):
        raise TypeError("sampling identity is required for fused replay")
    if not sampling.has_identity_sampling_transforms:
        raise ValueError(
            "fused processed-logprob replay requires identity sampling transforms"
        )
    if (
        hidden_states.ndim != 3
        or token_ids.ndim != 2
        or sampled_positions.ndim != 2
    ):
        raise ValueError(
            "fused replay expects hidden [B,S,H], token IDs [B,S], and positions [B,K]"
        )
    if hidden_states.shape[:2] != token_ids.shape:
        raise ValueError("fused replay hidden states and token IDs must align")
    if sampled_positions.shape[0] != token_ids.shape[0]:
        raise ValueError("fused replay positions must share the batch dimension")
    if sampled_positions.numel() and (
        sampled_positions.min() < 1
        or sampled_positions.max() >= token_ids.shape[1]
    ):
        raise ValueError("fused replay positions must be in [1, sequence_length)")

    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("fused replay LM head must expose a floating weight tensor")
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - pinned torch always provides it
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(weight, DTensor):
        weight = weight.full_tensor().to(hidden_states.device)
    # This mirrors veRL's Qwen3-VL torch backend.  FSDP can retain an FP32
    # master LM-head while decoder activations use BF16.
    weight = weight.to(device=hidden_states.device, dtype=hidden_states.dtype)

    batch_indices = torch.arange(
        token_ids.shape[0], device=hidden_states.device
    ).unsqueeze(1)
    positions = sampled_positions.to(device=hidden_states.device, dtype=torch.long)
    predictive_hidden = hidden_states[batch_indices, positions - 1]
    selected_ids = token_ids.to(device=hidden_states.device)[batch_indices, positions]

    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    logprobs, _entropy = FusedLinearForPPO().forward(
        hidden_states=predictive_hidden,
        vocab_weights=weight,
        input_ids=selected_ids,
        temperature=1.0,
    )
    return logprobs


__all__ = ["fused_selected_next_token_logprobs"]
