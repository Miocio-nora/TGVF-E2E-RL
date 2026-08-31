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


class FusedExactReplayMicrobatchMaterializer:
    """Reuse one differentiable full LM head within one replay microbatch.

    Exact replay evaluates each trajectory row separately and performs one
    backward only after every row has contributed to the microbatch loss.  A
    stateless DTensor ``full_tensor()`` per row therefore leaves one full
    vocabulary-weight copy in every row's autograd graph.  This callable is
    deliberately created once by the replay port factory and is not shared
    across microbatches or optimizer steps.
    """

    def __init__(self) -> None:
        self._lm_head: Any | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._weight: torch.Tensor | None = None

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head: Any,
        token_ids: torch.Tensor,
        sampled_positions: torch.Tensor,
        sampling: SamplingIdentity,
    ) -> torch.Tensor:
        _validate_fused_replay_inputs(
            hidden_states=hidden_states,
            token_ids=token_ids,
            sampled_positions=sampled_positions,
            sampling=sampling,
        )
        device = hidden_states.device
        dtype = hidden_states.dtype
        if self._weight is None:
            self._lm_head = lm_head
            self._device = device
            self._dtype = dtype
            self._weight = _materialize_lm_head_weight(
                lm_head=lm_head,
                device=device,
                dtype=dtype,
            )
        elif (
            lm_head is not self._lm_head
            or device != self._device
            or dtype != self._dtype
        ):
            raise RuntimeError(
                "fused replay materializer cannot cross an LM head, device, "
                "or dtype boundary"
            )
        return _fused_selected_next_token_logprobs_with_weight(
            hidden_states=hidden_states,
            weight=self._weight,
            token_ids=token_ids,
            sampled_positions=sampled_positions,
        )


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

    _validate_fused_replay_inputs(
        hidden_states=hidden_states,
        token_ids=token_ids,
        sampled_positions=sampled_positions,
        sampling=sampling,
    )
    weight = _materialize_lm_head_weight(
        lm_head=lm_head,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return _fused_selected_next_token_logprobs_with_weight(
        hidden_states=hidden_states,
        weight=weight,
        token_ids=token_ids,
        sampled_positions=sampled_positions,
    )


def _validate_fused_replay_inputs(
    *,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    sampled_positions: torch.Tensor,
    sampling: SamplingIdentity,
) -> None:
    if not isinstance(sampling, SamplingIdentity):
        raise TypeError("sampling identity is required for fused replay")
    if not sampling.has_identity_sampling_transforms:
        raise ValueError(
            "fused processed-logprob replay requires identity sampling transforms"
        )
    if hidden_states.ndim != 3 or token_ids.ndim != 2 or sampled_positions.ndim != 2:
        raise ValueError(
            "fused replay expects hidden [B,S,H], token IDs [B,S], and positions [B,K]"
        )
    if hidden_states.shape[:2] != token_ids.shape:
        raise ValueError("fused replay hidden states and token IDs must align")
    if sampled_positions.shape[0] != token_ids.shape[0]:
        raise ValueError("fused replay positions must share the batch dimension")
    if sampled_positions.numel() and (
        sampled_positions.min() < 1 or sampled_positions.max() >= token_ids.shape[1]
    ):
        raise ValueError("fused replay positions must be in [1, sequence_length)")


def _materialize_lm_head_weight(
    *, lm_head: Any, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("fused replay LM head must expose a floating weight tensor")
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - pinned torch always provides it
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(weight, DTensor):
        # Keep the existing actor gradient contract: materialize the FP32
        # DTensor first, then cast the replicated tensor to BF16.  Casting the
        # shard first would make the reduce-scatter gradient BF16 instead of
        # the run's configured FP32 reduce dtype.
        weight = weight.full_tensor().to(device)
    # This mirrors veRL's Qwen3-VL torch backend.  FSDP can retain an FP32
    # master LM-head while decoder activations use BF16.
    return weight.to(device=device, dtype=dtype)


def _fused_selected_next_token_logprobs_with_weight(
    *,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    sampled_positions: torch.Tensor,
) -> torch.Tensor:
    if weight.device != hidden_states.device or weight.dtype != hidden_states.dtype:
        raise RuntimeError("materialized LM head differs from hidden-state placement")

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


__all__ = [
    "FusedExactReplayMicrobatchMaterializer",
    "fused_selected_next_token_logprobs",
]
