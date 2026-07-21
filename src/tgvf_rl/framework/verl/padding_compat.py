"""Torch-only padding primitives for the accepted veRL SDPA runtime.

Pinned veRL routes padding removal through ``flash_attn.bert_padding`` even
when the model itself uses SDPA.  The Policy Pilot does not require a Flash
Attention kernel for this indexing operation, so this module installs the
mathematically equivalent PyTorch primitives at veRL's existing boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select first-axis rows with the same ordering as FlashAttention."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        raise TypeError("index_first_axis tensor must have at least one dimension")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise TypeError("index_first_axis indices must be a rank-1 tensor")
    if indices.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise TypeError("index_first_axis indices must be integral")
    normalized = indices.to(device=tensor.device, dtype=torch.long)
    return torch.index_select(tensor, 0, normalized)


def torch_unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Remove masked rows and return FlashAttention-compatible bookkeeping."""

    if unused_mask is not None:
        raise ValueError("Policy Pilot torch_unpad_input does not accept unused_mask")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 2:
        raise TypeError("hidden_states must have batch and sequence dimensions")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise TypeError("attention_mask must be a rank-2 tensor")
    if tuple(hidden_states.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("hidden_states and attention_mask dimensions differ")

    mask = attention_mask.to(dtype=torch.bool)
    sequence_lengths = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    unpadded = torch_index_first_axis(flat, indices)
    cumulative_lengths = F.pad(
        torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32),
        (1, 0),
    )
    maximum_length = (
        int(sequence_lengths.max().item()) if sequence_lengths.numel() else 0
    )
    return (
        unpadded,
        indices,
        cumulative_lengths,
        maximum_length,
        sequence_lengths,
    )


def install_verl_sdpa_padding_compat() -> None:
    """Install the audited Torch primitives into pinned veRL's padding module."""

    from verl.workers.utils import padding

    if not callable(getattr(padding, "left_right_2_no_padding", None)):
        raise RuntimeError("pinned veRL padding boundary is unavailable")
    if not callable(getattr(padding, "no_padding_2_padding", None)):
        raise RuntimeError("pinned veRL reverse-padding boundary is unavailable")
    padding.index_first_axis = torch_index_first_axis
    padding.unpad_input = torch_unpad_input


__all__ = [
    "install_verl_sdpa_padding_compat",
    "torch_index_first_axis",
    "torch_unpad_input",
]
