"""Pluggable selected-token log-probability materialization boundary."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from tgvf_rl.contracts.tokens import SamplingIdentity


class SelectedTokenLogprobMaterializer(Protocol):
    """Compute next-token log probabilities from decoder hidden states."""

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head: Any,
        token_ids: torch.Tensor,
        sampled_positions: torch.Tensor,
        sampling: SamplingIdentity,
    ) -> torch.Tensor: ...


__all__ = ["SelectedTokenLogprobMaterializer"]
