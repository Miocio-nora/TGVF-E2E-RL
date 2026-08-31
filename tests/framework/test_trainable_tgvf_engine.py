from __future__ import annotations

import torch
from torch import nn

from tgvf_rl.framework.verl.trainable_tgvf_engine import (
    _adapter_state_for_runtime_dtype,
    assert_frozen_rp66_optimizer_scope,
)
from tgvf_rl.policy.trainable_tgvf_replay import (
    TRAINABLE_TGVF_ADAPTER_ATTRIBUTE,
)
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter


class _ToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, 4, 3).mean(dim=1))


def _frozen_adapter() -> TGVFAdapter:
    ports = tuple(
        FrozenProjectionPort(
            _ToyMerger(),
            identity=f"engine-test-merger-{index}",
            input_dim=3,
            output_dim=5,
            spatial_merge_size=2,
        )
        for index in range(4)
    )
    adapter = TGVFAdapter(
        d_lm=5,
        d_v=3,
        attn_dim=4,
        main_projection=ports[0],
        deepstack_projections=ports[1:],
        branch_layers=(8, 16, 24),
    )
    adapter.requires_grad_(False)
    return adapter


class _ToyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = nn.Linear(5, 7)
        self.add_module(TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, _frozen_adapter())


def test_runtime_dtype_cast_preserves_source_and_nonfloating_state() -> None:
    floating = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    integer = torch.tensor([1, 2], dtype=torch.int64)

    normalized = _adapter_state_for_runtime_dtype(
        {"weight": floating, "counter": integer}, dtype=torch.float32
    )

    assert normalized["weight"].dtype is torch.float32
    assert torch.equal(normalized["weight"], floating.float())
    assert floating.dtype is torch.bfloat16
    assert normalized["counter"] is integer


def test_runtime_dtype_cast_rejects_nonfloating_target() -> None:
    try:
        _adapter_state_for_runtime_dtype(
            {"weight": torch.ones(1, dtype=torch.bfloat16)}, dtype=torch.int64
        )
    except TypeError as error:
        assert "must be floating point" in str(error)
    else:  # pragma: no cover - explicit assertion without pytest dependency
        raise AssertionError("non-floating RP66 runtime dtype was accepted")


def test_frozen_optimizer_owns_qwen_and_excludes_adapter() -> None:
    actor = _ToyActor()
    optimizer = torch.optim.AdamW(actor.qwen.parameters(), lr=1e-6)

    assert_frozen_rp66_optimizer_scope(actor, optimizer)


def test_frozen_optimizer_rejects_adapter_parameter() -> None:
    actor = _ToyActor()
    adapter_parameter = next(actor.tgvf_adapter.parameters())
    optimizer = torch.optim.AdamW(
        (*tuple(actor.qwen.parameters()), adapter_parameter), lr=1e-6
    )

    try:
        assert_frozen_rp66_optimizer_scope(actor, optimizer)
    except RuntimeError as error:
        assert "entered optimizer" in str(error)
    else:  # pragma: no cover - explicit assertion without pytest dependency
        raise AssertionError("frozen RP66 parameter was accepted by optimizer")
