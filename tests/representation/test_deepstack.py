from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.representation.deepstack import (
    DDeepStackProjectionPorts,
    FrozenProjectionPort,
    build_original_image_key_block_mask,
)


class ToyMerger(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, group_size: int = 4) -> None:
        super().__init__()
        self.group_size = group_size
        self.linear = nn.Linear(input_dim * group_size, output_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens.reshape(-1, tokens.shape[-1] * self.group_size))


def _port(identity: str, *, output_dim: int = 5) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        ToyMerger(3, output_dim),
        identity=identity,
        input_dim=3,
        output_dim=output_dim,
        spatial_merge_size=2,
    )


def test_frozen_projection_port_supports_unbatched_and_batched_inputs() -> None:
    port = _port("qwen.visual.merger")
    tokens = torch.randn(2, 8, 3, requires_grad=True)

    output = port(tokens)

    assert output.shape == (2, 2, 5)
    assert torch.allclose(output[0], port(tokens[0]))
    assert not port.projection.training
    assert all(
        not parameter.requires_grad for parameter in port.projection.parameters()
    )
    output.sum().backward()
    assert tokens.grad is not None
    assert all(parameter.grad is None for parameter in port.projection.parameters())


def test_projection_port_rejects_invalid_merge_layout_and_output_shape() -> None:
    port = _port("qwen.visual.merger")
    with pytest.raises(ValueError, match="divisible"):
        port(torch.randn(6, 3))

    wrong = FrozenProjectionPort(
        nn.Identity(),
        identity="wrong-shape",
        input_dim=3,
        output_dim=5,
        spatial_merge_size=2,
    )
    with pytest.raises(ValueError, match="output shape"):
        wrong(torch.randn(8, 3))


def test_deepstack_ports_preserve_required_layer_order() -> None:
    ports = DDeepStackProjectionPorts(
        branch_layers=(8, 16, 24),
        projections=(_port("branch-8"), _port("branch-16"), _port("branch-24")),
    )
    branches = tuple(torch.randn(8, 3) for _ in range(3))

    payload = ports(branches)

    assert payload.branch_layers == (8, 16, 24)
    assert payload.projection_identities == ("branch-8", "branch-16", "branch-24")
    assert all(branch.shape == (2, 5) for branch in payload.branches)
    with pytest.raises(ValueError, match="branch count"):
        ports(branches[:2])


def test_original_image_key_block_mask_is_batched_and_causal() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    blocked = build_original_image_key_block_mask(
        attention_mask=attention_mask,
        original_image_token_indices=torch.tensor([1]),
        block_query_start=2,
        dtype=torch.float32,
    )
    minimum = torch.finfo(torch.float32).min

    assert blocked.shape == (2, 1, 4, 4)
    assert blocked[0, 0, 1, 1] == 0
    assert blocked[0, 0, 2, 1] == minimum
    assert blocked[0, 0, 0, 1] == minimum  # normal causal masking
    assert blocked[1, 0, :, 3].eq(minimum).all()  # key padding
