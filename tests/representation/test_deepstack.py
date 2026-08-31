from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.representation.deepstack import (
    DDeepStackProjectionPorts,
    FrozenProjectionPort,
    TrainableBorrowedProjectionPort,
    _build_original_image_key_block_mask_from_positions,
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


def test_trainable_borrowed_projection_is_non_owning_and_keeps_autograd() -> None:
    merger = ToyMerger(3, 5)
    port = TrainableBorrowedProjectionPort(
        merger,
        identity="qwen.visual.merger",
        input_dim=3,
        output_dim=5,
        spatial_merge_size=2,
    )
    owner = nn.Module()
    owner.merger = merger
    owner.port = port

    assert port._modules == {}  # noqa: SLF001
    assert tuple(port.parameters()) == ()
    assert port.state_dict() == {}
    assert tuple(dict(owner.named_parameters())) == ("merger.linear.weight",)

    tokens = torch.randn(2, 8, 3, requires_grad=True)
    output = port(tokens)
    output.square().sum().backward()

    assert output.shape == (2, 2, 5)
    assert tokens.grad is not None
    assert merger.linear.weight.grad is not None


def test_deepstack_accepts_trainable_non_owning_projection_ports() -> None:
    mergers = tuple(ToyMerger(3, 5) for _ in range(3))
    ports = DDeepStackProjectionPorts(
        branch_layers=(8, 16, 24),
        projections=tuple(
            TrainableBorrowedProjectionPort(
                merger,
                identity=f"branch-{layer}",
                input_dim=3,
                output_dim=5,
                spatial_merge_size=2,
            )
            for merger, layer in zip(mergers, (8, 16, 24), strict=True)
        ),
    )
    branches = tuple(torch.randn(8, 3, requires_grad=True) for _ in range(3))

    payload = ports(branches)
    sum(branch.sum() for branch in payload.branches).backward()

    assert ports.state_dict() == {}
    assert payload.projection_identities == ("branch-8", "branch-16", "branch-24")
    assert all(branch.grad is not None for branch in branches)
    assert all(merger.linear.weight.grad is not None for merger in mergers)


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


def test_internal_cpu_position_mask_path_has_no_tensor_content_host_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_mask = torch.ones(1, 4, dtype=torch.bool)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("trusted CPU positions must not inspect tensor contents")

    monkeypatch.setattr(torch.Tensor, "min", forbidden)
    monkeypatch.setattr(torch.Tensor, "max", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch, "unique", forbidden)
    blocked = _build_original_image_key_block_mask_from_positions(
        attention_mask=attention_mask,
        original_image_token_positions=(1,),
        block_query_start=2,
    )

    assert blocked.shape == (1, 1, 4, 4)
    with pytest.raises(ValueError, match="outside"):
        _build_original_image_key_block_mask_from_positions(
            attention_mask=attention_mask,
            original_image_token_positions=(4,),
            block_query_start=2,
        )
    with pytest.raises(ValueError, match="unique"):
        _build_original_image_key_block_mask_from_positions(
            attention_mask=attention_mask,
            original_image_token_positions=(1, 1),
            block_query_start=2,
        )
