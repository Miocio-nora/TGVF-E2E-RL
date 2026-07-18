from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter


class ToyMerger(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 4, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens.reshape(-1, tokens.shape[-1] * 4))


def _projection(identity: str) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        ToyMerger(4, 6),
        identity=identity,
        input_dim=4,
        output_dim=6,
        spatial_merge_size=2,
    )


def _adapter() -> TGVFAdapter:
    torch.manual_seed(11)
    return TGVFAdapter(
        d_lm=6,
        d_v=4,
        attn_dim=5,
        main_projection=_projection("main-merger"),
        deepstack_projections=tuple(
            _projection(identity) for identity in ("merger-8", "merger-16", "merger-24")
        ),
        branch_layers=(8, 16, 24),
    )


def test_adapter_produces_main_and_required_independent_deepstack_branches() -> None:
    adapter = _adapter()
    target = torch.randn(3, 6, requires_grad=True)
    main_visual = torch.randn(8, 4, requires_grad=True)
    branch_visual = tuple(torch.randn(8, 4, requires_grad=True) for _ in range(3))

    output = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=main_visual,
        deepstack_pre_merge_visual_tokens=branch_visual,
    )

    assert output.main_d.shape == (2, 6)
    assert output.d_deepstack.branch_layers == (8, 16, 24)
    assert len(output.deepstack_visual_embeds) == 3
    assert all(branch.shape == (2, 6) for branch in output.deepstack_visual_embeds)
    assert (
        len({id(module) for module in adapter.d_deepstack_branch_adapters.values()})
        == 3
    )

    loss = output.main_d.sum() + sum(
        branch.sum() for branch in output.deepstack_visual_embeds
    )
    loss.backward()
    assert target.grad is not None
    assert main_visual.grad is not None
    assert all(branch.grad is not None for branch in branch_visual)
    projection_parameters = list(adapter.main_projection.projection.parameters()) + [
        parameter
        for port in adapter.d_deepstack_projections.projections
        for parameter in port.projection.parameters()
    ]
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in projection_parameters
    )
    assert adapter.target_proj.weight.grad is not None


def test_adapter_supports_explicit_batch_dimension() -> None:
    adapter = _adapter()
    output = adapter(
        target_hidden_states=torch.randn(2, 3, 6),
        pre_merge_visual_tokens=torch.randn(2, 8, 4),
        deepstack_pre_merge_visual_tokens=tuple(torch.randn(2, 8, 4) for _ in range(3)),
    )

    assert output.main_d.shape == (2, 2, 6)
    assert output.metadata.batched
    assert output.metadata.batch_size == 2
    assert all(branch.shape == (2, 2, 6) for branch in output.deepstack_visual_embeds)


def test_adapter_rejects_implicit_broadcasting_and_missing_branches() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="mix batched"):
        adapter(
            target_hidden_states=torch.randn(2, 3, 6),
            pre_merge_visual_tokens=torch.randn(8, 4),
            deepstack_pre_merge_visual_tokens=tuple(
                torch.randn(8, 4) for _ in range(3)
            ),
        )
    with pytest.raises(ValueError, match="branch count"):
        adapter(
            target_hidden_states=torch.randn(3, 6),
            pre_merge_visual_tokens=torch.randn(8, 4),
            deepstack_pre_merge_visual_tokens=(torch.randn(8, 4),),
        )


def test_adapter_branch_conditioning_does_not_leak_between_branches() -> None:
    adapter = _adapter().eval()
    target = torch.randn(3, 6)
    main = torch.randn(8, 4)
    branches = [torch.randn(8, 4) for _ in range(3)]
    first = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=main,
        deepstack_pre_merge_visual_tokens=branches,
    )
    changed = list(branches)
    changed[0] = changed[0] + 2.0
    second = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=main,
        deepstack_pre_merge_visual_tokens=changed,
    )

    assert torch.equal(first.main_d, second.main_d)
    assert not torch.equal(
        first.deepstack_visual_embeds[0], second.deepstack_visual_embeds[0]
    )
    assert torch.equal(
        first.deepstack_visual_embeds[1], second.deepstack_visual_embeds[1]
    )
    assert torch.equal(
        first.deepstack_visual_embeds[2], second.deepstack_visual_embeds[2]
    )
