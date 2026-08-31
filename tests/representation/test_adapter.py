from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.representation import (
    FrozenProjectionPort,
    TGVFAdapter,
    TGVFAdapterVariant,
)


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


def _adapter(
    variant: TGVFAdapterVariant = TGVFAdapterVariant.FULL_D_DEEPSTACK,
) -> TGVFAdapter:
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
        variant=variant,
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
    gated_delta = output.main_attention.conditioned_visual_tokens - main_visual
    expected_salience = torch.softmax(
        torch.linalg.vector_norm(gated_delta.float(), dim=-1), dim=-1
    ).to(dtype=gated_delta.dtype)
    assert output.main_attention.visual_salience.shape == (1, 8)
    assert torch.allclose(
        output.main_attention.visual_salience, expected_salience.unsqueeze(0)
    )
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
    assert output.main_attention.visual_salience.shape == (2, 8)
    assert torch.allclose(
        output.main_attention.visual_salience.sum(dim=-1), torch.ones(2)
    )


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


def test_adapter_artifact_state_excludes_every_borrowed_qwen_merger() -> None:
    source = _adapter()
    full_state = source.state_dict()
    artifact_state = {
        name: value.clone() for name, value in source.artifact_state_dict().items()
    }
    component_names = (
        "target_norm",
        "target_proj",
        "visual_norm",
        "visual_proj",
        "target_q_proj",
        "visual_k_proj",
        "visual_v_proj",
        "enriched_target_norm",
        "visual_q_proj",
        "target_k_proj",
        "target_v_proj",
        "context_to_delta",
        "gate_proj",
    )
    expected_keys = {
        f"{component}.{suffix}"
        for component in component_names
        for suffix in ("weight", "bias")
    }
    expected_keys.update(
        f"d_deepstack_branch_adapters.{layer}.{component}.{suffix}"
        for layer in (8, 16, 24)
        for component in component_names
        for suffix in ("weight", "bias")
    )

    assert len(artifact_state) == 104
    assert set(artifact_state) == expected_keys
    assert "target_norm.weight" in artifact_state
    assert any(name.startswith("main_projection.") for name in full_state)
    assert any(name.startswith("d_deepstack_projections.") for name in full_state)
    assert not any(name.startswith("main_projection.") for name in artifact_state)
    assert not any(
        name.startswith("d_deepstack_projections.") for name in artifact_state
    )

    target = _adapter()
    with torch.no_grad():
        target.target_norm.weight.zero_()
        next(target.main_projection.projection.parameters()).add_(7)
    projection_before = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if name not in artifact_state
    }
    assert not torch.equal(
        target.target_norm.weight, artifact_state["target_norm.weight"]
    )

    target.load_artifact_state_dict(artifact_state)

    projection_after = target.state_dict()
    assert torch.equal(target.target_norm.weight, artifact_state["target_norm.weight"])
    assert all(
        torch.equal(value, projection_after[name])
        for name, value in projection_before.items()
    )

    polluted = dict(artifact_state)
    polluted["main_projection.projection.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="artifact keys mismatch"):
        target.load_artifact_state_dict(polluted)


def test_main_d_only_has_no_learned_branch_parameters_or_branch_objective() -> None:
    adapter = _adapter(TGVFAdapterVariant.MAIN_D_ONLY)
    target = torch.randn(3, 6, requires_grad=True)
    main_visual = torch.randn(8, 4, requires_grad=True)
    branch_visual = tuple(torch.randn(8, 4, requires_grad=True) for _ in range(3))

    output = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=main_visual,
        deepstack_pre_merge_visual_tokens=branch_visual,
    )

    assert len(adapter.d_deepstack_branch_adapters) == 0
    assert len(adapter.artifact_state_dict()) == 26
    assert not any(
        name.startswith("d_deepstack_branch_adapters.")
        for name in adapter.artifact_state_dict()
    )
    assert output.metadata.variant is TGVFAdapterVariant.MAIN_D_ONLY
    assert output.metadata.deepstack_projection_identities == ()
    assert output.conditioned_deepstack_pre_merge_visual_tokens == ()
    assert output.deepstack_attention == ()
    assert all(
        torch.count_nonzero(branch).item() == 0
        for branch in output.deepstack_visual_embeds
    )
    assert all(not branch.requires_grad for branch in output.deepstack_visual_embeds)

    output.main_d.sum().backward()
    assert target.grad is not None
    assert main_visual.grad is not None
    assert all(branch.grad is None for branch in branch_visual)


def test_vision_routing_variant_retains_full_outputs_without_target_value_path() -> (
    None
):
    adapter = _adapter(TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING)
    historical = _adapter()
    target = torch.randn(3, 6, requires_grad=True)
    main_visual = torch.randn(8, 4, requires_grad=True)
    branch_visual = tuple(torch.randn(8, 4, requires_grad=True) for _ in range(3))

    output = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=main_visual,
        deepstack_pre_merge_visual_tokens=branch_visual,
    )

    assert output.metadata.variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING
    )
    assert output.main_d.shape == (2, 6)
    assert len(output.deepstack_visual_embeds) == 3
    assert all(branch.shape == (2, 6) for branch in output.deepstack_visual_embeds)
    assert adapter.vision_routing_only
    assert all(
        branch.vision_routing_only
        for branch in adapter.d_deepstack_branch_adapters.values()
    )
    assert len(adapter.artifact_state_dict()) == 104
    assert (
        adapter.artifact_state_dict().keys() == historical.artifact_state_dict().keys()
    )
    assert all(
        torch.equal(value, historical.artifact_state_dict()[name])
        for name, value in adapter.artifact_state_dict().items()
    )

    output.main_d.sum().backward()
    assert target.grad is not None
    assert main_visual.grad is not None


def test_visual_barycentric_variant_retains_parameters_but_only_routes_raw_visuals() -> (
    None
):
    adapter = _adapter(TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC)
    historical = _adapter()
    target = torch.randn(3, 6, requires_grad=True)
    main_visual = torch.randn(8, 4, requires_grad=True)
    branch_visual = tuple(torch.randn(8, 4, requires_grad=True) for _ in range(3))
    captured_delta: list[torch.Tensor] = []
    hook = adapter.context_to_delta.register_forward_hook(
        lambda _module, _inputs, output: captured_delta.append(output.detach().clone())
    )
    try:
        output = adapter(
            target_hidden_states=target,
            pre_merge_visual_tokens=main_visual,
            deepstack_pre_merge_visual_tokens=branch_visual,
        )
    finally:
        hook.remove()

    assert output.metadata.variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC
    )
    assert adapter.vision_routing_only
    assert adapter.visual_barycentric_writer
    assert all(
        branch.visual_barycentric_writer
        for branch in adapter.d_deepstack_branch_adapters.values()
    )
    assert len(adapter.artifact_state_dict()) == 104
    assert (
        adapter.artifact_state_dict().keys() == historical.artifact_state_dict().keys()
    )

    assert len(captured_delta) == 1
    query = output.main_attention.gate.detach() * captured_delta[0]
    keys = adapter.visual_norm(main_visual.detach())
    weights = torch.softmax(
        torch.matmul(query, keys.transpose(-2, -1)) / adapter.d_v**0.5,
        dim=-1,
    )
    expected = torch.matmul(weights, main_visual.detach())
    torch.testing.assert_close(
        output.conditioned_pre_merge_visual_tokens.detach(),
        expected,
    )
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(8))
    lower = main_visual.detach().amin(dim=-2)
    upper = main_visual.detach().amax(dim=-2)
    assert torch.all(output.conditioned_pre_merge_visual_tokens.detach() >= lower)
    assert torch.all(output.conditioned_pre_merge_visual_tokens.detach() <= upper)

    loss = output.main_d.sum() + sum(
        branch.sum() for branch in output.deepstack_visual_embeds
    )
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in adapter.artifact_state_dict(keep_vars=True).values()
    )


def test_vision_routing_target_affects_only_scalar_visual_routing() -> None:
    adapter = _adapter(TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING).eval()
    visual = torch.randn(8, 4)
    branches = tuple(torch.randn(8, 4) for _ in range(3))
    target_a = torch.randn(3, 6)
    target_b = torch.randn(3, 6)
    second_stage_value_inputs: list[torch.Tensor] = []

    value_hook = adapter.target_v_proj.register_forward_pre_hook(
        lambda _module, values: second_stage_value_inputs.append(
            values[0].detach().clone()
        )
    )
    try:
        first = adapter(
            target_hidden_states=target_a,
            pre_merge_visual_tokens=visual,
            deepstack_pre_merge_visual_tokens=branches,
        )
        second = adapter(
            target_hidden_states=target_b,
            pre_merge_visual_tokens=visual,
            deepstack_pre_merge_visual_tokens=branches,
        )
    finally:
        value_hook.remove()

    # The second attention receives exactly the first attention's weighted
    # visual values. Target state determines the scalar weights but is never
    # added to the value/payload stream.
    normalized_visual = adapter.visual_norm(visual)
    visual_projected = adapter.visual_proj(normalized_visual)
    visual_values = adapter.visual_v_proj(visual_projected)
    assert len(second_stage_value_inputs) == 2
    for captured, result in zip(
        second_stage_value_inputs,
        (first, second),
        strict=True,
    ):
        expected = torch.matmul(
            result.main_attention.target_to_visual_attention,
            visual_values,
        )
        torch.testing.assert_close(captured, expected)
    assert not torch.equal(
        first.main_attention.target_to_visual_attention,
        second.main_attention.target_to_visual_attention,
    )
    assert not torch.equal(first.main_attention.gate, second.main_attention.gate)
    assert torch.all(first.main_attention.gate >= 0)
    assert torch.all(first.main_attention.gate <= 1)
    assert torch.allclose(
        first.main_attention.visual_salience.sum(dim=-1), torch.ones(1)
    )


@pytest.mark.parametrize(
    "variant",
    (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING,
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC,
    ),
)
def test_vision_routing_cannot_encode_target_when_visual_tokens_are_indistinguishable(
    variant: TGVFAdapterVariant,
) -> None:
    adapter = _adapter(variant).eval()
    repeated_main = torch.randn(1, 4).expand(8, -1).clone()
    repeated_branches = tuple(torch.randn(1, 4).expand(8, -1).clone() for _ in range(3))

    first = adapter(
        target_hidden_states=torch.randn(3, 6),
        pre_merge_visual_tokens=repeated_main,
        deepstack_pre_merge_visual_tokens=repeated_branches,
    )
    second = adapter(
        target_hidden_states=torch.randn(3, 6),
        pre_merge_visual_tokens=repeated_main,
        deepstack_pre_merge_visual_tokens=repeated_branches,
    )

    torch.testing.assert_close(
        first.conditioned_pre_merge_visual_tokens,
        second.conditioned_pre_merge_visual_tokens,
    )
    torch.testing.assert_close(first.main_d, second.main_d)
    for first_branch, second_branch in zip(
        first.deepstack_visual_embeds,
        second.deepstack_visual_embeds,
        strict=True,
    ):
        torch.testing.assert_close(first_branch, second_branch)
