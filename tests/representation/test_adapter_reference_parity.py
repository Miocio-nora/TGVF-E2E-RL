from __future__ import annotations

from math import sqrt

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter


# Independent functional oracle for the two whitelisted equations in
# tgvf_foveal.py@f2244980599510c976a20dbbe227523fde5af72f26e8253188e04c072456853f.
# It deliberately does not import or execute the legacy repository.


class _Merger(nn.Module):
    def __init__(self, d_v: int, d_lm: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_v * 4, d_lm)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens.reshape(-1, tokens.shape[-1] * 4))


def _projection(identity: str, *, d_v: int, d_lm: int) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        _Merger(d_v, d_lm),
        identity=identity,
        input_dim=d_v,
        output_dim=d_lm,
        spatial_merge_size=2,
    )


def _adapter(*, dtype: torch.dtype) -> TGVFAdapter:
    adapter = TGVFAdapter(
        d_lm=6,
        d_v=4,
        attn_dim=5,
        main_projection=_projection("main", d_v=4, d_lm=6),
        deepstack_projections=tuple(
            _projection(f"branch-{layer}", d_v=4, d_lm=6) for layer in (8, 16, 24)
        ),
        branch_layers=(8, 16, 24),
    )
    return adapter.to(dtype=dtype)


def _linear(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    return F.linear(value, module.weight, module.bias)


def _layer_norm(module: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        value,
        module.normalized_shape,
        module.weight,
        module.bias,
        module.eps,
    )


def _cross_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = queries @ keys.transpose(0, 1) / sqrt(queries.shape[-1])
    attention = F.softmax(scores, dim=-1)
    return attention @ values, attention


def _reference_attention(
    module: nn.Module,
    target: torch.Tensor,
    visual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.to(dtype=module.target_norm.weight.dtype)
    visual = visual.to(dtype=module.visual_norm.weight.dtype)
    target_tokens = _linear(
        module.target_proj,
        _layer_norm(module.target_norm, target),
    )
    visual_tokens = _layer_norm(module.visual_norm, visual)
    visual_projected = _linear(module.visual_proj, visual_tokens)
    target_context, target_to_visual = _cross_attention(
        _linear(module.target_q_proj, target_tokens),
        _linear(module.visual_k_proj, visual_projected),
        _linear(module.visual_v_proj, visual_projected),
    )
    enriched_target = _layer_norm(
        module.enriched_target_norm,
        target_tokens + target_context,
    )
    visual_context, visual_to_target = _cross_attention(
        _linear(module.visual_q_proj, visual_projected),
        _linear(module.target_k_proj, enriched_target),
        _linear(module.target_v_proj, enriched_target),
    )
    delta = _linear(module.context_to_delta, visual_context)
    gate = torch.sigmoid(
        _linear(module.gate_proj, torch.cat([visual_tokens, visual_context], dim=-1))
    )
    return visual + gate * delta, target_to_visual, visual_to_target


def _reference_adapter(
    adapter: TGVFAdapter,
    target: torch.Tensor,
    main_visual: torch.Tensor,
    branch_visual: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    main_conditioned, _, _ = _reference_attention(adapter, target, main_visual)
    main = adapter.main_projection.projection(main_conditioned)
    branches = []
    for layer, visual, port in zip(
        adapter.d_deepstack_branch_layers,
        branch_visual,
        adapter.d_deepstack_projections.projections,
        strict=True,
    ):
        conditioned, _, _ = _reference_attention(
            adapter.d_deepstack_branch_adapters[str(layer)],
            target,
            visual,
        )
        branches.append(port.projection(conditioned))
    return main, tuple(branches)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2e-6, 2e-6),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_tgvf_adapter_matches_pinned_functional_output_and_gradient_oracle(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    torch.manual_seed(20260719)
    actual_adapter = _adapter(dtype=dtype)
    reference_adapter = _adapter(dtype=dtype)
    reference_adapter.load_state_dict(actual_adapter.state_dict())

    target_actual = torch.randn(3, 6, dtype=dtype, requires_grad=True)
    main_actual = torch.randn(8, 4, dtype=dtype, requires_grad=True)
    branches_actual = tuple(
        torch.randn(8, 4, dtype=dtype, requires_grad=True) for _ in range(3)
    )
    target_reference = target_actual.detach().clone().requires_grad_(True)
    main_reference = main_actual.detach().clone().requires_grad_(True)
    branches_reference = tuple(
        value.detach().clone().requires_grad_(True) for value in branches_actual
    )

    actual = actual_adapter(
        target_hidden_states=target_actual,
        pre_merge_visual_tokens=main_actual,
        deepstack_pre_merge_visual_tokens=branches_actual,
    )
    expected_main, expected_branches = _reference_adapter(
        reference_adapter,
        target_reference,
        main_reference,
        branches_reference,
    )
    torch.testing.assert_close(actual.main_d, expected_main, atol=atol, rtol=rtol)
    for observed, expected in zip(
        actual.deepstack_visual_embeds,
        expected_branches,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)

    coefficients = tuple(
        torch.linspace(0.2 + index, 1.2 + index, value.numel(), dtype=dtype).reshape_as(
            value
        )
        for index, value in enumerate((actual.main_d, *actual.deepstack_visual_embeds))
    )
    actual_loss = sum(
        (value * coefficient).sum()
        for value, coefficient in zip(
            (actual.main_d, *actual.deepstack_visual_embeds),
            coefficients,
            strict=True,
        )
    )
    reference_loss = sum(
        (value * coefficient).sum()
        for value, coefficient in zip(
            (expected_main, *expected_branches),
            coefficients,
            strict=True,
        )
    )
    actual_owned = tuple(actual_adapter.artifact_state_dict(keep_vars=True).values())
    reference_owned = tuple(
        reference_adapter.artifact_state_dict(keep_vars=True).values()
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (target_actual, main_actual, *branches_actual, *actual_owned),
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (
            target_reference,
            main_reference,
            *branches_reference,
            *reference_owned,
        ),
    )
    assert len(actual_owned) == len(reference_owned) == 104
    for observed, expected in zip(
        actual_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)

    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in (
            *actual_adapter.main_projection.projection.parameters(),
            *(
                parameter
                for port in actual_adapter.d_deepstack_projections.projections
                for parameter in port.projection.parameters()
            ),
        )
    )
