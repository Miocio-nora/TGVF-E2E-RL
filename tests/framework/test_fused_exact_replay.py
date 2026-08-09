from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity
from tgvf_rl.framework.verl.exact_replay_engine import (
    _validate_execution_capabilities,
)
from tgvf_rl.framework.verl.fused_exact_replay import (
    fused_selected_next_token_logprobs,
)
from tgvf_rl.framework.verl.trainable_tgvf_engine import (
    _validate_trainable_execution_capabilities,
)


def _sampling() -> SamplingIdentity:
    return SamplingIdentity(
        policy_version=PolicyVersion("fused-test", 0, "0" * 64),
        backend="vllm",
        backend_version="0.12.0",
        seed=42,
        rng_state_sha256="1" * 64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )


def _eager_selected_logprobs(
    hidden: torch.Tensor,
    head: nn.Linear,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(token_ids.shape[0]).unsqueeze(1)
    predictive_logits = head(hidden)[batch, positions - 1]
    selected_ids = token_ids[batch, positions]
    return (
        torch.log_softmax(predictive_logits.float(), dim=-1)
        .gather(-1, selected_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def test_fused_selected_logprobs_match_eager_values_and_gradients() -> None:
    torch.manual_seed(17)
    eager_hidden = torch.randn(2, 7, 11, requires_grad=True)
    fused_hidden = eager_hidden.detach().clone().requires_grad_(True)
    eager_head = nn.Linear(11, 19, bias=False)
    fused_head = nn.Linear(11, 19, bias=False)
    fused_head.load_state_dict(eager_head.state_dict())
    token_ids = torch.randint(0, 19, (2, 7))
    positions = torch.tensor([[1, 3, 6], [2, 4, 5]])
    coefficients = torch.randn(2, 3)

    eager = _eager_selected_logprobs(
        eager_hidden, eager_head, token_ids, positions
    )
    fused = fused_selected_next_token_logprobs(
        hidden_states=fused_hidden,
        lm_head=fused_head,
        token_ids=token_ids,
        sampled_positions=positions,
        sampling=_sampling(),
    )
    torch.testing.assert_close(fused, eager, atol=2e-6, rtol=2e-6)

    (eager * coefficients).sum().backward()
    (fused * coefficients).sum().backward()
    torch.testing.assert_close(
        fused_hidden.grad, eager_hidden.grad, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        fused_head.weight.grad,
        eager_head.weight.grad,
        atol=2e-6,
        rtol=2e-6,
    )


def test_exact_replay_accepts_fused_only_for_a_materializing_port() -> None:
    micro_batch = {"use_fused_kernels": True}
    supported = type("SupportedPort", (), {"materializes_fused_kernels": True})()
    _validate_execution_capabilities(micro_batch, supported)

    unsupported = type("UnsupportedPort", (), {})()
    with pytest.raises(
        ValueError, match="does not implement fused selected-token logprobs"
    ):
        _validate_execution_capabilities(micro_batch, unsupported)


def test_trainable_worker_preflights_the_selected_fused_backend() -> None:
    _validate_trainable_execution_capabilities(
        SimpleNamespace(
            use_fused_kernels=True,
            fused_kernel_options={"impl_backend": "torch"},
        )
    )
    with pytest.raises(ValueError, match="torch fused backend"):
        _validate_trainable_execution_capabilities(
            SimpleNamespace(
                use_fused_kernels=True,
                fused_kernel_options={"impl_backend": "triton"},
            )
        )
