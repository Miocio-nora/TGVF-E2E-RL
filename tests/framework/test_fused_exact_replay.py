from __future__ import annotations

import multiprocessing
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.framework.verl import fused_exact_replay as fused_replay
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity
from tgvf_rl.framework.verl.exact_replay_engine import (
    _validate_execution_capabilities,
)
from tgvf_rl.framework.verl.fused_exact_replay import (
    FusedExactReplayMicrobatchMaterializer,
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


def test_microbatch_materializer_reuses_one_weight_and_sums_row_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    eager_head = nn.Linear(13, 23, bias=False)
    fused_head = nn.Linear(13, 23, bias=False)
    fused_head.load_state_dict(eager_head.state_dict())
    eager_rows = [torch.randn(1, 8, 13, requires_grad=True) for _ in range(3)]
    fused_rows = [
        row.detach().clone().requires_grad_(True) for row in eager_rows
    ]
    token_rows = [torch.randint(0, 23, (1, 8)) for _ in range(3)]
    positions = torch.tensor([[1, 3, 5, 7]])
    coefficients = [torch.randn(1, 4) for _ in range(3)]

    original = fused_replay._materialize_lm_head_weight
    materializations: list[torch.Tensor] = []

    def counted_materialization(**kwargs: object) -> torch.Tensor:
        materialized = original(**kwargs)
        materializations.append(materialized)
        return materialized

    monkeypatch.setattr(
        fused_replay,
        "_materialize_lm_head_weight",
        counted_materialization,
    )
    materializer = FusedExactReplayMicrobatchMaterializer()
    eager_loss = torch.zeros(())
    fused_loss = torch.zeros(())
    for eager_hidden, fused_hidden, token_ids, coefficient in zip(
        eager_rows,
        fused_rows,
        token_rows,
        coefficients,
        strict=True,
    ):
        eager = _eager_selected_logprobs(
            eager_hidden, eager_head, token_ids, positions
        )
        fused = materializer(
            hidden_states=fused_hidden,
            lm_head=fused_head,
            token_ids=token_ids,
            sampled_positions=positions,
            sampling=_sampling(),
        )
        torch.testing.assert_close(fused, eager, atol=2e-6, rtol=2e-6)
        eager_loss = eager_loss + (eager * coefficient).sum()
        fused_loss = fused_loss + (fused * coefficient).sum()

    assert len(materializations) == 1
    eager_loss.backward()
    fused_loss.backward()
    for eager_hidden, fused_hidden in zip(eager_rows, fused_rows, strict=True):
        torch.testing.assert_close(
            fused_hidden.grad,
            eager_hidden.grad,
            atol=2e-6,
            rtol=2e-6,
        )
    torch.testing.assert_close(
        fused_head.weight.grad,
        eager_head.weight.grad,
        atol=2e-6,
        rtol=2e-6,
    )


def test_microbatch_materializer_never_crosses_its_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = nn.Linear(7, 11, bias=False)
    other_head = nn.Linear(7, 11, bias=False)
    hidden = torch.randn(1, 4, 7, requires_grad=True)
    token_ids = torch.randint(0, 11, (1, 4))
    positions = torch.tensor([[1, 3]])
    original = fused_replay._materialize_lm_head_weight
    calls = 0

    def counted_materialization(**kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        fused_replay,
        "_materialize_lm_head_weight",
        counted_materialization,
    )
    first = FusedExactReplayMicrobatchMaterializer()
    second = FusedExactReplayMicrobatchMaterializer()
    call = {
        "hidden_states": hidden,
        "lm_head": head,
        "token_ids": token_ids,
        "sampled_positions": positions,
        "sampling": _sampling(),
    }
    first(**call)
    second(**call)
    assert calls == 2

    with pytest.raises(RuntimeError, match="cannot cross an LM head"):
        first(**{**call, "lm_head": other_head})


def _run_dtensor_materializer_regression(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_queue,
) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous}",
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        torch.manual_seed(41)
        full_weight = torch.randn(8, 5)
        weight = distribute_tensor(full_weight, mesh, placements=(Shard(0),))
        weight.requires_grad_(True)
        head = SimpleNamespace(weight=weight)
        materializer = FusedExactReplayMicrobatchMaterializer()

        torch.manual_seed(100 + rank)
        actual_hidden = [
            torch.randn(1, 6, 5, requires_grad=True) for _ in range(2)
        ]
        expected_hidden = [
            row.detach().clone().requires_grad_(True) for row in actual_hidden
        ]
        token_rows = [torch.randint(0, 8, (1, 6)) for _ in range(2)]
        positions = torch.tensor([[1, 3, 5]])
        coefficients = [torch.randn(1, 3) for _ in range(2)]
        expected_weight = full_weight.detach().clone().requires_grad_(True)

        full_tensor_calls = 0
        original_full_tensor = DTensor.full_tensor

        def counted_full_tensor(self, *args, **kwargs):
            nonlocal full_tensor_calls
            full_tensor_calls += 1
            return original_full_tensor(self, *args, **kwargs)

        DTensor.full_tensor = counted_full_tensor
        try:
            actual_loss = torch.zeros(())
            expected_loss = torch.zeros(())
            values_match = True
            for actual, expected, token_ids, coefficient in zip(
                actual_hidden,
                expected_hidden,
                token_rows,
                coefficients,
                strict=True,
            ):
                actual_values = materializer(
                    hidden_states=actual,
                    lm_head=head,
                    token_ids=token_ids,
                    sampled_positions=positions,
                    sampling=_sampling(),
                )
                batch = torch.arange(token_ids.shape[0]).unsqueeze(1)
                logits = expected[batch, positions - 1] @ expected_weight.t()
                selected_ids = token_ids[batch, positions]
                expected_values = (
                    logits.float()
                    .log_softmax(dim=-1)
                    .gather(-1, selected_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                values_match = values_match and torch.allclose(
                    actual_values,
                    expected_values,
                    atol=2e-6,
                    rtol=2e-6,
                )
                actual_loss = actual_loss + (actual_values * coefficient).sum()
                expected_loss = expected_loss + (expected_values * coefficient).sum()
            actual_loss.backward()
            expected_loss.backward()
        finally:
            DTensor.full_tensor = original_full_tensor

        expected_weight_grad = expected_weight.grad
        assert expected_weight_grad is not None
        actual_weight_grad = weight.grad
        assert isinstance(actual_weight_grad, DTensor)
        # This raw-DTensor fixture verifies cache reuse and each rank's row
        # gradient. Cross-rank FSDP reduction is outside this regression's
        # scope and must not be inferred from this local-shard comparison.
        expected_local = expected_weight_grad.chunk(world_size, dim=0)[rank]
        hidden_grads_match = all(
            torch.allclose(actual.grad, expected.grad, atol=2e-6, rtol=2e-6)
            for actual, expected in zip(
                actual_hidden, expected_hidden, strict=True
            )
        )
        output_queue.put(
            (
                rank,
                full_tensor_calls,
                values_match,
                hidden_grads_match,
                torch.allclose(
                    actual_weight_grad.to_local(),
                    expected_local,
                    atol=2e-6,
                    rtol=2e-6,
                ),
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def test_microbatch_materializer_reuses_real_dtensor_and_preserves_row_gradients(
    tmp_path,
) -> None:
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("two-rank DTensor regression requires torch.distributed gloo")
    if torch.distributed.is_initialized():
        pytest.skip("two-rank DTensor regression requires process-group ownership")

    world_size = 2
    context = multiprocessing.get_context("spawn")
    output_queue = context.SimpleQueue()
    torch.multiprocessing.start_processes(
        _run_dtensor_materializer_regression,
        args=(
            world_size,
            str(tmp_path / "fused-dtensor-rendezvous"),
            output_queue,
        ),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    observed = sorted(output_queue.get() for _ in range(world_size))
    assert observed == [
        (0, 1, True, True, True),
        (1, 1, True, True, True),
    ]


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
