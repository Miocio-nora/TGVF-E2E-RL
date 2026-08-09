from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import multiprocessing
from types import MethodType

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tgvf_rl.policy.trainable_tgvf_replay import (
    trainable_parameter_zero_anchor,
)


class _LifecycleRoot(nn.Module):
    """Small analogue of Qwen children plus RP66-owned root parameters."""

    def __init__(self, width: int = 2048) -> None:
        super().__init__()
        self.visual = nn.Linear(width, width, bias=False)
        self.root_adapter = nn.Linear(width, width, bias=False)
        self.language = nn.Linear(width, width, bias=False)

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise AssertionError("probe must enter through the registered root method")


@dataclass(frozen=True, slots=True)
class _LifecycleResult:
    logprobs: torch.Tensor


def _dispatch_lifecycle_root(
    _module: nn.Module, *, operation
) -> tuple[torch.Tensor, _LifecycleResult]:
    result = operation()
    return result.logprobs, result


def _run_registered_root_lifecycle(
    rank: int,
    world_size: int,
    rendezvous: str,
    reshard_after_forward: bool,
    output_queue,
) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        MixedPrecisionPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(73)
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        model = _LifecycleRoot().to(device=device, dtype=torch.bfloat16)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": policy,
            "reshard_after_forward": reshard_after_forward,
        }
        fully_shard(model.visual, **fsdp_kwargs)
        fully_shard(model.language, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        method_name = "_tgvf_registered_root_lifecycle_probe"
        setattr(model, method_name, MethodType(_dispatch_lifecycle_root, model))
        register_fsdp_forward_method(model, method_name)
        root_forward = getattr(model, method_name)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        for micro_index in range(12):
            # Different local shapes reproduce data-parallel replay imbalance
            # while preserving one identical FSDP module schedule per rank.
            values = torch.randn(
                (rank + 1, 2048), device=device, dtype=torch.bfloat16
            )

            def operation() -> _LifecycleResult:
                visual = checkpoint(model.visual, values, use_reentrant=False)
                # Even ranks model a TGVF observation; odd ranks model a
                # direct/no-tool row.  The zero anchor makes the root FSDP2
                # gradient parameter set identical without changing either
                # row's numerical result.
                adapted = (
                    visual + model.root_adapter(visual)
                    if rank % 2 == 0
                    else visual
                )
                language = checkpoint(model.language, adapted, use_reentrant=False)
                coverage = trainable_parameter_zero_anchor(model.root_adapter)
                return _LifecycleResult(
                    logprobs=language.float().mean(dim=-1) + coverage.float()
                )

            anchor, result = root_forward(operation=operation)
            if anchor is not result.logprobs:
                raise RuntimeError("registered root changed its autograd anchor")
            (anchor.square().mean() / 12).backward()

        model.reshard()
        grad = model.root_adapter.weight.grad
        finite_grad = grad is not None and bool(torch.isfinite(grad).all().item())
        optimizer.step()
        output_queue.put((rank, reshard_after_forward, finite_grad))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("reshard_after_forward", (False, True))
def test_registered_root_survives_repeated_checkpointed_microbatches(
    tmp_path, reshard_after_forward: bool
) -> None:
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 4:
        pytest.skip("registered-root lifecycle regression requires four CUDA GPUs")
    if torch.distributed.is_initialized():
        pytest.skip("registered-root lifecycle regression owns its process group")

    world_size = 4
    context = multiprocessing.get_context("spawn")
    output_queue = context.SimpleQueue()
    torch.multiprocessing.start_processes(
        _run_registered_root_lifecycle,
        args=(
            world_size,
            str(tmp_path / f"registered-root-{reshard_after_forward}"),
            reshard_after_forward,
            output_queue,
        ),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    assert sorted(output_queue.get() for _ in range(world_size)) == [
        (rank, reshard_after_forward, True) for rank in range(world_size)
    ]
