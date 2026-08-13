from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import multiprocessing
from queue import Empty
import time
from types import MethodType

import pytest
import torch
from torch import nn

from tgvf_rl.policy.trainable_tgvf_replay import (
    extract_live_qwen3_vision_feature_batch,
)


class _PackedMerger(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, 4, 8).mean(dim=1))


class _PackedVisual(nn.Module):
    """Small Qwen-vision analogue with the same four-merger contract."""

    spatial_merge_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(8, 8, bias=False)
        self.merger = _PackedMerger(8)
        self.deepstack_merger_list = nn.ModuleList(_PackedMerger(8) for _ in range(3))
        self.forward_calls = 0

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        self.forward_calls += 1
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise RuntimeError("packed grid_thw must have shape [images,3]")
        expected_rows = int(torch.prod(grid_thw, dim=1).sum().item())
        if expected_rows != int(pixel_values.shape[0]):
            raise RuntimeError("packed pixels and grid_thw disagree")
        hidden = torch.nn.functional.gelu(self.block(pixel_values))
        main = self.merger(hidden)
        deepstack = tuple(
            merger(hidden * (branch_index + 2))
            for branch_index, merger in enumerate(self.deepstack_merger_list)
        )
        return main, deepstack


class _PackedReplayRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _PackedVisual()

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise AssertionError("probe must enter through the registered root method")


@dataclass(frozen=True, slots=True)
class _PackedReplayResult:
    loss: torch.Tensor
    image_count: int


def _dispatch_packed_replay_root(
    _module: nn.Module, *, operation
) -> tuple[torch.Tensor, _PackedReplayResult]:
    result = operation()
    return result.loss, result


def _local_finite_nonzero_gradient(parameter: nn.Parameter) -> tuple[bool, bool]:
    gradient = parameter.grad
    if gradient is None:
        return False, False
    local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
    return (
        bool(torch.isfinite(local_gradient).all().item()),
        bool(torch.count_nonzero(local_gradient).item()),
    )


def _run_asymmetric_packed_vision_replay(
    rank: int,
    world_size: int,
    rendezvous: str,
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
        torch.manual_seed(211)
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        model = _PackedReplayRoot().to(device=device, dtype=torch.bfloat16)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": policy,
            "reshard_after_forward": True,
        }
        # This independently wrapped vision block is the important collective:
        # every rank must enter it once even though its packed image count differs.
        fully_shard(model.model.visual.block, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        method_name = "_tgvf_asymmetric_packed_vision_replay_probe"
        setattr(model, method_name, MethodType(_dispatch_packed_replay_root, model))
        register_fsdp_forward_method(model, method_name)
        root_forward = getattr(model, method_name)

        image_count = rank + 1  # source + rank crops => 1/2/3/4 images.
        grids = tuple((1, 2, 4) for _ in range(image_count))
        pixels = tuple(
            torch.randn((8, 8), device=device, dtype=torch.bfloat16)
            for _ in range(image_count)
        )

        def operation() -> _PackedReplayResult:
            # The production invariant: exactly one current-vision call per
            # trajectory, independent of how many crops that trajectory used.
            features = extract_live_qwen3_vision_feature_batch(
                model,
                pixel_values=pixels,
                image_grid_thw=grids,
            )
            if len(features) != image_count:
                raise RuntimeError("packed replay lost an image feature partition")
            loss = (
                sum(
                    image.merged_main.float().square().mean()
                    + sum(
                        branch.float().square().mean()
                        for branch in image.merged_deepstack
                    )
                    for image in features
                )
                / image_count
            )
            return _PackedReplayResult(loss=loss, image_count=image_count)

        loss, result = root_forward(operation=operation)
        if loss is not result.loss:
            raise RuntimeError("registered root changed the replay autograd anchor")
        loss.backward()

        parameters = dict(model.named_parameters())
        block_finite, block_nonzero = _local_finite_nonzero_gradient(
            parameters["model.visual.block.weight"]
        )
        merger_finite, merger_nonzero = _local_finite_nonzero_gradient(
            parameters["model.visual.merger.projection.weight"]
        )
        output_queue.put(
            (
                rank,
                result.image_count,
                model.model.visual.forward_calls,
                bool(torch.isfinite(loss).item()),
                block_finite,
                block_nonzero,
                merger_finite,
                merger_nonzero,
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def _join_with_deadline(process_context, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process_context.join(
            timeout=min(1.0, max(0.0, deadline - time.monotonic())),
            grace_period=2.0,
        ):
            return
    alive = [process.pid for process in process_context.processes if process.is_alive()]
    for process in process_context.processes:
        if process.is_alive():
            process.terminate()
    for process in process_context.processes:
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join()
    pytest.fail(
        "four-rank packed-vision replay did not finish within "
        f"{timeout_seconds:.0f}s; probable FSDP collective mismatch; "
        f"alive child pids before termination: {alive}"
    )


def test_registered_root_packs_asymmetric_crop_counts_into_one_vision_call(
    tmp_path,
) -> None:
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 4:
        pytest.skip("asymmetric packed-vision regression requires four CUDA GPUs")
    if torch.distributed.is_initialized():
        pytest.skip("asymmetric packed-vision regression owns its process group")

    world_size = 4
    context = multiprocessing.get_context("spawn")
    output_queue = context.Queue()
    process_context = torch.multiprocessing.start_processes(
        _run_asymmetric_packed_vision_replay,
        args=(
            world_size,
            str(tmp_path / "asymmetric-packed-vision-rendezvous"),
            output_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )
    _join_with_deadline(process_context, timeout_seconds=90.0)
    try:
        observed = sorted(output_queue.get(timeout=5.0) for _ in range(world_size))
    except Empty:
        pytest.fail("packed-vision workers exited without four rank results")
    finally:
        output_queue.close()
        output_queue.join_thread()

    assert observed == [
        (rank, rank + 1, 1, True, True, True, True, True) for rank in range(world_size)
    ]
