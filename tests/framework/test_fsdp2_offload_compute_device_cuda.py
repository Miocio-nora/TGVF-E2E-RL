from __future__ import annotations

import multiprocessing
from types import MethodType

import pytest
import torch
from torch import nn

from tgvf_rl.policy.trainable_tgvf_replay import (
    extract_live_qwen3_vision_features,
)
from tgvf_rl.qwen.base import InjectedForwardRequest, materialize_inputs_embeds
from tgvf_rl.representation.deepstack import TrainableBorrowedProjectionPort
from tgvf_rl.tensor_device import tensor_compute_device


class _OffloadedLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens


class _OffloadedMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, 4, 4).mean(dim=1))


class _OffloadedVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_merge_size = 2
        self.stem = nn.Linear(4, 4, bias=False)
        self.merger = _OffloadedMerger()
        self.deepstack_merger_list = nn.ModuleList(
            _OffloadedMerger() for _ in range(3)
        )

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if tuple(grid_thw.shape) != (1, 3):
            raise ValueError("grid_thw lost its accelerator placement")
        hidden = self.stem(pixel_values)
        main = self.merger(hidden)
        branches = tuple(
            merger(hidden * (index + 2))
            for index, merger in enumerate(self.deepstack_merger_list)
        )
        return main, branches


class _OffloadedReplayRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _OffloadedLanguage()
        self.model.visual = _OffloadedVisual()
        self.tgvf_adapter = nn.Linear(4, 4, bias=False)

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise AssertionError("the focused replay probe must use the registered root")


def _dispatch_probe_root(self: nn.Module, *, operation) -> torch.Tensor:
    del self
    return operation()


def _run_offloaded_compute_device_probe(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_queue,
) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
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
    )
    try:
        torch.manual_seed(31)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        model = _OffloadedReplayRoot().to(device=torch.device("cuda", rank))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": policy,
            "offload_policy": CPUOffloadPolicy(pin_memory=True),
            "reshard_after_forward": True,
        }
        language = model.model.language_model
        visual = model.model.visual
        fully_shard(language.embed_tokens, **fsdp_kwargs)
        fully_shard(visual.stem, **fsdp_kwargs)
        for merger in (visual.merger, *tuple(visual.deepstack_merger_list)):
            fully_shard(merger, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        offloaded_before = (
            language.embed_tokens.weight.device.type == "cpu"
            and visual.stem.weight.device.type == "cpu"
            and visual.merger.projection.weight.device.type == "cpu"
        )
        request = InjectedForwardRequest(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
            position_ids=torch.arange(4).view(1, 4),
            visual_blocks=(),
        )
        pixels = torch.randn(8, 4)
        hq = torch.randn(2, 4)
        borrowed = TrainableBorrowedProjectionPort(
            visual.merger,
            identity="probe.visual.merger",
            input_dim=4,
            output_dim=4,
            spatial_merge_size=2,
        )

        method_name = "_tgvf_offloaded_compute_device_probe"
        setattr(model, method_name, MethodType(_dispatch_probe_root, model))
        register_fsdp_forward_method(model, method_name)
        root_forward = getattr(model, method_name)

        def operation() -> torch.Tensor:
            device_type = tensor_compute_device(next(model.parameters())).type
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                embeddings, _ = materialize_inputs_embeds(model, request)
                features = extract_live_qwen3_vision_features(
                    model,
                    pixel_values=pixels,
                    image_grid_thw=(1, 2, 4),
                )
                projected = borrowed(features.premerge_main)
                adapter_owner = next(model.tgvf_adapter.parameters())
                adapted = model.tgvf_adapter(
                    hq.to(
                        device=tensor_compute_device(adapter_owner),
                        dtype=adapter_owner.dtype,
                    )
                )
                return (
                    embeddings.sum()
                    + features.merged_main.sum()
                    + projected.sum()
                    + adapted.sum()
                )

        loss = root_forward(operation=operation)
        loss.backward()
        parameters = dict(model.named_parameters())
        output_queue.put(
            (
                rank,
                offloaded_before,
                loss.device == torch.device("cuda", rank),
                request.input_ids.device.type == "cpu",
                parameters["model.language_model.embed_tokens.weight"].grad
                is not None,
                parameters["model.visual.stem.weight"].grad is not None,
                parameters["model.visual.merger.projection.weight"].grad
                is not None,
                parameters["tgvf_adapter.weight"].grad is not None,
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def test_four_rank_fsdp2_cpu_offload_uses_mesh_compute_device(tmp_path) -> None:
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 4:
        pytest.skip("four-rank FSDP2 CPU-offload regression requires four CUDA GPUs")
    if torch.distributed.is_initialized():
        pytest.skip("four-rank FSDP2 regression requires process-group ownership")

    world_size = 4
    context = multiprocessing.get_context("spawn")
    output_queue = context.SimpleQueue()
    torch.multiprocessing.start_processes(
        _run_offloaded_compute_device_probe,
        args=(
            world_size,
            str(tmp_path / "fsdp2-offload-compute-device-rendezvous"),
            output_queue,
        ),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    assert sorted(output_queue.get() for _ in range(world_size)) == [
        (rank, True, True, True, True, True, True, True)
        for rank in range(world_size)
    ]
