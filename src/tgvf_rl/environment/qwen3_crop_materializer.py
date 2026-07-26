"""Real Qwen3-VL crop preprocessing and frozen-vision materialization.

This module is shared by the plain crop tool and the atomic crop+TGVF tool.
It turns the exact RGB crop into the same pre-merge, merged-main, and
DeepStack tensors produced by the selected Qwen3 vision tower.  It never
reloads pixels from a path and never substitutes a different vision model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image
from torch import nn

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.protocol.native import (
    NativeProtocolRenderer,
    native_assistant_dialect_for_model,
)

from .crop_tool import CropVisualTensorBundle
from .focus_tool import SourceVisualTensorBundle


QWEN3_VISION_COMPONENT_PATH = "model.visual"
QWEN3_MAIN_MERGER_COMPONENT_PATH = "model.visual.merger"
QWEN3_DEEPSTACK_MERGER_COMPONENT_PATHS = (
    "model.visual.deepstack_merger_list.0",
    "model.visual.deepstack_merger_list.1",
    "model.visual.deepstack_merger_list.2",
)
QWEN3_DEEPSTACK_BRANCH_LAYERS = (8, 16, 24)


class Qwen3CropVisualMaterializer:
    """Materialize one exact crop with frozen Qwen3 vision components.

    Forward hooks are installed only while the vision forward is protected by
    a process-local lock.  This prevents concurrent tool calls from mixing
    merger captures.  A later batched implementation may replace the lock, but
    it must preserve the same output contract and content identities.
    """

    def __init__(
        self,
        *,
        processor: Any,
        model_identity: ModelIdentity,
        vision_tower: nn.Module,
        main_merger: nn.Module,
        deepstack_mergers: tuple[nn.Module, ...],
        image_max_pixels: int,
        branch_layers: tuple[int, ...] = QWEN3_DEEPSTACK_BRANCH_LAYERS,
    ) -> None:
        if not isinstance(model_identity, ModelIdentity):
            raise TypeError("crop materializer requires a ModelIdentity")
        if model_identity.family != "qwen3_vl":
            raise ValueError("Qwen3 crop materializer received another model family")
        if type(image_max_pixels) is not int or image_max_pixels <= 0:
            raise ValueError("image_max_pixels must be a positive integer")
        if not isinstance(vision_tower, nn.Module) or not isinstance(
            main_merger, nn.Module
        ):
            raise TypeError("Qwen3 vision tower and main merger must be modules")
        deepstack_mergers = tuple(deepstack_mergers)
        branch_layers = tuple(branch_layers)
        if branch_layers != QWEN3_DEEPSTACK_BRANCH_LAYERS:
            raise ValueError("Qwen3 crop DeepStack layers must be (8, 16, 24)")
        if len(deepstack_mergers) != len(branch_layers) or any(
            not isinstance(module, nn.Module) for module in deepstack_mergers
        ):
            raise ValueError("Qwen3 crop materializer requires three mergers")
        mergers = (main_merger, *deepstack_mergers)
        if len({id(module) for module in mergers}) != len(mergers):
            raise ValueError("Qwen3 crop merger modules must be distinct")

        renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=model_identity.tokenizer_length,
            assistant_dialect=native_assistant_dialect_for_model(
                model_identity.model_name
            ),
        )
        if renderer.chat_template_sha256 != model_identity.chat_template_sha256:
            raise ValueError("crop processor chat template differs from model identity")
        tokenizer_path = getattr(renderer.tokenizer, "name_or_path", None)
        if tokenizer_path != model_identity.revision_or_path:
            raise ValueError("crop processor tokenizer path differs from model identity")

        self.processor = processor
        self.model_identity = model_identity
        self.vision_tower = vision_tower
        self.mergers = mergers
        self.image_max_pixels = image_max_pixels
        self.branch_layers = branch_layers
        self._forward_lock = Lock()
        self._assert_frozen_vision()

    @classmethod
    def from_model(
        cls,
        *,
        model: nn.Module,
        processor: Any,
        model_identity: ModelIdentity,
        image_max_pixels: int,
        vision_component_path: str = QWEN3_VISION_COMPONENT_PATH,
        main_merger_component_path: str = QWEN3_MAIN_MERGER_COMPONENT_PATH,
        deepstack_merger_component_paths: tuple[str, ...] = (
            QWEN3_DEEPSTACK_MERGER_COMPONENT_PATHS
        ),
    ) -> "Qwen3CropVisualMaterializer":
        """Bind the accepted Qwen3 component paths from one live policy model."""

        if not isinstance(model, nn.Module):
            raise TypeError("crop materializer model must be an nn.Module")
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "qwen3_vl":
            raise ValueError("crop materializer requires a Qwen3-VL model")
        configured_path = getattr(config, "_name_or_path", None)
        if configured_path != model_identity.revision_or_path:
            raise ValueError("crop model path differs from ModelIdentity")
        if Path(model_identity.model_name).name != Path(
            model_identity.revision_or_path
        ).name:
            raise ValueError("crop model name/path identities differ")
        vision_config = getattr(config, "vision_config", None)
        raw_layers = getattr(vision_config, "deepstack_visual_indexes", None)
        if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)):
            raise TypeError("Qwen3 vision config must expose DeepStack layer indexes")
        branch_layers = tuple(int(layer) for layer in raw_layers)
        return cls(
            processor=processor,
            model_identity=model_identity,
            vision_tower=_resolve_module(model, vision_component_path),
            main_merger=_resolve_module(model, main_merger_component_path),
            deepstack_mergers=tuple(
                _resolve_module(model, path)
                for path in deepstack_merger_component_paths
            ),
            image_max_pixels=image_max_pixels,
            branch_layers=branch_layers,
        )

    def materialize(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: object,
        call_index: int,
    ) -> CropVisualTensorBundle:
        """Implement the plain crop tool's concrete visual-materializer port."""

        source = self.materialize_source_visual(
            crop_rgb,
            parsed_call=parsed_call,
            call_index=call_index,
        )
        return CropVisualTensorBundle(
            merged_main=source.merged_main,
            merged_deepstack=source.merged_deepstack,
            image_grid_thw=source.image_grid_thw,
            spatial_merge_size=source.spatial_merge_size,
            deepstack_branch_layers=self.branch_layers,
        )

    def materialize_source_visual(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: object,
        call_index: int,
    ) -> SourceVisualTensorBundle:
        """Return the crop visual source needed by atomic crop+TGVF."""

        if type(call_index) is not int or call_index < 0:
            raise ValueError("crop materializer call_index must be non-negative")
        if parsed_call is None:
            raise TypeError("crop materializer requires the exact parsed call")
        crop = _validate_crop_rgb(crop_rgb)
        self._assert_frozen_vision()
        pixel_values, grid = self._preprocess(crop)
        captures: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in self.mergers
        ]
        with self._forward_lock:
            self._assert_frozen_vision()
            handles = tuple(
                merger.register_forward_hook(
                    _capture_merger_call(captures[index]), with_kwargs=True
                )
                for index, merger in enumerate(self.mergers)
            )
            try:
                with torch.no_grad():
                    self.vision_tower(pixel_values, grid_thw=grid)
            finally:
                for handle in handles:
                    handle.remove()
            self._assert_frozen_vision()
        if any(len(rows) != 1 for rows in captures):
            raise RuntimeError(
                "each Qwen3 crop merger must execute exactly once; "
                f"observed={tuple(len(rows) for rows in captures)}"
            )
        premerge = tuple(rows[0][0] for rows in captures)
        merged = tuple(rows[0][1] for rows in captures)
        grid_tuple = tuple(int(value) for value in grid[0].tolist())
        merge_size = _spatial_merge_size(self.vision_tower)
        _validate_feature_geometry(
            grid=grid_tuple,
            merge_size=merge_size,
            premerge=premerge,
            merged=merged,
        )
        crop_sha256 = tensor_checksum(crop)
        return SourceVisualTensorBundle(
            image_sha256=crop_sha256,
            premerge_main=premerge[0],
            premerge_deepstack=premerge[1:],
            merged_main=merged[0],
            merged_deepstack=merged[1:],
            image_grid_thw=grid_tuple,
            spatial_merge_size=merge_size,
            decoded_rgb_sha256=crop_sha256,
        )

    def _preprocess(self, crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values, grid = preprocess_qwen3_rgb(
            processor=self.processor,
            rgb=crop,
            image_max_pixels=self.image_max_pixels,
        )
        owner = _first_module_tensor(self.vision_tower)
        if owner is None:
            raise ValueError("Qwen3 vision tower owns no parameter or buffer")
        return (
            pixel_values.to(device=owner.device, dtype=owner.dtype),
            grid.to(device=owner.device, dtype=torch.long),
        )

    def _assert_frozen_vision(self) -> None:
        modules = (self.vision_tower, *self.mergers)
        if any(module.training for module in modules):
            raise RuntimeError("crop materialization requires eval-mode Qwen vision")
        if any(
            parameter.requires_grad
            for module in modules
            for parameter in module.parameters()
        ):
            raise RuntimeError("crop materialization requires frozen Qwen vision")
        owner = _first_module_tensor(self.vision_tower)
        if owner is None or not owner.dtype.is_floating_point:
            raise ValueError("Qwen vision tower requires a floating owner tensor")
        for merger in self.mergers:
            merger_owner = _first_module_tensor(merger)
            if merger_owner is None or merger_owner.device != owner.device:
                raise RuntimeError("Qwen crop mergers must share the vision device")


def preprocess_qwen3_rgb(
    *,
    processor: Any,
    rgb: torch.Tensor,
    image_max_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only Qwen's CPU image processor for an exact RGB tensor.

    The colocated Policy runtime uses this boundary in its model-free
    AgentLoop process, then sends the resulting processor tensors to the
    already-resident vLLM vision tower.  It deliberately owns no model and
    performs no visual forward.
    """

    crop = _validate_crop_rgb(rgb)
    if type(image_max_pixels) is not int or image_max_pixels <= 0:
        raise ValueError("image_max_pixels must be a positive integer")
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if not isinstance(size, Mapping):
        raise TypeError("Qwen3 image processor must expose a size mapping")
    shortest_edge = size.get("shortest_edge")
    if type(shortest_edge) is not int or shortest_edge <= 0:
        raise ValueError("Qwen3 image processor shortest_edge must be positive")
    if image_max_pixels < shortest_edge:
        raise ValueError("image_max_pixels is below the processor minimum")
    image = Image.fromarray(crop.numpy(), mode="RGB")
    batch = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[image],
        padding=False,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": shortest_edge,
                "longest_edge": image_max_pixels,
            }
        },
    )
    if not isinstance(batch, Mapping):
        raise TypeError("Qwen3 crop processor output must be a mapping")
    pixel_values = batch.get("pixel_values")
    grid = batch.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(
        grid, torch.Tensor
    ):
        raise ValueError("Qwen3 crop processor omitted pixel_values/image_grid_thw")
    if pixel_values.ndim != 2 or grid.shape != (1, 3):
        raise ValueError("Qwen3 crop processor returned invalid tensor geometry")
    return pixel_values.detach().cpu().contiguous(), grid.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()


def _validate_crop_rgb(value: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
    ):
        raise ValueError("crop RGB must be uint8 [H,W,3]")
    if value.device.type != "cpu":
        raise ValueError("exact crop RGB must be materialized on CPU")
    return value.detach().contiguous().clone()


def _capture_merger_call(destination: list[tuple[torch.Tensor, torch.Tensor]]):
    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        positional = args[0] if args and isinstance(args[0], torch.Tensor) else None
        keyword = kwargs.get("hidden_states")
        if keyword is not None and not isinstance(keyword, torch.Tensor):
            raise TypeError("Qwen3 merger hidden_states must be a tensor")
        if positional is not None and keyword is not None and positional is not keyword:
            raise ValueError("Qwen3 merger exposed ambiguous hidden-state inputs")
        source = positional if positional is not None else keyword
        if not isinstance(source, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError("Qwen3 merger must expose tensor input and output")
        if source.ndim != 2 or output.ndim != 2:
            raise ValueError("Qwen3 merger boundaries must be rank-two")
        destination.append((source.detach().clone(), output.detach().clone()))

    return hook


def _validate_feature_geometry(
    *,
    grid: tuple[int, int, int],
    merge_size: int,
    premerge: tuple[torch.Tensor, ...],
    merged: tuple[torch.Tensor, ...],
) -> None:
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError("crop image grid must contain three positive values")
    if grid[1] % merge_size or grid[2] % merge_size:
        raise ValueError("crop image grid is not spatial-merge divisible")
    if len(premerge) != 4 or len(merged) != 4:
        raise ValueError("crop materialization requires main plus three branches")
    pre_count = grid[0] * grid[1] * grid[2]
    merged_count = pre_count // (merge_size**2)
    if any(tensor.shape[0] != pre_count for tensor in premerge):
        raise ValueError("crop pre-merge token count differs from image grid")
    if any(tensor.shape[0] != merged_count for tensor in merged):
        raise ValueError("crop merged token count differs from image grid")
    if any(tensor.requires_grad or tensor.grad_fn is not None for tensor in (*premerge, *merged)):
        raise RuntimeError("frozen crop vision unexpectedly retained autograd state")


def _spatial_merge_size(vision_tower: nn.Module) -> int:
    config = getattr(vision_tower, "config", None)
    value = getattr(config, "spatial_merge_size", None)
    if type(value) is not int or value <= 0:
        raise ValueError("Qwen3 vision tower lacks a positive spatial merge size")
    return value


def _first_module_tensor(module: nn.Module) -> torch.Tensor | None:
    parameter = next(module.parameters(), None)
    return parameter if parameter is not None else next(module.buffers(), None)


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if part.isdecimal():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(f"Qwen3 component path is unavailable: {path}") from error
        else:
            if not hasattr(current, part):
                raise ValueError(f"Qwen3 component path is unavailable: {path}")
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Qwen3 component path is not a module: {path}")
    return current


__all__ = [
    "QWEN3_DEEPSTACK_BRANCH_LAYERS",
    "Qwen3CropVisualMaterializer",
    "preprocess_qwen3_rgb",
]
