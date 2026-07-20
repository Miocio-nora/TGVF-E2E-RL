"""Execute an original-image crop and materialize its exact replay state once."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropVisualState,
    SourceVisualState,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    tensor_checksum,
)
from tgvf_rl.protocol.schema import IMAGE_ZOOM_IN_TOOL_NAME, ParsedImageZoomInCall

from .focus_tool import SourceVisualTensorBundle


@dataclass(frozen=True, slots=True)
class CropVisualTensorBundle:
    """Rollout-time vision-tower output for the exact crop pixels."""

    merged_main: torch.Tensor
    merged_deepstack: tuple[torch.Tensor, ...]
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    deepstack_branch_layers: tuple[int, ...]


class CropVisualMaterializer(Protocol):
    def materialize(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: ParsedImageZoomInCall,
        call_index: int,
    ) -> CropVisualTensorBundle: ...


@dataclass(frozen=True, slots=True)
class CropReplayLayout:
    sequence_length: int
    original_image_positions: tuple[int, ...]
    crop_positions: tuple[int, ...]
    deepstack_injection_positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CropToolExecutionRequest:
    trajectory_id: str
    call_index: int
    parsed_call: ParsedImageZoomInCall
    source_rgb: torch.Tensor
    source_visual: SourceVisualTensorBundle
    layout: CropReplayLayout
    model: ModelIdentity
    policy_version: PolicyVersion


@dataclass(frozen=True, slots=True)
class CropToolExecutionResult:
    handle: ObservationHandle
    record: CropObservationRecord
    crop_rgb: torch.Tensor
    visual: CropVisualTensorBundle


class ImageZoomInTool:
    """DeepEyes-compatible public crop call with project-owned replay semantics."""

    name = IMAGE_ZOOM_IN_TOOL_NAME

    def __init__(
        self,
        materializer: CropVisualMaterializer,
        store: ObservationStore,
    ) -> None:
        self.materializer = materializer
        self.store = store

    def execute(self, request: CropToolExecutionRequest) -> CropToolExecutionResult:
        if request.call_index < 0:
            raise ValueError("call_index must be non-negative")
        if request.parsed_call.name != self.name:
            raise ValueError("parsed call is not image_zoom_in_tool")
        source = _validate_source_rgb(request.source_rgb)
        height, width, _ = source.shape
        requested = request.parsed_call.bbox_2d
        effective = clamp_bbox_to_image(requested, width=width, height=height)
        left, top, right, bottom = effective
        crop = source[top:bottom, left:right, :].contiguous().clone()
        visual = self.materializer.materialize(
            crop.clone(),
            parsed_call=request.parsed_call,
            call_index=request.call_index,
        )
        _validate_visual(visual, request.layout)

        source_state = _store_source_visual(
            self.store,
            request.source_visual,
            call_index=request.call_index,
            trajectory_id=request.trajectory_id,
        )
        crop_pixels = self.store.put_tensor(
            f"call.{request.call_index}.crop.rgb",
            crop,
            trajectory_id=request.trajectory_id,
        )
        merged_main = self.store.put_tensor(
            f"call.{request.call_index}.crop.merged.main",
            visual.merged_main,
            trajectory_id=request.trajectory_id,
        )
        merged_deepstack = tuple(
            self.store.put_tensor(
                f"call.{request.call_index}.crop.merged.deepstack.{layer}",
                tensor,
                trajectory_id=request.trajectory_id,
            )
            for layer, tensor in zip(
                visual.deepstack_branch_layers,
                visual.merged_deepstack,
                strict=True,
            )
        )
        source_digest = tensor_checksum(source)
        record = CropObservationRecord(
            schema_version="crop-observation-v1",
            observation_id=_observation_id(
                request, source_digest, crop_pixels.address.digest
            ),
            call_index=request.call_index,
            model=request.model,
            policy_version=request.policy_version,
            trajectory_id=request.trajectory_id,
            source_pixels_sha256=source_digest,
            source_width=width,
            source_height=height,
            requested_bbox_2d=requested,
            effective_bbox_2d=effective,
            source_visual=source_state,
            sequence_length=request.layout.sequence_length,
            original_image_positions=request.layout.original_image_positions,
            crop_visual=CropVisualState(
                crop_pixels=crop_pixels,
                merged_main=merged_main,
                merged_deepstack=merged_deepstack,
                image_grid_thw=visual.image_grid_thw,
                spatial_merge_size=visual.spatial_merge_size,
                positions=request.layout.crop_positions,
                deepstack_branch_layers=visual.deepstack_branch_layers,
                deepstack_injection_positions=request.layout.deepstack_injection_positions,
            ),
        )
        return CropToolExecutionResult(
            self.store.put(record), record, crop.clone(), visual
        )


def clamp_bbox_to_image(
    bbox_2d: tuple[int, int, int, int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a requested half-open bbox and reject an empty effective crop."""

    if width <= 0 or height <= 0:
        raise ValueError("source image dimensions must be positive")
    if len(bbox_2d) != 4 or any(type(value) is not int for value in bbox_2d):
        raise ValueError("bbox_2d must contain exactly four integers")
    left, top, right, bottom = bbox_2d
    if right <= left or bottom <= top:
        raise ValueError("requested bbox must be non-empty")
    effective = (
        min(max(left, 0), width),
        min(max(top, 0), height),
        min(max(right, 0), width),
        min(max(bottom, 0), height),
    )
    if effective[2] <= effective[0] or effective[3] <= effective[1]:
        raise ValueError("bbox is empty after clamping to the source image")
    return effective


def _validate_source_rgb(source: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(source, torch.Tensor)
        or source.dtype != torch.uint8
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.shape[0] <= 0
        or source.shape[1] <= 0
    ):
        raise ValueError("source image must be RGB uint8 [H,W,3]")
    return source.detach().to(device="cpu").contiguous()


def _validate_visual(visual: CropVisualTensorBundle, layout: CropReplayLayout) -> None:
    if visual.spatial_merge_size <= 0:
        raise ValueError("crop spatial merge size must be positive")
    if len(visual.merged_deepstack) != len(visual.deepstack_branch_layers):
        raise ValueError("crop DeepStack tensors and layer identities differ")
    if len(visual.merged_deepstack) != len(layout.deepstack_injection_positions):
        raise ValueError("crop DeepStack tensors and layout positions differ")
    main_count = (
        visual.merged_main.shape[-2] if visual.merged_main.ndim in {2, 3} else -1
    )
    if main_count != len(layout.crop_positions):
        raise ValueError("crop merged feature count and positions differ")


def _store_source_visual(
    store: ObservationStore,
    source: SourceVisualTensorBundle,
    *,
    call_index: int,
    trajectory_id: str,
) -> SourceVisualState:
    prefix = f"call.{call_index}.source"
    return SourceVisualState(
        image_sha256=source.image_sha256,
        premerge_main=store.put_tensor(
            f"{prefix}.premerge.main",
            source.premerge_main,
            trajectory_id=trajectory_id,
        ),
        premerge_deepstack=tuple(
            store.put_tensor(
                f"{prefix}.premerge.deepstack.{index}",
                tensor,
                trajectory_id=trajectory_id,
            )
            for index, tensor in enumerate(source.premerge_deepstack)
        ),
        merged_main=store.put_tensor(
            f"{prefix}.merged.main",
            source.merged_main,
            trajectory_id=trajectory_id,
        ),
        merged_deepstack=tuple(
            store.put_tensor(
                f"{prefix}.merged.deepstack.{index}",
                tensor,
                trajectory_id=trajectory_id,
            )
            for index, tensor in enumerate(source.merged_deepstack)
        ),
        image_grid_thw=source.image_grid_thw,
        spatial_merge_size=source.spatial_merge_size,
    )


def _observation_id(
    request: CropToolExecutionRequest,
    source_digest: str,
    crop_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(request.trajectory_id.encode())
    digest.update(str(request.call_index).encode())
    digest.update(request.parsed_call.raw_tool_call.encode())
    digest.update(source_digest.encode())
    digest.update(crop_digest.encode())
    return digest.hexdigest()
