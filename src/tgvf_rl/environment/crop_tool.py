"""Execute a plain crop from immutable trajectory pixels exactly once."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropVisualState,
    TrajectorySourceVisual,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    tensor_checksum,
)
from tgvf_rl.protocol.schema import IMAGE_ZOOM_IN_TOOL_NAME, ParsedImageZoomInCall
from tgvf_rl.qwen.crop_coordinates import (
    CropCoordinateMapper,
    CropCoordinateMapping,
)


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


class CropReplayLayoutBuilder(Protocol):
    """Late-bind crop positions after the exact crop token count is known."""

    def build(
        self,
        *,
        trajectory_id: str,
        call_index: int,
        parsed_call: ParsedImageZoomInCall,
        trajectory_source_visual: TrajectorySourceVisual,
        crop_visual: CropVisualTensorBundle,
    ) -> CropReplayLayout: ...


@dataclass(frozen=True, slots=True)
class CropToolExecutionRequest:
    trajectory_id: str
    call_index: int
    parsed_call: ParsedImageZoomInCall
    trajectory_source_visual: TrajectorySourceVisual
    layout_builder: CropReplayLayoutBuilder
    model: ModelIdentity
    policy_version: PolicyVersion
    crop_processor_identity: ArtifactIdentity
    crop_layout_identity: ArtifactIdentity


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
        *,
        coordinate_mapper: CropCoordinateMapper,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> None:
        if not callable(getattr(materializer, "materialize", None)):
            raise TypeError("plain crop requires a visual materializer")
        if not isinstance(store, ObservationStore):
            raise TypeError("plain crop requires an ObservationStore")
        if not callable(getattr(coordinate_mapper, "map_crop_bbox_to_source", None)):
            raise TypeError("plain crop requires an explicit coordinate mapper")
        self.materializer = materializer
        self.store = store
        self.coordinate_mapper = coordinate_mapper
        self.processor_resized_size = processor_resized_size

    def execute(self, request: CropToolExecutionRequest) -> CropToolExecutionResult:
        _validate_request(request)
        source_ref = request.trajectory_source_visual.source_pixels
        if source_ref is None:
            raise RuntimeError(
                "plain crop requires rollout-recorded immutable source RGB"
            )
        source = self.store.resolve_verified_for_trajectory(
            source_ref,
            trajectory_id=request.trajectory_id,
        )
        source = _validate_source_rgb(source)
        if tensor_checksum(source) != source_ref.address.digest:
            raise RuntimeError("resolved source pixels changed after rollout recording")
        _verify_source_visual_ownership(
            self.store,
            request.trajectory_source_visual,
            trajectory_id=request.trajectory_id,
        )
        height, width, _ = source.shape
        mapping = self.coordinate_mapper.map_crop_bbox_to_source(
            request.parsed_call.bbox_2d,
            source_width=width,
            source_height=height,
            processor_resized_size=self.processor_resized_size,
        )
        if not isinstance(mapping, CropCoordinateMapping):
            raise TypeError("crop coordinate mapper returned an invalid mapping")
        requested = mapping.model_bbox_2d
        source_bbox = mapping.source_bbox_2d
        effective = clamp_bbox_to_image(source_bbox, width=width, height=height)
        left, top, right, bottom = effective
        crop = source[top:bottom, left:right, :].contiguous().clone()
        with torch.no_grad():
            visual = self.materializer.materialize(
                crop.clone(),
                parsed_call=request.parsed_call,
                call_index=request.call_index,
            )
        _validate_materialized_visual(visual, request.trajectory_source_visual)
        layout = request.layout_builder.build(
            trajectory_id=request.trajectory_id,
            call_index=request.call_index,
            parsed_call=request.parsed_call,
            trajectory_source_visual=request.trajectory_source_visual,
            crop_visual=visual,
        )
        _validate_layout(layout, visual, request.trajectory_source_visual)
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
        record = CropObservationRecord(
            schema_version="crop-observation-v2",
            observation_id=_observation_id(
                request,
                mapping,
                source_ref.address.digest,
                crop_pixels.address.digest,
            ),
            call_index=request.call_index,
            model=request.model,
            policy_version=request.policy_version,
            processor_identity=request.crop_processor_identity,
            layout_identity=request.crop_layout_identity,
            trajectory_id=request.trajectory_id,
            source_pixels_sha256=source_ref.address.digest,
            source_width=width,
            source_height=height,
            model_coordinate_space=mapping.coordinate_space,
            coordinate_conversion_version=mapping.conversion_version,
            coordinate_reference_width=mapping.coordinate_reference_width,
            coordinate_reference_height=mapping.coordinate_reference_height,
            model_bbox_2d=requested,
            source_bbox_2d=source_bbox,
            effective_bbox_2d=effective,
            source_visual=request.trajectory_source_visual.state,
            sequence_length=layout.sequence_length,
            original_image_positions=layout.original_image_positions,
            crop_visual=CropVisualState(
                crop_pixels=crop_pixels,
                merged_main=merged_main,
                merged_deepstack=merged_deepstack,
                image_grid_thw=visual.image_grid_thw,
                spatial_merge_size=visual.spatial_merge_size,
                positions=layout.crop_positions,
                deepstack_branch_layers=visual.deepstack_branch_layers,
                deepstack_injection_positions=layout.deepstack_injection_positions,
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


def _validate_request(request: CropToolExecutionRequest) -> None:
    if not isinstance(request, CropToolExecutionRequest):
        raise TypeError("request must be CropToolExecutionRequest")
    if not request.trajectory_id or request.call_index < 0:
        raise ValueError("plain crop trajectory/call identity is invalid")
    if not isinstance(request.parsed_call, ParsedImageZoomInCall) or (
        request.parsed_call.name != IMAGE_ZOOM_IN_TOOL_NAME
    ):
        raise ValueError("parsed call is not image_zoom_in_tool")
    if not isinstance(request.trajectory_source_visual, TrajectorySourceVisual):
        raise TypeError("plain crop requires the immutable trajectory source")
    if not callable(getattr(request.layout_builder, "build", None)):
        raise TypeError("plain crop requires a late-bound layout builder")
    if not isinstance(request.model, ModelIdentity) or not isinstance(
        request.policy_version, PolicyVersion
    ):
        raise TypeError("plain crop model/policy identities must be explicit")
    if not isinstance(
        request.crop_processor_identity, ArtifactIdentity
    ) or not isinstance(request.crop_layout_identity, ArtifactIdentity):
        raise TypeError("plain crop processor/layout identities must be explicit")


def _validate_materialized_visual(
    visual: CropVisualTensorBundle,
    source: TrajectorySourceVisual,
) -> None:
    if not isinstance(visual, CropVisualTensorBundle):
        raise TypeError("crop materializer returned the wrong visual bundle type")
    if visual.spatial_merge_size <= 0:
        raise ValueError("crop spatial merge size must be positive")
    if len(visual.image_grid_thw) != 3 or any(
        type(value) is not int or value <= 0 for value in visual.image_grid_thw
    ):
        raise ValueError("crop image grid must contain three positive integers")
    if len(visual.merged_deepstack) != len(visual.deepstack_branch_layers):
        raise ValueError("crop DeepStack tensors and layer identities differ")
    if visual.deepstack_branch_layers != source.deepstack_branch_layers:
        raise ValueError("crop DeepStack layers differ from the trajectory model")
    tensors = (visual.merged_main, *visual.merged_deepstack)
    if any(
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or tensor.ndim not in {2, 3}
        or tensor.requires_grad
        or tensor.grad_fn is not None
        for tensor in tensors
    ):
        raise ValueError("crop visual features must be detached floating tensors")


def _validate_layout(
    layout: CropReplayLayout,
    visual: CropVisualTensorBundle,
    source: TrajectorySourceVisual,
) -> None:
    if not isinstance(layout, CropReplayLayout):
        raise TypeError("crop layout builder returned the wrong layout type")
    if layout.sequence_length <= 0:
        raise ValueError("crop replay sequence length must be positive")
    if layout.original_image_positions != source.positions:
        raise ValueError("crop replay changed original-image positions")
    if len(layout.deepstack_injection_positions) != len(visual.merged_deepstack):
        raise ValueError("crop DeepStack tensors and layout positions differ")
    main_count = visual.merged_main.shape[-2]
    if main_count != len(layout.crop_positions):
        raise ValueError("crop merged feature count and positions differ")
    if any(
        tensor.shape[-2] != len(positions)
        for tensor, positions in zip(
            visual.merged_deepstack,
            layout.deepstack_injection_positions,
            strict=True,
        )
    ):
        raise ValueError("crop DeepStack feature counts and positions differ")


def _verify_source_visual_ownership(
    store: ObservationStore,
    source: TrajectorySourceVisual,
    *,
    trajectory_id: str,
) -> None:
    state = source.state
    refs = (
        state.premerge_main,
        *state.premerge_deepstack,
        state.merged_main,
        *state.merged_deepstack,
    )
    for ref in refs:
        store.resolve_verified_for_trajectory(ref, trajectory_id=trajectory_id)


def _observation_id(
    request: CropToolExecutionRequest,
    mapping: CropCoordinateMapping,
    source_digest: str,
    crop_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(request.trajectory_id.encode())
    digest.update(str(request.call_index).encode())
    digest.update(request.parsed_call.raw_tool_call.encode())
    digest.update(mapping.coordinate_space.encode())
    digest.update(mapping.conversion_version.encode())
    digest.update(str(mapping.coordinate_reference_width).encode())
    digest.update(str(mapping.coordinate_reference_height).encode())
    digest.update(str(mapping.source_bbox_2d).encode())
    digest.update(source_digest.encode())
    digest.update(crop_digest.encode())
    digest.update(request.crop_processor_identity.sha256.encode())
    digest.update(request.crop_layout_identity.sha256.encode())
    return digest.hexdigest()


__all__ = [
    "CropReplayLayout",
    "CropReplayLayoutBuilder",
    "CropToolExecutionRequest",
    "CropToolExecutionResult",
    "CropVisualMaterializer",
    "CropVisualTensorBundle",
    "ImageZoomInTool",
    "clamp_bbox_to_image",
]
