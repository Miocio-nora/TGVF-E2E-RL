"""Atomic immutable-source crop followed by one frozen TGVF Adapter execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import torch
from torch import nn

from tgvf_rl.conditioning.base import TargetConditioningOutput
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    CropTGVFObservationRecord,
    CropTGVFVisualState,
    DeepStackBranchRecord,
    ObservationMasks,
    SourceVisualState,
    TrajectorySourceVisual,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    tensor_checksum,
)
from tgvf_rl.protocol.schema import TGVF_CROP_TOOL_NAME, ParsedCropTGVFCall
from tgvf_rl.representation.adapter import (
    TGVFAdapter,
    TGVFAdapterInput,
    TGVFAdapterOutput,
)

from .crop_tool import clamp_bbox_to_image
from .focus_tool import ReplayLayoutTensors, SourceVisualTensorBundle


class CropTGVFVisualMaterializer(Protocol):
    """Process exact crop pixels into the visual state consumed by the Adapter."""

    def materialize_source_visual(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: ParsedCropTGVFCall,
        call_index: int,
    ) -> SourceVisualTensorBundle: ...


class CropTGVFReplayLayoutBuilder(Protocol):
    """Late-bind replay positions after the crop visual token count is known."""

    def build(
        self,
        *,
        trajectory_id: str,
        call_index: int,
        parsed_call: ParsedCropTGVFCall,
        trajectory_source_visual: TrajectorySourceVisual,
        crop_visual: SourceVisualTensorBundle,
    ) -> ReplayLayoutTensors: ...


@dataclass(frozen=True, slots=True)
class CropTGVFToolExecutionRequest:
    trajectory_id: str
    call_index: int
    parsed_call: ParsedCropTGVFCall
    condition: TargetConditioningOutput
    trajectory_source_visual: TrajectorySourceVisual
    layout_builder: CropTGVFReplayLayoutBuilder
    model: ModelIdentity
    policy_version: PolicyVersion
    contextual_forward_identity: ArtifactIdentity | None
    representation: ArtifactIdentity
    branch_merger_identities: tuple[ArtifactIdentity, ...]
    crop_processor_identity: ArtifactIdentity
    crop_layout_identity: ArtifactIdentity


@dataclass(frozen=True, slots=True)
class CropTGVFToolExecutionResult:
    handle: ObservationHandle
    record: CropTGVFObservationRecord
    crop_rgb: torch.Tensor
    crop_visual: SourceVisualTensorBundle
    adapter_output: TGVFAdapterOutput


class AtomicCropTGVFTool:
    """Execute crop and TGVF as one sampled call and one immutable record."""

    name = TGVF_CROP_TOOL_NAME

    def __init__(
        self,
        *,
        materializer: CropTGVFVisualMaterializer,
        adapter: TGVFAdapter,
        store: ObservationStore,
    ) -> None:
        if not callable(getattr(materializer, "materialize_source_visual", None)):
            raise TypeError("atomic crop+TGVF requires a visual materializer")
        if not isinstance(adapter, nn.Module):
            raise TypeError("atomic crop+TGVF requires a torch Adapter module")
        if not isinstance(store, ObservationStore):
            raise TypeError("atomic crop+TGVF requires an ObservationStore")
        self.materializer = materializer
        self.adapter = adapter
        self.store = store
        _assert_frozen_adapter(adapter)

    def execute(
        self, request: CropTGVFToolExecutionRequest
    ) -> CropTGVFToolExecutionResult:
        _validate_request(request)
        _assert_frozen_adapter(self.adapter)
        source_ref = request.trajectory_source_visual.source_pixels
        if source_ref is None:
            raise RuntimeError(
                "atomic crop+TGVF requires rollout-recorded immutable source RGB"
            )
        source = self.store.resolve_verified_for_trajectory(
            source_ref,
            trajectory_id=request.trajectory_id,
        )
        _validate_source_rgb(source)
        if tensor_checksum(source) != source_ref.address.digest:
            raise RuntimeError("resolved source pixels changed after rollout recording")
        _verify_source_visual_ownership(
            self.store,
            request.trajectory_source_visual,
            trajectory_id=request.trajectory_id,
        )
        height, width, _ = source.shape
        requested = request.parsed_call.bbox_2d
        effective = clamp_bbox_to_image(requested, width=width, height=height)
        left, top, right, bottom = effective
        crop = source[top:bottom, left:right, :].contiguous().clone()

        crop_visual = self.materializer.materialize_source_visual(
            crop.clone(),
            parsed_call=request.parsed_call,
            call_index=request.call_index,
        )
        _validate_crop_visual(crop_visual, crop, request)
        layout = request.layout_builder.build(
            trajectory_id=request.trajectory_id,
            call_index=request.call_index,
            parsed_call=request.parsed_call,
            trajectory_source_visual=request.trajectory_source_visual,
            crop_visual=crop_visual,
        )
        _validate_layout(layout, request)
        with torch.no_grad():
            adapter_output = self.adapter(
                TGVFAdapterInput.from_conditioning(
                    request.condition,
                    pre_merge_visual_tokens=crop_visual.premerge_main,
                    deepstack_pre_merge_visual_tokens=crop_visual.premerge_deepstack,
                )
            )
        _validate_adapter_output(adapter_output, layout)

        def put(name: str, tensor: torch.Tensor):
            return self.store.put_tensor(
                name,
                tensor,
                trajectory_id=request.trajectory_id,
            )

        prefix = f"call.{request.call_index}.crop_tgvf"
        crop_pixels = put(f"{prefix}.crop.rgb", crop)
        crop_premerge = put(
            f"{prefix}.crop.premerge.main", crop_visual.premerge_main
        )
        crop_premerge_branches = tuple(
            put(f"{prefix}.crop.premerge.deepstack.{index}", tensor)
            for index, tensor in enumerate(crop_visual.premerge_deepstack)
        )
        crop_merged = put(f"{prefix}.crop.merged.main", crop_visual.merged_main)
        crop_merged_branches = tuple(
            put(f"{prefix}.crop.merged.deepstack.{index}", tensor)
            for index, tensor in enumerate(crop_visual.merged_deepstack)
        )
        main_d = put(f"{prefix}.main_d", adapter_output.main_d)
        d_branches = tuple(
            put(f"{prefix}.d_deepstack.{layer}", tensor)
            for layer, tensor in zip(
                adapter_output.metadata.branch_layers,
                adapter_output.deepstack_visual_embeds,
                strict=True,
            )
        )
        position_ids = put(f"{prefix}.position_ids", layout.position_ids)
        attention_mask = put(f"{prefix}.attention_mask", layout.attention_mask)
        policy_mask = put(f"{prefix}.policy_visible", layout.policy_visible_mask)
        reference_mask = put(
            f"{prefix}.reference_visible", layout.reference_visible_mask
        )
        teacher_mask = put(f"{prefix}.teacher_visible", layout.teacher_visible_mask)
        token_type_ids = (
            put(f"{prefix}.token_type_ids", layout.token_type_ids)
            if layout.token_type_ids is not None
            else None
        )
        key_block = (
            put(
                f"{prefix}.original_image_key_block",
                layout.original_image_key_block_mask,
            )
            if layout.original_image_key_block_mask is not None
            else None
        )
        cache_position = (
            put(f"{prefix}.cache_position", layout.cache_position)
            if layout.cache_position is not None
            else None
        )
        rope_delta = (
            put(f"{prefix}.rope_delta", layout.rope_delta)
            if layout.rope_delta is not None
            else None
        )

        provenance = request.condition.provenance
        condition = ConditionProvenance(
            provider=provenance.provider,
            sampled_target_text_sha256=hashlib.sha256(
                request.parsed_call.target.encode("utf-8")
            ).hexdigest(),
            sampled_target_token_start=request.parsed_call.target_span.token_start,
            sampled_target_token_end=request.parsed_call.target_span.token_end,
            conditioning_target_token_start=provenance.target_span.start,
            conditioning_target_token_end=provenance.target_span.end,
            source_sequence_length=provenance.source_sequence_length,
            source_input_ids_sha256=provenance.source_input_ids_sha256,
            trajectory_ids=provenance.trajectory_ids,
            call_indices=provenance.call_indices,
            hidden_layer=provenance.hidden_layer,
            contextual_forward_identity=request.contextual_forward_identity,
            policy_version=request.policy_version,
            embedding_identity=provenance.embedding_identity,
        )
        record = CropTGVFObservationRecord(
            schema_version="crop-tgvf-observation-v1",
            observation_id=_observation_id(request, source_ref.address.digest, crop_pixels.address.digest),
            call_index=request.call_index,
            model=request.model,
            representation=request.representation,
            condition=condition,
            source_pixels_sha256=source_ref.address.digest,
            source_width=width,
            source_height=height,
            requested_bbox_2d=requested,
            effective_bbox_2d=effective,
            sampled_target_char_span=(
                request.parsed_call.target_span.offsets.char_start,
                request.parsed_call.target_span.offsets.char_end,
            ),
            source_visual=request.trajectory_source_visual.state,
            crop_visual=CropTGVFVisualState(
                crop_pixels=crop_pixels,
                processor_identity=request.crop_processor_identity,
                layout_identity=request.crop_layout_identity,
                source=SourceVisualState(
                    image_sha256=crop_pixels.address.digest,
                    premerge_main=crop_premerge,
                    premerge_deepstack=crop_premerge_branches,
                    merged_main=crop_merged,
                    merged_deepstack=crop_merged_branches,
                    image_grid_thw=crop_visual.image_grid_thw,
                    spatial_merge_size=crop_visual.spatial_merge_size,
                    decoded_rgb_sha256=crop_pixels.address.digest,
                ),
            ),
            payload=TensorPayloadSet(
                main_d=main_d,
                deepstack=d_branches,
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
            branches=tuple(
                DeepStackBranchRecord(layer, ref, positions, merger)
                for layer, ref, positions, merger in zip(
                    adapter_output.metadata.branch_layers,
                    d_branches,
                    layout.visual_layout.deepstack_injection_positions,
                    request.branch_merger_identities,
                    strict=True,
                )
            ),
            layout=layout.visual_layout,
            masks=ObservationMasks(
                policy_mask,
                reference_mask,
                teacher_mask,
                key_block,
            ),
            cache=CacheContract(
                mode=layout.cache_mode,
                prefix_length=layout.cache_prefix_length,
                cache_position=cache_position,
                rope_delta=rope_delta,
                deterministic_forward=True,
                adapter_dropout=0.0,
            ),
        )
        handle = self.store.put(record)
        return CropTGVFToolExecutionResult(
            handle=handle,
            record=record,
            crop_rgb=crop.clone(),
            crop_visual=crop_visual,
            adapter_output=adapter_output,
        )


def _validate_request(request: CropTGVFToolExecutionRequest) -> None:
    if not isinstance(request, CropTGVFToolExecutionRequest):
        raise TypeError("request must be CropTGVFToolExecutionRequest")
    if not request.trajectory_id or request.call_index < 0:
        raise ValueError("atomic crop+TGVF trajectory/call identity is invalid")
    if request.parsed_call.name != TGVF_CROP_TOOL_NAME:
        raise ValueError("parsed call is not tgvf_crop_tool")
    provenance = request.condition.provenance
    if provenance.trajectory_ids != (request.trajectory_id,) or (
        provenance.call_indices != (request.call_index,)
    ):
        raise ValueError("atomic conditioning provenance differs from call identity")
    if provenance.target_token_ids != (request.parsed_call.target_span.token_ids,):
        raise ValueError("atomic conditioning differs from sampled target tokens")
    if provenance.model != request.model:
        raise ValueError("atomic conditioning differs from runtime model")
    if provenance.provider == "contextual_hidden_state":
        if not isinstance(request.contextual_forward_identity, ArtifactIdentity):
            raise ValueError("contextual conditioning requires exact forward identity")
    elif request.contextual_forward_identity is not None:
        raise ValueError("embedding conditioning cannot name a contextual forward")
    source = request.trajectory_source_visual
    if len(request.branch_merger_identities) != len(source.deepstack_branch_layers):
        raise ValueError("atomic branch merger identities are incomplete")
    if not callable(getattr(request.layout_builder, "build", None)):
        raise TypeError("atomic execution requires a late-bound layout builder")
    for identity in (
        request.representation,
        request.crop_processor_identity,
        request.crop_layout_identity,
        *request.branch_merger_identities,
    ):
        if not isinstance(identity, ArtifactIdentity):
            raise TypeError("atomic execution artifact identities must be explicit")


def _validate_source_rgb(source: torch.Tensor) -> None:
    if (
        source.dtype != torch.uint8
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.shape[0] <= 0
        or source.shape[1] <= 0
    ):
        raise ValueError("recorded source image must be RGB uint8 [H,W,3]")


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


def _validate_crop_visual(
    visual: SourceVisualTensorBundle,
    crop: torch.Tensor,
    request: CropTGVFToolExecutionRequest,
) -> None:
    if not isinstance(visual, SourceVisualTensorBundle):
        raise TypeError("crop materializer returned the wrong visual bundle type")
    crop_sha256 = tensor_checksum(crop)
    if visual.image_sha256 != crop_sha256:
        raise ValueError("crop visual identity differs from exact crop pixels")
    if visual.decoded_rgb_sha256 != crop_sha256:
        raise ValueError(
            "crop visual features are not bound to the exact decoded crop pixels"
        )
    branch_count = len(request.branch_merger_identities)
    if len(visual.premerge_deepstack) != branch_count or len(
        visual.merged_deepstack
    ) != branch_count:
        raise ValueError("crop visual lacks a model-supported DeepStack branch")
    if visual.spatial_merge_size <= 0 or any(
        value <= 0 for value in visual.image_grid_thw
    ):
        raise ValueError("crop visual grid/merge identity is invalid")
    tensors = (
        visual.premerge_main,
        *visual.premerge_deepstack,
        visual.merged_main,
        *visual.merged_deepstack,
    )
    if any(
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or tensor.requires_grad
        or tensor.grad_fn is not None
        for tensor in tensors
    ):
        raise ValueError("crop visual tensors must be detached floating tensors")
    premerge = (visual.premerge_main, *visual.premerge_deepstack)
    merged = (visual.merged_main, *visual.merged_deepstack)
    if any(tensor.ndim != 2 for tensor in (*premerge, *merged)):
        raise ValueError("crop visual tensors must have shape [tokens, hidden]")
    if any(tensor.shape != premerge[0].shape for tensor in premerge[1:]) or any(
        tensor.shape != merged[0].shape for tensor in merged[1:]
    ):
        raise ValueError("crop main/DeepStack feature geometries differ")
    premerge_count = int(premerge[0].shape[0])
    merge_group = visual.spatial_merge_size**2
    if (
        visual.image_grid_thw[0]
        * visual.image_grid_thw[1]
        * visual.image_grid_thw[2]
        != premerge_count
        or premerge_count % merge_group
        or int(merged[0].shape[0]) != premerge_count // merge_group
    ):
        raise ValueError("crop visual grid/premerge/merged token geometry differs")


def _validate_adapter_output(
    output: TGVFAdapterOutput,
    layout: ReplayLayoutTensors,
) -> None:
    if not isinstance(output, TGVFAdapterOutput):
        raise TypeError("TGVF Adapter returned the wrong output type")
    if output.metadata.branch_layers != layout.visual_layout.deepstack_branch_layers:
        raise ValueError("atomic Adapter branches differ from replay layout")
    if output.main_d.shape[-2] != len(layout.visual_layout.d_positions) or any(
        tensor.shape[-2] != len(positions)
        for tensor, positions in zip(
            output.deepstack_visual_embeds,
            layout.visual_layout.deepstack_injection_positions,
            strict=True,
        )
    ):
        raise ValueError("atomic Adapter token counts differ from replay positions")
    tensors = (output.main_d, *output.deepstack_visual_embeds)
    if any(tensor.requires_grad or tensor.grad_fn is not None for tensor in tensors):
        raise RuntimeError("frozen atomic Adapter built an autograd graph")


def _validate_layout(
    layout: ReplayLayoutTensors,
    request: CropTGVFToolExecutionRequest,
) -> None:
    if not isinstance(layout, ReplayLayoutTensors):
        raise TypeError("atomic late-bound layout builder returned the wrong type")
    visual = layout.visual_layout
    source = request.trajectory_source_visual
    if visual.original_image_positions != source.positions:
        raise ValueError("atomic layout differs from trajectory source positions")
    if visual.deepstack_branch_layers != source.deepstack_branch_layers:
        raise ValueError("atomic layout differs from source DeepStack layers")
    sequence = visual.sequence_length
    if layout.attention_mask.shape != (1, sequence) or (
        layout.attention_mask.dtype != torch.bool
    ):
        raise ValueError("atomic replay attention mask must be bool [1,S]")
    for name, mask in (
        ("policy", layout.policy_visible_mask),
        ("reference", layout.reference_visible_mask),
        ("teacher", layout.teacher_visible_mask),
    ):
        if mask.shape != (1, sequence) or mask.dtype != torch.bool:
            raise ValueError(f"atomic replay {name} mask must be bool [1,S]")
    position_shape = tuple(layout.position_ids.shape)
    if position_shape != (1, sequence) and not (
        len(position_shape) == 3 and position_shape[-2:] == (1, sequence)
    ):
        raise ValueError("atomic replay position IDs have the wrong shape")


def _assert_frozen_adapter(adapter: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("atomic crop+TGVF requires a frozen TGVF Adapter")
    if any(module.training for module in adapter.modules()):
        raise RuntimeError("atomic crop+TGVF requires an eval-mode TGVF Adapter")


def _observation_id(
    request: CropTGVFToolExecutionRequest,
    source_digest: str,
    crop_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(request.trajectory_id.encode("utf-8"))
    digest.update(str(request.call_index).encode("ascii"))
    digest.update(request.parsed_call.sampled_text.encode("utf-8"))
    digest.update(source_digest.encode("ascii"))
    digest.update(crop_digest.encode("ascii"))
    digest.update(request.representation.sha256.encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "AtomicCropTGVFTool",
    "CropTGVFToolExecutionRequest",
    "CropTGVFToolExecutionResult",
    "CropTGVFReplayLayoutBuilder",
    "CropTGVFVisualMaterializer",
]
