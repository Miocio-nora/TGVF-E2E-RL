"""Execute TGVF and materialize the exact replay observation once."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from tgvf_rl.conditioning.base import TargetConditioningOutput
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    DeepStackBranchRecord,
    FocusedObservationRecord,
    ObservationMasks,
    SourceVisualState,
    VisualLayout,
)
from tgvf_rl.observations.store import ObservationHandle, ObservationStore
from tgvf_rl.protocol.schema import ParsedToolCall, TGVF_FOCUS_TOOL_NAME
from tgvf_rl.representation.adapter import (
    TGVFAdapter,
    TGVFAdapterInput,
    TGVFAdapterOutput,
)


@dataclass(frozen=True, slots=True)
class SourceVisualTensorBundle:
    image_sha256: str
    premerge_main: torch.Tensor
    premerge_deepstack: tuple[torch.Tensor, ...]
    merged_main: torch.Tensor
    merged_deepstack: tuple[torch.Tensor, ...]
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    decoded_rgb_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayLayoutTensors:
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    policy_visible_mask: torch.Tensor
    reference_visible_mask: torch.Tensor
    teacher_visible_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    original_image_key_block_mask: torch.Tensor | None
    cache_position: torch.Tensor | None
    rope_delta: torch.Tensor | None
    visual_layout: VisualLayout
    cache_mode: str = "no_cache"
    cache_prefix_length: int = 0


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    trajectory_id: str
    call_index: int
    parsed_call: ParsedToolCall
    condition: TargetConditioningOutput
    source_visual: SourceVisualTensorBundle
    layout: ReplayLayoutTensors
    model: ModelIdentity
    policy_version: PolicyVersion
    contextual_forward_identity: ArtifactIdentity | None
    representation: ArtifactIdentity
    branch_merger_identities: tuple[ArtifactIdentity, ...]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    handle: ObservationHandle
    record: FocusedObservationRecord
    adapter_output: TGVFAdapterOutput


class TGVFFocusTool:
    name = TGVF_FOCUS_TOOL_NAME

    def __init__(self, adapter: TGVFAdapter, store: ObservationStore) -> None:
        self.adapter = adapter
        self.store = store

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        if request.call_index < 0:
            raise ValueError("call_index must be non-negative")
        if request.parsed_call.name != self.name:
            raise ValueError("parsed call is not tgvf_focus_tool")
        provenance = request.condition.provenance
        if tuple(provenance.trajectory_ids) != (request.trajectory_id,):
            raise ValueError("conditioning provenance differs from trajectory")
        if tuple(provenance.call_indices) != (request.call_index,):
            raise ValueError("conditioning provenance differs from call index")
        if provenance.target_token_ids != (request.parsed_call.target_span.token_ids,):
            raise ValueError(
                "conditioning provenance differs from exact sampled target tokens"
            )
        if provenance.model != request.model:
            raise ValueError("conditioning provenance differs from runtime model")
        if provenance.provider == "contextual_hidden_state":
            if not isinstance(request.contextual_forward_identity, ArtifactIdentity):
                raise ValueError(
                    "contextual conditioning requires its exact forward identity"
                )
        elif request.contextual_forward_identity is not None:
            raise ValueError(
                "target embedding conditioning cannot name a contextual forward"
            )
        source = request.source_visual
        if len(source.premerge_deepstack) != len(request.branch_merger_identities):
            raise ValueError("source DeepStack and merger identities differ")
        if len(source.merged_deepstack) != len(source.premerge_deepstack):
            raise ValueError("source merged/pre-merge DeepStack branches differ")

        adapter_output = self.adapter(
            TGVFAdapterInput.from_conditioning(
                request.condition,
                pre_merge_visual_tokens=source.premerge_main,
                deepstack_pre_merge_visual_tokens=source.premerge_deepstack,
            )
        )
        if (
            adapter_output.metadata.branch_layers
            != request.layout.visual_layout.deepstack_branch_layers
        ):
            raise ValueError("adapter and replay layout DeepStack layers differ")

        def put(name: str, tensor: torch.Tensor):
            return self.store.put_tensor(
                name, tensor, trajectory_id=request.trajectory_id
            )

        source_premerge = put(
            f"call.{request.call_index}.source.premerge.main", source.premerge_main
        )
        source_premerge_branches = tuple(
            put(f"call.{request.call_index}.source.premerge.deepstack.{index}", tensor)
            for index, tensor in enumerate(source.premerge_deepstack)
        )
        source_merged = put(
            f"call.{request.call_index}.source.merged.main", source.merged_main
        )
        source_merged_branches = tuple(
            put(f"call.{request.call_index}.source.merged.deepstack.{index}", tensor)
            for index, tensor in enumerate(source.merged_deepstack)
        )
        main_d = put(f"call.{request.call_index}.main_d", adapter_output.main_d)
        d_branches = tuple(
            put(f"call.{request.call_index}.d_deepstack.{layer}", tensor)
            for layer, tensor in zip(
                adapter_output.metadata.branch_layers,
                adapter_output.deepstack_visual_embeds,
                strict=True,
            )
        )
        layout = request.layout
        position_ids = put(
            f"call.{request.call_index}.position_ids", layout.position_ids
        )
        attention_mask = put(
            f"call.{request.call_index}.attention_mask", layout.attention_mask
        )
        policy_mask = put(
            f"call.{request.call_index}.policy_visible", layout.policy_visible_mask
        )
        reference_mask = put(
            f"call.{request.call_index}.reference_visible",
            layout.reference_visible_mask,
        )
        teacher_mask = put(
            f"call.{request.call_index}.teacher_visible", layout.teacher_visible_mask
        )
        token_type_ids = (
            put(f"call.{request.call_index}.token_type_ids", layout.token_type_ids)
            if layout.token_type_ids is not None
            else None
        )
        key_block = (
            put(
                f"call.{request.call_index}.original_image_key_block",
                layout.original_image_key_block_mask,
            )
            if layout.original_image_key_block_mask is not None
            else None
        )
        cache_position = (
            put(f"call.{request.call_index}.cache_position", layout.cache_position)
            if layout.cache_position is not None
            else None
        )
        rope_delta = (
            put(f"call.{request.call_index}.rope_delta", layout.rope_delta)
            if layout.rope_delta is not None
            else None
        )

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
        record = FocusedObservationRecord(
            schema_version="focused-observation-v1",
            observation_id=_observation_id(request, adapter_output),
            call_index=request.call_index,
            model=request.model,
            representation=request.representation,
            condition=condition,
            source_visual=SourceVisualState(
                image_sha256=source.image_sha256,
                premerge_main=source_premerge,
                premerge_deepstack=source_premerge_branches,
                merged_main=source_merged,
                merged_deepstack=source_merged_branches,
                image_grid_thw=source.image_grid_thw,
                spatial_merge_size=source.spatial_merge_size,
                decoded_rgb_sha256=source.decoded_rgb_sha256,
            ),
            payload=TensorPayloadSet(
                main_d=main_d,
                deepstack=d_branches,
                position_ids=position_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ),
            branches=tuple(
                DeepStackBranchRecord(layer, tensor_ref, positions, merger)
                for layer, tensor_ref, positions, merger in zip(
                    adapter_output.metadata.branch_layers,
                    d_branches,
                    layout.visual_layout.deepstack_injection_positions,
                    request.branch_merger_identities,
                    strict=True,
                )
            ),
            layout=layout.visual_layout,
            masks=ObservationMasks(
                policy_mask, reference_mask, teacher_mask, key_block
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
        return ToolExecutionResult(self.store.put(record), record, adapter_output)


def _observation_id(request: ToolExecutionRequest, output: TGVFAdapterOutput) -> str:
    digest = hashlib.sha256()
    digest.update(request.trajectory_id.encode())
    digest.update(str(request.call_index).encode())
    digest.update(request.parsed_call.sampled_text.encode())
    digest.update(request.condition.provenance.source_input_ids_sha256.encode())
    digest.update(output.metadata.schema_version.encode())
    return digest.hexdigest()
