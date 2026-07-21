from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.crop_tool import (
    CropReplayLayout,
    CropToolExecutionRequest,
    CropVisualTensorBundle,
    ImageZoomInTool,
    clamp_bbox_to_image,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.environment.tool_registry import (
    NativeToolRuntimeRegistry,
    ToolRuntimeBinding,
)
from tgvf_rl.framework.vllm.packer import pack_qwen3_vllm_replay
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    DeepStackBranchRecord,
    FocusedObservationRecord,
    ObservationMasks,
    VisualLayout,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
    tensor_checksum,
)
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan
from tgvf_rl.qwen.base import ReplayConsumer, resolve_replay_request
from tgvf_rl.trajectories.schema import TrajectoryIdentity
from tests.support import populated_observation_store, trajectory_source_visual


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
BRANCH_LAYERS = (8, 16, 24)


def _turn(text: str) -> SampledAssistantTurn:
    ids = tuple(1000 + index for index in range(len(text)))
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(ids)
    )
    return SampledAssistantTurn(text, ids, spans)


def _crop_call(bbox: str = "[-2,1,4,8]"):
    text = (
        '<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":'
        f"{bbox}}}}}</tool_call>"
    )
    return StrictToolCallParser().parse(_turn(text))


class FixtureCropMaterializer:
    def __init__(self) -> None:
        self.received: torch.Tensor | None = None

    def materialize(self, crop_rgb, *, parsed_call, call_index):
        self.received = crop_rgb.clone()
        return CropVisualTensorBundle(
            merged_main=torch.full((1, 1, 8), 3.0),
            merged_deepstack=tuple(
                torch.full((1, 1, 8), float(index + 4)) for index in range(3)
            ),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
            deepstack_branch_layers=BRANCH_LAYERS,
        )


class FixtureCropLayoutBuilder:
    def __init__(self, materializer: FixtureCropMaterializer) -> None:
        self.materializer = materializer
        self.received: CropVisualTensorBundle | None = None
        self.calls = 0

    def build(self, *, crop_visual, **kwargs):
        assert self.materializer.received is not None
        self.received = crop_visual
        self.calls += 1
        return CropReplayLayout(
            sequence_length=16,
            original_image_positions=(1, 2, 3, 4),
            crop_positions=(6,),
            deepstack_injection_positions=((6,), (6,), (6,)),
        )


def _source() -> tuple[torch.Tensor, SourceVisualTensorBundle]:
    pixels = torch.arange(4 * 5 * 3, dtype=torch.uint8).view(4, 5, 3)
    main = torch.arange(32, dtype=torch.float32).view(1, 4, 8)
    branches = tuple(torch.full((1, 4, 8), float(index + 1)) for index in range(3))
    return pixels, SourceVisualTensorBundle(
        image_sha256=tensor_checksum(pixels),
        premerge_main=main,
        premerge_deepstack=branches,
        merged_main=main,
        merged_deepstack=branches,
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=1,
        decoded_rgb_sha256=tensor_checksum(pixels),
    )


def _execute_crop(store: ObservationStore):
    pixels, source = _source()
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA1)
    policy = PolicyVersion("run", 0, SHA0)
    materializer = FixtureCropMaterializer()
    layout_builder = FixtureCropLayoutBuilder(materializer)
    trajectory_source = record_trajectory_source_visual(
        trajectory_id="trajectory",
        source_visual=source,
        source_positions=(1, 2, 3, 4),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1, 2, 3, 4),) * 3,
        observation_store=store,
        source_rgb=pixels,
    )
    result = ImageZoomInTool(materializer, store).execute(
        CropToolExecutionRequest(
            trajectory_id="trajectory",
            call_index=0,
            parsed_call=_crop_call(),
            trajectory_source_visual=trajectory_source,
            layout_builder=layout_builder,
            model=model,
            policy_version=policy,
            crop_processor_identity=ArtifactIdentity(
                "qwen", "crop-processor", "fixture", SHA1
            ),
            crop_layout_identity=ArtifactIdentity(
                "qwen", "crop-layout", "fixture", SHA2
            ),
        )
    )
    return (
        result,
        pixels,
        trajectory_source,
        model,
        policy,
        materializer,
        layout_builder,
    )


def test_crop_tool_clamps_original_image_bbox_and_records_exact_rgb() -> None:
    store = ObservationStore()
    result, pixels, _, _, _, materializer, layout_builder = _execute_crop(store)
    expected = pixels[1:4, 0:4, :].contiguous()
    assert result.record.requested_bbox_2d == (-2, 1, 4, 8)
    assert result.record.effective_bbox_2d == (0, 1, 4, 4)
    torch.testing.assert_close(result.crop_rgb, expected, rtol=0, atol=0)
    torch.testing.assert_close(materializer.received, expected, rtol=0, atol=0)
    assert layout_builder.received is result.visual
    assert layout_builder.calls == 1
    torch.testing.assert_close(
        store.resolve_verified(result.record.crop_visual.crop_pixels),
        expected,
        rtol=0,
        atol=0,
    )
    assert (
        result.record.source_pixels_sha256
        == hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
    )
    assert result.record.processor_identity.sha256 == SHA1
    assert result.record.layout_identity.sha256 == SHA2

    restored = ObservationStore.from_checkpoint_state(store.checkpoint_state())
    restored_record = restored.resolve_record(result.handle)
    torch.testing.assert_close(
        restored.resolve_verified(restored_record.crop_visual.crop_pixels),
        expected,
        rtol=0,
        atol=0,
    )


def test_crop_rejects_empty_box_after_source_bound_clamp() -> None:
    assert clamp_bbox_to_image((-1, -1, 2, 2), width=4, height=3) == (0, 0, 2, 2)
    with pytest.raises(ValueError, match="empty after clamping"):
        clamp_bbox_to_image((8, 1, 10, 3), width=4, height=3)


def test_crop_then_tgvf_share_exact_qwen_and_vllm_replay_order() -> None:
    store = ObservationStore()
    crop, _, trajectory_source, model, policy, _, _ = _execute_crop(store)
    source = crop.record.source_visual
    sequence = 16
    common_mask = store.put_tensor(
        "mixed.common_mask", torch.ones(1, sequence, dtype=torch.bool)
    )
    position_ids = store.put_tensor(
        "mixed.position_ids", torch.arange(sequence).view(1, sequence)
    )
    main_d = store.put_tensor("mixed.call.1.main_d", torch.full((1, 4, 8), 8.0))
    d_branches = tuple(
        store.put_tensor(
            f"mixed.call.1.branch.{layer}", torch.full((1, 4, 8), float(layer))
        )
        for layer in BRANCH_LAYERS
    )
    d_positions = (8, 9, 10, 11)
    merger = ArtifactIdentity("qwen", "merger", "fixture", SHA1)
    focus_record = FocusedObservationRecord(
        schema_version="focused-observation-v1",
        observation_id="mixed-focus-1",
        call_index=1,
        model=model,
        representation=ArtifactIdentity("tgvf", "adapter", "fixture", SHA0),
        condition=ConditionProvenance(
            provider="contextual_hidden_state",
            sampled_target_text_sha256=hashlib.sha256(b"serial number").hexdigest(),
            sampled_target_token_start=1,
            sampled_target_token_end=2,
            conditioning_target_token_start=1,
            conditioning_target_token_end=2,
            source_sequence_length=2,
            source_input_ids_sha256=SHA0,
            trajectory_ids=("trajectory",),
            call_indices=(1,),
            hidden_layer=18,
            contextual_forward_identity=ArtifactIdentity(
                "policy", "contextual-forward", "fixture", SHA1
            ),
            policy_version=policy,
        ),
        source_visual=source,
        payload=TensorPayloadSet(
            main_d=main_d,
            deepstack=d_branches,
            position_ids=position_ids,
            attention_mask=common_mask,
        ),
        branches=tuple(
            DeepStackBranchRecord(layer, ref, d_positions, merger)
            for layer, ref in zip(BRANCH_LAYERS, d_branches, strict=True)
        ),
        layout=VisualLayout(
            sequence_length=sequence,
            original_image_positions=(1, 2, 3, 4),
            d_positions=d_positions,
            deepstack_branch_layers=BRANCH_LAYERS,
            deepstack_injection_positions=(d_positions, d_positions, d_positions),
        ),
        masks=ObservationMasks(common_mask, common_mask, common_mask, None),
        cache=CacheContract("no_cache", 0, None, None, True, 0.0),
    )
    focus_handle = store.put(focus_record)
    input_ids = store.put_tensor(
        "mixed.input_ids", torch.arange(sequence).view(1, sequence)
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="mixed-crop-focus",
        trajectory_id="trajectory",
        model=model,
        behavior_policy=policy,
        source_visual=trajectory_source,
        observation_handles=(crop.handle, focus_handle),
        crop_vision_replay_mode="shared_frozen_recorded_features",
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=common_mask,
            policy_attention_mask=common_mask,
            reference_attention_mask=common_mask,
            teacher_attention_mask=common_mask,
        ),
    )
    with pytest.raises(ReplayMismatchError, match="shared frozen-vision"):
        store.put_replay(replace(replay, crop_vision_replay_mode="no_crop"))
    with pytest.raises(ReplayMismatchError, match="exact immutable source pixels"):
        store.put_replay(
            replace(
                replay,
                source_visual=replace(trajectory_source, source_pixels=None),
            )
        )
    replay_handle = store.put_replay(replay)

    policy_request = resolve_replay_request(store, replay_handle, ReplayConsumer.POLICY)
    reference_request = resolve_replay_request(
        store, replay_handle, ReplayConsumer.REFERENCE
    )
    assert tuple(block.kind for block in policy_request.visual_blocks) == (
        "source_image",
        "crop_image",
        "focused_d",
    )
    assert tuple(block.positions for block in policy_request.visual_blocks) == (
        (1, 2, 3, 4),
        (6,),
        d_positions,
    )
    for policy_block, reference_block in zip(
        policy_request.visual_blocks, reference_request.visual_blocks, strict=True
    ):
        torch.testing.assert_close(
            policy_block.embeddings, reference_block.embeddings, rtol=0, atol=0
        )

    packed = pack_qwen3_vllm_replay(store, replay_handle)
    assert tuple(item.kind for item in packed.items) == (
        "source_image",
        "crop_image",
        "focused_d",
    )
    assert tuple(item.call_index for item in packed.items) == (None, 0, 1)


def test_runtime_registry_dispatches_both_tools_without_reordering() -> None:
    calls: list[tuple[str, int]] = []
    binding_store, binding_handle = populated_observation_store()
    source_binding = trajectory_source_visual(
        binding_store.resolve_record(binding_handle)
    )

    policy = PolicyVersion("run", 0, SHA0)
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA1)
    sampled_text = "reason</think>"
    sampled_ids = tuple(range(100, 100 + len(sampled_text)))
    sampled = SampledPolicyTurn(
        text=sampled_text,
        token_ids=sampled_ids,
        token_byte_spans=tuple(
            TokenByteSpan(index, token_id, index, index + 1)
            for index, token_id in enumerate(sampled_ids)
        ),
        behavior_logprobs=tuple(-0.1 for _ in sampled_ids),
        sampling=SamplingIdentity(
            policy,
            "vllm",
            "fixture",
            7,
            SHA2,
            1.0,
            1.0,
            -1,
            0.0,
            1.0,
            (),
            LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            0,
        ),
        think_token_span=TokenSpan(0, len(sampled_ids)),
        stop_reason="stop",
        backend_request_sha256=SHA1,
        backend_response_sha256=SHA2,
    )

    def context(call_index: int) -> ToolExecutionContext:
        return ToolExecutionContext(
            trajectory_identity=TrajectoryIdentity("run", "sample", 0, "group"),
            model=model,
            behavior_policy=policy,
            trajectory_source_visual=source_binding,
            prior_observation_handles=(binding_handle,)[:call_index],
            prompt_token_ids_before_turn=(1, 2),
            sampled_turn=sampled,
            assistant_turn_index=call_index,
            attempt_index=call_index,
            call_index=call_index,
        )

    def execute(parsed, execution_context):
        calls.append((parsed.name, execution_context.call_index))
        return ObservationHandle(
            f"observation-{execution_context.call_index}",
            str(execution_context.call_index) * 64,
        )

    registry = NativeToolRuntimeRegistry(
        (
            ToolRuntimeBinding("image_zoom_in_tool", execute),
            ToolRuntimeBinding("tgvf_focus_tool", execute),
        )
    )
    crop = _crop_call("[0,0,2,2]")
    focus = StrictToolCallParser().parse(
        _turn(
            '<tool_call>{"name":"tgvf_focus_tool",'
            '"arguments":{"target":"serial number"}}</tool_call>'
        )
    )
    registry.execute(crop, context(0))
    registry.execute(focus, context(1))
    assert calls == [("image_zoom_in_tool", 0), ("tgvf_focus_tool", 1)]
