from __future__ import annotations

import hashlib

import pytest
import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.environment.crop_tool import (
    CropReplayLayout,
    CropToolExecutionRequest,
    CropVisualTensorBundle,
    ImageZoomInTool,
    clamp_bbox_to_image,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
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
)
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan
from tgvf_rl.qwen.base import ReplayConsumer, resolve_replay_request


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


def _source() -> tuple[torch.Tensor, SourceVisualTensorBundle]:
    pixels = torch.arange(4 * 5 * 3, dtype=torch.uint8).view(4, 5, 3)
    main = torch.arange(32, dtype=torch.float32).view(1, 4, 8)
    branches = tuple(torch.full((1, 4, 8), float(index + 1)) for index in range(3))
    return pixels, SourceVisualTensorBundle(
        image_sha256=SHA2,
        premerge_main=main,
        premerge_deepstack=branches,
        merged_main=main,
        merged_deepstack=branches,
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=1,
    )


def _execute_crop(store: ObservationStore):
    pixels, source = _source()
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA1)
    policy = PolicyVersion("run", 0, SHA0)
    materializer = FixtureCropMaterializer()
    result = ImageZoomInTool(materializer, store).execute(
        CropToolExecutionRequest(
            trajectory_id="trajectory",
            call_index=0,
            parsed_call=_crop_call(),
            source_rgb=pixels,
            source_visual=source,
            layout=CropReplayLayout(
                sequence_length=16,
                original_image_positions=(1, 2, 3, 4),
                crop_positions=(6,),
                deepstack_injection_positions=((6,), (6,), (6,)),
            ),
            model=model,
            policy_version=policy,
        )
    )
    return result, pixels, source, model, policy, materializer


def test_crop_tool_clamps_original_image_bbox_and_records_exact_rgb() -> None:
    store = ObservationStore()
    result, pixels, _, _, _, materializer = _execute_crop(store)
    expected = pixels[1:4, 0:4, :].contiguous()
    assert result.record.requested_bbox_2d == (-2, 1, 4, 8)
    assert result.record.effective_bbox_2d == (0, 1, 4, 4)
    torch.testing.assert_close(result.crop_rgb, expected, rtol=0, atol=0)
    torch.testing.assert_close(materializer.received, expected, rtol=0, atol=0)
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
    crop, _, _, model, policy, _ = _execute_crop(store)
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
            source_input_ids_sha256=SHA0,
            trajectory_ids=("trajectory",),
            call_indices=(1,),
            hidden_layer=18,
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
        observation_handles=(crop.handle, focus_handle),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=common_mask,
            policy_attention_mask=common_mask,
            reference_attention_mask=common_mask,
            teacher_attention_mask=common_mask,
        ),
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

    def execute(parsed, call_index):
        calls.append((parsed.name, call_index))
        return ObservationHandle(f"observation-{call_index}", str(call_index) * 64)

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
    registry.execute(crop, 0)
    registry.execute(focus, 1)
    assert calls == [("image_zoom_in_tool", 0), ("tgvf_focus_tool", 1)]
