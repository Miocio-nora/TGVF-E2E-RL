from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    ContextualHiddenStateConditionProvider,
    TargetConditioningRequest,
)
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.environment.crop_tgvf_tool import (
    AtomicCropTGVFTool,
    CropTGVFToolExecutionRequest,
)
from tgvf_rl.environment.focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
)
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.schema import VisualLayout
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
    tensor_checksum,
)
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.deepstack import FrozenProjectionPort


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
TRAJECTORY_ID = "run/sample/0/group"


class Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 8, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens.reshape(-1, 16))


def _parsed_call():
    text = (
        '<tool_call>{"name":"crop_tgvf_tool","arguments":'
        '{"bbox_2d":[-1,1,4,8],"target":"red label"}}</tool_call>'
    )
    ids = tuple(ord(char) for char in text)
    spans = tuple(
        TokenByteSpan(index, token, index, index + 1)
        for index, token in enumerate(ids)
    )
    return StrictToolCallParser().parse(SampledAssistantTurn(text, ids, spans))


class Materializer:
    def __init__(self) -> None:
        self.received: list[torch.Tensor] = []

    def materialize_source_visual(self, crop_rgb, *, parsed_call, call_index):
        assert parsed_call.name == "crop_tgvf_tool"
        assert call_index == 0
        self.received.append(crop_rgb.clone())
        return SourceVisualTensorBundle(
            image_sha256=tensor_checksum(crop_rgb),
            premerge_main=torch.arange(16, dtype=torch.float32).view(4, 4),
            premerge_deepstack=(torch.full((4, 4), 2.0),),
            merged_main=torch.full((1, 8), 3.0),
            merged_deepstack=(torch.full((1, 8), 4.0),),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
            decoded_rgb_sha256=tensor_checksum(crop_rgb),
        )


class LayoutBuilder:
    def __init__(self) -> None:
        self.crop_token_counts: list[int] = []

    def build(
        self,
        *,
        trajectory_id,
        call_index,
        parsed_call,
        trajectory_source_visual,
        crop_visual,
    ):
        assert trajectory_id == TRAJECTORY_ID
        assert call_index == 0
        assert parsed_call.target == "red label"
        assert trajectory_source_visual.positions == (1,)
        self.crop_token_counts.append(crop_visual.premerge_main.shape[0])
        mask = torch.ones(1, 10, dtype=torch.bool)
        return ReplayLayoutTensors(
            position_ids=torch.arange(10).view(1, 10),
            attention_mask=mask,
            policy_visible_mask=mask,
            reference_visible_mask=mask,
            teacher_visible_mask=mask,
            token_type_ids=None,
            original_image_key_block_mask=None,
            cache_position=None,
            rope_delta=None,
            visual_layout=VisualLayout(
                sequence_length=10,
                original_image_positions=(1,),
                d_positions=(6,),
                deepstack_branch_layers=(8,),
                deepstack_injection_positions=((6,),),
            ),
        )


def _fixture(*, record_pixels: bool = True):
    torch.manual_seed(3)
    store = ObservationStore()
    pixels = torch.arange(4 * 5 * 3, dtype=torch.uint8).view(4, 5, 3)
    source = SourceVisualTensorBundle(
        image_sha256=SHA2,
        premerge_main=torch.randn(4, 4),
        premerge_deepstack=(torch.randn(4, 4),),
        merged_main=torch.randn(1, 8),
        merged_deepstack=(torch.randn(1, 8),),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
        decoded_rgb_sha256=tensor_checksum(pixels),
    )
    trajectory_source = record_trajectory_source_visual(
        trajectory_id=TRAJECTORY_ID,
        source_visual=source,
        source_positions=(1,),
        deepstack_branch_layers=(8,),
        deepstack_injection_positions=((1,),),
        observation_store=store,
        source_rgb=pixels if record_pixels else None,
    )
    parsed = _parsed_call()
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA2)
    input_ids = torch.tensor(parsed.sampled_token_ids)
    condition = ContextualHiddenStateConditionProvider(
        model_identity=model,
        hidden_layer=2,
    ).build(
        TargetConditioningRequest(
            input_ids=input_ids,
            target_span=TokenSpan(
                parsed.target_span.token_start,
                parsed.target_span.token_end,
            ),
            expected_target_token_ids=parsed.target_span.token_ids,
            trajectory_id=TRAJECTORY_ID,
            call_index=0,
            model_identity=model,
            contextual_hidden_states=torch.randn(len(input_ids), 8),
        )
    )
    adapter = TGVFAdapter(
        d_lm=8,
        d_v=4,
        main_projection=FrozenProjectionPort(
            Merger(),
            identity="main",
            input_dim=4,
            output_dim=8,
            spatial_merge_size=2,
        ),
        deepstack_projections=(
            FrozenProjectionPort(
                Merger(),
                identity="branch8",
                input_dim=4,
                output_dim=8,
                spatial_merge_size=2,
            ),
        ),
        branch_layers=(8,),
    )
    adapter.requires_grad_(False).eval()
    materializer = Materializer()
    layout_builder = LayoutBuilder()
    policy = PolicyVersion("run", 0, SHA0)
    request = CropTGVFToolExecutionRequest(
        trajectory_id=TRAJECTORY_ID,
        call_index=0,
        parsed_call=parsed,
        condition=condition,
        trajectory_source_visual=trajectory_source,
        layout_builder=layout_builder,
        model=model,
        policy_version=policy,
        contextual_forward_identity=ArtifactIdentity(
            "policy", "contextual-forward", "fixture", SHA1
        ),
        representation=ArtifactIdentity("tgvf", "adapter", "fixture", SHA0),
        branch_merger_identities=(
            ArtifactIdentity("qwen", "merger-8", "fixture", SHA1),
        ),
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity(
            "qwen", "crop-layout", "fixture", SHA2
        ),
    )
    return store, pixels, trajectory_source, materializer, layout_builder, adapter, request


def test_atomic_crop_tgvf_materializes_one_complete_exact_record() -> None:
    store, pixels, source, materializer, layout_builder, adapter, request = _fixture()
    result = AtomicCropTGVFTool(
        materializer=materializer,
        adapter=adapter,
        store=store,
    ).execute(request)

    expected_crop = pixels[1:4, 0:4, :].contiguous()
    assert result.record.requested_bbox_2d == (-1, 1, 4, 8)
    assert result.record.effective_bbox_2d == (0, 1, 4, 4)
    assert len(materializer.received) == 1
    assert layout_builder.crop_token_counts == [4]
    torch.testing.assert_close(materializer.received[0], expected_crop, rtol=0, atol=0)
    torch.testing.assert_close(
        store.resolve_verified(result.record.crop_visual.crop_pixels),
        expected_crop,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        store.resolve_verified(result.record.crop_visual.source.premerge_main),
        result.crop_visual.premerge_main,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        store.resolve_verified(result.record.payload.main_d),
        result.adapter_output.main_d,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        store.resolve_verified(source.source_pixels), pixels, rtol=0, atol=0
    )

    mask = result.record.payload.attention_mask
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="atomic-replay",
        trajectory_id=TRAJECTORY_ID,
        model=request.model,
        behavior_policy=request.policy_version,
        source_visual=source,
        observation_handles=(result.handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=store.put_tensor(
                "replay.input_ids",
                torch.arange(10).view(1, 10),
                trajectory_id=TRAJECTORY_ID,
            ),
            position_ids=result.record.payload.position_ids,
            attention_mask=mask,
            policy_attention_mask=result.record.masks.policy_visible,
            reference_attention_mask=result.record.masks.reference_visible,
            teacher_attention_mask=result.record.masks.teacher_visible,
        ),
        crop_vision_replay_mode="shared_frozen_recorded_features",
    )
    replay_handle = store.put_replay(replay)
    bundle = store.export_replay_bundle(replay_handle)
    restored, restored_handle = ObservationStore.from_replay_bundle(bundle)
    restored_record = restored.resolve_record(
        restored.resolve_replay(restored_handle).observation_handles[0]
    )
    torch.testing.assert_close(
        restored.resolve_verified(restored_record.crop_visual.crop_pixels),
        expected_crop,
        rtol=0,
        atol=0,
    )

    with pytest.raises(ReplayMismatchError, match="exact immutable source pixels"):
        store.put_replay(replace(replay, source_visual=replace(source, source_pixels=None)))


def test_atomic_crop_tgvf_refuses_external_source_pixel_reconstruction() -> None:
    store, _, _, materializer, _, adapter, request = _fixture(record_pixels=False)
    with pytest.raises(RuntimeError, match="rollout-recorded immutable source RGB"):
        AtomicCropTGVFTool(
            materializer=materializer,
            adapter=adapter,
            store=store,
        ).execute(request)
    assert materializer.received == []
