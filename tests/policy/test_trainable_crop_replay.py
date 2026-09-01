from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.environment.test_crop_runtime import (
    BRANCH_LAYERS,
    SHA1,
    SHA2,
    _LayoutBuilder,
    _Materializer,
    _context,
    _crop_contract,
    _model,
    _parsed_call,
)
from tests.qwen.test_family_contract import TinyLanguageModel, _replay
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.environment.crop_runtime import (
    CropExecutionLedger,
    ImageZoomInToolRuntime,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
    tensor_checksum,
)
from tgvf_rl.policy.trainable_crop_replay import (
    TrainableCropCurrentReplayPort,
    build_trainable_crop_current_request,
)
from tgvf_rl.framework.verl.exact_replay_engine import ExactReplayResponseRequest
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    ReplayConsumer,
    resolve_replay_request,
)
from tgvf_rl.qwen.crop_coordinates import (
    CanonicalSourcePixelCropCoordinateMapper,
)


class _ToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grouped = hidden_states.reshape(-1, 4, 4).mean(dim=1)
        return self.projection(grouped)


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_merge_size = 2
        self.stem = nn.Linear(3, 4, bias=False)
        self.merger = _ToyMerger()
        self.deepstack_merger_list = nn.ModuleList(_ToyMerger() for _ in range(3))
        self.seen_pixels: list[torch.Tensor] = []
        self.seen_grids: list[torch.Tensor] = []

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        self.seen_pixels.append(pixel_values.detach().cpu().clone())
        self.seen_grids.append(grid_thw.detach().cpu().clone())
        hidden = self.stem(pixel_values)
        outputs = [self.merger(hidden)]
        outputs.extend(
            merger(hidden * (index + 2))
            for index, merger in enumerate(self.deepstack_merger_list)
        )
        return outputs[0], tuple(outputs[1:])


class _ToyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _ToyVisual()


class _CropReplayQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"tgvf_native_deepstack_enabled": True})()
        self.model = nn.Module()
        self.model.language_model = TinyLanguageModel()
        self.model.visual = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 32, bias=False)


def _bundle_with_sequence(sequence: int):
    store, handle = _replay(branches=3, calls=0)
    replay = store.resolve_replay(handle)
    trajectory_id = replay.trajectory_id
    ids = store.put_tensor(
        f"crop-batch.ids.{sequence}",
        torch.arange(1, sequence + 1, dtype=torch.long).view(1, sequence),
        trajectory_id=trajectory_id,
    )
    positions = store.put_tensor(
        f"crop-batch.positions.{sequence}",
        torch.arange(sequence, dtype=torch.long).view(1, sequence),
        trajectory_id=trajectory_id,
    )
    mask = store.put_tensor(
        f"crop-batch.mask.{sequence}",
        torch.ones(1, sequence, dtype=torch.bool),
        trajectory_id=trajectory_id,
    )
    tensors = replace(
        replay.tensors,
        input_ids=ids,
        position_ids=positions,
        attention_mask=mask,
        policy_attention_mask=mask,
        reference_attention_mask=mask,
        teacher_attention_mask=mask,
    )
    varied = replace(replay, replay_id=f"crop-batch-{sequence}", tensors=tensors)
    return store.export_replay_bundle(store.put_replay(varied))


def _sampling(bundle) -> SamplingIdentity:
    return SamplingIdentity(
        policy_version=bundle.replay_record.behavior_policy,
        backend="vllm",
        backend_version="decoder-batch",
        seed=17,
        rng_state_sha256="0" * 64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )


def _live_crop_request(*, model, store, replay_handle) -> InjectedForwardRequest:
    recorded = resolve_replay_request(store, replay_handle, ReplayConsumer.POLICY)
    blocks = tuple(
        InjectedVisualBlock(
            kind=block.kind,
            positions=block.positions,
            embeddings=model.model.visual(block.embeddings),
            deepstack=tuple(model.model.visual(branch) for branch in block.deepstack),
            deepstack_positions=block.deepstack_positions,
        )
        for block in recorded.visual_blocks
    )
    return InjectedForwardRequest(
        input_ids=recorded.input_ids,
        attention_mask=recorded.attention_mask,
        position_ids=recorded.position_ids,
        visual_blocks=blocks,
    )


def _crop_replay(*, legacy_crop_pixels: bool = False):
    model_identity = _model()
    store = ObservationStore()
    trajectory_id = "run/sample/0/group"
    source_rgb = torch.arange(4 * 5 * 3, dtype=torch.uint8).reshape(4, 5, 3)
    source_main = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    source_premerge = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    source = record_trajectory_source_visual(
        trajectory_id=trajectory_id,
        source_visual=SourceVisualTensorBundle(
            image_sha256=SHA2,
            premerge_main=source_premerge,
            premerge_deepstack=tuple(source_premerge + index for index in range(3)),
            merged_main=source_main,
            merged_deepstack=tuple(source_main + index for index in range(3)),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
            decoded_rgb_sha256=tensor_checksum(source_rgb),
        ),
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,),) * 3,
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
        source_rgb=source_rgb,
    )
    trace: list[str] = []
    runtime = ImageZoomInToolRuntime(
        model=model_identity,
        materializer=_Materializer(model_identity, trace),
        layout_builder=_LayoutBuilder(trace),
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=CropExecutionLedger(),
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        observation_contract=_crop_contract(),
    )
    parsed = _parsed_call()
    context = _context(parsed_call=parsed, source=source, model=model_identity)
    observation_handle = runtime.execute(parsed, context)
    record = store.resolve_record(observation_handle)
    if legacy_crop_pixels:
        legacy = replace(
            record,
            observation_id=f"{record.observation_id}-legacy",
            crop_visual=replace(
                record.crop_visual,
                preprocessed_pixel_values=None,
            ),
        )
        observation_handle = store.put(legacy)
        record = legacy
    sequence = record.sequence_length
    attention = store.put_tensor(
        "crop-current.attention",
        torch.ones((1, sequence), dtype=torch.bool),
        trajectory_id=context.trajectory_identity.canonical_id,
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id=(
            "crop-current-replay-legacy"
            if legacy_crop_pixels
            else "crop-current-replay"
        ),
        trajectory_id=context.trajectory_identity.canonical_id,
        model=context.model,
        behavior_policy=context.behavior_policy,
        source_visual=context.trajectory_source_visual,
        observation_handles=(observation_handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=store.put_tensor(
                "crop-current.input_ids",
                torch.arange(sequence, dtype=torch.long).view(1, sequence),
                trajectory_id=context.trajectory_identity.canonical_id,
            ),
            position_ids=store.put_tensor(
                "crop-current.position_ids",
                torch.arange(sequence, dtype=torch.long).view(1, sequence),
                trajectory_id=context.trajectory_identity.canonical_id,
            ),
            attention_mask=attention,
            policy_attention_mask=attention,
            reference_attention_mask=attention,
            teacher_attention_mask=attention,
        ),
        crop_vision_replay_mode=(
            "shared_frozen_recorded_features"
            if legacy_crop_pixels
            else "current_live_reference_recorded_features"
        ),
    )
    replay_handle = store.put_replay(replay)
    return store, replay_handle, context, record


def test_current_crop_replay_reruns_source_and_crop_vision_with_gradients() -> None:
    store, replay_handle, context, record = _crop_replay()
    bundle = store.export_replay_bundle(replay_handle)
    replay_store, transported_handle = ObservationStore.from_replay_bundle(bundle)
    source_pixels = replay_store.resolve_verified(
        context.trajectory_source_visual.preprocessed_pixel_values
    )
    assert record.crop_visual.preprocessed_pixel_values is not None
    crop_pixels = replay_store.resolve_verified(
        record.crop_visual.preprocessed_pixel_values
    )
    model = _ToyQwen()

    request = build_trainable_crop_current_request(
        model=model,
        store=replay_store,
        replay_handle=transported_handle,
    )

    assert tuple(block.kind for block in request.visual_blocks) == (
        "source_image",
        "crop_image",
    )
    assert len(model.model.visual.seen_pixels) == 1
    torch.testing.assert_close(
        model.model.visual.seen_pixels[0],
        torch.cat((source_pixels, crop_pixels), dim=0),
        rtol=0,
        atol=0,
    )
    assert model.model.visual.seen_grids[0].tolist() == [
        [1, 2, 2],
        [1, 2, 2],
    ]
    recorded_crop = replay_store.resolve_verified(record.crop_visual.merged_main)
    assert not torch.allclose(
        request.visual_blocks[1].embeddings,
        recorded_crop,
    )

    loss = sum(
        block.embeddings.square().sum()
        + sum(branch.square().sum() for branch in block.deepstack)
        for block in request.visual_blocks
    )
    loss.backward()

    assert model.model.visual.stem.weight.grad is not None
    assert torch.count_nonzero(model.model.visual.stem.weight.grad).item() > 0
    assert model.model.visual.merger.projection.weight.grad is not None
    assert all(
        merger.projection.weight.grad is not None
        for merger in model.model.visual.deepstack_merger_list
    )


def test_current_crop_replay_rejects_legacy_frozen_crop_features() -> None:
    store, replay_handle, _context_value, _record = _crop_replay(
        legacy_crop_pixels=True
    )

    with pytest.raises(ReplayMismatchError, match="crop pixel_values"):
        build_trainable_crop_current_request(
            model=_ToyQwen(),
            store=store,
            replay_handle=replay_handle,
        )


def test_direct_answer_crop_replay_runs_live_source_vision_once() -> None:
    store, replay_handle, context, record = _crop_replay()
    replay = store.resolve_replay(replay_handle)
    direct = replace(
        replay,
        replay_id="crop-current-direct-replay",
        observation_handles=(),
        crop_vision_replay_mode="no_crop",
    )
    direct_handle = store.put_replay(direct)
    source_pixels = store.resolve_verified(
        context.trajectory_source_visual.preprocessed_pixel_values
    )
    model = _ToyQwen()

    request = build_trainable_crop_current_request(
        model=model,
        store=store,
        replay_handle=direct_handle,
    )

    assert tuple(block.kind for block in request.visual_blocks) == ("source_image",)
    assert len(model.model.visual.seen_pixels) == 1
    torch.testing.assert_close(
        model.model.visual.seen_pixels[0], source_pixels, rtol=0, atol=0
    )
    assert model.model.visual.seen_grids[0].tolist() == [[1, 2, 2]]
    assert record.sequence_length == request.input_ids.shape[1]


def test_current_crop_nonfused_decoder_batch_matches_variable_row_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(127)
    bundles = (_bundle_with_sequence(8), _bundle_with_sequence(10))
    rows = []
    for bundle, sequence in zip(bundles, (8, 10), strict=True):
        response = OwnedTokenSequence(
            token_ids=tuple(range(4, sequence + 1)),
            ownership=(TokenOwnership.POLICY_SAMPLED,) * (sequence - 3),
        )
        rows.append(
            ExactReplayResponseRequest(
                bundle=bundle,
                prompt_token_ids=(1, 2, 3),
                response=response,
                sampling=_sampling(bundle),
            )
        )

    rowwise_model = _CropReplayQwen()
    batched_model = copy.deepcopy(rowwise_model)
    monkeypatch.setattr(
        "tgvf_rl.policy.trainable_crop_replay.build_trainable_crop_current_request",
        _live_crop_request,
    )
    rowwise_port = TrainableCropCurrentReplayPort(
        engine=type("Engine", (), {"_autocast_dtype": torch.float32})(),
        model=rowwise_model,
    )
    batched_port = TrainableCropCurrentReplayPort(
        engine=type("Engine", (), {"_autocast_dtype": torch.float32})(),
        model=batched_model,
    )

    expected = tuple(
        rowwise_port.replay_response_logprobs(
            bundle=row.bundle,
            prompt_token_ids=row.prompt_token_ids,
            response=row.response,
            sampling=row.sampling,
        )
        for row in rows
    )
    actual = batched_port.replay_response_logprobs_batch(rows=tuple(rows))
    expected_loss = sum(result.logprobs.sum() for result in expected)
    actual_loss = sum(result.logprobs.sum() for result in actual)
    expected_loss.backward()
    actual_loss.backward()

    assert tuple(result.bundle_sha256 for result in actual) == tuple(
        row.bundle.bundle_sha256 for row in rows
    )
    for rowwise, batched in zip(expected, actual, strict=True):
        torch.testing.assert_close(batched.logprobs, rowwise.logprobs)
        assert batched.response_token_ids == rowwise.response_token_ids
    for name in (
        "model.language_model.proj.weight",
        "model.visual.weight",
        "lm_head.weight",
    ):
        expected_grad = dict(rowwise_model.named_parameters())[name].grad
        actual_grad = dict(batched_model.named_parameters())[name].grad
        assert expected_grad is not None and actual_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad)
