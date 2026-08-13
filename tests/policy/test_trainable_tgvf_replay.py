from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.environment.test_crop_tgvf_runtime import _fixture as _atomic_fixture
from tests.policy.test_exact_replay import _payload
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.policy import trainable_tgvf_replay as replay_module
from tgvf_rl.policy.trainable_tgvf_replay import (
    TRAINABLE_TGVF_ADAPTER_ATTRIBUTE,
    TrainableTGVFCurrentReplayPort,
    build_trainable_tgvf_current_request,
    extract_live_qwen3_vision_features,
    trainable_parameter_zero_anchor,
)
from tgvf_rl.observations.store import (
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.qwen.base import (
    ReplayConsumer,
    injected_request_from_recorded,
    resolve_replay_request,
)
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.deepstack import TrainableBorrowedProjectionPort


class _ToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grouped = hidden_states.reshape(-1, 4, 3).mean(dim=1)
        return self.projection(grouped)


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 3, bias=False)
        self.merger = _ToyMerger()
        self.deepstack_merger_list = nn.ModuleList(_ToyMerger() for _ in range(3))

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        assert tuple(grid_thw.shape) == (1, 3)
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


class _AtomicToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grouped = hidden_states.reshape(-1, 4, 4).mean(dim=1)
        return self.projection(grouped)


class _AtomicToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 4, bias=False)
        self.merger = _AtomicToyMerger()
        self.deepstack_merger_list = nn.ModuleList(
            _AtomicToyMerger() for _ in range(3)
        )
        self.seen_pixel_values: list[torch.Tensor] = []

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        assert tuple(grid_thw.shape) == (1, 3)
        self.seen_pixel_values.append(pixel_values.detach().cpu().clone())
        hidden = self.stem(pixel_values)
        outputs = [self.merger(hidden)]
        outputs.extend(
            merger(hidden * (index + 2))
            for index, merger in enumerate(self.deepstack_merger_list)
        )
        return outputs[0], tuple(outputs[1:])


class _AtomicToyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _AtomicToyVisual()
        mergers = (
            self.model.visual.merger,
            *tuple(self.model.visual.deepstack_merger_list),
        )
        ports = tuple(
            TrainableBorrowedProjectionPort(
                merger,
                identity=f"atomic-current-merger-{index}",
                input_dim=4,
                output_dim=8,
                spatial_merge_size=2,
            )
            for index, merger in enumerate(mergers)
        )
        adapter = TGVFAdapter(
            d_lm=8,
            d_v=4,
            attn_dim=4,
            main_projection=ports[0],
            deepstack_projections=ports[1:],
            branch_layers=(8, 16, 24),
        )
        adapter.requires_grad_(False).train(True)
        self.add_module(TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, adapter)


def _trainable_adapter() -> TGVFAdapter:
    projections = tuple(
        FrozenProjectionPort(
            _ToyMerger(),
            identity=f"mixed-precision-merger-{index}",
            input_dim=3,
            output_dim=5,
            spatial_merge_size=2,
        )
        for index in range(4)
    )
    return TGVFAdapter(
        d_lm=5,
        d_v=3,
        attn_dim=4,
        main_projection=projections[0],
        deepstack_projections=projections[1:],
        branch_layers=(8, 16, 24),
    )


class _AutocastBoundaryQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(5, 128, bias=False)
        self.add_module(TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, _trainable_adapter())


def test_live_vision_capture_keeps_pixel_stem_and_merger_autograd() -> None:
    model = _ToyQwen()
    pixels = torch.randn(8, 3, requires_grad=True)

    features = extract_live_qwen3_vision_features(
        model,
        pixel_values=pixels,
        image_grid_thw=(1, 2, 4),
    )
    loss = features.merged_main.square().sum() + sum(
        branch.square().sum() for branch in features.merged_deepstack
    )
    loss.backward()

    assert features.premerge_main.shape == (8, 3)
    assert features.merged_main.shape == (2, 5)
    assert pixels.grad is not None
    assert model.model.visual.stem.weight.grad is not None
    assert model.model.visual.merger.projection.weight.grad is not None
    assert all(
        merger.projection.weight.grad is not None
        for merger in model.model.visual.deepstack_merger_list
    )


def test_zero_anchor_materializes_exact_zero_gradient_for_every_parameter() -> None:
    module = nn.Sequential(
        nn.Linear(5, 7),
        nn.LayerNorm(7),
        nn.Linear(7, 3, bias=False),
    )

    anchor = trainable_parameter_zero_anchor(module)
    anchor.backward()

    assert anchor.item() == 0.0
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(
        torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.parameters()
    )


def test_frozen_adapter_keeps_autograd_to_visual_and_target_inputs() -> None:
    """Frozen RP66 is a fixed differentiable transform, not a detached one."""

    adapter = _trainable_adapter()
    adapter.requires_grad_(False)
    target = torch.randn(3, 5, requires_grad=True)
    visual = torch.randn(8, 3, requires_grad=True)
    branches = tuple(torch.randn(8, 3, requires_grad=True) for _ in range(3))

    output = adapter(
        target_hidden_states=target,
        pre_merge_visual_tokens=visual,
        deepstack_pre_merge_visual_tokens=branches,
    )
    loss = output.main_d.square().sum() + sum(
        branch.square().sum() for branch in output.d_deepstack.branches
    )
    loss.backward()

    assert target.grad is not None
    assert torch.count_nonzero(target.grad).item() > 0
    assert visual.grad is not None
    assert torch.count_nonzero(visual.grad).item() > 0
    assert all(branch.grad is not None for branch in branches)
    assert all(parameter.grad is None for parameter in adapter.parameters())


def test_atomic_crop_tgvf_current_replay_reruns_crop_vision_and_injects_only_live_d(
    tmp_path,
) -> None:
    """Current actor uses exact crop pixels, not rollout D or raw crop features."""

    runtime, _materializer, store, _pixels, _capture, context, parsed = (
        _atomic_fixture(tmp_path, provider_kind="contextual_hidden_state")
    )
    observation_handle = runtime.execute(parsed, context)
    record = store.resolve_record(observation_handle)
    sequence = record.layout.sequence_length
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="atomic-current-replay",
        trajectory_id=context.trajectory_identity.canonical_id,
        model=context.model,
        behavior_policy=context.behavior_policy,
        source_visual=context.trajectory_source_visual,
        observation_handles=(observation_handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=store.put_tensor(
                "atomic-current.input_ids",
                torch.arange(sequence, dtype=torch.long).view(1, sequence),
                trajectory_id=context.trajectory_identity.canonical_id,
            ),
            position_ids=record.payload.position_ids,
            attention_mask=record.payload.attention_mask,
            policy_attention_mask=record.masks.policy_visible,
            reference_attention_mask=record.masks.reference_visible,
            teacher_attention_mask=record.masks.teacher_visible,
        ),
        crop_vision_replay_mode="current_live_reference_recorded_features",
    )
    replay_handle = store.put_replay(replay)
    recorded_d = store.resolve_verified(record.payload.main_d)
    source_pixels = store.resolve_verified(
        context.trajectory_source_visual.preprocessed_pixel_values
    )
    crop_pixels = store.resolve_verified(
        record.crop_visual.preprocessed_pixel_values
    )

    torch.manual_seed(91)
    model = _AtomicToyQwen()
    adapter = model.tgvf_adapter
    request = build_trainable_tgvf_current_request(
        model=model,
        adapter=adapter,
        store=store,
        replay_handle=replay_handle,
    )

    assert tuple(block.kind for block in request.visual_blocks) == (
        "source_image",
        "crop_focused_d",
    )
    assert len(model.model.visual.seen_pixel_values) == 2
    torch.testing.assert_close(
        model.model.visual.seen_pixel_values[0], source_pixels, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.model.visual.seen_pixel_values[1], crop_pixels, rtol=0, atol=0
    )
    assert not torch.allclose(
        request.visual_blocks[1].embeddings.squeeze(0), recorded_d
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
    assert all(
        parameter.grad is None
        for parameter in adapter.artifact_state_dict(keep_vars=True).values()
    )


def test_current_replay_accepts_fully_frozen_adapter() -> None:
    model = _AutocastBoundaryQwen()
    model.tgvf_adapter.requires_grad_(False)

    port = TrainableTGVFCurrentReplayPort(
        engine=SimpleNamespace(_autocast_dtype=torch.bfloat16),
        model=model,
        selected_logprob_materializer=lambda **_kwargs: torch.ones(1),
    )

    assert port.adapter_trainable is False


def test_current_replay_autocast_starts_before_live_vision_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live vision/RP66 request and decoder share one precision scope."""

    payload, _observation = _payload()
    bundle = payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0]
    model = _AutocastBoundaryQwen()
    port = TrainableTGVFCurrentReplayPort(
        engine=SimpleNamespace(_autocast_dtype=torch.bfloat16),
        model=model,
        selected_logprob_materializer=lambda **kwargs: kwargs["hidden_states"][
            :, : kwargs["sampled_positions"].shape[1], 0
        ],
    )
    saw_autocast: list[bool] = []

    def build_request(*, model, adapter, store, replay_handle):
        del model, adapter
        saw_autocast.append(torch.is_autocast_enabled("cpu"))
        recorded = resolve_replay_request(store, replay_handle, ReplayConsumer.POLICY)
        return injected_request_from_recorded(recorded)

    class _Family:
        def forward_injected_hidden(self, current_model, request):
            assert torch.is_autocast_enabled("cpu")
            hidden = (
                current_model.lm_head.weight[0]
                .view(1, 1, 5)
                .expand(1, request.input_ids.shape[1], 5)
            )
            return SimpleNamespace(hidden_states=hidden)

    monkeypatch.setattr(
        replay_module, "build_trainable_tgvf_current_request", build_request
    )
    port.family_adapter = _Family()
    response = OwnedTokenSequence(
        (20, 21, 22, 90, 91, 30, 31, 32, 33),
        (
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.TOOL_OBSERVATION,
            TokenOwnership.TOOL_OBSERVATION,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
        ),
    )
    sampling = SamplingIdentity(
        policy_version=bundle.replay_record.behavior_policy,
        backend="vllm",
        backend_version="autocast-regression",
        seed=42,
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

    result = port.replay_response_logprobs(
        bundle=bundle,
        prompt_token_ids=(101, 102, 103),
        response=response,
        sampling=sampling,
    )

    assert saw_autocast == [True]
    assert result.logprobs.requires_grad
