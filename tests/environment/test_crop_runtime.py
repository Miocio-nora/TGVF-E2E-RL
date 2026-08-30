from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.errors import RecoverableToolExecutionError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.crop_runtime import (
    CropExecutionLedger,
    ImageZoomInToolRuntime,
)
from tgvf_rl.environment.crop_tool import CropReplayLayout, CropVisualTensorBundle
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT,
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
    QwenNativeToolObservationAppender,
)
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.schema import TrajectorySourceVisual
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)
from tgvf_rl.protocol.schema import (
    NativeToolCapabilityProfile,
    SampledAssistantTurn,
    TokenByteSpan,
)
from tgvf_rl.qwen.crop_coordinates import (
    CanonicalSourcePixelCropCoordinateMapper,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.trajectories.schema import TrajectoryIdentity


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
BRANCH_LAYERS = (8, 16, 24)


class _Materializer:
    def __init__(self, model: ModelIdentity, trace: list[str]) -> None:
        self.model_identity = model
        self.trace = trace
        self.calls = 0

    def materialize(self, crop_rgb, *, parsed_call, call_index):
        self.trace.append("materialize")
        self.calls += 1
        return CropVisualTensorBundle(
            merged_main=torch.full((1, 1, 8), 3.0),
            merged_deepstack=tuple(
                torch.full((1, 1, 8), float(index + 4)) for index in range(3)
            ),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
            deepstack_branch_layers=BRANCH_LAYERS,
        )


class _LayoutBuilder:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.calls = 0
        self.visual: CropVisualTensorBundle | None = None
        self.environment_success_texts: list[str] = []

    def build_crop(
        self,
        context,
        crop_visual,
        parsed_call,
        *,
        environment_success_text,
    ):
        self.trace.append("layout")
        self.calls += 1
        self.visual = crop_visual
        self.environment_success_texts.append(environment_success_text)
        assert parsed_call.sampled_text == context.sampled_turn.text
        return CropReplayLayout(
            sequence_length=16,
            original_image_positions=context.trajectory_source_visual.positions,
            crop_positions=(6,),
            deepstack_injection_positions=((6,), (6,), (6,)),
        )


class _GradientMaterializer(_Materializer):
    def materialize(self, crop_rgb, *, parsed_call, call_index):
        visual = super().materialize(
            crop_rgb,
            parsed_call=parsed_call,
            call_index=call_index,
        )
        return replace(
            visual,
            merged_main=torch.ones((1, 1, 8), requires_grad=True),
        )


class _AspectRatioRejectingMaterializer(_Materializer):
    def materialize(self, crop_rgb, *, parsed_call, call_index):
        self.trace.append("materialize")
        self.calls += 1
        raise ValueError("absolute aspect ratio must be smaller than 200, got 330.0")


class _NativeTokenizer:
    name_or_path = "/fixture"
    _native = {
        "<|vision_start|>": 1,
        "<|image_pad|>": 2,
        "<|vision_end|>": 3,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._native[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            for token, token_id in self._native.items():
                if text.startswith(token, cursor):
                    result.append(token_id)
                    cursor += len(token)
                    break
            else:
                result.append(10 + (ord(text[cursor]) % 200))
                cursor += 1
        return result


def _model() -> ModelIdentity:
    return ModelIdentity("qwen3_vl", "fixture", "/fixture", 256, SHA2)


def _policy() -> PolicyVersion:
    return PolicyVersion("run", 0, SHA0)


def _crop_contract(
    protocol_id: NativeSuccessObservationProtocolId = (
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    ),
) -> NativeSuccessObservationContract:
    return NativeSuccessObservationContract(
        protocol_id=protocol_id,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )


def _parsed_call(bbox: str = "[0,1,4,8]"):
    text = (
        "reason</think>"
        '<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":'
        f"{bbox}}}}}</tool_call>"
    )
    token_ids = tuple(1000 + index for index in range(len(text)))
    byte_spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    return StrictToolCallParser().parse(
        SampledAssistantTurn(text, token_ids, byte_spans)
    )


def _source(
    store: ObservationStore,
    trajectory_id: str,
) -> tuple[TrajectorySourceVisual, torch.Tensor]:
    pixels = torch.arange(4 * 5 * 3, dtype=torch.uint8).reshape(4, 5, 3)
    main = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    branches = tuple(torch.full((1, 4, 8), float(index)) for index in range(3))
    bundle = SourceVisualTensorBundle(
        # SourceVisualState keeps the dataset/file identity; source_pixels owns
        # the separately content-addressed decoded RGB tensor.
        image_sha256=SHA2,
        premerge_main=main,
        premerge_deepstack=branches,
        merged_main=main,
        merged_deepstack=branches,
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=1,
        decoded_rgb_sha256=tensor_checksum(pixels),
    )
    return (
        record_trajectory_source_visual(
            trajectory_id=trajectory_id,
            source_visual=bundle,
            source_positions=(1, 2, 3, 4),
            deepstack_branch_layers=BRANCH_LAYERS,
            deepstack_injection_positions=((1, 2, 3, 4),) * 3,
            observation_store=store,
            source_rgb=pixels,
        ),
        pixels,
    )


def _context(
    *,
    parsed_call,
    source: TrajectorySourceVisual,
    model: ModelIdentity,
) -> ToolExecutionContext:
    policy = _policy()
    sampled = SampledPolicyTurn(
        text=parsed_call.sampled_text,
        token_ids=parsed_call.sampled_token_ids,
        token_byte_spans=parsed_call.sampled_token_byte_spans,
        behavior_logprobs=tuple(-0.1 for _ in parsed_call.sampled_token_ids),
        sampling=SamplingIdentity(
            policy_version=policy,
            backend="vllm",
            backend_version="fixture",
            seed=42,
            rng_state_sha256=SHA1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        ),
        think_token_span=TokenSpan(0, 1),
        stop_reason="tool_call",
        backend_request_sha256=SHA1,
        backend_response_sha256=SHA2,
    )
    return ToolExecutionContext(
        trajectory_identity=TrajectoryIdentity("run", "sample", 0, "group"),
        model=model,
        behavior_policy=policy,
        trajectory_source_visual=source,
        prior_observation_handles=(),
        prompt_token_ids_before_turn=(1, 2, 3),
        sampled_turn=sampled,
        assistant_turn_index=0,
        attempt_index=0,
        call_index=0,
    )


def _runtime():
    model = _model()
    store = ObservationStore()
    trajectory_id = "run/sample/0/group"
    source, pixels = _source(store, trajectory_id)
    trace: list[str] = []
    materializer = _Materializer(model, trace)
    layout = _LayoutBuilder(trace)
    ledger = CropExecutionLedger()
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=materializer,
        layout_builder=layout,
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=ledger,
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        observation_contract=_crop_contract(),
    )
    return runtime, store, source, pixels, materializer, layout, ledger, trace


def test_plain_crop_runtime_materializes_once_then_late_binds_layout() -> None:
    runtime, store, source, pixels, materializer, layout, ledger, trace = _runtime()
    parsed = _parsed_call()
    context = _context(parsed_call=parsed, source=source, model=_model())

    first = runtime.execute(parsed, context)
    second = runtime.execute(parsed, context)

    assert second == first
    assert materializer.calls == 1
    assert layout.calls == 1
    assert trace == ["materialize", "layout"]
    assert layout.environment_success_texts == [QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT]
    assert ledger.entry_count() == 1
    record = store.resolve_record(first)
    assert record.source_visual is source.state
    assert record.source_pixels_sha256 == source.source_pixels.address.digest
    expected_crop = pixels[1:4, 0:4, :].contiguous()
    torch.testing.assert_close(
        store.resolve_verified(record.crop_visual.crop_pixels),
        expected_crop,
        rtol=0,
        atol=0,
    )


def test_crop_runtime_and_appender_share_one_explicit_byte_contract() -> None:
    runtime, _store, source, _pixels, _materializer, layout, _ledger, _trace = (
        _runtime()
    )
    parsed = _parsed_call()
    context = _context(parsed_call=parsed, source=source, model=_model())
    handle = runtime.execute(parsed, context)
    tokenizer = _NativeTokenizer()

    class Registrar:
        def register_tool_turn(self, **_kwargs) -> None:
            return None

    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=Registrar(),
        observation_contract=runtime.observation_contract,
    )
    _updated, environment_ids = appender.append(
        context.prompt_token_ids_before_turn,
        context.sampled_turn,
        handle,
        call_index=0,
        parsed_call=parsed,
    )

    assert layout.environment_success_texts == [QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT]
    assert environment_ids == tuple(
        tokenizer.encode(
            layout.environment_success_texts[0],
            add_special_tokens=False,
        )
    )


def test_matched_crop_bytes_survive_two_calls_and_final_recorded_replay() -> None:
    model = _model()
    store = ObservationStore()
    source, _pixels = _source(store, "run/sample/0/group")
    tokenizer = _NativeTokenizer()
    canonical_prompt = tuple(
        tokenizer.encode(
            QWEN_NATIVE_IMAGE_PLACEHOLDER + "Q",
            add_special_tokens=False,
        )
    )
    initial_prompt = (
        canonical_prompt[:1]
        + (canonical_prompt[1],) * len(source.positions)
        + canonical_prompt[2:]
    )

    def get_rope_index(*, input_ids, **_kwargs):
        sequence = input_ids.shape[-1]
        positions = torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1)
        return positions, torch.zeros(1, 1, dtype=torch.long)

    layout_builder = Qwen3NativeToolLayoutBuilder(
        tokenizer=tokenizer,
        model_identity=model,
        observation_store=store,
        get_rope_index=get_rope_index,
    )
    contract = _crop_contract()
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=_Materializer(model, []),
        layout_builder=layout_builder,
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=CropExecutionLedger(),
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        observation_contract=contract,
    )

    class Registrar:
        def register_tool_turn(self, **_kwargs) -> None:
            return None

    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=Registrar(),
        observation_contract=contract,
    )

    first_call = _parsed_call("[0,1,4,4]")
    first_context = replace(
        _context(parsed_call=first_call, source=source, model=model),
        prompt_token_ids_before_turn=initial_prompt,
    )
    first_handle = runtime.execute(first_call, first_context)
    first_prompt, first_environment = appender.append(
        initial_prompt,
        first_context.sampled_turn,
        first_handle,
        call_index=0,
        parsed_call=first_call,
    )

    second_call = _parsed_call("[1,0,5,3]")
    second_context = replace(
        _context(parsed_call=second_call, source=source, model=model),
        prior_observation_handles=(first_handle,),
        prompt_token_ids_before_turn=first_prompt,
        assistant_turn_index=1,
        call_index=1,
    )
    second_handle = runtime.execute(second_call, second_context)
    final_prompt, second_environment = appender.append(
        first_prompt,
        second_context.sampled_turn,
        second_handle,
        call_index=1,
        parsed_call=second_call,
    )

    expanded = layout_builder.expand_recorded_visual_sequence(
        final_prompt,
        trajectory_source_visual=source,
        observation_handles=(first_handle, second_handle),
    )
    first_record = store.resolve_record(first_handle)
    second_record = store.resolve_record(second_handle)
    assert expanded.existing_positions == (
        source.positions,
        first_record.crop_visual.positions,
    )
    assert expanded.new_positions == second_record.crop_visual.positions
    assert tuple(expanded.input_ids[0].tolist()) == final_prompt
    expected_environment = tuple(
        tokenizer.encode(
            QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
            add_special_tokens=False,
        )
    )
    assert first_environment == expected_environment
    assert second_environment == expected_environment


@pytest.mark.parametrize(
    ("protocol_id", "environment_success_text"),
    (
        (
            NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
            QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
        ),
        (
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1,
            QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT,
        ),
    ),
)
def test_qwen3_plain_crop_layout_uses_exact_contract_response_bytes(
    protocol_id: NativeSuccessObservationProtocolId,
    environment_success_text: str,
) -> None:
    model = _model()
    store = ObservationStore()
    source, _pixels = _source(store, "run/sample/0/group")
    parsed = _parsed_call()
    tokenizer = _NativeTokenizer()
    prompt_ids = tuple(
        tokenizer.encode(
            QWEN_NATIVE_IMAGE_PLACEHOLDER + "Q",
            add_special_tokens=False,
        )
    )
    context = replace(
        _context(parsed_call=parsed, source=source, model=model),
        prompt_token_ids_before_turn=prompt_ids,
    )
    captured_input_ids: list[tuple[int, ...]] = []

    def get_rope_index(*, input_ids, image_grid_thw, **_kwargs):
        assert image_grid_thw.tolist() == [[1, 2, 2], [1, 2, 2]]
        captured_input_ids.append(tuple(input_ids[0].tolist()))
        sequence = input_ids.shape[-1]
        positions = torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1)
        return positions, torch.zeros(1, 1, dtype=torch.long)

    builder = Qwen3NativeToolLayoutBuilder(
        tokenizer=tokenizer,
        model_identity=model,
        observation_store=store,
        get_rope_index=get_rope_index,
    )
    visual = CropVisualTensorBundle(
        merged_main=torch.ones((1, 1, 8)),
        merged_deepstack=tuple(torch.ones((1, 1, 8)) for _ in BRANCH_LAYERS),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
        deepstack_branch_layers=BRANCH_LAYERS,
    )
    contract = _crop_contract(protocol_id)
    assert contract.render(parsed) == environment_success_text
    layout = builder.build_crop(
        context,
        visual,
        parsed,
        environment_success_text=environment_success_text,
    )

    expected_environment_ids = tuple(
        tokenizer.encode(
            environment_success_text,
            add_special_tokens=False,
        )
    )
    expected_ids = (
        (1, 2, 2, 2, 2, 3)
        + prompt_ids[3:]
        + context.sampled_turn.token_ids
        + expected_environment_ids
    )
    assert captured_input_ids == [expected_ids]
    assert layout.sequence_length == len(expected_ids)


def test_plain_crop_runtime_call_key_rejects_changed_sampled_content() -> None:
    runtime, _, source, _, materializer, _, _, _ = _runtime()
    first = _parsed_call("[0,0,2,2]")
    runtime.execute(first, _context(parsed_call=first, source=source, model=_model()))
    changed = _parsed_call("[1,0,3,2]")

    with pytest.raises(ValueError, match="different content"):
        runtime.execute(
            changed,
            _context(parsed_call=changed, source=source, model=_model()),
        )
    assert materializer.calls == 1


def test_plain_crop_runtime_call_key_includes_observation_protocol_id() -> None:
    runtime, store, source, _, materializer, layout, ledger, _ = _runtime()
    parsed = _parsed_call()
    context = _context(parsed_call=parsed, source=source, model=_model())
    runtime.execute(parsed, context)
    legacy_runtime = ImageZoomInToolRuntime(
        model=runtime.model,
        materializer=materializer,
        layout_builder=layout,
        observation_store=store,
        crop_processor_identity=runtime.crop_processor_identity,
        crop_layout_identity=runtime.crop_layout_identity,
        execution_ledger=ledger,
        coordinate_mapper=runtime.coordinate_mapper,
        observation_contract=_crop_contract(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
        ),
    )

    with pytest.raises(ValueError, match="different content"):
        legacy_runtime.execute(parsed, context)

    assert materializer.calls == 1


def test_plain_crop_runtime_fails_closed_without_recorded_source_pixels() -> None:
    runtime, _, source, _, materializer, _, ledger, _ = _runtime()
    parsed = _parsed_call()
    missing_pixels = replace(source, source_pixels=None)

    with pytest.raises(RuntimeError, match="immutable source RGB"):
        runtime.execute(
            parsed,
            _context(parsed_call=parsed, source=missing_pixels, model=_model()),
        )
    assert materializer.calls == 0
    assert ledger.entry_count() == 0


def test_plain_crop_runtime_maps_empty_clamped_box_to_recoverable_error() -> None:
    runtime, _, source, _, materializer, _, ledger, _ = _runtime()
    parsed = _parsed_call("[20,20,30,30]")

    with pytest.raises(RecoverableToolExecutionError, match="empty after clamping"):
        runtime.execute(
            parsed,
            _context(parsed_call=parsed, source=source, model=_model()),
        )

    assert materializer.calls == 0
    assert ledger.entry_count() == 0


def test_plain_crop_runtime_maps_invalid_qwen3_grid_to_recoverable_error() -> None:
    model = _model()
    store = ObservationStore()
    source, _ = _source(store, "run/sample/0/group")
    trace: list[str] = []
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=_Materializer(model, trace),
        layout_builder=_LayoutBuilder(trace),
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=CropExecutionLedger(),
        coordinate_mapper=Qwen3VLAdapter(),
        observation_contract=_crop_contract(),
    )
    parsed = _parsed_call("[0,0,1001,1000]")

    with pytest.raises(RecoverableToolExecutionError, match="within 0..1000"):
        runtime.execute(
            parsed,
            _context(parsed_call=parsed, source=source, model=model),
        )

    assert trace == []


def test_plain_crop_runtime_maps_processor_aspect_ratio_to_recoverable_error() -> None:
    model = _model()
    store = ObservationStore()
    source, _ = _source(store, "run/sample/0/group")
    trace: list[str] = []
    materializer = _AspectRatioRejectingMaterializer(model, trace)
    ledger = CropExecutionLedger()
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=materializer,
        layout_builder=_LayoutBuilder(trace),
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=ledger,
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        observation_contract=_crop_contract(),
    )
    parsed = _parsed_call()

    with pytest.raises(RecoverableToolExecutionError, match="absolute aspect ratio"):
        runtime.execute(
            parsed,
            _context(parsed_call=parsed, source=source, model=model),
        )

    assert materializer.calls == 1
    assert ledger.entry_count() == 0


def test_plain_crop_runtime_rejects_gradient_carrying_recorded_features() -> None:
    model = _model()
    store = ObservationStore()
    source, _ = _source(store, "run/sample/0/group")
    trace: list[str] = []
    materializer = _GradientMaterializer(model, trace)
    ledger = CropExecutionLedger()
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=materializer,
        layout_builder=_LayoutBuilder(trace),
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "qwen", "crop-processor", "fixture", SHA1
        ),
        crop_layout_identity=ArtifactIdentity("qwen", "crop-layout", "fixture", SHA2),
        execution_ledger=ledger,
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        observation_contract=_crop_contract(),
    )
    parsed = _parsed_call()

    with pytest.raises(ValueError, match="detached floating tensors"):
        runtime.execute(
            parsed,
            _context(parsed_call=parsed, source=source, model=model),
        )
    assert materializer.calls == 1
    assert ledger.entry_count() == 0


def test_plain_crop_runtime_rejects_materializer_bound_to_another_model() -> None:
    model = _model()
    other = replace(model, revision_or_path="/other")
    trace: list[str] = []

    with pytest.raises(ValueError, match="materializer model"):
        ImageZoomInToolRuntime(
            model=model,
            materializer=_Materializer(other, trace),
            layout_builder=_LayoutBuilder(trace),
            observation_store=ObservationStore(),
            crop_processor_identity=ArtifactIdentity(
                "qwen", "crop-processor", "fixture", SHA1
            ),
            crop_layout_identity=ArtifactIdentity(
                "qwen", "crop-layout", "fixture", SHA2
            ),
            execution_ledger=CropExecutionLedger(),
            coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
            observation_contract=_crop_contract(),
        )
