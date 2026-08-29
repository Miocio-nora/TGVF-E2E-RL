"""Real-Qwen CPU canary for matched Crop layout/appender replay parity.

This deliberately loads only the local processor and model config.  It never
loads model weights, starts vLLM/Ray, or touches CUDA.  Synthetic detached
feature tensors stand in for the vision tower because this regression is at
the tokenizer/layout/appender boundary, while every image grid and M-RoPE
position is still produced by the real Qwen3-VL processor/config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    SamplingIdentity,
    TokenSpan,
)
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.crop_runtime import CropExecutionLedger, ImageZoomInToolRuntime
from tgvf_rl.environment.crop_tool import CropVisualTensorBundle
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.native_appender import (
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
    QwenNativeToolObservationAppender,
    render_qwen_native_matched_crop_success_environment_text,
)
from tgvf_rl.environment.qwen3_crop_materializer import preprocess_qwen3_rgb
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.verl.policy_live_runtime import _source_visual_positions
from tgvf_rl.framework.verl.smoke_dataset import (
    _materialize_source_image_prompt_token_ids,
)
from tgvf_rl.framework.vllm import (
    FastTokenizerTokenByteSpanDecoder,
    QWEN3_DEEPSTACK_BRANCH_LAYERS,
    VLLMOutputDecodingContract,
)
from tgvf_rl.observations.schema import CropObservationRecord
from tgvf_rl.observations.store import ObservationHandle, ObservationStore, tensor_checksum
from tgvf_rl.policy.deepeyes_official_protocol import build_visual_messages
from tgvf_rl.protocol import NativeAssistantDialect, StrictToolCallParser
from tgvf_rl.qwen import Qwen3VLAdapter
from tgvf_rl.trajectories.schema import TrajectoryIdentity


_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
_IMAGE_MAX_PIXELS = 262_144
_SHA0 = "0" * 64
_SHA1 = "1" * 64
_SHA2 = "2" * 64
_DECODING = VLLMOutputDecodingContract(True, False, False, "final_only")


class _CPUProcessorCropMaterializer:
    """Run the real image processor and provide detached shape-faithful features."""

    def __init__(self, processor: object, model_identity: ModelIdentity) -> None:
        self.processor = processor
        self.model_identity = model_identity
        self.merge_size = int(processor.image_processor.merge_size)
        self.grids: list[tuple[int, int, int]] = []

    def materialize(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: object,
        call_index: int,
    ) -> CropVisualTensorBundle:
        del parsed_call, call_index
        pixel_values, grid_tensor = preprocess_qwen3_rgb(
            processor=self.processor,
            rgb=crop_rgb,
            image_max_pixels=_IMAGE_MAX_PIXELS,
        )
        grid = tuple(int(value) for value in grid_tensor[0].tolist())
        merged_count = _merged_token_count(grid, self.merge_size)
        main = torch.zeros((1, merged_count, 4), dtype=torch.float32)
        branches = tuple(torch.zeros_like(main) for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS)
        self.grids.append(grid)
        return CropVisualTensorBundle(
            merged_main=main,
            merged_deepstack=branches,
            image_grid_thw=grid,
            spatial_merge_size=self.merge_size,
            deepstack_branch_layers=QWEN3_DEEPSTACK_BRANCH_LAYERS,
            preprocessed_pixel_values=pixel_values,
        )


class _ObservationTokenCountResolver:
    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    def resolve_visual_token_count(self, handle: ObservationHandle) -> int:
        record = self.store.resolve_record(handle)
        if not isinstance(record, CropObservationRecord):
            raise TypeError("matched Crop canary resolved another observation type")
        return len(record.crop_visual.positions)


class _Registrar:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def register_tool_turn(self, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))


class _RecordingMatchedRenderer:
    def __init__(self, *, drift_appender_calls: bool = False) -> None:
        self.drift_appender_calls = drift_appender_calls
        self.calls: list[tuple[str, NativeAssistantDialect, str]] = []

    def __call__(
        self,
        parsed_call: object,
        *,
        assistant_dialect: NativeAssistantDialect,
    ) -> str:
        rendered = render_qwen_native_matched_crop_success_environment_text(
            parsed_call,
            assistant_dialect=assistant_dialect,
        )
        # Runtime/layout renders first and appender renders second for each call.
        # Prefix drift moves the native visual block and must fail replay closed.
        if self.drift_appender_calls and len(self.calls) % 2 == 1:
            rendered = "\n" + rendered
        self.calls.append((parsed_call.sampled_text, assistant_dialect, rendered))
        return rendered


def _merged_token_count(grid: tuple[int, int, int], merge_size: int) -> int:
    count = grid[0] * grid[1] * grid[2]
    assert count % (merge_size**2) == 0
    return count // (merge_size**2)


def _source_rgb() -> torch.Tensor:
    height, width = 504, 672
    rows = torch.arange(height, dtype=torch.int32).view(height, 1)
    columns = torch.arange(width, dtype=torch.int32).view(1, width)
    return torch.stack(
        (
            (columns.expand(height, width) % 256).to(torch.uint8),
            (rows.expand(height, width) % 256).to(torch.uint8),
            ((rows + columns) % 256).to(torch.uint8),
        ),
        dim=-1,
    ).contiguous()


def _model_identity(processor: object) -> ModelIdentity:
    template = processor.chat_template or processor.tokenizer.chat_template
    assert isinstance(template, str) and template
    return ModelIdentity(
        family="qwen3_vl",
        model_name="Qwen3-VL-8B-Instruct",
        revision_or_path=str(_MODEL_PATH),
        tokenizer_length=len(processor.tokenizer),
        chat_template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )


def _initial_prompt_ids(processor: object, source_rgb: torch.Tensor) -> tuple[int, ...]:
    text = processor.apply_chat_template(
        list(build_visual_messages("Inspect the two marked regions.")),
        tools=[],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(text, str) and text
    canonical_ids = tuple(processor.tokenizer.encode(text, add_special_tokens=False))
    return _materialize_source_image_prompt_token_ids(
        processor=processor,
        canonical_token_ids=canonical_ids,
        prompt_text=text,
        source_rgb=source_rgb,
        image_max_pixels=_IMAGE_MAX_PIXELS,
    )


def _source_visual(
    *,
    processor: object,
    source_rgb: torch.Tensor,
    initial_prompt_ids: tuple[int, ...],
    store: ObservationStore,
    trajectory_id: str,
) -> object:
    pixel_values, grid_tensor = preprocess_qwen3_rgb(
        processor=processor,
        rgb=source_rgb,
        image_max_pixels=_IMAGE_MAX_PIXELS,
    )
    grid = tuple(int(value) for value in grid_tensor[0].tolist())
    merge_size = int(processor.image_processor.merge_size)
    premerge_count = grid[0] * grid[1] * grid[2]
    merged_count = _merged_token_count(grid, merge_size)
    premerge = torch.zeros((1, premerge_count, 4), dtype=torch.float32)
    merged = torch.zeros((1, merged_count, 4), dtype=torch.float32)
    premerge_branches = tuple(
        torch.zeros_like(premerge) for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS
    )
    merged_branches = tuple(
        torch.zeros_like(merged) for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS
    )
    source_positions = _source_visual_positions(
        initial_prompt_ids,
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        expected_count=merged_count,
    )
    bundle = SourceVisualTensorBundle(
        image_sha256=tensor_checksum(source_rgb),
        premerge_main=premerge,
        premerge_deepstack=premerge_branches,
        merged_main=merged,
        merged_deepstack=merged_branches,
        image_grid_thw=grid,
        spatial_merge_size=merge_size,
        decoded_rgb_sha256=tensor_checksum(source_rgb),
    )
    return record_trajectory_source_visual(
        trajectory_id=trajectory_id,
        source_visual=bundle,
        source_positions=source_positions,
        deepstack_branch_layers=QWEN3_DEEPSTACK_BRANCH_LAYERS,
        deepstack_injection_positions=tuple(
            source_positions for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS
        ),
        observation_store=store,
        preprocessed_pixel_values=pixel_values,
        source_rgb=source_rgb,
    )


def _sampled_crop_turn(
    *,
    tokenizer: object,
    policy: PolicyVersion,
    bbox: tuple[int, int, int, int],
    label: str,
    turn_index: int,
) -> tuple[SampledPolicyTurn, object]:
    payload = json.dumps(
        {
            "name": "image_zoom_in_tool",
            "arguments": {"bbox_2d": list(bbox), "label": label},
        },
        separators=(",", ":"),
    )
    text = f"<think>inspect region {turn_index}</think><tool_call>{payload}</tool_call>"
    token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    assert tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
    ) == text
    spans = FastTokenizerTokenByteSpanDecoder(tokenizer).spans_for_output(
        text=text,
        token_ids=token_ids,
        decoding=_DECODING,
    )
    think_open = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    start = token_ids.index(think_open) + 1
    end = token_ids.index(think_close)
    sampled = SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.1 for _ in token_ids),
        sampling=SamplingIdentity(
            policy_version=policy,
            backend="vllm",
            backend_version="cpu-canary",
            seed=42 + turn_index,
            rng_state_sha256=_SHA1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        ),
        think_token_span=TokenSpan(start, end),
        stop_reason="tool_call",
        backend_request_sha256=_SHA1,
        backend_response_sha256=_SHA2,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    parsed = StrictToolCallParser(
        enabled_tool_names=("image_zoom_in_tool",)
    ).parse(sampled.parser_turn())
    return sampled, parsed


def _visual_runs(token_ids: tuple[int, ...], image_pad_id: int) -> tuple[tuple[int, ...], ...]:
    runs: list[tuple[int, ...]] = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != image_pad_id:
            index += 1
            continue
        start = index
        while index < len(token_ids) and token_ids[index] == image_pad_id:
            index += 1
        runs.append(tuple(range(start, index)))
    return tuple(runs)


def _canary_components(processor: object, renderer: object) -> dict[str, object]:
    model = _model_identity(processor)
    store = ObservationStore()
    identity = TrajectoryIdentity("prl27-cpu-canary", "sample", 0, "group")
    policy = PolicyVersion("prl27-cpu-canary", 0, _SHA0)
    source_rgb = _source_rgb()
    initial_ids = _initial_prompt_ids(processor, source_rgb)
    source = _source_visual(
        processor=processor,
        source_rgb=source_rgb,
        initial_prompt_ids=initial_ids,
        store=store,
        trajectory_id=identity.canonical_id,
    )
    layout = Qwen3NativeToolLayoutBuilder.from_processor_config(
        processor=processor,
        model_identity=model,
        observation_store=store,
    )
    materializer = _CPUProcessorCropMaterializer(processor, model)
    runtime = ImageZoomInToolRuntime(
        model=model,
        materializer=materializer,
        layout_builder=layout,
        observation_store=store,
        crop_processor_identity=ArtifactIdentity(
            "prl27-canary", "real-qwen-processor", "v1", _SHA1
        ),
        crop_layout_identity=ArtifactIdentity(
            "prl27-canary", "real-qwen-layout", "v1", _SHA2
        ),
        execution_ledger=CropExecutionLedger(),
        coordinate_mapper=Qwen3VLAdapter(),
        success_environment_text_renderer=renderer,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=processor.tokenizer,
        registrar=registrar,
        visual_token_count_resolver=_ObservationTokenCountResolver(store),
        success_environment_text_renderer=renderer,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    return {
        "model": model,
        "store": store,
        "identity": identity,
        "policy": policy,
        "source": source,
        "initial_ids": initial_ids,
        "layout": layout,
        "materializer": materializer,
        "runtime": runtime,
        "registrar": registrar,
        "appender": appender,
    }


@pytest.fixture(scope="module")
def real_qwen_processor() -> object:
    if not _MODEL_PATH.is_dir():
        pytest.skip(f"local Qwen processor is unavailable: {_MODEL_PATH}")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(
        _MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )


def test_real_qwen_processor_two_matched_crops_and_final_replay_are_identical(
    real_qwen_processor: object,
) -> None:
    renderer = _RecordingMatchedRenderer()
    state = _canary_components(real_qwen_processor, renderer)
    prompt = state["initial_ids"]
    handles: list[ObservationHandle] = []
    environment_rows: list[tuple[int, ...]] = []
    calls = (
        ((40, 60, 590, 640), "upper-left gauge"),
        ((410, 300, 960, 940), "lower-right table"),
    )

    for call_index, (bbox, label) in enumerate(calls):
        sampled, parsed = _sampled_crop_turn(
            tokenizer=real_qwen_processor.tokenizer,
            policy=state["policy"],
            bbox=bbox,
            label=label,
            turn_index=call_index,
        )
        context = ToolExecutionContext(
            trajectory_identity=state["identity"],
            model=state["model"],
            behavior_policy=state["policy"],
            trajectory_source_visual=state["source"],
            prior_observation_handles=tuple(handles),
            prompt_token_ids_before_turn=prompt,
            sampled_turn=sampled,
            assistant_turn_index=call_index,
            attempt_index=call_index,
            call_index=call_index,
        )
        handle = state["runtime"].execute(parsed, context)
        prompt, environment_ids = state["appender"].append(
            prompt,
            sampled,
            handle,
            call_index=call_index,
            parsed_call=parsed,
        )
        handles.append(handle)
        environment_rows.append(environment_ids)
        record = state["store"].resolve_record(handle)
        assert isinstance(record, CropObservationRecord)
        assert record.sequence_length == len(prompt)

    records = tuple(state["store"].resolve_record(handle) for handle in handles)
    image_pad_id = real_qwen_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    runs = _visual_runs(prompt, image_pad_id)
    assert runs == (
        state["source"].positions,
        records[0].crop_visual.positions,
        records[1].crop_visual.positions,
    )
    assert len(set(state["materializer"].grids)) == 2
    assert len(state["registrar"].rows) == 2

    canonical_environment_ids = tuple(
        real_qwen_processor.tokenizer.encode(
            QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
            add_special_tokens=False,
        )
    )
    assert len(canonical_environment_ids) == 60
    assert hashlib.sha256(
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT.encode("utf-8")
    ).hexdigest() == QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
    for environment_ids, record in zip(environment_rows, records, strict=True):
        assert len(environment_ids) == (
            len(canonical_environment_ids) - 1 + len(record.crop_visual.positions)
        )

    # Each call is rendered once for runtime/layout and once for the appender.
    assert len(renderer.calls) == 4
    for first, second in zip(renderer.calls[::2], renderer.calls[1::2], strict=True):
        assert first == second
        assert first[1] is NativeAssistantDialect.QWEN3_VL_INSTRUCT
        assert first[2] == QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT

    final = state["layout"].expand_recorded_visual_sequence(
        prompt,
        trajectory_source_visual=state["source"],
        observation_handles=tuple(handles),
    )
    assert tuple(int(value) for value in final.input_ids[0].tolist()) == prompt
    assert final.existing_positions == runs[:-1]
    assert final.new_positions == runs[-1]
    assert final.position_ids.shape[-1] == len(prompt)
    assert final.attention_mask.shape == (1, len(prompt))
    assert bool(final.attention_mask.all())


def test_real_qwen_processor_detects_one_token_layout_appender_drift(
    real_qwen_processor: object,
) -> None:
    from tgvf_rl.contracts.errors import ReplayMismatchError

    renderer = _RecordingMatchedRenderer(drift_appender_calls=True)
    state = _canary_components(real_qwen_processor, renderer)
    sampled, parsed = _sampled_crop_turn(
        tokenizer=real_qwen_processor.tokenizer,
        policy=state["policy"],
        bbox=(40, 60, 590, 640),
        label="upper-left gauge",
        turn_index=0,
    )
    context = ToolExecutionContext(
        trajectory_identity=state["identity"],
        model=state["model"],
        behavior_policy=state["policy"],
        trajectory_source_visual=state["source"],
        prior_observation_handles=(),
        prompt_token_ids_before_turn=state["initial_ids"],
        sampled_turn=sampled,
        assistant_turn_index=0,
        attempt_index=0,
        call_index=0,
    )
    handle = state["runtime"].execute(parsed, context)
    drifted_prompt, _environment_ids = state["appender"].append(
        state["initial_ids"],
        sampled,
        handle,
        call_index=0,
        parsed_call=parsed,
    )

    # A one-Crop trajectory must already fail at final replay; a second call
    # would discover the same mismatch earlier while validating prior records.
    with pytest.raises(
        ReplayMismatchError,
        match="expanded native visual positions differ from rollout record",
    ):
        state["layout"].expand_recorded_visual_sequence(
            drifted_prompt,
            trajectory_source_visual=state["source"],
            observation_handles=(handle,),
        )
