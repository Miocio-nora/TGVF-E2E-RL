from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    ContextualHiddenStateConditionProvider,
    TargetTokenEmbeddingConditionProvider,
)
from tgvf_rl.contracts.errors import (
    IdentityMismatchError,
    RecoverableToolExecutionError,
)
from tgvf_rl.contracts.identity import ArtifactIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.adapter_runtime import load_frozen_tgvf_adapter
from tgvf_rl.environment.crop_tgvf_runtime import AtomicCropTGVFToolRuntime
from tgvf_rl.environment.crop_tgvf_tool import AtomicCropTGVFTool
from tgvf_rl.environment.focus_runtime import (
    BehaviorHiddenStateCapture,
    FocusExecutionLedger,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.native_appender import (
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    render_qwen_native_success_environment_text,
)
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.schema import CropTGVFObservationRecord
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import TokenByteSpan
from tgvf_rl.qwen.crop_coordinates import (
    CanonicalSourcePixelCropCoordinateMapper,
)
from tgvf_rl.trajectories.schema import TrajectoryIdentity

from tests.environment.test_adapter_runtime import _branch_bindings, _write_artifact


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
BRANCH_LAYERS = (8, 16, 24)


class _Tokenizer:
    name_or_path = "/qwen3_vl/fixture"
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


class _Materializer:
    def __init__(self) -> None:
        self.received: list[torch.Tensor] = []

    def materialize_source_visual(self, crop_rgb, *, parsed_call, call_index):
        assert parsed_call.name == "tgvf_crop_tool"
        assert call_index == 0
        self.received.append(crop_rgb.clone())
        premerge = torch.arange(16, dtype=torch.float32).view(4, 4)
        return SourceVisualTensorBundle(
            image_sha256=tensor_checksum(crop_rgb),
            premerge_main=premerge,
            premerge_deepstack=tuple(
                premerge.add(index + 1) for index in range(len(BRANCH_LAYERS))
            ),
            merged_main=torch.full((1, 8), 3.0),
            merged_deepstack=tuple(
                torch.full((1, 8), float(4 + index))
                for index in range(len(BRANCH_LAYERS))
            ),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
            decoded_rgb_sha256=tensor_checksum(crop_rgb),
        )


class _HiddenCapture:
    def __init__(self) -> None:
        self.forward_identity = _artifact("contextual-forward", SHA3)
        self.requests = []
        self.results = []

    def capture(self, request):
        self.requests.append(request)
        hidden_states = torch.arange(
            request.input_ids.shape[0] * 8,
            dtype=torch.float32,
            device=request.input_ids.device,
        ).reshape(request.input_ids.shape[0], 8)
        result = BehaviorHiddenStateCapture(
            identity=request.call.identity,
            input_ids=request.input_ids,
            hidden_layer=request.hidden_layer,
            forward_identity=self.forward_identity,
            hidden_states=hidden_states,
        )
        self.results.append(result)
        return result


def _artifact(name: str, sha256: str) -> ArtifactIdentity:
    return ArtifactIdentity("fixture", name, "v1", sha256)


def _source(store: ObservationStore, trajectory_id: str):
    torch.manual_seed(7)
    pixels = torch.arange(4 * 5 * 3, dtype=torch.uint8).view(4, 5, 3)
    visual = SourceVisualTensorBundle(
        image_sha256=SHA2,
        premerge_main=torch.randn(4, 4),
        premerge_deepstack=tuple(torch.randn(4, 4) for _ in BRANCH_LAYERS),
        merged_main=torch.randn(1, 8),
        merged_deepstack=tuple(torch.randn(1, 8) for _ in BRANCH_LAYERS),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
        decoded_rgb_sha256=tensor_checksum(pixels),
    )
    binding = record_trajectory_source_visual(
        trajectory_id=trajectory_id,
        source_visual=visual,
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=tuple((1,) for _ in BRANCH_LAYERS),
        observation_store=store,
        source_rgb=pixels,
    )
    return pixels, binding


def _sampled(
    tokenizer: _Tokenizer,
    policy: PolicyVersion,
    *,
    bbox: tuple[int, int, int, int] = (0, 1, 4, 4),
    target: str = "red label",
):
    bbox_json = ",".join(str(value) for value in bbox)
    text = (
        "inspect</think>\n<tool_call>"
        '{"name":"tgvf_crop_tool","arguments":'
        f'{{"bbox_2d":[{bbox_json}],"target":"{target}"}}'
        "}</tool_call>"
    )
    token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    sampled = SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.1 for _ in token_ids),
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
        think_token_span=TokenSpan(0, text.index("</think>") + len("</think>")),
        stop_reason="stop",
        backend_request_sha256=SHA2,
        backend_response_sha256=SHA3,
    )
    return sampled, StrictToolCallParser().parse(sampled.parser_turn())


def _fixture(
    tmp_path: Path,
    *,
    bbox: tuple[int, int, int, int] = (0, 1, 4, 4),
    target: str = "red label",
    provider_kind: str = "target_token_embedding",
):
    tokenizer = _Tokenizer()
    binding, adapter = _write_artifact(
        tmp_path / f"{provider_kind}.pt",
        provider=provider_kind,
    )
    loaded_adapter = load_frozen_tgvf_adapter(
        binding=binding,
        adapter=adapter,
    )
    model = binding.model
    policy = PolicyVersion("pilot", 0, SHA1)
    identity = TrajectoryIdentity("pilot", "sample", 0, "group")
    store = ObservationStore()
    pixels, source = _source(store, identity.canonical_id)
    sampled, parsed = _sampled(tokenizer, policy, bbox=bbox, target=target)
    context = ToolExecutionContext(
        trajectory_identity=identity,
        model=model,
        behavior_policy=policy,
        trajectory_source_visual=source,
        prior_observation_handles=(),
        prompt_token_ids_before_turn=tuple(
            tokenizer.encode(
                QWEN_NATIVE_IMAGE_PLACEHOLDER + "Q",
                add_special_tokens=False,
            )
        ),
        sampled_turn=sampled,
        assistant_turn_index=0,
        attempt_index=0,
        call_index=0,
    )
    if provider_kind == "target_token_embedding":
        provider_owner = (
            nn.Embedding(model.tokenizer_length, 8).requires_grad_(False).eval()
        )
        provider = TargetTokenEmbeddingConditionProvider(
            model_identity=model,
            embedding=provider_owner,
            embedding_identity=binding.conditioning.embedding_identity or "",
        )
        hidden_state_capture = None
        contextual_forward_identity = None
    elif provider_kind == "contextual_hidden_state":
        provider_owner = _HiddenCapture()
        provider = ContextualHiddenStateConditionProvider(
            model_identity=model,
            hidden_layer=binding.conditioning.hidden_layer or -1,
        )
        hidden_state_capture = provider_owner
        contextual_forward_identity = provider_owner.forward_identity
    else:
        raise ValueError(f"unsupported fixture provider {provider_kind!r}")

    def get_rope_index(*, input_ids, image_grid_thw, **_kwargs):
        assert image_grid_thw.tolist() == [[1, 2, 2], [1, 2, 2]]
        sequence = input_ids.shape[-1]
        positions = torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1)
        return positions, torch.zeros(1, 1, dtype=torch.long)

    layout = Qwen3NativeToolLayoutBuilder(
        tokenizer=tokenizer,
        model_identity=model,
        observation_store=store,
        get_rope_index=get_rope_index,
    )
    materializer = _Materializer()
    runtime = AtomicCropTGVFToolRuntime(
        conditioning_provider=provider,
        hidden_state_capture=hidden_state_capture,
        atomic_tool=AtomicCropTGVFTool(
            materializer=materializer,
            adapter=loaded_adapter.adapter,
            store=store,
            coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        ),
        layout_builder=layout,
        loaded_adapter=loaded_adapter,
        branch_mergers=_branch_bindings(binding.adapter_contract),
        crop_processor_identity=_artifact("crop-processor", SHA2),
        crop_layout_identity=_artifact("crop-layout", SHA3),
        conditioning_input_device=torch.device("cpu"),
        contextual_forward_identity=contextual_forward_identity,
        execution_ledger=FocusExecutionLedger(),
    )
    return runtime, materializer, store, pixels, provider_owner, context, parsed


def test_embedding_runtime_executes_atomic_crop_and_tgvf_once(
    tmp_path: Path,
) -> None:
    runtime, materializer, store, pixels, embedding, context, parsed = _fixture(
        tmp_path
    )

    handle = runtime.execute(parsed, context)
    repeated = runtime.execute(parsed, context)

    assert repeated == handle
    assert len(materializer.received) == 1
    torch.testing.assert_close(
        materializer.received[0], pixels[1:4, 0:4, :], rtol=0, atol=0
    )
    record = store.resolve_record(handle)
    assert isinstance(record, CropTGVFObservationRecord)
    assert record.requested_bbox_2d == (0, 1, 4, 4)
    assert record.effective_bbox_2d == (0, 1, 4, 4)
    assert record.condition.provider == "target_token_embedding"
    assert (
        record.condition.embedding_identity
        == "language_model.input_embeddings@fixture"
    )
    assert record.condition.contextual_forward_identity is None
    assert record.condition.conditioning_target_token_start == (
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_start
    )
    assert record.layout.original_image_positions == (1,)
    assert len(record.layout.d_positions) == 1
    assert record.layout.deepstack_branch_layers == BRANCH_LAYERS
    tokenizer = runtime.layout_builder.tokenizer
    expected_environment_ids = tuple(
        tokenizer.encode(
            render_qwen_native_success_environment_text(parsed),
            add_special_tokens=False,
        )
    )
    assert record.layout.sequence_length == len(
        context.prompt_token_ids_before_turn
        + context.sampled_turn.token_ids
        + expected_environment_ids
    )
    assert all(not parameter.requires_grad for parameter in embedding.parameters())


def test_contextual_runtime_binds_exact_capture_and_fused_condition_provenance(
    tmp_path: Path,
) -> None:
    runtime, materializer, store, _pixels, capture, context, parsed = _fixture(
        tmp_path,
        provider_kind="contextual_hidden_state"
    )
    assert isinstance(capture, _HiddenCapture)

    handle = runtime.execute(parsed, context)
    assert runtime.execute(parsed, context) == handle

    assert len(capture.requests) == len(capture.results) == 1
    assert len(materializer.received) == 1
    capture_request = capture.requests[0]
    captured = capture.results[0]
    expected_span = TokenSpan(
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_start,
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_end,
    )
    assert capture_request.call.target_span == expected_span
    assert capture_request.call.parsed_call.target_span == parsed.target_span
    assert capture_request.call.identity.trajectory_id == (
        context.trajectory_identity.canonical_id
    )
    assert capture_request.call.identity.behavior_policy == context.behavior_policy
    assert capture_request.call.identity.model == context.model
    assert capture_request.call.identity.contextual_forward_identity == (
        capture.forward_identity
    )
    assert capture_request.hidden_layer == -1
    assert tuple(capture_request.input_ids.tolist()) == context.conditioning_input_ids
    assert captured.identity == capture_request.call.identity
    assert captured.input_ids is capture_request.input_ids
    assert captured.forward_identity == capture.forward_identity

    record = store.resolve_record(handle)
    assert isinstance(record, CropTGVFObservationRecord)
    assert record.condition.provider == "contextual_hidden_state"
    assert record.condition.hidden_layer == -1
    assert record.condition.contextual_forward_identity == capture.forward_identity
    assert record.condition.embedding_identity is None
    assert record.condition.policy_version == context.behavior_policy
    assert record.condition.trajectory_ids == (
        context.trajectory_identity.canonical_id,
    )
    assert record.condition.call_indices == (context.call_index,)
    assert record.condition.source_input_ids_sha256 == (
        capture_request.call.identity.source_input_ids_sha256
    )
    assert (
        record.condition.sampled_target_token_start,
        record.condition.sampled_target_token_end,
    ) == (parsed.target_span.token_start, parsed.target_span.token_end)
    assert (
        record.condition.conditioning_target_token_start,
        record.condition.conditioning_target_token_end,
    ) == (expected_span.start, expected_span.end)


def test_runtime_rejects_provider_not_named_by_loaded_artifact(
    tmp_path: Path,
) -> None:
    runtime, materializer, store, _pixels, _capture, _context, _parsed = _fixture(
        tmp_path,
        provider_kind="contextual_hidden_state",
    )
    embedding = (
        nn.Embedding(runtime.loaded_adapter.binding.model.tokenizer_length, 8)
        .requires_grad_(False)
        .eval()
    )
    wrong_provider = TargetTokenEmbeddingConditionProvider(
        model_identity=runtime.loaded_adapter.binding.model,
        embedding=embedding,
        embedding_identity="language_model.input_embeddings@fixture",
    )

    with pytest.raises(
        IdentityMismatchError,
        match="provider differs from representation artifact",
    ):
        AtomicCropTGVFToolRuntime(
            conditioning_provider=wrong_provider,
            hidden_state_capture=None,
            atomic_tool=AtomicCropTGVFTool(
                materializer=materializer,
                adapter=runtime.loaded_adapter.adapter,
                store=store,
                coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
            ),
            layout_builder=runtime.layout_builder,
            loaded_adapter=runtime.loaded_adapter,
            branch_mergers=runtime.branch_mergers,
            crop_processor_identity=runtime.crop_processor_identity,
            crop_layout_identity=runtime.crop_layout_identity,
            conditioning_input_device=torch.device("cpu"),
            contextual_forward_identity=None,
            execution_ledger=FocusExecutionLedger(),
        )


def test_runtime_rejects_adapter_not_loaded_by_selected_binding(
    tmp_path: Path,
) -> None:
    runtime, materializer, store, _pixels, _embedding, _context, _parsed = _fixture(
        tmp_path
    )
    other_binding, other_adapter = _write_artifact(
        tmp_path / "other.pt",
        provider="target_token_embedding",
    )
    other_loaded = load_frozen_tgvf_adapter(
        binding=other_binding,
        adapter=other_adapter,
    )

    with pytest.raises(
        IdentityMismatchError,
        match="differs from the loaded representation artifact",
    ):
        AtomicCropTGVFToolRuntime(
            conditioning_provider=runtime.conditioning_provider,
            hidden_state_capture=None,
            atomic_tool=AtomicCropTGVFTool(
                materializer=materializer,
                adapter=other_loaded.adapter,
                store=store,
                coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
            ),
            layout_builder=runtime.layout_builder,
            loaded_adapter=runtime.loaded_adapter,
            branch_mergers=runtime.branch_mergers,
            crop_processor_identity=runtime.crop_processor_identity,
            crop_layout_identity=runtime.crop_layout_identity,
            conditioning_input_device=torch.device("cpu"),
            contextual_forward_identity=None,
            execution_ledger=FocusExecutionLedger(),
        )


def test_runtime_rejects_branch_bindings_outside_artifact_architecture(
    tmp_path: Path,
) -> None:
    runtime, materializer, store, _pixels, _embedding, _context, _parsed = _fixture(
        tmp_path
    )
    wrong_branches = (
        replace(runtime.branch_mergers[0], projection_identity="wrong.projection"),
        *runtime.branch_mergers[1:],
    )

    with pytest.raises(
        IdentityMismatchError,
        match="branch merger bindings differ from representation architecture",
    ):
        AtomicCropTGVFToolRuntime(
            conditioning_provider=runtime.conditioning_provider,
            hidden_state_capture=None,
            atomic_tool=AtomicCropTGVFTool(
                materializer=materializer,
                adapter=runtime.loaded_adapter.adapter,
                store=store,
                coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
            ),
            layout_builder=runtime.layout_builder,
            loaded_adapter=runtime.loaded_adapter,
            branch_mergers=wrong_branches,
            crop_processor_identity=runtime.crop_processor_identity,
            crop_layout_identity=runtime.crop_layout_identity,
            conditioning_input_device=torch.device("cpu"),
            contextual_forward_identity=None,
            execution_ledger=FocusExecutionLedger(),
        )


def test_execute_once_rejects_changed_atomic_call_without_reexecution(
    tmp_path: Path,
) -> None:
    runtime, materializer, _store, _pixels, _embedding, context, parsed = _fixture(
        tmp_path
    )
    runtime.execute(parsed, context)
    changed_sampled, changed_parsed = _sampled(
        _Tokenizer(),
        context.behavior_policy,
        bbox=(1, 1, 5, 4),
        target="blue label",
    )

    with pytest.raises(ValueError, match="reused with different content"):
        runtime.execute(
            changed_parsed,
            replace(context, sampled_turn=changed_sampled),
        )

    assert len(materializer.received) == 1


def test_execute_once_binds_exact_sampled_token_byte_identity(
    tmp_path: Path,
) -> None:
    runtime, materializer, _store, _pixels, _embedding, context, parsed = _fixture(
        tmp_path
    )
    handle = runtime.execute(parsed, context)

    changed_spans = list(context.sampled_turn.token_byte_spans)
    changed_spans[0] = TokenByteSpan(
        token_index=0,
        token_id=context.sampled_turn.token_ids[0],
        byte_start=0,
        byte_end=0,
    )
    changed_spans[1] = TokenByteSpan(
        token_index=1,
        token_id=context.sampled_turn.token_ids[1],
        byte_start=0,
        byte_end=2,
    )
    changed_turn = replace(
        context.sampled_turn,
        token_byte_spans=tuple(changed_spans),
    )
    changed_parsed = StrictToolCallParser().parse(changed_turn.parser_turn())

    assert changed_parsed.sampled_text == parsed.sampled_text
    assert changed_parsed.sampled_token_ids == parsed.sampled_token_ids
    assert changed_parsed.sampled_token_byte_spans != parsed.sampled_token_byte_spans
    with pytest.raises(ValueError, match="reused with different content"):
        runtime.execute(
            changed_parsed,
            replace(context, sampled_turn=changed_turn),
        )

    assert runtime.execute(parsed, context) == handle
    assert len(materializer.received) == 1


def test_empty_clamped_bbox_is_a_recoverable_tool_error(tmp_path: Path) -> None:
    runtime, materializer, _store, _pixels, _embedding, context, parsed = _fixture(
        tmp_path,
        bbox=(20, 20, 30, 30)
    )

    with pytest.raises(RecoverableToolExecutionError, match="empty after clamping"):
        runtime.execute(parsed, context)

    assert materializer.received == []
