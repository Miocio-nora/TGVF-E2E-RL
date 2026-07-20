from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread
import time

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    ContextualHiddenStateConditionProvider,
    TargetTokenEmbeddingConditionProvider,
)
from tgvf_rl.conditioning.base import _bind_canonical_input_ids
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    SamplingIdentity,
    TokenSpan,
)
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.focus_runtime import (
    BehaviorHiddenStateCapture,
    BoundReplayLayout,
    BoundSourceVisual,
    FocusExecutionLedger,
    FocusRuntimeCallIdentity,
    TGVFFocusToolRuntime,
    _LedgerEntry,
    _source_binding_sha256,
)
from tgvf_rl.environment.focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
    TGVFFocusTool,
)
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.schema import VisualLayout
from tgvf_rl.observations.store import ObservationHandle, ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import TokenByteSpan
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.deepstack import FrozenProjectionPort
from tgvf_rl.trajectories.schema import TrajectoryIdentity


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(16, 8, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.reshape(-1, 16))


class _HiddenCapture:
    def __init__(self) -> None:
        self.requests = []
        self.identity_override = None
        self.layer_offset = 0
        self.forward_identity = ArtifactIdentity(
            "policy", "contextual-forward", "fixture", SHA3
        )
        self.forward_identity_override = None

    def capture(self, request):
        self.requests.append(request)
        sequence_length = request.input_ids.shape[0]
        hidden = torch.arange(
            sequence_length * 8,
            dtype=torch.float32,
            device=request.input_ids.device,
        ).reshape(sequence_length, 8)
        return BehaviorHiddenStateCapture(
            identity=self.identity_override or request.call.identity,
            input_ids=request.input_ids,
            hidden_layer=request.hidden_layer + self.layer_offset,
            forward_identity=(self.forward_identity_override or self.forward_identity),
            hidden_states=hidden,
        )


class _SourcePort:
    def __init__(self, tensors: SourceVisualTensorBundle) -> None:
        self.tensors = tensors
        self.requests = []
        self.identity_override = None

    def resolve(self, request):
        self.requests.append(request)
        return BoundSourceVisual(
            self.identity_override or request.identity,
            self.tensors,
        )


class _LayoutPort:
    def __init__(self, tensors: ReplayLayoutTensors) -> None:
        self.tensors = tensors
        self.requests = []
        self.sources = []
        self.identity_override = None

    def resolve(self, request, source_visual):
        self.requests.append(request)
        self.sources.append(source_visual)
        return BoundReplayLayout(
            self.identity_override or request.identity,
            self.tensors,
        )


class _CountingFocusTool(TGVFFocusTool):
    def __init__(self, adapter: TGVFAdapter, store: ObservationStore) -> None:
        super().__init__(adapter, store)
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return super().execute(request)


def _model(name: str = "fixture") -> ModelIdentity:
    return ModelIdentity("qwen3_vl", name, f"/{name}", 256, SHA0)


def _policy(step: int = 4) -> PolicyVersion:
    return PolicyVersion("pilot", step, SHA1)


def _sampled(target: str = "red label"):
    text = (
        "inspect</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"'
        f"{target}"
        '"}}</tool_call>'
    )
    token_ids = tuple(ord(character) for character in text)
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    policy = _policy()
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
            rng_state_sha256=SHA2,
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


def _context(source_binding, *, model: ModelIdentity | None = None):
    sampled, _ = _sampled()
    return ToolExecutionContext(
        trajectory_identity=TrajectoryIdentity("pilot", "sample", 2, "group"),
        model=model or _model(),
        behavior_policy=_policy(),
        trajectory_source_visual=source_binding,
        prior_observation_handles=(ObservationHandle("prior", SHA0),),
        prompt_token_ids_before_turn=(7, 8, 9, 10),
        sampled_turn=sampled,
        assistant_turn_index=1,
        attempt_index=2,
        call_index=1,
    )


def _visual() -> SourceVisualTensorBundle:
    return SourceVisualTensorBundle(
        image_sha256=SHA2,
        premerge_main=torch.randn(4, 4),
        premerge_deepstack=(torch.randn(4, 4),),
        merged_main=torch.randn(1, 8),
        merged_deepstack=(torch.randn(1, 8),),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
    )


def _layout() -> ReplayLayoutTensors:
    mask = torch.ones(1, 8, dtype=torch.bool)
    return ReplayLayoutTensors(
        position_ids=torch.arange(8).view(1, 8),
        attention_mask=mask,
        policy_visible_mask=mask.clone(),
        reference_visible_mask=mask.clone(),
        teacher_visible_mask=mask.clone(),
        token_type_ids=None,
        original_image_key_block_mask=None,
        cache_position=None,
        rope_delta=None,
        visual_layout=VisualLayout(
            sequence_length=8,
            original_image_positions=(1,),
            d_positions=(4,),
            deepstack_branch_layers=(8,),
            deepstack_injection_positions=((4,),),
        ),
    )


def _adapter() -> TGVFAdapter:
    return TGVFAdapter(
        d_lm=8,
        d_v=4,
        main_projection=FrozenProjectionPort(
            _Merger(),
            identity="main",
            input_dim=4,
            output_dim=8,
            spatial_merge_size=2,
        ),
        deepstack_projections=(
            FrozenProjectionPort(
                _Merger(),
                identity="branch-8",
                input_dim=4,
                output_dim=8,
                spatial_merge_size=2,
            ),
        ),
        branch_layers=(8,),
    )


def _runtime(provider_kind: str = "contextual_hidden_state"):
    model = _model()
    if provider_kind == "contextual_hidden_state":
        embedding_owner = None
        provider = ContextualHiddenStateConditionProvider(
            model_identity=model,
            hidden_layer=3,
        )
        hidden_layer = 3
    else:
        embedding_owner = nn.Embedding(model.tokenizer_length, 8)
        provider = TargetTokenEmbeddingConditionProvider(
            model_identity=model,
            embedding=embedding_owner,
            embedding_identity="model.embed_tokens",
        )
        hidden_layer = None
    hidden = _HiddenCapture()
    visual = _visual()
    source = _SourcePort(visual)
    layout = _LayoutPort(_layout())
    store = ObservationStore()
    source_binding = record_trajectory_source_visual(
        trajectory_id="pilot/sample/2/group",
        source_visual=visual,
        source_positions=(1,),
        deepstack_branch_layers=(8,),
        deepstack_injection_positions=((1,),),
        observation_store=store,
    )
    tool = _CountingFocusTool(_adapter(), store)
    runtime = TGVFFocusToolRuntime(
        conditioning_provider=provider,
        hidden_state_capture=hidden,
        source_visual=source,
        replay_layout=layout,
        focus_tool=tool,
        representation=ArtifactIdentity("tgvf", "adapter", "pilot", SHA2),
        branch_merger_identities=(ArtifactIdentity("qwen", "merger-8", "pilot", SHA3),),
        conditioning_input_device=torch.device("cpu"),
        contextual_hidden_layer=hidden_layer,
        contextual_forward_identity=(
            hidden.forward_identity
            if provider_kind == "contextual_hidden_state"
            else None
        ),
        execution_ledger=FocusExecutionLedger(),
    )
    return runtime, hidden, source, layout, tool, source_binding, embedding_owner


@pytest.mark.parametrize(
    ("provider_kind", "expected_hidden_calls"),
    (("contextual_hidden_state", 1), ("target_token_embedding", 0)),
)
def test_runtime_offsets_exact_span_and_executes_every_source_once(
    provider_kind: str,
    expected_hidden_calls: int,
) -> None:
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime(
        provider_kind
    )
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())

    handle = runtime.execute(parsed, context)
    repeated = runtime.execute(parsed, context)

    assert repeated == handle
    assert len(hidden.requests) == expected_hidden_calls
    assert len(source.requests) == len(layout.requests) == len(tool.requests) == 1
    call = source.requests[0]
    expected_span = TokenSpan(
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_start,
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_end,
    )
    assert call.conditioning_input_ids == (
        context.prompt_token_ids_before_turn + context.sampled_turn.token_ids
    )
    assert call.target_span == expected_span
    assert tool.requests[0].condition.provenance.target_span == expected_span
    assert tool.requests[0].source_visual is source.tensors
    assert tool.requests[0].layout is layout.tensors
    assert layout.sources[0].tensors is source.tensors
    record = tool.store.resolve_record(handle)
    assert record.call_index == context.call_index
    assert (
        record.condition.sampled_target_token_start,
        record.condition.sampled_target_token_end,
    ) == (parsed.target_span.token_start, parsed.target_span.token_end)
    assert (
        record.condition.conditioning_target_token_start,
        record.condition.conditioning_target_token_end,
    ) == (expected_span.start, expected_span.end)
    if provider_kind == "contextual_hidden_state":
        capture = hidden.requests[0]
        assert capture.hidden_layer == 3
        assert tuple(capture.input_ids.tolist()) == call.conditioning_input_ids
        assert capture.call.identity.prior_observation_handles == (
            ObservationHandle("prior", SHA0),
        )
        assert record.condition.contextual_forward_identity == (hidden.forward_identity)
    else:
        assert record.condition.contextual_forward_identity is None
    different_sampled, different_parsed = _sampled("blue label")
    with pytest.raises(ValueError, match="reused with different content"):
        runtime.execute(
            different_parsed,
            replace(context, sampled_turn=different_sampled),
        )
    assert len(source.requests) == len(layout.requests) == len(tool.requests) == 1


def test_runtime_rejects_parsed_turn_and_bound_model_mismatch_before_ports() -> None:
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime()
    context = _context(source_binding)
    _, other = _sampled("blue label")
    with pytest.raises(ValueError, match="exact sampled assistant turn"):
        runtime.execute(other, context)

    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())
    with pytest.raises(ValueError, match="target-conditioning provider"):
        runtime.execute(parsed, _context(source_binding, model=_model("other")))
    assert not hidden.requests
    assert not source.requests
    assert not layout.requests
    assert not tool.requests


def test_execute_once_rejects_changed_sampled_token_byte_identity() -> None:
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime()
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())

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
    assert changed_parsed.sampled_token_byte_spans != (parsed.sampled_token_byte_spans)
    with pytest.raises(ValueError, match="reused with different content"):
        runtime.execute(changed_parsed, replace(context, sampled_turn=changed_turn))

    assert runtime.execute(parsed, context) == handle
    assert len(hidden.requests) == len(source.requests) == 1
    assert len(layout.requests) == len(tool.requests) == 1


def test_ledger_release_wakes_waiter_without_reexecuting_released_call() -> None:
    ledger = FocusExecutionLedger()
    trajectory_id = "pilot/waiter-release/0/group"
    key = (trajectory_id, 0)
    handle = ObservationHandle("completed-before-release", SHA0)
    fingerprint = "waiter-release-fingerprint"
    with ledger._condition:
        ledger._entries[key] = _LedgerEntry(fingerprint=fingerprint)

    operation_started = Event()
    errors: list[BaseException] = []

    def wait_for_existing_call() -> None:
        try:
            ledger.execute_once(
                key=key,
                fingerprint=fingerprint,
                operation=lambda: operation_started.set() or handle,
            )
        except BaseException as error:
            errors.append(error)

    waiter = Thread(target=wait_for_existing_call)
    waiter.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with ledger._condition:
            if ledger._condition._waiters:
                break
        time.sleep(0.001)
    else:
        pytest.fail("focus execute-once caller did not enter the wait state")

    # Hold the condition across completion and release so the notified waiter
    # can resume only after the completed entry has been removed.
    with ledger._condition:
        entry = ledger._entries[key]
        entry.handle = handle
        entry.running = False
        ledger._condition.notify_all()
        assert ledger.release_trajectories((trajectory_id,)) == 1

    waiter.join(timeout=2.0)
    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "already been released" in str(errors[0])
    assert not operation_started.is_set()


def test_second_call_capture_includes_prior_observation_and_full_prefix_offset() -> (
    None
):
    runtime, hidden, _, _, _, source_binding, _ = _runtime()
    prior = ObservationHandle("first-focus-observation", SHA3)
    context = replace(
        _context(source_binding),
        prompt_token_ids_before_turn=(11, 12, 201, 13, 202, 203),
        prior_observation_handles=(prior,),
    )
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())

    runtime.execute(parsed, context)

    captured_call = hidden.requests[0].call
    assert captured_call.identity.prior_observation_handles == (prior,)
    assert captured_call.target_span == TokenSpan(
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_start,
        len(context.prompt_token_ids_before_turn) + parsed.target_span.token_end,
    )


def test_runtime_rejects_behavior_policy_and_call_identity_drift() -> None:
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime()
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())

    wrong_policy = PolicyVersion("pilot", 5, SHA3)
    hidden.identity_override = replace(
        _expected_identity(runtime, parsed, context), behavior_policy=wrong_policy
    )
    with pytest.raises(ValueError, match="capture identity"):
        runtime.execute(parsed, context)
    assert not source.requests and not layout.requests and not tool.requests

    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime(
        "target_token_embedding"
    )
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())
    source.identity_override = replace(
        _expected_identity(runtime, parsed, context), behavior_policy=wrong_policy
    )
    with pytest.raises(ValueError, match="source visual identity"):
        runtime.execute(parsed, context)
    assert not hidden.requests and not layout.requests and not tool.requests

    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime(
        "target_token_embedding"
    )
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())
    layout.identity_override = replace(
        _expected_identity(runtime, parsed, context),
        call_index=2,
        prior_observation_handles=(
            *context.prior_observation_handles,
            ObservationHandle("other-prior", SHA1),
        ),
    )
    with pytest.raises(ValueError, match="replay layout identity"):
        runtime.execute(parsed, context)
    assert not hidden.requests
    assert len(source.requests) == len(layout.requests) == 1
    assert not tool.requests


def test_contextual_runtime_rejects_capture_layer_drift_without_tool_execution() -> (
    None
):
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime()
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())
    hidden.layer_offset = 1

    with pytest.raises(ValueError, match="different layer"):
        runtime.execute(parsed, context)

    assert len(hidden.requests) == 1
    assert not source.requests and not layout.requests and not tool.requests


def test_contextual_runtime_rejects_capture_forward_identity_drift() -> None:
    runtime, hidden, source, layout, tool, source_binding, _embedding_owner = _runtime()
    context = _context(source_binding)
    parsed = StrictToolCallParser().parse(context.sampled_turn.parser_turn())
    hidden.forward_identity_override = ArtifactIdentity(
        "policy", "other-contextual-forward", "fixture", "9" * 64
    )

    with pytest.raises(ValueError, match="different forward identity"):
        runtime.execute(parsed, context)

    assert len(hidden.requests) == 1
    assert not source.requests and not layout.requests and not tool.requests


def _expected_identity(runtime, parsed, context):
    del parsed
    input_ids = torch.tensor(context.conditioning_input_ids, dtype=torch.long)
    proof = _bind_canonical_input_ids(input_ids, context.conditioning_input_ids)
    return FocusRuntimeCallIdentity(
        trajectory_id=context.trajectory_identity.canonical_id,
        assistant_turn_index=context.assistant_turn_index,
        attempt_index=context.attempt_index,
        call_index=context.call_index,
        model=context.model,
        behavior_policy=context.behavior_policy,
        contextual_forward_identity=runtime.contextual_forward_identity,
        source_input_ids_sha256=proof.digest,
        source_binding_sha256=_source_binding_sha256(context.trajectory_source_visual),
        prior_observation_handles=context.prior_observation_handles,
    )
