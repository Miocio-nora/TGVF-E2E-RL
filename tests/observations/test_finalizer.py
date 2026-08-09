from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.framework.verl.objective_bridge import make_objective_sentinels
from tgvf_rl.observations import (
    MaterializedTrajectoryReplayTensors,
    ObservationStore,
    TrajectoryReplayFinalizationRequest,
    finalize_trajectory_replay,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    ToolCallRecord,
    ToolObservationRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tests.support import (
    SHA0,
    SHA1,
    populated_observation_store,
    policy_version,
    trajectory_source_visual,
)


FINAL_SEQUENCE_LENGTH = 12


def _completed_trajectory(call_count: int):
    if call_count not in {0, 1, 2}:
        raise ValueError("fixture supports zero, one, or two calls")
    store, first_handle = populated_observation_store()
    first = store.resolve_record(first_handle)
    handles = [first_handle]
    targets = ["red label"]
    if call_count == 2:
        target = "lower label"
        d_positions = (8, 9)
        second = replace(
            first,
            observation_id="observation-1-finalizer",
            call_index=1,
            condition=replace(
                first.condition,
                sampled_target_text_sha256=hashlib.sha256(
                    target.encode("utf-8")
                ).hexdigest(),
                call_indices=(1,),
            ),
            branches=tuple(
                replace(branch, injection_positions=d_positions)
                for branch in first.branches
            ),
            layout=replace(
                first.layout,
                d_positions=d_positions,
                deepstack_injection_positions=tuple(
                    d_positions for _ in first.branches
                ),
            ),
        )
        handles.append(store.put(second))
        targets.append(target)

    identity = TrajectoryIdentity("smoke", "sample", 0, "group")
    policy = policy_version()
    behavior_store = BehaviorTraceStore()
    recorder = VLLMBehaviorRecorder(behavior_store)
    assistant_turns = []
    tool_calls = []
    observations = []
    response_ids: tuple[int, ...] = ()
    native_rows: tuple[tuple[int, ...], ...]

    if call_count == 0:
        native_rows = ()
        token_rows = ((20, 21, 22, 23),)
    else:
        native_rows = tuple(
            (100 + 2 * index, 101 + 2 * index) for index in range(call_count)
        )
        token_rows = tuple(
            (20 + 3 * index, 21 + 3 * index, 22 + 3 * index)
            for index in range(call_count)
        )

    for turn_index, token_ids in enumerate(token_rows):
        tokens = OwnedTokenSequence(
            token_ids,
            (TokenOwnership.POLICY_SAMPLED,) * len(token_ids),
        )
        sampling = SamplingIdentity(
            policy_version=policy,
            backend="vllm",
            backend_version="finalizer-fixture",
            seed=42 + turn_index,
            rng_state_sha256=SHA0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        )
        trace = recorder.record(
            trajectory_id=identity.canonical_id,
            assistant_turn_index=turn_index,
            tokens=tokens,
            actual_sampled_logprobs=tuple(-0.1 * (index + 1) for index in token_ids),
            sampling=sampling,
            behavior_policy=policy,
            backend_request_sha256=SHA0,
            backend_response_sha256=SHA1,
        )
        is_tool_call = call_count > 0
        assistant_turns.append(
            AssistantTurnRecord(
                turn_index=turn_index,
                raw_text=(
                    "reason</think><tool_call>fixture</tool_call>"
                    if is_tool_call
                    else "reason</think>answer"
                ),
                tokens=tokens,
                behavior_trace=trace,
                think_span=TokenSpan(0, min(2, len(token_ids))),
                is_tool_call=is_tool_call,
            )
        )
        response_ids += token_ids
        if is_tool_call:
            target = targets[turn_index]
            tool_calls.append(
                ToolCallRecord(
                    call_index=turn_index,
                    assistant_turn_index=turn_index,
                    function_name="tgvf_focus_tool",
                    target=target,
                    target_token_span=TokenSpan(1, 3),
                    target_char_span=(0, len(target)),
                    raw_call_text=f"fixture-{turn_index}",
                )
            )
            observations.append(
                ToolObservationRecord(
                    call_index=turn_index,
                    handle=handles[turn_index],
                    template_token_ids=native_rows[turn_index],
                )
            )
            response_ids += native_rows[turn_index]

    trajectory = TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=identity,
        model=first.model,
        behavior_policy=policy,
        assistant_turns=tuple(assistant_turns),
        tool_calls=tuple(tool_calls),
        observations=tuple(observations),
        final_answer="answer" if call_count == 0 else None,
        stop=(
            TrajectoryStop.DIRECT_ANSWER
            if call_count == 0
            else TrajectoryStop.MAX_TOKENS
        ),
    )
    prompt_length = FINAL_SEQUENCE_LENGTH - len(response_ids)
    prompt_ids = tuple(range(1, prompt_length + 1))
    final_ids = prompt_ids + response_ids
    assert len(final_ids) == FINAL_SEQUENCE_LENGTH
    mask = torch.ones(1, FINAL_SEQUENCE_LENGTH, dtype=torch.bool)
    tensors = MaterializedTrajectoryReplayTensors(
        input_ids=torch.tensor([final_ids], dtype=torch.long),
        position_ids=torch.arange(FINAL_SEQUENCE_LENGTH).view(1, -1),
        base_attention_mask=mask,
        policy_attention_mask=mask.clone(),
        reference_attention_mask=mask.clone(),
        teacher_attention_mask=mask.clone(),
        token_type_ids=torch.zeros(1, FINAL_SEQUENCE_LENGTH, dtype=torch.long),
        cache_position=(
            torch.arange(100, 100 + FINAL_SEQUENCE_LENGTH, dtype=torch.long)
            if call_count == 2
            else None
        ),
        rope_delta=(torch.tensor([0], dtype=torch.long) if call_count == 2 else None),
    )
    request = TrajectoryReplayFinalizationRequest(
        trajectory=trajectory,
        source_visual=trajectory_source_visual(first),
        tensors=tensors,
        replay_schema_version="trajectory-replay-v1",
        replay_id=f"finalized-{call_count}",
        trajectory_id=identity.canonical_id,
        model=trajectory.model,
        behavior_policy=policy,
        crop_vision_replay_mode="no_crop",
        cache_mode="recorded_cache" if call_count == 2 else "no_cache",
        cache_prefix_length=4 if call_count == 2 else 0,
        deterministic_forward=True,
        adapter_dropout=0.0,
        maximum_policy_staleness=0,
        initial_prompt_token_ids=prompt_ids,
        native_tool_appended_token_ids=native_rows,
        sentinel_fields=make_objective_sentinels(f"finalizer-{call_count}"),
    )
    return store, behavior_store, request


@pytest.mark.parametrize("call_count", (0, 1, 2))
def test_finalizer_freezes_zero_one_and_multi_call_replay(call_count: int) -> None:
    store, behavior_store, request = _completed_trajectory(call_count)

    bridge = finalize_trajectory_replay(
        request,
        observation_store=store,
        behavior_store=behavior_store,
    )

    replay = bridge.replay_bundle.replay_record
    assert replay.source_visual == request.source_visual
    assert replay.observation_handles == tuple(
        item.handle for item in request.trajectory.observations
    )
    assert len(bridge.replay_bundle.observation_records) == call_count
    assert bridge.prompt_ids == request.initial_prompt_token_ids
    assert bridge.replay_handle == bridge.replay_bundle.replay_handle
    assert (replay.tensors.cache_position is not None) is (call_count == 2)
    assert (replay.tensors.rope_delta is not None) is (call_count == 2)


def test_finalizer_preserves_direct_only_invalid_tool_marker_replay() -> None:
    store, behavior_store, request = _completed_trajectory(0)
    original = request.trajectory
    trajectory = replace(
        original,
        assistant_turns=(
            replace(
                original.assistant_turns[0],
                raw_text=(
                    '<tool_call>{"name":"tgvf_focus_tool",'
                    '"arguments":{"target":"x"}}</tool_call>'
                ),
                think_span=None,
                is_tool_call=True,
            ),
        ),
        final_answer=None,
        stop=TrajectoryStop.INVALID_FORMAT,
    )

    bridge = finalize_trajectory_replay(
        replace(request, trajectory=trajectory, replay_id="direct-only-marker"),
        observation_store=store,
        behavior_store=behavior_store,
    )

    assert bridge.trajectory_payload == trajectory
    assert bridge.response_ids == trajectory.assistant_turns[0].tokens.token_ids
    assert bridge.response_mask == (1,) * len(bridge.response_ids)
    assert bridge.exact_observation_handles == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("trajectory_id", "another/trajectory", "trajectory identity"),
        (
            "model",
            ModelIdentity("qwen3_vl", "other", "/other", 151669, "f" * 64),
            "model identity",
        ),
        (
            "behavior_policy",
            PolicyVersion("other", 1, "e" * 64),
            "policy identity",
        ),
    ),
)
def test_finalizer_rejects_explicit_trajectory_model_and_policy_mismatch(
    field: str, value: object, message: str
) -> None:
    store, behavior_store, request = _completed_trajectory(0)

    with pytest.raises(ValueError, match=message):
        finalize_trajectory_replay(
            replace(request, **{field: value}),
            observation_store=store,
            behavior_store=behavior_store,
        )


def test_finalizer_rejects_source_visual_mismatch_with_tool_observation() -> None:
    store, behavior_store, request = _completed_trajectory(2)
    mismatched_source = replace(
        request.source_visual,
        state=replace(request.source_visual.state, image_sha256="d" * 64),
    )

    with pytest.raises(ReplayMismatchError, match="trajectory source"):
        finalize_trajectory_replay(
            replace(request, source_visual=mismatched_source),
            observation_store=store,
            behavior_store=behavior_store,
        )


def test_finalizer_clones_raw_tensors_and_bundle_detects_later_mutation() -> None:
    store, behavior_store, request = _completed_trajectory(0)
    expected_ids = request.tensors.input_ids.clone()
    bridge = finalize_trajectory_replay(
        request,
        observation_store=store,
        behavior_store=behavior_store,
    )

    request.tensors.input_ids.zero_()
    stored_ids = store.resolve_verified(bridge.replay_bundle.replay_record.tensors.input_ids)
    torch.testing.assert_close(stored_ids, expected_ids, rtol=0, atol=0)

    input_digest = bridge.replay_bundle.replay_record.tensors.input_ids.address.digest
    transported_input = next(
        payload
        for payload in bridge.replay_bundle.tensor_payloads
        if payload.sha256 == input_digest
    )
    transported_input.tensor.add_(1)
    with pytest.raises(ReplayMismatchError, match="replay tensor payload"):
        ObservationStore.from_replay_bundle(bridge.replay_bundle)
