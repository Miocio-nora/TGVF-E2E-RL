from __future__ import annotations

import json

import pytest

from tgvf_rl.protocol import (
    AgentEvent,
    AgentPhase,
    CapErrorBehavior,
    InvalidTransitionError,
    MultiCallStateMachine,
    SampledAssistantTurn,
    StrictToolCallParser,
    TGVF_FOCUS_TOOL_NAME,
    TerminationReason,
    TokenByteSpan,
)


def _parsed_call(target: str):
    payload = json.dumps(
        {"name": TGVF_FOCUS_TOOL_NAME, "arguments": {"target": target}},
        separators=(",", ":"),
    )
    text = f"<tool_call>{payload}</tool_call>"
    token_ids = tuple(range(1, len(text) + 1))
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    return StrictToolCallParser().parse(SampledAssistantTurn(text, token_ids, spans))


@pytest.mark.parametrize("cap", [True, False, -1, 0, 1, 2.5, "3", None])
def test_cap_must_be_an_integer_greater_than_one(cap: object) -> None:
    with pytest.raises(ValueError, match="greater than one"):
        MultiCallStateMachine(cap)


def test_two_sequential_calls_and_responses_then_final_answer() -> None:
    machine = MultiCallStateMachine(max_tool_calls=2)
    state = machine.initial_state()

    first = machine.apply(state, AgentEvent.valid_tool_call(_parsed_call("left label")))
    assert first.execute_tool is True
    assert first.call_index == 0
    assert first.state.phase is AgentPhase.AWAITING_TOOL_RESPONSE
    assert first.state.tool_call_count == 1
    assert first.state.successful_tool_call_count == 0

    state = machine.apply(first.state, AgentEvent.tool_response()).state
    assert state.phase is AgentPhase.AWAITING_ASSISTANT

    second = machine.apply(
        state, AgentEvent.valid_tool_call(_parsed_call("right value"))
    )
    assert second.execute_tool is True
    assert second.call_index == 1
    assert second.state.tool_call_count == 2

    state = machine.apply(second.state, AgentEvent.tool_response()).state
    finished = machine.apply(state, AgentEvent.final_answer())
    assert finished.execute_tool is False
    assert finished.state.phase is AgentPhase.TERMINATED
    assert finished.state.termination_reason is TerminationReason.FINAL_ANSWER
    assert finished.state.tool_call_count == 2
    assert finished.state.successful_tool_call_count == 2


def test_call_beyond_cap_emits_one_error_then_terminates_without_execution() -> None:
    machine = MultiCallStateMachine(max_tool_calls=2)
    state = machine.initial_state()
    for target in ("first", "second"):
        call = machine.apply(state, AgentEvent.valid_tool_call(_parsed_call(target)))
        assert call.execute_tool
        state = machine.apply(call.state, AgentEvent.tool_response()).state

    capped = machine.apply(state, AgentEvent.valid_tool_call(_parsed_call("third")))
    assert capped.execute_tool is False
    assert capped.emit_error is True
    assert capped.call_index is None
    assert capped.attempt_index == 2
    assert capped.state.phase is AgentPhase.AWAITING_TOOL_RESPONSE
    assert capped.state.tool_call_count == 3
    terminated = machine.apply(capped.state, AgentEvent.tool_error())
    assert terminated.state.phase is AgentPhase.TERMINATED
    assert terminated.state.termination_reason is TerminationReason.TOOL_CALL_CAP


def test_malformed_action_consumes_attempt_emits_error_and_recovers() -> None:
    machine = MultiCallStateMachine(3)
    result = machine.apply(machine.initial_state(), AgentEvent.malformed_action())
    assert result.emit_error is True
    assert result.attempt_index == 0
    recovered = machine.apply(result.state, AgentEvent.tool_error()).state
    assert recovered.phase is AgentPhase.AWAITING_ASSISTANT
    assert recovered.tool_attempt_count == 1
    assert recovered.successful_tool_call_count == 0


def test_assistant_timeout_terminates() -> None:
    machine = MultiCallStateMachine(3)
    result = machine.apply(machine.initial_state(), AgentEvent.timeout())
    assert result.state.termination_reason is TerminationReason.TIMEOUT


def test_tool_error_returns_to_assistant_without_counting_success() -> None:
    machine = MultiCallStateMachine(3)
    pending = machine.apply(
        machine.initial_state(),
        AgentEvent.valid_tool_call(_parsed_call("serial number")),
    ).state
    result = machine.apply(pending, AgentEvent.tool_error())
    assert result.state.phase is AgentPhase.AWAITING_ASSISTANT
    assert result.state.tool_call_count == 1
    assert result.state.successful_tool_call_count == 0


def test_cap_error_can_allow_exactly_one_final_answer_turn() -> None:
    machine = MultiCallStateMachine(
        2, CapErrorBehavior.ONE_FINAL_ANSWER_TURN
    )
    state = machine.initial_state()
    for target in ("first", "second"):
        call = machine.apply(state, AgentEvent.valid_tool_call(_parsed_call(target)))
        state = machine.apply(call.state, AgentEvent.tool_response()).state
    cap = machine.apply(state, AgentEvent.valid_tool_call(_parsed_call("third")))
    final_only = machine.apply(cap.state, AgentEvent.tool_error()).state
    assert final_only.phase is AgentPhase.AWAITING_FINAL_ANSWER
    terminated = machine.apply(final_only, AgentEvent.final_answer()).state
    assert terminated.termination_reason is TerminationReason.FINAL_ANSWER


def test_invalid_event_order_and_post_termination_transition_fail_closed() -> None:
    machine = MultiCallStateMachine(3)
    with pytest.raises(InvalidTransitionError):
        machine.apply(machine.initial_state(), AgentEvent.tool_response())

    pending = machine.apply(
        machine.initial_state(), AgentEvent.valid_tool_call(_parsed_call("logo"))
    ).state
    with pytest.raises(InvalidTransitionError):
        machine.apply(pending, AgentEvent.final_answer())

    terminated = machine.apply(machine.initial_state(), AgentEvent.final_answer()).state
    with pytest.raises(InvalidTransitionError, match="terminated"):
        machine.apply(terminated, AgentEvent.final_answer())
