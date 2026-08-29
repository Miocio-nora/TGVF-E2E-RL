from __future__ import annotations

import pytest

from tgvf_rl.protocol.action_boundary import (
    AssistantTurnDisposition,
    NativeActionBoundaryProtocolId,
    classify_assistant_action_boundary,
)


LEGACY = NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
STRICT = NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
CALL_A = '<tool_call>{"name":"crop","arguments":{"x":1}}</tool_call>'
CALL_B = '<tool_call>{"name":"crop","arguments":{"x":2}}</tool_call>'


def test_no_tool_marker_is_a_direct_final_under_both_protocols() -> None:
    for protocol in (LEGACY, STRICT):
        boundary = classify_assistant_action_boundary(
            "<think>reason</think>answer",
            protocol_id=protocol,
        )
        assert boundary.disposition is AssistantTurnDisposition.DIRECT_FINAL
        assert boundary.final_text == "<think>reason</think>answer"
        assert boundary.tool_call_blocks == ()


def test_strict_protocol_accepts_one_terminal_call_modulo_whitespace() -> None:
    boundary = classify_assistant_action_boundary(
        f"<think>inspect</think>{CALL_A}\n",
        protocol_id=STRICT,
    )
    assert boundary.disposition is AssistantTurnDisposition.TOOL_ACTION
    assert boundary.selected_tool_call == CALL_A
    assert boundary.trailing_text == "\n"
    assert boundary.violation_code is None


def test_strict_protocol_rejects_answer_after_action() -> None:
    boundary = classify_assistant_action_boundary(
        f"<think>inspect</think>{CALL_A} blue",
        protocol_id=STRICT,
    )
    assert boundary.disposition is AssistantTurnDisposition.INVALID_ACTION
    assert boundary.selected_tool_call is None
    assert boundary.final_text is None
    assert boundary.violation_code == "tool_call_terminal_suffix"


def test_strict_protocol_rejects_multiple_calls_in_one_turn() -> None:
    boundary = classify_assistant_action_boundary(
        CALL_A + CALL_B,
        protocol_id=STRICT,
    )
    assert boundary.disposition is AssistantTurnDisposition.INVALID_ACTION
    assert boundary.tool_call_blocks == (CALL_A, CALL_B)
    assert boundary.violation_code == "multiple_tool_calls"


def test_legacy_protocol_preserves_answer_over_action_and_last_call() -> None:
    answer = classify_assistant_action_boundary(
        CALL_A + " blue",
        protocol_id=LEGACY,
    )
    assert answer.disposition is AssistantTurnDisposition.DIRECT_FINAL
    assert answer.final_text == "blue"

    action = classify_assistant_action_boundary(
        CALL_A + CALL_B,
        protocol_id=LEGACY,
    )
    assert action.disposition is AssistantTurnDisposition.TOOL_ACTION
    assert action.selected_tool_call == CALL_B


@pytest.mark.parametrize(
    "text",
    (
        "<tool_call>{}",
        "{}</tool_call>",
        "<tool_call><tool_call>{}</tool_call>",
    ),
)
def test_malformed_tags_are_invalid_under_both_protocols(text: str) -> None:
    for protocol in (LEGACY, STRICT):
        boundary = classify_assistant_action_boundary(text, protocol_id=protocol)
        assert boundary.disposition is AssistantTurnDisposition.INVALID_ACTION
        assert boundary.violation_code == "malformed_tool_call_tags"


def test_unknown_protocol_fails_closed() -> None:
    with pytest.raises(ValueError, match="protocol ID is invalid"):
        classify_assistant_action_boundary(CALL_A, protocol_id="implicit")
