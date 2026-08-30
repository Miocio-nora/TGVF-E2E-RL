from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tgvf_rl.framework.verl.native_deepeyes_agent_loop import (
    NativeDeepEyesAgentLoop,
    StrictNativeDeepEyesAgentLoopV2,
    _current_policy_turn_token_ids,
    _decide_native_deepeyes_action,
)
from tgvf_rl.protocol.action_boundary import NativeActionBoundaryProtocolId


def test_native_deepeyes_loop_classes_bind_distinct_action_protocols() -> None:
    assert NativeDeepEyesAgentLoop.action_boundary_protocol_id is (
        NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
    )
    assert StrictNativeDeepEyesAgentLoopV2.action_boundary_protocol_id is (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    )
    assert NativeDeepEyesAgentLoop is not StrictNativeDeepEyesAgentLoopV2


def test_current_turn_slice_excludes_prior_action_and_observation_tokens() -> None:
    prior_action = tuple(map(ord, "<tool_call>old</tool_call>"))
    observation = tuple(map(ord, "<tool_response>crop</tool_response>"))
    current_final = tuple(map(ord, "<think>done</think>blue"))
    prompt_ids = (11, 12) + prior_action + observation + current_final
    prior_mask = (1,) * len(prior_action) + (0,) * len(observation)
    response_mask = prior_mask + (1,) * len(current_final)

    assert _current_policy_turn_token_ids(
        prompt_ids=prompt_ids,
        response_mask=response_mask,
        response_mask_length_before=len(prior_mask),
    ) == current_final


def test_current_turn_slice_accepts_legal_empty_generation() -> None:
    assert (
        _current_policy_turn_token_ids(
            prompt_ids=(11, 12, 13),
            response_mask=(1,),
            response_mask_length_before=1,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("text", "processing", "parsed_count", "expected_violation", "terminates"),
    (
        (
            '<think>x</think><tool_call>{"name":"image_zoom_in_tool"}',
            False,
            0,
            "malformed_tool_call_tags",
            True,
        ),
        (
            '<think>x</think>{"name":"image_zoom_in_tool"}</tool_call>',
            False,
            0,
            "malformed_tool_call_tags",
            True,
        ),
        (
            '<think>x</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":BROKEN}</tool_call>',
            False,
            0,
            "parsed_tool_call_count_mismatch",
            True,
        ),
        (
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>'
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>',
            True,
            2,
            "multiple_tool_calls",
            True,
        ),
        (
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>'
            "blue",
            True,
            1,
            "tool_call_terminal_suffix",
            True,
        ),
        ("<think>done</think>blue", False, 0, None, True),
        (
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>',
            True,
            1,
            None,
            False,
        ),
    ),
)
def test_dependency_free_strict_action_decision_covers_parser_early_exit(
    text: str,
    processing: bool,
    parsed_count: int,
    expected_violation: str | None,
    terminates: bool,
) -> None:
    decision = _decide_native_deepeyes_action(
        text,
        protocol_id=(
            NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ),
        upstream_processing_tools=processing,
        parsed_tool_call_count=parsed_count,
        upstream_limit=False,
    )

    assert decision.violation_code == expected_violation
    assert decision.terminate is terminates


class _CharacterTokenizer:
    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(map(ord, text))

    def decode(self, token_ids: object, **_kwargs: object) -> str:
        return "".join(map(chr, token_ids))


class _QueuedServer:
    def __init__(self, tokenizer: _CharacterTokenizer, *texts: str) -> None:
        self.tokenizer = tokenizer
        self.texts = list(texts)

    async def generate(self, **_kwargs: object) -> object:
        text = self.texts.pop(0)
        return SimpleNamespace(
            token_ids=self.tokenizer.encode(text),
            num_preempted=0,
            extra_fields={},
            log_probs=None,
            routed_experts=None,
        )


def _upstream_harness(*texts: str) -> tuple[object, object, object]:
    pytest.importorskip(
        "verl",
        reason="state-machine fixture requires the pinned optional veRL",
    )
    from verl.experimental.agent_loop.tool_agent_loop import AgentData
    from verl.experimental.agent_loop.tool_parser import HermesToolParser

    tokenizer = _CharacterTokenizer()
    loop = object.__new__(StrictNativeDeepEyesAgentLoopV2)
    loop.tokenizer = tokenizer
    loop.tool_parser = HermesToolParser(tokenizer)
    loop.server_manager = _QueuedServer(tokenizer, *texts)
    loop.tools = {}
    loop.enable_continuous_token = False
    loop.response_length = 8192
    loop.max_assistant_turns = 7
    loop.max_user_turns = 6
    data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="boundary-fixture",
        tools_kwargs={},
    )
    data.prompt_ids = [17, 18]
    return loop, data, tokenizer


@pytest.mark.parametrize(
    ("text", "expected_code"),
    (
        (
            '<think>x</think><tool_call>{"name":"image_zoom_in_tool"}',
            "malformed_tool_call_tags",
        ),
        (
            '<think>x</think>{"name":"image_zoom_in_tool"}</tool_call>',
            "malformed_tool_call_tags",
        ),
        (
            '<think>x</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":BROKEN}</tool_call>',
            "parsed_tool_call_count_mismatch",
        ),
        (
            '<think>x</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":BROKEN}</tool_call>blue',
            "tool_call_terminal_suffix",
        ),
        (
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>'
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>',
            "multiple_tool_calls",
        ),
        (
            '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>'
            "blue",
            "tool_call_terminal_suffix",
        ),
    ),
)
def test_pinned_upstream_state_machine_records_strict_boundary_violations(
    text: str,
    expected_code: str,
) -> None:
    pytest.importorskip("verl")
    from verl.experimental.agent_loop.tool_agent_loop import AgentState

    loop, data, _tokenizer = _upstream_harness(text)

    state = asyncio.run(loop._handle_generating_state(data, {}))

    assert state is AgentState.TERMINATED
    assert data.tool_calls == []
    assert data.extra_fields["action_boundary_violation_count"] == 1
    assert data.extra_fields["action_boundary_violation_codes"] == [expected_code]


def test_pinned_upstream_state_machine_accepts_valid_tool_then_current_turn_final() -> (
    None
):
    pytest.importorskip("verl")
    from verl.experimental.agent_loop.tool_agent_loop import AgentState

    tool_text = (
        '<think>inspect</think><tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[10,20,100,200]}}</tool_call>'
    )
    final_text = "<think>done</think>blue"
    loop, data, tokenizer = _upstream_harness(tool_text, final_text)

    first = asyncio.run(loop._handle_generating_state(data, {}))
    assert first is AgentState.PROCESSING_TOOLS
    assert len(data.tool_calls) == 1

    # Model one environment-owned observation between assistant turns.  The
    # second decision must inspect only final_text, never the prior call.
    observation_ids = tokenizer.encode("<tool_response>crop</tool_response>")
    data.prompt_ids.extend(observation_ids)
    data.response_mask.extend([0] * len(observation_ids))
    data.user_turns += 1

    second = asyncio.run(loop._handle_generating_state(data, {}))
    assert second is AgentState.TERMINATED
    assert data.tool_calls == []
    assert data.extra_fields.get("action_boundary_violation_count", 0) == 0
    assert data.extra_fields.get("action_boundary_violation_codes", []) == []
