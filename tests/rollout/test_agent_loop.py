from __future__ import annotations

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import (
    FrameworkNeutralAgentLoop,
    RolloutRequest,
    SampledPolicyTurn,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import TokenByteSpan
from tgvf_rl.trajectories.schema import TrajectoryIdentity, TrajectoryStop
from tgvf_rl.trajectories.schema import CropToolCallRecord, ToolCallRecord
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder


SHA = "0" * 64


def _sample(text: str, sampling: SamplingIdentity) -> SampledPolicyTurn:
    ids = tuple(ord(char) for char in text)
    spans = tuple(
        TokenByteSpan(index, token, index, index + 1) for index, token in enumerate(ids)
    )
    close_end = text.index("</think>") + len("</think>")
    return SampledPolicyTurn(
        text=text,
        token_ids=ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.1 for _ in ids),
        sampling=sampling,
        think_token_span=TokenSpan(0, close_end),
        stop_reason="stop",
        backend_request_sha256=SHA,
        backend_response_sha256=SHA,
    )


class Sampler:
    def __init__(self, turns):
        self.turns = list(turns)

    def sample(self, prompt_token_ids, sampling_parameters, *, turn_index):
        return self.turns[turn_index]


class Runtime:
    def execute(self, parsed_call, call_index):
        return ObservationHandle(f"observation-{call_index}", str(call_index) * 64)


class Appender:
    def append(self, prompt_token_ids, sampled_turn, observation, *, call_index):
        environment = (151665, 151655, 151666, 151644, 151667)
        return prompt_token_ids + sampled_turn.token_ids + environment, environment


def test_framework_neutral_loop_preserves_two_calls_and_actual_logprobs() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        7,
        SHA,
        0.7,
        0.9,
        20,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )

    def call(target: str) -> str:
        return (
            "reason\n</think>\n\n<tool_call>"
            f'{{"name":"tgvf_focus_tool","arguments":{{"target":"{target}"}}}}'
            "</tool_call>"
        )

    turns = (
        _sample(call("red label"), sampling),
        _sample(call("lower label"), sampling),
        _sample("answer reasoning\n</think>\n\nblue", sampling),
    )
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler(turns),
        tool_runtime=Runtime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=3,
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "sample", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            (1, 2, 3),
            {"temperature": 0.7},
        )
    )
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert tuple(call.target for call in trajectory.tool_calls) == (
        "red label",
        "lower label",
    )
    assert len(trajectory.observations) == 2
    assert all(
        len(loop.behavior_recorder.store.resolve(turn.behavior_trace).behavior.logprobs)
        == len(turn.tokens.token_ids)
        for turn in trajectory.assistant_turns
    )
    assert trajectory.final_answer == "blue"


def test_framework_neutral_loop_preserves_mixed_crop_then_tgvf_order() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        9,
        SHA,
        0.7,
        0.9,
        20,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )
    crop = _sample(
        "reason\n</think>\n<tool_call>"
        '{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[1,2,9,10]}}'
        "</tool_call>",
        sampling,
    )
    focus = _sample(
        "reason\n</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"serial number"}}'
        "</tool_call>",
        sampling,
    )
    answer = _sample("reason\n</think>\n42", sampling)
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((crop, focus, answer)),
        tool_runtime=Runtime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=3,
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "mixed", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            (1, 2, 3),
            {},
        )
    )
    assert isinstance(trajectory.tool_calls[0], CropToolCallRecord)
    assert isinstance(trajectory.tool_calls[1], ToolCallRecord)
    assert trajectory.tool_calls[0].bbox_2d == (1, 2, 9, 10)
    assert trajectory.tool_calls[1].target == "serial number"
    assert tuple(item.call_index for item in trajectory.observations) == (0, 1)
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER


def test_tool_error_does_not_fabricate_a_call_or_observation() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        7,
        SHA,
        0.7,
        0.9,
        20,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )
    sampled = _sample(
        "reason\n</think>\n\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"red"}}'
        "</tool_call>",
        sampling,
    )

    class FailingRuntime:
        def execute(self, parsed_call, call_index):
            raise RuntimeError("fixture tool failure")

    behavior_store = BehaviorTraceStore()
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((sampled,)),
        tool_runtime=FailingRuntime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(behavior_store),
        max_tool_calls=3,
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "sample", 1, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            (1, 2, 3),
            {},
        )
    )
    assert trajectory.stop is TrajectoryStop.TOOL_ERROR
    assert trajectory.tool_calls == ()
    assert trajectory.observations == ()
    assert behavior_store.resolve(trajectory.assistant_turns[0].behavior_trace)
