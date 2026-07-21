from __future__ import annotations

from dataclasses import replace

import pytest

from tgvf_rl.contracts.errors import (
    IdentityMismatchError,
    RecoverableToolExecutionError,
)
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import (
    FrameworkNeutralAgentLoop,
    RolloutRequest,
    SampledPolicyTurn,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.policy import PilotSamplingConfig
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import StandardToolError, TokenByteSpan
from tgvf_rl.trajectories.schema import TrajectoryIdentity, TrajectoryStop
from tgvf_rl.trajectories.schema import (
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolCallRecord,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tests.support import populated_observation_store, trajectory_source_visual


SHA = "0" * 64
_SOURCE_STORE, _SOURCE_HANDLE = populated_observation_store()
SOURCE_VISUAL = trajectory_source_visual(
    _SOURCE_STORE.resolve_record(_SOURCE_HANDLE)
)


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
    def __init__(self):
        self.contexts = []

    def execute(self, parsed_call, context):
        self.contexts.append(context)
        return ObservationHandle(
            f"observation-{context.call_index}", str(context.call_index) * 64
        )


class Appender:
    def append(
        self,
        prompt_token_ids,
        sampled_turn,
        observation,
        *,
        call_index,
        parsed_call,
    ):
        if isinstance(observation, StandardToolError):
            assert parsed_call is None
        else:
            assert parsed_call is not None
            assert parsed_call.sampled_text == sampled_turn.text
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
    runtime = Runtime()
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler(turns),
        tool_runtime=runtime,
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
            SOURCE_VISUAL,
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
    assert runtime.contexts[0].prompt_token_ids_before_turn == (1, 2, 3)
    assert runtime.contexts[0].conditioning_input_ids == (
        (1, 2, 3) + turns[0].token_ids
    )
    assert runtime.contexts[0].behavior_policy == version
    assert tuple(context.call_index for context in runtime.contexts) == (0, 1)


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
        '{"name":"image_zoom_in_tool","arguments":'
        '{"bbox_2d":[1,2,9,10],"label":"serial-number plate"}}'
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
            SOURCE_VISUAL,
            (1, 2, 3),
            {},
        )
    )
    assert isinstance(trajectory.tool_calls[0], CropToolCallRecord)
    assert isinstance(trajectory.tool_calls[1], ToolCallRecord)
    assert trajectory.tool_calls[0].bbox_2d == (1, 2, 9, 10)
    assert trajectory.tool_calls[0].label == "serial-number plate"
    assert trajectory.tool_calls[1].target == "serial number"
    assert tuple(item.call_index for item in trajectory.observations) == (0, 1)
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER


def test_framework_neutral_loop_records_atomic_crop_tgvf_call() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        10,
        SHA,
        1.0,
        1.0,
        -1,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )
    fused = _sample(
        "reason\n</think>\n<tool_call>"
        '{"name":"tgvf_crop_tool","arguments":'
        '{"bbox_2d":[3,4,19,23],"target":"small red digits"}}'
        "</tool_call>",
        sampling,
    )
    answer = _sample("reason\n</think>\n17", sampling)
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((fused, answer)),
        tool_runtime=Runtime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=4,
        enabled_tool_names=("tgvf_crop_tool",),
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "fused", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            SOURCE_VISUAL,
            (1, 2, 3),
            {},
        )
    )
    call = trajectory.tool_calls[0]
    assert isinstance(call, CropTGVFToolCallRecord)
    assert call.bbox_2d == (3, 4, 19, 23)
    assert call.target == "small red digits"
    assert call.target_token_span.end - call.target_token_span.start == len(
        "small red digits"
    )
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER


def test_tool_error_returns_standard_observation_and_allows_recovery() -> None:
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
        def execute(self, parsed_call, context):
            raise RecoverableToolExecutionError("fixture tool failure")

    behavior_store = BehaviorTraceStore()
    answer = _sample("recovered\n</think>\nblue", sampling)
    appender_values = []

    class RecordingAppender(Appender):
        def append(
            self,
            prompt_token_ids,
            sampled_turn,
            observation,
            *,
            call_index,
            parsed_call,
        ):
            appender_values.append((observation, parsed_call))
            return super().append(
                prompt_token_ids,
                sampled_turn,
                observation,
                call_index=call_index,
                parsed_call=parsed_call,
            )

    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((sampled, answer)),
        tool_runtime=FailingRuntime(),
        appender=RecordingAppender(),
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
            SOURCE_VISUAL,
            (1, 2, 3),
            {},
        )
    )
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert trajectory.tool_calls == ()
    assert trajectory.observations == ()
    assert len(trajectory.tool_errors) == 1
    assert trajectory.tool_errors[0].code == "tool_execution_failed"
    assert trajectory.tool_errors[0].attempt_index == 0
    assert isinstance(appender_values[0][0], StandardToolError)
    assert appender_values[0][1] is None
    assert trajectory.final_answer == "blue"
    assert behavior_store.resolve(trajectory.assistant_turns[0].behavior_trace)


def test_tool_contract_failure_propagates_instead_of_becoming_tool_error() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        7,
        SHA,
        1.0,
        1.0,
        -1,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )
    sampled = _sample(
        "reason</think><tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"red"}}'
        "</tool_call>",
        sampling,
    )

    class ContractFailingRuntime:
        def execute(self, parsed_call, context):
            raise IdentityMismatchError("fixture identity drift")

    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((sampled,)),
        tool_runtime=ContractFailingRuntime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=3,
    )
    with pytest.raises(IdentityMismatchError, match="identity drift"):
        loop.run(
            RolloutRequest(
                "trajectory-v1",
                TrajectoryIdentity("smoke", "fatal", 0, "group"),
                ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
                version,
                SOURCE_VISUAL,
                (1, 2, 3),
                {},
            )
        )


def test_fifth_attempt_is_not_executed_and_receives_cap_error() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        7,
        SHA,
        1.0,
        1.0,
        -1,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
    )

    def call(target: str) -> SampledPolicyTurn:
        return _sample(
            "reason\n</think>\n<tool_call>"
            f'{{"name":"tgvf_focus_tool","arguments":{{"target":"{target}"}}}}'
            "</tool_call>",
            sampling,
        )

    executed = []

    class CountingRuntime(Runtime):
        def execute(self, parsed_call, context):
            executed.append((parsed_call.target, context.call_index))
            return super().execute(parsed_call, context)

    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler(tuple(call(str(index)) for index in range(5))),
        tool_runtime=CountingRuntime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=4,
        enabled_tool_names=("tgvf_focus_tool",),
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "cap", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            SOURCE_VISUAL,
            (1, 2, 3),
            {},
        )
    )

    assert len(executed) == 4
    assert len(trajectory.tool_calls) == 4
    assert len(trajectory.observations) == 4
    assert trajectory.stop is TrajectoryStop.CALL_CAP
    assert len(trajectory.tool_errors) == 1
    assert trajectory.tool_errors[0].attempt_index == 4
    assert trajectory.tool_errors[0].code == "tool_call_limit_exceeded"


def test_disabled_crop_returns_error_without_executing_runtime() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling = SamplingIdentity(
        version,
        "vllm",
        "fixture",
        8,
        SHA,
        1.0,
        1.0,
        -1,
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
    answer = _sample("reason\n</think>\nanswer", sampling)

    class ForbiddenRuntime:
        def execute(self, parsed_call, context):
            raise AssertionError("disabled crop must not execute")

    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((crop, answer)),
        tool_runtime=ForbiddenRuntime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=4,
        enabled_tool_names=("tgvf_focus_tool",),
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "disabled", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            SOURCE_VISUAL,
            (1,),
            {},
        )
    )
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert trajectory.tool_calls == ()
    assert trajectory.tool_errors[0].code == "tool_not_enabled"


def test_typed_pilot_sampling_contract_controls_each_remaining_turn_budget() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling_contract = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(),
        stop_strings=(),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )
    sampling = SamplingIdentity(
        version,
        "vllm",
        "0.12.0",
        42,
        SHA,
        1.0,
        1.0,
        -1,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
        max_tokens=8192,
        do_sample=True,
        stop_token_ids=(),
        stop_strings=(),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )
    call = _sample(
        "reason\n</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"label"}}'
        "</tool_call>",
        sampling,
    )
    answer = _sample(
        "reason\n</think>\nanswer",
        replace(sampling, seed=43, max_tokens=8192 - len(call.token_ids)),
    )

    class RecordingSampler(Sampler):
        def __init__(self, turns):
            super().__init__(turns)
            self.parameters = []

        def sample(self, prompt_token_ids, sampling_parameters, *, turn_index):
            self.parameters.append(dict(sampling_parameters))
            return super().sample(
                prompt_token_ids, sampling_parameters, turn_index=turn_index
            )

    sampler = RecordingSampler((call, answer))
    loop = FrameworkNeutralAgentLoop(
        sampler=sampler,
        tool_runtime=Runtime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=4,
        enabled_tool_names=("tgvf_focus_tool",),
    )
    trajectory = loop.run(
        RolloutRequest(
            "trajectory-v1",
            TrajectoryIdentity("smoke", "sampling", 0, "group"),
            ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
            version,
            SOURCE_VISUAL,
            (1,),
            {},
            sampling_contract,
        )
    )

    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert sampler.parameters[0]["max_tokens"] == 8192
    assert sampler.parameters[1]["max_tokens"] == 8192 - len(call.token_ids)
    assert sampler.parameters[0]["top_k"] == -1
    assert sampler.parameters[0]["min_p"] == 0.0


def test_typed_pilot_sampling_rejects_backend_probability_mismatch() -> None:
    version = PolicyVersion("smoke", 0, SHA)
    sampling_contract = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(),
        stop_strings=(),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )
    bad_sampling = SamplingIdentity(
        version,
        "vllm",
        "0.12.0",
        42,
        SHA,
        0.9,
        1.0,
        -1,
        0.0,
        1.0,
        (),
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        0,
        max_tokens=8192,
        do_sample=True,
        stop_token_ids=(),
        stop_strings=(),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )
    answer = _sample("reason\n</think>\nanswer", bad_sampling)
    loop = FrameworkNeutralAgentLoop(
        sampler=Sampler((answer,)),
        tool_runtime=Runtime(),
        appender=Appender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(BehaviorTraceStore()),
        max_tool_calls=4,
    )
    with pytest.raises(IdentityMismatchError, match="SamplingIdentity differs"):
        loop.run(
            RolloutRequest(
                "trajectory-v1",
                TrajectoryIdentity("smoke", "bad-sampling", 0, "group"),
                ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA),
                version,
                SOURCE_VISUAL,
                (1,),
                {},
                sampling_contract,
            )
        )
