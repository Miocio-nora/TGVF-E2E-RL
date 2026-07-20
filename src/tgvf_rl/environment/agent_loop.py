"""Framework-neutral native multi-call rollout loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, ToolCallParseError
from tgvf_rl.protocol.schema import ParsedImageZoomInCall
from tgvf_rl.protocol.state_machine import AgentEvent, AgentPhase, MultiCallStateMachine
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    CropToolCallRecord,
    NativeToolCallRecord,
    ToolCallRecord,
    ToolObservationRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tgvf_rl.trajectories.behavior import VLLMBehaviorRecorder


@dataclass(frozen=True, slots=True)
class SampledPolicyTurn:
    text: str
    token_ids: tuple[int, ...]
    token_byte_spans: tuple[object, ...]
    behavior_logprobs: tuple[float, ...]
    sampling: SamplingIdentity
    think_token_span: TokenSpan
    stop_reason: str
    backend_request_sha256: str
    backend_response_sha256: str

    def __post_init__(self) -> None:
        if not self.token_ids or len(self.token_ids) != len(self.behavior_logprobs):
            raise ValueError("sampled tokens and actual behavior logprobs must align")
        if len(self.token_byte_spans) != len(self.token_ids):
            raise ValueError("sampled tokens and byte spans must align")
        if "<think>" in self.text:
            raise ValueError("policy sampled a duplicate <think> opener")
        if self.text.count("</think>") != 1:
            raise ValueError("sampled assistant turn must close exactly one think span")
        if self.think_token_span.end > len(self.token_ids):
            raise ValueError("think token span lies outside sampled tokens")

    def parser_turn(self) -> SampledAssistantTurn:
        return SampledAssistantTurn(
            self.text, self.token_ids, tuple(self.token_byte_spans)
        )


class PolicySamplerPort(Protocol):
    def sample(
        self,
        prompt_token_ids: tuple[int, ...],
        sampling_parameters: Mapping[str, object],
        *,
        turn_index: int,
    ) -> SampledPolicyTurn: ...


class ToolRuntimePort(Protocol):
    def execute(self, parsed_call: object, call_index: int) -> ObservationHandle: ...


class ToolObservationAppender(Protocol):
    def append(
        self,
        prompt_token_ids: tuple[int, ...],
        sampled_turn: SampledPolicyTurn,
        observation: ObservationHandle,
        *,
        call_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return updated prompt and exact environment-owned appended token IDs."""


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    schema_version: str
    identity: TrajectoryIdentity
    model: ModelIdentity
    behavior_policy: PolicyVersion
    initial_prompt_token_ids: tuple[int, ...]
    sampling_parameters: Mapping[str, object]


class FrameworkNeutralAgentLoop:
    def __init__(
        self,
        *,
        sampler: PolicySamplerPort,
        tool_runtime: ToolRuntimePort,
        appender: ToolObservationAppender,
        parser: StrictToolCallParser,
        behavior_recorder: VLLMBehaviorRecorder,
        max_tool_calls: int,
    ) -> None:
        self.sampler = sampler
        self.tool_runtime = tool_runtime
        self.appender = appender
        self.parser = parser
        self.behavior_recorder = behavior_recorder
        self.machine = MultiCallStateMachine(max_tool_calls)

    def run(self, request: RolloutRequest) -> TrajectoryRecord:
        prompt = tuple(request.initial_prompt_token_ids)
        state = self.machine.initial_state()
        turns: list[AssistantTurnRecord] = []
        calls: list[NativeToolCallRecord] = []
        observations: list[ToolObservationRecord] = []
        final_answer: str | None = None
        stop: TrajectoryStop | None = None

        while state.phase is not AgentPhase.TERMINATED:
            sampled = self.sampler.sample(
                prompt, request.sampling_parameters, turn_index=len(turns)
            )
            if sampled.sampling.policy_version != request.behavior_policy:
                raise ValueError("sampler policy version differs from rollout request")
            owned_tokens = OwnedTokenSequence(
                sampled.token_ids,
                tuple(TokenOwnership.POLICY_SAMPLED for _ in sampled.token_ids),
            )
            behavior_trace = self.behavior_recorder.record(
                trajectory_id=request.identity.canonical_id,
                assistant_turn_index=len(turns),
                tokens=owned_tokens,
                actual_sampled_logprobs=sampled.behavior_logprobs,
                sampling=sampled.sampling,
                behavior_policy=request.behavior_policy,
                backend_request_sha256=sampled.backend_request_sha256,
                backend_response_sha256=sampled.backend_response_sha256,
            )
            has_tool_marker = (
                "<tool_call>" in sampled.text or "</tool_call>" in sampled.text
            )
            turns.append(
                AssistantTurnRecord(
                    turn_index=len(turns),
                    raw_text=sampled.text,
                    tokens=owned_tokens,
                    behavior_trace=behavior_trace,
                    think_span=sampled.think_token_span,
                    is_tool_call=has_tool_marker,
                )
            )
            if not has_tool_marker:
                transition = self.machine.apply(state, AgentEvent.final_answer())
                state = transition.state
                final_answer = sampled.text.split("</think>", 1)[1].strip()
                stop = (
                    TrajectoryStop.DIRECT_ANSWER
                    if not calls
                    else TrajectoryStop.FINAL_ANSWER
                )
                break

            try:
                parsed = self.parser.parse(sampled.parser_turn())
            except ToolCallParseError:
                state = self.machine.apply(state, AgentEvent.malformed_action()).state
                stop = TrajectoryStop.MALFORMED_CALL
                break

            transition = self.machine.apply(state, AgentEvent.valid_tool_call(parsed))
            state = transition.state
            if not transition.execute_tool:
                stop = TrajectoryStop.CALL_CAP
                break
            call_index = transition.call_index
            assert call_index is not None
            try:
                handle = self.tool_runtime.execute(parsed, call_index)
                previous_prompt = prompt
                prompt, environment_tokens = self.appender.append(
                    prompt, sampled, handle, call_index=call_index
                )
                expected_prompt = (
                    previous_prompt + sampled.token_ids + tuple(environment_tokens)
                )
                if tuple(prompt) != expected_prompt:
                    raise ValueError(
                        "tool appender must preserve the exact old prompt and sampled turn"
                    )
                if not environment_tokens:
                    raise ValueError(
                        "tool response must append native environment tokens"
                    )
            except Exception:
                state = self.machine.apply(state, AgentEvent.tool_error()).state
                stop = TrajectoryStop.TOOL_ERROR
                break
            if isinstance(parsed, ParsedImageZoomInCall):
                calls.append(
                    CropToolCallRecord(
                        call_index=call_index,
                        assistant_turn_index=len(turns) - 1,
                        function_name=parsed.name,
                        bbox_2d=parsed.bbox_2d,
                        raw_call_text=parsed.raw_tool_call,
                    )
                )
            else:
                calls.append(
                    ToolCallRecord(
                        call_index=call_index,
                        assistant_turn_index=len(turns) - 1,
                        function_name=parsed.name,
                        target=parsed.target,
                        target_token_span=TokenSpan(
                            parsed.target_span.token_start, parsed.target_span.token_end
                        ),
                        target_char_span=(
                            parsed.target_span.offsets.char_start,
                            parsed.target_span.offsets.char_end,
                        ),
                        raw_call_text=parsed.raw_tool_call,
                    )
                )
            observations.append(
                ToolObservationRecord(call_index, handle, environment_tokens)
            )
            state = self.machine.apply(state, AgentEvent.tool_response()).state

        if stop is None:
            raise RuntimeError("agent loop terminated without a recorded stop reason")
        return TrajectoryRecord(
            schema_version=request.schema_version,
            identity=request.identity,
            model=request.model,
            behavior_policy=request.behavior_policy,
            assistant_turns=tuple(turns),
            tool_calls=tuple(calls),
            observations=tuple(observations),
            final_answer=final_answer,
            stop=stop,
        )
