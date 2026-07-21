"""Framework-neutral native multi-call rollout loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from tgvf_rl.contracts.errors import RecoverableToolExecutionError
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.observations.schema import TrajectorySourceVisual
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.policy.config import PilotSamplingConfig
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import (
    POLICY_RL_TOOL_NAMES,
    SampledAssistantTurn,
    StandardToolError,
    ToolCallParseError,
    ToolErrorCode,
)
from tgvf_rl.protocol.schema import ParsedCropTGVFCall, ParsedImageZoomInCall
from tgvf_rl.protocol.state_machine import (
    AgentEvent,
    AgentPhase,
    CapErrorBehavior,
    MultiCallStateMachine,
)
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    NativeToolCallRecord,
    ToolCallRecord,
    ToolErrorRecord,
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
        if not self.stop_reason:
            raise ValueError("sampler stop reason must be recorded")

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


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Exact behavior-forward context required to materialize one observation."""

    trajectory_identity: TrajectoryIdentity
    model: ModelIdentity
    behavior_policy: PolicyVersion
    trajectory_source_visual: TrajectorySourceVisual
    prior_observation_handles: tuple[ObservationHandle, ...]
    prompt_token_ids_before_turn: tuple[int, ...]
    sampled_turn: SampledPolicyTurn
    assistant_turn_index: int
    attempt_index: int
    call_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_token_ids_before_turn",
            tuple(self.prompt_token_ids_before_turn),
        )
        object.__setattr__(
            self, "prior_observation_handles", tuple(self.prior_observation_handles)
        )
        if not self.prompt_token_ids_before_turn:
            raise ValueError("tool execution requires the exact non-empty prompt prefix")
        if self.sampled_turn.sampling.policy_version != self.behavior_policy:
            raise ValueError("tool execution context changed behavior policy")
        if self.assistant_turn_index < 0 or self.attempt_index < 0:
            raise ValueError("tool execution turn/attempt indices must be non-negative")
        if self.call_index < 0:
            raise ValueError("tool execution call index must be non-negative")
        if not isinstance(self.trajectory_source_visual, TrajectorySourceVisual):
            raise TypeError("tool execution requires an immutable source visual")
        if any(
            not isinstance(handle, ObservationHandle)
            for handle in self.prior_observation_handles
        ):
            raise TypeError("prior observations must be ObservationHandle values")
        if len(self.prior_observation_handles) != self.call_index:
            raise ValueError("prior observations must exactly precede the call index")

    @property
    def conditioning_input_ids(self) -> tuple[int, ...]:
        """The exact policy prefix through the sampled native tool call."""

        return self.prompt_token_ids_before_turn + self.sampled_turn.token_ids


class ToolRuntimePort(Protocol):
    def execute(
        self, parsed_call: object, context: ToolExecutionContext
    ) -> ObservationHandle: ...


class ToolObservationAppender(Protocol):
    def append(
        self,
        prompt_token_ids: tuple[int, ...],
        sampled_turn: SampledPolicyTurn,
        observation: ObservationHandle | StandardToolError,
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
    trajectory_source_visual: TrajectorySourceVisual
    initial_prompt_token_ids: tuple[int, ...]
    sampling_parameters: Mapping[str, object]
    sampling_contract: PilotSamplingConfig | None = None

    def __post_init__(self) -> None:
        if not self.initial_prompt_token_ids:
            raise ValueError("rollout initial prompt tokens must be non-empty")
        if not isinstance(self.trajectory_source_visual, TrajectorySourceVisual):
            raise TypeError("rollout requires an immutable source visual")
        if self.sampling_contract is not None:
            if not isinstance(self.sampling_contract, PilotSamplingConfig):
                raise TypeError("sampling_contract must be PilotSamplingConfig")
            if self.sampling_parameters:
                raise ValueError(
                    "typed Pilot sampling and untyped sampling parameters cannot coexist"
                )


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
        enabled_tool_names: tuple[str, ...] = POLICY_RL_TOOL_NAMES,
        cap_error_behavior: CapErrorBehavior = CapErrorBehavior.TERMINATE_AFTER_ERROR,
    ) -> None:
        self.sampler = sampler
        self.tool_runtime = tool_runtime
        self.appender = appender
        self.parser = parser
        self.behavior_recorder = behavior_recorder
        names = tuple(enabled_tool_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("enabled tool names must be non-empty and unique")
        unknown = set(names) - set(POLICY_RL_TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown enabled tools: {sorted(unknown)!r}")
        self.enabled_tool_names = names
        self.machine = MultiCallStateMachine(max_tool_calls, cap_error_behavior)

    def run(self, request: RolloutRequest) -> TrajectoryRecord:
        prompt = tuple(request.initial_prompt_token_ids)
        state = self.machine.initial_state()
        turns: list[AssistantTurnRecord] = []
        calls: list[NativeToolCallRecord] = []
        observations: list[ToolObservationRecord] = []
        errors: list[ToolErrorRecord] = []
        final_answer: str | None = None
        stop: TrajectoryStop | None = None
        generated_policy_tokens = 0

        while state.phase is not AgentPhase.TERMINATED:
            if (
                request.sampling_contract is not None
                and generated_policy_tokens
                == request.sampling_contract.max_response_length
            ):
                stop = TrajectoryStop.MAX_TOKENS
                break
            sampling_parameters: Mapping[str, object]
            expected_max_tokens: int | None = None
            if request.sampling_contract is None:
                sampling_parameters = request.sampling_parameters
            else:
                remaining = request.sampling_contract.remaining_response_tokens(
                    generated_policy_tokens
                )
                expected_max_tokens = remaining
                sampling_parameters = request.sampling_contract.as_vllm_parameters(
                    max_tokens=remaining
                )
            sampled = self.sampler.sample(
                prompt, sampling_parameters, turn_index=len(turns)
            )
            if sampled.sampling.policy_version != request.behavior_policy:
                raise ValueError("sampler policy version differs from rollout request")
            if request.sampling_contract is not None:
                assert expected_max_tokens is not None
                request.sampling_contract.validate_sampling_identity(
                    sampled.sampling,
                    expected_max_tokens=expected_max_tokens,
                )
                generated_policy_tokens += len(sampled.token_ids)
                if (
                    generated_policy_tokens
                    > request.sampling_contract.max_response_length
                ):
                    raise ValueError(
                        "sampler exceeded the cumulative trajectory response budget"
                    )
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
                    stop_reason=sampled.stop_reason,
                )
            )
            if not has_tool_marker:
                transition = self.machine.apply(state, AgentEvent.final_answer())
                state = transition.state
                final_answer = sampled.text.split("</think>", 1)[1].strip()
                stop = (
                    TrajectoryStop.DIRECT_ANSWER
                    if state.tool_attempt_count == 0
                    else TrajectoryStop.FINAL_ANSWER
                )
                break

            try:
                parsed = self.parser.parse(sampled.parser_turn())
            except ToolCallParseError as parse_error:
                transition = self.machine.apply(state, AgentEvent.malformed_action())
                state = transition.state
                if not transition.emit_error:
                    stop = TrajectoryStop.CALL_CAP
                    break
                assert transition.attempt_index is not None
                error = self._standard_error(
                    code=(
                        ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value
                        if state.pending_cap_error
                        else f"tool_parse.{parse_error.code.value}"
                    ),
                    message=(
                        self._cap_error_message()
                        if state.pending_cap_error
                        else "The tool call is invalid and was not executed."
                    ),
                    attempt_index=transition.attempt_index,
                    recoverable=not state.pending_cap_error
                    or self.machine.cap_error_behavior
                    is CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
                )
                prompt, environment_tokens = self._append_environment(
                    prompt,
                    sampled,
                    error,
                    call_index=transition.attempt_index,
                )
                errors.append(
                    self._error_record(
                        error,
                        assistant_turn_index=len(turns) - 1,
                        template_token_ids=environment_tokens,
                    )
                )
                state = self.machine.apply(state, AgentEvent.tool_error()).state
                if state.phase is AgentPhase.TERMINATED:
                    stop = TrajectoryStop.CALL_CAP
                    break
                continue

            transition = self.machine.apply(state, AgentEvent.valid_tool_call(parsed))
            state = transition.state
            if transition.emit_error:
                assert transition.attempt_index is not None
                error = self._standard_error(
                    code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
                    message=self._cap_error_message(),
                    attempt_index=transition.attempt_index,
                    recoverable=self.machine.cap_error_behavior
                    is CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
                )
                prompt, environment_tokens = self._append_environment(
                    prompt,
                    sampled,
                    error,
                    call_index=transition.attempt_index,
                )
                errors.append(
                    self._error_record(
                        error,
                        assistant_turn_index=len(turns) - 1,
                        template_token_ids=environment_tokens,
                        function_name=parsed.name,
                    )
                )
                state = self.machine.apply(state, AgentEvent.tool_error()).state
                if state.phase is AgentPhase.TERMINATED:
                    stop = TrajectoryStop.CALL_CAP
                    break
                continue
            if not transition.execute_tool:
                stop = TrajectoryStop.CALL_CAP
                break
            attempt_index = transition.attempt_index
            call_index = transition.call_index
            assert attempt_index is not None and call_index is not None

            if parsed.name not in self.enabled_tool_names:
                error = self._standard_error(
                    code=ToolErrorCode.TOOL_NOT_ENABLED.value,
                    message="This tool is not enabled for the current experiment.",
                    attempt_index=attempt_index,
                    recoverable=True,
                )
                prompt, environment_tokens = self._append_environment(
                    prompt, sampled, error, call_index=attempt_index
                )
                errors.append(
                    self._error_record(
                        error,
                        assistant_turn_index=len(turns) - 1,
                        template_token_ids=environment_tokens,
                        function_name=parsed.name,
                    )
                )
                state = self.machine.apply(state, AgentEvent.tool_error()).state
                continue

            try:
                handle = self.tool_runtime.execute(
                    parsed,
                    ToolExecutionContext(
                        trajectory_identity=request.identity,
                        model=request.model,
                        behavior_policy=request.behavior_policy,
                        trajectory_source_visual=request.trajectory_source_visual,
                        prior_observation_handles=tuple(
                            observation.handle for observation in observations
                        ),
                        prompt_token_ids_before_turn=prompt,
                        sampled_turn=sampled,
                        assistant_turn_index=len(turns) - 1,
                        attempt_index=attempt_index,
                        call_index=call_index,
                    ),
                )
            except RecoverableToolExecutionError:
                error = self._standard_error(
                    code=ToolErrorCode.TOOL_EXECUTION_FAILED.value,
                    message="The tool failed to execute and produced no observation.",
                    attempt_index=attempt_index,
                    recoverable=True,
                )
                prompt, environment_tokens = self._append_environment(
                    prompt, sampled, error, call_index=attempt_index
                )
                errors.append(
                    self._error_record(
                        error,
                        assistant_turn_index=len(turns) - 1,
                        template_token_ids=environment_tokens,
                        function_name=parsed.name,
                    )
                )
                state = self.machine.apply(state, AgentEvent.tool_error()).state
                continue
            prompt, environment_tokens = self._append_environment(
                prompt, sampled, handle, call_index=call_index
            )
            if isinstance(parsed, ParsedImageZoomInCall):
                calls.append(
                    CropToolCallRecord(
                        call_index=call_index,
                        assistant_turn_index=len(turns) - 1,
                        function_name=parsed.name,
                        bbox_2d=parsed.bbox_2d,
                        raw_call_text=parsed.raw_tool_call,
                        attempt_index=attempt_index,
                    )
                )
            elif isinstance(parsed, ParsedCropTGVFCall):
                calls.append(
                    CropTGVFToolCallRecord(
                        call_index=call_index,
                        assistant_turn_index=len(turns) - 1,
                        function_name=parsed.name,
                        bbox_2d=parsed.bbox_2d,
                        target=parsed.target,
                        target_token_span=TokenSpan(
                            parsed.target_span.token_start,
                            parsed.target_span.token_end,
                        ),
                        target_char_span=(
                            parsed.target_span.offsets.char_start,
                            parsed.target_span.offsets.char_end,
                        ),
                        raw_call_text=parsed.raw_tool_call,
                        attempt_index=attempt_index,
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
                        attempt_index=attempt_index,
                    )
                )
            observations.append(
                ToolObservationRecord(call_index, handle, environment_tokens)
            )
            state = self.machine.apply(state, AgentEvent.tool_response()).state
            if (
                request.sampling_contract is not None
                and generated_policy_tokens
                == request.sampling_contract.max_response_length
            ):
                stop = TrajectoryStop.MAX_TOKENS
                break

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
            tool_errors=tuple(errors),
        )

    def _append_environment(
        self,
        prompt: tuple[int, ...],
        sampled: SampledPolicyTurn,
        value: ObservationHandle | StandardToolError,
        *,
        call_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        previous_prompt = prompt
        updated_prompt, environment_tokens = self.appender.append(
            prompt, sampled, value, call_index=call_index
        )
        environment_tokens = tuple(environment_tokens)
        expected_prompt = previous_prompt + sampled.token_ids + environment_tokens
        if tuple(updated_prompt) != expected_prompt:
            raise ValueError(
                "tool appender must preserve the exact old prompt and sampled turn"
            )
        if not environment_tokens:
            raise ValueError("tool response must append native environment tokens")
        return tuple(updated_prompt), environment_tokens

    def _standard_error(
        self,
        *,
        code: str,
        message: str,
        attempt_index: int,
        recoverable: bool,
    ) -> StandardToolError:
        return StandardToolError(
            code=code,
            message=message,
            attempt_index=attempt_index,
            recoverable=recoverable,
            maximum_tool_calls=self.machine.max_tool_calls,
        )

    def _cap_error_message(self) -> str:
        return (
            f"The maximum of {self.machine.max_tool_calls} tool-call attempts "
            "has been reached; this call was not executed."
        )

    @staticmethod
    def _error_record(
        error: StandardToolError,
        *,
        assistant_turn_index: int,
        template_token_ids: tuple[int, ...],
        function_name: str | None = None,
    ) -> ToolErrorRecord:
        return ToolErrorRecord(
            attempt_index=error.attempt_index,
            assistant_turn_index=assistant_turn_index,
            code=error.code,
            payload_json=error.canonical_json,
            payload_sha256=error.payload_sha256,
            template_token_ids=template_token_ids,
            recoverable=error.recoverable,
            function_name=function_name,
        )
