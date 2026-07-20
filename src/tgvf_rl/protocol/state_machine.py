"""Fail-closed repeated-call state machine for the native tool environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schema import NativeToolCall, TerminationReason


class AgentPhase(str, Enum):
    AWAITING_ASSISTANT = "awaiting_assistant"
    AWAITING_TOOL_RESPONSE = "awaiting_tool_response"
    AWAITING_FINAL_ANSWER = "awaiting_final_answer"
    TERMINATED = "terminated"


class CapErrorBehavior(str, Enum):
    TERMINATE_AFTER_ERROR = "terminate_after_error"
    ONE_FINAL_ANSWER_TURN = "one_final_answer_turn"


class AgentEventType(str, Enum):
    VALID_TOOL_CALL = "valid_tool_call"
    TOOL_RESPONSE = "tool_response"
    FINAL_ANSWER = "final_answer"
    MALFORMED_ACTION = "malformed_action"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"


class InvalidTransitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventType
    tool_call: NativeToolCall | None = None

    def __post_init__(self) -> None:
        if self.kind is AgentEventType.VALID_TOOL_CALL and self.tool_call is None:
            raise ValueError("a valid-tool-call event requires the parsed immutable call")
        if self.kind is not AgentEventType.VALID_TOOL_CALL and self.tool_call is not None:
            raise ValueError("only a valid-tool-call event may carry a parsed call")

    @classmethod
    def valid_tool_call(cls, call: NativeToolCall) -> "AgentEvent":
        return cls(AgentEventType.VALID_TOOL_CALL, call)

    @classmethod
    def tool_response(cls) -> "AgentEvent":
        return cls(AgentEventType.TOOL_RESPONSE)

    @classmethod
    def final_answer(cls) -> "AgentEvent":
        return cls(AgentEventType.FINAL_ANSWER)

    @classmethod
    def malformed_action(cls) -> "AgentEvent":
        return cls(AgentEventType.MALFORMED_ACTION)

    @classmethod
    def tool_error(cls) -> "AgentEvent":
        return cls(AgentEventType.TOOL_ERROR)

    @classmethod
    def timeout(cls) -> "AgentEvent":
        return cls(AgentEventType.TIMEOUT)


@dataclass(frozen=True, slots=True)
class AgentState:
    phase: AgentPhase = AgentPhase.AWAITING_ASSISTANT
    tool_attempt_count: int = 0
    successful_tool_call_count: int = 0
    pending_attempt_index: int | None = None
    pending_call_index: int | None = None
    pending_cap_error: bool = False
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        if self.tool_attempt_count < 0 or self.successful_tool_call_count < 0:
            raise ValueError("tool counts must be non-negative")
        if self.successful_tool_call_count > self.tool_attempt_count:
            raise ValueError("successful calls cannot exceed attempted calls")
        if self.phase is AgentPhase.AWAITING_TOOL_RESPONSE:
            if self.pending_attempt_index is None:
                raise ValueError("a pending tool response requires an attempt index")
            if self.pending_cap_error and self.pending_call_index is not None:
                raise ValueError("a cap error cannot reserve a successful call index")
        elif self.pending_attempt_index is not None or self.pending_call_index is not None:
            raise ValueError("only a pending tool response may retain call indices")
        elif self.pending_cap_error:
            raise ValueError("only a pending tool response may retain a cap error")
        if self.phase is AgentPhase.TERMINATED:
            if self.termination_reason is None:
                raise ValueError("a terminated state requires a reason")
        elif self.termination_reason is not None:
            raise ValueError("a live state cannot have a termination reason")

    @property
    def tool_call_count(self) -> int:
        """Compatibility alias; the safety budget counts attempts, not successes."""

        return self.tool_attempt_count


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: AgentState
    execute_tool: bool = False
    emit_error: bool = False
    attempt_index: int | None = None
    call_index: int | None = None

    def __post_init__(self) -> None:
        if self.execute_tool and self.emit_error:
            raise ValueError("a transition cannot execute a tool and emit an error")
        if self.execute_tool != (self.call_index is not None):
            raise ValueError("execute_tool and call_index must agree")
        if (self.execute_tool or self.emit_error) != (self.attempt_index is not None):
            raise ValueError("tool actions and attempt_index must agree")


@dataclass(frozen=True, slots=True)
class MultiCallStateMachine:
    """Bound tool attempts and make every error response an explicit transition."""

    max_tool_calls: int
    cap_error_behavior: CapErrorBehavior = CapErrorBehavior.TERMINATE_AFTER_ERROR

    def __post_init__(self) -> None:
        if type(self.max_tool_calls) is not int or self.max_tool_calls <= 1:
            raise ValueError("max_tool_calls must be an integer greater than one")
        if not isinstance(self.cap_error_behavior, CapErrorBehavior):
            raise TypeError("cap_error_behavior must be CapErrorBehavior")

    def initial_state(self) -> AgentState:
        return AgentState()

    def apply(self, state: AgentState, event: AgentEvent) -> TransitionResult:
        if state.tool_attempt_count > self.max_tool_calls + 1:
            raise InvalidTransitionError("state exceeds the bounded cap-error attempt")
        if state.phase is AgentPhase.TERMINATED:
            raise InvalidTransitionError("terminated trajectories cannot transition")
        if state.phase is AgentPhase.AWAITING_ASSISTANT:
            return self._apply_assistant_event(state, event)
        if state.phase is AgentPhase.AWAITING_FINAL_ANSWER:
            return self._apply_final_only_event(state, event)
        return self._apply_tool_event(state, event)

    def _apply_assistant_event(
        self, state: AgentState, event: AgentEvent
    ) -> TransitionResult:
        if event.kind in {
            AgentEventType.VALID_TOOL_CALL,
            AgentEventType.MALFORMED_ACTION,
        }:
            return self._begin_attempt(
                state,
                parsed=event.kind is AgentEventType.VALID_TOOL_CALL,
            )
        if event.kind is AgentEventType.FINAL_ANSWER:
            return self._terminate(state, TerminationReason.FINAL_ANSWER)
        if event.kind is AgentEventType.TIMEOUT:
            return self._terminate(state, TerminationReason.TIMEOUT)
        raise InvalidTransitionError(
            f"{event.kind.value} is invalid while awaiting an assistant action"
        )

    def _begin_attempt(self, state: AgentState, *, parsed: bool) -> TransitionResult:
        attempt_index = state.tool_attempt_count
        if attempt_index >= self.max_tool_calls:
            return TransitionResult(
                state=AgentState(
                    phase=AgentPhase.AWAITING_TOOL_RESPONSE,
                    tool_attempt_count=state.tool_attempt_count + 1,
                    successful_tool_call_count=state.successful_tool_call_count,
                    pending_attempt_index=attempt_index,
                    pending_cap_error=True,
                ),
                emit_error=True,
                attempt_index=attempt_index,
            )
        pending_call_index = (
            state.successful_tool_call_count if parsed else None
        )
        return TransitionResult(
            state=AgentState(
                phase=AgentPhase.AWAITING_TOOL_RESPONSE,
                tool_attempt_count=state.tool_attempt_count + 1,
                successful_tool_call_count=state.successful_tool_call_count,
                pending_attempt_index=attempt_index,
                pending_call_index=pending_call_index,
            ),
            execute_tool=parsed,
            emit_error=not parsed,
            attempt_index=attempt_index,
            call_index=pending_call_index,
        )

    def _apply_tool_event(
        self, state: AgentState, event: AgentEvent
    ) -> TransitionResult:
        if event.kind is AgentEventType.TOOL_RESPONSE:
            if state.pending_call_index is None or state.pending_cap_error:
                raise InvalidTransitionError(
                    "a successful response requires one executed pending call"
                )
            return TransitionResult(
                AgentState(
                    phase=AgentPhase.AWAITING_ASSISTANT,
                    tool_attempt_count=state.tool_attempt_count,
                    successful_tool_call_count=state.successful_tool_call_count + 1,
                )
            )
        if event.kind is AgentEventType.TOOL_ERROR:
            if state.pending_cap_error:
                if self.cap_error_behavior is CapErrorBehavior.ONE_FINAL_ANSWER_TURN:
                    return TransitionResult(
                        AgentState(
                            phase=AgentPhase.AWAITING_FINAL_ANSWER,
                            tool_attempt_count=state.tool_attempt_count,
                            successful_tool_call_count=state.successful_tool_call_count,
                        )
                    )
                return self._terminate(state, TerminationReason.TOOL_CALL_CAP)
            return TransitionResult(
                AgentState(
                    phase=AgentPhase.AWAITING_ASSISTANT,
                    tool_attempt_count=state.tool_attempt_count,
                    successful_tool_call_count=state.successful_tool_call_count,
                )
            )
        if event.kind is AgentEventType.TIMEOUT:
            return self._terminate(state, TerminationReason.TIMEOUT)
        raise InvalidTransitionError(
            f"{event.kind.value} is invalid while awaiting a tool response"
        )

    def _apply_final_only_event(
        self, state: AgentState, event: AgentEvent
    ) -> TransitionResult:
        if event.kind is AgentEventType.FINAL_ANSWER:
            return self._terminate(state, TerminationReason.FINAL_ANSWER)
        if event.kind is AgentEventType.TIMEOUT:
            return self._terminate(state, TerminationReason.TIMEOUT)
        if event.kind in {
            AgentEventType.VALID_TOOL_CALL,
            AgentEventType.MALFORMED_ACTION,
        }:
            return self._terminate(state, TerminationReason.TOOL_CALL_CAP)
        raise InvalidTransitionError(
            f"{event.kind.value} is invalid while awaiting the final answer"
        )

    @staticmethod
    def _terminate(state: AgentState, reason: TerminationReason) -> TransitionResult:
        return TransitionResult(
            AgentState(
                phase=AgentPhase.TERMINATED,
                tool_attempt_count=state.tool_attempt_count,
                successful_tool_call_count=state.successful_tool_call_count,
                termination_reason=reason,
            )
        )
