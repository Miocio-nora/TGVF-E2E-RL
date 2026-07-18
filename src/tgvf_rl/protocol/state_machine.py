"""Fail-closed repeated-call state machine for the native tool environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schema import ParsedToolCall, TerminationReason


class AgentPhase(str, Enum):
    AWAITING_ASSISTANT = "awaiting_assistant"
    AWAITING_TOOL_RESPONSE = "awaiting_tool_response"
    TERMINATED = "terminated"


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
    tool_call: ParsedToolCall | None = None

    def __post_init__(self) -> None:
        if self.kind is AgentEventType.VALID_TOOL_CALL and self.tool_call is None:
            raise ValueError(
                "a valid-tool-call event requires the parsed immutable call"
            )
        if (
            self.kind is not AgentEventType.VALID_TOOL_CALL
            and self.tool_call is not None
        ):
            raise ValueError("only a valid-tool-call event may carry a parsed call")

    @classmethod
    def valid_tool_call(cls, call: ParsedToolCall) -> "AgentEvent":
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
    tool_call_count: int = 0
    pending_call_index: int | None = None
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        if self.tool_call_count < 0:
            raise ValueError("tool_call_count must be non-negative")
        if self.phase is AgentPhase.AWAITING_TOOL_RESPONSE:
            if self.pending_call_index is None:
                raise ValueError("a pending tool response requires a call index")
        elif self.pending_call_index is not None:
            raise ValueError("only a pending tool response may retain a call index")
        if self.phase is AgentPhase.TERMINATED:
            if self.termination_reason is None:
                raise ValueError("a terminated state requires a reason")
        elif self.termination_reason is not None:
            raise ValueError("a live state cannot have a termination reason")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: AgentState
    execute_tool: bool = False
    call_index: int | None = None

    def __post_init__(self) -> None:
        if self.execute_tool != (self.call_index is not None):
            raise ValueError("execute_tool and call_index must agree")


@dataclass(frozen=True, slots=True)
class MultiCallStateMachine:
    """Allow repeated calls up to an explicit safety cap greater than one."""

    max_tool_calls: int

    def __post_init__(self) -> None:
        if type(self.max_tool_calls) is not int or self.max_tool_calls <= 1:
            raise ValueError("max_tool_calls must be an integer greater than one")

    def initial_state(self) -> AgentState:
        return AgentState()

    def apply(self, state: AgentState, event: AgentEvent) -> TransitionResult:
        if state.tool_call_count > self.max_tool_calls:
            raise InvalidTransitionError("state exceeds this machine's tool-call cap")
        if state.phase is AgentPhase.TERMINATED:
            raise InvalidTransitionError("terminated trajectories cannot transition")
        if state.phase is AgentPhase.AWAITING_ASSISTANT:
            return self._apply_assistant_event(state, event)
        return self._apply_tool_event(state, event)

    def _apply_assistant_event(
        self, state: AgentState, event: AgentEvent
    ) -> TransitionResult:
        if event.kind is AgentEventType.VALID_TOOL_CALL:
            if state.tool_call_count >= self.max_tool_calls:
                return TransitionResult(
                    AgentState(
                        phase=AgentPhase.TERMINATED,
                        tool_call_count=state.tool_call_count,
                        termination_reason=TerminationReason.TOOL_CALL_CAP,
                    )
                )
            call_index = state.tool_call_count
            return TransitionResult(
                state=AgentState(
                    phase=AgentPhase.AWAITING_TOOL_RESPONSE,
                    tool_call_count=state.tool_call_count + 1,
                    pending_call_index=call_index,
                ),
                execute_tool=True,
                call_index=call_index,
            )
        if event.kind is AgentEventType.FINAL_ANSWER:
            return self._terminate(state, TerminationReason.FINAL_ANSWER)
        if event.kind is AgentEventType.MALFORMED_ACTION:
            return self._terminate(state, TerminationReason.MALFORMED_ACTION)
        if event.kind is AgentEventType.TIMEOUT:
            return self._terminate(state, TerminationReason.TIMEOUT)
        raise InvalidTransitionError(
            f"{event.kind.value} is invalid while awaiting an assistant action"
        )

    def _apply_tool_event(
        self, state: AgentState, event: AgentEvent
    ) -> TransitionResult:
        if event.kind is AgentEventType.TOOL_RESPONSE:
            return TransitionResult(
                AgentState(
                    phase=AgentPhase.AWAITING_ASSISTANT,
                    tool_call_count=state.tool_call_count,
                )
            )
        if event.kind is AgentEventType.TOOL_ERROR:
            return self._terminate(state, TerminationReason.TOOL_ERROR)
        if event.kind is AgentEventType.TIMEOUT:
            return self._terminate(state, TerminationReason.TIMEOUT)
        raise InvalidTransitionError(
            f"{event.kind.value} is invalid while awaiting a tool response"
        )

    @staticmethod
    def _terminate(state: AgentState, reason: TerminationReason) -> TransitionResult:
        return TransitionResult(
            AgentState(
                phase=AgentPhase.TERMINATED,
                tool_call_count=state.tool_call_count,
                termination_reason=reason,
            )
        )
