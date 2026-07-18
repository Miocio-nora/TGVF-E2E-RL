"""Native Qwen TGVF tool protocol."""

from .parser import StrictToolCallParser
from .native import NativeProtocolRenderer, RenderedTranscript
from .schema import (
    ParseErrorCode,
    ParsedToolCall,
    SampledAssistantTurn,
    TGVF_FOCUS_TOOL_NAME,
    TGVF_FOCUS_TOOL_SCHEMA,
    TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON,
    TGVF_FOCUS_TOOL_SCHEMA_SHA256,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TargetSpan,
    TerminationReason,
    TextOffsets,
    TokenByteSpan,
    ToolCallParseError,
    build_tgvf_focus_tool_schema,
)
from .state_machine import (
    AgentEvent,
    AgentEventType,
    AgentPhase,
    AgentState,
    InvalidTransitionError,
    MultiCallStateMachine,
    TransitionResult,
)

__all__ = [
    "AgentEvent",
    "AgentEventType",
    "AgentPhase",
    "AgentState",
    "InvalidTransitionError",
    "MultiCallStateMachine",
    "NativeProtocolRenderer",
    "ParseErrorCode",
    "ParsedToolCall",
    "SampledAssistantTurn",
    "RenderedTranscript",
    "StrictToolCallParser",
    "TGVF_FOCUS_TOOL_NAME",
    "TGVF_FOCUS_TOOL_SCHEMA",
    "TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON",
    "TGVF_FOCUS_TOOL_SCHEMA_SHA256",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "TargetSpan",
    "TerminationReason",
    "TextOffsets",
    "TokenByteSpan",
    "ToolCallParseError",
    "TransitionResult",
    "build_tgvf_focus_tool_schema",
]
