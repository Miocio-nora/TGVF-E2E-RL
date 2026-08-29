"""Versioned interpretation of assistant tool-action boundaries.

This module classifies only the outer assistant-turn shape. Tool-name,
argument and schema validation remain the responsibility of the strict tool
parser. Keeping the boundary policy separate prevents an observation-renderer
change from silently changing whether a sampled turn executes a tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .schema import TOOL_CALL_CLOSE, TOOL_CALL_OPEN


class NativeActionBoundaryProtocolId(str, Enum):
    """Stable identities for historical and canonical action precedence."""

    LEGACY_ANSWER_OVER_ACTION_V1 = (
        "qwen-native-action-boundary-answer-over-action-legacy-v1"
    )
    STRICT_SINGLE_TERMINAL_TOOL_CALL_V2 = (
        "qwen-native-action-boundary-single-terminal-tool-call-v2"
    )


class AssistantTurnDisposition(str, Enum):
    DIRECT_FINAL = "direct_final"
    TOOL_ACTION = "tool_action"
    INVALID_ACTION = "invalid_action"


@dataclass(frozen=True, slots=True)
class AssistantActionBoundary:
    """Auditable outer-boundary classification for one sampled turn."""

    protocol_id: NativeActionBoundaryProtocolId
    disposition: AssistantTurnDisposition
    tool_call_blocks: tuple[str, ...]
    selected_tool_call: str | None
    final_text: str | None
    trailing_text: str
    violation_code: str | None


_TOOL_CALL_BLOCK = re.compile(
    re.escape(TOOL_CALL_OPEN) + r".*?" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


def classify_assistant_action_boundary(
    text: str,
    *,
    protocol_id: NativeActionBoundaryProtocolId | str,
) -> AssistantActionBoundary:
    """Classify a sampled turn without parsing the tool-call JSON payload.

    The strict protocol admits exactly one well-formed tool-call block and
    requires it to be terminal modulo whitespace. The legacy protocol records
    the historical answer-over-action and last-action behavior so old evidence
    can be reproduced only by naming that identity explicitly.
    """

    if not isinstance(text, str):
        raise TypeError("assistant action-boundary text must be str")
    try:
        selected_protocol = NativeActionBoundaryProtocolId(protocol_id)
    except (TypeError, ValueError) as error:
        raise ValueError("assistant action-boundary protocol ID is invalid") from error

    matches = tuple(_TOOL_CALL_BLOCK.finditer(text))
    blocks = tuple(match.group(0) for match in matches)
    open_count = text.count(TOOL_CALL_OPEN)
    close_count = text.count(TOOL_CALL_CLOSE)
    if open_count == 0 and close_count == 0:
        return AssistantActionBoundary(
            protocol_id=selected_protocol,
            disposition=AssistantTurnDisposition.DIRECT_FINAL,
            tool_call_blocks=(),
            selected_tool_call=None,
            final_text=text,
            trailing_text="",
            violation_code=None,
        )
    if open_count != close_count or len(matches) != open_count:
        return _invalid_boundary(
            selected_protocol,
            blocks=blocks,
            trailing_text="",
            violation_code="malformed_tool_call_tags",
        )

    trailing_text = text[matches[-1].end() :]
    if selected_protocol is NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1:
        if trailing_text.strip():
            return AssistantActionBoundary(
                protocol_id=selected_protocol,
                disposition=AssistantTurnDisposition.DIRECT_FINAL,
                tool_call_blocks=blocks,
                selected_tool_call=None,
                final_text=trailing_text.strip(),
                trailing_text=trailing_text,
                violation_code=None,
            )
        return AssistantActionBoundary(
            protocol_id=selected_protocol,
            disposition=AssistantTurnDisposition.TOOL_ACTION,
            tool_call_blocks=blocks,
            selected_tool_call=blocks[-1],
            final_text=None,
            trailing_text=trailing_text,
            violation_code=None,
        )

    if len(blocks) != 1:
        return _invalid_boundary(
            selected_protocol,
            blocks=blocks,
            trailing_text=trailing_text,
            violation_code="multiple_tool_calls",
        )
    if trailing_text.strip():
        return _invalid_boundary(
            selected_protocol,
            blocks=blocks,
            trailing_text=trailing_text,
            violation_code="tool_call_terminal_suffix",
        )
    return AssistantActionBoundary(
        protocol_id=selected_protocol,
        disposition=AssistantTurnDisposition.TOOL_ACTION,
        tool_call_blocks=blocks,
        selected_tool_call=blocks[0],
        final_text=None,
        trailing_text=trailing_text,
        violation_code=None,
    )


def _invalid_boundary(
    protocol_id: NativeActionBoundaryProtocolId,
    *,
    blocks: tuple[str, ...],
    trailing_text: str,
    violation_code: str,
) -> AssistantActionBoundary:
    return AssistantActionBoundary(
        protocol_id=protocol_id,
        disposition=AssistantTurnDisposition.INVALID_ACTION,
        tool_call_blocks=blocks,
        selected_tool_call=None,
        final_text=None,
        trailing_text=trailing_text,
        violation_code=violation_code,
    )


__all__ = [
    "AssistantActionBoundary",
    "AssistantTurnDisposition",
    "NativeActionBoundaryProtocolId",
    "classify_assistant_action_boundary",
]
