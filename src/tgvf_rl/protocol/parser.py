"""Strict parser for one registered native visual-tool call."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .schema import (
    CROP_TGVF_TOOL_NAME,
    ParseErrorCode,
    ParsedCropTGVFCall,
    ParsedImageZoomInCall,
    ParsedToolCall,
    NativeToolCall,
    SampledAssistantTurn,
    IMAGE_ZOOM_IN_TOOL_NAME,
    POLICY_RL_TOOL_NAMES,
    TGVF_FOCUS_TOOL_NAME,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TargetSpan,
    TextOffsets,
    ToolCallParseError,
)


@dataclass(frozen=True, slots=True)
class _JsonNode:
    value: Any
    start: int
    end: int
    content_start: int | None = None
    content_end: int | None = None
    members: dict[str, "_JsonNode"] | None = None


class _JsonSpanScanner:
    """Locate JSON value spans after strict ``json.loads`` validation."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)

    def parse(self) -> _JsonNode:
        node, cursor = self._parse_value(self._skip_ws(0))
        if self._skip_ws(cursor) != self.length:
            raise ValueError("trailing data after JSON value")
        return node

    def _skip_ws(self, cursor: int) -> int:
        while cursor < self.length and self.source[cursor] in " \t\r\n":
            cursor += 1
        return cursor

    def _parse_value(self, cursor: int) -> tuple[_JsonNode, int]:
        if cursor >= self.length:
            raise ValueError("expected JSON value")
        marker = self.source[cursor]
        if marker == '"':
            return self._parse_string(cursor)
        if marker == "{":
            return self._parse_object(cursor)
        if marker == "[":
            return self._parse_array(cursor)
        return self._parse_primitive(cursor)

    def _parse_string(self, cursor: int) -> tuple[_JsonNode, int]:
        start = cursor
        cursor += 1
        while cursor < self.length:
            marker = self.source[cursor]
            if marker == "\\":
                cursor += 2
                continue
            if marker == '"':
                end = cursor + 1
                raw = self.source[start:end]
                value = json.loads(raw, strict=True)
                return (
                    _JsonNode(
                        value=value,
                        start=start,
                        end=end,
                        content_start=start + 1,
                        content_end=end - 1,
                    ),
                    end,
                )
            cursor += 1
        raise ValueError("unterminated JSON string")

    def _parse_object(self, cursor: int) -> tuple[_JsonNode, int]:
        start = cursor
        cursor = self._skip_ws(cursor + 1)
        members: dict[str, _JsonNode] = {}
        value: dict[str, Any] = {}
        if cursor < self.length and self.source[cursor] == "}":
            end = cursor + 1
            return _JsonNode(value, start, end, members=members), end

        while True:
            key_node, cursor = self._parse_string(cursor)
            key = key_node.value
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string")
            if key in members:
                raise ValueError(f"duplicate JSON key: {key!r}")
            cursor = self._skip_ws(cursor)
            if cursor >= self.length or self.source[cursor] != ":":
                raise ValueError("expected ':' after JSON object key")
            child, cursor = self._parse_value(self._skip_ws(cursor + 1))
            members[key] = child
            value[key] = child.value
            cursor = self._skip_ws(cursor)
            if cursor >= self.length:
                raise ValueError("unterminated JSON object")
            if self.source[cursor] == "}":
                end = cursor + 1
                return _JsonNode(value, start, end, members=members), end
            if self.source[cursor] != ",":
                raise ValueError("expected ',' or '}' in JSON object")
            cursor = self._skip_ws(cursor + 1)

    def _parse_array(self, cursor: int) -> tuple[_JsonNode, int]:
        start = cursor
        cursor = self._skip_ws(cursor + 1)
        value: list[Any] = []
        if cursor < self.length and self.source[cursor] == "]":
            end = cursor + 1
            return _JsonNode(value, start, end), end
        while True:
            child, cursor = self._parse_value(cursor)
            value.append(child.value)
            cursor = self._skip_ws(cursor)
            if cursor >= self.length:
                raise ValueError("unterminated JSON array")
            if self.source[cursor] == "]":
                end = cursor + 1
                return _JsonNode(value, start, end), end
            if self.source[cursor] != ",":
                raise ValueError("expected ',' or ']' in JSON array")
            cursor = self._skip_ws(cursor + 1)

    def _parse_primitive(self, cursor: int) -> tuple[_JsonNode, int]:
        start = cursor
        while cursor < self.length and self.source[cursor] not in " \t\r\n,]}":
            cursor += 1
        if cursor == start:
            raise ValueError("expected JSON primitive")
        raw = self.source[start:cursor]
        value = json.loads(raw, parse_constant=_reject_json_constant, strict=True)
        return _JsonNode(value, start, cursor), cursor


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(source: str) -> Any:
    return json.loads(
        source,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
        strict=True,
    )


def _text_offsets(text: str, char_start: int, char_end: int) -> TextOffsets:
    return TextOffsets(
        char_start=char_start,
        char_end=char_end,
        byte_start=len(text[:char_start].encode("utf-8")),
        byte_end=len(text[:char_end].encode("utf-8")),
    )


def _map_target_tokens(
    turn: SampledAssistantTurn, offsets: TextOffsets
) -> tuple[int, int, tuple[int, ...]]:
    overlapping = [
        span
        for span in turn.token_byte_spans
        if span.byte_end > offsets.byte_start and span.byte_start < offsets.byte_end
    ]
    if not overlapping:
        raise ToolCallParseError(
            ParseErrorCode.AMBIGUOUS_TARGET_TOKEN_SPAN,
            "no sampled token covers the raw target value",
        )

    if any(span.byte_start == span.byte_end for span in overlapping):
        raise ToolCallParseError(
            ParseErrorCode.AMBIGUOUS_TARGET_TOKEN_SPAN,
            "a zero-width sampled token overlaps the raw target value",
        )

    token_start = overlapping[0].token_index
    token_end = overlapping[-1].token_index + 1
    expected_indices = list(range(token_start, token_end))
    actual_indices = [span.token_index for span in overlapping]
    if actual_indices != expected_indices:
        raise ToolCallParseError(
            ParseErrorCode.AMBIGUOUS_TARGET_TOKEN_SPAN,
            "target token coverage is not contiguous",
        )
    if (
        overlapping[0].byte_start > offsets.byte_start
        or overlapping[-1].byte_end < offsets.byte_end
    ):
        raise ToolCallParseError(
            ParseErrorCode.AMBIGUOUS_TARGET_TOKEN_SPAN,
            "sampled target tokens do not cover the raw target value",
        )
    token_ids = turn.token_ids[token_start:token_end]
    return token_start, token_end, token_ids


@dataclass(frozen=True, slots=True)
class StrictToolCallParser:
    """Parse exactly one complete native tool call without rewriting it."""

    allowed_terminal_suffixes: tuple[str, ...] = ()
    enabled_tool_names: tuple[str, ...] = POLICY_RL_TOOL_NAMES

    def __post_init__(self) -> None:
        suffixes = tuple(self.allowed_terminal_suffixes)
        if any(not suffix for suffix in suffixes) or len(set(suffixes)) != len(
            suffixes
        ):
            raise ValueError(
                "allowed terminal suffixes must be unique non-empty strings"
            )
        object.__setattr__(self, "allowed_terminal_suffixes", suffixes)
        names = tuple(self.enabled_tool_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("enabled tool names must be non-empty and unique")
        unknown = set(names) - set(POLICY_RL_TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown enabled tool names: {sorted(unknown)!r}")
        object.__setattr__(self, "enabled_tool_names", names)

    def parse(self, turn: SampledAssistantTurn) -> NativeToolCall:
        text = turn.sampled_text
        opening_count = text.count(TOOL_CALL_OPEN)
        closing_count = text.count(TOOL_CALL_CLOSE)
        if opening_count == 0 and closing_count == 0:
            raise ToolCallParseError(
                ParseErrorCode.MISSING_TOOL_CALL, "no native tool call was sampled"
            )
        if opening_count > 1 or closing_count > 1:
            raise ToolCallParseError(
                ParseErrorCode.MULTIPLE_TOOL_CALLS,
                "an assistant action turn must contain exactly one call object",
            )
        if opening_count != 1 or closing_count != 1:
            raise ToolCallParseError(
                ParseErrorCode.INCOMPLETE_TOOL_CALL,
                "native tool-call markers are incomplete",
            )

        call_start = text.index(TOOL_CALL_OPEN)
        json_start = call_start + len(TOOL_CALL_OPEN)
        json_end = text.index(TOOL_CALL_CLOSE)
        if json_end < json_start:
            raise ToolCallParseError(
                ParseErrorCode.INCOMPLETE_TOOL_CALL,
                "closing marker precedes opening marker",
            )
        call_end = json_end + len(TOOL_CALL_CLOSE)
        trailing = text[call_end:]
        if not self._terminal_suffix_is_allowed(trailing):
            raise ToolCallParseError(
                ParseErrorCode.TRAILING_ASSISTANT_TEXT,
                "assistant text after the tool call is forbidden",
            )

        raw_json = text[json_start:json_end]
        try:
            decoded = _strict_json_loads(raw_json)
            root = _JsonSpanScanner(raw_json).parse()
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ToolCallParseError(ParseErrorCode.INVALID_JSON, str(error)) from error
        if root.value != decoded:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_JSON,
                "JSON decoder and span scanner disagree",
            )

        if type(decoded) is not dict or set(decoded) != {"name", "arguments"}:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_CALL_SHAPE,
                "call object must contain exactly 'name' and 'arguments'",
            )
        tool_name = decoded["name"]
        if tool_name not in self.enabled_tool_names:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_TOOL_NAME,
                f"expected one of {self.enabled_tool_names!r}",
            )
        arguments = decoded["arguments"]
        if tool_name == IMAGE_ZOOM_IN_TOOL_NAME:
            return self._parse_crop_call(
                decoded=decoded,
                arguments=arguments,
                turn=turn,
                text=text,
                raw_json=raw_json,
                call_start=call_start,
                call_end=call_end,
                json_start=json_start,
                json_end=json_end,
            )
        if tool_name == CROP_TGVF_TOOL_NAME:
            return self._parse_crop_tgvf_call(
                root=root,
                decoded=decoded,
                arguments=arguments,
                turn=turn,
                text=text,
                raw_json=raw_json,
                call_start=call_start,
                call_end=call_end,
                json_start=json_start,
                json_end=json_end,
            )
        if type(arguments) is not dict or set(arguments) != {"target"}:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS,
                "arguments must contain exactly one 'target' string",
            )
        target = arguments["target"]
        if not isinstance(target, str):
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS, "target must be a string"
            )
        if not target.strip():
            raise ToolCallParseError(
                ParseErrorCode.EMPTY_TARGET, "target must be non-empty"
            )

        if root.members is None or root.members["arguments"].members is None:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_CALL_SHAPE, "arguments must be an object"
            )
        target_node = root.members["arguments"].members["target"]
        if target_node.content_start is None or target_node.content_end is None:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS, "target must be a JSON string"
            )

        target_char_start = json_start + target_node.content_start
        target_char_end = json_start + target_node.content_end
        target_offsets = _text_offsets(text, target_char_start, target_char_end)
        token_start, token_end, target_token_ids = _map_target_tokens(
            turn, target_offsets
        )

        return ParsedToolCall(
            name=TGVF_FOCUS_TOOL_NAME,
            target=target,
            sampled_text=text,
            sampled_token_ids=turn.token_ids,
            sampled_token_byte_spans=turn.token_byte_spans,
            raw_tool_call=text[call_start:call_end],
            raw_json=raw_json,
            call_offsets=_text_offsets(text, call_start, call_end),
            json_offsets=_text_offsets(text, json_start, json_end),
            target_span=TargetSpan(
                target_text=target,
                raw_json_value=text[target_char_start:target_char_end],
                offsets=target_offsets,
                token_start=token_start,
                token_end=token_end,
                token_ids=target_token_ids,
            ),
        )

    @staticmethod
    def _parse_crop_call(
        *,
        decoded: dict[str, Any],
        arguments: Any,
        turn: SampledAssistantTurn,
        text: str,
        raw_json: str,
        call_start: int,
        call_end: int,
        json_start: int,
        json_end: int,
    ) -> ParsedImageZoomInCall:
        if type(arguments) is not dict or set(arguments) != {"bbox_2d"}:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS,
                "crop arguments must contain exactly 'bbox_2d'",
            )
        bbox = arguments["bbox_2d"]
        if (
            type(bbox) is not list
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise ToolCallParseError(
                ParseErrorCode.INVALID_BBOX,
                "bbox_2d must be a JSON array of exactly four integers",
            )
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_BBOX,
                "bbox_2d must have positive requested width and height",
            )
        return ParsedImageZoomInCall(
            name=decoded["name"],
            bbox_2d=(left, top, right, bottom),
            sampled_text=text,
            sampled_token_ids=turn.token_ids,
            sampled_token_byte_spans=turn.token_byte_spans,
            raw_tool_call=text[call_start:call_end],
            raw_json=raw_json,
            call_offsets=_text_offsets(text, call_start, call_end),
            json_offsets=_text_offsets(text, json_start, json_end),
        )

    @staticmethod
    def _parse_crop_tgvf_call(
        *,
        root: _JsonNode,
        decoded: dict[str, Any],
        arguments: Any,
        turn: SampledAssistantTurn,
        text: str,
        raw_json: str,
        call_start: int,
        call_end: int,
        json_start: int,
        json_end: int,
    ) -> ParsedCropTGVFCall:
        if type(arguments) is not dict or set(arguments) != {"bbox_2d", "target"}:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS,
                "atomic crop/TGVF arguments must contain exactly 'bbox_2d' and "
                "'target'",
            )
        bbox = arguments["bbox_2d"]
        if (
            type(bbox) is not list
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise ToolCallParseError(
                ParseErrorCode.INVALID_BBOX,
                "bbox_2d must be a JSON array of exactly four integers",
            )
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_BBOX,
                "bbox_2d must have positive requested width and height",
            )
        target = arguments["target"]
        if not isinstance(target, str):
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS, "target must be a string"
            )
        if not target.strip():
            raise ToolCallParseError(
                ParseErrorCode.EMPTY_TARGET, "target must be non-empty"
            )
        if root.members is None or root.members["arguments"].members is None:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_CALL_SHAPE, "arguments must be an object"
            )
        target_node = root.members["arguments"].members["target"]
        if target_node.content_start is None or target_node.content_end is None:
            raise ToolCallParseError(
                ParseErrorCode.INVALID_ARGUMENTS, "target must be a JSON string"
            )
        target_char_start = json_start + target_node.content_start
        target_char_end = json_start + target_node.content_end
        target_offsets = _text_offsets(text, target_char_start, target_char_end)
        token_start, token_end, target_token_ids = _map_target_tokens(
            turn, target_offsets
        )
        return ParsedCropTGVFCall(
            name=decoded["name"],
            bbox_2d=(left, top, right, bottom),
            target=target,
            sampled_text=text,
            sampled_token_ids=turn.token_ids,
            sampled_token_byte_spans=turn.token_byte_spans,
            raw_tool_call=text[call_start:call_end],
            raw_json=raw_json,
            call_offsets=_text_offsets(text, call_start, call_end),
            json_offsets=_text_offsets(text, json_start, json_end),
            target_span=TargetSpan(
                target_text=target,
                raw_json_value=text[target_char_start:target_char_end],
                offsets=target_offsets,
                token_start=token_start,
                token_end=token_end,
                token_ids=target_token_ids,
            ),
        )

    def _terminal_suffix_is_allowed(self, suffix: str) -> bool:
        stripped = suffix.strip()
        if not stripped:
            return True
        return stripped in self.allowed_terminal_suffixes
