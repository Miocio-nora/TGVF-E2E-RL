"""Framework-neutral records for Qwen's native TGVF tool protocol.

This module deliberately has no tokenizer dependency.  A caller supplies the
already-sampled text, token IDs, and exact byte coverage for every token.  The
parser may inspect those records, but it never decodes, rerenders, or retokenizes
the assistant turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from copy import deepcopy
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence


TGVF_FOCUS_TOOL_NAME = "tgvf_focus_tool"
IMAGE_ZOOM_IN_TOOL_NAME = "image_zoom_in_tool"
TGVF_CROP_TOOL_NAME = "tgvf_crop_tool"
REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_VERSION = "tgvf-focus-tool-v1"
TGVF_FOCUS_TOOL_SCHEMA_VERSION = "tgvf-focus-tool-policy-v1"
IMAGE_ZOOM_IN_TOOL_SCHEMA_VERSION = "image-zoom-in-tool-v2"
TGVF_CROP_TOOL_SCHEMA_VERSION = "tgvf-crop-tool-v2"
POLICY_RL_TOOL_NAMES = (
    TGVF_FOCUS_TOOL_NAME,
    IMAGE_ZOOM_IN_TOOL_NAME,
    TGVF_CROP_TOOL_NAME,
)
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TARGET_TOKEN_SPAN_RULE = "minimal_overlapping_sampled_token_cover_v1"
STANDARD_TOOL_ERROR_SCHEMA_VERSION = "tgvf-native-tool-error-v1"
NATIVE_TARGET_ECHO_FORBIDDEN_FRAGMENTS = (
    "\r",
    "\n",
    "<|",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


def validate_native_tool_target_for_echo(target: str) -> str:
    """Reject content that could become a native control token when echoed."""

    if not isinstance(target, str):
        raise TypeError("tool target must be a string")
    if not target.strip():
        raise ValueError("tool target must be non-empty")
    if any(fragment in target for fragment in NATIVE_TARGET_ECHO_FORBIDDEN_FRAGMENTS):
        raise ValueError("tool target cannot contain newlines or native control tags")
    try:
        target.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("tool target must contain valid Unicode scalar values") from error
    return target

_REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TGVF_FOCUS_TOOL_NAME,
        "description": "Inspect one specific visual region before answering.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "A neutral, visually locatable region description.",
                }
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
}

_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TGVF_FOCUS_TOOL_NAME,
        "description": (
            "Generate a target-conditioned visual observation for a visual "
            "inspection request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "A concise, self-contained visual query specifying both "
                        "what to inspect and what evidence or relation to extract, "
                        "read, compare, count, or verify. Do not include a guessed "
                        "final answer."
                    ),
                }
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
}

_IMAGE_ZOOM_IN_TOOL_SCHEMA_MUTABLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": IMAGE_ZOOM_IN_TOOL_NAME,
        "description": "Crop and enlarge a selected region of the original image.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The crop bounding box [x1, y1, x2, y2]. For Qwen3-VL, "
                        "use integer coordinates on the original-image-relative "
                        "0..1000 grid, with x2 > x1 and y2 > y1."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "A short description of the object or region inside the "
                        "bounding box."
                    ),
                },
            },
            "required": ["bbox_2d"],
            "additionalProperties": False,
        },
    },
}

_TGVF_CROP_TOOL_SCHEMA_MUTABLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TGVF_CROP_TOOL_NAME,
        "description": (
            "Crop a selected region from the original image and return a "
            "target-conditioned visual representation of that crop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The crop bounding box [x1, y1, x2, y2]. For Qwen3-VL, "
                        "use integer coordinates on the original-image-relative "
                        "0..1000 grid, with x2 > x1 and y2 > y1."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "A concise, self-contained visual query specifying both "
                        "what to inspect inside the crop and what evidence or "
                        "relation to extract, read, compare, count, or verify. Do "
                        "not include a guessed final answer."
                    ),
                },
            },
            "required": ["bbox_2d", "target"],
            "additionalProperties": False,
        },
    },
}


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA: Mapping[str, Any] = _freeze_json(
    _REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE
)
REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON = json.dumps(
    _REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)
REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_SHA256 = sha256(
    REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON.encode("utf-8")
).hexdigest()
TGVF_FOCUS_TOOL_SCHEMA: Mapping[str, Any] = _freeze_json(
    _TGVF_FOCUS_TOOL_SCHEMA_MUTABLE
)
TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON = json.dumps(
    _TGVF_FOCUS_TOOL_SCHEMA_MUTABLE,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)
TGVF_FOCUS_TOOL_SCHEMA_SHA256 = sha256(
    TGVF_FOCUS_TOOL_SCHEMA_CANONICAL_JSON.encode("utf-8")
).hexdigest()
IMAGE_ZOOM_IN_TOOL_SCHEMA: Mapping[str, Any] = _freeze_json(
    _IMAGE_ZOOM_IN_TOOL_SCHEMA_MUTABLE
)
IMAGE_ZOOM_IN_TOOL_SCHEMA_CANONICAL_JSON = json.dumps(
    _IMAGE_ZOOM_IN_TOOL_SCHEMA_MUTABLE,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)
IMAGE_ZOOM_IN_TOOL_SCHEMA_SHA256 = sha256(
    IMAGE_ZOOM_IN_TOOL_SCHEMA_CANONICAL_JSON.encode("utf-8")
).hexdigest()
TGVF_CROP_TOOL_SCHEMA: Mapping[str, Any] = _freeze_json(
    _TGVF_CROP_TOOL_SCHEMA_MUTABLE
)
TGVF_CROP_TOOL_SCHEMA_CANONICAL_JSON = json.dumps(
    _TGVF_CROP_TOOL_SCHEMA_MUTABLE,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)
TGVF_CROP_TOOL_SCHEMA_SHA256 = sha256(
    TGVF_CROP_TOOL_SCHEMA_CANONICAL_JSON.encode("utf-8")
).hexdigest()


class NativeToolCapabilityProfile(str, Enum):
    """One separately identified visual-tool capability surface."""

    CROP_ONLY = "crop_only"
    TGVF_ONLY = "tgvf_only"
    CROP_TGVF = "crop_tgvf"

    @property
    def tool_names(self) -> tuple[str, ...]:
        return {
            NativeToolCapabilityProfile.CROP_ONLY: (IMAGE_ZOOM_IN_TOOL_NAME,),
            NativeToolCapabilityProfile.TGVF_ONLY: (TGVF_FOCUS_TOOL_NAME,),
            NativeToolCapabilityProfile.CROP_TGVF: (TGVF_CROP_TOOL_NAME,),
        }[self]

    @property
    def tool_set_sha256(self) -> str:
        return native_tool_set_sha256(self.tool_names)


def build_tgvf_focus_tool_schema() -> dict[str, Any]:
    """Return a fresh JSON-compatible copy of the policy-RL v1 schema."""

    # Preserve the declared schema field order because it is part of the Qwen
    # system-prompt token identity, while the separate canonical JSON remains
    # the order-independent schema digest.
    return deepcopy(_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE)


def build_representation_tgvf_focus_tool_schema() -> dict[str, Any]:
    """Return the frozen representation-phase schema as a fresh JSON object."""

    return deepcopy(_REPRESENTATION_TGVF_FOCUS_TOOL_SCHEMA_MUTABLE)


def build_image_zoom_in_tool_schema() -> dict[str, Any]:
    """Return a fresh copy of the accepted DeepEyes-compatible crop schema."""

    return deepcopy(_IMAGE_ZOOM_IN_TOOL_SCHEMA_MUTABLE)


def build_tgvf_crop_tool_schema() -> dict[str, Any]:
    """Return a fresh copy of the atomic crop-then-foveate schema."""

    return deepcopy(_TGVF_CROP_TOOL_SCHEMA_MUTABLE)


def build_native_tool_schemas(
    tool_names: tuple[str, ...] = POLICY_RL_TOOL_NAMES,
) -> list[dict[str, Any]]:
    """Build an ordered, duplicate-free set of registered native tool schemas."""

    names = tuple(tool_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("native tool names must be non-empty and unique")
    builders = {
        TGVF_FOCUS_TOOL_NAME: build_tgvf_focus_tool_schema,
        IMAGE_ZOOM_IN_TOOL_NAME: build_image_zoom_in_tool_schema,
        TGVF_CROP_TOOL_NAME: build_tgvf_crop_tool_schema,
    }
    unknown = tuple(name for name in names if name not in builders)
    if unknown:
        raise ValueError(f"unknown native tool names: {unknown!r}")
    return [builders[name]() for name in names]


def native_tool_set_sha256(tool_names: tuple[str, ...]) -> str:
    """Hash the exact ordered tool set passed to the Qwen chat template."""

    names = tuple(tool_names)
    return native_tool_schemas_sha256(build_native_tool_schemas(names))


def native_tool_schemas_sha256(
    schemas: Sequence[Mapping[str, Any]],
) -> str:
    """Hash one canonical schema object or an ordered list of schema objects."""

    schema_items = tuple(schemas)
    if not schema_items:
        raise ValueError("native tool schemas must be non-empty")
    if any(not isinstance(schema, Mapping) for schema in schema_items):
        raise TypeError("each native tool schema must be a mapping")
    json_items = tuple(_thaw_json(schema) for schema in schema_items)
    payload: Any = json_items[0] if len(json_items) == 1 else list(json_items)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


class ParseErrorCode(str, Enum):
    MISSING_THINK_CLOSER = "missing_think_closer"
    MULTIPLE_THINK_CLOSERS = "multiple_think_closers"
    DUPLICATE_THINK_OPENER = "duplicate_think_opener"
    THINK_CLOSE_AFTER_TOOL_OPEN = "think_close_after_tool_open"
    MISSING_TOOL_CALL = "missing_tool_call"
    INCOMPLETE_TOOL_CALL = "incomplete_tool_call"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    TRAILING_ASSISTANT_TEXT = "trailing_assistant_text"
    INVALID_JSON = "invalid_json"
    INVALID_CALL_SHAPE = "invalid_call_shape"
    INVALID_TOOL_NAME = "invalid_tool_name"
    INVALID_ARGUMENTS = "invalid_arguments"
    EMPTY_TARGET = "empty_target"
    INVALID_BBOX = "invalid_bbox"
    AMBIGUOUS_TARGET_TOKEN_SPAN = "ambiguous_target_token_span"


class ToolErrorCode(str, Enum):
    TOOL_NOT_ENABLED = "tool_not_enabled"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_RESPONSE_APPEND_FAILED = "tool_response_append_failed"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"


@dataclass(frozen=True, slots=True)
class StandardToolError:
    """Deterministic environment-owned native tool-error observation."""

    code: str
    message: str
    attempt_index: int
    recoverable: bool
    maximum_tool_calls: int
    schema_version: str = STANDARD_TOOL_ERROR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.schema_version or not self.code or not self.message:
            raise ValueError("tool error identity/code/message must be non-empty")
        if self.attempt_index < 0:
            raise ValueError("tool error attempt index must be non-negative")
        if self.maximum_tool_calls <= 1:
            raise ValueError("tool error maximum_tool_calls must be greater than one")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "schema_version": self.schema_version,
                "code": self.code,
                "message": self.message,
                "attempt_index": self.attempt_index,
                "maximum_tool_calls": self.maximum_tool_calls,
                "recoverable": self.recoverable,
            },
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def payload_sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


class ToolCallParseError(ValueError):
    """Fail-closed parse error with a stable machine-readable code."""

    def __init__(self, code: ParseErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True, slots=True)
class TextOffsets:
    """Half-open character and UTF-8 byte offsets into sampled assistant text."""

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.byte_start < 0:
            raise ValueError("text offsets must be non-negative")
        if self.char_end < self.char_start or self.byte_end < self.byte_start:
            raise ValueError("text offset ends must not precede their starts")


@dataclass(frozen=True, slots=True)
class TokenByteSpan:
    """Exact half-open UTF-8 byte coverage of one already-sampled token."""

    token_index: int
    token_id: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise ValueError("invalid token byte span")


@dataclass(frozen=True, slots=True)
class SampledAssistantTurn:
    """Immutable sampled assistant text and its caller-provided token identity."""

    sampled_text: str
    token_ids: tuple[int, ...]
    token_byte_spans: tuple[TokenByteSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_byte_spans", tuple(self.token_byte_spans))

        if len(self.token_ids) != len(self.token_byte_spans):
            raise ValueError("each sampled token must have exactly one byte span")

        total_bytes = len(self.sampled_text.encode("utf-8"))
        cursor = 0
        for expected_index, (token_id, span) in enumerate(
            zip(self.token_ids, self.token_byte_spans, strict=True)
        ):
            if span.token_index != expected_index:
                raise ValueError(
                    "token byte spans must be ordered and index-contiguous"
                )
            if span.token_id != token_id:
                raise ValueError("token byte span token_id does not match token_ids")
            if span.byte_start != cursor:
                raise ValueError(
                    "token byte spans must provide contiguous exact coverage"
                )
            cursor = span.byte_end

        if cursor != total_bytes:
            raise ValueError("token byte spans must cover sampled_text exactly")


@dataclass(frozen=True, slots=True)
class TargetSpan:
    """Decoded target plus its exact raw JSON and sampled-token identity."""

    target_text: str
    raw_json_value: str
    offsets: TextOffsets
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.target_text.strip():
            raise ValueError("target_text must be non-empty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("target token span must be non-empty and half-open")
        if len(self.token_ids) != self.token_end - self.token_start:
            raise ValueError("target token IDs do not match target token span")


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """A validated call while retaining the original sampled representation."""

    name: str
    target: str
    sampled_text: str
    sampled_token_ids: tuple[int, ...]
    sampled_token_byte_spans: tuple[TokenByteSpan, ...]
    raw_tool_call: str
    raw_json: str
    call_offsets: TextOffsets
    json_offsets: TextOffsets
    target_span: TargetSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampled_token_ids", tuple(self.sampled_token_ids))
        object.__setattr__(
            self,
            "sampled_token_byte_spans",
            tuple(self.sampled_token_byte_spans),
        )
        SampledAssistantTurn(
            self.sampled_text,
            self.sampled_token_ids,
            self.sampled_token_byte_spans,
        )
        if self.name != TGVF_FOCUS_TOOL_NAME:
            raise ValueError(f"unsupported tool name: {self.name!r}")
        validate_native_tool_target_for_echo(self.target)
        if self.target != self.target_span.target_text:
            raise ValueError("parsed target and target span disagree")


@dataclass(frozen=True, slots=True)
class ParsedImageZoomInCall:
    """A validated crop call retaining the exact sampled representation."""

    name: str
    bbox_2d: tuple[int, int, int, int]
    sampled_text: str
    sampled_token_ids: tuple[int, ...]
    sampled_token_byte_spans: tuple[TokenByteSpan, ...]
    raw_tool_call: str
    raw_json: str
    call_offsets: TextOffsets
    json_offsets: TextOffsets
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_2d", tuple(self.bbox_2d))
        object.__setattr__(self, "sampled_token_ids", tuple(self.sampled_token_ids))
        object.__setattr__(
            self,
            "sampled_token_byte_spans",
            tuple(self.sampled_token_byte_spans),
        )
        SampledAssistantTurn(
            self.sampled_text,
            self.sampled_token_ids,
            self.sampled_token_byte_spans,
        )
        if self.name != IMAGE_ZOOM_IN_TOOL_NAME:
            raise ValueError(f"unsupported crop tool name: {self.name!r}")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("crop label must be a string when provided")
        if len(self.bbox_2d) != 4 or any(
            type(value) is not int for value in self.bbox_2d
        ):
            raise ValueError("bbox_2d must contain exactly four integers")
        left, top, right, bottom = self.bbox_2d
        if right <= left or bottom <= top:
            raise ValueError("bbox_2d must have positive requested width and height")


@dataclass(frozen=True, slots=True)
class ParsedCropTGVFCall:
    """One atomic crop-and-foveate call with exact target-token identity."""

    name: str
    bbox_2d: tuple[int, int, int, int]
    target: str
    sampled_text: str
    sampled_token_ids: tuple[int, ...]
    sampled_token_byte_spans: tuple[TokenByteSpan, ...]
    raw_tool_call: str
    raw_json: str
    call_offsets: TextOffsets
    json_offsets: TextOffsets
    target_span: TargetSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_2d", tuple(self.bbox_2d))
        object.__setattr__(self, "sampled_token_ids", tuple(self.sampled_token_ids))
        object.__setattr__(
            self,
            "sampled_token_byte_spans",
            tuple(self.sampled_token_byte_spans),
        )
        SampledAssistantTurn(
            self.sampled_text,
            self.sampled_token_ids,
            self.sampled_token_byte_spans,
        )
        if self.name != TGVF_CROP_TOOL_NAME:
            raise ValueError(f"unsupported atomic crop/TGVF tool name: {self.name!r}")
        validate_native_tool_target_for_echo(self.target)
        if len(self.bbox_2d) != 4 or any(
            type(value) is not int for value in self.bbox_2d
        ):
            raise ValueError("bbox_2d must contain exactly four integers")
        left, top, right, bottom = self.bbox_2d
        if right <= left or bottom <= top:
            raise ValueError("bbox_2d must have positive requested width and height")
        if self.target != self.target_span.target_text:
            raise ValueError("parsed target and target span disagree")


NativeToolCall = ParsedToolCall | ParsedImageZoomInCall | ParsedCropTGVFCall


class TerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MALFORMED_ACTION = "malformed_action"
    TOOL_ERROR = "tool_error"
    TOOL_CALL_CAP = "tool_call_cap"
    TIMEOUT = "timeout"
