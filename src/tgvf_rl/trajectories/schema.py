"""Immutable multi-turn trajectory schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import OwnedTokenSequence, TokenSpan
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.schema import TGVF_CROP_TOOL_NAME

from .behavior import BehaviorTraceHandle


class TrajectoryStop(str, Enum):
    FINAL_ANSWER = "final_answer"
    DIRECT_ANSWER = "direct_answer"
    MALFORMED_CALL = "malformed_call"
    TOOL_ERROR = "tool_error"
    CALL_CAP = "call_cap"
    TIMEOUT = "timeout"
    MAX_TOKENS = "max_tokens"
    INVALID_FORMAT = "invalid_format"


@dataclass(frozen=True, slots=True)
class TrajectoryIdentity:
    run_id: str
    sample_id: str
    rollout_index: int
    group_id: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.sample_id or not self.group_id:
            raise ValueError("trajectory identity fields must be non-empty")
        if self.rollout_index < 0:
            raise ValueError("rollout_index must be non-negative")

    @property
    def canonical_id(self) -> str:
        return f"{self.run_id}/{self.sample_id}/{self.rollout_index}/{self.group_id}"


@dataclass(frozen=True, slots=True)
class AssistantTurnRecord:
    turn_index: int
    raw_text: str
    tokens: OwnedTokenSequence
    behavior_trace: BehaviorTraceHandle
    think_span: TokenSpan | None
    is_tool_call: bool
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    call_index: int
    assistant_turn_index: int
    function_name: str
    target: str
    target_token_span: TokenSpan
    target_char_span: tuple[int, int]
    raw_call_text: str
    attempt_index: int | None = None

    def __post_init__(self) -> None:
        if self.function_name != "tgvf_focus_tool":
            raise ValueError("unexpected tool function")
        if not self.target.strip():
            raise ValueError("tool target must be non-empty")
        if self.call_index < 0 or self.assistant_turn_index < 0:
            raise ValueError("tool call indices must be non-negative")
        if self.attempt_index is None:
            object.__setattr__(self, "attempt_index", self.call_index)
        elif self.attempt_index < 0:
            raise ValueError("tool attempt index must be non-negative")


@dataclass(frozen=True, slots=True)
class CropToolCallRecord:
    call_index: int
    assistant_turn_index: int
    function_name: str
    bbox_2d: tuple[int, int, int, int]
    raw_call_text: str
    label: str | None = None
    attempt_index: int | None = None

    def __post_init__(self) -> None:
        if self.function_name != "image_zoom_in_tool":
            raise ValueError("unexpected crop tool function")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("crop label must be a string when provided")
        if self.call_index < 0 or self.assistant_turn_index < 0:
            raise ValueError("crop call indices must be non-negative")
        if self.attempt_index is None:
            object.__setattr__(self, "attempt_index", self.call_index)
        elif self.attempt_index < 0:
            raise ValueError("crop attempt index must be non-negative")
        if len(self.bbox_2d) != 4 or any(
            type(value) is not int for value in self.bbox_2d
        ):
            raise ValueError("crop bbox must contain exactly four integers")
        left, top, right, bottom = self.bbox_2d
        if right <= left or bottom <= top:
            raise ValueError("crop bbox must be non-empty")


@dataclass(frozen=True, slots=True)
class CropTGVFToolCallRecord:
    """One atomic sampled call carrying both crop box and TGVF target."""

    call_index: int
    assistant_turn_index: int
    function_name: str
    bbox_2d: tuple[int, int, int, int]
    target: str
    target_token_span: TokenSpan
    target_char_span: tuple[int, int]
    raw_call_text: str
    attempt_index: int | None = None

    def __post_init__(self) -> None:
        if self.function_name != TGVF_CROP_TOOL_NAME:
            raise ValueError("unexpected atomic crop+TGVF tool function")
        if self.call_index < 0 or self.assistant_turn_index < 0:
            raise ValueError("atomic crop+TGVF call indices must be non-negative")
        if self.attempt_index is None:
            object.__setattr__(self, "attempt_index", self.call_index)
        elif self.attempt_index < 0:
            raise ValueError("atomic crop+TGVF attempt index must be non-negative")
        if len(self.bbox_2d) != 4 or any(
            type(value) is not int for value in self.bbox_2d
        ):
            raise ValueError("atomic crop+TGVF bbox must contain four integers")
        left, top, right, bottom = self.bbox_2d
        if right <= left or bottom <= top:
            raise ValueError("atomic crop+TGVF bbox must be non-empty")
        if not self.target.strip():
            raise ValueError("atomic crop+TGVF target must be non-empty")
        if (
            len(self.target_char_span) != 2
            or self.target_char_span[0] < 0
            or self.target_char_span[1] <= self.target_char_span[0]
        ):
            raise ValueError("atomic crop+TGVF target char span must be non-empty")


NativeToolCallRecord = ToolCallRecord | CropToolCallRecord | CropTGVFToolCallRecord


@dataclass(frozen=True, slots=True)
class ToolObservationRecord:
    call_index: int
    handle: ObservationHandle
    template_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ToolErrorRecord:
    """One environment-owned error response; never a visual observation."""

    attempt_index: int
    assistant_turn_index: int
    code: str
    payload_json: str
    payload_sha256: str
    template_token_ids: tuple[int, ...]
    recoverable: bool
    function_name: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_index < 0 or self.assistant_turn_index < 0:
            raise ValueError("tool error indices must be non-negative")
        if not self.code or not self.payload_json or not self.template_token_ids:
            raise ValueError("tool error code/payload/environment tokens are required")
        actual = hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()
        if actual != self.payload_sha256:
            raise ValueError("tool error payload checksum mismatch")


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    source: str
    text: str
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    schema_version: str
    identity: TrajectoryIdentity
    model: ModelIdentity
    behavior_policy: PolicyVersion
    assistant_turns: tuple[AssistantTurnRecord, ...]
    tool_calls: tuple[NativeToolCallRecord, ...]
    observations: tuple[ToolObservationRecord, ...]
    final_answer: str | None
    stop: TrajectoryStop
    tool_errors: tuple[ToolErrorRecord, ...] = ()
    rewards: tuple[tuple[str, float], ...] = ()
    feedback: tuple[FeedbackEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    trajectories: tuple[TrajectoryRecord, ...]


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def trajectory_checksum(trajectory: TrajectoryRecord) -> str:
    """Canonical identity of the complete immutable trajectory payload."""

    payload = json.dumps(
        _canonical(asdict(trajectory)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
