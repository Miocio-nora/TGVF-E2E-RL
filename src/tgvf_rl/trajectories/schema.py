"""Immutable multi-turn trajectory schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import OwnedTokenSequence, TokenSpan
from tgvf_rl.observations.store import ObservationHandle

from .behavior import BehaviorTraceHandle


class TrajectoryStop(str, Enum):
    FINAL_ANSWER = "final_answer"
    DIRECT_ANSWER = "direct_answer"
    MALFORMED_CALL = "malformed_call"
    TOOL_ERROR = "tool_error"
    CALL_CAP = "call_cap"
    TIMEOUT = "timeout"
    MAX_TOKENS = "max_tokens"


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


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    call_index: int
    assistant_turn_index: int
    function_name: str
    target: str
    target_token_span: TokenSpan
    target_char_span: tuple[int, int]
    raw_call_text: str

    def __post_init__(self) -> None:
        if self.function_name != "tgvf_focus_tool":
            raise ValueError("unexpected tool function")
        if not self.target.strip():
            raise ValueError("tool target must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolObservationRecord:
    call_index: int
    handle: ObservationHandle
    template_token_ids: tuple[int, ...]


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
    tool_calls: tuple[ToolCallRecord, ...]
    observations: tuple[ToolObservationRecord, ...]
    final_answer: str | None
    stop: TrajectoryStop
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
