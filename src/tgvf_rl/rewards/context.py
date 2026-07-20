"""Lossless conversion from an immutable trajectory to reward inputs."""

from __future__ import annotations

from tgvf_rl.protocol import ToolErrorCode
from tgvf_rl.trajectories.schema import (
    ToolCallRecord,
    TrajectoryRecord,
    TrajectoryStop,
)

from .schema import AnswerTaskKind, RewardContext


_PROTOCOL_INVALIDATING_ERROR_PREFIXES = ("tool_parse.",)
_PROTOCOL_INVALIDATING_ERROR_CODES = frozenset(
    {
        ToolErrorCode.TOOL_NOT_ENABLED.value,
        ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
    }
)


def reward_context_from_trajectory(
    trajectory: TrajectoryRecord,
    *,
    question: str,
    expected_answer: str,
    task_kind: AnswerTaskKind,
) -> RewardContext:
    """Retain invalid/error trajectories while deriving their reward facts."""

    if not isinstance(trajectory, TrajectoryRecord):
        raise TypeError("trajectory must be TrajectoryRecord")
    has_final_answer = (
        trajectory.stop in {TrajectoryStop.DIRECT_ANSWER, TrajectoryStop.FINAL_ANSWER}
        and trajectory.final_answer is not None
        and bool(trajectory.final_answer.strip())
    )
    invalidating_error = any(
        error.code in _PROTOCOL_INVALIDATING_ERROR_CODES
        or error.code.startswith(_PROTOCOL_INVALIDATING_ERROR_PREFIXES)
        for error in trajectory.tool_errors
    )
    protocol_valid = has_final_answer and not invalidating_error
    successful_tgvf = sum(
        isinstance(call, ToolCallRecord) for call in trajectory.tool_calls
    )
    return RewardContext(
        sample_id=trajectory.identity.sample_id,
        question=question,
        candidate_answer=trajectory.final_answer or "",
        expected_answer=expected_answer,
        tool_call_count=len(trajectory.tool_calls) + len(trajectory.tool_errors),
        task_kind=task_kind,
        protocol_valid=protocol_valid,
        has_valid_final_answer=has_final_answer,
        successful_tgvf_observation_count=successful_tgvf,
        tool_error_codes=tuple(error.code for error in trajectory.tool_errors),
    )


__all__ = ["reward_context_from_trajectory"]
