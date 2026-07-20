"""Fail-closed invariants for actual-logprob multi-call trajectories."""

from __future__ import annotations

import hashlib

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.tokens import TokenOwnership
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.observations.schema import CropObservationRecord, FocusedObservationRecord

from .behavior import BehaviorTraceStore
from .schema import (
    CropToolCallRecord,
    TrajectoryBatch,
    TrajectoryRecord,
    TrajectoryStop,
)


class TrajectoryValidator:
    def __init__(
        self,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        *,
        maximum_policy_staleness: int = 0,
    ) -> None:
        if maximum_policy_staleness < 0:
            raise ValueError("maximum_policy_staleness must be non-negative")
        self.store = observation_store
        self.behavior_store = behavior_store
        self.maximum_policy_staleness = maximum_policy_staleness

    def validate(self, trajectory: TrajectoryRecord) -> None:
        if tuple(turn.turn_index for turn in trajectory.assistant_turns) != tuple(
            range(len(trajectory.assistant_turns))
        ):
            raise ReplayMismatchError("assistant turns must have contiguous indices")
        call_indices = tuple(call.call_index for call in trajectory.tool_calls)
        observation_indices = tuple(item.call_index for item in trajectory.observations)
        if call_indices != tuple(range(len(call_indices))):
            raise ReplayMismatchError("tool calls must have contiguous call indices")
        if observation_indices != call_indices:
            raise ReplayMismatchError(
                "every valid tool call must have one ordered observation"
            )
        representation = None
        for item, call in zip(
            trajectory.observations, trajectory.tool_calls, strict=True
        ):
            record = self.store.resolve_record(item.handle)
            if record.call_index != item.call_index:
                raise ReplayMismatchError(
                    "observation call index differs from trajectory"
                )
            if record.model != trajectory.model:
                raise IdentityMismatchError(
                    "observation model differs from trajectory model"
                )
            if isinstance(call, CropToolCallRecord):
                if not isinstance(record, CropObservationRecord):
                    raise IdentityMismatchError(
                        "crop call received a non-crop observation"
                    )
                if record.policy_version != trajectory.behavior_policy:
                    raise IdentityMismatchError(
                        "crop materialization policy differs from behavior policy"
                    )
                if record.trajectory_id != trajectory.identity.canonical_id:
                    raise IdentityMismatchError(
                        "crop observation differs from trajectory identity"
                    )
                if record.requested_bbox_2d != call.bbox_2d:
                    raise IdentityMismatchError(
                        "crop observation bbox differs from sampled tool call"
                    )
            else:
                if not isinstance(record, FocusedObservationRecord):
                    raise IdentityMismatchError(
                        "TGVF call received a non-TGVF observation"
                    )
                if record.condition.policy_version != trajectory.behavior_policy:
                    raise IdentityMismatchError(
                        "observation materialization policy differs from behavior policy"
                    )
                if record.condition.trajectory_ids != (
                    trajectory.identity.canonical_id,
                ):
                    raise IdentityMismatchError(
                        "observation provenance differs from trajectory identity"
                    )
                if record.condition.call_indices != (call.call_index,):
                    raise IdentityMismatchError(
                        "observation provenance differs from trajectory call"
                    )
                expected_target_sha = hashlib.sha256(
                    call.target.encode("utf-8")
                ).hexdigest()
                if record.condition.sampled_target_text_sha256 != expected_target_sha:
                    raise IdentityMismatchError(
                        "observation target differs from sampled tool target"
                    )
                if (
                    record.condition.sampled_target_token_start,
                    record.condition.sampled_target_token_end,
                ) != (call.target_token_span.start, call.target_token_span.end):
                    raise IdentityMismatchError(
                        "observation target token span differs from sampled tool call"
                    )
                if representation is None:
                    representation = record.representation
                elif record.representation != representation:
                    raise IdentityMismatchError(
                        "representation artifact changed within one trajectory"
                    )
            if call.assistant_turn_index >= len(trajectory.assistant_turns):
                raise ReplayMismatchError(
                    "tool call references a missing assistant turn"
                )
            assistant_turn = trajectory.assistant_turns[call.assistant_turn_index]
            if not assistant_turn.is_tool_call:
                raise ReplayMismatchError(
                    "tool call references a non-tool assistant turn"
                )
            if call.assistant_turn_index != call.call_index:
                raise ReplayMismatchError(
                    "each accepted action turn must contain exactly one ordered tool call"
                )

        for turn in trajectory.assistant_turns:
            trace = self.behavior_store.resolve(turn.behavior_trace)
            if trace.trajectory_id != trajectory.identity.canonical_id:
                raise IdentityMismatchError(
                    "behavior trace differs from trajectory identity"
                )
            if trace.assistant_turn_index != turn.turn_index:
                raise IdentityMismatchError(
                    "behavior trace differs from assistant turn index"
                )
            if trace.tokens != turn.tokens:
                raise ReplayMismatchError(
                    "assistant tokens differ from content-addressed behavior trace"
                )
            if trace.behavior_policy != trajectory.behavior_policy:
                raise IdentityMismatchError(
                    "behavior trace policy differs from trajectory"
                )
            policy_indices = turn.tokens.policy_indices
            if policy_indices != trace.behavior.sampled_token_indices:
                raise ReplayMismatchError(
                    "policy ownership mask and behavior logprobs differ"
                )
            sampled_ids = tuple(
                turn.tokens.token_ids[index] for index in policy_indices
            )
            if sampled_ids != trace.behavior.sampled_token_ids:
                raise ReplayMismatchError(
                    "sampled token IDs differ from behavior record"
                )
            sampling = trace.behavior.sampling
            if sampling.policy_version != trajectory.behavior_policy:
                raise IdentityMismatchError(
                    "turn behavior policy differs from trajectory"
                )
            if sampling.asynchronous_staleness_steps > self.maximum_policy_staleness:
                raise ReplayMismatchError("rollout staleness exceeds accepted bound")
            if turn.think_span is not None:
                span = turn.think_span
                if span.end > len(turn.tokens.token_ids):
                    raise ReplayMismatchError("think span lies outside assistant turn")
            if not all(
                isinstance(owner, TokenOwnership) for owner in turn.tokens.ownership
            ):
                raise ReplayMismatchError("unknown token ownership value")

        if trajectory.stop in {
            TrajectoryStop.MALFORMED_CALL,
            TrajectoryStop.TOOL_ERROR,
            TrajectoryStop.CALL_CAP,
        }:
            if (
                not trajectory.assistant_turns
                or not trajectory.assistant_turns[-1].is_tool_call
            ):
                raise ReplayMismatchError(
                    "tool-terminal trajectory must end in a tool-call assistant turn"
                )
            if len(trajectory.assistant_turns) != len(trajectory.tool_calls) + 1:
                raise ReplayMismatchError(
                    "failed/capped tool attempt must not fabricate a call or observation"
                )

    def validate_batch(self, batch: TrajectoryBatch) -> None:
        seen: set[tuple[str, int]] = set()
        for trajectory in batch.trajectories:
            key = (trajectory.identity.sample_id, trajectory.identity.rollout_index)
            if key in seen:
                raise IdentityMismatchError("duplicate trajectory batch identity")
            seen.add(key)
            self.validate(trajectory)


class TrajectoryBatcher:
    def __init__(self, validator: TrajectoryValidator) -> None:
        self.validator = validator

    def batch(self, trajectories: tuple[TrajectoryRecord, ...]) -> TrajectoryBatch:
        batch = TrajectoryBatch(trajectories)
        self.validator.validate_batch(batch)
        return batch
