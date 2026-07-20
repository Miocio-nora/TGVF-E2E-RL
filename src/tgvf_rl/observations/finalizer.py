"""Pure assembly boundary for one completed rollout trajectory.

This module stores tensors that were already materialized during rollout.  It
does not render tokens, run a processor, execute a tool, or recompute any
visual/conditioning state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import torch

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.trajectories.behavior import BehaviorTraceStore
from tgvf_rl.trajectories.schema import TrajectoryRecord
from tgvf_rl.trajectories.validation import TrajectoryValidator

from .schema import TrajectorySourceVisual
from .store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)

if TYPE_CHECKING:
    from tgvf_rl.framework.verl.rollout_bridge import RolloutBridgeRecord


@dataclass(frozen=True, slots=True)
class MaterializedTrajectoryReplayTensors:
    """Raw final-sequence tensors captured by rollout before this boundary."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    base_attention_mask: torch.Tensor
    policy_attention_mask: torch.Tensor
    reference_attention_mask: torch.Tensor
    teacher_attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    original_image_key_block: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
    rope_delta: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_ids",
            "position_ids",
            "base_attention_mask",
            "policy_attention_mask",
            "reference_attention_mask",
            "teacher_attention_mask",
        ):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"materialized replay {name} must be a torch.Tensor")
        for name in (
            "token_type_ids",
            "original_image_key_block",
            "cache_position",
            "rope_delta",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"materialized replay {name} must be a torch.Tensor or None"
                )


@dataclass(frozen=True, slots=True)
class TrajectoryReplayFinalizationRequest:
    """All explicit inputs needed to freeze replay and mint its veRL bridge."""

    trajectory: TrajectoryRecord
    source_visual: TrajectorySourceVisual
    tensors: MaterializedTrajectoryReplayTensors
    replay_schema_version: str
    replay_id: str
    trajectory_id: str
    model: ModelIdentity
    behavior_policy: PolicyVersion
    crop_vision_replay_mode: str
    cache_mode: str
    cache_prefix_length: int
    deterministic_forward: bool
    adapter_dropout: float
    maximum_policy_staleness: int
    initial_prompt_token_ids: tuple[int, ...]
    native_tool_appended_token_ids: tuple[tuple[int, ...], ...]
    sentinel_fields: Mapping[str, object]
    extra_fields: Mapping[str, object] = field(default_factory=dict)
    reward_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryRecord):
            raise TypeError("finalization requires a completed TrajectoryRecord")
        if not isinstance(self.source_visual, TrajectorySourceVisual):
            raise TypeError("finalization requires a trajectory source visual")
        if not isinstance(self.tensors, MaterializedTrajectoryReplayTensors):
            raise TypeError("finalization requires materialized replay tensors")
        if not self.replay_schema_version or not self.replay_id or not self.trajectory_id:
            raise ValueError("all replay identities must be explicit and non-empty")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("finalization model identity must be explicit")
        if not isinstance(self.behavior_policy, PolicyVersion):
            raise TypeError("finalization behavior policy identity must be explicit")
        if type(self.maximum_policy_staleness) is not int or (
            self.maximum_policy_staleness < 0
        ):
            raise ValueError("maximum policy staleness must be a non-negative integer")
        if not isinstance(self.initial_prompt_token_ids, tuple) or not (
            self.initial_prompt_token_ids
        ):
            raise ValueError("exact initial prompt token IDs must be supplied")
        if not isinstance(self.native_tool_appended_token_ids, tuple) or any(
            not isinstance(row, tuple)
            for row in self.native_tool_appended_token_ids
        ):
            raise TypeError("native appended tool token IDs must be tuple rows")
        if not isinstance(self.sentinel_fields, Mapping):
            raise TypeError("objective sentinel fields must be an explicit mapping")
        if not isinstance(self.extra_fields, Mapping):
            raise TypeError("extra bridge fields must be a mapping")


def finalize_trajectory_replay(
    request: TrajectoryReplayFinalizationRequest,
    *,
    observation_store: ObservationStore,
    behavior_store: BehaviorTraceStore,
) -> RolloutBridgeRecord:
    """Freeze one exact replay and convert it without recomputation or inference."""

    if not isinstance(request, TrajectoryReplayFinalizationRequest):
        raise TypeError("request must be TrajectoryReplayFinalizationRequest")
    if not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if not isinstance(behavior_store, BehaviorTraceStore):
        raise TypeError("behavior_store must be a BehaviorTraceStore")

    trajectory = request.trajectory
    if request.trajectory_id != trajectory.identity.canonical_id:
        raise ValueError("explicit replay trajectory identity differs from trajectory")
    if request.model != trajectory.model:
        raise ValueError("explicit replay model identity differs from trajectory")
    if request.behavior_policy != trajectory.behavior_policy:
        raise ValueError("explicit replay policy identity differs from trajectory")

    validator = TrajectoryValidator(
        observation_store,
        behavior_store,
        maximum_policy_staleness=request.maximum_policy_staleness,
    )
    validator.validate(trajectory)

    refs = _store_materialized_tensors(
        observation_store,
        request.replay_id,
        request.tensors,
    )
    replay = TrajectoryReplayRecord(
        schema_version=request.replay_schema_version,
        replay_id=request.replay_id,
        trajectory_id=request.trajectory_id,
        model=request.model,
        behavior_policy=request.behavior_policy,
        source_visual=request.source_visual,
        observation_handles=tuple(
            observation.handle for observation in trajectory.observations
        ),
        tensors=refs,
        crop_vision_replay_mode=request.crop_vision_replay_mode,
        cache_mode=request.cache_mode,
        cache_prefix_length=request.cache_prefix_length,
        deterministic_forward=request.deterministic_forward,
        adapter_dropout=request.adapter_dropout,
    )
    replay_handle = observation_store.put_replay(replay)

    # Imported lazily so observations.store remains usable by the veRL bridge
    # without a package initialization cycle.
    from tgvf_rl.framework.verl.rollout_bridge import trajectory_to_rollout_bridge

    return trajectory_to_rollout_bridge(
        trajectory,
        validator=validator,
        initial_prompt_token_ids=request.initial_prompt_token_ids,
        native_tool_appended_token_ids=request.native_tool_appended_token_ids,
        replay_handle=replay_handle,
        sentinel_fields=request.sentinel_fields,
        extra_fields=request.extra_fields,
        reward_score=request.reward_score,
    )


def _store_materialized_tensors(
    store: ObservationStore,
    replay_id: str,
    tensors: MaterializedTrajectoryReplayTensors,
) -> TrajectoryReplayTensorRefs:
    prefix = f"trajectory-replay.{replay_id}"

    def put(name: str, tensor: torch.Tensor | None):
        if tensor is None:
            return None
        return store.put_tensor(f"{prefix}.{name}", tensor)

    return TrajectoryReplayTensorRefs(
        input_ids=put("input_ids", tensors.input_ids),
        position_ids=put("position_ids", tensors.position_ids),
        attention_mask=put("attention_mask", tensors.base_attention_mask),
        policy_attention_mask=put(
            "policy_attention_mask", tensors.policy_attention_mask
        ),
        reference_attention_mask=put(
            "reference_attention_mask", tensors.reference_attention_mask
        ),
        teacher_attention_mask=put(
            "teacher_attention_mask", tensors.teacher_attention_mask
        ),
        token_type_ids=put("token_type_ids", tensors.token_type_ids),
        original_image_key_block=put(
            "original_image_key_block", tensors.original_image_key_block
        ),
        cache_position=put("cache_position", tensors.cache_position),
        rope_delta=put("rope_delta", tensors.rope_delta),
    )


__all__ = [
    "MaterializedTrajectoryReplayTensors",
    "TrajectoryReplayFinalizationRequest",
    "finalize_trajectory_replay",
]
