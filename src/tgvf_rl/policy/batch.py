"""Project-owned n=8 group batching for Policy Pilot v1.

Pinned veRL GRPO groups rows by ``non_tensor_batch["uid"]`` and consumes the
sum of ``rm_scores`` as each outcome reward. This module materializes those
public fields without inheriting veRL's reward-placement or padding defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from tgvf_rl.framework.verl.data_bridge import (
    DataProtoPayload,
    build_padded_data_proto_payload,
)
from tgvf_rl.framework.verl.rollout_bridge import RolloutBridgeRecord


POLICY_PILOT_V1_GROUP_SIZE = 8
POLICY_PILOT_V1_GROUP_BATCH_SCHEMA = "policy-pilot-v1-group-batch-v1"
VERL_GRPO_GROUP_UID_FIELD = "uid"
PILOT_EXACT_GROUP_UID_FIELD = "tgvf_exact_group_uid"
PILOT_EXACT_REWARD_FIELD = "tgvf_exact_trajectory_reward"
PILOT_GROUP_BATCH_SCHEMA_FIELD = "tgvf_group_batch_schema_version"
_GROUP_FIELDS = frozenset(
    {
        VERL_GRPO_GROUP_UID_FIELD,
        PILOT_EXACT_GROUP_UID_FIELD,
        PILOT_EXACT_REWARD_FIELD,
        PILOT_GROUP_BATCH_SCHEMA_FIELD,
    }
)


@dataclass(frozen=True, slots=True)
class PilotGroupedRollout:
    """One completed rollout plus the explicit GRPO group identity."""

    group_uid: str
    rollout: RolloutBridgeRecord

    def __post_init__(self) -> None:
        if not isinstance(self.group_uid, str) or not self.group_uid.strip():
            raise ValueError("group_uid must be a non-empty string")
        if not isinstance(self.rollout, RolloutBridgeRecord):
            raise TypeError("rollout must be a RolloutBridgeRecord")
        trajectory_group = self.rollout.trajectory_payload.identity.group_id
        if trajectory_group != self.group_uid:
            raise ValueError(
                "explicit group_uid differs from the trajectory group identity"
            )


def materialize_policy_pilot_group_batch(
    grouped_rollouts: Iterable[PilotGroupedRollout],
    *,
    pad_token_id: int,
) -> DataProtoPayload:
    """Materialize complete n=8 groups without filtering or regenerating rows."""

    entries = tuple(grouped_rollouts)
    if not entries:
        raise ValueError("at least one completed Pilot group is required")
    if any(not isinstance(entry, PilotGroupedRollout) for entry in entries):
        raise TypeError("grouped_rollouts must contain PilotGroupedRollout values")

    grouped: dict[str, list[RolloutBridgeRecord]] = {}
    seen_trajectories: set[str] = set()
    rewards: list[float] = []
    for entry in entries:
        record = entry.rollout
        trajectory_id = record.trajectory_id
        if trajectory_id in seen_trajectories:
            raise ValueError("Pilot group batch contains a duplicate trajectory")
        seen_trajectories.add(trajectory_id)
        reward = record.reward_score
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ValueError(
                "every Pilot trajectory requires a finite reward_score before batching"
            )
        rewards.append(float(reward))
        grouped.setdefault(entry.group_uid, []).append(record)

    for group_uid, records in grouped.items():
        if len(records) != POLICY_PILOT_V1_GROUP_SIZE:
            raise ValueError(
                f"Pilot group {group_uid!r} must contain exactly "
                f"{POLICY_PILOT_V1_GROUP_SIZE} trajectories"
            )
        expected_prompt_ids = records[0].prompt_ids
        if any(record.prompt_ids != expected_prompt_ids for record in records[1:]):
            raise ValueError(
                f"Pilot group {group_uid!r} contains different exact prompt IDs"
            )

    records = tuple(entry.rollout for entry in entries)
    padded = build_padded_data_proto_payload(records, pad_token_id=pad_token_id)
    tensor_batch = dict(padded.tensor_batch)
    if "rm_scores" in tensor_batch:
        raise ValueError("padded rollout payload already contains rm_scores")
    non_tensor_batch = dict(padded.non_tensor_batch)
    collisions = _GROUP_FIELDS & set(non_tensor_batch)
    if collisions:
        raise ValueError(
            "rollout sidecars collide with Pilot group fields: "
            f"{sorted(collisions)}"
        )

    response_mask = tensor_batch["response_mask"]
    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    for row_index, reward in enumerate(rewards):
        policy_positions = torch.nonzero(
            response_mask[row_index].to(dtype=torch.bool), as_tuple=False
        ).flatten()
        if policy_positions.numel() == 0:
            raise ValueError("Pilot trajectory has no policy-owned response token")
        rm_scores[row_index, int(policy_positions[-1].item())] = reward
    tensor_batch["rm_scores"] = rm_scores

    group_uids = tuple(entry.group_uid for entry in entries)
    non_tensor_batch.update(
        {
            # This exact public name is consumed by pinned veRL GRPO.
            VERL_GRPO_GROUP_UID_FIELD: group_uids,
            # Project-owned duplicates make the public grouping field auditable.
            PILOT_EXACT_GROUP_UID_FIELD: group_uids,
            PILOT_EXACT_REWARD_FIELD: tuple(rewards),
            PILOT_GROUP_BATCH_SCHEMA_FIELD: tuple(
                POLICY_PILOT_V1_GROUP_BATCH_SCHEMA for _ in entries
            ),
        }
    )
    meta_info = dict(padded.meta_info)
    meta_info["tgvf_group_batch_schema_version"] = (
        POLICY_PILOT_V1_GROUP_BATCH_SCHEMA
    )
    return DataProtoPayload(
        tensor_batch=tensor_batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )


__all__ = [
    "PILOT_EXACT_GROUP_UID_FIELD",
    "PILOT_EXACT_REWARD_FIELD",
    "PILOT_GROUP_BATCH_SCHEMA_FIELD",
    "POLICY_PILOT_V1_GROUP_BATCH_SCHEMA",
    "POLICY_PILOT_V1_GROUP_SIZE",
    "VERL_GRPO_GROUP_UID_FIELD",
    "PilotGroupedRollout",
    "materialize_policy_pilot_group_batch",
]
