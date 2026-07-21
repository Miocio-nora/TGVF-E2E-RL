"""Policy Pilot reward and exact-GRPO fields on veRL's public DataProto path."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

import torch

from tgvf_rl.objectives import (
    ReferenceKLEstimator,
    compute_group_advantages,
    policy_pilot_v1_grpo_spec,
)
from tgvf_rl.rewards.schema import RewardResult
from tgvf_rl.rewards.verl_adapter import (
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
    PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD,
    PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD,
    PilotVerlTrajectoryReward,
    PilotVerlTrajectoryRewardScorer,
)
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .rollout_bridge import (
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    RolloutBridgeRecord,
    build_agent_loop_output,
)


POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA = "tgvf-pilot-verl-reward-batch-v1"
POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD = "tgvf_reward_batch_schema_version"


class RewardedTrajectoryFinalizerPort(Protocol):
    def finalize(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        reward: RewardResult,
    ) -> RolloutBridgeRecord: ...


class VerlRewardedAgentLoopOutputBuilder:
    """Score, finalize, and emit one public AgentLoopOutput with reward_score."""

    def __init__(
        self,
        *,
        request: object,
        scorer: PilotVerlTrajectoryRewardScorer,
        finalizer: RewardedTrajectoryFinalizerPort,
        metrics_factory: Callable[
            [TrajectoryRecord, PilotVerlTrajectoryReward], object
        ],
        agent_loop_output_cls: type[Any] | None = None,
    ) -> None:
        if not hasattr(request, "identity"):
            raise TypeError("rewarded output request must expose identity")
        if not isinstance(scorer, PilotVerlTrajectoryRewardScorer):
            raise TypeError("scorer must be PilotVerlTrajectoryRewardScorer")
        if not callable(getattr(finalizer, "finalize", None)):
            raise TypeError("finalizer must implement finalize()")
        if not callable(metrics_factory):
            raise TypeError("metrics_factory must be callable")
        self.request = request
        self.scorer = scorer
        self.finalizer = finalizer
        self.metrics_factory = metrics_factory
        self.agent_loop_output_cls = agent_loop_output_cls

    def __call__(self, trajectory: TrajectoryRecord) -> object:
        scored = self.scorer.score(request=self.request, trajectory=trajectory)
        record = self.finalizer.finalize(
            request=self.request,
            trajectory=trajectory,
            reward=scored.result,
        )
        if not isinstance(record, RolloutBridgeRecord):
            raise TypeError("rewarded finalizer must return RolloutBridgeRecord")
        if record.trajectory_payload != trajectory:
            raise ValueError("rewarded finalizer changed the trajectory")
        if record.trajectory_id != scored.trajectory_id:
            raise ValueError("rewarded finalizer changed the trajectory identity")
        if record.reward_score is None or not math.isclose(
            float(record.reward_score),
            scored.total,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("rewarded finalizer changed or omitted reward_score")
        output = build_agent_loop_output(
            record,
            metrics=self.metrics_factory(trajectory, scored),
            agent_loop_output_cls=self.agent_loop_output_cls,
        )
        extras = getattr(output, "extra_fields", None)
        if type(extras) is not dict:
            raise TypeError("public AgentLoopOutput.extra_fields must be a dict")
        sidecars = _agent_loop_reward_sidecars(scored)
        collisions = (set(sidecars) | {"reward_extra_info"}) & set(extras)
        if collisions:
            raise ValueError(
                "rollout extra fields collide with exact reward fields: "
                f"{sorted(collisions)!r}"
            )
        extras.update(sidecars)
        extras["reward_extra_info"] = scored.reward_extra_info()
        if getattr(output, "reward_score", None) != scored.total:
            raise RuntimeError("public AgentLoopOutput lost exact reward_score")
        return output


@dataclass(frozen=True, slots=True)
class VerlPilotRewardBatchView:
    """Validated row identities recovered from an upstream DataProto."""

    trajectory_ids: tuple[str, ...]
    group_uids: tuple[str, ...]
    upstream_group_uids: tuple[object, ...]
    rewards: tuple[float, ...]
    pipeline_sha256: str


def validate_policy_pilot_reward_data_proto(
    data: object,
) -> VerlPilotRewardBatchView:
    """Prove upstream used exact trajectory rewards and retained every n=8 row."""

    batch, non_tensors = _data_parts(data)
    required_tensors = {
        name: _required(batch, name, "DataProto.batch")
        for name in (
            "prompts",
            "responses",
            "response_mask",
            "attention_mask",
            "rm_scores",
        )
    }
    if any(not isinstance(value, torch.Tensor) for value in required_tensors.values()):
        raise TypeError("Policy Pilot reward DataProto fields must be tensors")
    responses = required_tensors["responses"]
    response_mask = required_tensors["response_mask"]
    rm_scores = required_tensors["rm_scores"]
    prompts = required_tensors["prompts"]
    attention_mask = required_tensors["attention_mask"]
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must share [batch,response]")
    if rm_scores.shape != responses.shape or not rm_scores.dtype.is_floating_point:
        raise ValueError("rm_scores must be floating [batch,response]")
    if prompts.ndim != 2 or prompts.shape[0] != responses.shape[0]:
        raise ValueError("prompts must share the reward batch dimension")
    if attention_mask.shape != (
        responses.shape[0],
        prompts.shape[1] + responses.shape[1],
    ):
        raise ValueError("attention_mask differs from prompt/response layout")
    if not bool(torch.isfinite(rm_scores).all().item()):
        raise ValueError("rm_scores must be finite")
    if not bool(((response_mask == 0) | (response_mask == 1)).all().item()):
        raise ValueError("response_mask must remain binary")
    if not bool(response_mask.bool().any(dim=-1).all().item()):
        raise ValueError("every retained trajectory requires policy-owned tokens")

    batch_size = responses.shape[0]
    if batch_size == 0 or batch_size % 8:
        raise ValueError("Policy Pilot reward batch must contain complete n=8 groups")
    exact_group_field, exact_reward_field, group_schema_field, group_schema = (
        _policy_batch_fields()
    )
    fields = {
        name: _row_values(
            _required(non_tensors, name, "DataProto.non_tensor_batch"),
            batch_size,
            name,
        )
        for name in (
            "uid",
            exact_group_field,
            exact_reward_field,
            group_schema_field,
            POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD,
            PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
            PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD,
            PILOT_VERL_REWARD_COMPONENTS_FIELD,
            PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD,
            TRAJECTORY_ID_FIELD,
            TRAJECTORY_PAYLOAD_FIELD,
        )
    }
    if any(value != group_schema for value in fields[group_schema_field]):
        raise ValueError("Policy Pilot group-batch schema was changed")
    if any(
        value != POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA
        for value in fields[POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD]
    ):
        raise ValueError("Policy Pilot reward-batch schema was changed")
    if any(
        value != PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
        for value in fields[PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD]
    ):
        raise ValueError("trajectory reward bridge schema was changed")

    trajectory_ids: list[str] = []
    group_uids: list[str] = []
    upstream_uids: list[object] = []
    rewards: list[float] = []
    pipeline_shas: list[str] = []
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    upstream_to_exact: dict[object, set[str]] = defaultdict(set)
    exact_to_upstream: dict[str, set[object]] = defaultdict(set)
    for row_index in range(batch_size):
        trajectory = fields[TRAJECTORY_PAYLOAD_FIELD][row_index]
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("reward DataProto trajectory sidecar is invalid")
        trajectory_id = fields[TRAJECTORY_ID_FIELD][row_index]
        reward_trajectory_id = fields[PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD][row_index]
        if (
            not isinstance(trajectory_id, str)
            or trajectory_id != trajectory.identity.canonical_id
            or reward_trajectory_id != trajectory_id
        ):
            raise ValueError("reward row trajectory identities differ")
        exact_group = fields[exact_group_field][row_index]
        if (
            not isinstance(exact_group, str)
            or not exact_group
            or exact_group != trajectory.identity.group_id
        ):
            raise ValueError("reward row exact group identity differs")
        upstream_uid = _stable_uid(fields["uid"][row_index])
        exact_reward = fields[exact_reward_field][row_index]
        if (
            isinstance(exact_reward, bool)
            or not isinstance(exact_reward, (int, float))
            or not math.isfinite(float(exact_reward))
        ):
            raise ValueError("exact trajectory reward sidecar is invalid")
        pipeline_sha = fields[PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD][row_index]
        _require_sha256(pipeline_sha, "reward pipeline")
        _validate_component_sidecar(
            fields[PILOT_VERL_REWARD_COMPONENTS_FIELD][row_index],
            expected_total=float(exact_reward),
        )
        trajectory_ids.append(trajectory_id)
        group_uids.append(exact_group)
        upstream_uids.append(upstream_uid)
        rewards.append(float(exact_reward))
        pipeline_shas.append(pipeline_sha)
        grouped_rows[exact_group].append(row_index)
        upstream_to_exact[upstream_uid].add(exact_group)
        exact_to_upstream[exact_group].add(upstream_uid)

    if len(set(trajectory_ids)) != batch_size:
        raise ValueError("reward batch contains duplicate trajectories")
    for group_uid, row_indices in grouped_rows.items():
        if len(row_indices) != 8:
            raise ValueError(f"reward group {group_uid!r} is not complete n=8")
        group_trajectories = tuple(
            fields[TRAJECTORY_PAYLOAD_FIELD][index] for index in row_indices
        )
        if len({item.identity.sample_id for item in group_trajectories}) != 1:
            raise ValueError("one reward group must contain one exact sample")
        first_prompt = prompts[row_indices[0]]
        if any(
            not torch.equal(prompts[index], first_prompt) for index in row_indices[1:]
        ):
            raise ValueError("one reward group must contain one exact prompt")
        rollout_indices = {
            fields[TRAJECTORY_PAYLOAD_FIELD][index].identity.rollout_index
            for index in row_indices
        }
        if rollout_indices != set(range(8)):
            raise ValueError("reward group rollout indices must be exactly 0..7")
    if any(len(groups) != 1 for groups in upstream_to_exact.values()) or any(
        len(groups) != 1 for groups in exact_to_upstream.values()
    ):
        raise ValueError("upstream uid and exact group_uid are not one-to-one")
    if len(set(pipeline_shas)) != 1:
        raise ValueError("one reward batch cannot mix pipeline identities")

    expected_rewards = torch.tensor(
        rewards,
        dtype=rm_scores.dtype,
        device=rm_scores.device,
    )
    if not torch.equal(rm_scores.sum(dim=-1), expected_rewards):
        raise ValueError("upstream rm_scores differ from exact trajectory rewards")
    response_attention = attention_mask[:, prompts.shape[1] :]
    response_lengths = response_attention.to(dtype=torch.int64).sum(dim=-1)
    if bool(((response_lengths <= 0) | (response_lengths > responses.shape[1])).any()):
        raise ValueError("reward rows have invalid response lengths")
    expected_rm_scores = torch.zeros_like(rm_scores)
    expected_rm_scores[
        torch.arange(batch_size, device=rm_scores.device), response_lengths - 1
    ] = expected_rewards
    if not torch.equal(rm_scores, expected_rm_scores):
        raise ValueError("rm_scores placement differs from pinned AgentLoopOutput")
    for name in ("token_level_scores", "token_level_rewards"):
        if name in batch and not torch.equal(batch[name], rm_scores):
            raise ValueError(f"{name} differs from exact zero-KL rm_scores")

    return VerlPilotRewardBatchView(
        trajectory_ids=tuple(trajectory_ids),
        group_uids=tuple(group_uids),
        upstream_group_uids=tuple(upstream_uids),
        rewards=tuple(rewards),
        pipeline_sha256=pipeline_shas[0],
    )


def bind_policy_pilot_exact_grpo_fields(
    data: object,
    *,
    diagnostic_kl_estimator: ReferenceKLEstimator,
) -> VerlPilotRewardBatchView:
    """Attach repo-owned scores/rewards/advantages consumed by the exact loss."""

    view = validate_policy_pilot_reward_data_proto(data)
    batch, _ = _data_parts(data)
    rm_scores = batch["rm_scores"]
    response_mask = batch["response_mask"].to(dtype=torch.bool)
    rewards = torch.tensor(
        view.rewards,
        dtype=rm_scores.dtype,
        device=rm_scores.device,
    )
    group_ids = _integer_group_ids(view.group_uids, device=rm_scores.device)
    spec = policy_pilot_v1_grpo_spec(diagnostic_kl_estimator=diagnostic_kl_estimator)
    sequence_advantages = compute_group_advantages(rewards, group_ids, spec)
    advantages = sequence_advantages[:, None] * response_mask
    for name, expected in (
        ("token_level_scores", rm_scores),
        ("token_level_rewards", rm_scores),
        ("advantages", advantages),
        ("returns", advantages),
    ):
        _set_or_validate_tensor(batch, name, expected)
    return validate_policy_pilot_reward_data_proto(data)


def _agent_loop_reward_sidecars(
    scored: PilotVerlTrajectoryReward,
) -> dict[str, object]:
    exact_group_field, exact_reward_field, group_schema_field, group_schema = (
        _policy_batch_fields()
    )
    fields = scored.reward_sidecars()
    fields.update(
        {
            exact_group_field: scored.group_uid,
            exact_reward_field: scored.total,
            group_schema_field: group_schema,
            POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD: (
                POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA
            ),
        }
    )
    return fields


def _policy_batch_fields() -> tuple[str, str, str, str]:
    # Lazy import avoids a framework-package/policy.batch initialization cycle.
    from tgvf_rl.policy.batch import (
        PILOT_EXACT_GROUP_UID_FIELD,
        PILOT_EXACT_REWARD_FIELD,
        PILOT_GROUP_BATCH_SCHEMA_FIELD,
        POLICY_PILOT_V1_GROUP_BATCH_SCHEMA,
    )

    return (
        PILOT_EXACT_GROUP_UID_FIELD,
        PILOT_EXACT_REWARD_FIELD,
        PILOT_GROUP_BATCH_SCHEMA_FIELD,
        POLICY_PILOT_V1_GROUP_BATCH_SCHEMA,
    )


def _data_parts(data: object) -> tuple[Any, Mapping[str, Any]]:
    batch = getattr(data, "batch", None)
    non_tensors = getattr(data, "non_tensor_batch", None)
    if (
        batch is None
        or not hasattr(batch, "__getitem__")
        or not hasattr(batch, "__contains__")
    ):
        raise TypeError("reward bridge requires mapping-like DataProto.batch")
    if not isinstance(non_tensors, Mapping):
        raise TypeError("reward bridge requires DataProto.non_tensor_batch")
    return batch, non_tensors


def _required(mapping: Any, name: str, owner: str) -> Any:
    if name not in mapping:
        raise ValueError(f"{owner} is missing required field {name!r}")
    return mapping[name]


def _row_values(value: object, size: int, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) and not (
        hasattr(value, "__len__") and hasattr(value, "__getitem__")
    ):
        raise TypeError(f"non-tensor reward field {name!r} must be indexable")
    if len(value) != size:  # type: ignore[arg-type]
        raise ValueError(f"non-tensor reward field {name!r} has wrong row count")
    return tuple(value[index] for index in range(size))  # type: ignore[index]


def _stable_uid(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not value:
        raise ValueError("upstream GRPO uid must be a stable non-empty scalar")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _validate_component_sidecar(value: object, *, expected_total: float) -> None:
    try:
        components = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("reward component sidecar must be iterable") from error
    names = ("answer_reward", "format_reward", "conditional_tool_reward")
    if len(components) != 3 or tuple(item[0] for item in components) != names:
        raise ValueError("reward component sidecar differs from Pilot equation")
    raw = tuple(float(item[1]) for item in components)
    if (
        raw[0] not in {0.0, 1.0}
        or raw[1] not in {-1.0, 0.0}
        or raw[2]
        not in {
            0.0,
            1.0,
        }
    ):
        raise ValueError("reward component sidecar has invalid raw values")
    total = 0.8 * raw[0] + 0.2 * raw[1] + 1.2 * raw[2]
    if not math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reward component sidecar differs from exact total")


def _integer_group_ids(
    group_uids: tuple[str, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    identities: dict[str, int] = {}
    values: list[int] = []
    for uid in group_uids:
        if uid not in identities:
            identities[uid] = len(identities)
        values.append(identities[uid])
    return torch.tensor(values, dtype=torch.int64, device=device)


def _set_or_validate_tensor(batch: Any, name: str, expected: torch.Tensor) -> None:
    if name in batch:
        actual = batch[name]
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise ValueError(f"existing {name} differs from repo-owned GRPO value")
        return
    batch[name] = expected


__all__ = [
    "POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA",
    "POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD",
    "RewardedTrajectoryFinalizerPort",
    "VerlPilotRewardBatchView",
    "VerlRewardedAgentLoopOutputBuilder",
    "bind_policy_pilot_exact_grpo_fields",
    "validate_policy_pilot_reward_data_proto",
]
