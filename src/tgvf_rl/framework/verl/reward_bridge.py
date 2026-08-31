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
from tgvf_rl.policy.metrics import (
    POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT,
)
from tgvf_rl.rewards.schema import (
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PILOT_REWARD_WEIGHTS_BY_EQUATION,
    RewardResult,
    deepeyes_reward_equation_for_data_source,
)
from tgvf_rl.rewards.stage3_shaped import (
    STAGE3_ANSWER_REWARD_SCALE,
    STAGE3_PROTOCOL_ERROR_PENALTY,
    STAGE3_REPEATED_CALL_PENALTY,
    Stage3ShapedRewardResult,
)
from tgvf_rl.rewards.stage3_verl_adapter import (
    STAGE3_VERL_QUALITY_APPLICABLE_FIELD,
    STAGE3_VERL_QUALITY_COVERED_FIELD,
    STAGE3_VERL_QUALITY_FAILURE_FIELD,
    STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD,
    STAGE3_VERL_TOOL_LABEL_FIELD,
    STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD,
    STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD,
    STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD,
    Stage3VerlTrajectoryReward,
    Stage3VerlTrajectoryRewardScorer,
)
from tgvf_rl.rewards.verl_adapter import (
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
    PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD,
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
        reward: RewardResult | Stage3ShapedRewardResult,
    ) -> RolloutBridgeRecord: ...


class VerlRewardedAgentLoopOutputBuilder:
    """Score, finalize, and emit one public AgentLoopOutput with reward_score."""

    def __init__(
        self,
        *,
        request: object,
        scorer: PilotVerlTrajectoryRewardScorer | Stage3VerlTrajectoryRewardScorer,
        finalizer: RewardedTrajectoryFinalizerPort,
        metrics_factory: Callable[
            [TrajectoryRecord, PilotVerlTrajectoryReward | Stage3VerlTrajectoryReward],
            object,
        ],
        agent_loop_output_cls: type[Any] | None = None,
    ) -> None:
        if not hasattr(request, "identity"):
            raise TypeError("rewarded output request must expose identity")
        if not isinstance(
            scorer,
            (PilotVerlTrajectoryRewardScorer, Stage3VerlTrajectoryRewardScorer),
        ):
            raise TypeError("scorer must be a supported trajectory reward scorer")
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
    reward_bridge_schema_version: str


def validate_policy_pilot_reward_data_proto(
    data: object,
    *,
    expected_group_size: int = POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT,
    expected_stage3_answer_reward_scale: float = STAGE3_ANSWER_REWARD_SCALE,
    expected_stage3_repeated_call_penalty: float = STAGE3_REPEATED_CALL_PENALTY,
    expected_stage3_protocol_error_penalty: float = STAGE3_PROTOCOL_ERROR_PENALTY,
    expected_stage3_tool_utility_reward_enabled: bool = True,
    expected_stage3_visual_quality_enabled: bool = True,
) -> VerlPilotRewardBatchView:
    """Prove upstream used exact rewards and retained every configured group."""

    if type(expected_group_size) is not int or expected_group_size <= 0:
        raise ValueError("expected reward group size must be a positive integer")
    for field_name, value in (
        ("answer reward scale", expected_stage3_answer_reward_scale),
        ("repeated-call penalty", expected_stage3_repeated_call_penalty),
        ("protocol error penalty", expected_stage3_protocol_error_penalty),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Stage3 {field_name} must be finite and non-negative")
    for field_name, value in (
        (
            "tool-utility reward switch",
            expected_stage3_tool_utility_reward_enabled,
        ),
        ("visual-quality reward switch", expected_stage3_visual_quality_enabled),
    ):
        if type(value) is not bool:
            raise ValueError(f"Stage3 {field_name} must be bool")

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
    if batch_size == 0 or batch_size % expected_group_size:
        raise ValueError(
            f"Policy reward batch must contain complete n={expected_group_size} groups"
        )
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
    bridge_schemas = set(fields[PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD])
    if len(bridge_schemas) != 1:
        raise ValueError("one reward batch cannot mix reward bridge schemas")
    bridge_schema = next(iter(bridge_schemas))
    if bridge_schema not in {
        PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
        STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    }:
        raise ValueError("trajectory reward bridge schema was changed")
    if bridge_schema == PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION:
        fields.update(
            {
                name: _row_values(
                    _required(non_tensors, name, "DataProto.non_tensor_batch"),
                    batch_size,
                    name,
                )
                for name in (
                    PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD,
                    PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD,
                )
            }
        )
        fields["data_source"] = _row_values(
            _required(non_tensors, "data_source", "DataProto.non_tensor_batch"),
            batch_size,
            "data_source",
        )
    stage3_fields: dict[str, tuple[object, ...]] = {}
    if bridge_schema == STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION:
        stage3_fields = {
            name: _row_values(
                _required(non_tensors, name, "DataProto.non_tensor_batch"),
                batch_size,
                name,
            )
            for name in (
                STAGE3_VERL_TOOL_LABEL_FIELD,
                STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD,
                STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD,
                STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD,
                STAGE3_VERL_QUALITY_APPLICABLE_FIELD,
                STAGE3_VERL_QUALITY_COVERED_FIELD,
                STAGE3_VERL_QUALITY_FAILURE_FIELD,
                STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD,
            )
        }

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
        stage3_tool_utility_bound = None
        if stage3_fields:
            stage3_tool_utility_bound = _validate_stage3_row_sidecars(
                stage3_fields,
                row_index=row_index,
                expected_tool_utility_reward_enabled=(
                    expected_stage3_tool_utility_reward_enabled
                ),
                expected_visual_quality_enabled=(
                    expected_stage3_visual_quality_enabled
                ),
                successful_observation_count=len(trajectory.observations),
            )
        _validate_component_sidecar(
            fields[PILOT_VERL_REWARD_COMPONENTS_FIELD][row_index],
            expected_total=float(exact_reward),
            bridge_schema=bridge_schema,
            equation_route=(
                fields[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD][row_index]
                if bridge_schema == PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
                else None
            ),
            applied_weights=(
                fields[PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD][row_index]
                if bridge_schema == PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
                else None
            ),
            data_source=(
                fields["data_source"][row_index]
                if bridge_schema == PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
                else None
            ),
            expected_stage3_answer_reward_scale=float(
                expected_stage3_answer_reward_scale
            ),
            expected_stage3_repeated_call_penalty=float(
                expected_stage3_repeated_call_penalty
            ),
            expected_stage3_protocol_error_penalty=float(
                expected_stage3_protocol_error_penalty
            ),
            expected_stage3_visual_quality_enabled=(
                expected_stage3_visual_quality_enabled
            ),
            stage3_tool_utility_bound=stage3_tool_utility_bound,
            stage3_tool_call_count=(
                len(trajectory.observations) + len(trajectory.tool_errors)
            ),
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
        if len(row_indices) != expected_group_size:
            raise ValueError(
                f"reward group {group_uid!r} is not complete n={expected_group_size}"
            )
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
        if rollout_indices != set(range(expected_group_size)):
            raise ValueError(
                "reward group rollout indices must be exactly "
                f"0..{expected_group_size - 1}"
            )
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
        reward_bridge_schema_version=str(bridge_schema),
    )


def bind_policy_pilot_exact_grpo_fields(
    data: object,
    *,
    diagnostic_kl_estimator: ReferenceKLEstimator,
    expected_group_size: int = POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT,
    expected_stage3_answer_reward_scale: float = STAGE3_ANSWER_REWARD_SCALE,
    expected_stage3_repeated_call_penalty: float = STAGE3_REPEATED_CALL_PENALTY,
    expected_stage3_protocol_error_penalty: float = STAGE3_PROTOCOL_ERROR_PENALTY,
    expected_stage3_tool_utility_reward_enabled: bool = True,
    expected_stage3_visual_quality_enabled: bool = True,
) -> VerlPilotRewardBatchView:
    """Attach repo-owned scores/rewards/advantages consumed by the exact loss."""

    view = validate_policy_pilot_reward_data_proto(
        data,
        expected_group_size=expected_group_size,
        expected_stage3_answer_reward_scale=expected_stage3_answer_reward_scale,
        expected_stage3_repeated_call_penalty=(expected_stage3_repeated_call_penalty),
        expected_stage3_protocol_error_penalty=(expected_stage3_protocol_error_penalty),
        expected_stage3_tool_utility_reward_enabled=(
            expected_stage3_tool_utility_reward_enabled
        ),
        expected_stage3_visual_quality_enabled=(expected_stage3_visual_quality_enabled),
    )
    batch, _ = _data_parts(data)
    rm_scores = batch["rm_scores"]
    response_mask = batch["response_mask"].to(dtype=torch.bool)
    rewards = torch.tensor(
        view.rewards,
        dtype=rm_scores.dtype,
        device=rm_scores.device,
    )
    group_ids = _integer_group_ids(view.group_uids, device=rm_scores.device)
    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=diagnostic_kl_estimator,
        expected_group_size=expected_group_size,
    )
    sequence_advantages = compute_group_advantages(rewards, group_ids, spec)
    advantages = sequence_advantages[:, None] * response_mask
    for name, expected in (
        ("token_level_scores", rm_scores),
        ("token_level_rewards", rm_scores),
        ("advantages", advantages),
        ("returns", advantages),
    ):
        _set_or_validate_tensor(batch, name, expected)
    return validate_policy_pilot_reward_data_proto(
        data,
        expected_group_size=expected_group_size,
        expected_stage3_answer_reward_scale=expected_stage3_answer_reward_scale,
        expected_stage3_repeated_call_penalty=(expected_stage3_repeated_call_penalty),
        expected_stage3_protocol_error_penalty=(expected_stage3_protocol_error_penalty),
        expected_stage3_tool_utility_reward_enabled=(
            expected_stage3_tool_utility_reward_enabled
        ),
        expected_stage3_visual_quality_enabled=(expected_stage3_visual_quality_enabled),
    )


def _agent_loop_reward_sidecars(
    scored: PilotVerlTrajectoryReward | Stage3VerlTrajectoryReward,
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


def _validate_component_sidecar(
    value: object,
    *,
    expected_total: float,
    bridge_schema: object = PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    equation_route: object = None,
    applied_weights: object = None,
    data_source: object = None,
    expected_stage3_answer_reward_scale: float = STAGE3_ANSWER_REWARD_SCALE,
    expected_stage3_repeated_call_penalty: float = STAGE3_REPEATED_CALL_PENALTY,
    expected_stage3_protocol_error_penalty: float = STAGE3_PROTOCOL_ERROR_PENALTY,
    expected_stage3_visual_quality_enabled: bool = True,
    stage3_tool_utility_bound: bool | None = None,
    stage3_tool_call_count: int = 0,
) -> None:
    try:
        components = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("reward component sidecar must be iterable") from error
    if bridge_schema == STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION:
        _validate_stage3_component_sidecar(
            components,
            expected_total=expected_total,
            expected_answer_reward_scale=expected_stage3_answer_reward_scale,
            expected_repeated_call_penalty=expected_stage3_repeated_call_penalty,
            expected_protocol_error_penalty=expected_stage3_protocol_error_penalty,
            visual_quality_enabled=expected_stage3_visual_quality_enabled,
            tool_utility_bound=stage3_tool_utility_bound,
            tool_call_count=stage3_tool_call_count,
        )
        return
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
    if equation_route not in PILOT_REWARD_WEIGHTS_BY_EQUATION:
        raise ValueError("reward equation route sidecar is unsupported")
    try:
        weights = tuple(float(weight) for weight in applied_weights)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise ValueError("reward applied-weight sidecar is malformed") from error
    expected_weights = PILOT_REWARD_WEIGHTS_BY_EQUATION[equation_route]
    if weights != expected_weights:
        raise ValueError("reward applied weights differ from its equation route")
    if isinstance(equation_route, str) and equation_route.startswith("deepeyes-"):
        expected_route, source_weights = deepeyes_reward_equation_for_data_source(
            data_source
        )
        if equation_route != expected_route or weights != source_weights:
            raise ValueError(
                "DeepEyes reward equation differs from DataProto data_source"
            )
    if equation_route == PILOT_REWARD_EQUATION_DEEPEYES_MATH and raw[2] != 0.0:
        raise ValueError("DeepEyes math reward cannot contain a tool component")
    total = sum(score * weight for score, weight in zip(raw, weights, strict=True))
    if not math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reward component sidecar differs from exact total")


def _validate_stage3_component_sidecar(
    components: tuple[object, ...],
    *,
    expected_total: float,
    expected_answer_reward_scale: float,
    expected_repeated_call_penalty: float,
    expected_protocol_error_penalty: float,
    visual_quality_enabled: bool,
    tool_utility_bound: bool | None,
    tool_call_count: int,
) -> None:
    names = ("answer", "tool", "focus", "grounding", "protocol")
    try:
        actual_names = tuple(item[0] for item in components)  # type: ignore[index]
        values = tuple(float(item[1]) for item in components)  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("Stage3 reward component sidecar is malformed") from error
    if len(components) != 5 or actual_names != names:
        raise ValueError("reward component sidecar differs from Stage3 equation")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Stage3 reward components must be finite")
    answer, tool, focus, grounding, protocol = values
    if (
        answer not in {0.0, expected_answer_reward_scale}
        or focus not in {0.0, 0.5, 1.0}
        or grounding not in {-1.0, 0.0, 0.5, 1.0}
        or protocol not in {-expected_protocol_error_penalty, 0.0}
    ):
        raise ValueError("Stage3 reward component sidecar has invalid values")
    if tool_utility_bound is False:
        expected_tool = -expected_repeated_call_penalty * max(0, tool_call_count - 1)
        if not math.isclose(tool, expected_tool, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "T-free Stage3 tool component differs from repeated-call penalty"
            )
    if not visual_quality_enabled and (focus != 0.0 or grounding != 0.0):
        raise ValueError("T-free Stage3 cannot carry visual-quality rewards")
    if not math.isclose(
        math.fsum(values), expected_total, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("reward component sidecar differs from exact total")


def _validate_stage3_row_sidecars(
    fields: Mapping[str, tuple[object, ...]],
    *,
    row_index: int,
    expected_tool_utility_reward_enabled: bool,
    expected_visual_quality_enabled: bool,
    successful_observation_count: int,
) -> bool:
    label = fields[STAGE3_VERL_TOOL_LABEL_FIELD][row_index]
    confidence = fields[STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD][row_index]
    label_row_sha256 = fields[STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD][row_index]
    sidecar_sha256 = fields[STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD][row_index]
    utility_fields = (label, confidence, label_row_sha256, sidecar_sha256)
    tool_utility_bound = utility_fields != (None, None, None, None)
    if tool_utility_bound != expected_tool_utility_reward_enabled:
        raise ValueError(
            "Stage3 tool-utility sidecars differ from the configured switch"
        )
    if tool_utility_bound:
        if any(value is None for value in utility_fields):
            raise ValueError("Stage3 tool-utility sidecars are partially bound")
        if label not in {"needed", "optional", "unnecessary"}:
            raise ValueError("Stage3 tool label sidecar is invalid")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("Stage3 tool label confidence is invalid")
        _require_sha256(label_row_sha256, "Stage3 tool label row")
        _require_sha256(sidecar_sha256, "Stage3 tool sidecar")
    applicable = fields[STAGE3_VERL_QUALITY_APPLICABLE_FIELD][row_index]
    covered = fields[STAGE3_VERL_QUALITY_COVERED_FIELD][row_index]
    failure = fields[STAGE3_VERL_QUALITY_FAILURE_FIELD][row_index]
    visual_usage = fields[STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD][row_index]
    if type(applicable) is not bool or type(covered) is not bool:
        raise ValueError("Stage3 visual-quality coverage sidecars must be bool")
    if not applicable and (covered or failure is not None):
        raise ValueError("non-applicable Stage3 visual judge cannot be covered/failed")
    if applicable and covered == (failure is not None):
        raise ValueError("Stage3 visual-quality coverage/failure sidecars differ")
    if failure is not None and (not isinstance(failure, str) or not failure.strip()):
        raise ValueError("Stage3 visual-quality failure sidecar is invalid")
    if visual_usage is not None:
        try:
            prompt_tokens, completion_tokens, total_tokens, cost = visual_usage  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Stage3 visual judge usage sidecar is malformed"
            ) from error
        if (
            type(prompt_tokens) is not int
            or type(completion_tokens) is not int
            or type(total_tokens) is not int
            or min(prompt_tokens, completion_tokens, total_tokens) < 0
            or total_tokens != prompt_tokens + completion_tokens
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError("Stage3 visual judge usage sidecar is invalid")
    if visual_usage is not None and not applicable:
        raise ValueError("Stage3 visual judge usage requires an applicable call")
    expected_applicable = (
        expected_visual_quality_enabled and successful_observation_count >= 1
    )
    if applicable != expected_applicable:
        raise ValueError(
            "Stage3 visual judge applicability differs from the configured switch"
        )
    return tool_utility_bound


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
