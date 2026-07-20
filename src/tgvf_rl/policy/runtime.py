"""CPU-only composition boundary for one complete Policy Pilot GRPO group.

This layer owns orchestration, not model semantics. Prompt rendering, sampling,
tool execution, observation/Hq/D materialization, reward-context construction,
trajectory replay finalization, and policy/reference replay are all injected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Protocol

import torch

from tgvf_rl.framework.verl.data_bridge import DataProtoPayload
from tgvf_rl.framework.verl.rollout_bridge import (
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    RolloutBridgeRecord,
)
from tgvf_rl.objectives import (
    GRPOSpec,
    ObjectiveResult,
    PolicyLogProbSet,
    compute_grpo_loss,
)
from tgvf_rl.rewards import PilotRewardPipeline, RewardContext, RewardResult
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .batch import (
    PILOT_EXACT_REWARD_FIELD,
    POLICY_PILOT_V1_GROUP_SIZE,
    VERL_GRPO_GROUP_UID_FIELD,
    PilotGroupedRollout,
    materialize_policy_pilot_group_batch,
)
from .lifecycle import (
    PolicyBatchLifecycle,
    PolicyBatchLifecycleManager,
    PolicyBatchMilestone,
    PolicyBatchTransientState,
)

if TYPE_CHECKING:
    from tgvf_rl.environment.agent_loop import (
        FrameworkNeutralAgentLoop,
        RolloutRequest,
    )


class RewardContextProvider(Protocol):
    def build(
        self,
        *,
        request: RolloutRequest,
        trajectory: TrajectoryRecord,
    ) -> RewardContext: ...


class TrajectoryReplayFinalizerPort(Protocol):
    def finalize(
        self,
        *,
        request: RolloutRequest,
        trajectory: TrajectoryRecord,
        reward: RewardResult,
    ) -> RolloutBridgeRecord: ...


@dataclass(frozen=True, slots=True)
class PolicyReplayMaterialization:
    """Policy roles plus proof that policy/reference used each exact bundle."""

    logprobs: PolicyLogProbSet
    policy_replay_bundle_sha256s: tuple[str, ...]
    reference_replay_bundle_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.logprobs, PolicyLogProbSet):
            raise TypeError("logprobs must be a PolicyLogProbSet")
        object.__setattr__(
            self,
            "policy_replay_bundle_sha256s",
            tuple(self.policy_replay_bundle_sha256s),
        )
        object.__setattr__(
            self,
            "reference_replay_bundle_sha256s",
            tuple(self.reference_replay_bundle_sha256s),
        )


class PolicyReplayMaterializerPort(Protocol):
    def materialize(self, payload: DataProtoPayload) -> PolicyReplayMaterialization: ...


@dataclass(frozen=True, slots=True)
class PilotGroupRuntimeRequest:
    """One n=8 group in a unique rollout-batch lifecycle instance.

    ``group_uid`` identifies this rollout/update instance.  It must include an
    execution nonce/step and must never be a reusable dataset sample ID.
    """

    group_uid: str
    rollout_requests: tuple[RolloutRequest, ...]
    pad_token_id: int
    lifecycle: PolicyBatchLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.group_uid, str) or not self.group_uid.strip():
            raise ValueError("group_uid must be a non-empty string")
        object.__setattr__(self, "rollout_requests", tuple(self.rollout_requests))
        if len(self.rollout_requests) != POLICY_PILOT_V1_GROUP_SIZE:
            raise ValueError(
                f"Policy Pilot runtime requires exactly {POLICY_PILOT_V1_GROUP_SIZE} "
                "rollout requests"
            )
        if any(
            not isinstance(request, _rollout_request_type())
            for request in self.rollout_requests
        ):
            raise TypeError("rollout_requests must contain RolloutRequest values")
        identities = tuple(
            request.identity.canonical_id for request in self.rollout_requests
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Policy Pilot runtime requests must be trajectory-unique")
        if any(
            request.identity.group_id != self.group_uid
            for request in self.rollout_requests
        ):
            raise ValueError("rollout request group identity differs from group_uid")
        expected_prompt = self.rollout_requests[0].initial_prompt_token_ids
        if any(
            request.initial_prompt_token_ids != expected_prompt
            for request in self.rollout_requests[1:]
        ):
            raise ValueError("one Pilot group requires identical exact prompt IDs")
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be an explicit non-negative integer")
        if not isinstance(self.lifecycle, PolicyBatchLifecycle):
            raise TypeError("request lifecycle must be PolicyBatchLifecycle")
        if self.lifecycle.batch_id != self.group_uid:
            raise ValueError("request lifecycle batch identity differs from group_uid")
        if self.lifecycle.trajectory_ids != identities:
            raise ValueError(
                "request lifecycle trajectories differ from rollout requests"
            )
        self.lifecycle.assert_open()


class PilotGroupRuntimeResult:
    """Runtime outputs whose replay-heavy fields expire with their batch."""

    __slots__ = (
        "trajectories",
        "reward_contexts",
        "rewards",
        "replay_bundle_sha256s",
        "lifecycle",
        "_transient_state",
    )

    def __init__(
        self,
        *,
        trajectories: tuple[TrajectoryRecord, ...],
        reward_contexts: tuple[RewardContext, ...],
        rewards: tuple[RewardResult, ...],
        grouped_rollouts: tuple[PilotGroupedRollout, ...],
        payload: DataProtoPayload,
        policy_replay: PolicyReplayMaterialization,
        objective: ObjectiveResult,
        lifecycle: PolicyBatchLifecycle,
    ) -> None:
        self.trajectories = tuple(trajectories)
        self.reward_contexts = tuple(reward_contexts)
        self.rewards = tuple(rewards)
        self.replay_bundle_sha256s = tuple(
            grouped.rollout.replay_bundle.bundle_sha256 for grouped in grouped_rollouts
        )
        self.lifecycle = lifecycle
        self._transient_state = PolicyBatchTransientState(
            {
                "grouped_rollouts": tuple(grouped_rollouts),
                "payload": payload,
                "policy_replay": policy_replay,
                "objective": objective,
            }
        )

    @property
    def grouped_rollouts(self) -> tuple[PilotGroupedRollout, ...]:
        value = self._transient_state.get("grouped_rollouts")
        assert isinstance(value, tuple)
        return value

    @property
    def payload(self) -> DataProtoPayload:
        value = self._transient_state.get("payload")
        assert isinstance(value, DataProtoPayload)
        return value

    @property
    def policy_replay(self) -> PolicyReplayMaterialization:
        value = self._transient_state.get("policy_replay")
        assert isinstance(value, PolicyReplayMaterialization)
        return value

    @property
    def objective(self) -> ObjectiveResult:
        value = self._transient_state.get("objective")
        assert isinstance(value, ObjectiveResult)
        return value

    @property
    def released(self) -> bool:
        return self._transient_state.released


class PolicyPilotRuntime:
    """Compose one synchronous, zero-staleness CPU Pilot group update."""

    def __init__(
        self,
        *,
        agent_loop: FrameworkNeutralAgentLoop,
        reward_pipeline: PilotRewardPipeline,
        reward_context_provider: RewardContextProvider,
        replay_finalizer: TrajectoryReplayFinalizerPort,
        policy_replay_materializer: PolicyReplayMaterializerPort,
        grpo_spec: GRPOSpec,
        batch_lifecycle_manager: PolicyBatchLifecycleManager,
    ) -> None:
        if not isinstance(agent_loop, _agent_loop_type()):
            raise TypeError("agent_loop must be FrameworkNeutralAgentLoop")
        if not isinstance(reward_pipeline, PilotRewardPipeline):
            raise TypeError("reward_pipeline must be PilotRewardPipeline")
        for name, value, method in (
            ("reward_context_provider", reward_context_provider, "build"),
            ("replay_finalizer", replay_finalizer, "finalize"),
            (
                "policy_replay_materializer",
                policy_replay_materializer,
                "materialize",
            ),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{name} must implement {method}()")
        if not isinstance(grpo_spec, GRPOSpec):
            raise TypeError("grpo_spec must be GRPOSpec")
        if not isinstance(batch_lifecycle_manager, PolicyBatchLifecycleManager):
            raise TypeError(
                "batch_lifecycle_manager must be PolicyBatchLifecycleManager"
            )
        if grpo_spec.expected_group_size != POLICY_PILOT_V1_GROUP_SIZE:
            raise ValueError("runtime GRPO spec must require Pilot n=8 groups")
        _validate_lifecycle_store_assembly(
            agent_loop=agent_loop,
            replay_finalizer=replay_finalizer,
            manager=batch_lifecycle_manager,
        )
        self.agent_loop = agent_loop
        self.reward_pipeline = reward_pipeline
        self.reward_context_provider = reward_context_provider
        self.replay_finalizer = replay_finalizer
        self.policy_replay_materializer = policy_replay_materializer
        self.grpo_spec = grpo_spec
        self.batch_lifecycle_manager = batch_lifecycle_manager

    def run_group(self, request: PilotGroupRuntimeRequest) -> PilotGroupRuntimeResult:
        if not isinstance(request, PilotGroupRuntimeRequest):
            raise TypeError("request must be PilotGroupRuntimeRequest")

        lifecycle = request.lifecycle
        self.batch_lifecycle_manager.assert_owns(lifecycle)
        try:
            return self._run_group_open(request, lifecycle)
        except BaseException as error:
            try:
                lifecycle.abort()
            except BaseException as cleanup_error:
                error.add_note(
                    "Policy batch cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _run_group_open(
        self,
        request: PilotGroupRuntimeRequest,
        lifecycle: PolicyBatchLifecycle,
    ) -> PilotGroupRuntimeResult:
        with lifecycle.consume(PolicyBatchMilestone.BEHAVIOR_REPLAY):
            (
                trajectories,
                contexts,
                rewards,
                grouped_rollouts,
                payload,
            ) = self._rollout_and_materialize_payload(request)
            lifecycle.attach_data_proto(payload)

        with (
            lifecycle.consume(PolicyBatchMilestone.CURRENT_REPLAY),
            lifecycle.consume(PolicyBatchMilestone.REFERENCE_REPLAY),
        ):
            policy_replay = self.policy_replay_materializer.materialize(payload)
            if not isinstance(policy_replay, PolicyReplayMaterialization):
                raise TypeError(
                    "policy replay materializer must return PolicyReplayMaterialization"
                )
            self._validate_policy_replay(payload, policy_replay)

        exact_rewards = torch.tensor(
            tuple(payload.non_tensor_batch[PILOT_EXACT_REWARD_FIELD]),
            dtype=policy_replay.logprobs.current.values.dtype,
            device=policy_replay.logprobs.current.values.device,
        )
        group_ids = _integer_group_ids(
            tuple(payload.non_tensor_batch[VERL_GRPO_GROUP_UID_FIELD]),
            device=exact_rewards.device,
        )
        objective = compute_grpo_loss(
            self.grpo_spec,
            policy_replay.logprobs,
            exact_rewards,
            group_ids,
        )
        result = PilotGroupRuntimeResult(
            trajectories=trajectories,
            reward_contexts=contexts,
            rewards=rewards,
            grouped_rollouts=grouped_rollouts,
            payload=payload,
            policy_replay=policy_replay,
            objective=objective,
            lifecycle=lifecycle,
        )
        lifecycle.attach_transient_state(result._transient_state)
        return result

    def _rollout_and_materialize_payload(
        self, request: PilotGroupRuntimeRequest
    ) -> tuple[
        tuple[TrajectoryRecord, ...],
        tuple[RewardContext, ...],
        tuple[RewardResult, ...],
        tuple[PilotGroupedRollout, ...],
        DataProtoPayload,
    ]:
        trajectories: list[TrajectoryRecord] = []
        contexts: list[RewardContext] = []
        rewards: list[RewardResult] = []
        grouped_rollouts: list[PilotGroupedRollout] = []
        for rollout_request in request.rollout_requests:
            trajectory = self.agent_loop.run(rollout_request)
            context = self.reward_context_provider.build(
                request=rollout_request,
                trajectory=trajectory,
            )
            if not isinstance(context, RewardContext):
                raise TypeError("reward context provider must return RewardContext")
            reward = self.reward_pipeline.score(context)
            if not isinstance(reward, RewardResult) or not math.isfinite(reward.total):
                raise ValueError("reward pipeline must return a finite RewardResult")
            bridge = self.replay_finalizer.finalize(
                request=rollout_request,
                trajectory=trajectory,
                reward=reward,
            )
            if not isinstance(bridge, RolloutBridgeRecord):
                raise TypeError("replay finalizer must return RolloutBridgeRecord")
            if bridge.trajectory_payload != trajectory:
                raise ValueError("replay finalizer changed the completed trajectory")
            if bridge.prompt_ids != rollout_request.initial_prompt_token_ids:
                raise ValueError("replay finalizer changed the exact prompt token IDs")
            if bridge.reward_score != reward.total:
                raise ValueError("replay finalizer changed the trajectory reward")
            grouped = PilotGroupedRollout(request.group_uid, bridge)
            trajectories.append(trajectory)
            contexts.append(context)
            rewards.append(reward)
            grouped_rollouts.append(grouped)

        payload = materialize_policy_pilot_group_batch(
            grouped_rollouts,
            pad_token_id=request.pad_token_id,
        )
        if payload.tensor_batch["responses"].shape[0] != len(request.rollout_requests):
            raise RuntimeError("Pilot materializer dropped or duplicated a trajectory")
        return (
            tuple(trajectories),
            tuple(contexts),
            tuple(rewards),
            tuple(grouped_rollouts),
            payload,
        )

    @staticmethod
    def _validate_policy_replay(
        payload: DataProtoPayload,
        replay: PolicyReplayMaterialization,
    ) -> None:
        policy = replay.logprobs
        if policy.current.values.device.type != "cpu":
            raise ValueError("Policy Pilot composition runtime is CPU-only")
        if not policy.current.values.requires_grad:
            raise ValueError("current policy replay must retain an autograd graph")
        response_mask = payload.tensor_batch["response_mask"].to(dtype=torch.bool)
        if policy.policy_sampled_mask.device.type != "cpu" or not torch.equal(
            policy.policy_sampled_mask, response_mask
        ):
            raise ValueError("policy replay mask differs from exact response ownership")
        expected_behavior = payload.tensor_batch["rollout_log_probs"].to(
            dtype=policy.behavior.values.dtype
        )
        if not torch.equal(policy.behavior.values, expected_behavior):
            raise ValueError("policy replay changed actual behavior log probabilities")

        bundles = tuple(payload.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD])
        expected_bundle_sha256s = tuple(bundle.bundle_sha256 for bundle in bundles)
        if replay.policy_replay_bundle_sha256s != expected_bundle_sha256s:
            raise ValueError("policy replay did not consume the exact rollout bundles")
        if replay.reference_replay_bundle_sha256s != expected_bundle_sha256s:
            raise ValueError(
                "reference replay did not consume the exact rollout bundles"
            )

        rm_scores = payload.tensor_batch["rm_scores"]
        exact_rewards = torch.tensor(
            tuple(payload.non_tensor_batch[PILOT_EXACT_REWARD_FIELD]),
            dtype=rm_scores.dtype,
        )
        if not torch.equal(rm_scores.sum(dim=-1), exact_rewards):
            raise ValueError("rm_scores differ from exact trajectory rewards")


def _integer_group_ids(
    group_uids: tuple[object, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    identities: dict[str, int] = {}
    values: list[int] = []
    for uid in group_uids:
        if not isinstance(uid, str) or not uid:
            raise ValueError("veRL GRPO uid sidecar is malformed")
        if uid not in identities:
            identities[uid] = len(identities)
        values.append(identities[uid])
    return torch.tensor(values, dtype=torch.int64, device=device)


def _agent_loop_type() -> type[FrameworkNeutralAgentLoop]:
    from tgvf_rl.environment.agent_loop import FrameworkNeutralAgentLoop

    return FrameworkNeutralAgentLoop


def _rollout_request_type() -> type[RolloutRequest]:
    from tgvf_rl.environment.agent_loop import RolloutRequest

    return RolloutRequest


def _validate_lifecycle_store_assembly(
    *,
    agent_loop: FrameworkNeutralAgentLoop,
    replay_finalizer: TrajectoryReplayFinalizerPort,
    manager: PolicyBatchLifecycleManager,
) -> None:
    if getattr(agent_loop.behavior_recorder, "store", None) is not (
        manager.behavior_store
    ):
        raise ValueError(
            "Policy lifecycle manager and agent loop must share BehaviorTraceStore"
        )
    if getattr(replay_finalizer, "observation_store", None) is not (
        manager.observation_store
    ):
        raise ValueError(
            "Policy lifecycle manager and replay finalizer must share ObservationStore"
        )
    tool_runtime = agent_loop.tool_runtime
    tool_store = getattr(tool_runtime, "store", None)
    if tool_store is None:
        focus_tool = getattr(tool_runtime, "focus_tool", None)
        tool_store = getattr(focus_tool, "store", None)
    if tool_store is not manager.observation_store:
        raise ValueError(
            "Policy lifecycle manager and tool runtime must share ObservationStore"
        )
    if getattr(tool_runtime, "execution_ledger", None) is not (
        manager.focus_execution_ledger
    ):
        raise ValueError(
            "Policy lifecycle manager and tool runtime must share FocusExecutionLedger"
        )


__all__ = [
    "PilotGroupRuntimeRequest",
    "PilotGroupRuntimeResult",
    "PolicyPilotRuntime",
    "PolicyReplayMaterialization",
    "PolicyReplayMaterializerPort",
    "RewardContextProvider",
    "TrajectoryReplayFinalizerPort",
]
