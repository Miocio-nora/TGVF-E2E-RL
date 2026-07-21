from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import hashlib
import weakref

import pytest
import torch

from tgvf_rl.contracts.identity import (
    ArtifactIdentity,
    ComponentRole,
    PolicyVersion,
)
from tgvf_rl.contracts.errors import RecoverableToolExecutionError
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    SamplingIdentity,
    TokenSpan,
)
from tgvf_rl.environment import (
    FocusExecutionLedger,
    FrameworkNeutralAgentLoop,
    RolloutRequest,
    SampledPolicyTurn,
)
from tgvf_rl.framework.verl import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    make_objective_sentinels,
)
from tgvf_rl.framework.verl.rollout_bridge import (
    _trajectory_response_materialization,
)
from tgvf_rl.objectives import (
    LogProbSource,
    PolicyLogProbSet,
    ReferenceKLEstimator,
    RoleLogProbs,
    policy_pilot_v1_grpo_spec,
)
from tgvf_rl.observations import (
    MaterializedTrajectoryReplayTensors,
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayFinalizationRequest,
    finalize_trajectory_replay,
)
from tgvf_rl.policy import (
    PilotGroupRuntimeRequest,
    PolicyBatchLifecycleManager,
    PolicyBatchMilestone,
    PolicyPilotRuntime,
    PolicyReplayMaterialization,
)
from tgvf_rl.protocol import StandardToolError, StrictToolCallParser, TokenByteSpan
from tgvf_rl.rewards import (
    AnswerTaskKind,
    AnswerVerificationResult,
    PilotRewardPipeline,
    PilotRewardSpec,
    RewardContext,
    reward_context_from_trajectory,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tests.support import populated_observation_store, trajectory_source_visual


SHA = "9" * 64
BEHAVIOR_VERSION = PolicyVersion("policy-pilot-runtime", 0, "0" * 64)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tool_call(target: str, reasoning: str = "inspect") -> str:
    return (
        f"{reasoning}</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"'
        f"{target}"
        '"}}</tool_call>'
    )


SCRIPTS = (
    ("brief</think>\ncorrect",),
    (_tool_call("label"), "read result</think>\ncorrect"),
    (
        _tool_call("upper", "inspect upper region"),
        _tool_call("lower", "inspect lower region carefully"),
        "compare both observations</think>\ncorrect",
    ),
    (_tool_call("fail"), "recover after error</think>\ncorrect"),
    ("long direct reasoning with no tool</think>\nwrong",),
    (_tool_call("serial"), "use visible serial</think>\ncorrect"),
    (
        _tool_call("left", "inspect"),
        _tool_call("right", "inspect another side"),
        "combine evidence</think>\nwrong",
    ),
    (_tool_call("fail", "attempt unavailable focus"), "recover</think>\nwrong"),
)


class _ScriptedSampler:
    def __init__(self) -> None:
        self.trajectory_index = -1

    def sample(self, prompt_token_ids, sampling_parameters, *, turn_index):
        del prompt_token_ids, sampling_parameters
        if turn_index == 0:
            self.trajectory_index += 1
        text = SCRIPTS[self.trajectory_index][turn_index]
        token_ids = tuple(ord(character) for character in text)
        byte_spans = tuple(
            TokenByteSpan(index, token_id, index, index + 1)
            for index, token_id in enumerate(token_ids)
        )
        seed = 1000 + 10 * self.trajectory_index + turn_index
        sampling = SamplingIdentity(
            policy_version=BEHAVIOR_VERSION,
            backend="vllm",
            backend_version="fake-runtime-v1",
            seed=seed,
            rng_state_sha256=_digest(f"rng-{seed}"),
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        )
        behavior_logprobs = tuple(
            -0.01 * (1 + self.trajectory_index + turn_index + token_index)
            for token_index in range(len(token_ids))
        )
        return SampledPolicyTurn(
            text=text,
            token_ids=token_ids,
            token_byte_spans=byte_spans,
            behavior_logprobs=behavior_logprobs,
            sampling=sampling,
            think_token_span=TokenSpan(0, text.index("</think>") + len("</think>")),
            stop_reason="stop",
            backend_request_sha256=_digest(f"request-{seed}"),
            backend_response_sha256=_digest(f"response-{seed}"),
        )


class _FakeTGVFToolRuntime:
    def __init__(
        self,
        store: ObservationStore,
        source: ObservationHandle,
        execution_ledger: FocusExecutionLedger,
    ) -> None:
        self.store = store
        self.base = store.resolve_record(source)
        self.execution_ledger = execution_ledger

    def execute(self, parsed_call, context) -> ObservationHandle:
        if parsed_call.target == "fail":
            raise RecoverableToolExecutionError("intentional fixture tool failure")
        d_positions = (6 + 2 * context.call_index, 7 + 2 * context.call_index)
        record = replace(
            self.base,
            observation_id=(
                "runtime-observation-"
                f"{context.trajectory_identity.rollout_index}-{context.call_index}"
            ),
            call_index=context.call_index,
            model=context.model,
            condition=replace(
                self.base.condition,
                sampled_target_text_sha256=_digest(parsed_call.target),
                sampled_target_token_start=parsed_call.target_span.token_start,
                sampled_target_token_end=parsed_call.target_span.token_end,
                conditioning_target_token_start=(
                    len(context.prompt_token_ids_before_turn)
                    + parsed_call.target_span.token_start
                ),
                conditioning_target_token_end=(
                    len(context.prompt_token_ids_before_turn)
                    + parsed_call.target_span.token_end
                ),
                source_sequence_length=len(context.conditioning_input_ids),
                source_input_ids_sha256=_digest(
                    ",".join(str(value) for value in context.conditioning_input_ids)
                ),
                trajectory_ids=(context.trajectory_identity.canonical_id,),
                call_indices=(context.call_index,),
                policy_version=context.behavior_policy,
            ),
            branches=tuple(
                replace(branch, injection_positions=d_positions)
                for branch in self.base.branches
            ),
            layout=replace(
                self.base.layout,
                d_positions=d_positions,
                deepstack_injection_positions=tuple(
                    d_positions for _ in self.base.branches
                ),
            ),
        )
        return self.store.put(record)


class _FakeNativeAppender:
    def append(
        self,
        prompt_token_ids,
        sampled_turn,
        observation,
        *,
        call_index,
        parsed_call,
    ):
        if isinstance(observation, StandardToolError):
            appended = (910, 911, 912, 913)
        else:
            appended = tuple(range(900 + 10 * call_index, 903 + 11 * call_index))
        return prompt_token_ids + sampled_turn.token_ids + appended, appended


@dataclass(frozen=True)
class _RewardContextProvider:
    def build(
        self,
        *,
        request: RolloutRequest,
        trajectory: TrajectoryRecord,
    ) -> RewardContext:
        del request
        return reward_context_from_trajectory(
            trajectory,
            question="What is the answer?",
            expected_answer="correct",
            task_kind=AnswerTaskKind.OPEN_VQA,
        )


@dataclass(frozen=True)
class _ExactAnswerVerifier:
    identity: ArtifactIdentity

    def verify(self, context: RewardContext) -> AnswerVerificationResult:
        correct = context.candidate_answer.strip() == context.expected_answer
        return AnswerVerificationResult(
            correct=correct,
            route="fake_exact",
            evidence=f"exact={correct}",
            verifier_identity=self.identity,
        )


class _ReplayFinalizer:
    def __init__(
        self,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        source_visual,
    ) -> None:
        self.observation_store = observation_store
        self.behavior_store = behavior_store
        self.source_visual = source_visual
        self.records = []

    def finalize(self, *, request, trajectory, reward):
        behavior_records = tuple(
            self.behavior_store.resolve(turn.behavior_trace)
            for turn in trajectory.assistant_turns
        )
        response_ids, _, _, _ = _trajectory_response_materialization(
            trajectory, behavior_records
        )
        final_ids = request.initial_prompt_token_ids + response_ids
        sequence_length = len(final_ids)
        visible = torch.ones((1, sequence_length), dtype=torch.bool)
        observations_by_turn = {
            call.assistant_turn_index: observation.template_token_ids
            for call, observation in zip(
                trajectory.tool_calls, trajectory.observations, strict=True
            )
        }
        errors_by_turn = {
            error.assistant_turn_index: error.template_token_ids
            for error in trajectory.tool_errors
        }
        native_rows = tuple(
            observations_by_turn.get(
                turn.turn_index, errors_by_turn.get(turn.turn_index)
            )
            for turn in trajectory.assistant_turns
            if turn.turn_index in observations_by_turn
            or turn.turn_index in errors_by_turn
        )
        bridge = finalize_trajectory_replay(
            TrajectoryReplayFinalizationRequest(
                trajectory=trajectory,
                source_visual=self.source_visual,
                tensors=MaterializedTrajectoryReplayTensors(
                    input_ids=torch.tensor([final_ids], dtype=torch.int64),
                    position_ids=torch.arange(sequence_length).view(1, sequence_length),
                    base_attention_mask=visible,
                    policy_attention_mask=visible,
                    reference_attention_mask=visible,
                    teacher_attention_mask=visible,
                ),
                replay_schema_version="trajectory-replay-v1",
                replay_id=f"runtime-replay-{trajectory.identity.rollout_index}",
                trajectory_id=trajectory.identity.canonical_id,
                model=trajectory.model,
                behavior_policy=trajectory.behavior_policy,
                crop_vision_replay_mode="no_crop",
                cache_mode="no_cache",
                cache_prefix_length=0,
                deterministic_forward=True,
                adapter_dropout=0.0,
                maximum_policy_staleness=0,
                initial_prompt_token_ids=request.initial_prompt_token_ids,
                native_tool_appended_token_ids=native_rows,
                sentinel_fields=make_objective_sentinels(
                    f"runtime-{trajectory.identity.rollout_index}"
                ),
                reward_score=reward.total,
            ),
            observation_store=self.observation_store,
            behavior_store=self.behavior_store,
        )
        self.records.append(bridge.replay_bundle.bundle_sha256)
        return bridge


def _role(
    role: ComponentRole,
    values: torch.Tensor,
    *,
    step: int,
    digit: str,
) -> RoleLogProbs:
    return RoleLogProbs(
        role=role,
        values=values,
        policy_version=PolicyVersion("policy-pilot-runtime", step, digit * 64),
        source=(
            LogProbSource.ROLLOUT_RECORDED
            if role is ComponentRole.BEHAVIOR
            else LogProbSource.DETERMINISTIC_REPLAY
        ),
        sampling_transform_sha256=SHA,
    )


class _PolicyReplayMaterializer:
    def __init__(self) -> None:
        self.current = None
        self.bundle_sha256s = ()

    def materialize(self, payload):
        behavior = payload.tensor_batch["rollout_log_probs"].to(torch.float64).clone()
        policy_mask = payload.tensor_batch["response_mask"].to(torch.bool)
        offsets = torch.linspace(
            -0.03,
            0.03,
            behavior.numel(),
            dtype=behavior.dtype,
        ).view_as(behavior)
        self.current = (behavior + offsets).detach().clone().requires_grad_(True)
        bundles = tuple(payload.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD])
        bundle_sha256s = tuple(bundle.bundle_sha256 for bundle in bundles)
        self.bundle_sha256s = bundle_sha256s
        return PolicyReplayMaterialization(
            logprobs=PolicyLogProbSet(
                behavior=_role(
                    ComponentRole.BEHAVIOR,
                    behavior,
                    step=0,
                    digit="0",
                ),
                proximal_old=_role(
                    ComponentRole.PROXIMAL_OLD,
                    behavior.clone(),
                    step=0,
                    digit="1",
                ),
                current=_role(
                    ComponentRole.CURRENT,
                    self.current,
                    step=1,
                    digit="2",
                ),
                reference=_role(
                    ComponentRole.REFERENCE,
                    (behavior - 0.02).clone(),
                    step=0,
                    digit="3",
                ),
                policy_sampled_mask=policy_mask,
            ),
            policy_replay_bundle_sha256s=bundle_sha256s,
            reference_replay_bundle_sha256s=bundle_sha256s,
        )


def _reward_pipeline() -> PilotRewardPipeline:
    def identity(name: str, digit: str) -> ArtifactIdentity:
        return ArtifactIdentity("runtime-test", name, "v1", digit * 64)

    spec = PilotRewardSpec(
        pipeline_identity=identity("reward", "1"),
        answer_verifier_identity=identity("answer", "2"),
        format_verifier_identity=identity("format", "3"),
        tool_verifier_identity=identity("tool", "4"),
    )
    return PilotRewardPipeline(
        spec,
        _ExactAnswerVerifier(spec.answer_verifier_identity),
    )


def test_cpu_policy_pilot_runtime_preserves_mixed_n8_group_and_backpropagates() -> None:
    observation_store, source_handle = populated_observation_store()
    source_record = observation_store.resolve_record(source_handle)
    behavior_store = BehaviorTraceStore()
    execution_ledger = FocusExecutionLedger()
    sampler = _ScriptedSampler()
    agent_loop = FrameworkNeutralAgentLoop(
        sampler=sampler,
        tool_runtime=_FakeTGVFToolRuntime(
            observation_store, source_handle, execution_ledger
        ),
        appender=_FakeNativeAppender(),
        parser=StrictToolCallParser(),
        behavior_recorder=VLLMBehaviorRecorder(behavior_store),
        max_tool_calls=4,
        enabled_tool_names=("tgvf_focus_tool",),
    )
    replay_finalizer = _ReplayFinalizer(
        observation_store,
        behavior_store,
        trajectory_source_visual(source_record),
    )
    policy_replay = _PolicyReplayMaterializer()
    lifecycle_manager = PolicyBatchLifecycleManager(
        observation_store=observation_store,
        behavior_store=behavior_store,
        focus_execution_ledger=execution_ledger,
    )
    runtime = PolicyPilotRuntime(
        agent_loop=agent_loop,
        reward_pipeline=_reward_pipeline(),
        reward_context_provider=_RewardContextProvider(),
        replay_finalizer=replay_finalizer,
        policy_replay_materializer=policy_replay,
        grpo_spec=policy_pilot_v1_grpo_spec(
            diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
        ),
        batch_lifecycle_manager=lifecycle_manager,
    )
    model = source_record.model
    requests = tuple(
        RolloutRequest(
            schema_version="trajectory-v1",
            identity=_trajectory_identity(index),
            model=model,
            behavior_policy=BEHAVIOR_VERSION,
            trajectory_source_visual=trajectory_source_visual(source_record),
            initial_prompt_token_ids=(101, 102, 103),
            sampling_parameters={"temperature": 1.0},
        )
        for index in range(8)
    )
    result = runtime.run_group(
        PilotGroupRuntimeRequest(
            group_uid="runtime-group",
            rollout_requests=requests,
            pad_token_id=0,
            lifecycle=lifecycle_manager.open_batch(
                batch_id="runtime-group",
                trajectory_ids=tuple(
                    request.identity.canonical_id for request in requests
                ),
            ),
        )
    )

    expected_ids = tuple(request.identity.canonical_id for request in requests)
    assert tuple(item.identity.canonical_id for item in result.trajectories) == (
        expected_ids
    )
    assert len(result.grouped_rollouts) == len(replay_finalizer.records) == 8
    assert tuple(item.stop for item in result.trajectories) == (
        TrajectoryStop.DIRECT_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
        TrajectoryStop.DIRECT_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
        TrajectoryStop.FINAL_ANSWER,
    )
    assert tuple(len(item.tool_calls) for item in result.trajectories) == (
        0,
        1,
        2,
        0,
        0,
        1,
        2,
        0,
    )
    assert tuple(len(item.tool_errors) for item in result.trajectories) == (
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    )
    assert len({len(item.rollout.response_ids) for item in result.grouped_rollouts}) > 1
    assert tuple(reward.total for reward in result.rewards) == (
        0.8,
        2.0,
        2.0,
        0.8,
        0.0,
        2.0,
        0.0,
        0.0,
    )

    exact_logprobs = tuple(
        result.payload.non_tensor_batch[ACTUAL_RESPONSE_LOGPROBS_FIELD]
    )
    exact_bundles = tuple(
        result.payload.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD]
    )
    replay_tensor_ref = weakref.ref(exact_bundles[0].tensor_payloads[0].tensor)
    current_values_ref = weakref.ref(result.policy_replay.logprobs.current.values)
    objective_loss_ref = weakref.ref(result.objective.loss)
    payload = result.payload
    for row, grouped in enumerate(result.grouped_rollouts):
        bridge = grouped.rollout
        assert exact_logprobs[row] == bridge.response_logprobs
        assert exact_bundles[row] is bridge.replay_bundle
        assert policy_replay.bundle_sha256s[row] == bridge.replay_bundle.bundle_sha256
        for turn, trace in zip(
            bridge.trajectory_payload.assistant_turns,
            bridge.behavior_trace_records,
            strict=True,
        ):
            assert behavior_store.resolve(turn.behavior_trace) == trace
            assert trace.behavior.logprobs

    with result.lifecycle.consume(PolicyBatchMilestone.LOSS_BACKWARD):
        result.objective.loss.backward()
        assert policy_replay.current is not None
        assert policy_replay.current.grad is not None
        mask = result.payload.tensor_batch["response_mask"].bool()
        assert torch.count_nonzero(policy_replay.current.grad[mask]).item() > 0
        assert torch.count_nonzero(policy_replay.current.grad[~mask]).item() == 0
        assert torch.isfinite(result.objective.loss)
    policy_replay.current = None
    with result.lifecycle.consume(PolicyBatchMilestone.OPTIMIZER_STEP):
        pass
    with result.lifecycle.consume(PolicyBatchMilestone.ZERO_STALENESS_BARRIER):
        pass
    del bridge, grouped, exact_bundles
    report = result.lifecycle.close()
    assert report.replay_records == 8
    assert report.behavior_traces == sum(
        len(trajectory.assistant_turns) for trajectory in result.trajectories
    )
    assert report.transient_owners == 1
    assert payload.sidecars_released
    assert not payload.non_tensor_batch
    assert result.released
    assert result.replay_bundle_sha256s == tuple(replay_finalizer.records)
    for name in ("grouped_rollouts", "payload", "policy_replay", "objective"):
        with pytest.raises(RuntimeError, match="transient state has been released"):
            getattr(result, name)
    gc.collect()
    assert replay_tensor_ref() is None
    assert current_values_ref() is None
    assert objective_loss_ref() is None


def _trajectory_identity(index: int) -> TrajectoryIdentity:
    return TrajectoryIdentity(
        run_id="policy-pilot-runtime",
        sample_id="shared-prompt",
        rollout_index=index,
        group_id="runtime-group",
    )
