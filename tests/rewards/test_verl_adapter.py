from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.framework.test_verl_bridges import _record

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import JudgeUsage
from tgvf_rl.framework.verl.reward_bridge import (
    VerlRewardedAgentLoopOutputBuilder,
)
from tgvf_rl.rewards import (
    AnswerTaskKind,
    PilotRewardSpec,
    PilotVerlTrajectoryRewardScorer,
)
from tgvf_rl.rewards.context import reward_context_from_trajectory
from tgvf_rl.rewards.pipeline import PilotRewardPipeline
from tgvf_rl.rewards.schema import (
    NormalizationSpec,
    PILOT_REWARD_DEEPEYES_MATH_WEIGHTS,
    PILOT_REWARD_EQUATION_ANSWER_PRIMARY,
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    PILOT_REWARD_EQUATION_LEGACY,
    PILOT_REWARD_LEGACY_WEIGHTS,
)
from tgvf_rl.rewards.verifiers import RuleFirstAnswerVerifier
from tgvf_rl.rewards.verl_adapter import (
    PILOT_VERL_ANSWER_ROUTE_FIELD,
    PILOT_VERL_JUDGE_USAGE_FIELD,
    PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD,
    PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD,
)
from tgvf_rl.trajectories.schema import TrajectoryStop


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("pilot-verl-reward-test", name, "v1", digit * 64)


class _ExplodingJudge:
    calls = 0

    def judge(self, request):
        del request
        self.calls += 1
        raise AssertionError("MCQ exact reward must not invoke a judge")


class _ContextProvider:
    def __init__(
        self,
        *,
        data_source: str | None = None,
        task_kind: AnswerTaskKind = AnswerTaskKind.MULTIPLE_CHOICE,
    ) -> None:
        self.data_source = data_source
        self.task_kind = task_kind

    def build(self, *, request, trajectory):
        assert request.identity == trajectory.identity
        return reward_context_from_trajectory(
            trajectory,
            question="Which option is correct?",
            expected_answer="fixture answer",
            task_kind=self.task_kind,
            data_source=self.data_source,
        )


def _scorer(
    *,
    conditional_tool_weight: float = 1.2,
    deepeyes_source_aware: bool = False,
    data_source: str | None = None,
    task_kind: AnswerTaskKind = AnswerTaskKind.MULTIPLE_CHOICE,
) -> tuple[PilotVerlTrajectoryRewardScorer, _ExplodingJudge]:
    spec = PilotRewardSpec(
        pipeline_identity=_identity("pipeline", "1"),
        answer_verifier_identity=_identity("answer", "2"),
        format_verifier_identity=_identity("format", "3"),
        tool_verifier_identity=_identity("tool", "4"),
        conditional_tool_weight=conditional_tool_weight,
        deepeyes_source_aware=deepeyes_source_aware,
    )
    judge = _ExplodingJudge()
    verifier = RuleFirstAnswerVerifier(
        rule_identity=spec.answer_verifier_identity,
        normalization=NormalizationSpec(True, True, True),
        judge=judge,
        judge_prompt_identity=_identity("judge-prompt", "5"),
        judge_model_identity=_identity("judge-model", "6"),
        judge_service_identity=_identity("judge-service", "7"),
        judge_sampling_identity=_identity("judge-sampling", "8"),
        judge_calibration_identity=_identity("judge-calibration", "9"),
    )
    return (
        PilotVerlTrajectoryRewardScorer(
            pipeline=PilotRewardPipeline(spec, verifier),
            context_provider=_ContextProvider(
                data_source=data_source,
                task_kind=task_kind,
            ),
        ),
        judge,
    )


def test_mcq_trajectory_reward_is_exact_and_never_calls_judge() -> None:
    bridge_record = _record(tool_call_count=0, reward_score=0.8)
    trajectory = bridge_record.trajectory_payload
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, judge = _scorer()

    scored = scorer.score(request=request, trajectory=trajectory)

    assert scored.total == pytest.approx(0.8)
    assert scored.group_uid == trajectory.identity.group_id
    assert scored.trajectory_id == trajectory.identity.canonical_id
    assert scored.raw_components == (
        ("answer_reward", 1.0),
        ("format_reward", 0.0),
        ("conditional_tool_reward", 0.0),
    )
    assert scored.result.components[0].evidence.startswith(
        "route=multiple_choice_rule;"
    )
    assert scored.reward_extra_info()["tgvf_exact_trajectory_reward"] == 0.8
    assert scored.equation_route == PILOT_REWARD_EQUATION_LEGACY
    assert scored.applied_weights == PILOT_REWARD_LEGACY_WEIGHTS
    assert (
        scored.reward_sidecars()[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD]
        == PILOT_REWARD_EQUATION_LEGACY
    )
    assert (
        scored.reward_sidecars()[PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD]
        == PILOT_REWARD_LEGACY_WEIGHTS
    )
    assert judge.calls == 0

    usage = JudgeUsage(201, 17, 218, 0.00007916)
    verification = replace(
        scored.result.answer_verification,
        route="qwen2.5_72b_semantic_fallback",
        judge_usage=usage,
    )
    with_usage = replace(
        scored,
        result=replace(scored.result, answer_verification=verification),
    )
    sidecars = with_usage.reward_sidecars()
    assert sidecars[PILOT_VERL_ANSWER_ROUTE_FIELD] == ("qwen2.5_72b_semantic_fallback")
    assert sidecars[PILOT_VERL_JUDGE_USAGE_FIELD] == (
        201,
        17,
        218,
        pytest.approx(0.00007916),
    )

    with pytest.raises(ValueError, match="identities differ"):
        scorer.score(
            request=SimpleNamespace(identity=object()),
            trajectory=trajectory,
        )


def test_answer_primary_profile_crosses_verl_reward_scorer() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, judge = _scorer(conditional_tool_weight=0.2)

    scored = scorer.score(request=request, trajectory=trajectory)

    assert scored.total == pytest.approx(1.0)
    assert tuple(
        component.weighted_score for component in scored.result.components
    ) == pytest.approx((0.8, 0.0, 0.2))
    assert scored.equation_route == PILOT_REWARD_EQUATION_ANSWER_PRIMARY
    assert judge.calls == 0


@pytest.mark.parametrize(
    (
        "data_source",
        "task_kind",
        "expected_route",
        "expected_weights",
        "expected_raw_tool",
        "expected_total",
    ),
    (
        pytest.param(
            "vstar",
            AnswerTaskKind.MULTIPLE_CHOICE,
            PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
            PILOT_REWARD_LEGACY_WEIGHTS,
            1.0,
            2.0,
            id="visual",
        ),
        pytest.param(
            "thinklite",
            AnswerTaskKind.OPEN_VQA,
            PILOT_REWARD_EQUATION_DEEPEYES_MATH,
            PILOT_REWARD_DEEPEYES_MATH_WEIGHTS,
            0.0,
            1.2,
            id="math",
        ),
    ),
)
def test_deepeyes_source_route_and_weights_cross_verl_reward_sidecars(
    data_source: str,
    task_kind: AnswerTaskKind,
    expected_route: str,
    expected_weights: tuple[float, float, float],
    expected_raw_tool: float,
    expected_total: float,
) -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, judge = _scorer(
        deepeyes_source_aware=True,
        data_source=data_source,
        task_kind=task_kind,
    )

    scored = scorer.score(request=request, trajectory=trajectory)
    sidecars = scored.reward_sidecars()

    assert scored.total == pytest.approx(expected_total)
    assert scored.raw_components[-1] == (
        "conditional_tool_reward",
        expected_raw_tool,
    )
    assert scored.equation_route == expected_route
    assert scored.applied_weights == expected_weights
    assert sidecars[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD] == expected_route
    assert sidecars[PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD] == expected_weights
    assert judge.calls == 0


@pytest.mark.parametrize(
    "record_kwargs",
    (
        pytest.param(
            {"tool_call_count": 0, "invalid_format": True},
            id="invalid-format",
        ),
        pytest.param({"tool_call_count": 1}, id="max-tokens"),
    ),
)
def test_unanswered_mcq_trajectory_is_retained_and_never_calls_judge(
    record_kwargs: dict[str, object],
) -> None:
    bridge_record = _record(**record_kwargs)
    trajectory = bridge_record.trajectory_payload
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, judge = _scorer()

    scored = scorer.score(request=request, trajectory=trajectory)

    assert scored.trajectory_id == trajectory.identity.canonical_id
    assert scored.total == pytest.approx(-0.2)
    assert scored.raw_components == (
        ("answer_reward", 0.0),
        ("format_reward", -1.0),
        ("conditional_tool_reward", 0.0),
    )
    assert scored.result.components[0].evidence.startswith(
        "route=missing_final_answer;"
    )
    assert judge.calls == 0


def test_rewarded_output_builder_sets_public_reward_score_and_exact_sidecars() -> None:
    pytest.importorskip("verl")
    from verl.experimental.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    bridge_record = _record(tool_call_count=0, reward_score=0.8)
    trajectory = bridge_record.trajectory_payload
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, judge = _scorer()

    class Finalizer:
        def finalize(self, *, request, trajectory, reward):
            assert request.identity == trajectory.identity
            assert reward.total == pytest.approx(0.8)
            return bridge_record

    builder = VerlRewardedAgentLoopOutputBuilder(
        request=request,
        scorer=scorer,
        finalizer=Finalizer(),
        metrics_factory=lambda trajectory, scored: AgentLoopMetrics(),
        agent_loop_output_cls=AgentLoopOutput,
    )
    output = builder(trajectory)
    async_output = asyncio.run(builder.build_async(trajectory))
    public = output.as_dict()

    assert output.reward_score == pytest.approx(0.8)
    assert public["rm_scores"].sum().item() == pytest.approx(0.8)
    assert output.extra_fields["tgvf_exact_group_uid"] == (trajectory.identity.group_id)
    assert output.extra_fields["tgvf_exact_trajectory_reward"] == pytest.approx(0.8)
    assert output.extra_fields["tgvf_reward_trajectory_id"] == (
        trajectory.identity.canonical_id
    )
    assert async_output.reward_score == pytest.approx(output.reward_score)
    assert async_output.extra_fields == output.extra_fields
    assert judge.calls == 0
