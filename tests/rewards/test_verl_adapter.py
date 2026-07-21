from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.framework.test_verl_bridges import _record

from tgvf_rl.contracts.identity import ArtifactIdentity
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
from tgvf_rl.rewards.schema import NormalizationSpec
from tgvf_rl.rewards.verifiers import RuleFirstAnswerVerifier


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("pilot-verl-reward-test", name, "v1", digit * 64)


class _ExplodingJudge:
    calls = 0

    def judge(self, request):
        del request
        self.calls += 1
        raise AssertionError("MCQ exact reward must not invoke a judge")


class _ContextProvider:
    def build(self, *, request, trajectory):
        assert request.identity == trajectory.identity
        return reward_context_from_trajectory(
            trajectory,
            question="Which option is correct?",
            expected_answer="fixture answer",
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )


def _scorer() -> tuple[PilotVerlTrajectoryRewardScorer, _ExplodingJudge]:
    spec = PilotRewardSpec(
        pipeline_identity=_identity("pipeline", "1"),
        answer_verifier_identity=_identity("answer", "2"),
        format_verifier_identity=_identity("format", "3"),
        tool_verifier_identity=_identity("tool", "4"),
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
            context_provider=_ContextProvider(),
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
    assert judge.calls == 0

    with pytest.raises(ValueError, match="identities differ"):
        scorer.score(
            request=SimpleNamespace(identity=object()),
            trajectory=trajectory,
        )


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
    public = output.as_dict()

    assert output.reward_score == pytest.approx(0.8)
    assert public["rm_scores"].sum().item() == pytest.approx(0.8)
    assert output.extra_fields["tgvf_exact_group_uid"] == (trajectory.identity.group_id)
    assert output.extra_fields["tgvf_exact_trajectory_reward"] == pytest.approx(0.8)
    assert output.extra_fields["tgvf_reward_trajectory_id"] == (
        trajectory.identity.canonical_id
    )
    assert judge.calls == 0
