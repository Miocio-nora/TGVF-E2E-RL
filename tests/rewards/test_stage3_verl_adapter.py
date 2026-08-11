from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tests.framework.test_verl_bridges import _record

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityLabelBinding,
    TGVFToolUtilityRuntimeBinding,
)
from tgvf_rl.rewards.context import reward_context_from_trajectory
from tgvf_rl.framework.verl.reward_bridge import VerlRewardedAgentLoopOutputBuilder
from tgvf_rl.framework.verl.rollout_bridge import rollout_provenance_checksum
from tgvf_rl.rewards.schema import (
    AnswerTaskKind,
    AnswerVerificationResult,
)
from tgvf_rl.rewards.stage3_shaped import QualityJudgeScore
from tgvf_rl.rewards.stage3_verl_adapter import (
    Stage3ShapedRewardSpec,
    Stage3VerlTrajectoryRewardScorer,
    Stage3VisualJudgeSampleFailure,
    Stage3VisualQualityJudgement,
)
from tgvf_rl.trajectories.schema import TrajectoryStop, trajectory_checksum


SHA = "a" * 64


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("stage3-test", name, "v1", digit * 64)


class _ContextProvider:
    def build(self, *, request, trajectory):
        assert request.identity == trajectory.identity
        return reward_context_from_trajectory(
            trajectory,
            question="Which option is correct?",
            expected_answer="fixture answer",
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )


class _AnswerVerifier:
    def __init__(
        self,
        identity: ArtifactIdentity,
        *,
        judge_identity: ArtifactIdentity | None = None,
        route: str = "fixture-rule",
    ) -> None:
        self.identity = identity
        self.route = route
        if judge_identity is not None:
            self.judge_model_identity = judge_identity

    def verify(self, context):
        return AnswerVerificationResult(
            correct=context.candidate_answer == context.expected_answer,
            route=self.route,
            evidence="fixture exact answer",
            verifier_identity=self.identity,
        )


class _VisualJudge:
    def __init__(self, identity: ArtifactIdentity, *, mode: str = "success") -> None:
        self.identity = identity
        self.mode = mode
        self.calls = 0

    def judge(self, *, request, trajectory, context):
        del request
        self.calls += 1
        if self.mode == "sample-failure":
            raise Stage3VisualJudgeSampleFailure("transport")
        sample_id = context.sample_id if self.mode != "wrong-sample" else "wrong"
        return Stage3VisualQualityJudgement(
            trajectory_id=trajectory.identity.canonical_id,
            sample_id=sample_id,
            successful_observation_count=context.successful_tgvf_observation_count,
            focus_score=QualityJudgeScore.PASS,
            grounding_score=QualityJudgeScore.PASS,
            judge_identity=self.identity,
        )


def _scorer(
    *,
    mode: str = "success",
    answer_result_identity: ArtifactIdentity | None = None,
    answer_judge_identity: ArtifactIdentity | None = None,
    answer_route: str = "fixture-rule",
    visual_quality_enabled: bool = True,
):
    answer_identity = _identity("answer", "1")
    visual_identity = _identity("visual", "2")
    spec = Stage3ShapedRewardSpec(
        pipeline_identity=_identity("pipeline", "3"),
        answer_verifier_identity=answer_identity,
        visual_judge_identity=(visual_identity if visual_quality_enabled else None),
        tool_utility_sidecar_sha256="4" * 64,
        tool_utility_manifest_sha256="5" * 64,
        visual_quality_enabled=visual_quality_enabled,
    )
    label = TGVFToolUtilityLabelBinding(
        sample_id="sample-0",
        training_index=0,
        utility_label="needed",
        confidence=0.5,
        row_sha256="6" * 64,
    )
    utility = TGVFToolUtilityRuntimeBinding(
        sidecar_path=Path("/fixture/tool-utility.jsonl"),
        sidecar_sha256="4" * 64,
        manifest_path=Path("/fixture/manifest.json"),
        manifest_sha256="5" * 64,
        dataset_iteration_identity_sha256="7" * 64,
        labels=MappingProxyType({label.sample_id: label}),
    )
    judge = _VisualJudge(visual_identity, mode=mode)
    return (
        Stage3VerlTrajectoryRewardScorer(
            spec=spec,
            answer_verifier=_AnswerVerifier(
                answer_result_identity or answer_identity,
                judge_identity=answer_judge_identity,
                route=answer_route,
            ),
            context_provider=_ContextProvider(),
            tool_utility=utility,
            visual_quality_judge=(judge if visual_quality_enabled else None),
        ),
        judge,
    )


def test_stage3_runtime_bridge_emits_exact_five_component_reward() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    scorer, judge = _scorer()

    scored = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    assert scored.total == pytest.approx(4.5)
    assert scored.raw_components == (
        ("answer", 2.0),
        ("tool", 0.5),
        ("focus", 1.0),
        ("grounding", 1.0),
        ("protocol", 0.0),
    )
    assert scored.result.quality_judge_covered is True
    assert scored.reward_extra_info()["stage3_grounding_reward"] == 1.0
    assert judge.calls == 1


def test_stage3_accepts_configured_answer_judge_fallback_identity() -> None:
    judge_identity = _identity("answer-judge", "8")
    scorer, _ = _scorer(
        answer_result_identity=judge_identity,
        answer_judge_identity=judge_identity,
        answer_route="qwen2.5_72b_semantic_fallback",
    )
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )

    scored = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    assert scored.answer_verification.verifier_identity == judge_identity


def test_stage3_rejects_unbound_answer_verifier_identity() -> None:
    scorer, _ = _scorer()
    scorer.answer_verifier = _AnswerVerifier(_identity("unknown-answer", "9"))
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )

    with pytest.raises(IdentityMismatchError, match="route and configured identity"):
        scorer.score(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )


def test_only_successful_observation_counts_as_tool_use() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        observations=(),
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    scorer, judge = _scorer()

    scored = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    assert dict(scored.raw_components)["answer"] == 0.0
    assert dict(scored.raw_components)["tool"] == -1.0
    assert scored.result.answer_gated is True
    assert judge.calls == 0


def test_sample_local_visual_failure_records_zero_and_coverage() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    scorer, _ = _scorer(mode="sample-failure")

    scored = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    components = dict(scored.raw_components)
    assert components["focus"] == 0.0
    assert components["grounding"] == 0.0
    assert scored.result.quality_judge_failure == "transport"
    assert scored.reward_extra_info()["stage3_quality_judge_failed"] == 1


def test_disabled_visual_quality_never_calls_provider() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    scorer, judge = _scorer(visual_quality_enabled=False)

    scored = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    assert scored.total == pytest.approx(2.5)
    assert scored.result.quality_judge_applicable is False
    assert scored.reward_extra_info()["visual_judge_calls"] == 0
    assert judge.calls == 0


def test_visual_result_identity_error_fails_closed() -> None:
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    scorer, _ = _scorer(mode="wrong-sample")

    with pytest.raises(IdentityMismatchError, match="judgement identity"):
        scorer.score(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )


def test_visual_failure_type_cannot_downgrade_identity_errors() -> None:
    with pytest.raises(ValueError, match="transport or malformed_output"):
        Stage3VisualJudgeSampleFailure("identity_mismatch")


def test_stage3_reward_crosses_public_agent_loop_output_with_five_components() -> None:
    pytest.importorskip("verl")
    from verl.experimental.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    base_record = _record(tool_call_count=1, reward_score=4.5)
    trajectory = replace(
        base_record.trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    exact_trajectory_sha256 = trajectory_checksum(trajectory)
    bridge_record = replace(
        base_record,
        trajectory_payload=trajectory,
        trajectory_sha256=exact_trajectory_sha256,
        rollout_provenance_sha256=rollout_provenance_checksum(
            trajectory_id=base_record.trajectory_id,
            trajectory_sha256=exact_trajectory_sha256,
            replay_handle=base_record.replay_handle,
            replay_bundle_sha256=base_record.replay_bundle.bundle_sha256,
            observation_handles=base_record.exact_observation_handles,
            behavior_trace_handles=base_record.behavior_trace_handles,
            token_ownership_sha256=base_record.token_ownership_sha256,
        ),
    )
    request = SimpleNamespace(identity=trajectory.identity)
    scorer, _ = _scorer()

    class Finalizer:
        def finalize(self, *, request, trajectory, reward):
            assert request.identity == trajectory.identity
            assert reward.total == pytest.approx(4.5)
            return bridge_record

    output = VerlRewardedAgentLoopOutputBuilder(
        request=request,
        scorer=scorer,
        finalizer=Finalizer(),
        metrics_factory=lambda trajectory, scored: AgentLoopMetrics(),
        agent_loop_output_cls=AgentLoopOutput,
    )(trajectory)

    assert output.reward_score == pytest.approx(4.5)
    assert output.extra_fields["tgvf_reward_components"] == (
        ("answer", 2.0),
        ("tool", 0.5),
        ("focus", 1.0),
        ("grounding", 1.0),
        ("protocol", 0.0),
    )
    assert output.extra_fields["reward_extra_info"]["stage3_quality_judge_covered"] == 1
