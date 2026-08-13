from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tests.rewards.test_deepeyes_async_tgvf import (
    FakeAsyncJudge,
    _outcome,
    _trajectory,
)

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityLabelBinding,
    TGVFToolUtilityRuntimeBinding,
)
from tgvf_rl.rewards.deepeyes_async_tgvf import (
    DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY,
    AsyncDeepEyesTGVFTrajectoryRewardScorer,
)
from tgvf_rl.rewards.stage3_async_tgvf import (
    AsyncStage3ShapedTGVFTrajectoryRewardScorer,
)
from tgvf_rl.rewards.stage3_verl_adapter import Stage3ShapedRewardSpec
from tgvf_rl.rewards.stage3_verl_adapter import Stage3VisualQualityJudgement
from tgvf_rl.rewards.stage3_shaped import QualityJudgeScore


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("async-stage3-test", name, "v1", digit * 64)


def _scorer(
    *, tool_utility_reward_enabled: bool = True
) -> AsyncStage3ShapedTGVFTrajectoryRewardScorer:
    label = TGVFToolUtilityLabelBinding(
        sample_id="sample",
        training_index=0,
        utility_label="needed",
        confidence=0.5,
        row_sha256="4" * 64,
    )
    utility = TGVFToolUtilityRuntimeBinding(
        sidecar_path=Path("/fixture/tool-utility.jsonl"),
        sidecar_sha256="5" * 64,
        manifest_path=Path("/fixture/manifest.json"),
        manifest_sha256="6" * 64,
        dataset_iteration_identity_sha256="7" * 64,
        labels=MappingProxyType({label.sample_id: label}),
    )
    spec = Stage3ShapedRewardSpec(
        pipeline_identity=_identity("pipeline", "1"),
        answer_verifier_identity=DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY,
        visual_judge_identity=None,
        tool_utility_sidecar_sha256=(
            utility.sidecar_sha256 if tool_utility_reward_enabled else None
        ),
        tool_utility_manifest_sha256=(
            utility.manifest_sha256 if tool_utility_reward_enabled else None
        ),
        visual_quality_enabled=False,
        tool_utility_reward_enabled=tool_utility_reward_enabled,
    )
    answer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="What color?",
        reference_answer="blue",
        task_kind="open",
        data_source="vstar",
        judge_transport=FakeAsyncJudge(_outcome(True)),
    )
    return AsyncStage3ShapedTGVFTrajectoryRewardScorer(
        answer_scorer=answer,
        spec=spec,
        tool_utility=utility if tool_utility_reward_enabled else None,
    )


def test_async_shaped_reuses_answer_judge_and_disables_visual_quality() -> None:
    trajectory = _trajectory(
        "<think>see</think>blue",
        successful_tgvf=True,
    )
    scorer = _scorer()

    result = asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )

    assert result.total == pytest.approx(2.5)
    assert dict(result.raw_components) == {
        "answer": 2.0,
        "tool": 0.5,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": 0.0,
    }
    assert result.result.quality_judge_applicable is False
    assert result.reward_extra_info()["visual_judge_calls"] == 0


def test_async_shaped_gates_needed_answer_without_successful_tool() -> None:
    trajectory = _trajectory("<think>see</think>blue")
    scorer = _scorer()

    result = asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )

    assert result.total == pytest.approx(-1.0)
    assert result.result.answer_gated is True


def test_async_tfree_scores_answer_without_sidecar_or_gate() -> None:
    trajectory = _trajectory("<think>see</think>blue")
    scorer = _scorer(tool_utility_reward_enabled=False)

    result = asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )

    assert result.total == pytest.approx(2.0)
    assert result.result.answer_gated is False
    assert result.tool_label is None
    assert dict(result.raw_components) == {
        "answer": 2.0,
        "tool": 0.0,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": 0.0,
    }


class _FakeAsyncVisualJudge:
    def __init__(self, identity: ArtifactIdentity) -> None:
        self.identity = identity
        self.calls = 0

    async def judge_async(self, *, request, trajectory, context):
        self.calls += 1
        return Stage3VisualQualityJudgement(
            trajectory_id=trajectory.identity.canonical_id,
            sample_id=context.sample_id,
            successful_observation_count=(
                context.successful_tgvf_observation_count
            ),
            focus_score=QualityJudgeScore.PARTIAL,
            grounding_score=QualityJudgeScore.PASS,
            judge_identity=self.identity,
        )


def _visual_tfree_scorer():
    visual_identity = _identity("visual", "9")
    visual = _FakeAsyncVisualJudge(visual_identity)
    spec = Stage3ShapedRewardSpec(
        pipeline_identity=_identity("pipeline-visual", "8"),
        answer_verifier_identity=DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY,
        visual_judge_identity=visual_identity,
        tool_utility_sidecar_sha256=None,
        tool_utility_manifest_sha256=None,
        visual_quality_enabled=True,
        tool_utility_reward_enabled=False,
    )
    answer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="What color?",
        reference_answer="blue",
        task_kind="open",
        data_source="vstar",
        judge_transport=FakeAsyncJudge(_outcome(True)),
    )
    return (
        AsyncStage3ShapedTGVFTrajectoryRewardScorer(
            answer_scorer=answer,
            spec=spec,
            tool_utility=None,
            visual_quality_judge=visual,
        ),
        visual,
    )


def test_async_tfree_visual_scores_focus_and_grounding() -> None:
    trajectory = _trajectory("<think>see</think>blue", successful_tgvf=True)
    scorer, visual = _visual_tfree_scorer()

    result = asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )

    assert result.total == pytest.approx(3.5)
    assert visual.calls == 1
    assert dict(result.raw_components) == {
        "answer": 2.0,
        "tool": 0.0,
        "focus": 0.5,
        "grounding": 1.0,
        "protocol": 0.0,
    }
    assert result.result.quality_judge_covered is True


def test_async_visual_is_not_called_without_successful_observation() -> None:
    trajectory = _trajectory("<think>see</think>blue")
    scorer, visual = _visual_tfree_scorer()

    result = asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )

    assert result.total == pytest.approx(2.0)
    assert visual.calls == 0
    assert result.result.quality_judge_applicable is False
