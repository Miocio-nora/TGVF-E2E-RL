from __future__ import annotations

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.rewards.pipeline import ExactTextVerifier, RewardPipeline
from tgvf_rl.rewards.schema import (
    NormalizationSpec,
    RewardComponentSpec,
    RewardContext,
    RewardPipelineSpec,
)


def test_explicit_reward_pipeline_is_decomposed_and_deterministic() -> None:
    identity = ArtifactIdentity("smoke", "exact", "v1", "0" * 64)
    spec = RewardPipelineSpec(identity, (RewardComponentSpec("answer", 2.0, identity),))
    pipeline = RewardPipeline(
        spec,
        {"answer": ExactTextVerifier(NormalizationSpec(True, True, True))},
    )
    result = pipeline.score(RewardContext("s", "q", " Blue  label ", "blue label", 1))
    assert result.total == 2.0
    assert result.components[0].raw_score == 1.0
