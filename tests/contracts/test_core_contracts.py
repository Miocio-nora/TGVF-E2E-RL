from __future__ import annotations

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    BehaviorLogProbBlock,
    LogProbMeasurement,
    SamplingIdentity,
)


SHA = "0" * 64


def test_sha_identity_is_fail_closed() -> None:
    with pytest.raises(ValueError):
        ArtifactIdentity("project", "artifact", "v1", "not-a-digest")


def test_behavior_logprobs_must_align() -> None:
    sampling = SamplingIdentity(
        policy_version=PolicyVersion("run", 0, SHA),
        backend="vllm",
        backend_version="test",
        seed=7,
        rng_state_sha256=SHA,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    with pytest.raises(ValueError):
        BehaviorLogProbBlock((1, 2), (10,), (-0.1, -0.2), sampling)


def test_only_vllm_sampling_identity_is_accepted() -> None:
    with pytest.raises(ValueError, match="vLLM"):
        SamplingIdentity(
            policy_version=PolicyVersion("run", 0, SHA),
            backend="other",
            backend_version="test",
            seed=7,
            rng_state_sha256=SHA,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.RAW_MODEL,
            asynchronous_staleness_steps=0,
        )


def test_greedy_behavior_requires_point_mass_logprobability() -> None:
    sampling = SamplingIdentity(
        policy_version=PolicyVersion("run", 0, "a" * 64),
        backend="vllm",
        backend_version="0.12.0",
        seed=3,
        rng_state_sha256="b" * 64,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    with pytest.raises(ValueError, match="point mass"):
        BehaviorLogProbBlock((1,), (7,), (-0.5,), sampling)
    assert BehaviorLogProbBlock((1,), (7,), (0.0,), sampling).logprobs == (0.0,)
