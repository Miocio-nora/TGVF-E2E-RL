from __future__ import annotations

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    ToolCallRecord,
    ToolObservationRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tgvf_rl.trajectories.validation import TrajectoryValidator
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tests.support import SHA0, populated_observation_store, policy_version


def _trajectory(
    *, sampled_indices: tuple[int, ...] = (1, 2)
) -> tuple[object, BehaviorTraceStore, TrajectoryRecord]:
    store, observation_handle = populated_observation_store()
    version = policy_version()
    sampling = SamplingIdentity(
        policy_version=version,
        backend="vllm",
        backend_version="fixture",
        seed=1,
        rng_state_sha256=SHA0,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    tokens = OwnedTokenSequence(
        token_ids=(151667, 101, 102),
        ownership=(
            TokenOwnership.TEMPLATE,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.POLICY_SAMPLED,
        ),
    )
    behavior_store = BehaviorTraceStore()
    behavior_handle = VLLMBehaviorRecorder(behavior_store).record(
        trajectory_id="smoke/sample/0/group",
        assistant_turn_index=0,
        tokens=tokens,
        actual_sampled_logprobs=(-0.2, -0.3),
        sampling=sampling,
        behavior_policy=version,
        backend_request_sha256=SHA0,
        backend_response_sha256=SHA0,
    )
    turn_tokens = tokens
    if sampled_indices != (1, 2):
        turn_tokens = OwnedTokenSequence(
            token_ids=tokens.token_ids,
            ownership=(
                TokenOwnership.POLICY_SAMPLED,
                TokenOwnership.POLICY_SAMPLED,
                TokenOwnership.TEMPLATE,
            ),
        )
    turn = AssistantTurnRecord(
        0,
        "<think>inspect</think><tool_call>...</tool_call>",
        turn_tokens,
        behavior_handle,
        TokenSpan(0, 3),
        True,
    )
    trajectory = TrajectoryRecord(
        "trajectory-v1",
        TrajectoryIdentity("smoke", "sample", 0, "group"),
        store.resolve_record(observation_handle).model,
        version,
        (turn,),
        (
            ToolCallRecord(
                0, 0, "tgvf_focus_tool", "red label", TokenSpan(1, 3), (10, 19), "raw"
            ),
        ),
        (ToolObservationRecord(0, observation_handle, (151665, 151666)),),
        None,
        TrajectoryStop.MAX_TOKENS,
    )
    return store, behavior_store, trajectory


def test_actual_behavior_logprobs_align_with_policy_owned_tokens() -> None:
    store, behavior_store, trajectory = _trajectory()
    TrajectoryValidator(store, behavior_store).validate(trajectory)


def test_old_logprob_alignment_cannot_be_fabricated() -> None:
    store, behavior_store, trajectory = _trajectory(sampled_indices=(0, 1))
    with pytest.raises(ReplayMismatchError, match="content-addressed"):
        TrajectoryValidator(store, behavior_store).validate(trajectory)


def test_vllm_recorder_preserves_nontrivial_post_transform_logprob_oracle() -> None:
    version = policy_version()
    sampling = SamplingIdentity(
        policy_version=version,
        backend="vllm",
        backend_version="fixture-processed-logprobs",
        seed=9,
        rng_state_sha256=SHA0,
        temperature=0.5,
        top_p=1.0,
        top_k=2,
        min_p=0.0,
        repetition_penalty=1.25,
        presence_penalty=0.25,
        frequency_penalty=0.0,
        logit_processors=("fixture_bias_v1",),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    raw_logits = torch.tensor([2.0, 1.5, 0.5, -1.0], dtype=torch.float64)
    processed = raw_logits.clone()
    processed[0] = processed[0] / sampling.repetition_penalty
    processed[0] -= sampling.presence_penalty
    processed[1] += 0.125  # fixture_bias_v1
    processed /= sampling.temperature
    threshold = torch.topk(processed, sampling.top_k).values[-1]
    processed[processed < threshold] = -torch.inf
    oracle = float(torch.log_softmax(processed, dim=-1)[1].item())

    tokens = OwnedTokenSequence((1,), (TokenOwnership.POLICY_SAMPLED,))
    store = BehaviorTraceStore()
    handle = VLLMBehaviorRecorder(store).record(
        trajectory_id="smoke/oracle/0/group",
        assistant_turn_index=0,
        tokens=tokens,
        actual_sampled_logprobs=(oracle,),
        sampling=sampling,
        behavior_policy=version,
        backend_request_sha256=SHA0,
        backend_response_sha256=SHA0,
    )
    assert store.resolve(handle).behavior.logprobs == (oracle,)
    assert sampling.has_identity_sampling_transforms is False
