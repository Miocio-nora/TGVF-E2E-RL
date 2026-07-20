from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion
from tgvf_rl.framework.verl import (
    EXACT_RESPONSE_IDS_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    to_verl_data_proto,
    validate_data_proto_integrity,
)
from tgvf_rl.objectives import (
    LogProbSource,
    PolicyLogProbSet,
    ReferenceKLEstimator,
    RoleLogProbs,
    compute_grpo_loss,
    compute_group_advantages,
    policy_pilot_v1_grpo_spec,
)
from tgvf_rl.policy import (
    PILOT_EXACT_GROUP_UID_FIELD,
    PILOT_EXACT_REWARD_FIELD,
    VERL_GRPO_GROUP_UID_FIELD,
    PilotGroupedRollout,
    materialize_policy_pilot_group_batch,
)
from tests.framework.test_verl_bridges import _FakeDataProto, _record


SHA = "9" * 64


def _entries() -> tuple[PilotGroupedRollout, ...]:
    call_counts = (0, 1, 2, 0, 1, 2, 0, 2)
    rewards = (-0.2, 0.0, 0.8, 1.2, 2.0, 0.8, 0.0, 2.0)
    return tuple(
        PilotGroupedRollout(
            "group",
            _record(
                suffix=index,
                tool_call_count=call_count,
                prompt_ids=(1, 2),
                reward_score=rewards[index],
            ),
        )
        for index, call_count in enumerate(call_counts)
    )


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
        policy_version=PolicyVersion("pilot-batch", step, digit * 64),
        source=(
            LogProbSource.ROLLOUT_RECORDED
            if role is ComponentRole.BEHAVIOR
            else LogProbSource.DETERMINISTIC_REPLAY
        ),
        sampling_transform_sha256=SHA,
    )


def test_mixed_n8_group_materialization_advantages_and_backward() -> None:
    entries = _entries()
    payload = materialize_policy_pilot_group_batch(entries, pad_token_id=99)
    data = to_verl_data_proto(payload, data_proto_cls=_FakeDataProto)
    integrity = validate_data_proto_integrity(data)

    assert data.batch["responses"].shape == (8, 11)
    assert tuple(data.non_tensor_batch[VERL_GRPO_GROUP_UID_FIELD]) == ("group",) * 8
    assert tuple(data.non_tensor_batch[PILOT_EXACT_GROUP_UID_FIELD]) == (
        "group",
    ) * 8
    rewards = torch.tensor(
        tuple(data.non_tensor_batch[PILOT_EXACT_REWARD_FIELD]), dtype=torch.float64
    )
    torch.testing.assert_close(
        data.batch["rm_scores"].sum(dim=-1).to(torch.float64), rewards
    )
    for row_index, entry in enumerate(entries):
        mask = data.batch["response_mask"][row_index].bool()
        last_policy_position = int(torch.nonzero(mask)[-1].item())
        nonzero_positions = torch.nonzero(data.batch["rm_scores"][row_index]).flatten()
        if entry.rollout.reward_score == 0.0:
            assert nonzero_positions.numel() == 0
        else:
            assert tuple(nonzero_positions.tolist()) == (last_policy_position,)
        assert tuple(data.non_tensor_batch[EXACT_RESPONSE_IDS_FIELD][row_index]) == (
            entry.rollout.response_ids
        )
        assert (
            data.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD][row_index]
            is entry.rollout.replay_bundle
        )
        assert integrity.actual_response_logprobs[row_index] == (
            entry.rollout.response_logprobs
        )

    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    group_ids = torch.zeros(8, dtype=torch.int64)
    advantages = compute_group_advantages(rewards, group_ids, spec)
    expected = (rewards - rewards.mean()) / (
        rewards.std(correction=1) + spec.group_std_epsilon
    )
    torch.testing.assert_close(advantages, expected)
    equal_advantages = compute_group_advantages(
        torch.full((8,), 0.8, dtype=torch.float64), group_ids, spec
    )
    assert torch.equal(equal_advantages, torch.zeros_like(equal_advantages))

    policy_mask = data.batch["response_mask"].bool()
    broadcast = advantages[:, None] * policy_mask
    assert torch.count_nonzero(broadcast[~policy_mask]).item() == 0

    behavior = data.batch["rollout_log_probs"].to(torch.float64).clone()
    current = behavior.clone().requires_grad_(True)
    policy = PolicyLogProbSet(
        behavior=_role(ComponentRole.BEHAVIOR, behavior, step=0, digit="0"),
        proximal_old=_role(
            ComponentRole.PROXIMAL_OLD,
            behavior.clone(),
            step=0,
            digit="1",
        ),
        current=_role(ComponentRole.CURRENT, current, step=1, digit="2"),
        reference=_role(
            ComponentRole.REFERENCE,
            behavior.clone(),
            step=0,
            digit="3",
        ),
        policy_sampled_mask=policy_mask,
    )
    result = compute_grpo_loss(spec, policy, rewards, group_ids)
    result.loss.backward()
    assert current.grad is not None
    assert torch.count_nonzero(current.grad[~policy_mask]).item() == 0
    assert torch.count_nonzero(current.grad[policy_mask]).item() > 0


def test_group_materializer_fails_closed_without_filtering_rows() -> None:
    entries = _entries()
    with pytest.raises(ValueError, match="exactly 8"):
        materialize_policy_pilot_group_batch(entries[:-1], pad_token_id=99)
    with pytest.raises(ValueError, match="duplicate trajectory"):
        materialize_policy_pilot_group_batch((entries[0],) * 8, pad_token_id=99)

    missing_reward = PilotGroupedRollout(
        "group", _record(suffix=20, tool_call_count=0, reward_score=None)
    )
    with pytest.raises(ValueError, match="finite reward_score"):
        materialize_policy_pilot_group_batch(
            entries[:-1] + (missing_reward,), pad_token_id=99
        )

    different_prompt = PilotGroupedRollout(
        "group",
        _record(
            suffix=21,
            tool_call_count=0,
            prompt_ids=(7,),
            reward_score=0.8,
        ),
    )
    with pytest.raises(ValueError, match="different exact prompt IDs"):
        materialize_policy_pilot_group_batch(
            entries[:-1] + (different_prompt,), pad_token_id=99
        )

    with pytest.raises(ValueError, match="trajectory group identity"):
        replace(entries[0], group_uid="different-group")
