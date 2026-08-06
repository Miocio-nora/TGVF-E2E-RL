from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.framework.test_verl_bridges import _record

from tgvf_rl.framework.verl.reward_bridge import (
    POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA,
    POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD,
    bind_policy_pilot_exact_grpo_fields,
    validate_policy_pilot_reward_data_proto,
)
from tgvf_rl.framework.verl.rollout_bridge import (
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
)
from tgvf_rl.objectives import (
    ReferenceKLEstimator,
    compute_group_advantages,
    policy_pilot_v1_grpo_spec,
)
from tgvf_rl.policy.batch import (
    PILOT_EXACT_GROUP_UID_FIELD,
    PILOT_EXACT_REWARD_FIELD,
    PILOT_GROUP_BATCH_SCHEMA_FIELD,
    POLICY_PILOT_V1_GROUP_BATCH_SCHEMA,
)
from tgvf_rl.rewards.verl_adapter import (
    PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
    PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD,
    PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD,
    PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD,
)
from tgvf_rl.rewards.stage3_verl_adapter import (
    STAGE3_VERL_QUALITY_APPLICABLE_FIELD,
    STAGE3_VERL_QUALITY_COVERED_FIELD,
    STAGE3_VERL_QUALITY_FAILURE_FIELD,
    STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD,
    STAGE3_VERL_TOOL_LABEL_FIELD,
    STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD,
    STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD,
    STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD,
)
from tgvf_rl.rewards.schema import (
    PILOT_REWARD_ANSWER_PRIMARY_WEIGHTS,
    PILOT_REWARD_DEEPEYES_MATH_WEIGHTS,
    PILOT_REWARD_EQUATION_ANSWER_PRIMARY,
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    PILOT_REWARD_EQUATION_LEGACY,
    PILOT_REWARD_LEGACY_WEIGHTS,
)
from tgvf_rl.trajectories.schema import TrajectoryIdentity, TrajectoryStop


_PIPELINE_SHA256 = "a" * 64


def _components(total: float) -> tuple[tuple[str, float], ...]:
    raw_by_total = {
        -0.2: (0.0, -1.0, 0.0),
        0.0: (0.0, 0.0, 0.0),
        0.8: (1.0, 0.0, 0.0),
        1.0: (1.0, 0.0, 1.0),
        2.0: (1.0, 0.0, 1.0),
    }
    answer, format_score, tool = raw_by_total[total]
    return (
        ("answer_reward", answer),
        ("format_reward", format_score),
        ("conditional_tool_reward", tool),
    )


def _real_data_proto(
    *,
    incomplete: bool = False,
    tool_weight: float = 1.2,
    group_size: int = 8,
    equation_route: str | None = None,
):
    pytest.importorskip("verl")
    from verl.protocol import DataProto

    correct_with_tool = 0.8 + tool_weight
    if equation_route is None:
        equation_route = {
            1.2: PILOT_REWARD_EQUATION_LEGACY,
            0.2: PILOT_REWARD_EQUATION_ANSWER_PRIMARY,
        }[tool_weight]
    applied_weights = {
        PILOT_REWARD_EQUATION_LEGACY: PILOT_REWARD_LEGACY_WEIGHTS,
        PILOT_REWARD_EQUATION_ANSWER_PRIMARY: PILOT_REWARD_ANSWER_PRIMARY_WEIGHTS,
        PILOT_REWARD_EQUATION_DEEPEYES_VISUAL: PILOT_REWARD_LEGACY_WEIGHTS,
    }[equation_route]
    reward_pattern = [
        0.8,
        0.0,
        -0.2,
        correct_with_tool,
        0.8,
        0.0,
        -0.2,
        correct_with_tool,
    ]
    assert group_size % len(reward_pattern) == 0
    rewards = reward_pattern * (group_size // len(reward_pattern))
    rewards.extend([0.8] * (group_size - 1 if incomplete else group_size))
    trajectories = []
    exact_groups = []
    upstream_groups = []
    components = []
    for row_index, reward in enumerate(rewards):
        group_index = row_index // group_size
        rollout_index = row_index % group_size
        exact_group = f"exact-group-{group_index}"
        upstream_group = f"upstream-group-{group_index}"
        template = _record(suffix=0, tool_call_count=0).trajectory_payload
        trajectory = replace(
            template,
            identity=TrajectoryIdentity(
                run_id="reward-bridge-test",
                sample_id=f"sample-{group_index}",
                rollout_index=rollout_index,
                group_id=exact_group,
            ),
        )
        if reward == -0.2:
            trajectory = replace(
                trajectory,
                assistant_turns=(
                    replace(
                        trajectory.assistant_turns[0],
                        raw_text="unfinished reasoning",
                        think_span=None,
                    ),
                ),
                final_answer=None,
                stop=TrajectoryStop.INVALID_FORMAT,
            )
        trajectories.append(trajectory)
        exact_groups.append(exact_group)
        upstream_groups.append(upstream_group)
        components.append(_components(reward))

    batch_size = len(rewards)
    prompts = torch.tensor(
        [
            [11, 12] if row_index < group_size else [21, 22]
            for row_index in range(batch_size)
        ],
        dtype=torch.int64,
    )
    responses = torch.arange(batch_size * 4, dtype=torch.int64).reshape(batch_size, 4)
    response_mask = torch.tensor([[1, 1, 0, 0]] * batch_size, dtype=torch.int64)
    attention_mask = torch.ones((batch_size, 6), dtype=torch.int64)
    rm_scores = torch.zeros((batch_size, 4), dtype=torch.float32)
    rm_scores[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    trajectory_ids = [item.identity.canonical_id for item in trajectories]
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
            "rm_scores": rm_scores,
        },
        non_tensors={
            "uid": upstream_groups,
            "data_source": ["vstar"] * batch_size,
            PILOT_EXACT_GROUP_UID_FIELD: exact_groups,
            PILOT_EXACT_REWARD_FIELD: rewards,
            PILOT_GROUP_BATCH_SCHEMA_FIELD: [POLICY_PILOT_V1_GROUP_BATCH_SCHEMA]
            * batch_size,
            POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA_FIELD: [
                POLICY_PILOT_VERL_REWARD_BATCH_SCHEMA
            ]
            * batch_size,
            PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD: [
                PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
            ]
            * batch_size,
            PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD: [_PIPELINE_SHA256] * batch_size,
            PILOT_VERL_REWARD_COMPONENTS_FIELD: components,
            PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD: [equation_route] * batch_size,
            PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD: [applied_weights] * batch_size,
            PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD: trajectory_ids,
            TRAJECTORY_ID_FIELD: trajectory_ids,
            TRAJECTORY_PAYLOAD_FIELD: trajectories,
        },
    )


def test_real_dataproto_binds_repo_owned_exact_grpo_fields() -> None:
    data = _real_data_proto()

    view = bind_policy_pilot_exact_grpo_fields(
        data,
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
    )

    assert len(view.trajectory_ids) == 16
    assert (
        sum(
            trajectory.stop is TrajectoryStop.INVALID_FORMAT
            for trajectory in data.non_tensor_batch[TRAJECTORY_PAYLOAD_FIELD]
        )
        == 2
    )
    assert view.group_uids[:8] == ("exact-group-0",) * 8
    assert view.group_uids[8:] == ("exact-group-1",) * 8
    assert torch.equal(data.batch["token_level_scores"], data.batch["rm_scores"])
    assert torch.equal(data.batch["token_level_rewards"], data.batch["rm_scores"])
    assert torch.equal(data.batch["returns"], data.batch["advantages"])
    assert torch.equal(
        data.batch["advantages"][8:],
        torch.zeros_like(data.batch["advantages"][8:]),
    )
    assert torch.equal(
        data.batch["advantages"][:, 2:],
        torch.zeros_like(data.batch["advantages"][:, 2:]),
    )

    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    expected_sequences = compute_group_advantages(
        torch.tensor(view.rewards, dtype=torch.float32),
        torch.tensor([0] * 8 + [1] * 8, dtype=torch.int64),
        spec,
    )
    expected = expected_sequences[:, None] * data.batch["response_mask"].bool()
    assert torch.equal(data.batch["advantages"], expected)


def test_real_dataproto_binds_deepeyes_n16_groups() -> None:
    data = _real_data_proto(group_size=16)

    view = bind_policy_pilot_exact_grpo_fields(
        data,
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
        expected_group_size=16,
    )

    assert len(view.trajectory_ids) == 32
    assert view.group_uids[:16] == ("exact-group-0",) * 16
    assert view.group_uids[16:] == ("exact-group-1",) * 16
    assert torch.equal(
        data.batch["advantages"][16:],
        torch.zeros_like(data.batch["advantages"][16:]),
    )
    with pytest.raises(ValueError, match="complete n=16"):
        validate_policy_pilot_reward_data_proto(
            _real_data_proto(group_size=16, incomplete=True),
            expected_group_size=16,
        )
    assert torch.equal(
        data.batch["advantages"][:, 2:],
        torch.zeros_like(data.batch["advantages"][:, 2:]),
    )

    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
        expected_group_size=16,
    )
    expected_sequences = compute_group_advantages(
        torch.tensor(view.rewards, dtype=torch.float32),
        torch.tensor([0] * 16 + [1] * 16, dtype=torch.int64),
        spec,
    )
    expected = expected_sequences[:, None] * data.batch["response_mask"].bool()
    assert torch.equal(data.batch["advantages"], expected)


def test_answer_primary_reward_crosses_dataproto_sidecar_gate() -> None:
    data = _real_data_proto(tool_weight=0.2)

    view = validate_policy_pilot_reward_data_proto(data)

    assert view.rewards[3] == pytest.approx(1.0)
    assert view.rewards[7] == pytest.approx(1.0)


def test_deepeyes_thinklite_math_reward_crosses_dataproto_sidecar_gate() -> None:
    data = _real_data_proto(
        group_size=16,
        equation_route=PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    )
    for row_index, reward in ((0, 1.2), (2, -0.4)):
        data.non_tensor_batch["data_source"][row_index] = "thinklite"
        data.non_tensor_batch[PILOT_EXACT_REWARD_FIELD][row_index] = reward
        data.non_tensor_batch[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD][row_index] = (
            PILOT_REWARD_EQUATION_DEEPEYES_MATH
        )
        data.non_tensor_batch[PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD][row_index] = (
            PILOT_REWARD_DEEPEYES_MATH_WEIGHTS
        )
        data.batch["rm_scores"][row_index, -1] = reward

    view = validate_policy_pilot_reward_data_proto(
        data,
        expected_group_size=16,
    )

    assert view.rewards[0] == pytest.approx(1.2)
    assert view.rewards[2] == pytest.approx(-0.4)
    assert (
        data.non_tensor_batch[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD][1]
        == PILOT_REWARD_EQUATION_DEEPEYES_VISUAL
    )


def test_deepeyes_dataproto_rejects_route_that_differs_from_data_source() -> None:
    data = _real_data_proto(
        group_size=16,
        equation_route=PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    )
    data.non_tensor_batch["data_source"][0] = "thinklite"

    with pytest.raises(ValueError, match="differs from DataProto data_source"):
        validate_policy_pilot_reward_data_proto(data, expected_group_size=16)


def test_deepeyes_math_dataproto_rejects_nonzero_tool_component() -> None:
    data = _real_data_proto(
        group_size=16,
        equation_route=PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    )
    row_index = 3
    data.non_tensor_batch["data_source"][row_index] = "thinklite"
    data.non_tensor_batch[PILOT_VERL_REWARD_EQUATION_ROUTE_FIELD][row_index] = (
        PILOT_REWARD_EQUATION_DEEPEYES_MATH
    )
    data.non_tensor_batch[PILOT_VERL_REWARD_APPLIED_WEIGHTS_FIELD][row_index] = (
        PILOT_REWARD_DEEPEYES_MATH_WEIGHTS
    )
    data.non_tensor_batch[PILOT_EXACT_REWARD_FIELD][row_index] = 1.2
    data.batch["rm_scores"][row_index, -1] = 1.2

    with pytest.raises(ValueError, match="cannot contain a tool component"):
        validate_policy_pilot_reward_data_proto(data, expected_group_size=16)


def test_stage3_five_component_reward_crosses_dataproto_sidecar_gate() -> None:
    data = _real_data_proto()
    batch_size = data.batch["rm_scores"].shape[0]
    patterns = (
        (
            ("answer", 2.0),
            ("tool", 0.0),
            ("focus", 0.0),
            ("grounding", 0.0),
            ("protocol", 0.0),
        ),
        (
            ("answer", 0.0),
            ("tool", 0.0),
            ("focus", 0.0),
            ("grounding", 0.0),
            ("protocol", 0.0),
        ),
        (
            ("answer", 0.0),
            ("tool", 0.0),
            ("focus", 0.0),
            ("grounding", 0.0),
            ("protocol", -1.0),
        ),
        (
            ("answer", 2.0),
            ("tool", 0.5),
            ("focus", 1.0),
            ("grounding", 1.0),
            ("protocol", 0.0),
        ),
    )
    components = [patterns[index % len(patterns)] for index in range(batch_size)]
    rewards = [sum(score for _name, score in row) for row in components]
    data.non_tensor_batch[PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD] = [
        STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION
    ] * batch_size
    data.non_tensor_batch[PILOT_VERL_REWARD_COMPONENTS_FIELD] = components
    data.non_tensor_batch[PILOT_EXACT_REWARD_FIELD] = rewards
    data.non_tensor_batch[STAGE3_VERL_TOOL_LABEL_FIELD] = ["optional"] * batch_size
    data.non_tensor_batch[STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD] = [0.5] * batch_size
    data.non_tensor_batch[STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD] = [
        "b" * 64
    ] * batch_size
    data.non_tensor_batch[STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD] = [
        "c" * 64
    ] * batch_size
    data.non_tensor_batch[STAGE3_VERL_QUALITY_APPLICABLE_FIELD] = [False] * batch_size
    data.non_tensor_batch[STAGE3_VERL_QUALITY_COVERED_FIELD] = [False] * batch_size
    data.non_tensor_batch[STAGE3_VERL_QUALITY_FAILURE_FIELD] = [None] * batch_size
    data.non_tensor_batch[STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD] = [None] * batch_size
    data.batch["rm_scores"].zero_()
    data.batch["rm_scores"][:, -1] = torch.tensor(rewards)

    view = validate_policy_pilot_reward_data_proto(data)

    assert view.reward_bridge_schema_version == STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION
    assert view.rewards[3] == pytest.approx(4.5)


def test_dataproto_sidecar_rejects_unnamed_reward_total() -> None:
    data = _real_data_proto()
    data.non_tensor_batch[PILOT_EXACT_REWARD_FIELD][3] = 1.4
    data.batch["rm_scores"][3, -1] = 1.4

    with pytest.raises(ValueError, match="component sidecar differs from exact total"):
        validate_policy_pilot_reward_data_proto(data)


def test_reward_dataproto_fails_closed_without_exact_upstream_reward() -> None:
    data = _real_data_proto()
    del data.batch["rm_scores"]

    with pytest.raises(ValueError, match="rm_scores"):
        validate_policy_pilot_reward_data_proto(data)


def test_reward_dataproto_never_accepts_incomplete_n8_groups() -> None:
    data = _real_data_proto(incomplete=True)

    with pytest.raises(ValueError, match="complete n=8"):
        validate_policy_pilot_reward_data_proto(data)


def test_reward_dataproto_rejects_identity_reward_and_advantage_overwrites() -> None:
    data = _real_data_proto()
    data.non_tensor_batch[PILOT_EXACT_GROUP_UID_FIELD][0] = "forged-group"
    with pytest.raises(ValueError, match="group identity"):
        validate_policy_pilot_reward_data_proto(data)

    data = _real_data_proto()
    data.batch["rm_scores"][0, -1] = 0.0
    with pytest.raises(ValueError, match="exact trajectory rewards"):
        validate_policy_pilot_reward_data_proto(data)

    data = _real_data_proto()
    data.batch["advantages"] = torch.zeros_like(data.batch["rm_scores"])
    with pytest.raises(ValueError, match="repo-owned GRPO value"):
        bind_policy_pilot_exact_grpo_fields(
            data,
            diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
        )
