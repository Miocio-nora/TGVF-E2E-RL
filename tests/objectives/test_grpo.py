from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion
from tgvf_rl.objectives import (
    GRPOSpec,
    GroupStdMode,
    LogProbSource,
    LossReduction,
    POLICY_PILOT_V1_GRPO_CONTRACT_ID,
    PolicyLogProbSet,
    RatioDenominator,
    ReductionSpec,
    ReferenceKLEstimator,
    ReferenceKLSpec,
    RoleLogProbs,
    ZeroVarianceBehavior,
    compute_grpo_loss,
    compute_group_advantages,
    policy_pilot_v1_grpo_spec,
)


def _version(step: int, digit: str) -> PolicyVersion:
    return PolicyVersion("objective-test", step, digit * 64)


def _policy(
    current_values: torch.Tensor,
    *,
    behavior_values: torch.Tensor | None = None,
    proximal_values: torch.Tensor | None = None,
    reference_values: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> PolicyLogProbSet:
    shape = current_values.shape
    behavior_values = (
        torch.zeros(shape, dtype=current_values.dtype)
        if behavior_values is None
        else behavior_values
    )
    proximal_values = (
        torch.full(shape, 0.15, dtype=current_values.dtype)
        if proximal_values is None
        else proximal_values
    )
    reference_values = (
        torch.full(shape, -0.10, dtype=current_values.dtype)
        if reference_values is None
        else reference_values
    )
    mask = torch.ones(shape, dtype=torch.bool) if mask is None else mask
    return PolicyLogProbSet(
        behavior=RoleLogProbs(
            ComponentRole.BEHAVIOR,
            behavior_values.detach().clone(),
            _version(0, "0"),
            LogProbSource.ROLLOUT_RECORDED,
            "9" * 64,
        ),
        proximal_old=RoleLogProbs(
            ComponentRole.PROXIMAL_OLD,
            proximal_values.detach().clone(),
            _version(1, "1"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        current=RoleLogProbs(
            ComponentRole.CURRENT,
            current_values,
            _version(2, "2"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        reference=RoleLogProbs(
            ComponentRole.REFERENCE,
            reference_values.detach().clone(),
            _version(0, "3"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        policy_sampled_mask=mask,
    )


def _spec(*, denominator: RatioDenominator = RatioDenominator.BEHAVIOR) -> GRPOSpec:
    return GRPOSpec(
        center_rewards=True,
        scale_by_group_std=True,
        group_std_mode=GroupStdMode.POPULATION,
        group_std_epsilon=1.0e-8,
        zero_variance_behavior=ZeroVarianceBehavior.ZERO_ADVANTAGE,
        ratio_denominator=denominator,
        clip_ratio_min=0.8,
        clip_ratio_max=1.2,
        dual_clip=None,
        reference_kl=ReferenceKLSpec(
            estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
            coefficient=0.17,
        ),
        reduction=ReductionSpec(
            mode=LossReduction.TOKEN_MEAN,
            fixed_token_normalizer=None,
        ),
    )


def test_population_group_advantages_and_zero_variance_are_explicit() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 6.0, 5.0, 5.0], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    actual = compute_group_advantages(rewards, groups, _spec())
    torch.testing.assert_close(
        actual,
        torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0], dtype=torch.float64),
    )

    epsilon_spec = replace(
        _spec(),
        zero_variance_behavior=ZeroVarianceBehavior.EPSILON_DIVISION,
    )
    epsilon_actual = compute_group_advantages(rewards, groups, epsilon_spec)
    torch.testing.assert_close(epsilon_actual[-2:], torch.zeros(2, dtype=torch.float64))


def test_grpo_value_and_gradient_match_cpu_oracle_and_ignore_template_tokens() -> None:
    current = torch.tensor(
        [
            [0.00, 0.30, -0.40],
            [0.20, -0.10, 0.45],
            [-0.35, 0.10, 0.00],
            [0.25, -0.25, 0.05],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ]
    )
    policy = _policy(current, mask=mask)
    rewards = torch.tensor([1.0, 3.0, 2.0, 6.0], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1])
    spec = _spec()

    result = compute_grpo_loss(spec, policy, rewards, groups)

    advantage = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)[:, None]
    ratio = current.exp()
    clipped = ratio.clamp(0.8, 1.2)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    log_ratio_to_reference = current - policy.reference.values
    k3 = torch.exp(-log_ratio_to_reference) + log_ratio_to_reference - 1.0
    manual_tokens = -surrogate + 0.17 * k3
    expected = manual_tokens[mask].mean()

    torch.testing.assert_close(result.loss, expected)
    assert torch.count_nonzero(result.per_token_loss[~mask]).item() == 0
    expected_gradient = torch.autograd.grad(expected, current, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(result.loss, current)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual_gradient[~mask]).item() == 0


def test_ratio_denominator_is_a_hashed_mathematical_choice() -> None:
    current = torch.tensor(
        [[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float64, requires_grad=True
    )
    rewards = torch.tensor([0.0, 2.0], dtype=torch.float64)
    groups = torch.tensor([7, 7])
    policy = _policy(current)
    behavior_spec = _spec(denominator=RatioDenominator.BEHAVIOR)
    proximal_spec = _spec(denominator=RatioDenominator.PROXIMAL_OLD)

    behavior_loss = compute_grpo_loss(behavior_spec, policy, rewards, groups).loss
    proximal_loss = compute_grpo_loss(proximal_spec, policy, rewards, groups).loss
    assert behavior_spec.identity_sha256 != proximal_spec.identity_sha256
    assert not torch.isclose(behavior_loss, proximal_loss)
    assert behavior_spec.identity_sha256 == _spec().identity_sha256


def test_detached_current_cannot_masquerade_as_recorded_behavior() -> None:
    current = torch.tensor([[0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ReplayMismatchError, match="share storage"):
        PolicyLogProbSet(
            behavior=RoleLogProbs(
                ComponentRole.BEHAVIOR,
                current.detach(),
                _version(0, "0"),
                LogProbSource.ROLLOUT_RECORDED,
                "9" * 64,
            ),
            proximal_old=RoleLogProbs(
                ComponentRole.PROXIMAL_OLD,
                current.detach().clone(),
                _version(0, "1"),
                LogProbSource.DETERMINISTIC_REPLAY,
                "9" * 64,
            ),
            current=RoleLogProbs(
                ComponentRole.CURRENT,
                current,
                _version(1, "2"),
                LogProbSource.DETERMINISTIC_REPLAY,
                "9" * 64,
            ),
            reference=RoleLogProbs(
                ComponentRole.REFERENCE,
                current.detach().clone(),
                _version(0, "3"),
                LogProbSource.DETERMINISTIC_REPLAY,
                "9" * 64,
            ),
            policy_sampled_mask=torch.ones_like(current, dtype=torch.bool),
        )


def test_sample_standard_deviation_rejects_singleton_group() -> None:
    spec = replace(_spec(), group_std_mode=GroupStdMode.SAMPLE)
    with pytest.raises(ValueError, match="at least two"):
        compute_group_advantages(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0, 1]),
            spec,
        )


def test_policy_pilot_v1_factory_freezes_math_but_requires_kl_identity() -> None:
    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )

    assert POLICY_PILOT_V1_GRPO_CONTRACT_ID == "POLICY-PILOT-V1-20260720"
    assert spec.center_rewards is True
    assert spec.scale_by_group_std is True
    assert spec.group_std_mode is GroupStdMode.SAMPLE
    assert spec.group_std_epsilon == 1.0e-6
    assert spec.zero_variance_behavior is ZeroVarianceBehavior.ZERO_ADVANTAGE
    assert spec.expected_group_size == 8
    assert spec.ratio_denominator is RatioDenominator.BEHAVIOR
    assert spec.clip_ratio_min == 0.8
    assert spec.clip_ratio_max == 1.2
    assert spec.dual_clip == 3.0
    assert spec.reference_kl.coefficient == 0.0
    assert spec.reduction == ReductionSpec(LossReduction.TOKEN_MEAN, None)

    same = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    different_diagnostic = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K1_SIGNED_LOG_RATIO
    )
    assert spec.identity_sha256 == same.identity_sha256
    assert spec.identity_sha256 != different_diagnostic.identity_sha256
    assert (
        spec.identity_sha256 != replace(spec, expected_group_size=None).identity_sha256
    )

    deepeyes = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
        expected_group_size=16,
    )
    assert deepeyes.expected_group_size == 16
    assert deepeyes.identity_sha256 != spec.identity_sha256

    for unsupported_group_size in (1, 7, 32):
        with pytest.raises(ValueError, match="one of"):
            policy_pilot_v1_grpo_spec(
                diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE,
                expected_group_size=unsupported_group_size,
            )

    with pytest.raises(TypeError, match="diagnostic_kl_estimator"):
        policy_pilot_v1_grpo_spec(diagnostic_kl_estimator=None)  # type: ignore[arg-type]


def test_policy_pilot_v1_uses_sample_std_plus_epsilon_and_zeroes_equal_group() -> None:
    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    varying = torch.arange(8, dtype=torch.float64)
    equal = torch.full((8,), 7.0, dtype=torch.float64)
    rewards = torch.stack((varying, equal), dim=1).reshape(-1)
    groups = torch.tensor([41, -7] * 8)

    actual = compute_group_advantages(rewards, groups, spec)
    varying_centered = varying - varying.mean()
    expected_varying = varying_centered / (
        varying.std(correction=1) + spec.group_std_epsilon
    )
    expected = torch.stack(
        (expected_varying, torch.zeros_like(expected_varying)), dim=1
    ).reshape(-1)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(
        actual[::2],
        varying_centered / varying.std(correction=1),
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.equal(actual[1::2], torch.zeros(8, dtype=torch.float64))

    equal_decimal = compute_group_advantages(
        torch.full((8,), 0.8, dtype=torch.float64),
        torch.full((8,), 5, dtype=torch.int64),
        spec,
    )
    assert torch.equal(equal_decimal, torch.zeros_like(equal_decimal))


def test_policy_pilot_v1_rejects_non_eight_groups_without_breaking_generic_mode() -> (
    None
):
    pilot = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    rewards = torch.arange(15, dtype=torch.float64)
    groups = torch.tensor([3] * 8 + [9] * 7)

    with pytest.raises(ReplayMismatchError, match="exactly 8"):
        compute_group_advantages(rewards, groups, pilot)

    generic = replace(pilot, expected_group_size=None)
    actual = compute_group_advantages(rewards, groups, generic)
    assert actual.shape == rewards.shape
    assert torch.isfinite(actual).all()


def test_policy_pilot_v1_loss_matches_dual_clip_global_token_mean_oracle() -> None:
    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    rewards = torch.arange(8, dtype=torch.float64)
    groups = torch.full((8,), 73, dtype=torch.int64)
    sequence_ratios = torch.tensor(
        [5.0, 0.1, 1.0, 2.0, 0.1, 5.0, 0.7, 1.1], dtype=torch.float64
    )
    behavior = torch.full((8, 4), -2.0, dtype=torch.float64)
    current = (behavior + sequence_ratios.log()[:, None]).clone().requires_grad_(True)
    reference = torch.full((8, 4), -2.4, dtype=torch.float64)
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    policy = _policy(
        current,
        behavior_values=behavior,
        reference_values=reference,
        mask=mask,
    )

    result = compute_grpo_loss(spec, policy, rewards, groups)

    advantages = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1.0e-6)
    ratios = torch.exp(current - behavior)
    broadcast_advantages = advantages[:, None]
    clipped_surrogate = torch.minimum(
        ratios * broadcast_advantages,
        ratios.clamp(0.8, 1.2) * broadcast_advantages,
    )
    expected_surrogate = torch.where(
        broadcast_advantages < 0,
        torch.maximum(clipped_surrogate, 3.0 * broadcast_advantages),
        clipped_surrogate,
    )
    expected_loss = (-expected_surrogate)[mask].mean()
    per_token_k3 = torch.exp(-(current - reference)) + (current - reference) - 1.0
    expected_kl_diagnostic = per_token_k3[mask].mean()

    torch.testing.assert_close(result.loss, expected_loss)
    torch.testing.assert_close(result.metrics["policy_loss"], expected_loss.detach())
    torch.testing.assert_close(
        result.metrics["reference_kl"], expected_kl_diagnostic.detach()
    )
    assert result.metrics["reference_kl_contribution"].item() == 0.0
    assert torch.count_nonzero(result.per_token_loss[~mask]).item() == 0

    sequence_means = torch.stack(
        [(-expected_surrogate[row])[mask[row]].mean() for row in range(8)]
    ).mean()
    assert not torch.isclose(expected_loss, sequence_means)

    expected_gradient = torch.autograd.grad(expected_loss, current, retain_graph=True)[
        0
    ]
    actual_gradient = torch.autograd.grad(result.loss, current)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual_gradient[~mask]).item() == 0
