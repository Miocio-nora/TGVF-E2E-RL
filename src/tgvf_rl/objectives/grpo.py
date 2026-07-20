"""Pure-tensor Group Relative Policy Optimization objective.

This module is intentionally independent from a trainer or veRL worker.  It is
the small numerical contract that those integrations must call and reproduce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError

from .base import (
    LossReduction,
    ObjectiveResult,
    PolicyLogProbSet,
    RatioDenominator,
    ReductionSpec,
    ReferenceKLEstimator,
    ReferenceKLSpec,
    reduce_token_loss,
    reference_kl_per_token,
    spec_identity_sha256,
)


POLICY_PILOT_V1_GRPO_CONTRACT_ID = "POLICY-PILOT-V1-20260720"


class GroupStdMode(str, Enum):
    POPULATION = "population"
    SAMPLE = "sample"


class ZeroVarianceBehavior(str, Enum):
    ZERO_ADVANTAGE = "zero_advantage"
    EPSILON_DIVISION = "epsilon_division"


@dataclass(frozen=True, slots=True)
class GRPOSpec:
    """Every mathematical selection required by the pure GRPO objective."""

    center_rewards: bool
    scale_by_group_std: bool
    group_std_mode: GroupStdMode
    group_std_epsilon: float
    zero_variance_behavior: ZeroVarianceBehavior
    ratio_denominator: RatioDenominator
    clip_ratio_min: float
    clip_ratio_max: float
    dual_clip: float | None
    reference_kl: ReferenceKLSpec
    reduction: ReductionSpec
    expected_group_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.center_rewards, bool) or not isinstance(
            self.scale_by_group_std, bool
        ):
            raise TypeError(
                "reward centering and group-std scaling choices must be bool"
            )
        if not isinstance(self.group_std_mode, GroupStdMode):
            raise TypeError("group_std_mode must be GroupStdMode")
        if not isinstance(self.zero_variance_behavior, ZeroVarianceBehavior):
            raise TypeError("zero_variance_behavior must be ZeroVarianceBehavior")
        if not isinstance(self.ratio_denominator, RatioDenominator):
            raise TypeError("ratio_denominator must be RatioDenominator")
        if not isinstance(self.reference_kl, ReferenceKLSpec):
            raise TypeError("reference_kl must be ReferenceKLSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise TypeError("reduction must be ReductionSpec")
        if self.expected_group_size is not None and (
            type(self.expected_group_size) is not int or self.expected_group_size <= 0
        ):
            raise ValueError("expected_group_size must be None or a positive integer")
        _require_real(self.group_std_epsilon, "group_std_epsilon")
        if not math.isfinite(self.group_std_epsilon) or self.group_std_epsilon <= 0:
            raise ValueError("group_std_epsilon must be finite and positive")
        _require_real(self.clip_ratio_min, "clip_ratio_min")
        _require_real(self.clip_ratio_max, "clip_ratio_max")
        if not math.isfinite(self.clip_ratio_min) or not math.isfinite(
            self.clip_ratio_max
        ):
            raise ValueError("clip ratios must be finite")
        if not 0 < self.clip_ratio_min <= 1.0 <= self.clip_ratio_max:
            raise ValueError("clip ratios must satisfy 0 < min <= 1 <= max")
        if self.dual_clip is not None:
            _require_real(self.dual_clip, "dual_clip")
            if not math.isfinite(self.dual_clip) or self.dual_clip <= 1.0:
                raise ValueError(
                    "dual_clip must be None or a finite value greater than one"
                )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


def policy_pilot_v1_grpo_spec(
    *, diagnostic_kl_estimator: ReferenceKLEstimator
) -> GRPOSpec:
    """Build the exact pure-loss contract accepted for Policy Pilot v1.

    The reference-KL estimator deliberately has no default because section 0.8
    leaves that diagnostic identity unresolved.  Its coefficient is fixed to
    zero, so it is reported without contributing to the loss.
    """

    if not isinstance(diagnostic_kl_estimator, ReferenceKLEstimator):
        raise TypeError("diagnostic_kl_estimator must be ReferenceKLEstimator")
    return GRPOSpec(
        center_rewards=True,
        scale_by_group_std=True,
        group_std_mode=GroupStdMode.SAMPLE,
        group_std_epsilon=1.0e-6,
        zero_variance_behavior=ZeroVarianceBehavior.ZERO_ADVANTAGE,
        ratio_denominator=RatioDenominator.BEHAVIOR,
        clip_ratio_min=0.8,
        clip_ratio_max=1.2,
        dual_clip=3.0,
        reference_kl=ReferenceKLSpec(
            estimator=diagnostic_kl_estimator,
            coefficient=0.0,
        ),
        reduction=ReductionSpec(
            mode=LossReduction.TOKEN_MEAN,
            fixed_token_normalizer=None,
        ),
        expected_group_size=8,
    )


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    spec: GRPOSpec,
) -> torch.Tensor:
    """Compute one scalar group-relative advantage per sampled sequence."""

    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
        raise TypeError("rewards must be a one-dimensional tensor")
    if not rewards.dtype.is_floating_point:
        raise TypeError("rewards must use a floating dtype")
    if not isinstance(group_ids, torch.Tensor) or group_ids.ndim != 1:
        raise TypeError("group_ids must be a one-dimensional tensor")
    if group_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("group_ids must use an integer dtype")
    if rewards.shape != group_ids.shape:
        raise ReplayMismatchError("rewards and group_ids shape mismatch")
    if rewards.device != group_ids.device:
        raise ReplayMismatchError("rewards and group_ids must be on the same device")
    if rewards.numel() == 0:
        raise ValueError("GRPO requires at least one reward")
    if rewards.requires_grad:
        raise ValueError("rewards must not carry gradients")
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("rewards must be finite")

    _, inverse, counts = torch.unique(
        group_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    if spec.expected_group_size is not None and bool(
        torch.any(counts != spec.expected_group_size).item()
    ):
        raise ReplayMismatchError(
            f"every GRPO group must contain exactly {spec.expected_group_size} trajectories"
        )
    if spec.group_std_mode is GroupStdMode.SAMPLE and bool(
        torch.any(counts < 2).item()
    ):
        raise ValueError(
            "sample standard deviation requires at least two items per group"
        )

    group_sums = torch.zeros(
        counts.shape[0], dtype=rewards.dtype, device=rewards.device
    )
    group_sums.scatter_add_(0, inverse, rewards)
    group_means = group_sums / counts.to(dtype=rewards.dtype)
    centered_rewards = rewards - group_means[inverse]
    advantages = centered_rewards if spec.center_rewards else rewards

    if not spec.scale_by_group_std:
        return advantages

    squared_deviation_sums = torch.zeros_like(group_sums)
    squared_deviation_sums.scatter_add_(0, inverse, centered_rewards.square())
    correction = 1 if spec.group_std_mode is GroupStdMode.SAMPLE else 0
    variance_denominators = (counts - correction).to(dtype=rewards.dtype)
    group_variances = squared_deviation_sums / variance_denominators
    group_standard_deviations = torch.sqrt(group_variances)
    advantages = advantages / (
        group_standard_deviations[inverse] + spec.group_std_epsilon
    )
    if spec.zero_variance_behavior is ZeroVarianceBehavior.ZERO_ADVANTAGE:
        # The accepted Pilot contract says *identical input rewards* produce
        # exact zero. Computing the mean first can make equal non-binary values
        # such as 0.8 acquire the same tiny nonzero centered residual, so test
        # equality on the original rewards rather than using a tolerance.
        group_rewards_are_identical = torch.stack(
            tuple(
                torch.all(
                    rewards[inverse == group_index]
                    == rewards[inverse == group_index][0]
                )
                for group_index in range(counts.numel())
            )
        )
        zero_variance_rows = group_rewards_are_identical[inverse] | (
            group_variances[inverse] == 0
        )
        advantages = torch.where(
            zero_variance_rows,
            torch.zeros_like(advantages),
            advantages,
        )
    return advantages


def compute_grpo_loss(
    spec: GRPOSpec,
    policy: PolicyLogProbSet,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
) -> ObjectiveResult:
    """Compute clipped pure GRPO plus the explicitly selected reference KL."""

    if rewards.shape[0] != policy.current.values.shape[0]:
        raise ReplayMismatchError("one reward is required per policy sequence")
    if (
        rewards.device != policy.current.values.device
        or group_ids.device != rewards.device
    ):
        raise ReplayMismatchError(
            "GRPO rewards, groups, and policy tensors must share a device"
        )

    sequence_advantages = compute_group_advantages(rewards, group_ids, spec)
    advantages = sequence_advantages[:, None].to(dtype=policy.current.values.dtype)
    denominator = policy.ratio_denominator(spec.ratio_denominator)
    ratios = torch.exp(policy.current.values - denominator)
    clipped_ratios = torch.clamp(
        ratios, min=spec.clip_ratio_min, max=spec.clip_ratio_max
    )
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    if spec.dual_clip is not None:
        dual_floor = spec.dual_clip * advantages
        surrogate = torch.where(
            advantages < 0,
            torch.maximum(surrogate, dual_floor),
            surrogate,
        )

    kl = reference_kl_per_token(
        policy.current.values,
        policy.reference.values,
        spec.reference_kl,
    )
    raw_per_token_loss = -surrogate + spec.reference_kl.coefficient * kl
    mask = policy.policy_sampled_mask
    per_token_loss = torch.where(
        mask, raw_per_token_loss, torch.zeros_like(raw_per_token_loss)
    )
    loss = reduce_token_loss(per_token_loss, mask, spec.reduction)

    policy_loss = reduce_token_loss(
        torch.where(mask, -surrogate, torch.zeros_like(surrogate)),
        mask,
        spec.reduction,
    )
    kl_loss = reduce_token_loss(
        torch.where(mask, kl, torch.zeros_like(kl)),
        mask,
        spec.reduction,
    )
    clip_fraction = (
        ((ratios < spec.clip_ratio_min) | (ratios > spec.clip_ratio_max))[mask]
        .to(dtype=torch.float32)
        .mean()
    )
    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "reference_kl": kl_loss.detach(),
        "reference_kl_contribution": (spec.reference_kl.coefficient * kl_loss).detach(),
        "clip_fraction": clip_fraction.detach(),
        "mean_advantage": sequence_advantages.mean().detach(),
        "mean_ratio": ratios[mask].mean().detach(),
    }
    return ObjectiveResult(
        loss=loss,
        per_token_loss=per_token_loss,
        metrics=metrics,
        spec_identity_sha256=spec.identity_sha256,
    )


def _require_real(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
