"""Shared, fail-closed contracts for policy objectives.

The objective layer deliberately does not infer any mathematical choice from a
library default.  Every choice which changes a loss is represented by a frozen
specification and therefore contributes to a stable SHA256 identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError, TGVFContractError
from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion


class LossReduction(str, Enum):
    """Explicit token/sequence normalization used by an objective."""

    TOKEN_MEAN = "token_mean"
    SEQUENCE_MEAN_TOKEN_MEAN = "sequence_mean_token_mean"
    SEQUENCE_MEAN_TOKEN_SUM = "sequence_mean_token_sum"
    FIXED_TOKEN_NORMALIZER = "fixed_token_normalizer"


class RatioDenominator(str, Enum):
    """Policy whose likelihood is the denominator of a policy ratio."""

    BEHAVIOR = "behavior"
    PROXIMAL_OLD = "proximal_old"


class ReferenceKLEstimator(str, Enum):
    """Per-sampled-token estimators for ``KL(current || reference)``."""

    K1_SIGNED_LOG_RATIO = "k1_signed_log_ratio"
    K2_SQUARED_LOG_RATIO = "k2_squared_log_ratio"
    K3_LOW_VARIANCE = "k3_low_variance"


class LogProbSource(str, Enum):
    """Where a role-specific log-probability tensor came from."""

    ROLLOUT_RECORDED = "rollout_recorded"
    DETERMINISTIC_REPLAY = "deterministic_replay"


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    mode: LossReduction
    fixed_token_normalizer: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, LossReduction):
            raise TypeError("reduction mode must be LossReduction")
        if self.mode is LossReduction.FIXED_TOKEN_NORMALIZER:
            if self.fixed_token_normalizer is None:
                raise ValueError(
                    "fixed-token reduction requires an explicit normalizer"
                )
            _require_real(self.fixed_token_normalizer, "fixed token normalizer")
            if (
                not math.isfinite(self.fixed_token_normalizer)
                or self.fixed_token_normalizer <= 0
            ):
                raise ValueError("fixed token normalizer must be finite and positive")
        elif self.fixed_token_normalizer is not None:
            raise ValueError(
                "fixed_token_normalizer must be None for the selected reduction"
            )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class ReferenceKLSpec:
    estimator: ReferenceKLEstimator
    coefficient: float

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, ReferenceKLEstimator):
            raise TypeError("reference KL estimator must be ReferenceKLEstimator")
        _require_real(self.coefficient, "reference KL coefficient")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("reference KL coefficient must be finite and non-negative")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RoleLogProbs:
    """Log probabilities with an explicit policy role and provenance."""

    role: ComponentRole
    values: torch.Tensor
    policy_version: PolicyVersion
    source: LogProbSource
    sampling_transform_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ComponentRole):
            raise TypeError("policy log-probability role must be ComponentRole")
        if not isinstance(self.source, LogProbSource):
            raise TypeError("policy log-probability source must be LogProbSource")
        if not isinstance(self.policy_version, PolicyVersion):
            raise TypeError("policy_version must be PolicyVersion")
        _validate_sha256(self.sampling_transform_sha256)
        if self.role not in {
            ComponentRole.BEHAVIOR,
            ComponentRole.PROXIMAL_OLD,
            ComponentRole.CURRENT,
            ComponentRole.REFERENCE,
        }:
            raise ValueError(
                f"role {self.role.value!r} is not a policy log-probability role"
            )
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("log probabilities must be a torch.Tensor")
        if self.values.ndim != 2:
            raise ValueError(
                "policy log probabilities must have shape [batch, sequence]"
            )
        if not self.values.dtype.is_floating_point:
            raise TypeError("policy log probabilities must use a floating dtype")
        if not bool(torch.isfinite(self.values.detach()).all().item()):
            raise ValueError("policy log probabilities must be finite")
        expected_source = (
            LogProbSource.ROLLOUT_RECORDED
            if self.role is ComponentRole.BEHAVIOR
            else LogProbSource.DETERMINISTIC_REPLAY
        )
        if self.source is not expected_source:
            raise ReplayMismatchError(
                f"{self.role.value} log probabilities require source {expected_source.value}"
            )
        if self.role is not ComponentRole.CURRENT and self.values.requires_grad:
            raise ValueError(
                f"{self.role.value} log probabilities must be gradient-free"
            )


@dataclass(frozen=True, slots=True)
class PolicyLogProbSet:
    """Four deliberately separate policy identities used by RL objectives.

    Separate role objects are required even when two policies intentionally
    carry equal weights.  Storage aliasing is rejected so that
    ``old_logprobs = new_logprobs.detach()`` cannot masquerade as recorded
    behavior or an independently replayed proximal policy.
    """

    behavior: RoleLogProbs
    proximal_old: RoleLogProbs
    current: RoleLogProbs
    reference: RoleLogProbs
    policy_sampled_mask: torch.Tensor

    def __post_init__(self) -> None:
        expected = (
            ("behavior", self.behavior, ComponentRole.BEHAVIOR),
            ("proximal_old", self.proximal_old, ComponentRole.PROXIMAL_OLD),
            ("current", self.current, ComponentRole.CURRENT),
            ("reference", self.reference, ComponentRole.REFERENCE),
        )
        for field_name, block, role in expected:
            if not isinstance(block, RoleLogProbs):
                raise TypeError(f"{field_name} must be RoleLogProbs")
            if block.role is not role:
                raise ValueError(f"{field_name} must carry role {role.value}")

        shape = self.current.values.shape
        device = self.current.values.device
        for field_name, block, _ in expected:
            if block.values.shape != shape:
                raise ReplayMismatchError(
                    f"{field_name} log-probability shape mismatch"
                )
            if block.values.device != device:
                raise ReplayMismatchError(
                    f"{field_name} log probabilities are on a different device"
                )

        if not isinstance(self.policy_sampled_mask, torch.Tensor):
            raise TypeError("policy_sampled_mask must be a torch.Tensor")
        if self.policy_sampled_mask.dtype is not torch.bool:
            raise TypeError("policy_sampled_mask must have dtype bool")
        if self.policy_sampled_mask.shape != shape:
            raise ReplayMismatchError("policy_sampled_mask shape mismatch")
        if self.policy_sampled_mask.device != device:
            raise ReplayMismatchError("policy_sampled_mask is on a different device")
        if not bool(self.policy_sampled_mask.any().item()):
            raise ValueError("policy_sampled_mask must select at least one token")

        tensors = [(name, block.values) for name, block, _ in expected]
        for left_index, (left_name, left) in enumerate(tensors):
            for right_name, right in tensors[left_index + 1 :]:
                if tensors_share_storage(left, right):
                    raise ReplayMismatchError(
                        f"{left_name} and {right_name} log probabilities share storage; "
                        "recorded/replayed policy roles must be independently materialized"
                    )
        transform_identities = {
            block.sampling_transform_sha256 for _, block, _ in expected
        }
        if len(transform_identities) != 1:
            raise ReplayMismatchError(
                "behavior/current/old/reference log probabilities use different sampling measures"
            )

    def ratio_denominator(self, role: RatioDenominator) -> torch.Tensor:
        if not isinstance(role, RatioDenominator):
            raise TypeError("ratio denominator must be RatioDenominator")
        if role is RatioDenominator.BEHAVIOR:
            return self.behavior.values
        if role is RatioDenominator.PROXIMAL_OLD:
            return self.proximal_old.values
        raise ValueError(f"unknown ratio denominator: {role!r}")


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    loss: torch.Tensor
    per_token_loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | float]
    spec_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0:
            raise TypeError("objective loss must be a scalar tensor")
        if (
            not isinstance(self.per_token_loss, torch.Tensor)
            or self.per_token_loss.ndim != 2
        ):
            raise TypeError("per_token_loss must have shape [batch, sequence]")
        if not bool(torch.isfinite(self.loss.detach()).item()):
            raise TGVFContractError("objective produced a non-finite loss")
        _validate_sha256(self.spec_identity_sha256)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def tensors_share_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return true for aliases/views backed by the same storage allocation."""

    if left.device != right.device:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except RuntimeError:
        # Meta tensors and unusual backends do not expose storage.  Object
        # identity is still a forbidden alias in those environments.
        return left is right


def reduce_token_loss(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    spec: ReductionSpec,
) -> torch.Tensor:
    """Reduce a [batch, sequence] loss using only selected tokens."""

    if per_token_loss.ndim != 2 or mask.shape != per_token_loss.shape:
        raise ValueError("loss and mask must have equal [batch, sequence] shapes")
    if mask.dtype is not torch.bool:
        raise TypeError("loss mask must have dtype bool")
    if mask.device != per_token_loss.device:
        raise ValueError("loss and mask must be on the same device")
    if not bool(mask.any().item()):
        raise ValueError("loss mask must select at least one token")
    selected = torch.where(mask, per_token_loss, torch.zeros_like(per_token_loss))
    if not bool(torch.isfinite(selected.detach()).all().item()):
        raise TGVFContractError("selected per-token objective values must be finite")

    if spec.mode is LossReduction.TOKEN_MEAN:
        return selected.sum() / mask.sum().to(dtype=per_token_loss.dtype)

    token_counts = mask.sum(dim=-1)
    active_sequences = token_counts > 0
    sequence_sums = selected.sum(dim=-1)
    if spec.mode is LossReduction.SEQUENCE_MEAN_TOKEN_MEAN:
        sequence_values = sequence_sums[active_sequences] / token_counts[
            active_sequences
        ].to(dtype=per_token_loss.dtype)
        return sequence_values.mean()
    if spec.mode is LossReduction.SEQUENCE_MEAN_TOKEN_SUM:
        return sequence_sums[active_sequences].mean()
    if spec.mode is LossReduction.FIXED_TOKEN_NORMALIZER:
        assert spec.fixed_token_normalizer is not None
        return selected.sum() / spec.fixed_token_normalizer
    raise ValueError(f"unknown reduction mode: {spec.mode!r}")


def reference_kl_per_token(
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    spec: ReferenceKLSpec,
) -> torch.Tensor:
    """Compute the selected sampled-token KL estimator without reducing it."""

    if current_log_probs.shape != reference_log_probs.shape:
        raise ReplayMismatchError("current/reference log-probability shape mismatch")
    log_ratio = current_log_probs - reference_log_probs
    if spec.estimator is ReferenceKLEstimator.K1_SIGNED_LOG_RATIO:
        return log_ratio
    if spec.estimator is ReferenceKLEstimator.K2_SQUARED_LOG_RATIO:
        return 0.5 * log_ratio.square()
    if spec.estimator is ReferenceKLEstimator.K3_LOW_VARIANCE:
        # Schulman's k3 estimator: exp(-r) + r - 1 for r=log(pi/ref).
        return torch.exp(-log_ratio) + log_ratio - 1.0
    raise ValueError(f"unknown reference KL estimator: {spec.estimator!r}")


def spec_identity_sha256(spec: object) -> str:
    """Hash a frozen mathematical spec using a canonical JSON encoding."""

    if not is_dataclass(spec):
        raise TypeError("objective specs must be dataclass instances")
    parameters = getattr(type(spec), "__dataclass_params__", None)
    if parameters is None or not parameters.frozen:
        raise TypeError("objective specs must be frozen dataclasses")
    payload = {
        "type": f"{type(spec).__module__}.{type(spec).__qualname__}",
        "fields": {
            field.name: _canonical_spec_value(getattr(spec, field.name))
            for field in fields(spec)
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_spec_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _canonical_spec_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_spec_value(item) for key, item in sorted(value.items())
        }
    if isinstance(value, tuple):
        return [_canonical_spec_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported objective-spec value: {type(value).__name__}")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"expected lowercase SHA256, got {value!r}")


def _require_real(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
