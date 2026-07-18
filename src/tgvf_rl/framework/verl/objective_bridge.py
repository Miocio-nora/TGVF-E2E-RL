"""Public veRL policy-loss registration and objective-sentinel contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from .compatibility import load_verl_public_api


OBJECTIVE_SENTINEL_FIELDS = (
    "group_construction",
    "group_standard_deviation",
    "advantage_scaling",
    "behavior_policy_ratio",
    "clipping",
    "reference_kl",
    "policy_token_mask",
    "token_sequence_normalization",
    "global_denominator",
    "gradient_accumulation",
)


def make_objective_sentinels(prefix: str = "tgvf-sentinel") -> Mapping[str, str]:
    """Create distinct values for the public-dataflow overwrite probe."""

    if not prefix:
        raise ValueError("sentinel prefix must be non-empty")
    return MappingProxyType(
        {
            field: f"{prefix}:{index}:{field}"
            for index, field in enumerate(OBJECTIVE_SENTINEL_FIELDS)
        }
    )


def validate_objective_sentinels(values: Mapping[str, object]) -> Mapping[str, object]:
    """Require all future objective-owned fields and distinct stable scalars."""

    if not isinstance(values, Mapping):
        raise TypeError("objective sentinels must be a mapping")
    if set(values) != set(OBJECTIVE_SENTINEL_FIELDS):
        missing = sorted(set(OBJECTIVE_SENTINEL_FIELDS) - set(values))
        extra = sorted(set(values) - set(OBJECTIVE_SENTINEL_FIELDS))
        raise ValueError(
            f"objective sentinel fields differ: missing={missing} extra={extra}"
        )
    normalized: dict[str, object] = {}
    identities: set[tuple[type[object], object]] = set()
    for field in OBJECTIVE_SENTINEL_FIELDS:
        value = values[field]
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"sentinel {field!r} must be a stable scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"sentinel {field!r} must be finite")
        identity = (type(value), value)
        if identity in identities:
            raise ValueError("objective sentinel values must be distinct")
        identities.add(identity)
        normalized[field] = value
    return MappingProxyType(normalized)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except RuntimeError:
        return left is right


@dataclass(frozen=True, slots=True)
class VerlPolicyLossCall:
    """The exact seven-argument public ``register_policy_loss`` invocation."""

    old_log_prob: torch.Tensor
    log_prob: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    loss_agg_mode: str
    config: object
    rollout_log_probs: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.old_log_prob,
            self.log_prob,
            self.advantages,
            self.response_mask,
            self.rollout_log_probs,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("veRL policy-loss inputs must be tensors")
        shape = self.log_prob.shape
        if self.log_prob.ndim != 2 or any(value.shape != shape for value in tensors):
            raise ValueError(
                "veRL policy-loss tensors must share [batch, response] shape"
            )
        if any(value.device != self.log_prob.device for value in tensors):
            raise ValueError("veRL policy-loss tensors must share one device")
        if not self.log_prob.dtype.is_floating_point:
            raise TypeError("current log probabilities must use a floating dtype")
        for name, value in (
            ("old_log_prob", self.old_log_prob),
            ("log_prob", self.log_prob),
            ("advantages", self.advantages),
            ("rollout_log_probs", self.rollout_log_probs),
        ):
            if not value.dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating dtype")
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise ValueError(f"{name} must be finite")
        if (
            self.response_mask.dtype is not torch.bool
            and self.response_mask.dtype
            not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            raise TypeError("response_mask must be bool or integer")
        if not bool(self.response_mask.bool().any().item()):
            raise ValueError("response_mask must select at least one policy token")
        if not bool(
            ((self.response_mask == 0) | (self.response_mask == 1)).all().item()
        ):
            raise ValueError("response_mask must remain binary")
        if not self.loss_agg_mode:
            raise ValueError("loss_agg_mode must remain explicit")
        if _shares_storage(self.log_prob, self.rollout_log_probs):
            raise ValueError(
                "rollout_log_probs must be actual recorded behavior values, not log_prob.detach()"
            )


ProjectPolicyLoss = Callable[
    [VerlPolicyLossCall], tuple[torch.Tensor, Mapping[str, Any]]
]


def adapt_policy_loss(
    project_loss: ProjectPolicyLoss,
) -> Callable[..., tuple[torch.Tensor, dict[str, Any]]]:
    """Adapt a project-owned loss to veRL's maintained seven-argument hook."""

    if not callable(project_loss):
        raise TypeError("project_loss must be callable")

    def verl_policy_loss(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
        config: object,
        rollout_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if rollout_log_probs is None:
            raise ValueError(
                "actual rollout_log_probs are required; replayed current log probabilities are forbidden"
            )
        call = VerlPolicyLossCall(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_log_probs=rollout_log_probs,
        )
        result = project_loss(call)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("project policy loss must return (scalar_loss, metrics)")
        loss, metrics = result
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("project policy loss must return a scalar tensor")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise ValueError("project policy loss returned a non-finite value")
        if not isinstance(metrics, Mapping):
            raise TypeError("project policy loss metrics must be a mapping")
        return loss, dict(metrics)

    verl_policy_loss.__name__ = (
        f"verl_{getattr(project_loss, '__name__', 'project_policy_loss')}"
    )
    verl_policy_loss.__doc__ = (
        "veRL public-hook adapter for a project-owned exact objective."
    )
    return verl_policy_loss


_REGISTERED_POLICY_LOSSES: dict[str, Callable[..., Any]] = {}


def register_project_policy_loss(
    name: str,
    project_loss: ProjectPolicyLoss,
    *,
    registrar: Callable[[str], Callable[[Callable[..., Any]], Callable[..., Any]]]
    | None = None,
) -> Callable[..., Any]:
    """Register through veRL's public decorator without patching its trainer."""

    if (
        not isinstance(name, str)
        or not name.startswith("tgvf_")
        or len(name) <= len("tgvf_")
    ):
        raise ValueError(
            "project policy-loss names must use a non-empty 'tgvf_' prefix"
        )
    if name in _REGISTERED_POLICY_LOSSES:
        raise ValueError(f"project policy loss {name!r} is already registered")
    if registrar is None:
        registrar = load_verl_public_api().register_policy_loss
    wrapped = adapt_policy_loss(project_loss)
    registered = registrar(name)(wrapped)
    if registered is not wrapped:
        raise RuntimeError(
            "veRL policy-loss registrar did not preserve the registered callable"
        )
    _REGISTERED_POLICY_LOSSES[name] = wrapped
    return wrapped
