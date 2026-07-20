"""Public veRL policy-loss registration and objective-sentinel contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion
from tgvf_rl.objectives import (
    LogProbSource,
    PolicyLogProbSet,
    ReferenceKLEstimator,
    RoleLogProbs,
    compute_grpo_loss,
    compute_group_advantages,
    policy_pilot_v1_grpo_spec,
)
from tgvf_rl.policy.config import POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE

from .compatibility import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    VerlCompatibilityError,
    VerlPublicAPI,
    load_verl_public_api,
)


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


@dataclass(frozen=True, slots=True)
class VerlPilotGRPOParityReport:
    """Numerical evidence for the pinned veRL execution path."""

    verl_commit: str
    objective_identity_sha256: str
    selected_policy_token_count: int
    maximum_advantage_absolute_error: float
    loss_absolute_error: float
    maximum_gradient_absolute_error: float


class _ParityActorConfig:
    """Minimal public actor-config surface consumed by veRL's loss hook."""

    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    clip_ratio_c = 3.0
    policy_loss = {
        "rollout_correction": {
            "bypass_mode": True,
            "loss_type": "ppo_clip",
            "rollout_is": None,
            "rollout_rs": None,
            "rollout_is_batch_normalize": False,
        }
    }

    def __init__(self, selected_policy_token_count: int) -> None:
        self.global_batch_info = {
            "dp_size": 1,
            "batch_num_tokens": selected_policy_token_count,
            "global_batch_size": 16,
            "loss_scale_factor": None,
        }

    def get(self, name: str, default: object = None) -> object:
        return getattr(self, name, default)


def _parity_role(
    role: ComponentRole,
    values: torch.Tensor,
    *,
    optimizer_step: int,
    sha_digit: str,
) -> RoleLogProbs:
    source = (
        LogProbSource.ROLLOUT_RECORDED
        if role is ComponentRole.BEHAVIOR
        else LogProbSource.DETERMINISTIC_REPLAY
    )
    return RoleLogProbs(
        role=role,
        values=values,
        policy_version=PolicyVersion(
            "verl-pilot-parity", optimizer_step, sha_digit * 64
        ),
        source=source,
        sampling_transform_sha256="9" * 64,
    )


def validate_policy_pilot_v1_verl_grpo_parity(
    public_api: VerlPublicAPI | None = None,
    *,
    policy_loss: Callable[..., tuple[torch.Tensor, Mapping[str, Any]]] | None = None,
) -> VerlPilotGRPOParityReport:
    """Compare pinned veRL GRPO execution with the project-owned Pilot oracle.

    The fixture jointly exercises sample-standard-deviation normalization, an
    equal-reward group, behavior-policy ratios, asymmetric masks, both PPO
    clipping branches, dual clipping, and global policy-token mean reduction.
    A positive-advantage token uses a log ratio below -20 so an undocumented
    pre-exponential clamp cannot pass as the accepted unclamped equation.
    It deliberately calls veRL's maintained ``bypass_mode`` entry point: the
    ``old_log_prob`` argument is the rollout-recorded behavior probability.
    """

    api = load_verl_public_api() if public_api is None else public_api
    if not isinstance(api, VerlPublicAPI):
        raise TypeError("public_api must be VerlPublicAPI")
    if policy_loss is None:
        policy_loss = api.get_policy_loss_fn("bypass_mode")
        if getattr(policy_loss, "__module__", None) != (
            POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE
        ):
            raise VerlCompatibilityError(
                "actor worker did not register the repo-owned exact bypass loss"
            )
    elif not callable(policy_loss):
        raise TypeError("policy_loss must be callable")

    dtype = torch.float64
    rewards = torch.cat(
        (
            torch.tensor(
                [-1.0, 0.0, 0.5, 1.5, 3.0, 4.0, 6.0, 9.0], dtype=dtype
            ),
            torch.full((8,), 2.75, dtype=dtype),
        )
    )
    group_ids = torch.tensor([17] * 8 + [91] * 8, dtype=torch.int64)
    response_mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
        * 2,
        dtype=torch.bool,
    )
    token_level_rewards = torch.zeros((16, 4), dtype=dtype)
    token_level_rewards[:, 0] = rewards
    spec = policy_pilot_v1_grpo_spec(
        diagnostic_kl_estimator=ReferenceKLEstimator.K3_LOW_VARIANCE
    )
    project_advantages = compute_group_advantages(rewards, group_ids, spec)

    upstream_advantages, _ = api.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=group_ids.tolist(),
        epsilon=spec.group_std_epsilon,
        norm_adv_by_std_in_grpo=True,
    )
    expected_advantages = project_advantages[:, None] * response_mask
    if not isinstance(upstream_advantages, torch.Tensor):
        raise VerlCompatibilityError("veRL GRPO advantage did not return a tensor")
    if upstream_advantages.shape != expected_advantages.shape:
        raise VerlCompatibilityError(
            "veRL GRPO advantage shape differs from the Pilot oracle"
        )

    behavior = torch.full((16, 4), -2.0, dtype=dtype)
    sequence_ratios = torch.tensor(
        [5.0, 0.1, 1.0, 2.0, 0.1, 5.0, 0.7, math.exp(-21.0)] * 2,
        dtype=dtype,
    )
    current_values = behavior + sequence_ratios.log()[:, None]
    # Poison non-policy positions. A response/template mask regression therefore
    # changes the scalar loss and gradient instead of passing accidentally.
    current_values = torch.where(
        response_mask,
        current_values,
        torch.full_like(current_values, 7.0),
    )
    project_current = current_values.clone().requires_grad_(True)
    upstream_current = current_values.clone().requires_grad_(True)
    policy = PolicyLogProbSet(
        behavior=_parity_role(
            ComponentRole.BEHAVIOR,
            behavior.clone(),
            optimizer_step=0,
            sha_digit="0",
        ),
        proximal_old=_parity_role(
            ComponentRole.PROXIMAL_OLD,
            torch.full_like(behavior, -0.25),
            optimizer_step=1,
            sha_digit="1",
        ),
        current=_parity_role(
            ComponentRole.CURRENT,
            project_current,
            optimizer_step=2,
            sha_digit="2",
        ),
        reference=_parity_role(
            ComponentRole.REFERENCE,
            torch.full_like(behavior, -2.4),
            optimizer_step=0,
            sha_digit="3",
        ),
        policy_sampled_mask=response_mask,
    )
    project_result = compute_grpo_loss(spec, policy, rewards, group_ids)
    upstream_loss, _ = policy_loss(
        old_log_prob=behavior.clone(),
        log_prob=upstream_current,
        advantages=upstream_advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_ParityActorConfig(int(response_mask.sum().item())),
        rollout_is_weights=None,
    )
    if not isinstance(upstream_loss, torch.Tensor) or upstream_loss.ndim != 0:
        raise VerlCompatibilityError("veRL bypass-mode loss did not return a scalar")

    project_gradient = torch.autograd.grad(project_result.loss, project_current)[0]
    upstream_gradient = torch.autograd.grad(upstream_loss, upstream_current)[0]
    advantage_error = float(
        (upstream_advantages - expected_advantages).abs().max().item()
    )
    loss_error = float((upstream_loss - project_result.loss).abs().item())
    gradient_error = float(
        (upstream_gradient - project_gradient).abs().max().item()
    )
    try:
        torch.testing.assert_close(
            upstream_advantages,
            expected_advantages,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            upstream_loss,
            project_result.loss,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            upstream_gradient,
            project_gradient,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as error:
        raise VerlCompatibilityError(
            "pinned veRL bypass-mode GRPO differs from the Policy Pilot v1 "
            "oracle: "
            f"advantage_error={advantage_error:.17g}, "
            f"loss_error={loss_error:.17g}, "
            f"gradient_error={gradient_error:.17g}"
        ) from error

    return VerlPilotGRPOParityReport(
        verl_commit=SPIKE_CANDIDATE_VERL_COMMIT,
        objective_identity_sha256=spec.identity_sha256,
        selected_policy_token_count=int(response_mask.sum().item()),
        maximum_advantage_absolute_error=advantage_error,
        loss_absolute_error=loss_error,
        maximum_gradient_absolute_error=gradient_error,
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


@dataclass(frozen=True, slots=True)
class VerlPolicyLossCall:
    """The exact public ``register_policy_loss`` invocation.

    ``rollout_is_weights`` is veRL's optional rollout-correction importance
    weight. It is not a behavior log probability. Actual rollout log
    probabilities remain an independently validated DataProto/TransferQueue
    field and are not smuggled through this hook.
    """

    old_log_prob: torch.Tensor
    log_prob: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    loss_agg_mode: str
    config: object
    rollout_is_weights: torch.Tensor | None

    def __post_init__(self) -> None:
        tensors = (
            self.old_log_prob,
            self.log_prob,
            self.advantages,
            self.response_mask,
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
        if self.rollout_is_weights is not None:
            if not isinstance(self.rollout_is_weights, torch.Tensor):
                raise TypeError("rollout_is_weights must be a tensor when present")
            if self.rollout_is_weights.shape != shape:
                raise ValueError(
                    "rollout_is_weights must share [batch, response] shape"
                )
            if self.rollout_is_weights.device != self.log_prob.device:
                raise ValueError("rollout_is_weights must share the policy device")
            if not self.rollout_is_weights.dtype.is_floating_point:
                raise TypeError("rollout_is_weights must use a floating dtype")
            if not bool(torch.isfinite(self.rollout_is_weights.detach()).all().item()):
                raise ValueError("rollout_is_weights must be finite")
            if not bool((self.rollout_is_weights.detach() >= 0).all().item()):
                raise ValueError("rollout_is_weights must be non-negative")


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
        rollout_is_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        call = VerlPolicyLossCall(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
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
