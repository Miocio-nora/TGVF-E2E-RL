"""Method-matrix actor reduction behind veRL's forced bypass registry.

Pinned veRL's ``apply_bypass_mode`` always replaces
``actor.policy_loss.loss_mode`` with ``"bypass_mode"`` immediately before the
actor update.  Method runs therefore register this dispatcher under that exact
name.  The explicit ``actor.use_dynamic_bsz`` contract selects one of two
different mathematical identities:

* fixed micros retain the released DeepEyes equal-micro reduction;
* variable micros use a global policy-token mean.

The two underlying public losses keep their own strict registry checks.  This
module uses read-only config views to present the semantic identity selected by
the dispatcher, without mutating the actor config a second time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

from .deepeyes_actor_loss import (
    DEEPEYES_OFFICIAL_POLICY_LOSS_MODE,
    compute_deepeyes_official_micro_token_mean_loss,
)
from .dynamic_token_actor_loss import compute_dynamic_global_token_mean_loss
from .dynamic_token_loss_contract import (
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    METHOD_MATRIX_BYPASS_LOSS_MODULE,
    METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME,
)


def _value(container: object, name: str, default: object = None) -> object:
    if isinstance(container, Mapping):
        return container.get(name, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(container, name, default)


class _PolicyLossIdentityView:
    def __init__(self, value: object, *, loss_mode: str) -> None:
        self._value = value
        self._loss_mode = loss_mode

    def get(self, name: str, default: object = None) -> object:
        if name == "loss_mode":
            return self._loss_mode
        return _value(self._value, name, default)

    def __getattr__(self, name: str) -> object:
        sentinel = object()
        value = self.get(name, sentinel)
        if value is sentinel:
            raise AttributeError(name)
        return value


class _ActorIdentityView:
    def __init__(self, value: object, *, loss_mode: str) -> None:
        self._value = value
        self._policy_loss = _PolicyLossIdentityView(
            _value(value, "policy_loss"),
            loss_mode=loss_mode,
        )

    def get(self, name: str, default: object = None) -> object:
        if name == "policy_loss":
            return self._policy_loss
        return _value(self._value, name, default)

    def __getattr__(self, name: str) -> object:
        sentinel = object()
        value = self.get(name, sentinel)
        if value is sentinel:
            raise AttributeError(name)
        return value


def _require_exact_bypass(config: object) -> bool:
    if config is None:
        raise ValueError("method bypass loss requires an actor config")
    policy_loss = _value(config, "policy_loss")
    if _value(policy_loss, "loss_mode") != METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME:
        raise ValueError("method bypass loss requires veRL's bypass registry identity")
    rollout_correction = _value(policy_loss, "rollout_correction")
    expected = {
        "bypass_mode": True,
        "loss_type": "ppo_clip",
        "rollout_is": None,
        "rollout_rs": None,
        "rollout_is_batch_normalize": False,
    }
    mismatches = {
        name: (_value(rollout_correction, name), required)
        for name, required in expected.items()
        if _value(rollout_correction, name) != required
    }
    if mismatches:
        raise ValueError(
            f"method bypass rollout-correction config differs: {mismatches!r}"
        )
    dynamic = _value(config, "use_dynamic_bsz")
    if type(dynamic) is not bool:
        raise TypeError("method bypass requires explicit actor.use_dynamic_bsz")
    return dynamic


def compute_method_matrix_bypass_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: object = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dispatch the forced bypass name to the explicitly bound reduction."""

    dynamic = _require_exact_bypass(config)
    semantic_loss_mode = (
        DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE
        if dynamic
        else DEEPEYES_OFFICIAL_POLICY_LOSS_MODE
    )
    semantic_config = _ActorIdentityView(
        config,
        loss_mode=semantic_loss_mode,
    )
    loss_fn = (
        compute_dynamic_global_token_mean_loss
        if dynamic
        else compute_deepeyes_official_micro_token_mean_loss
    )
    loss, metrics = loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=semantic_config,
        rollout_is_weights=rollout_is_weights,
    )
    metrics["actor/tgvf_dynamic_token_batching"] = float(dynamic)
    return loss, metrics


def register_method_matrix_bypass_loss() -> Callable[..., object]:
    """Install the dispatcher, including after another external-lib reload."""

    return register_policy_loss(METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME)(
        compute_method_matrix_bypass_loss
    )


METHOD_MATRIX_BYPASS_LOSS = register_method_matrix_bypass_loss()


__all__ = [
    "METHOD_MATRIX_BYPASS_LOSS",
    "METHOD_MATRIX_BYPASS_LOSS_MODULE",
    "METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME",
    "compute_method_matrix_bypass_loss",
    "register_method_matrix_bypass_loss",
]
