"""Upstream full-Qwen sync with a typed behavior-version receipt.

NoTool and Crop method-matrix arms have no decoder LoRA and no RP66 Adapter.
This wrapper leaves the upstream transport untouched, but brackets it with the
existing request rendezvous and publishes a behavior identity only after the
upstream ``update_weights`` call has completed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from .policy_behavior_version import (
    FullQwenSyncReceipt,
    PolicyBehaviorPayload,
    PolicyBehaviorSnapshot,
    publish_policy_behavior_snapshot,
)
from .policy_weight_sync import (
    PolicyWeightSyncState,
    _auto_await,
    _nonnegative_step,
    publish_policy_weight_sync_request,
)


FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN = (
    "tgvf_rl.framework.verl.full_qwen_checkpoint_manager."
    "FullQwenBehaviorCheckpointEngineManager"
)


class FullQwenBehaviorCheckpointEngineManager:
    """Publish the accepted full-Qwen behavior version for NoTool/Crop."""

    # A checkpoint save performs a level-2 rollout sleep and discards the
    # synchronized full model.  The outer lifecycle wrapper must publish the
    # same step again rather than waking a weightless replica.
    requires_post_checkpoint_weight_resync = True

    def __init__(
        self,
        config: object,
        actor_wg: object,
        replicas: list[object],
        *,
        upstream_manager_factory: Callable[..., object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        factory = upstream_manager_factory or _load_upstream_checkpoint_manager_class()
        self._state = PolicyWeightSyncState.from_environment(environment)
        self._upstream = factory(config=config, actor_wg=actor_wg, replicas=replicas)
        if not callable(getattr(self._upstream, "update_weights", None)):
            raise TypeError(
                "upstream checkpoint manager must implement update_weights()"
            )
        self._last_behavior_snapshot: PolicyBehaviorSnapshot | None = None

    @property
    def last_behavior_snapshot(self) -> PolicyBehaviorSnapshot | None:
        return self._last_behavior_snapshot

    @_auto_await
    async def update_weights(self, global_steps: int | None = None) -> object:
        if global_steps is None:
            raise ValueError("full-Qwen behavior sync requires explicit global_steps")
        _nonnegative_step(global_steps)
        if (
            self._last_behavior_snapshot is not None
            and global_steps
            < self._last_behavior_snapshot.policy_version.optimizer_step
        ):
            raise ValueError("full-Qwen behavior sync step moved backwards")
        request = publish_policy_weight_sync_request(self._state, global_steps)
        result = self._upstream.update_weights(global_steps=global_steps)
        if inspect.isawaitable(result):
            result = await result
        receipt = FullQwenSyncReceipt.from_acknowledged_request(request)
        self._last_behavior_snapshot = publish_policy_behavior_snapshot(
            self._state,
            full_qwen=receipt,
            payload=PolicyBehaviorPayload.FULL_QWEN,
        )
        return result

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._upstream, name)


def _load_upstream_checkpoint_manager_class() -> type[Any]:
    from verl.checkpoint_engine import CheckpointEngineManager

    return CheckpointEngineManager


__all__ = [
    "FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN",
    "FullQwenBehaviorCheckpointEngineManager",
]
