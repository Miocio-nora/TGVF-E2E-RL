"""Atomic full-Qwen plus trainable-RP66 publication to rollout workers.

The upstream veRL checkpoint manager remains the sole owner of full Qwen
transport.  The actor engine removes RP66-owned tensors from that stream and
rank zero commits them to the request-scoped snapshot rendezvous.  This
manager waits for upstream Qwen synchronization, loads that exact RP66
snapshot, then waits for an exact ACK from every vLLM server before returning.

This is a dedicated PRL15 path.  It neither wraps the historical policy-LoRA
manager nor requires a policy LoRA configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any

from .policy_weight_sync import (
    PolicyWeightSyncState,
    _auto_await,
    _nonnegative_step,
    publish_policy_weight_sync_request,
)
from .trainable_tgvf_weight_sync import load_latest_trainable_rp66_snapshot
from .vllm_tool_runtime import (
    _validate_adapter_update_ack,
    adapter_owned_state_sha256,
    bind_tgvf_adapter_state_update_manager,
)


TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN = (
    "tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager."
    "TrainableTGVFCheckpointEngineManager"
)


@dataclass(frozen=True, slots=True)
class TrainableTGVFRolloutPublication:
    """Identity of one Qwen+RP66 rollout state accepted by every server."""

    optimizer_step: int
    adapter_state_sha256: str
    snapshot_storage_sha256: str
    request_sha256: str
    acknowledgement_count: int
    applied_count: int

    def __post_init__(self) -> None:
        _nonnegative_step(self.optimizer_step)
        for owner, value in (
            ("Adapter state", self.adapter_state_sha256),
            ("RP66 snapshot storage", self.snapshot_storage_sha256),
            ("RP66 sync request", self.request_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{owner} digest must be lowercase SHA256")
        if type(self.acknowledgement_count) is not int or self.acknowledgement_count <= 0:
            raise ValueError("RP66 publication requires positive ACK count")
        if (
            type(self.applied_count) is not int
            or self.applied_count < 0
            or self.applied_count > self.acknowledgement_count
        ):
            raise ValueError("RP66 publication applied count is invalid")


class TrainableTGVFCheckpointEngineManager:
    """Coordinate one full-Qwen sync and one RP66 fan-out per optimizer step."""

    def __init__(
        self,
        config: object,
        actor_wg: object,
        replicas: list[object],
        *,
        upstream_manager_factory: Callable[..., object] | None = None,
        rollout_manager_factory: Callable[[Sequence[object]], object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(replicas, list) or not replicas:
            raise ValueError("trainable RP66 sync requires rollout replicas")
        self._state = PolicyWeightSyncState.from_environment(environment)
        upstream_factory = (
            upstream_manager_factory or _load_upstream_checkpoint_manager_class()
        )
        self._upstream = upstream_factory(
            config=config,
            actor_wg=actor_wg,
            replicas=replicas,
        )
        if not callable(getattr(self._upstream, "update_weights", None)):
            raise TypeError("upstream checkpoint manager must implement update_weights()")

        rollout_factory = (
            rollout_manager_factory or bind_tgvf_adapter_state_update_manager
        )
        self._rollout_manager = rollout_factory(replicas)
        if not callable(
            getattr(self._rollout_manager, "update_adapter_owned_state", None)
        ):
            raise TypeError(
                "rollout manager must implement update_adapter_owned_state()"
            )
        self._expected_acknowledgement_count = _rollout_server_count(replicas)
        self._last_publication: TrainableTGVFRolloutPublication | None = None

    @property
    def last_publication(self) -> TrainableTGVFRolloutPublication | None:
        return self._last_publication

    @_auto_await
    async def update_weights(self, global_steps: int | None = None) -> object:
        """Block until Qwen and RP66 for exactly ``global_steps`` are accepted."""

        if global_steps is None:
            raise ValueError("trainable RP66 weight sync requires explicit global_steps")
        _nonnegative_step(global_steps)
        if (
            self._last_publication is not None
            and global_steps < self._last_publication.optimizer_step
        ):
            raise ValueError("trainable RP66 weight sync step moved backwards")

        request = publish_policy_weight_sync_request(self._state, global_steps)

        # Consuming the actor's full stream both updates Qwen through upstream
        # veRL and commits the rank-zero RP66 snapshot under ``request``.
        upstream_result = self._upstream.update_weights(global_steps=global_steps)
        if inspect.isawaitable(upstream_result):
            upstream_result = await upstream_result

        snapshot = load_latest_trainable_rp66_snapshot(
            self._state,
            expected_optimizer_step=global_steps,
            expected_request_sha256=request.request_sha256,
        )
        adapter_sha256 = adapter_owned_state_sha256(snapshot.tensors)
        acknowledgements = self._rollout_manager.update_adapter_owned_state(
            optimizer_step=global_steps,
            state_sha256=adapter_sha256,
            state=snapshot.tensors,
        )
        if inspect.isawaitable(acknowledgements):
            acknowledgements = await acknowledgements
        validated = _validate_all_acknowledgements(
            acknowledgements,
            optimizer_step=global_steps,
            state_sha256=adapter_sha256,
            tensor_count=len(snapshot.tensors),
            expected_count=self._expected_acknowledgement_count,
        )
        self._last_publication = TrainableTGVFRolloutPublication(
            optimizer_step=global_steps,
            adapter_state_sha256=adapter_sha256,
            snapshot_storage_sha256=snapshot.storage_sha256,
            request_sha256=request.request_sha256,
            acknowledgement_count=len(validated),
            applied_count=sum(bool(ack["applied"]) for ack in validated),
        )
        return upstream_result

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._upstream, name)


def _validate_all_acknowledgements(
    value: object,
    *,
    optimizer_step: int,
    state_sha256: str,
    tensor_count: int,
    expected_count: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("RP66 rollout publication ACKs must be a sequence")
    if len(value) != expected_count:
        raise RuntimeError(
            "RP66 rollout publication ACK count differs from rollout server count"
        )
    return tuple(
        _validate_adapter_update_ack(
            acknowledgement,
            expected_optimizer_step=optimizer_step,
            expected_state_sha256=state_sha256,
            expected_tensor_count=tensor_count,
        )
        for acknowledgement in value
    )


def _rollout_server_count(replicas: Sequence[object]) -> int:
    count = 0
    for replica in replicas:
        servers = getattr(replica, "servers", None)
        if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
            raise TypeError("each rollout replica must expose a server sequence")
        count += len(servers)
    if count <= 0:
        raise ValueError("trainable RP66 sync requires rollout servers")
    return count


def _load_upstream_checkpoint_manager_class() -> type[Any]:
    from verl.checkpoint_engine import CheckpointEngineManager

    return CheckpointEngineManager


__all__ = [
    "TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN",
    "TrainableTGVFCheckpointEngineManager",
    "TrainableTGVFRolloutPublication",
]
