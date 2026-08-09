"""Split trainable RP66 state from the standard full-Qwen rollout stream.

Pinned veRL/vLLM understands Qwen parameter names only.  RP66 is attached to
the actor before FSDP so it appears in the same full state stream; this module
removes that one prefix, publishes the complete Adapter-owned mapping under the
existing behavior-version rendezvous, and leaves every Qwen tensor unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError

from .policy_weight_sync import (
    PolicyWeightSyncRequest,
    PolicyWeightSyncState,
    _distributed_identity,
    _publish_lora_snapshot,
    _snapshot_tensor,
    load_latest_lora_snapshot,
    load_policy_weight_sync_request,
)


TRAINABLE_RP66_STATE_PREFIX = "tgvf_adapter."
TRAINABLE_RP66_REQUIRED_WORLD_SIZE = 8


@dataclass(frozen=True, slots=True)
class TrainableRP66Snapshot:
    """One exact rank-zero RP66-owned snapshot for rollout publication."""

    optimizer_step: int
    request_sha256: str
    storage_sha256: str
    tensors: Mapping[str, torch.Tensor]


def split_trainable_rp66_parameter_stream_for_snapshot(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    base_sync_done: bool,
    rank: int | None = None,
    world_size: int | None = None,
    global_steps: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield only Qwen weights and commit the complete RP66-owned state.

    The behavior-version snapshot storage envelope is reused for compatibility.
    Its tensor payload is RP66 Adapter-owned state in this experiment, never
    decoder LoRA state.  ``base_sync_done`` is accepted only because it is part
    of veRL's stream API: without policy LoRA, both values denote the one full
    model stream and therefore both must publish RP66 on rank zero.
    """

    if type(base_sync_done) is not bool:
        raise TypeError("base_sync_done must be bool")
    resolved_rank, resolved_world_size = _distributed_identity(
        rank=rank,
        world_size=world_size,
        environment=environment,
    )
    if resolved_world_size != TRAINABLE_RP66_REQUIRED_WORLD_SIZE:
        raise ValueError("PRL15 trainable RP66 sync requires exactly eight actor ranks")
    state = PolicyWeightSyncState.from_environment(environment)
    request = load_policy_weight_sync_request(state)
    if global_steps is not None and request.optimizer_step != global_steps:
        raise ValueError("RP66 stream optimizer step differs from sync request")
    return _split_parameter_iterator(
        iter(weights),
        state=state,
        request=request,
        publish=resolved_rank == 0,
    )


def load_latest_trainable_rp66_snapshot(
    state: PolicyWeightSyncState,
    *,
    expected_optimizer_step: int,
    expected_request_sha256: str,
) -> TrainableRP66Snapshot:
    """Load the exact RP66 payload emitted during one upstream Qwen sync."""

    try:
        snapshot = load_latest_lora_snapshot(
            state,
            expected_optimizer_step=expected_optimizer_step,
            expected_request_sha256=expected_request_sha256,
        )
    except ReplayMismatchError as error:
        raise ReplayMismatchError(
            "latest trainable RP66 snapshot is unavailable or invalid"
        ) from error
    return TrainableRP66Snapshot(
        optimizer_step=snapshot.policy_version.optimizer_step,
        request_sha256=snapshot.request_sha256,
        storage_sha256=snapshot.policy_version.weights_sha256,
        tensors=snapshot.tensors,
    )


def _split_parameter_iterator(
    iterator: Iterator[tuple[str, torch.Tensor]],
    *,
    state: PolicyWeightSyncState,
    request: PolicyWeightSyncRequest,
    publish: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    qwen_count = 0
    for item in iterator:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("full-model parameter items must be (name, tensor)")
        name, tensor = item
        if not isinstance(name, str) or not name:
            raise ValueError("full-model parameter names must be non-empty")
        marker = name.find(TRAINABLE_RP66_STATE_PREFIX)
        if marker >= 0 and (
            marker == 0 or name[marker - 1] == "."
        ):
            owned_name = name[marker + len(TRAINABLE_RP66_STATE_PREFIX) :]
            if not owned_name or owned_name in captured:
                raise ValueError("RP66 stream contains an invalid/duplicate name")
            if publish:
                captured[owned_name] = _snapshot_tensor(tensor)
            continue
        qwen_count += 1
        yield item
    if qwen_count == 0:
        raise RuntimeError("full-model sync stream contained no Qwen parameters")
    if publish:
        if not captured:
            raise RuntimeError("full-model sync stream contained no RP66 parameters")
        _publish_lora_snapshot(state, request=request, tensors=captured)


__all__ = [
    "TRAINABLE_RP66_REQUIRED_WORLD_SIZE",
    "TRAINABLE_RP66_STATE_PREFIX",
    "TrainableRP66Snapshot",
    "load_latest_trainable_rp66_snapshot",
    "split_trainable_rp66_parameter_stream_for_snapshot",
]
