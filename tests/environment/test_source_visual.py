from __future__ import annotations

import torch

from tgvf_rl.environment import record_trajectory_source_visual
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.observations.schema import SourceVisualState
from tgvf_rl.observations.store import ObservationStore


def _bundle() -> SourceVisualTensorBundle:
    return SourceVisualTensorBundle(
        image_sha256="a" * 64,
        premerge_main=torch.arange(32, dtype=torch.float32).view(8, 4),
        premerge_deepstack=(
            torch.arange(32, 64, dtype=torch.float32).view(8, 4),
            torch.arange(64, 96, dtype=torch.float32).view(8, 4),
        ),
        merged_main=torch.arange(16, dtype=torch.float32).view(2, 8),
        merged_deepstack=(
            torch.arange(16, 32, dtype=torch.float32).view(2, 8),
            torch.arange(32, 48, dtype=torch.float32).view(2, 8),
        ),
        image_grid_thw=(1, 2, 4),
        spatial_merge_size=2,
    )


def _record(
    store: ObservationStore,
    source: SourceVisualTensorBundle,
    *,
    trajectory_id: str,
):
    return record_trajectory_source_visual(
        trajectory_id=trajectory_id,
        source_visual=source,
        source_positions=(3, 4),
        deepstack_branch_layers=(8, 16),
        deepstack_injection_positions=((3, 4), (3, 4)),
        observation_store=store,
    )


def _state_identity(state: SourceVisualState) -> tuple[object, ...]:
    """Match the replay store's content identity for focused observations."""

    return (
        state.image_sha256,
        state.premerge_main.address.digest,
        tuple(ref.address.digest for ref in state.premerge_deepstack),
        state.merged_main.address.digest,
        tuple(ref.address.digest for ref in state.merged_deepstack),
        state.image_grid_thw,
        state.spatial_merge_size,
    )


def test_records_complete_source_visual_without_a_tool_call() -> None:
    store = ObservationStore()
    source = _bundle()
    recorded = _record(store, source, trajectory_id="trajectory-0")

    torch.testing.assert_close(
        store.resolve_verified(recorded.state.premerge_main), source.premerge_main
    )
    torch.testing.assert_close(
        store.resolve_verified(recorded.state.merged_main), source.merged_main
    )
    for ref, expected in zip(
        recorded.state.premerge_deepstack,
        source.premerge_deepstack,
        strict=True,
    ):
        torch.testing.assert_close(store.resolve_verified(ref), expected)
    for ref, expected in zip(
        recorded.state.merged_deepstack,
        source.merged_deepstack,
        strict=True,
    ):
        torch.testing.assert_close(store.resolve_verified(ref), expected)
    assert recorded.positions == (3, 4)
    assert recorded.deepstack_branch_layers == (8, 16)
    assert recorded.deepstack_injection_positions == ((3, 4), (3, 4))


def test_recording_is_mutation_safe_and_content_deduplicated() -> None:
    store = ObservationStore()
    source = _bundle()
    pristine = _bundle()
    first = _record(store, source, trajectory_id="trajectory-0")
    second = _record(store, pristine, trajectory_id="trajectory-1")

    source.premerge_main.add_(1000)
    source.premerge_deepstack[0].zero_()
    source.merged_main.mul_(-1)
    source.merged_deepstack[1].fill_(99)

    torch.testing.assert_close(
        store.resolve_verified(first.state.premerge_main), pristine.premerge_main
    )
    torch.testing.assert_close(
        store.resolve_verified(first.state.premerge_deepstack[0]),
        pristine.premerge_deepstack[0],
    )
    torch.testing.assert_close(
        store.resolve_verified(first.state.merged_main), pristine.merged_main
    )
    torch.testing.assert_close(
        store.resolve_verified(first.state.merged_deepstack[1]),
        pristine.merged_deepstack[1],
    )

    # Names remain trajectory-scoped, while the exact identity used to compare
    # a replay source with FocusedObservationRecord.source_visual is identical.
    assert first.state.merged_main.name != second.state.merged_main.name
    assert _state_identity(first.state) == _state_identity(second.state)
