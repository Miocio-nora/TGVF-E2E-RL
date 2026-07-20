"""Record the original-image visual state before any tool call occurs."""

from __future__ import annotations

from tgvf_rl.observations.schema import SourceVisualState, TrajectorySourceVisual
from tgvf_rl.observations.store import ObservationStore

from .focus_tool import SourceVisualTensorBundle


def record_trajectory_source_visual(
    *,
    trajectory_id: str,
    source_visual: SourceVisualTensorBundle,
    source_positions: tuple[int, ...],
    deepstack_branch_layers: tuple[int, ...],
    deepstack_injection_positions: tuple[tuple[int, ...], ...],
    observation_store: ObservationStore,
) -> TrajectorySourceVisual:
    """Freeze an already-materialized source visual for exact trajectory replay.

    This boundary only records tensors supplied by prompt preparation. It does
    not run an image processor, a vision encoder, or a tool.
    """

    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    if not isinstance(source_visual, SourceVisualTensorBundle):
        raise TypeError("source_visual must be a SourceVisualTensorBundle")
    if not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")

    prefix = f"trajectory.{trajectory_id}.source"
    state = SourceVisualState(
        image_sha256=source_visual.image_sha256,
        premerge_main=observation_store.put_tensor(
            f"{prefix}.premerge.main",
            source_visual.premerge_main,
            trajectory_id=trajectory_id,
        ),
        premerge_deepstack=tuple(
            observation_store.put_tensor(
                f"{prefix}.premerge.deepstack.{index}",
                tensor,
                trajectory_id=trajectory_id,
            )
            for index, tensor in enumerate(source_visual.premerge_deepstack)
        ),
        merged_main=observation_store.put_tensor(
            f"{prefix}.merged.main",
            source_visual.merged_main,
            trajectory_id=trajectory_id,
        ),
        merged_deepstack=tuple(
            observation_store.put_tensor(
                f"{prefix}.merged.deepstack.{index}",
                tensor,
                trajectory_id=trajectory_id,
            )
            for index, tensor in enumerate(source_visual.merged_deepstack)
        ),
        image_grid_thw=source_visual.image_grid_thw,
        spatial_merge_size=source_visual.spatial_merge_size,
    )
    return TrajectorySourceVisual(
        state=state,
        positions=source_positions,
        deepstack_branch_layers=deepstack_branch_layers,
        deepstack_injection_positions=deepstack_injection_positions,
    )


__all__ = ["record_trajectory_source_visual"]
