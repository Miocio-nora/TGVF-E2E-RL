"""Record the original-image visual state before any tool call occurs."""

from __future__ import annotations

import torch

from tgvf_rl.observations.schema import (
    TRAJECTORY_SOURCE_VISUAL_SCHEMA_V2,
    SourceVisualState,
    TrajectorySourceVisual,
    TrajectorySourceVisualV2,
)
from tgvf_rl.observations.store import ObservationStore, tensor_checksum

from .focus_tool import SourceVisualTensorBundle


def record_trajectory_source_visual(
    *,
    trajectory_id: str,
    source_visual: SourceVisualTensorBundle,
    source_positions: tuple[int, ...],
    deepstack_branch_layers: tuple[int, ...],
    deepstack_injection_positions: tuple[tuple[int, ...], ...],
    observation_store: ObservationStore,
    preprocessed_pixel_values: torch.Tensor | None = None,
    source_rgb: torch.Tensor | None = None,
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
    if preprocessed_pixel_values is not None:
        _validate_preprocessed_pixel_values(
            preprocessed_pixel_values,
            source_visual,
        )
    if source_rgb is not None and (
        not isinstance(source_rgb, torch.Tensor)
        or source_rgb.dtype != torch.uint8
        or source_rgb.ndim != 3
        or source_rgb.shape[-1] != 3
        or source_rgb.shape[0] <= 0
        or source_rgb.shape[1] <= 0
    ):
        raise ValueError("source_rgb must be RGB uint8 [H,W,3]")
    decoded_rgb_sha256 = source_visual.decoded_rgb_sha256
    if source_rgb is not None:
        actual_rgb_sha256 = tensor_checksum(source_rgb)
        if decoded_rgb_sha256 is None:
            raise ValueError(
                "source visual must identify the decoded RGB used to produce it"
            )
        if decoded_rgb_sha256 != actual_rgb_sha256:
            raise ValueError(
                "source visual features and immutable decoded RGB are not bound"
            )

    prefix = f"trajectory.{trajectory_id}.source"
    source_pixels = (
        observation_store.put_tensor(
            f"{prefix}.rgb",
            source_rgb,
            trajectory_id=trajectory_id,
        )
        if source_rgb is not None
        else None
    )
    pixel_values = (
        observation_store.put_tensor(
            f"{prefix}.pixel_values",
            preprocessed_pixel_values,
            trajectory_id=trajectory_id,
        )
        if preprocessed_pixel_values is not None
        else None
    )
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
        decoded_rgb_sha256=decoded_rgb_sha256,
    )
    common = {
        "state": state,
        "positions": source_positions,
        "deepstack_branch_layers": deepstack_branch_layers,
        "deepstack_injection_positions": deepstack_injection_positions,
        "source_pixels": source_pixels,
    }
    if pixel_values is None:
        return TrajectorySourceVisual(**common)
    return TrajectorySourceVisualV2(
        **common,
        preprocessed_pixel_values=pixel_values,
        schema_version=TRAJECTORY_SOURCE_VISUAL_SCHEMA_V2,
    )


def _validate_preprocessed_pixel_values(
    pixel_values: torch.Tensor,
    source_visual: SourceVisualTensorBundle,
) -> None:
    if (
        not isinstance(pixel_values, torch.Tensor)
        or pixel_values.ndim != 2
        or not pixel_values.is_floating_point()
        or pixel_values.shape[0] <= 0
        or pixel_values.shape[1] <= 0
    ):
        raise ValueError("preprocessed_pixel_values must be floating [N, patch_dim]")
    expected_tokens = 1
    for value in source_visual.image_grid_thw:
        expected_tokens *= value
    if int(pixel_values.shape[0]) != expected_tokens:
        raise ValueError(
            "preprocessed_pixel_values rows differ from source image_grid_thw"
        )
    premerge = source_visual.premerge_main
    if premerge.ndim not in {2, 3}:
        raise ValueError("source pre-merge main must have shape [N,H] or [1,N,H]")
    if int(pixel_values.shape[0]) != int(premerge.shape[-2]):
        raise ValueError(
            "preprocessed_pixel_values rows differ from source pre-merge tokens"
        )


__all__ = ["record_trajectory_source_visual"]
