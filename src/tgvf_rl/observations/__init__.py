"""Immutable focused-observation schema and storage."""

from .schema import (
    CROP_TGVF_OBSERVATION_SCHEMA_V3,
    CropObservationRecord,
    CropTGVFObservationRecord,
    CropTGVFVisualState,
    CropVisualState,
    FocusedObservationRecord,
    SourceVisualState,
    TrajectorySourceVisual,
)
from .store import (
    ObservationHandle,
    ObservationReleaseCounts,
    ObservationStore,
    ReplayTensorPayload,
    TrajectoryReplayBundle,
    TrajectoryReplayHandle,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
    validate_replay_bundle,
)


_FINALIZER_EXPORTS = {
    "MaterializedTrajectoryReplayTensors",
    "TrajectoryReplayFinalizationRequest",
    "finalize_trajectory_replay",
}


def __getattr__(name: str):
    """Load trajectory-dependent finalizers without a cold-import cycle."""

    if name not in _FINALIZER_EXPORTS:
        raise AttributeError(name)
    from . import finalizer

    return getattr(finalizer, name)

__all__ = [
    "CROP_TGVF_OBSERVATION_SCHEMA_V3",
    "FocusedObservationRecord",
    "MaterializedTrajectoryReplayTensors",
    "CropObservationRecord",
    "CropTGVFObservationRecord",
    "CropTGVFVisualState",
    "CropVisualState",
    "SourceVisualState",
    "TrajectorySourceVisual",
    "ObservationHandle",
    "ObservationReleaseCounts",
    "ObservationStore",
    "ReplayTensorPayload",
    "TrajectoryReplayBundle",
    "TrajectoryReplayHandle",
    "TrajectoryReplayFinalizationRequest",
    "TrajectoryReplayRecord",
    "TrajectoryReplayTensorRefs",
    "finalize_trajectory_replay",
    "validate_replay_bundle",
]
