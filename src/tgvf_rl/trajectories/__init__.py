"""Framework-neutral rollout records and actual behavior evidence."""

from .behavior import (
    BEHAVIOR_TRACE_SCHEMA_VERSION,
    BehaviorTraceHandle,
    BehaviorTraceRecord,
    BehaviorTraceStore,
    VLLMBehaviorRecorder,
    behavior_trace_checksum,
    verify_behavior_trace_pair,
)
from .schema import (
    CropToolCallRecord,
    NativeToolCallRecord,
    ToolCallRecord,
    ToolErrorRecord,
    TrajectoryRecord,
    trajectory_checksum,
)
from .validation import TrajectoryValidator

__all__ = [
    "BEHAVIOR_TRACE_SCHEMA_VERSION",
    "BehaviorTraceHandle",
    "BehaviorTraceRecord",
    "BehaviorTraceStore",
    "CropToolCallRecord",
    "NativeToolCallRecord",
    "ToolCallRecord",
    "ToolErrorRecord",
    "TrajectoryRecord",
    "TrajectoryValidator",
    "VLLMBehaviorRecorder",
    "behavior_trace_checksum",
    "trajectory_checksum",
    "verify_behavior_trace_pair",
]
