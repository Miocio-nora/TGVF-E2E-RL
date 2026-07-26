"""Live ``image_zoom_in_tool`` runtime with immutable exact replay state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from threading import Condition
from typing import Callable, Protocol

from tgvf_rl.contracts.errors import RecoverableToolExecutionError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity
from tgvf_rl.observations.schema import TrajectorySourceVisual
from tgvf_rl.observations.store import ObservationHandle, ObservationStore
from tgvf_rl.protocol.schema import IMAGE_ZOOM_IN_TOOL_NAME, ParsedImageZoomInCall
from tgvf_rl.qwen.crop_coordinates import CropCoordinateMapper

from .agent_loop import ToolExecutionContext
from .crop_tool import (
    CropReplayLayout,
    CropToolExecutionRequest,
    CropVisualMaterializer,
    CropVisualTensorBundle,
    ImageZoomInTool,
)


@dataclass(slots=True)
class _CropLedgerEntry:
    fingerprint: str
    handle: ObservationHandle | None = None
    running: bool = True


class CropExecutionLedger:
    """Thread-safe execute-once ledger keyed by trajectory and call index."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._entries: dict[tuple[str, int], _CropLedgerEntry] = {}
        self._released_trajectory_ids: set[str] = set()

    def execute_once(
        self,
        *,
        key: tuple[str, int],
        fingerprint: str,
        operation: Callable[[], ObservationHandle],
    ) -> ObservationHandle:
        with self._condition:
            while True:
                if key[0] in self._released_trajectory_ids:
                    raise RuntimeError("crop execution batch has already been released")
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _CropLedgerEntry(fingerprint)
                    break
                if entry.fingerprint != fingerprint:
                    raise ValueError("crop call key was reused with different content")
                if not entry.running:
                    assert entry.handle is not None
                    return entry.handle
                self._condition.wait()
        try:
            handle = operation()
            if not isinstance(handle, ObservationHandle):
                raise TypeError("crop execution must return an ObservationHandle")
        except BaseException:
            with self._condition:
                self._entries.pop(key, None)
                self._condition.notify_all()
            raise
        with self._condition:
            entry = self._entries[key]
            entry.handle = handle
            entry.running = False
            self._condition.notify_all()
        return handle

    def assert_releasable(self, trajectory_ids: tuple[str, ...]) -> None:
        identities = _trajectory_id_set(trajectory_ids)
        with self._condition:
            running = tuple(
                key
                for key, entry in self._entries.items()
                if key[0] in identities and entry.running
            )
            if running:
                raise RuntimeError(
                    "cannot release a batch while crop execution is active"
                )

    def release_trajectories(self, trajectory_ids: tuple[str, ...]) -> int:
        identities = _trajectory_id_set(trajectory_ids)
        with self._condition:
            running = tuple(
                key
                for key, entry in self._entries.items()
                if key[0] in identities and entry.running
            )
            if running:
                raise RuntimeError(
                    "cannot release a batch while crop execution is active"
                )
            keys = tuple(key for key in self._entries if key[0] in identities)
            for key in keys:
                del self._entries[key]
            self._released_trajectory_ids.update(identities)
            self._condition.notify_all()
            return len(keys)

    def entry_count(self) -> int:
        with self._condition:
            return len(self._entries)


class CropRuntimeLayoutPort(Protocol):
    """Family-owned layout port bound to the exact tool execution context."""

    def build_crop(
        self,
        context: ToolExecutionContext,
        crop_visual: CropVisualTensorBundle,
        parsed_call: ParsedImageZoomInCall,
    ) -> CropReplayLayout: ...


@dataclass(frozen=True, slots=True)
class _BoundCropReplayLayoutBuilder:
    owner: CropRuntimeLayoutPort
    context: ToolExecutionContext
    parsed_call: ParsedImageZoomInCall

    def build(
        self,
        *,
        trajectory_id: str,
        call_index: int,
        parsed_call: ParsedImageZoomInCall,
        trajectory_source_visual: TrajectorySourceVisual,
        crop_visual: CropVisualTensorBundle,
    ) -> CropReplayLayout:
        context = self.context
        if trajectory_id != context.trajectory_identity.canonical_id:
            raise ValueError("bound crop layout trajectory identity changed")
        if call_index != context.call_index:
            raise ValueError("bound crop layout call index changed")
        if parsed_call != self.parsed_call:
            raise ValueError("bound crop layout parsed call changed")
        if trajectory_source_visual != context.trajectory_source_visual:
            raise ValueError("bound crop layout source visual changed")
        return self.owner.build_crop(context, crop_visual, parsed_call)


class ImageZoomInToolRuntime:
    """Concrete ``ToolRuntimePort`` for one plain crop observation.

    The runtime accepts the real Qwen crop materializer through its existing
    ``materialize()`` port.  The source is always the trajectory-owned RGB
    artifact, and the layout is built only after that one materialization has
    revealed the exact crop visual-token geometry.
    """

    def __init__(
        self,
        *,
        model: ModelIdentity,
        materializer: CropVisualMaterializer,
        layout_builder: CropRuntimeLayoutPort,
        observation_store: ObservationStore,
        crop_processor_identity: ArtifactIdentity,
        crop_layout_identity: ArtifactIdentity,
        execution_ledger: CropExecutionLedger,
        coordinate_mapper: CropCoordinateMapper,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> None:
        if not isinstance(model, ModelIdentity):
            raise TypeError("plain crop runtime requires a ModelIdentity")
        if not callable(getattr(materializer, "materialize", None)):
            raise TypeError("plain crop runtime requires materialize()")
        if not callable(getattr(layout_builder, "build_crop", None)):
            raise TypeError("plain crop runtime requires a layout builder")
        if not isinstance(observation_store, ObservationStore):
            raise TypeError("plain crop runtime requires an ObservationStore")
        if not isinstance(crop_processor_identity, ArtifactIdentity) or not isinstance(
            crop_layout_identity, ArtifactIdentity
        ):
            raise TypeError("plain crop runtime identities must be explicit")
        if not isinstance(execution_ledger, CropExecutionLedger):
            raise TypeError("plain crop runtime requires a CropExecutionLedger")
        if not callable(getattr(coordinate_mapper, "map_crop_bbox_to_source", None)):
            raise TypeError("plain crop runtime requires an explicit coordinate mapper")
        bound_model = getattr(materializer, "model_identity", None)
        if bound_model is not None and bound_model != model:
            raise ValueError("crop materializer model differs from runtime model")
        layout_model = getattr(layout_builder, "model_identity", None)
        if layout_model is not None and layout_model != model:
            raise ValueError("crop layout model differs from runtime model")

        self.model = model
        self.materializer = materializer
        self.layout_builder = layout_builder
        self.observation_store = observation_store
        self.crop_processor_identity = crop_processor_identity
        self.crop_layout_identity = crop_layout_identity
        self.execution_ledger = execution_ledger
        self.coordinate_mapper = coordinate_mapper
        self.processor_resized_size = processor_resized_size
        self.crop_tool = ImageZoomInTool(
            materializer,
            observation_store,
            coordinate_mapper=coordinate_mapper,
            processor_resized_size=processor_resized_size,
        )

    def execute(
        self, parsed_call: object, context: ToolExecutionContext
    ) -> ObservationHandle:
        if not isinstance(parsed_call, ParsedImageZoomInCall):
            raise TypeError("plain crop runtime requires a ParsedImageZoomInCall")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("plain crop runtime requires ToolExecutionContext")
        if parsed_call.name != IMAGE_ZOOM_IN_TOOL_NAME:
            raise ValueError("plain crop runtime received another tool call")
        _validate_sampled_turn(parsed_call, context)
        if context.model != self.model:
            raise ValueError("plain crop runtime model differs from trajectory model")

        fingerprint = _call_fingerprint(
            parsed_call=parsed_call,
            context=context,
            crop_processor_identity=self.crop_processor_identity,
            crop_layout_identity=self.crop_layout_identity,
            coordinate_space=self.coordinate_mapper.crop_coordinate_space,
            coordinate_conversion_version=(
                self.coordinate_mapper.crop_coordinate_conversion_version
            ),
            processor_resized_size=self.processor_resized_size,
        )
        return self.execution_ledger.execute_once(
            key=(context.trajectory_identity.canonical_id, context.call_index),
            fingerprint=fingerprint,
            operation=lambda: self._execute_once(parsed_call, context),
        )

    def _execute_once(
        self,
        parsed_call: ParsedImageZoomInCall,
        context: ToolExecutionContext,
    ) -> ObservationHandle:
        try:
            result = self.crop_tool.execute(
                CropToolExecutionRequest(
                    trajectory_id=context.trajectory_identity.canonical_id,
                    call_index=context.call_index,
                    parsed_call=parsed_call,
                    trajectory_source_visual=context.trajectory_source_visual,
                    layout_builder=_BoundCropReplayLayoutBuilder(
                        self.layout_builder,
                        context,
                        parsed_call,
                    ),
                    model=context.model,
                    policy_version=context.behavior_policy,
                    crop_processor_identity=self.crop_processor_identity,
                    crop_layout_identity=self.crop_layout_identity,
                )
            )
        except ValueError as error:
            if _is_recoverable_crop_geometry_error(error):
                raise RecoverableToolExecutionError(str(error)) from error
            raise
        if not isinstance(result.handle, ObservationHandle):
            raise TypeError("plain crop tool returned an invalid observation handle")
        return result.handle


def _is_recoverable_crop_geometry_error(error: ValueError) -> bool:
    """Identify sampled crop geometries that the visual processor cannot use."""

    message = str(error)
    return any(
        marker in message
        for marker in (
            "bbox is empty after clamping",
            "model bbox must be non-empty",
            "converted source bbox must be non-empty",
            "Qwen3 crop coordinates must lie within 0..1000",
            "Qwen2.5-VL crop coordinates lie outside",
            "absolute aspect ratio must be smaller than",
        )
    )


def _validate_sampled_turn(
    parsed_call: ParsedImageZoomInCall,
    context: ToolExecutionContext,
) -> None:
    sampled = context.sampled_turn
    if (
        parsed_call.sampled_text != sampled.text
        or parsed_call.sampled_token_ids != sampled.token_ids
        or parsed_call.sampled_token_byte_spans != sampled.token_byte_spans
    ):
        raise ValueError("parsed crop call differs from the sampled assistant turn")


def _source_binding_sha256(source: TrajectorySourceVisual) -> str:
    payload = json.dumps(asdict(source), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _call_fingerprint(
    *,
    parsed_call: ParsedImageZoomInCall,
    context: ToolExecutionContext,
    crop_processor_identity: ArtifactIdentity,
    crop_layout_identity: ArtifactIdentity,
    coordinate_space: str,
    coordinate_conversion_version: str,
    processor_resized_size: tuple[int, int] | None,
) -> str:
    payload = {
        "trajectory_id": context.trajectory_identity.canonical_id,
        "assistant_turn_index": context.assistant_turn_index,
        "attempt_index": context.attempt_index,
        "call_index": context.call_index,
        "model": asdict(context.model),
        "behavior_policy": asdict(context.behavior_policy),
        "prompt_token_ids_before_turn": context.prompt_token_ids_before_turn,
        "conditioning_input_ids": context.conditioning_input_ids,
        "sampled_text": parsed_call.sampled_text,
        "sampled_token_ids": parsed_call.sampled_token_ids,
        "sampled_token_byte_spans": tuple(
            asdict(span) for span in parsed_call.sampled_token_byte_spans
        ),
        "raw_tool_call": parsed_call.raw_tool_call,
        "raw_json": parsed_call.raw_json,
        "bbox_2d": parsed_call.bbox_2d,
        "label": parsed_call.label,
        "source_binding_sha256": _source_binding_sha256(
            context.trajectory_source_visual
        ),
        "prior_observation_handles": tuple(
            asdict(handle) for handle in context.prior_observation_handles
        ),
        "crop_processor_identity": asdict(crop_processor_identity),
        "crop_layout_identity": asdict(crop_layout_identity),
        "coordinate_space": coordinate_space,
        "coordinate_conversion_version": coordinate_conversion_version,
        "processor_resized_size": processor_resized_size,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_id_set(trajectory_ids: tuple[str, ...]) -> tuple[str, ...]:
    identities = tuple(trajectory_ids)
    if not identities or any(
        not isinstance(identity, str) or not identity for identity in identities
    ):
        raise ValueError("trajectory_ids must contain non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("trajectory_ids must be unique")
    return identities


__all__ = [
    "CropExecutionLedger",
    "CropRuntimeLayoutPort",
    "ImageZoomInToolRuntime",
]
