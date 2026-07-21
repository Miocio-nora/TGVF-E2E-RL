"""Concrete Qwen3 native visual layout for crop and crop+TGVF calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropTGVFObservationRecord,
    FocusedObservationRecord,
    TrajectorySourceVisual,
    VisualLayout,
)
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.protocol.schema import (
    NativeToolCall,
    ParsedCropTGVFCall,
    ParsedImageZoomInCall,
)

from .agent_loop import ToolExecutionContext
from .crop_tool import CropReplayLayout, CropVisualTensorBundle
from .focus_tool import ReplayLayoutTensors, SourceVisualTensorBundle
from .native_appender import (
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    render_qwen_native_success_environment_text,
)


QWEN3_DEEPSTACK_BRANCH_LAYERS = (8, 16, 24)


@dataclass(frozen=True, slots=True)
class _ExpandedNativeVisualLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    existing_positions: tuple[tuple[int, ...], ...]
    new_positions: tuple[int, ...]


class BoundQwen3CropTGVFLayoutBuilder:
    """Bind one exact sampled call to the low-level atomic executor port."""

    def __init__(
        self,
        owner: "Qwen3NativeToolLayoutBuilder",
        context: ToolExecutionContext,
    ) -> None:
        self.owner = owner
        self.context = context

    def build(
        self,
        *,
        trajectory_id: str,
        call_index: int,
        parsed_call: object,
        trajectory_source_visual: TrajectorySourceVisual,
        crop_visual: SourceVisualTensorBundle,
    ) -> ReplayLayoutTensors:
        context = self.context
        if trajectory_id != context.trajectory_identity.canonical_id:
            raise IdentityMismatchError("bound atomic layout trajectory changed")
        if call_index != context.call_index:
            raise IdentityMismatchError("bound atomic layout call index changed")
        if not isinstance(parsed_call, ParsedCropTGVFCall):
            raise TypeError("bound atomic layout requires ParsedCropTGVFCall")
        _validate_parsed_call_context(parsed_call, context)
        if trajectory_source_visual != context.trajectory_source_visual:
            raise IdentityMismatchError("bound atomic layout source visual changed")
        return self.owner.build_crop_tgvf(context, crop_visual, parsed_call)


class Qwen3NativeToolLayoutBuilder:
    """Expand native image placeholders and run Qwen3's exact M-RoPE helper.

    The initial source block may already contain repeated image-pad IDs, while
    each environment-appended tool response contains one canonical image-pad
    ID.  Expansion is therefore performed per vision-start/end block using the
    exact recorded grid for each ordered visual item.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_identity: ModelIdentity,
        observation_store: ObservationStore,
        get_rope_index: Callable[..., object],
    ) -> None:
        if not isinstance(model_identity, ModelIdentity):
            raise TypeError("Qwen3 layout builder requires a ModelIdentity")
        if model_identity.family != "qwen3_vl":
            raise ValueError("Qwen3 layout builder received another model family")
        if not isinstance(observation_store, ObservationStore):
            raise TypeError("layout builder requires an ObservationStore")
        if not callable(getattr(tokenizer, "encode", None)) or not callable(
            getattr(tokenizer, "convert_tokens_to_ids", None)
        ):
            raise TypeError("Qwen3 layout builder requires a tokenizer")
        if getattr(tokenizer, "name_or_path", None) != model_identity.revision_or_path:
            raise ValueError("layout tokenizer path differs from ModelIdentity")
        if not callable(get_rope_index):
            raise TypeError("Qwen3 layout builder requires get_rope_index()")
        self.tokenizer = tokenizer
        self.model_identity = model_identity
        self.store = observation_store
        self.get_rope_index = get_rope_index
        self.vision_start_id = self._native_token_id("<|vision_start|>")
        self.image_pad_id = self._native_token_id("<|image_pad|>")
        self.vision_end_id = self._native_token_id("<|vision_end|>")
        placeholder_ids = tuple(
            tokenizer.encode(
                QWEN_NATIVE_IMAGE_PLACEHOLDER,
                add_special_tokens=False,
            )
        )
        if placeholder_ids != (
            self.vision_start_id,
            self.image_pad_id,
            self.vision_end_id,
        ):
            raise ValueError("Qwen3 native image placeholder is not three atomic IDs")

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        processor: Any,
        model_identity: ModelIdentity,
        observation_store: ObservationStore,
    ) -> "Qwen3NativeToolLayoutBuilder":
        core = getattr(model, "model", None)
        get_rope_index = getattr(core, "get_rope_index", None)
        if not callable(get_rope_index):
            raise TypeError("Qwen3 model core must expose get_rope_index()")
        tokenizer = getattr(processor, "tokenizer", processor)
        return cls(
            tokenizer=tokenizer,
            model_identity=model_identity,
            observation_store=observation_store,
            get_rope_index=get_rope_index,
        )

    def bind_crop_tgvf(
        self, context: ToolExecutionContext
    ) -> BoundQwen3CropTGVFLayoutBuilder:
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("atomic layout binding requires ToolExecutionContext")
        self._validate_context(context)
        return BoundQwen3CropTGVFLayoutBuilder(self, context)

    def build_crop_tgvf(
        self,
        context: ToolExecutionContext,
        crop_visual: SourceVisualTensorBundle,
        parsed_call: ParsedCropTGVFCall,
    ) -> ReplayLayoutTensors:
        if not isinstance(crop_visual, SourceVisualTensorBundle):
            raise TypeError("atomic layout requires crop SourceVisualTensorBundle")
        if not isinstance(parsed_call, ParsedCropTGVFCall):
            raise TypeError("atomic layout requires ParsedCropTGVFCall")
        _validate_parsed_call_context(parsed_call, context)
        expanded = self._expand(
            context,
            new_grid=crop_visual.image_grid_thw,
            new_merge_size=crop_visual.spatial_merge_size,
            parsed_call=parsed_call,
        )
        branch_count = len(crop_visual.merged_deepstack)
        if branch_count != len(QWEN3_DEEPSTACK_BRANCH_LAYERS):
            raise ValueError("atomic crop visual is missing Qwen3 DeepStack branches")
        positions = expanded.new_positions
        sequence = int(expanded.input_ids.shape[-1])
        visible = torch.ones((1, sequence), dtype=torch.bool)
        return ReplayLayoutTensors(
            position_ids=expanded.position_ids,
            attention_mask=expanded.attention_mask,
            policy_visible_mask=visible.clone(),
            reference_visible_mask=visible.clone(),
            teacher_visible_mask=visible.clone(),
            token_type_ids=None,
            original_image_key_block_mask=None,
            cache_position=None,
            rope_delta=None,
            visual_layout=VisualLayout(
                sequence_length=sequence,
                original_image_positions=context.trajectory_source_visual.positions,
                d_positions=positions,
                deepstack_branch_layers=QWEN3_DEEPSTACK_BRANCH_LAYERS,
                deepstack_injection_positions=tuple(
                    positions for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS
                ),
            ),
            cache_mode="no_cache",
            cache_prefix_length=0,
        )

    def build_crop(
        self,
        context: ToolExecutionContext,
        crop_visual: CropVisualTensorBundle,
        parsed_call: ParsedImageZoomInCall,
    ) -> CropReplayLayout:
        if not isinstance(crop_visual, CropVisualTensorBundle):
            raise TypeError("plain crop layout requires CropVisualTensorBundle")
        if not isinstance(parsed_call, ParsedImageZoomInCall):
            raise TypeError("plain crop layout requires ParsedImageZoomInCall")
        _validate_parsed_call_context(parsed_call, context)
        if crop_visual.deepstack_branch_layers != QWEN3_DEEPSTACK_BRANCH_LAYERS:
            raise ValueError("plain crop visual has incompatible DeepStack layers")
        expanded = self._expand(
            context,
            new_grid=crop_visual.image_grid_thw,
            new_merge_size=crop_visual.spatial_merge_size,
            parsed_call=parsed_call,
        )
        positions = expanded.new_positions
        return CropReplayLayout(
            sequence_length=int(expanded.input_ids.shape[-1]),
            original_image_positions=context.trajectory_source_visual.positions,
            crop_positions=positions,
            deepstack_injection_positions=tuple(
                positions for _ in QWEN3_DEEPSTACK_BRANCH_LAYERS
            ),
        )

    def _expand(
        self,
        context: ToolExecutionContext,
        *,
        new_grid: tuple[int, int, int],
        new_merge_size: int,
        parsed_call: NativeToolCall,
    ) -> _ExpandedNativeVisualLayout:
        self._validate_context(context)
        _validate_parsed_call_context(parsed_call, context)
        expected = [
            (
                context.trajectory_source_visual.state.image_grid_thw,
                context.trajectory_source_visual.state.spatial_merge_size,
                context.trajectory_source_visual.positions,
            )
        ]
        for expected_call, handle in enumerate(context.prior_observation_handles):
            record = self.store.resolve_record(handle)
            if record.call_index != expected_call:
                raise ReplayMismatchError("prior visual observations are out of order")
            grid, merge_size, positions = _record_visual_geometry(record)
            expected.append((grid, merge_size, positions))
        expected.append((new_grid, new_merge_size, None))

        environment_success_token_ids = tuple(
            self.tokenizer.encode(
                render_qwen_native_success_environment_text(parsed_call),
                add_special_tokens=False,
            )
        )
        if not environment_success_token_ids:
            raise ValueError("Qwen3 native success response encoded to no tokens")
        canonical_ids = (
            context.prompt_token_ids_before_turn
            + context.sampled_turn.token_ids
            + environment_success_token_ids
        )
        blocks = _vision_blocks(
            canonical_ids,
            vision_start_id=self.vision_start_id,
            image_pad_id=self.image_pad_id,
            vision_end_id=self.vision_end_id,
        )
        if len(blocks) != len(expected):
            raise ReplayMismatchError(
                "native visual block count differs from source/prior/new observations"
            )

        expanded: list[int] = []
        expanded_positions: list[tuple[int, ...]] = []
        cursor = 0
        for block, (grid, merge_size, recorded_positions) in zip(
            blocks, expected, strict=True
        ):
            start, pad_start, pad_end, end = block
            expanded.extend(canonical_ids[cursor:pad_start])
            count = _merged_token_count(grid, merge_size)
            positions = tuple(range(len(expanded), len(expanded) + count))
            expanded.extend((self.image_pad_id,) * count)
            expanded.extend(canonical_ids[pad_end:end])
            expanded_positions.append(positions)
            if recorded_positions is not None and positions != recorded_positions:
                raise ReplayMismatchError(
                    "expanded native visual positions differ from rollout record"
                )
            cursor = end
        expanded.extend(canonical_ids[cursor:])

        input_ids = torch.tensor((tuple(expanded),), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        grids = torch.tensor(tuple(item[0] for item in expected), dtype=torch.long)
        with torch.no_grad():
            result = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=grids,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Qwen3 get_rope_index must return positions and delta")
        position_ids, _rope_delta = result
        if not isinstance(position_ids, torch.Tensor) or (
            position_ids.ndim not in {2, 3}
            or position_ids.shape[-2:] != input_ids.shape
        ):
            raise ValueError("Qwen3 native position IDs do not align with input IDs")
        return _ExpandedNativeVisualLayout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids.detach().clone(),
            existing_positions=tuple(expanded_positions[:-1]),
            new_positions=expanded_positions[-1],
        )

    def _native_token_id(self, token: str) -> int:
        value = self.tokenizer.convert_tokens_to_ids(token)
        if type(value) is not int or value < 0:
            raise ValueError(f"Qwen3 native token {token!r} has no integer ID")
        return value

    def _validate_context(self, context: ToolExecutionContext) -> None:
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("Qwen3 native layout requires ToolExecutionContext")
        if context.model != self.model_identity:
            raise IdentityMismatchError("native layout model identity changed")
        if len(context.prior_observation_handles) != context.call_index:
            raise ReplayMismatchError("native layout prior-call count changed")


def _validate_parsed_call_context(
    parsed_call: NativeToolCall,
    context: ToolExecutionContext,
) -> None:
    if not isinstance(
        parsed_call,
        (ParsedImageZoomInCall, ParsedCropTGVFCall),
    ):
        raise TypeError("Qwen3 crop layout requires a parsed crop call")
    sampled = context.sampled_turn
    if (
        parsed_call.sampled_text != sampled.text
        or parsed_call.sampled_token_ids != sampled.token_ids
        or parsed_call.sampled_token_byte_spans != sampled.token_byte_spans
    ):
        raise ReplayMismatchError(
            "native layout parsed call differs from sampled assistant turn"
        )


def _record_visual_geometry(
    record: FocusedObservationRecord | CropObservationRecord | CropTGVFObservationRecord,
) -> tuple[tuple[int, int, int], int, tuple[int, ...]]:
    if isinstance(record, FocusedObservationRecord):
        return (
            record.source_visual.image_grid_thw,
            record.source_visual.spatial_merge_size,
            record.layout.d_positions,
        )
    if isinstance(record, CropObservationRecord):
        return (
            record.crop_visual.image_grid_thw,
            record.crop_visual.spatial_merge_size,
            record.crop_visual.positions,
        )
    if isinstance(record, CropTGVFObservationRecord):
        return (
            record.crop_visual.source.image_grid_thw,
            record.crop_visual.source.spatial_merge_size,
            record.layout.d_positions,
        )
    raise TypeError("unknown recorded visual observation")


def _merged_token_count(grid: tuple[int, int, int], merge_size: int) -> int:
    if len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid):
        raise ValueError("native image grid must contain three positive integers")
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("native spatial merge size must be positive")
    if grid[1] % merge_size or grid[2] % merge_size:
        raise ValueError("native image grid is not merge divisible")
    return grid[0] * grid[1] * grid[2] // (merge_size**2)


def _vision_blocks(
    token_ids: Sequence[int],
    *,
    vision_start_id: int,
    image_pad_id: int,
    vision_end_id: int,
) -> tuple[tuple[int, int, int, int], ...]:
    blocks: list[tuple[int, int, int, int]] = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != vision_start_id:
            index += 1
            continue
        start = index
        pad_start = start + 1
        pad_end = pad_start
        while pad_end < len(token_ids) and token_ids[pad_end] == image_pad_id:
            pad_end += 1
        if pad_end == pad_start or (
            pad_end >= len(token_ids) or token_ids[pad_end] != vision_end_id
        ):
            raise ReplayMismatchError("malformed native image placeholder block")
        end = pad_end + 1
        blocks.append((start, pad_start, pad_end, end))
        index = end
    return tuple(blocks)


__all__ = [
    "BoundQwen3CropTGVFLayoutBuilder",
    "Qwen3NativeToolLayoutBuilder",
]
