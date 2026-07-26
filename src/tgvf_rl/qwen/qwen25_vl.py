"""Qwen2.5-VL main-D replay adapter with explicit no-DeepStack support."""

from __future__ import annotations

from typing import Any

import torch

from tgvf_rl.contracts.identity import SupportLevel

from .base import (
    FamilyCapabilities,
    InjectedForwardRequest,
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
    injected_request_from_recorded,
    materialize_inputs_embeds,
    resolve_replay_request,
    resolve_language_model,
    resolve_lm_head,
)
from .crop_coordinates import (
    QWEN25_CROP_CONVERSION_VERSION,
    QWEN25_CROP_COORDINATE_SPACE,
    CropCoordinateMapping,
    map_qwen25_crop_bbox_to_source,
)


class Qwen25VLAdapter(QwenVLMFamilyAdapter):
    crop_coordinate_space = QWEN25_CROP_COORDINATE_SPACE
    crop_coordinate_conversion_version = QWEN25_CROP_CONVERSION_VERSION
    capabilities = FamilyCapabilities(
        family="qwen2_5_vl",
        support_level=SupportLevel.SYNTHETIC,
        native_thinking_prefill=False,
        deepstack_branch_count=0,
        recorded_d_forward=True,
        native_tool_template=True,
    )

    def map_crop_bbox_to_source(
        self,
        bbox_2d: tuple[int, int, int, int],
        *,
        source_width: int,
        source_height: int,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> CropCoordinateMapping:
        return map_qwen25_crop_bbox_to_source(
            bbox_2d,
            source_width=source_width,
            source_height=source_height,
            processor_resized_size=processor_resized_size,
        )

    def forward_recorded(
        self,
        model: Any,
        store: Any,
        replay_handle: Any,
        consumer: ReplayConsumer,
    ) -> RecordedReplayResult:
        recorded = resolve_replay_request(store, replay_handle, consumer)
        return self.forward_injected(model, injected_request_from_recorded(recorded))

    def forward_injected(
        self,
        model: Any,
        request: InjectedForwardRequest,
    ) -> RecordedReplayResult:
        if not isinstance(request, InjectedForwardRequest):
            raise TypeError("request must be InjectedForwardRequest")
        if any(block.deepstack for block in request.visual_blocks):
            raise ValueError(
                "Qwen2.5-VL has no accepted DeepStack-equivalent replay contract"
            )
        inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
        language_model = resolve_language_model(model)
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=request.attention_mask.to(inputs_embeds.device),
            position_ids=request.position_ids.to(inputs_embeds.device),
            use_cache=request.use_cache,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        logits = resolve_lm_head(model)(hidden)
        return RecordedReplayResult(
            logits=logits,
            hidden_states=hidden,
            past_key_values=getattr(outputs, "past_key_values", None),
            visual_position_mask=visual_mask,
        )

    def materialize_representation_supervision(
        self,
        model: Any,
        tokenizer: Any,
        canonical: Any,
        model_input_ids: torch.Tensor,
    ) -> Any:
        """Fail until Qwen2.5-VL has its own accepted transcript/artifact fixture."""

        raise NotImplementedError(
            "Qwen2.5-VL representation supervision is blocked until its "
            "family-specific native transcript and representation artifact pass"
        )
