"""Qwen2.5-VL main-D replay adapter with explicit no-DeepStack support."""

from __future__ import annotations

from typing import Any

from tgvf_rl.contracts.identity import SupportLevel

from .base import (
    FamilyCapabilities,
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
    materialize_inputs_embeds,
    resolve_replay_request,
    resolve_language_model,
    resolve_lm_head,
)


class Qwen25VLAdapter(QwenVLMFamilyAdapter):
    capabilities = FamilyCapabilities(
        family="qwen2_5_vl",
        support_level=SupportLevel.SYNTHETIC,
        native_thinking_prefill=False,
        deepstack_branch_count=0,
        recorded_d_forward=True,
        native_tool_template=True,
    )

    def forward_recorded(
        self,
        model: Any,
        store: Any,
        replay_handle: Any,
        consumer: ReplayConsumer,
    ) -> RecordedReplayResult:
        request = resolve_replay_request(store, replay_handle, consumer)
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
