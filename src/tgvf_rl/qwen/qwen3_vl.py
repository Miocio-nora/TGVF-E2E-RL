"""Qwen3-VL exact-D replay adapter."""

from __future__ import annotations

from typing import Any

from tgvf_rl.contracts.identity import SupportLevel

from .base import (
    FamilyCapabilities,
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
    materialize_deepstack,
    materialize_inputs_embeds,
    resolve_replay_request,
    resolve_language_model,
    resolve_lm_head,
)


class Qwen3VLAdapter(QwenVLMFamilyAdapter):
    capabilities = FamilyCapabilities(
        family="qwen3_vl",
        support_level=SupportLevel.EXECUTABLE,
        native_thinking_prefill=True,
        deepstack_branch_count=3,
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
        if any(
            len(block.deepstack) != self.capabilities.deepstack_branch_count
            for block in request.visual_blocks
        ):
            raise ValueError(
                f"Qwen3 replay requires {self.capabilities.deepstack_branch_count} exact DeepStack branches for every visual block"
            )
        inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
        deepstack = materialize_deepstack(request, visual_mask)
        language_model = resolve_language_model(model)
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=request.attention_mask.to(inputs_embeds.device),
            position_ids=request.position_ids.to(inputs_embeds.device),
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
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
