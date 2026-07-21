"""Qwen3-VL exact-D replay adapter."""

from __future__ import annotations

from typing import Any

import torch

from tgvf_rl.contracts.identity import SupportLevel
from tgvf_rl.representation.training.transcript import (
    CanonicalEvidenceSupervision,
    ModelEvidenceSupervision,
    _build_visual_token_expansion,
    _materialize_model_evidence_supervision,
)

from .base import (
    CachedTokenForwardRequest,
    FamilyCapabilities,
    InjectedForwardRequest,
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
    assert_model_vocabulary_compatible,
    injected_request_from_recorded,
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
        native_injected_kv_cache=True,
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
        if any(
            len(block.deepstack) != self.capabilities.deepstack_branch_count
            for block in request.visual_blocks
        ):
            raise ValueError(
                f"Qwen3 replay requires {self.capabilities.deepstack_branch_count} exact DeepStack branches for every visual block"
            )
        inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
        deepstack = materialize_deepstack(
            request,
            visual_mask,
            target_dtype=inputs_embeds.dtype,
        )
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

    def prefill_injected_cache(
        self,
        model: Any,
        request: InjectedForwardRequest,
    ) -> RecordedReplayResult:
        if not isinstance(request, InjectedForwardRequest):
            raise TypeError("request must be InjectedForwardRequest")
        if any(
            len(block.deepstack) != self.capabilities.deepstack_branch_count
            for block in request.visual_blocks
        ):
            raise ValueError(
                "Qwen3 cached prefill requires exact DeepStack branches for every visual block"
            )
        inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
        deepstack = materialize_deepstack(
            request,
            visual_mask,
            target_dtype=inputs_embeds.dtype,
        )
        language_model = resolve_language_model(model)
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=request.attention_mask.to(inputs_embeds.device),
            position_ids=request.position_ids.to(inputs_embeds.device),
            cache_position=cache_position,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            use_cache=True,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("Qwen3 cached prefill returned no past_key_values")
        return RecordedReplayResult(
            logits=resolve_lm_head(model)(hidden),
            hidden_states=hidden,
            past_key_values=past_key_values,
            visual_position_mask=visual_mask,
        )

    def forward_cached_token(
        self,
        model: Any,
        request: CachedTokenForwardRequest,
    ) -> RecordedReplayResult:
        if not isinstance(request, CachedTokenForwardRequest):
            raise TypeError("request must be CachedTokenForwardRequest")
        language_model = resolve_language_model(model)
        embedding_device = language_model.get_input_embeddings().weight.device
        outputs = language_model(
            input_ids=request.input_ids.to(embedding_device),
            inputs_embeds=None,
            attention_mask=request.attention_mask.to(embedding_device),
            position_ids=request.position_ids.to(embedding_device),
            past_key_values=request.past_key_values,
            cache_position=request.cache_position.to(embedding_device),
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            use_cache=True,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("Qwen3 cached token forward returned no past_key_values")
        visual_mask = torch.zeros(
            hidden.shape[:2],
            dtype=torch.bool,
            device=hidden.device,
        )
        return RecordedReplayResult(
            logits=resolve_lm_head(model)(hidden),
            hidden_states=hidden,
            past_key_values=past_key_values,
            visual_position_mask=visual_mask,
        )

    def materialize_representation_supervision(
        self,
        model: Any,
        tokenizer: Any,
        canonical: CanonicalEvidenceSupervision,
        model_input_ids: torch.Tensor,
    ) -> ModelEvidenceSupervision:
        """Map canonical labels across Qwen3 visual-placeholder expansion."""

        if not isinstance(canonical, CanonicalEvidenceSupervision):
            raise TypeError("canonical must be CanonicalEvidenceSupervision")
        sequence = _single_model_token_sequence(model_input_ids)
        self.assert_tokenizer_invariant(
            tokenizer, canonical.transcript.tokenizer_length
        )
        assert_model_vocabulary_compatible(
            model,
            tokenizer,
            expected_tokenizer_length=canonical.transcript.tokenizer_length,
            token_ids=sequence,
        )
        if not hasattr(tokenizer, "convert_tokens_to_ids") or not hasattr(
            tokenizer, "convert_ids_to_tokens"
        ):
            raise TypeError("Qwen3 tokenizer must expose token/id round trips")
        visual_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if isinstance(visual_token_id, bool) or not isinstance(visual_token_id, int):
            raise TypeError("Qwen3 image placeholder must resolve to one token id")
        if tokenizer.convert_ids_to_tokens(visual_token_id) != "<|image_pad|>":
            raise ValueError("Qwen3 image placeholder token does not round trip")
        if canonical.transcript.token_ids.count(visual_token_id) != 2:
            raise ValueError(
                "Qwen3 representation supervision requires exactly source-image "
                "and tool-observation placeholders"
            )
        expansion = _build_visual_token_expansion(
            family=self.capabilities.family,
            canonical_token_ids=canonical.transcript.token_ids,
            model_token_ids=tuple(int(token_id) for token_id in sequence.tolist()),
            visual_placeholder_token_id=visual_token_id,
        )
        result = _materialize_model_evidence_supervision(canonical, expansion)
        if len(result.visual_expansion_blocks) != 2:
            raise ValueError(
                "Qwen3 model input must preserve two ordered visual expansions"
            )
        self.assert_tokenizer_invariant(
            tokenizer, canonical.transcript.tokenizer_length
        )
        return result


def _single_model_token_sequence(input_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("model_input_ids must be a torch.Tensor")
    if input_ids.dtype != torch.long:
        raise TypeError("model_input_ids must have dtype torch.long")
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "representation supervision currently requires batch size one"
            )
        input_ids = input_ids[0]
    if input_ids.ndim != 1 or input_ids.shape[0] == 0:
        raise ValueError("model_input_ids must have shape [S] or [1,S]")
    return input_ids
