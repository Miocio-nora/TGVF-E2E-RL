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
    RecordedReplayHiddenResult,
    ReplayConsumer,
    assert_model_vocabulary_compatible,
    injected_request_from_recorded,
    materialize_deepstack,
    materialize_inputs_embeds,
    resolve_replay_request,
    resolve_language_model,
    resolve_lm_head,
)
from .crop_coordinates import (
    QWEN3_CROP_CONVERSION_VERSION,
    QWEN3_CROP_COORDINATE_SPACE,
    CropCoordinateMapping,
    map_qwen3_crop_bbox_to_source,
)
from .deepstack_control import native_deepstack_enabled_from_model


class Qwen3VLAdapter(QwenVLMFamilyAdapter):
    crop_coordinate_space = QWEN3_CROP_COORDINATE_SPACE
    crop_coordinate_conversion_version = QWEN3_CROP_CONVERSION_VERSION
    capabilities = FamilyCapabilities(
        family="qwen3_vl",
        support_level=SupportLevel.EXECUTABLE,
        # Assistant framing is checkpoint/template-bound: the Instruct primary
        # has no template-owned think opener, while historical Thinking does.
        native_thinking_prefill=False,
        deepstack_branch_count=3,
        recorded_d_forward=True,
        native_tool_template=True,
        native_injected_kv_cache=True,
    )

    def map_crop_bbox_to_source(
        self,
        bbox_2d: tuple[int, int, int, int],
        *,
        source_width: int,
        source_height: int,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> CropCoordinateMapping:
        return map_qwen3_crop_bbox_to_source(
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
        if any(
            len(block.deepstack) != self.capabilities.deepstack_branch_count
            for block in request.visual_blocks
        ):
            raise ValueError(
                f"Qwen3 replay requires {self.capabilities.deepstack_branch_count} exact DeepStack branches for every visual block"
            )
        inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
        deepstack = (
            materialize_deepstack(
                request,
                visual_mask,
                target_dtype=inputs_embeds.dtype,
            )
            if native_deepstack_enabled_from_model(model)
            else None
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

    def forward_injected_hidden(
        self,
        model: Any,
        request: InjectedForwardRequest,
    ) -> RecordedReplayHiddenResult:
        """Run one injected request through the decoder without its LM head."""

        rows = self.forward_injected_hidden_batch(model, (request,))
        return rows[0]

    def forward_injected_hidden_batch(
        self,
        model: Any,
        requests: tuple[InjectedForwardRequest, ...],
    ) -> tuple[RecordedReplayHiddenResult, ...]:
        """Batch only decoder forwards for independently constructed rows.

        Vision/TGVF/Crop construction intentionally remains outside this
        method.  Every request is validated and injected independently before
        right-padding, so no row can borrow another row's visual positions or
        DeepStack values.  The single decoder call is therefore an execution
        optimization over the same per-row inputs, not a replay reconstruction.
        """

        if not requests:
            raise ValueError("Qwen3 decoder replay batch cannot be empty")
        if any(not isinstance(request, InjectedForwardRequest) for request in requests):
            raise TypeError("Qwen3 decoder replay batch requires injected requests")
        if any(request.use_cache for request in requests):
            raise ValueError("batched exact replay is no_cache only")
        if any(request.attention_mask.ndim != 2 for request in requests):
            raise ValueError(
                "batched exact replay currently requires 2D attention masks"
            )
        if any(
            len(block.deepstack) != self.capabilities.deepstack_branch_count
            for request in requests
            for block in request.visual_blocks
        ):
            raise ValueError(
                f"Qwen3 replay requires {self.capabilities.deepstack_branch_count} exact DeepStack branches for every visual block"
            )

        injected: list[
            tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]
        ] = []
        deepstack_enabled = native_deepstack_enabled_from_model(model)
        for request in requests:
            inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)
            if inputs_embeds.shape[0] != 1:
                raise ValueError("each exact replay row must have batch size one")
            deepstack = (
                materialize_deepstack(
                    request,
                    visual_mask,
                    target_dtype=inputs_embeds.dtype,
                )
                if deepstack_enabled
                else None
            )
            injected.append((inputs_embeds, visual_mask, deepstack))

        devices = {row[0].device for row in injected}
        dtypes = {row[0].dtype for row in injected}
        hidden_sizes = {int(row[0].shape[-1]) for row in injected}
        if len(devices) != 1 or len(dtypes) != 1 or len(hidden_sizes) != 1:
            raise ValueError(
                "batched decoder replay rows differ in placement or hidden size"
            )
        device = injected[0][0].device
        hidden_size = int(injected[0][0].shape[-1])
        lengths = tuple(int(row[0].shape[1]) for row in injected)
        maximum = max(lengths)

        def pad_sequence(tensor: torch.Tensor, *, value: float = 0.0) -> torch.Tensor:
            if tensor.shape[1] == maximum:
                return tensor
            padding = torch.full(
                (1, maximum - tensor.shape[1], *tensor.shape[2:]),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat((tensor, padding), dim=1)

        inputs_embeds = torch.cat(
            tuple(pad_sequence(row[0]) for row in injected), dim=0
        )
        visual_mask = torch.cat(tuple(pad_sequence(row[1]) for row in injected), dim=0)
        attention_mask = torch.cat(
            tuple(
                pad_sequence(request.attention_mask.to(device), value=0)
                for request in requests
            ),
            dim=0,
        )
        position_rank = requests[0].position_ids.ndim
        if any(request.position_ids.ndim != position_rank for request in requests):
            raise ValueError("batched replay position-id ranks differ")
        position_rows: list[torch.Tensor] = []
        for request in requests:
            positions = request.position_ids.to(device)
            padding_length = maximum - positions.shape[-1]
            if padding_length:
                positions = torch.cat(
                    (
                        positions,
                        torch.zeros(
                            (*positions.shape[:-1], padding_length),
                            dtype=positions.dtype,
                            device=positions.device,
                        ),
                    ),
                    dim=-1,
                )
            position_rows.append(positions)
        position_batch_dimension = 0 if position_rank == 2 else 1
        position_ids = torch.cat(tuple(position_rows), dim=position_batch_dimension)

        deepstack_batch: list[torch.Tensor] | None = None
        if deepstack_enabled:
            branch_counts = {
                len(row[2]) if row[2] is not None else -1 for row in injected
            }
            if branch_counts != {self.capabilities.deepstack_branch_count}:
                raise ValueError("batched replay DeepStack branch counts differ")
            deepstack_batch = [
                torch.cat(
                    tuple(row[2][branch] for row in injected if row[2] is not None),
                    dim=0,
                )
                for branch in range(self.capabilities.deepstack_branch_count)
            ]

        language_model = resolve_language_model(model)
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack_batch,
            use_cache=False,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        if hidden.shape != (len(requests), maximum, hidden_size):
            raise ValueError("Qwen3 batched decoder returned an unexpected shape")
        return tuple(
            RecordedReplayHiddenResult(
                hidden_states=hidden[index : index + 1, :length],
                visual_position_mask=visual_mask[index : index + 1, :length],
            )
            for index, length in enumerate(lengths)
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
        deepstack = (
            materialize_deepstack(
                request,
                visual_mask,
                target_dtype=inputs_embeds.dtype,
            )
            if native_deepstack_enabled_from_model(model)
            else None
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
