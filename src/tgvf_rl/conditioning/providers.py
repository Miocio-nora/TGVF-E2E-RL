"""Target-conditioning providers that reuse the selected base model state."""

from __future__ import annotations

import weakref
import torch
from torch import nn

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.tokens import TokenSpan

from .base import (
    CONTEXTUAL_HIDDEN_STATE,
    TARGET_TOKEN_EMBEDDING,
    BoundTargetConditionProvider,
    TargetConditioningOutput,
    TargetConditioningProvenance,
    TargetConditioningRequest,
    _ValidatedTargetSelection,
    _validate_target_selection,
)


class ContextualHiddenStateConditionProvider(BoundTargetConditionProvider):
    """Slice the exact sampled target span from one explicitly selected layer."""

    provider_name = CONTEXTUAL_HIDDEN_STATE

    def __init__(self, *, model_identity: ModelIdentity, hidden_layer: int) -> None:
        super().__init__(model_identity=model_identity)
        if not isinstance(hidden_layer, int) or isinstance(hidden_layer, bool):
            raise TypeError("hidden_layer must be an integer selected by configuration")
        self.hidden_layer = int(hidden_layer)

    def build(
        self,
        request: TargetConditioningRequest,
        /,
    ) -> TargetConditioningOutput:
        if not isinstance(request, TargetConditioningRequest):
            raise TypeError("request must be a TargetConditioningRequest")
        self._check_runtime_model(request.model_identity)
        selection = _validate_target_selection(
            input_ids=request.input_ids,
            target_span=request.target_span,
            expected_target_token_ids=request.expected_target_token_ids,
            trajectory_id=request.trajectory_id,
            call_index=request.call_index,
            tokenizer_length=self.model_identity.tokenizer_length,
            canonical_input_ids_proof=request.canonical_input_ids_proof,
        )
        hidden_states = request.contextual_hidden_states
        if hidden_states is None:
            raise ValueError(
                "contextual_hidden_state requires contextual_hidden_states"
            )
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        expected_rank = 3 if selection.batched else 2
        if hidden_states.ndim != expected_rank:
            raise ValueError(
                f"hidden_states must have rank {expected_rank} to match input_ids"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")
        if hidden_states.shape[-1] <= 0:
            raise ValueError("hidden_states feature dimension must be positive")
        if selection.batched:
            if tuple(hidden_states.shape[:2]) != tuple(selection.input_ids.shape):
                raise ValueError(
                    "hidden_states batch/sequence shape must match input_ids"
                )
        elif hidden_states.shape[0] != selection.sequence_length:
            raise ValueError("hidden_states sequence length must match input_ids")

        values = hidden_states[
            ..., request.target_span.start : request.target_span.end, :
        ]
        return TargetConditioningOutput(
            values=values,
            provenance=_provenance(
                provider=self.provider_name,
                model=self.model_identity,
                target_span=request.target_span,
                selection=selection,
                hidden_layer=self.hidden_layer,
            ),
        )

    def forward(
        self, request: TargetConditioningRequest, /
    ) -> TargetConditioningOutput:
        return self.build(request)


class TargetTokenEmbeddingConditionProvider(BoundTargetConditionProvider):
    """Use exact rows from the selected model's existing input embedding."""

    provider_name = TARGET_TOKEN_EMBEDDING

    def __init__(
        self,
        *,
        model_identity: ModelIdentity,
        embedding: nn.Module,
        embedding_identity: str,
    ) -> None:
        super().__init__(model_identity=model_identity)
        if not isinstance(embedding, nn.Module):
            raise TypeError("embedding must be the selected model's nn.Module")
        if not embedding_identity or not embedding_identity.strip():
            raise ValueError("embedding_identity must be non-empty")
        self.embedding_identity = embedding_identity
        self._embedding_ref = weakref.ref(embedding)
        self._validate_embedding(embedding)

    @property
    def borrowed_embedding(self) -> nn.Module:
        embedding = self._embedding_ref()
        if embedding is None:
            raise RuntimeError("the borrowed base-model embedding no longer exists")
        return embedding

    def _validate_embedding(self, embedding: nn.Module) -> None:
        num_embeddings = getattr(embedding, "num_embeddings", None)
        weight = getattr(embedding, "weight", None)
        if not isinstance(num_embeddings, int) or not isinstance(weight, torch.Tensor):
            raise TypeError("embedding must expose num_embeddings and a tensor weight")
        if num_embeddings != self.model_identity.tokenizer_length:
            raise ValueError(
                "embedding vocabulary size differs from the bound tokenizer length; "
                "tokenizer growth is forbidden"
            )
        if weight.ndim != 2 or weight.shape[0] != num_embeddings:
            raise ValueError(
                "embedding weight shape is inconsistent with num_embeddings"
            )

    def build(
        self,
        request: TargetConditioningRequest,
        /,
    ) -> TargetConditioningOutput:
        if not isinstance(request, TargetConditioningRequest):
            raise TypeError("request must be a TargetConditioningRequest")
        self._check_runtime_model(request.model_identity)
        if request.contextual_hidden_states is not None:
            raise ValueError(
                "target_token_embedding request cannot carry contextual_hidden_states"
            )
        selection = _validate_target_selection(
            input_ids=request.input_ids,
            target_span=request.target_span,
            expected_target_token_ids=request.expected_target_token_ids,
            trajectory_id=request.trajectory_id,
            call_index=request.call_index,
            tokenizer_length=self.model_identity.tokenizer_length,
            canonical_input_ids_proof=request.canonical_input_ids_proof,
        )
        embedding = self.borrowed_embedding
        self._validate_embedding(embedding)
        weight = embedding.weight
        if request.input_ids.device != weight.device:
            raise ValueError(
                "input_ids and the borrowed model embedding must share a device"
            )

        selected_ids = selection.input_ids[
            :, request.target_span.start : request.target_span.end
        ]
        values = embedding(selected_ids)
        if not selection.batched:
            values = values.squeeze(0)
        return TargetConditioningOutput(
            values=values,
            provenance=_provenance(
                provider=self.provider_name,
                model=self.model_identity,
                target_span=request.target_span,
                selection=selection,
                embedding_identity=self.embedding_identity,
            ),
        )

    def forward(
        self, request: TargetConditioningRequest, /
    ) -> TargetConditioningOutput:
        return self.build(request)


def _provenance(
    *,
    provider: str,
    model: ModelIdentity,
    target_span: TokenSpan,
    selection: _ValidatedTargetSelection,
    hidden_layer: int | None = None,
    embedding_identity: str | None = None,
) -> TargetConditioningProvenance:
    return TargetConditioningProvenance(
        provider=provider,
        model=model,
        target_span=target_span,
        target_token_ids=selection.rows,
        trajectory_ids=selection.trajectories,
        call_indices=selection.call_indices,
        source_sequence_length=selection.sequence_length,
        source_batch_size=selection.batch_size,
        source_input_ids_sha256=selection.digest,
        batched=selection.batched,
        hidden_layer=hidden_layer,
        embedding_identity=embedding_identity,
    )


# Compact aliases are kept for configuration/CLI code; prose uses the full names.
ContextHiddenStateProvider = ContextualHiddenStateConditionProvider
TargetTokenEmbeddingProvider = TargetTokenEmbeddingConditionProvider


__all__ = [
    "ContextHiddenStateProvider",
    "ContextualHiddenStateConditionProvider",
    "TargetTokenEmbeddingConditionProvider",
    "TargetTokenEmbeddingProvider",
]
