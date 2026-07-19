from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    ContextualHiddenStateConditionProvider,
    TargetConditioningRequest,
    TargetTokenEmbeddingConditionProvider,
)
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.tokens import TokenSpan


SHA = "0" * 64


def _model(*, tokenizer_length: int = 32, name: str = "qwen-test") -> ModelIdentity:
    return ModelIdentity(
        family="qwen-vl",
        model_name=name,
        revision_or_path="synthetic",
        tokenizer_length=tokenizer_length,
        chat_template_sha256=SHA,
    )


def test_contextual_provider_selects_exact_span_and_records_provenance() -> None:
    provider = ContextualHiddenStateConditionProvider(
        model_identity=_model(), hidden_layer=17
    )
    input_ids = torch.tensor([1, 7, 8, 9, 2])
    hidden = torch.arange(30, dtype=torch.float32).reshape(5, 6).requires_grad_()

    output = provider.build(
        TargetConditioningRequest(
            input_ids=input_ids,
            target_span=TokenSpan(1, 4),
            expected_target_token_ids=(7, 8, 9),
            trajectory_id="trajectory-a",
            call_index=1,
            model_identity=provider.model_identity,
            contextual_hidden_states=hidden,
        )
    )

    assert torch.equal(output.values, hidden[1:4])
    assert output.provenance.hidden_layer == 17
    assert output.provenance.target_token_ids == ((7, 8, 9),)
    assert output.provenance.trajectory_ids == ("trajectory-a",)
    output.values.sum().backward()
    assert hidden.grad is not None


def test_contextual_provider_handles_per_trajectory_batched_targets() -> None:
    provider = ContextualHiddenStateConditionProvider(
        model_identity=_model(), hidden_layer=-1
    )
    input_ids = torch.tensor([[1, 5, 6, 2], [1, 8, 9, 2]])
    hidden = torch.randn(2, 4, 7)
    output = provider(
        TargetConditioningRequest(
            input_ids=input_ids,
            target_span=TokenSpan(1, 3),
            expected_target_token_ids=((5, 6), (8, 9)),
            trajectory_id=("trajectory-a", "trajectory-b"),
            call_index=(0, 2),
            model_identity=provider.model_identity,
            contextual_hidden_states=hidden,
        )
    )

    assert output.values.shape == (2, 2, 7)
    assert output.provenance.batched
    assert output.provenance.call_indices == (0, 2)


def test_contextual_provider_rejects_span_token_and_model_identity_drift() -> None:
    provider = ContextualHiddenStateConditionProvider(
        model_identity=_model(), hidden_layer=3
    )
    with pytest.raises(ValueError, match="do not exactly match"):
        provider.build(
            TargetConditioningRequest(
                input_ids=torch.tensor([1, 5, 6, 2]),
                target_span=TokenSpan(1, 3),
                expected_target_token_ids=(5, 7),
                trajectory_id="trajectory-a",
                call_index=0,
                model_identity=provider.model_identity,
                contextual_hidden_states=torch.randn(4, 6),
            )
        )
    with pytest.raises(ValueError, match="model identity"):
        provider.build(
            TargetConditioningRequest(
                input_ids=torch.tensor([1, 5, 6, 2]),
                target_span=TokenSpan(1, 3),
                expected_target_token_ids=(5, 6),
                trajectory_id="trajectory-a",
                call_index=0,
                model_identity=_model(name="different"),
                contextual_hidden_states=torch.randn(4, 6),
            )
        )
    with pytest.raises(ValueError, match="outside"):
        provider.build(
            TargetConditioningRequest(
                input_ids=torch.tensor([1, 5, 6, 2]),
                target_span=TokenSpan(3, 5),
                expected_target_token_ids=(2, 3),
                trajectory_id="trajectory-a",
                call_index=0,
                model_identity=provider.model_identity,
                contextual_hidden_states=torch.randn(4, 6),
            )
        )


def test_token_embedding_provider_borrows_exact_existing_rows_without_ownership() -> (
    None
):
    model = _model(tokenizer_length=16)
    embedding = nn.Embedding(16, 5)
    provider = TargetTokenEmbeddingConditionProvider(
        model_identity=model,
        embedding=embedding,
        embedding_identity="qwen.model.embed_tokens",
    )
    ids = torch.tensor([0, 3, 7, 4])

    output = provider.build(
        TargetConditioningRequest(
            input_ids=ids,
            target_span=TokenSpan(1, 3),
            expected_target_token_ids=(3, 7),
            trajectory_id="trajectory-a",
            call_index=0,
            model_identity=model,
        )
    )

    assert torch.equal(output.values, embedding(ids[1:3]))
    assert output.provenance.embedding_identity == "qwen.model.embed_tokens"
    assert list(provider.parameters()) == []
    output.values.sum().backward()
    assert embedding.weight.grad is not None


def test_token_embedding_provider_detects_vocab_growth() -> None:
    embedding = nn.Embedding(8, 4)
    provider = TargetTokenEmbeddingConditionProvider(
        model_identity=_model(tokenizer_length=8),
        embedding=embedding,
        embedding_identity="base-embedding",
    )
    embedding.num_embeddings = 9
    embedding.weight = nn.Parameter(torch.randn(9, 4))

    with pytest.raises(ValueError, match="tokenizer growth"):
        provider.build(
            TargetConditioningRequest(
                input_ids=torch.tensor([1, 2]),
                target_span=TokenSpan(0, 1),
                expected_target_token_ids=(1,),
                trajectory_id="trajectory-a",
                call_index=0,
                model_identity=provider.model_identity,
            )
        )


def test_token_embedding_provider_rejects_ids_outside_original_tokenizer() -> None:
    embedding = nn.Embedding(8, 4)
    provider = TargetTokenEmbeddingConditionProvider(
        model_identity=_model(tokenizer_length=8),
        embedding=embedding,
        embedding_identity="base-embedding",
    )
    with pytest.raises(ValueError, match="vocabulary"):
        provider.build(
            TargetConditioningRequest(
                input_ids=torch.tensor([1, 8]),
                target_span=TokenSpan(0, 1),
                expected_target_token_ids=(1,),
                trajectory_id="trajectory-a",
                call_index=0,
                model_identity=provider.model_identity,
            )
        )
