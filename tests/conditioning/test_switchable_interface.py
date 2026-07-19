from __future__ import annotations

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    ContextualHiddenStateConditionProvider,
    TargetConditionProvider,
    TargetConditioningConfig,
    TargetConditioningDependencies,
    TargetConditioningProviderKind,
    TargetConditioningRequest,
    TargetTokenEmbeddingConditionProvider,
    create_target_condition_provider,
)
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.representation import TGVFAdapterInput


def _model() -> ModelIdentity:
    return ModelIdentity(
        family="qwen-vl",
        model_name="switchable-fixture",
        revision_or_path="synthetic",
        tokenizer_length=16,
        chat_template_sha256="0" * 64,
    )


def test_provider_config_is_explicit_and_mutually_exclusive() -> None:
    with pytest.raises(TypeError, match="TargetConditioningProviderKind"):
        TargetConditioningConfig(provider="contextual_hidden_state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_layer"):
        TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        )
    with pytest.raises(ValueError, match="embedding_identity"):
        TargetConditioningConfig(
            provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
        )
    with pytest.raises(ValueError, match="cannot configure hidden_layer"):
        TargetConditioningConfig(
            provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
            hidden_layer=-1,
            embedding_identity="model.embed_tokens",
        )


def test_factory_fails_closed_without_selected_embedding_dependency() -> None:
    with pytest.raises(ValueError, match="base embedding"):
        create_target_condition_provider(
            config=TargetConditioningConfig(
                provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
                embedding_identity="model.embed_tokens",
            ),
            model_identity=_model(),
        )


@pytest.mark.parametrize(
    ("kind", "expected_type", "expected_name"),
    [
        (
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            ContextualHiddenStateConditionProvider,
            "contextual_hidden_state",
        ),
        (
            TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
            TargetTokenEmbeddingConditionProvider,
            "target_token_embedding",
        ),
    ],
)
def test_factory_switches_providers_behind_one_request_and_adapter_handoff(
    kind: TargetConditioningProviderKind,
    expected_type: type[object],
    expected_name: str,
) -> None:
    model = _model()
    embedding = nn.Embedding(model.tokenizer_length, 5)
    if kind is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        config = TargetConditioningConfig(provider=kind, hidden_layer=-1)
        hidden_states: torch.Tensor | None = torch.randn(4, 5)
    else:
        config = TargetConditioningConfig(
            provider=kind,
            embedding_identity="model.embed_tokens",
        )
        hidden_states = None

    provider = create_target_condition_provider(
        config=config,
        model_identity=model,
        dependencies=TargetConditioningDependencies(base_embedding=embedding),
    )
    assert isinstance(provider, expected_type)
    assert isinstance(provider, TargetConditionProvider)

    output = provider.build(
        TargetConditioningRequest(
            input_ids=torch.tensor([1, 5, 6, 2]),
            target_span=TokenSpan(1, 3),
            expected_target_token_ids=(5, 6),
            trajectory_id="trajectory-a",
            call_index=1,
            model_identity=model,
            contextual_hidden_states=hidden_states,
        )
    )
    adapter_input = TGVFAdapterInput.from_conditioning(
        output,
        pre_merge_visual_tokens=torch.randn(4, 3),
        deepstack_pre_merge_visual_tokens=(torch.randn(4, 3),),
    )

    assert output.provenance.provider == expected_name
    assert output.values.shape == (2, 5)
    assert adapter_input.target_hidden_states is output.values
    assert adapter_input.condition_provenance is output.provenance
