"""Configuration-driven construction of target-conditioning providers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from tgvf_rl.contracts.identity import ModelIdentity

from .base import (
    BoundTargetConditionProvider,
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from .providers import (
    ContextualHiddenStateConditionProvider,
    TargetTokenEmbeddingConditionProvider,
)


@dataclass(frozen=True, slots=True)
class TargetConditioningDependencies:
    """Model-owned dependencies injected without transferring parameter ownership."""

    base_embedding: nn.Module | None = None

    def __post_init__(self) -> None:
        if self.base_embedding is not None and not isinstance(
            self.base_embedding, nn.Module
        ):
            raise TypeError("base_embedding must be an nn.Module")


def create_target_condition_provider(
    *,
    config: TargetConditioningConfig,
    model_identity: ModelIdentity,
    dependencies: TargetConditioningDependencies | None = None,
) -> BoundTargetConditionProvider:
    """Create exactly the provider selected by the experiment configuration."""

    if not isinstance(config, TargetConditioningConfig):
        raise TypeError("config must be a TargetConditioningConfig")
    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")
    resolved_dependencies = dependencies or TargetConditioningDependencies()
    if not isinstance(resolved_dependencies, TargetConditioningDependencies):
        raise TypeError("dependencies must be TargetConditioningDependencies")

    if config.provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        hidden_layer = config.hidden_layer
        if hidden_layer is None:
            raise ValueError("contextual_hidden_state requires hidden_layer")
        return ContextualHiddenStateConditionProvider(
            model_identity=model_identity,
            hidden_layer=hidden_layer,
        )

    embedding = resolved_dependencies.base_embedding
    if embedding is None:
        raise ValueError(
            "target_token_embedding requires the selected model's base embedding"
        )
    embedding_identity = config.embedding_identity
    if embedding_identity is None:
        raise ValueError("target_token_embedding requires embedding_identity")
    return TargetTokenEmbeddingConditionProvider(
        model_identity=model_identity,
        embedding=embedding,
        embedding_identity=embedding_identity,
    )


__all__ = [
    "TargetConditioningDependencies",
    "create_target_condition_provider",
]
