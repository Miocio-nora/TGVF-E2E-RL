"""veRL external-lib entry for the matched full-model RP66 pilot."""

from __future__ import annotations

# Importing this module installs the exact actor reduction used by the
# successful native Crop control, plus its padding/FlexAttention compatibility.
from .deepeyes_actor_loss import (  # noqa: F401
    compute_deepeyes_official_micro_token_mean_loss,
)
from .trainable_tgvf_engine import register_trainable_tgvf_fsdp2_engine


TRAINABLE_TGVF_EXTERNAL_MODULE = (
    "tgvf_rl.framework.verl.trainable_tgvf_external"
)
TRAINABLE_TGVF_ENGINE_CLASS = register_trainable_tgvf_fsdp2_engine()


__all__ = [
    "TRAINABLE_TGVF_ENGINE_CLASS",
    "TRAINABLE_TGVF_EXTERNAL_MODULE",
]
