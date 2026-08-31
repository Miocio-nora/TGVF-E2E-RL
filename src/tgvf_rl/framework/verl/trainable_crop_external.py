"""veRL external-lib entry for matched full-model plain Crop."""

from __future__ import annotations

from .deepeyes_actor_loss import (  # noqa: F401
    compute_deepeyes_official_micro_token_mean_loss,
)
from .trainable_crop_engine import register_trainable_crop_fsdp2_engine


TRAINABLE_CROP_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_crop_external"
TRAINABLE_CROP_ENGINE_CLASS = register_trainable_crop_fsdp2_engine()


__all__ = [
    "TRAINABLE_CROP_ENGINE_CLASS",
    "TRAINABLE_CROP_EXTERNAL_MODULE",
]
