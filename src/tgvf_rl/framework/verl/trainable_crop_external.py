"""veRL external-lib entry for matched full-model plain Crop."""

from __future__ import annotations

from .deepeyes_actor_loss import (  # noqa: F401
    compute_deepeyes_official_micro_token_mean_loss,
)
from .dynamic_token_actor_loss import (  # noqa: F401
    compute_dynamic_global_token_mean_loss,
)
from .method_bypass_actor_loss import register_method_matrix_bypass_loss
from .trainable_crop_engine import register_trainable_crop_fsdp2_engine


TRAINABLE_CROP_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_crop_external"
METHOD_MATRIX_BYPASS_LOSS = register_method_matrix_bypass_loss()
TRAINABLE_CROP_ENGINE_CLASS = register_trainable_crop_fsdp2_engine()


__all__ = [
    "TRAINABLE_CROP_ENGINE_CLASS",
    "TRAINABLE_CROP_EXTERNAL_MODULE",
    "METHOD_MATRIX_BYPASS_LOSS",
    "compute_dynamic_global_token_mean_loss",
]
