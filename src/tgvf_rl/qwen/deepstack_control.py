"""One config-owned switch for native Qwen DeepStack execution.

The recorded replay format always retains all three branches.  This switch
controls whether Qwen's language stack consumes those branches; preserving the
transport keeps true/false runs comparable and avoids a second replay schema.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch


TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD = "tgvf_native_deepstack_enabled"


def native_deepstack_enabled_from_config(config: Any) -> bool:
    """Read the project override, retaining enabled semantics for old models."""

    value = getattr(config, TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD, True)
    if type(value) is not bool:
        raise TypeError(f"{TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD} must be bool")
    return value


def native_deepstack_enabled_from_model(model: Any) -> bool:
    """Resolve the override through common HF/FSDP wrapper boundaries."""

    pending: deque[Any] = deque((model,))
    visited: set[int] = set()
    while pending:
        candidate = pending.popleft()
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(
            config, TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD
        ):
            return native_deepstack_enabled_from_config(config)
        for attribute in ("module", "model", "_fsdp_wrapped_module"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)
    return True


def apply_native_deepstack_tensor_control(
    deepstack_visual_embeds: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Zero vLLM's native DeepStack input while retaining its main embedding."""

    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    if not isinstance(deepstack_visual_embeds, torch.Tensor):
        raise TypeError("deepstack_visual_embeds must be a tensor")
    return (
        deepstack_visual_embeds
        if enabled
        else torch.zeros_like(deepstack_visual_embeds)
    )


__all__ = [
    "TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD",
    "apply_native_deepstack_tensor_control",
    "native_deepstack_enabled_from_config",
    "native_deepstack_enabled_from_model",
]
