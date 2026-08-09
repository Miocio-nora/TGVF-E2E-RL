"""Narrow Qwen3-VL text-only FlexAttention compatibility.

Transformers 4.57 implements Qwen3-VL text attention through the common
``ALL_ATTENTION_FUNCTIONS`` interface and builds packed-document BlockMasks,
but the text model class does not advertise the corresponding capability.
The vision encoder has a different varlen execution path and deliberately
remains on SDPA.

The upstream FlexAttention wrapper also compiles training kernels with
``dynamic=None``.  Exact replay evaluates several variable-length trajectories
before one backward pass.  Torch may therefore replace the initial specialized
graph before non-reentrant activation checkpointing recomputes it, changing the
order of captured padding/BlockMask tensors and raising ``CheckpointError``.
Training compilation is made dynamic from its first graph so checkpoint
forward and recomputation retain one input ordering.  Inference keeps the
upstream wrapper unchanged.
"""

from __future__ import annotations


QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA = "tgvf.qwen3-vl-text-flex-compat.v2"
_DYNAMIC_TRAINING_MARKER = "_tgvf_dynamic_training_compile"


def _install_dynamic_training_flex_wrapper() -> None:
    import torch
    from transformers.integrations import flex_attention as flex_runtime

    upstream_cls = flex_runtime.WrappedFlexAttention
    if getattr(upstream_cls, _DYNAMIC_TRAINING_MARKER, False):
        return

    class TGVFDynamicTrainingFlexAttention(upstream_cls):
        """Preserve upstream inference and compile training for dynamic lengths."""

        _instance = None
        _is_flex_compiled = False
        _compiled_flex_attention = None
        _tgvf_dynamic_training_compile = True

        @torch.compiler.disable(recursive=False)
        def __init__(self, training: bool) -> None:
            if not training:
                super().__init__(training)
                return
            if not self._is_flex_compiled or training != self.training:
                self.training = training
                self._compiled_flex_attention = torch.compile(
                    flex_runtime.flex_attention,
                    dynamic=True,
                )
                self._is_flex_compiled = True

    TGVFDynamicTrainingFlexAttention.__name__ = upstream_cls.__name__
    TGVFDynamicTrainingFlexAttention.__qualname__ = upstream_cls.__qualname__
    TGVFDynamicTrainingFlexAttention.__module__ = upstream_cls.__module__
    flex_runtime.WrappedFlexAttention = TGVFDynamicTrainingFlexAttention


def install_qwen3_vl_text_flex_attention_compat() -> str:
    """Declare text FlexAttention and stabilize variable-length checkpointing."""

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    if not Qwen3VLTextModel._can_set_attn_implementation():
        raise RuntimeError(
            "installed Qwen3-VL text model does not use the common attention interface"
        )
    Qwen3VLTextModel._supports_flex_attn = True
    _install_dynamic_training_flex_wrapper()
    return QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA


__all__ = [
    "QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA",
    "install_qwen3_vl_text_flex_attention_compat",
]
