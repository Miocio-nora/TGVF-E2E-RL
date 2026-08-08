"""Narrow Qwen3-VL text-only FlexAttention capability declaration.

Transformers 4.57 implements Qwen3-VL text attention through the common
``ALL_ATTENTION_FUNCTIONS`` interface and builds packed-document BlockMasks,
but the text model class does not advertise the corresponding capability.
The vision encoder has a different varlen execution path and deliberately
remains on SDPA.
"""

from __future__ import annotations


QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA = "tgvf.qwen3-vl-text-flex-compat.v1"


def install_qwen3_vl_text_flex_attention_compat() -> str:
    """Declare FlexAttention only for the generic Qwen3-VL text model."""

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    if not Qwen3VLTextModel._can_set_attn_implementation():
        raise RuntimeError(
            "installed Qwen3-VL text model does not use the common attention interface"
        )
    Qwen3VLTextModel._supports_flex_attn = True
    return QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA


__all__ = [
    "QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA",
    "install_qwen3_vl_text_flex_attention_compat",
]
