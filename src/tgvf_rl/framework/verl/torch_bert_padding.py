"""Deterministic Torch fallback for veRL's FlashAttention padding helpers.

PRL13 uses Transformers SDPA for Qwen3-VL on B200, but pinned veRL's
remove-padding adapter still imports ``flash_attn.bert_padding`` solely for
four indexing/padding utilities.  The accepted environment intentionally has
no compatible ``flash_attn`` wheel.  veRL already vendors an exact PyTorch
copy of those utilities for its NPU compatibility path; those functions are
hardware-agnostic and operate normally on CUDA tensors.

Importing this module installs that implementation behind veRL's public
``attention_utils`` wrappers.  It does not change the model attention kernel,
token masks, sequence offsets, loss, or gradients.
"""

from __future__ import annotations

from collections.abc import Callable


PRL13_TORCH_BERT_PADDING_MODULE = (
    "tgvf_rl.framework.verl.torch_bert_padding"
)
PRL13_TORCH_BERT_PADDING_SCHEMA = "tgvf.prl13-torch-bert-padding.v1"


def _torch_bert_padding_functions() -> tuple[
    Callable[..., object],
    Callable[..., object],
    Callable[..., object],
    Callable[..., object],
]:
    # These are copied verbatim from flash_attn.bert_padding in pinned veRL.
    # The module name reflects their original NPU use, but the implementation
    # consists only of torch/einops gather, scatter, cumsum, and reshape ops.
    from verl.utils.npu_flash_attn_utils import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )

    return index_first_axis, pad_input, rearrange, unpad_input


def install_prl13_torch_bert_padding() -> str:
    """Install the project-owned backend into pinned veRL, idempotently."""

    from verl.utils import attention_utils

    attention_utils._get_attention_functions = _torch_bert_padding_functions
    attention_utils._prl13_padding_backend = PRL13_TORCH_BERT_PADDING_SCHEMA
    return PRL13_TORCH_BERT_PADDING_SCHEMA


def require_prl13_torch_bert_padding() -> str:
    """Fail closed unless this process uses the declared PRL13 backend."""

    from verl.utils import attention_utils

    if (
        attention_utils._get_attention_functions
        is not _torch_bert_padding_functions
        or getattr(attention_utils, "_prl13_padding_backend", None)
        != PRL13_TORCH_BERT_PADDING_SCHEMA
    ):
        raise RuntimeError("PRL13 Torch BERT-padding backend is not installed")
    actual = attention_utils._get_attention_functions()
    expected = _torch_bert_padding_functions()
    if any(left is not right for left, right in zip(actual, expected, strict=True)):
        raise RuntimeError("PRL13 Torch BERT-padding functions differ")
    return PRL13_TORCH_BERT_PADDING_SCHEMA


# ``ModelConfig.external_lib`` imports the project actor-loss module in each
# model worker, which imports this module before the no-padding path is used.
# The separate CPU-only TaskRunner installs the same backend explicitly at the
# first line of its ``run`` method.
install_prl13_torch_bert_padding()


__all__ = [
    "PRL13_TORCH_BERT_PADDING_MODULE",
    "PRL13_TORCH_BERT_PADDING_SCHEMA",
    "install_prl13_torch_bert_padding",
    "require_prl13_torch_bert_padding",
]
