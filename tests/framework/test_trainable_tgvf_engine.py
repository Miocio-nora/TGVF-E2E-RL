from __future__ import annotations

import torch

from tgvf_rl.framework.verl.trainable_tgvf_engine import (
    _adapter_state_for_runtime_dtype,
)


def test_runtime_dtype_cast_preserves_source_and_nonfloating_state() -> None:
    floating = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    integer = torch.tensor([1, 2], dtype=torch.int64)

    normalized = _adapter_state_for_runtime_dtype(
        {"weight": floating, "counter": integer}, dtype=torch.float32
    )

    assert normalized["weight"].dtype is torch.float32
    assert torch.equal(normalized["weight"], floating.float())
    assert floating.dtype is torch.bfloat16
    assert normalized["counter"] is integer


def test_runtime_dtype_cast_rejects_nonfloating_target() -> None:
    try:
        _adapter_state_for_runtime_dtype(
            {"weight": torch.ones(1, dtype=torch.bfloat16)}, dtype=torch.int64
        )
    except TypeError as error:
        assert "must be floating point" in str(error)
    else:  # pragma: no cover - explicit assertion without pytest dependency
        raise AssertionError("non-floating RP66 runtime dtype was accepted")
