from __future__ import annotations

import torch
from tensordict import TensorDict

from tgvf_rl.framework.verl.torch_bert_padding import (
    PRL13_TORCH_BERT_PADDING_SCHEMA,
    install_prl13_torch_bert_padding,
    require_prl13_torch_bert_padding,
)


def test_torch_bert_padding_matches_mask_and_round_trips() -> None:
    from verl.utils.attention_utils import pad_input, unpad_input

    assert install_prl13_torch_bert_padding() == PRL13_TORCH_BERT_PADDING_SCHEMA
    assert require_prl13_torch_bert_padding() == PRL13_TORCH_BERT_PADDING_SCHEMA

    hidden = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4)
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.int64,
    )
    unpadded, indices, offsets, maximum, used = unpad_input(hidden, mask)
    assert torch.equal(unpadded, hidden[mask.bool()])
    assert torch.equal(
        indices,
        torch.nonzero(mask.flatten(), as_tuple=False).flatten(),
    )
    assert offsets.tolist() == [0, 3, 7, 12]
    assert maximum == 5
    assert used.tolist() == [3, 4, 5]

    restored = pad_input(unpadded, indices, batch=3, seqlen=5)
    expected = torch.zeros_like(hidden)
    expected[mask.bool()] = hidden[mask.bool()]
    assert torch.equal(restored, expected)


def test_torch_bert_padding_preserves_index_and_pad_gradients() -> None:
    from verl.utils.attention_utils import index_first_axis, pad_input, unpad_input

    install_prl13_torch_bert_padding()
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int64)

    actual_source = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    actual_source.requires_grad_(True)
    actual, indices, *_ = unpad_input(actual_source, mask)
    actual_loss = pad_input(actual.square(), indices, 2, 3).sum()
    actual_loss.backward()

    expected_source = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    expected_source.requires_grad_(True)
    expected_loss = expected_source[mask.bool()].square().sum()
    expected_loss.backward()
    assert torch.equal(actual_source.grad, expected_source.grad)

    flat = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    flat.requires_grad_(True)
    selected = index_first_axis(flat, torch.tensor([0, 2, 5]))
    selected.sum().backward()
    expected_grad = torch.zeros_like(flat)
    expected_grad[[0, 2, 5]] = 1
    assert torch.equal(flat.grad, expected_grad)


def test_exact_old_logprob_no_padding_adapter_needs_no_flash_attn() -> None:
    from verl.workers.utils.padding import left_right_2_no_padding

    install_prl13_torch_bert_padding()
    input_ids = torch.tensor(
        [[0, 11, 12, 21, 22], [31, 32, 33, 41, 0]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.long
    )
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            "position_ids": torch.arange(5).repeat(2, 1),
        },
        batch_size=2,
    )
    converted = left_right_2_no_padding(batch)
    assert converted["input_ids"].is_nested
    assert torch.equal(
        converted["input_ids"].values(), input_ids[attention_mask.bool()]
    )
    assert converted["input_ids"].offsets().tolist() == [0, 4, 8]


def test_task_runner_installs_backend_before_upstream_run(monkeypatch) -> None:
    from verl.utils import attention_utils

    from tgvf_rl.framework.verl.prl13_main import run_prl13_task_runner

    original_getter = attention_utils._get_attention_functions
    original_schema = getattr(attention_utils, "_prl13_padding_backend", None)
    observed: dict[str, object] = {}

    def unpatched_getter():
        raise AssertionError("upstream run observed the unpatched backend")

    def upstream_run(runner: object, config: object) -> str:
        observed["runner"] = runner
        observed["config"] = config
        observed["schema"] = require_prl13_torch_bert_padding()
        return "upstream-result"

    monkeypatch.setattr(attention_utils, "_get_attention_functions", unpatched_getter)
    monkeypatch.delattr(attention_utils, "_prl13_padding_backend", raising=False)
    runner, config = object(), object()
    try:
        result = run_prl13_task_runner(upstream_run, runner, config)
    finally:
        attention_utils._get_attention_functions = original_getter
        if original_schema is None:
            delattr(attention_utils, "_prl13_padding_backend")
        else:
            attention_utils._prl13_padding_backend = original_schema

    assert result == "upstream-result"
    assert observed == {
        "runner": runner,
        "config": config,
        "schema": PRL13_TORCH_BERT_PADDING_SCHEMA,
    }
