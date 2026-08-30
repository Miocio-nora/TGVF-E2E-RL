from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from tgvf_rl.framework.verl.padding_compat import (
    install_verl_sdpa_padding_compat,
    torch_index_first_axis,
    torch_unpad_input,
)


def test_torch_unpad_input_matches_exact_row_major_mask_selection() -> None:
    values = torch.arange(10).reshape(2, 5, 1)
    mask = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
    )

    unpadded, indices, cumulative, maximum, lengths = torch_unpad_input(values, mask)

    assert indices.tolist() == [1, 2, 3, 5, 6, 7, 8, 9]
    assert cumulative.tolist() == [0, 3, 8]
    assert maximum == 5
    assert lengths.tolist() == [3, 5]
    assert unpadded.squeeze(-1).tolist() == indices.tolist()
    torch.testing.assert_close(
        torch_index_first_axis(values.reshape(10, 1), indices),
        unpadded,
    )


def test_pinned_verl_padding_round_trip_uses_torch_compat_without_flash_attn() -> None:
    pytest.importorskip(
        "verl",
        reason="padding integration requires the optional pinned veRL",
    )
    from verl.trainer.ppo import ray_trainer
    from verl.workers.utils import padding

    original_index = padding.index_first_axis
    original_unpad = padding.unpad_input
    try:
        install_verl_sdpa_padding_compat()
        data = TensorDict(
            {
                "input_ids": torch.arange(10).reshape(2, 5),
                "attention_mask": torch.tensor(
                    [
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                    ]
                ),
                "response_mask": torch.tensor([[1, 0], [1, 1]]),
                "position_ids": torch.arange(5).repeat(2, 1),
                "prompts": torch.zeros((2, 3), dtype=torch.long),
                "responses": torch.zeros((2, 2), dtype=torch.long),
            },
            batch_size=[2],
        )

        no_padding = ray_trainer.left_right_2_no_padding(data)
        assert no_padding["input_ids"].values().tolist() == [1, 2, 3, 5, 6, 7, 8, 9]
        assert no_padding["input_ids"].offsets().tolist() == [0, 3, 8]
        restored = ray_trainer.no_padding_2_padding(
            torch.arange(8, dtype=torch.float32),
            no_padding,
        )
        torch.testing.assert_close(
            restored,
            torch.tensor([[1.0, 0.0], [5.0, 6.0]]),
        )
    finally:
        padding.index_first_axis = original_index
        padding.unpad_input = original_unpad
