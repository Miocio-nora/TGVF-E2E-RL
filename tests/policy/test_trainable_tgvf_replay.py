from __future__ import annotations

import torch
from torch import nn

from tgvf_rl.policy.trainable_tgvf_replay import (
    extract_live_qwen3_vision_features,
    trainable_parameter_zero_anchor,
)


class _ToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grouped = hidden_states.reshape(-1, 4, 3).mean(dim=1)
        return self.projection(grouped)


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 3, bias=False)
        self.merger = _ToyMerger()
        self.deepstack_merger_list = nn.ModuleList(_ToyMerger() for _ in range(3))

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        assert tuple(grid_thw.shape) == (1, 3)
        hidden = self.stem(pixel_values)
        outputs = [self.merger(hidden)]
        outputs.extend(
            merger(hidden * (index + 2))
            for index, merger in enumerate(self.deepstack_merger_list)
        )
        return outputs[0], tuple(outputs[1:])


class _ToyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _ToyVisual()


def test_live_vision_capture_keeps_pixel_stem_and_merger_autograd() -> None:
    model = _ToyQwen()
    pixels = torch.randn(8, 3, requires_grad=True)

    features = extract_live_qwen3_vision_features(
        model,
        pixel_values=pixels,
        image_grid_thw=(1, 2, 4),
    )
    loss = features.merged_main.square().sum() + sum(
        branch.square().sum() for branch in features.merged_deepstack
    )
    loss.backward()

    assert features.premerge_main.shape == (8, 3)
    assert features.merged_main.shape == (2, 5)
    assert pixels.grad is not None
    assert model.model.visual.stem.weight.grad is not None
    assert model.model.visual.merger.projection.weight.grad is not None
    assert all(
        merger.projection.weight.grad is not None
        for merger in model.model.visual.deepstack_merger_list
    )


def test_zero_anchor_materializes_exact_zero_gradient_for_every_parameter() -> None:
    module = nn.Sequential(
        nn.Linear(5, 7),
        nn.LayerNorm(7),
        nn.Linear(7, 3, bias=False),
    )

    anchor = trainable_parameter_zero_anchor(module)
    anchor.backward()

    assert anchor.item() == 0.0
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(
        torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.parameters()
    )
