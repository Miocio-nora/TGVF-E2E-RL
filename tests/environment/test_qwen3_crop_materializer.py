from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.environment.qwen3_crop_materializer import (
    Qwen3CropVisualMaterializer,
)
from tgvf_rl.observations.store import tensor_checksum


class _Tokenizer:
    is_fast = True

    def __init__(self) -> None:
        self.name_or_path = "/fixture"
        self.chat_template = "fixture-template"

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 256 for character in text]


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.chat_template = self.tokenizer.chat_template
        self.image_processor = SimpleNamespace(size={"shortest_edge": 16})
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "rendered"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "image_grid_thw": torch.tensor(((1, 2, 2),), dtype=torch.long),
        }


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, 8))


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(spatial_merge_size=2)
        self.patch = nn.Linear(3, 2, bias=False)
        self.merger = _Merger()
        self.deepstack_merger_list = nn.ModuleList(
            (_Merger(), _Merger(), _Merger())
        )

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        assert tuple(grid_thw.shape) == (1, 3)
        hidden = self.patch(pixel_values)
        branches = tuple(
            merger(hidden + index + 1)
            for index, merger in enumerate(self.deepstack_merger_list)
        )
        return self.merger(hidden), branches


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Vision()
        self.model = SimpleNamespace(visual=self.visual)
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            _name_or_path="/fixture",
            vision_config=SimpleNamespace(deepstack_visual_indexes=(8, 16, 24)),
        )


def _identity() -> ModelIdentity:
    template_sha = hashlib.sha256(b"fixture-template").hexdigest()
    return ModelIdentity("qwen3_vl", "fixture", "/fixture", 256, template_sha)


def _frozen_model() -> _Model:
    model = _Model()
    model.requires_grad_(False)
    model.eval()
    return model


def test_real_crop_materializer_runs_processor_and_all_qwen3_mergers() -> None:
    model = _frozen_model()
    processor = _Processor()
    materializer = Qwen3CropVisualMaterializer.from_model(
        model=model,
        processor=processor,
        model_identity=_identity(),
        image_max_pixels=262144,
    )
    crop = torch.arange(6 * 7 * 3, dtype=torch.uint8).reshape(6, 7, 3)

    source = materializer.materialize_source_visual(
        crop, parsed_call=object(), call_index=0
    )
    plain = materializer.materialize(crop, parsed_call=object(), call_index=1)

    assert source.image_grid_thw == (1, 2, 2)
    assert source.decoded_rgb_sha256 == tensor_checksum(crop)
    assert source.spatial_merge_size == 2
    assert source.premerge_main.shape == (4, 2)
    assert tuple(branch.shape for branch in source.premerge_deepstack) == (
        (4, 2),
        (4, 2),
        (4, 2),
    )
    assert source.merged_main.shape == (1, 4)
    assert tuple(branch.shape for branch in source.merged_deepstack) == (
        (1, 4),
        (1, 4),
        (1, 4),
    )
    assert plain.deepstack_branch_layers == (8, 16, 24)
    assert not source.merged_main.requires_grad
    assert processor.calls[0]["images_kwargs"] == {
        "size": {"shortest_edge": 16, "longest_edge": 262144}
    }


def test_crop_materializer_rejects_trainable_vision() -> None:
    model = _Model()
    model.eval()
    processor = _Processor()

    try:
        Qwen3CropVisualMaterializer.from_model(
            model=model,
            processor=processor,
            model_identity=_identity(),
            image_max_pixels=262144,
        )
    except RuntimeError as error:
        assert "frozen Qwen vision" in str(error)
    else:  # pragma: no cover - fail loudly if the invariant regresses.
        raise AssertionError("trainable crop vision was unexpectedly accepted")
