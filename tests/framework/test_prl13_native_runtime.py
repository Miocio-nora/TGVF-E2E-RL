from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch

from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_THINKLITE_AGENT,
    NATIVE_DEEPEYES_VISUAL_AGENT,
    NativeFullModelTrainability,
    assert_native_multimodal_inputs,
    assert_native_pixel_row,
    assert_observation_mask,
    assert_trainable_parameter_groups,
    native_deepeyes_agent_name,
)
from tgvf_rl.framework.verl.native_deepeyes_smoke import (
    run_native_deepeyes_cpu_fake_smoke,
)


def _visual_row() -> dict[str, object]:
    return {
        "data_source": "vstar",
        "agent_name": NATIVE_DEEPEYES_VISUAL_AGENT,
        "raw_prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/original.png"},
                    {"type": "text", "text": "question"},
                ],
            }
        ],
    }


def test_native_source_routing_is_visual_crop_or_thinklite_no_tool() -> None:
    assert native_deepeyes_agent_name("vstar") == NATIVE_DEEPEYES_VISUAL_AGENT
    assert native_deepeyes_agent_name("arxivqa") == NATIVE_DEEPEYES_VISUAL_AGENT
    assert native_deepeyes_agent_name("teacher") == NATIVE_DEEPEYES_VISUAL_AGENT
    assert native_deepeyes_agent_name("thinklite") == NATIVE_DEEPEYES_THINKLITE_AGENT
    with pytest.raises(ValueError, match="unsupported"):
        native_deepeyes_agent_name("unknown")


def test_teacher_native_pixel_row_is_visual_without_gold_regions() -> None:
    row = {
        **_visual_row(),
        "data_source": "teacher",
        "task_kind": "mcq",
        "tools_kwargs": {"image_zoom_in_tool": {"create_kwargs": {"gt_regions": ()}}},
    }
    assert_native_pixel_row(row)
    with pytest.raises(ValueError, match="mcq.*open"):
        assert_native_pixel_row({**row, "task_kind": "math"})
    with pytest.raises(ValueError, match="cannot carry gt_regions"):
        assert_native_pixel_row(
            {
                **row,
                "tools_kwargs": {
                    "image_zoom_in_tool": {
                        "create_kwargs": {"gt_regions": ((1, 2, 10, 20),)}
                    }
                },
            }
        )


def test_native_pixel_row_rejects_embeds_replay_and_wrong_routing() -> None:
    row = _visual_row()
    assert_native_pixel_row(row)
    pil_row = _visual_row()
    pil_row["raw_prompt"][0]["content"][0]["image"] = Image.new(
        "RGB", (32, 24), color="red"
    )
    assert_native_pixel_row(pil_row)
    with pytest.raises(ValueError, match="image_embeds"):
        assert_native_pixel_row({**row, "image_embeds": [1.0]})
    with pytest.raises(ValueError, match="trajectory_replay_bundle"):
        assert_native_pixel_row({**row, "trajectory_replay_bundle": {}})
    with pytest.raises(ValueError, match="routing"):
        assert_native_pixel_row({**row, "agent_name": "single_turn_agent"})


def test_full_model_gate_rejects_lora_frozen_vision_and_legacy_adapter() -> None:
    valid = NativeFullModelTrainability(
        lora_rank=0,
        vision_trainable=True,
        vision_projection_trainable=True,
        language_trainable=True,
        native_pixels=True,
        precomputed_image_embeds=False,
    )
    valid.validate()
    invalid_values = (
        {"lora_rank": 8},
        {"vision_trainable": False},
        {"vision_projection_trainable": False},
        {"language_trainable": False},
        {"native_pixels": False},
        {"precomputed_image_embeds": True},
        {"legacy_adapter_path": "/checkpoint/rp66"},
        {"model_implementation": "Qwen3TGVFForConditionalGeneration"},
    )
    base = {
        "lora_rank": 0,
        "vision_trainable": True,
        "vision_projection_trainable": True,
        "language_trainable": True,
        "native_pixels": True,
        "precomputed_image_embeds": False,
        "legacy_adapter_path": None,
        "model_implementation": "native_qwen3_vl",
    }
    for mutation in invalid_values:
        with pytest.raises(ValueError):
            NativeFullModelTrainability(**{**base, **mutation}).validate()


def test_observation_mask_and_native_image_count_are_fail_closed() -> None:
    response_mask = [1, 1, 0, 0, 0, 1]
    assert_observation_mask(response_mask, [[2, 5]])
    with pytest.raises(ValueError, match="policy loss"):
        assert_observation_mask([1, 1, 0, 1, 0, 1], [[2, 5]])

    inputs = {
        "pixel_values": torch.ones((2, 3)),
        "image_grid_thw": torch.ones((2, 3), dtype=torch.long),
    }
    assert_native_multimodal_inputs(
        inputs, original_image_count=1, successful_crop_count=1
    )
    with pytest.raises(ValueError, match="image count"):
        assert_native_multimodal_inputs(
            inputs, original_image_count=1, successful_crop_count=0
        )


def test_parameter_gate_requires_trainable_vision_and_language() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_tower = torch.nn.Linear(2, 2)
            self.vision_merger = torch.nn.Linear(2, 2)
            self.language_model = torch.nn.Linear(2, 2)

    counts = assert_trainable_parameter_groups(Tiny().named_parameters())
    assert counts["vision"] > 0
    assert counts["projection"] > 0
    assert counts["language"] > 0


def test_single_process_cpu_fake_server_contract(tmp_path: Path) -> None:
    result = run_native_deepeyes_cpu_fake_smoke(
        checkpoint_root=tmp_path / "checkpoints"
    )
    assert result.sample_count == 4
    assert result.visual_sample_count == result.thinklite_sample_count == 2
    assert result.successful_crop_count == 1
    assert result.native_image_count == 2
    assert result.observation_mask_zero
    assert result.vision_gradient_norm > 0
    assert result.language_gradient_norm > 0
    assert result.vision_weight_changed
    assert result.language_weight_changed
    assert result.cpu_state_dict_copy_changed
    assert result.cpu_state_dict_copy_step == 1
    assert result.resume_advanced_one_step
    assert result.checkpoint_before_eval
    assert result.manifest["native_pixels"] is True
    assert result.manifest["precomputed_image_embeds"] is False
    assert result.manifest["legacy_adapter_loaded"] is False
    assert result.manifest["cpu_contract_only"] is True
    assert result.manifest["real_fsdp_vllm_sync_proven"] is False
    assert result.manifest["real_model_trainability_proven"] is False
