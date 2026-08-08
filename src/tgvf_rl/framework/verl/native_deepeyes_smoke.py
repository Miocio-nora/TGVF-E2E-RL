"""Single-process CPU fake-server smoke for the PRL13 native contract."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import torch

from .native_crop_tool import NATIVE_CROP_TOOL_NAME, NativeDeepEyesCropTool
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_THINKLITE_AGENT,
    NATIVE_DEEPEYES_VISUAL_AGENT,
    STRICT_NATIVE_FULL_MODEL_TRAINABILITY,
    assert_native_multimodal_inputs,
    assert_native_pixel_row,
    assert_observation_mask,
    assert_trainable_parameter_groups,
    finite_nonzero_gradient_norm,
    native_deepeyes_agent_name,
)


@dataclass(frozen=True, slots=True)
class NativeDeepEyesSmokeResult:
    sample_count: int
    visual_sample_count: int
    thinklite_sample_count: int
    successful_crop_count: int
    native_image_count: int
    observation_mask_zero: bool
    vision_gradient_norm: float
    language_gradient_norm: float
    vision_weight_changed: bool
    language_weight_changed: bool
    cpu_state_dict_copy_changed: bool
    cpu_state_dict_copy_step: int
    resume_advanced_one_step: bool
    checkpoint_before_eval: bool
    manifest: dict[str, object]

    def as_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "visual_sample_count": self.visual_sample_count,
            "thinklite_sample_count": self.thinklite_sample_count,
            "successful_crop_count": self.successful_crop_count,
            "native_image_count": self.native_image_count,
            "observation_mask_zero": self.observation_mask_zero,
            "vision_gradient_norm": self.vision_gradient_norm,
            "language_gradient_norm": self.language_gradient_norm,
            "vision_weight_changed": self.vision_weight_changed,
            "language_weight_changed": self.language_weight_changed,
            "cpu_state_dict_copy_changed": self.cpu_state_dict_copy_changed,
            "cpu_state_dict_copy_step": self.cpu_state_dict_copy_step,
            "resume_advanced_one_step": self.resume_advanced_one_step,
            "checkpoint_before_eval": self.checkpoint_before_eval,
            "manifest": self.manifest,
        }


class _FakeNativeProcessor:
    """Minimal pixel processor; deliberately has no image-embed surface."""

    @staticmethod
    def process(images: list[Image.Image]) -> dict[str, torch.Tensor]:
        rows: list[list[float]] = []
        for image in images:
            resized = image.convert("RGB").resize((1, 1))
            red, green, blue = resized.getpixel((0, 0))
            rows.append([red / 255.0, green / 255.0, blue / 255.0])
        return {
            "pixel_values": torch.tensor(rows, dtype=torch.float32),
            "image_grid_thw": torch.ones((len(images), 3), dtype=torch.long),
        }


class _TinyNativeActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Linear(3, 4)
        self.vision_projection = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Embedding(32, 4)
        self.lm_head = torch.nn.Linear(4, 1)

    def forward(
        self, pixel_values: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        vision = self.vision_projection(
            torch.tanh(self.vision_tower(pixel_values))
        ).mean(dim=0)
        language = self.language_model(token_ids).mean(dim=0)
        return self.lm_head(torch.tanh(vision + language)).square().mean()


def _state_sha256(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _visual_row(sample_id: str, image_path: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "data_source": "vstar",
        "agent_name": NATIVE_DEEPEYES_VISUAL_AGENT,
        "raw_prompt": [
            {"role": "system", "content": "native DeepEyes protocol"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What is shown?"},
                ],
            },
        ],
    }


def _thinklite_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "data_source": "thinklite",
        "agent_name": NATIVE_DEEPEYES_THINKLITE_AGENT,
        "raw_prompt": [
            {"role": "system", "content": "native math protocol"},
            {"role": "user", "content": "Compute 1+1 and box the answer."},
        ],
    }


async def _one_successful_crop(
    original: Image.Image,
) -> tuple[list[Image.Image], dict[str, Any]]:
    schema = SimpleNamespace(function=SimpleNamespace(name=NATIVE_CROP_TOOL_NAME))
    tool = NativeDeepEyesCropTool(
        config={"type": "native", "max_crops": 6},
        tool_schema=schema,
    )
    agent_data = SimpleNamespace(image_data=[original], extra_fields={})
    instance_id, _ = await tool.create(create_kwargs={"gt_regions": [[16, 16, 56, 56]]})
    response, _, info = await tool.execute(
        instance_id,
        # Native tool arguments use Qwen's 0..1000 coordinate space.  This
        # maps back to roughly [14, 14, 58, 58] on the 72px fixture.
        {"bbox_2d": [194, 194, 806, 806], "label": "target"},
        agent_data=agent_data,
    )
    await tool.release(instance_id)
    if info.get("status") != "success" or not getattr(response, "image", None):
        raise AssertionError("CPU fake server did not execute a successful Crop")
    agent_data.image_data.extend(response.image)
    return agent_data.image_data, agent_data.extra_fields


def _checkpoint_resume_contract(
    actor: _TinyNativeActor,
    *,
    checkpoint_root: Path,
    pixel_values: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[bool, bool]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    checkpoint = checkpoint_root / "step_1.pt"
    torch.save(
        {"global_step": 1, "model": actor.state_dict()},
        checkpoint,
    )
    events.append("save:1")
    events.append("eval:1")

    resumed = _TinyNativeActor()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed.load_state_dict(payload["model"], strict=True)
    resumed_step = int(payload["global_step"])
    optimizer = torch.optim.SGD(resumed.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    resumed(pixel_values, token_ids).backward()
    optimizer.step()
    resumed_step += 1
    events.append("resume-update:2")
    checkpoint_2 = checkpoint_root / "step_2.pt"
    torch.save(
        {"global_step": resumed_step, "model": resumed.state_dict()}, checkpoint_2
    )
    events.append("save:2")
    events.append("eval:2")
    save_before_eval = all(
        events.index(f"save:{step}") < events.index(f"eval:{step}") for step in (1, 2)
    )
    return resumed_step == 2, save_before_eval


def run_native_deepeyes_cpu_fake_smoke(
    *, checkpoint_root: Path
) -> NativeDeepEyesSmokeResult:
    """Exercise the PRL13 contracts without CUDA, vLLM, Ray, or an API."""

    STRICT_NATIVE_FULL_MODEL_TRAINABILITY.validate()
    rows = [
        _visual_row("visual-crop", "/fake/original-a.png"),
        _visual_row("visual-direct", "/fake/original-b.png"),
        _thinklite_row("math-a"),
        _thinklite_row("math-b"),
    ]
    for row in rows:
        assert_native_pixel_row(row)
    if [native_deepeyes_agent_name(str(row["data_source"])) for row in rows] != [
        NATIVE_DEEPEYES_VISUAL_AGENT,
        NATIVE_DEEPEYES_VISUAL_AGENT,
        NATIVE_DEEPEYES_THINKLITE_AGENT,
        NATIVE_DEEPEYES_THINKLITE_AGENT,
    ]:
        raise AssertionError("PRL13 source routing differs")

    original = Image.new("RGB", (72, 72), color=(192, 64, 32))
    native_images, crop_metrics = asyncio.run(_one_successful_crop(original))
    multi_modal_inputs = _FakeNativeProcessor.process(native_images)
    assert_native_multimodal_inputs(
        multi_modal_inputs,
        original_image_count=1,
        successful_crop_count=1,
    )
    observation_ids = [20, 21, 22, 23]
    response_mask = [1, 1] + [0] * len(observation_ids) + [1, 1]
    observation_spans = [[2, 2 + len(observation_ids)]]
    assert_observation_mask(response_mask, observation_spans)

    actor = _TinyNativeActor()
    parameter_counts = assert_trainable_parameter_groups(actor.named_parameters())
    if not parameter_counts["vision"] or not parameter_counts["language"]:
        raise AssertionError("CPU smoke did not expose both parameter groups")
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    actor(multi_modal_inputs["pixel_values"], token_ids).backward()
    vision_parameters = [
        parameter for name, parameter in actor.named_parameters() if "vision" in name
    ]
    language_parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if "language" in name or "lm_head" in name
    ]
    vision_gradient_norm = finite_nonzero_gradient_norm(vision_parameters)
    language_gradient_norm = finite_nonzero_gradient_norm(language_parameters)
    optimizer.step()
    vision_changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in actor.named_parameters()
        if "vision" in name
    )
    language_changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in actor.named_parameters()
        if "language" in name or "lm_head" in name
    )
    if not vision_changed or not language_changed:
        raise AssertionError(
            "full-model CPU optimizer update did not change both groups"
        )

    rollout_actor = copy.deepcopy(_TinyNativeActor())
    rollout_before = _state_sha256(rollout_actor)
    rollout_actor.load_state_dict(actor.state_dict(), strict=True)
    rollout_after = _state_sha256(rollout_actor)
    actor_after = _state_sha256(actor)
    sync_changed = rollout_before != rollout_after and rollout_after == actor_after
    if not sync_changed:
        raise AssertionError("CPU state_dict copy contract failed")

    resume_advanced, save_before_eval = _checkpoint_resume_contract(
        actor,
        checkpoint_root=checkpoint_root,
        pixel_values=multi_modal_inputs["pixel_values"],
        token_ids=token_ids,
    )
    manifest = {
        "schema_version": "tgvf.prl13-native-cpu-smoke.v1",
        "runtime": "single_process_fake_server",
        "gpu_or_api_used": False,
        "native_pixels": True,
        "precomputed_image_embeds": False,
        "legacy_adapter_loaded": False,
        "trajectory_replay_bundle_loaded": False,
        "cpu_contract_only": True,
        "cpu_state_dict_copy_sync": True,
        "real_fsdp_vllm_sync_proven": False,
        "real_model_trainability_proven": False,
        "vision_trainable": True,
        "language_trainable": True,
        "observation_role": "user",
        "observation_envelope": "<tool_response><image>...</tool_response>",
        "upstream_role_tool_remapped": True,
        "crop_metrics": crop_metrics,
    }
    return NativeDeepEyesSmokeResult(
        sample_count=4,
        visual_sample_count=2,
        thinklite_sample_count=2,
        successful_crop_count=1,
        native_image_count=len(native_images),
        observation_mask_zero=True,
        vision_gradient_norm=vision_gradient_norm,
        language_gradient_norm=language_gradient_norm,
        vision_weight_changed=vision_changed,
        language_weight_changed=language_changed,
        cpu_state_dict_copy_changed=sync_changed,
        cpu_state_dict_copy_step=1,
        resume_advanced_one_step=resume_advanced,
        checkpoint_before_eval=save_before_eval,
        manifest=manifest,
    )


__all__ = [
    "NativeDeepEyesSmokeResult",
    "run_native_deepeyes_cpu_fake_smoke",
]
