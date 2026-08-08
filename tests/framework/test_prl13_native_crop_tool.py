from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tgvf_rl.framework.verl.native_crop_tool import (
    NATIVE_CROP_TOOL_NAME,
    NativeCropBox,
    NativeDeepEyesCropTool,
    crop_original_image,
    ensure_native_crop_audit_fields,
    normalize_native_crop_box,
)


def test_visual_crop_audit_defaults_cover_direct_no_call() -> None:
    fields = ensure_native_crop_audit_fields({})
    assert fields["crop_call_count"] == 0
    assert fields["crop_action_count"] == 0
    assert fields["crop_error_count"] == 0
    assert fields["crop_boxes"] == []
    assert fields["crop_area_fractions"] == []
    assert fields["crop_observation_token_spans"] == []


def test_crop_maps_qwen3_grid_and_reports_grounding_metrics() -> None:
    original = Image.new("RGB", (500, 333), color="red")
    result = crop_original_image(
        original,
        [75, 306, 435, 710],
        gt_regions=(NativeCropBox(50, 110, 180, 220),),
    )
    assert result.model_box == NativeCropBox(75, 306, 435, 710)
    assert result.box == NativeCropBox(37, 101, 217, 236)
    assert result.image.size == (180, 135)
    assert result.coordinate_space == "qwen3_relative_0_1000"
    assert result.conversion_version == "qwen3-relative-1000-floor-v1"
    assert result.coordinate_reference_size == (1000, 1000)
    assert result.source_size == (500, 333)
    assert 0 < result.crop_area_fraction < 1
    assert result.first_call_iou is not None and result.first_call_iou > 0
    assert result.best_gt_coverage == 1.0


@pytest.mark.parametrize(
    "bbox",
    (
        [0, 0, 6, 40],
        [0, 0, 40, 6],
        [0, 0, float("nan"), 50],
        [-1, 0, 500, 500],
        [0, 0, 1001, 500],
        [0, 0, 1, 1],
    ),
)
def test_crop_rejects_reference_invalid_boxes(bbox: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize_native_crop_box(bbox, image_width=5000, image_height=5000)


def test_native_tool_always_crops_original_and_tracks_actions() -> None:
    async def exercise() -> None:
        schema = SimpleNamespace(function=SimpleNamespace(name=NATIVE_CROP_TOOL_NAME))
        tool = NativeDeepEyesCropTool(
            config={"type": "native", "max_crops": 6}, tool_schema=schema
        )
        original = Image.new("RGB", (120, 120), color="blue")
        agent_data = SimpleNamespace(image_data=[original], extra_fields={})
        instance, _ = await tool.create(
            create_kwargs={"gt_regions": [[20, 20, 70, 70]]}
        )
        first, _, first_info = await tool.execute(
            instance, {"bbox_2d": [80, 80, 800, 800]}, agent_data=agent_data
        )
        # Mimic ToolAgentLoop appending the first crop.  A second request is
        # still evaluated against image_data[0], never against this crop.
        agent_data.image_data.extend(first.image)
        second, _, second_info = await tool.execute(
            instance, {"bbox_2d": [500, 500, 950, 950]}, agent_data=agent_data
        )
        await tool.release(instance)
        assert first_info["status"] == second_info["status"] == "success"
        assert first.image[0].size == (87, 87)
        assert second.image[0].size == (54, 54)
        assert first_info["model_bbox_2d"] == [80, 80, 800, 800]
        assert first_info["source_bbox_2d"] == [9, 9, 96, 96]
        assert agent_data.extra_fields["crop_call_count"] == 2
        assert agent_data.extra_fields["crop_action_count"] == 2
        assert agent_data.extra_fields["crop_first_call_iou"] is not None
        assert agent_data.extra_fields["crop_best_call_iou"] is not None
        assert len(agent_data.extra_fields["crop_area_fractions"]) == 2
        assert agent_data.extra_fields["crop_model_boxes"] == [
            [80, 80, 800, 800],
            [500, 500, 950, 950],
        ]
        assert agent_data.extra_fields["crop_source_boxes"] == [
            [9, 9, 96, 96],
            [60, 60, 114, 114],
        ]

    asyncio.run(exercise())


def test_native_tool_invalid_call_is_a_call_but_not_an_action() -> None:
    async def exercise() -> None:
        schema = SimpleNamespace(function=SimpleNamespace(name=NATIVE_CROP_TOOL_NAME))
        tool = NativeDeepEyesCropTool(
            config={"type": "native", "max_crops": 6}, tool_schema=schema
        )
        agent_data = SimpleNamespace(
            image_data=[Image.new("RGB", (120, 120), color="blue")],
            extra_fields={},
        )
        instance, _ = await tool.create(create_kwargs={"gt_regions": []})
        response, _, info = await tool.execute(
            instance, {"bbox_2d": [0, 0, 1, 1]}, agent_data=agent_data
        )
        await tool.release(instance)
        assert info["status"] == "invalid_crop"
        assert response.image is None
        assert agent_data.extra_fields["crop_call_count"] == 1
        assert agent_data.extra_fields["crop_action_count"] == 0
        assert agent_data.extra_fields["crop_error_count"] == 1
        assert agent_data.extra_fields["crop_boxes"] == []
        assert agent_data.extra_fields["crop_observation_token_spans"] == []

    asyncio.run(exercise())


def test_upstream_tool_agent_loop_appends_crop_pixels_and_masks_observation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("verl")
    import torch
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import HermesToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.workers.rollout.replica import TokenOutput

    from tgvf_rl.framework.verl.native_deepeyes_agent_loop import (
        NativeDeepEyesAgentLoop,
    )

    assert issubclass(NativeDeepEyesAgentLoop, ToolAgentLoop)

    class FakeTokenizer:
        eos_token_id = 0

        def __init__(self) -> None:
            self.values: dict[int, str] = {}
            self.next_id = 1

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            result: list[int] = []
            for character in text:
                token_id = self.next_id
                self.next_id += 1
                self.values[token_id] = character
                result.append(token_id)
            return result

        def decode(self, ids: list[int], **_kwargs: object) -> str:
            return "".join(self.values[int(token_id)] for token_id in ids)

    class FakeProcessor:
        def __init__(self, tokenizer: FakeTokenizer) -> None:
            self.tokenizer = tokenizer
            self.image_processor = SimpleNamespace(patch_size=14)
            self.rendered_roles: list[list[str]] = []
            self.rendered_texts: list[str] = []

        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tokenize: bool,
            **_kwargs: object,
        ) -> str | list[int]:
            self.rendered_roles.append(
                [str(message.get("role")) for message in messages]
            )
            parts: list[str] = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    message_parts: list[str] = []
                    for item in content:
                        if item.get("type") == "image":
                            message_parts.append("<image>")
                        elif item.get("type") == "text":
                            message_parts.append(str(item.get("text", "")))
                    parts.append("".join(message_parts))
                else:
                    parts.append(str(content))
            text = "\n".join(parts) + "<assistant>"
            self.rendered_texts.append(text)
            return self.tokenizer.encode(text) if tokenize else text

        def __call__(
            self,
            *,
            text: list[str],
            images: list[Image.Image] | None = None,
            **_kwargs: object,
        ) -> dict[str, torch.Tensor]:
            del images
            return {"input_ids": torch.tensor([self.tokenizer.encode(text[0])])}

    class FakeServer:
        def __init__(self, tokenizer: FakeTokenizer) -> None:
            self.tokenizer = tokenizer
            self.calls = 0
            self.sampling_params: list[dict[str, object]] = []

        async def generate(self, **kwargs: object) -> TokenOutput:
            self.calls += 1
            self.sampling_params.append(dict(kwargs["sampling_params"]))
            if self.calls <= 4:
                text = (
                    '<think>zoom</think><tool_call>{"name":"image_zoom_in_tool",'
                    '"arguments":{"bbox_2d":[100,100,800,800]}}</tool_call>'
                )
            else:
                text = "<think>done</think>green"
            return TokenOutput(
                token_ids=self.tokenizer.encode(text),
                # Pinned sync vLLM emits global_steps only.  AgentLoopWorker
                # adds null min/max compatibility columns after this loop.
                extra_fields={"global_steps": 0},
            )

    async def exercise_loop() -> None:
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": NATIVE_CROP_TOOL_NAME,
                    "description": "crop original pixels",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {
                                "type": "array",
                                "description": "xyxy",
                            }
                        },
                        "required": ["bbox_2d"],
                    },
                },
            }
        )
        tool = NativeDeepEyesCropTool(
            config={"type": "native", "max_crops": 6}, tool_schema=schema
        )
        tokenizer = FakeTokenizer()
        image_path = tmp_path / "fake-original.png"
        Image.new("RGB", (96, 96), color="green").save(image_path)
        loop = NativeDeepEyesAgentLoop.__new__(NativeDeepEyesAgentLoop)
        loop.max_user_turns = 6
        loop.max_assistant_turns = 7
        loop.max_parallel_calls = 1
        loop.max_tool_response_length = 4096
        loop.tool_response_truncate_side = "right"
        loop.tools = {tool.name: tool}
        loop.tool_schemas = [schema.model_dump(exclude_none=True)]
        loop.tool_parser = HermesToolParser(tokenizer)
        loop.tool_parser_name = "hermes"
        loop.prompt_length = 4096
        loop.response_length = 4096
        loop.rollout_config = SimpleNamespace(prompt_length=4096, response_length=4096)
        loop.enable_continuous_token = False
        loop.processor = FakeProcessor(tokenizer)
        loop.tokenizer = tokenizer
        loop.dataset_cls = object
        loop.data_config = {}
        loop.apply_chat_template_kwargs = {}
        loop.mm_processor_kwargs = {}
        loop.server_manager = FakeServer(tokenizer)
        loop.loop = asyncio.get_running_loop()
        loop.turn_separator = []
        loop.system_prompt = []
        output = await loop.run(
            {},
            raw_prompt=[
                {"role": "system", "content": "official schema is visible"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.resolve())},
                        {"type": "text", "text": "What color?"},
                    ],
                },
            ],
            data_source="vstar",
            tools_kwargs={
                NATIVE_CROP_TOOL_NAME: {
                    "create_kwargs": {"gt_regions": [[12, 12, 68, 68]]}
                }
            },
        )
        assert len(output.multi_modal_data["images"]) == 5
        assert output.extra_fields["native_total_image_count"] == 5
        assert output.extra_fields["native_crop_image_count"] == 4
        spans = output.extra_fields["crop_observation_token_spans"]
        assert len(spans) == 4
        start, stop = spans[0]
        assert stop > start
        for start, stop in spans:
            assert output.response_mask[start:stop] == [0] * (stop - start)
        assert output.extra_fields["legacy_adapter_loaded"] is False
        assert output.extra_fields["observation_role"] == "user"
        assert output.extra_fields["global_steps"] == 0
        assert "min_global_steps" not in output.extra_fields
        assert "max_global_steps" not in output.extra_fields
        assert ["user"] in loop.processor.rendered_roles
        assert any(
            "<tool_response><image>" in text and "</tool_response>" in text
            for text in loop.processor.rendered_texts
        )
        assert loop.server_manager.calls == 5
        assert loop.server_manager.sampling_params[0]["max_tokens"] == 4096
        assert (
            loop.server_manager.sampling_params[1]["max_tokens"] == 4096 - spans[0][1]
        )

    asyncio.run(exercise_loop())


def test_real_hydra_qwen_processor_preserves_source_pixels_and_replays_crops(
    tmp_path: Path,
) -> None:
    pytest.importorskip("verl")
    import hydra
    from omegaconf import OmegaConf
    import torch
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopWorker,
        DictConfigWrap,
        ToolListWrap,
    )
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils.tokenizer import hf_processor
    from verl.workers.rollout.replica import TokenOutput

    from tgvf_rl.framework.verl.native_deepeyes_agent_loop import (
        NativeDeepEyesAgentLoop,
    )

    image_path = tmp_path / "source.png"
    Image.new("RGB", (137, 91), color=(33, 77, 121)).save(image_path)
    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": NATIVE_CROP_TOOL_NAME,
                "description": "map Qwen3 relative coordinates to original pixels",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "description": "Qwen3 relative 0..1000 xyxy",
                        },
                        "label": {"type": "string", "description": "optional"},
                    },
                    "required": ["bbox_2d"],
                },
            },
        }
    )
    tool = NativeDeepEyesCropTool(
        config={"type": "native", "max_crops": 6}, tool_schema=schema
    )
    processor = hf_processor(
        "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
        local_files_only=True,
    )

    class FakeServer:
        calls = 0
        sampling_params: list[dict[str, object]] = []

        async def generate(self, **kwargs: object) -> TokenOutput:
            self.calls += 1
            self.sampling_params.append(dict(kwargs["sampling_params"]))
            if self.calls == 1:
                text = (
                    '<think>zoom</think><tool_call>{"name":"image_zoom_in_tool",'
                    '"arguments":{"bbox_2d":[0,0,365,550],"label":"discarded"}}'
                    "</tool_call>"
                    '<tool_call>{"name":"image_zoom_in_tool",'
                    '"arguments":{"bbox_2d":[146,110,657,770],"label":"target"}}'
                    "</tool_call>"
                )
            else:
                text = "<think>done</think>blue"
            return TokenOutput(
                token_ids=processor.tokenizer.encode(text, add_special_tokens=False)
            )

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 8192,
                    "response_length": 20480,
                    "multi_turn": {
                        "max_user_turns": 6,
                        "max_assistant_turns": 7,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 4096,
                        "tool_response_truncate_side": "right",
                        "format": "hermes",
                    },
                }
            },
            "data": {
                "continuous_token": {"enable": False},
                "apply_chat_template_kwargs": {},
                "mm_processor_kwargs": {},
            },
        }
    )

    async def exercise_real_loop() -> tuple[NativeDeepEyesAgentLoop, object]:
        # AgentLoopWorker constructs each loop from inside its running event
        # loop; preserve that real lifecycle so executor futures bind correctly.
        loop = hydra.utils.instantiate(
            {
                "_target_": (
                    "tgvf_rl.framework.verl.native_deepeyes_agent_loop."
                    "NativeDeepEyesAgentLoop"
                )
            },
            trainer_config=DictConfigWrap(config),
            server_manager=FakeServer(),
            tokenizer=processor.tokenizer,
            processor=processor,
            dataset_cls=object,
            data_config=DictConfigWrap(config.data),
            tools=ToolListWrap([tool]),
        )
        output = await loop.run(
            {},
            raw_prompt=[
                {"role": "system", "content": "official schema is visible"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.resolve())},
                        {"type": "text", "text": "What color?"},
                    ],
                },
            ],
            data_source="vstar",
            tools_kwargs={
                NATIVE_CROP_TOOL_NAME: {
                    "create_kwargs": {"gt_regions": [[22, 12, 88, 68]]}
                }
            },
        )
        return loop, output

    loop, output = asyncio.run(exercise_real_loop())
    assert isinstance(loop, NativeDeepEyesAgentLoop)
    # The PIL objects own their pixels; source file lifetime is irrelevant.
    image_path.unlink()
    images = output.multi_modal_data["images"]
    assert [image.size for image in images] == [(137, 91), (70, 60)]
    assert images[0].getpixel((0, 0)) == (33, 77, 121)
    assert tool._instances == {}
    assert output.extra_fields["crop_action_count"] == 1
    assert loop.server_manager.sampling_params[0]["max_tokens"] == 10_240
    assert loop.server_manager.sampling_params[1]["max_tokens"] == 10_240
    start, stop = output.extra_fields["crop_observation_token_spans"][0]
    assert output.response_mask[start:stop] == [0] * (stop - start)
    rendered_observation = processor.tokenizer.decode(
        output.response_ids[start:stop], skip_special_tokens=False
    )
    assert "<|im_start|>user\n<tool_response>" in rendered_observation
    assert "<|vision_start|><|image_pad|>" in rendered_observation
    assert "<|image_pad|><|vision_end|>" in rendered_observation
    assert "</tool_response><|im_end|>" in rendered_observation

    # Exercise the installed AgentLoopManager actor replay path with the real
    # Qwen3 processor: both original and crop become pixel_values in order.
    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.processor = processor
    worker.tokenizer = processor.tokenizer
    worker.mm_processor_kwargs = {}
    input_ids = torch.tensor(
        [output.prompt_ids + output.response_ids], dtype=torch.long
    )
    multi_modal_inputs = worker._compute_multi_modal_inputs(output, input_ids)
    assert tuple(multi_modal_inputs["image_grid_thw"].shape) == (2, 3)
    assert multi_modal_inputs["pixel_values"].shape[0] > 0
    position_ids = worker._compute_position_ids(
        input_ids,
        torch.ones_like(input_ids),
        multi_modal_inputs,
        output.mm_processor_kwargs,
    )
    assert tuple(position_ids.shape) == (1, 4, input_ids.shape[-1])
