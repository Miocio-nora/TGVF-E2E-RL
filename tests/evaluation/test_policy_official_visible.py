from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.evaluation.policy_coredev import (
    PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
    CoreDevTask,
)
from tgvf_rl.evaluation.policy_official_visible import (
    OfficialVisiblePolicyEvaluator,
    _official_visible_mm_processor_kwargs,
    _render_native_prompt,
    normalize_official_visible_crop_box,
    normalize_qwen3_official_visible_crop_box,
    official_visible_observation_message,
    validate_official_visible_processor,
)
from tgvf_rl.evaluation.policy_no_tool_matched import (
    validate_no_tool_matched_processor,
)
from tgvf_rl.framework.verl.native_crop_tool import normalize_native_crop_box
from tgvf_rl.framework.vllm import ContentAddressedVLLMTurnRNG
from tgvf_rl.policy.deepeyes_official_protocol import (
    USER_PROMPT_V2,
    build_visual_messages,
)


_QWEN3_PROCESSOR_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")


class _Tokenizer:
    image_pad_id = 99

    def __init__(self) -> None:
        self.decoded: dict[int, str] = {}

    def __len__(self) -> int:
        return 123

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|image_pad|>"
        return self.image_pad_id

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        parts = text.split("<|image_pad|>")
        ids: list[int] = []
        for index, part in enumerate(parts):
            ids.extend(1 + (ord(character) % 80) for character in part)
            if index + 1 < len(parts):
                ids.append(self.image_pad_id)
        return ids

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
            "spaces_between_special_tokens": False,
        }
        return self.decoded[token_ids[0]]


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 65_536, "longest_edge": 16_777_216},
            patch_size=16,
            merge_size=2,
        )
        self.rendered: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[object],
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert tools == []
        chunks: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                body = content
            else:
                assert isinstance(content, list)
                body = "".join(
                    (
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        if item["type"] == "image"
                        else str(item["text"])
                    )
                    for item in content
                )
            chunks.append(
                f"<|im_start|>{message['role']}\n{body}<|im_end|>\n"
            )
        text = "".join(chunks) + "<|im_start|>assistant\n"
        self.rendered.append(text)
        return text

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        size: dict[str, int],
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert size == {
            "shortest_edge": 65_536,
            "longest_edge": 1_003_520,
        }
        assert return_tensors == "pt"
        canonical = self.tokenizer.encode(text[0], add_special_tokens=False)
        expanded: list[int] = []
        for token_id in canonical:
            expanded.extend(
                [token_id] * (4 if token_id == self.tokenizer.image_pad_id else 1)
            )
        return {
            "input_ids": torch.tensor([expanded], dtype=torch.long),
            "image_grid_thw": torch.tensor(
                [[1, 4, 4] for _image in images], dtype=torch.long
            ),
        }


def test_official_crop_maps_qwen3_grid_on_non_square_source() -> None:
    box = normalize_official_visible_crop_box(
        [75, 306, 435, 710], image_width=500, image_height=333
    )

    assert box.requested_bbox_2d == (75, 306, 435, 710)
    assert box.source_bbox_2d == (37, 101, 217, 236)
    assert (box.width, box.height) == (180, 135)


def test_training_and_evaluation_use_the_same_qwen3_family_mapping() -> None:
    for model_bbox, source_size, expected in (
        ([75, 306, 435, 710], (500, 333), (37, 101, 217, 236)),
        ([0, 0, 1000, 1000], (1440, 720), (0, 0, 1440, 720)),
        ([125, 200, 875, 800], (320, 640), (40, 128, 280, 512)),
    ):
        evaluation = normalize_official_visible_crop_box(
            model_bbox, image_width=source_size[0], image_height=source_size[1]
        )
        training = normalize_native_crop_box(
            model_bbox, image_width=source_size[0], image_height=source_size[1]
        )
        assert evaluation.source_bbox_2d == expected
        assert (training.left, training.top, training.right, training.bottom) == (
            expected
        )


def test_qwen3_official_crop_maps_relative_box_to_original_pixels() -> None:
    box = normalize_qwen3_official_visible_crop_box(
        [100, 100, 700, 700], image_width=100, image_height=80
    )

    assert box.requested_bbox_2d == (100.0, 100.0, 700.0, 700.0)
    assert box.source_bbox_2d == (10, 8, 70, 56)


@pytest.mark.parametrize(
    "bbox",
    (
        [-1, 0, 500, 500],
        [0, 0, 1001, 500],
        [500, 0, 500, 500],
        [0, 0, 500.5, 500],
        [0, 0, 1, 1],
    ),
)
def test_official_crop_rejects_invalid_model_geometry(bbox: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize_official_visible_crop_box(bbox, image_width=100, image_height=80)


def test_official_observation_is_user_framed_native_image() -> None:
    marker = object()

    assert official_visible_observation_message(image=marker) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "<tool_response>"},
            {"type": "image", "image": marker},
            {"type": "text", "text": USER_PROMPT_V2 + "</tool_response>"},
        ],
    }


def test_static_processor_proof_includes_native_visual_expansion() -> None:
    proof = validate_official_visible_processor(
        _Processor(), tokenizer_length=123, image_max_pixels=1_003_520
    )

    assert proof["tools_argument_empty"] is True
    assert proof["observation_role"] == "user"
    assert proof["continuation_parity"] is True
    assert proof["continuation_environment_text_sha256"] == (
        "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
    )
    assert proof["success_environment_renderer"] == (
        "render_qwen_native_matched_crop_success_environment_text"
    )
    assert proof["native_original_image_count"] == 1
    assert proof["native_crop_image_count"] == 1
    assert proof["synthetic_native_visual_token_counts"] == [4, 4]
    assert proof["configured_image_max_pixels"] == 1_003_520
    assert proof["processor_image_size"] == {
        "shortest_edge": 65_536,
        "longest_edge": 16_777_216,
    }
    assert proof["effective_processor_image_size"] == {
        "shortest_edge": 65_536,
        "longest_edge": 1_003_520,
    }
    assert proof["synthetic_native_source_pixel_areas"] == [3_145_728, 3_145_728]
    assert proof["synthetic_native_represented_pixel_areas"] == [4096, 4096]
    assert proof["runtime_mm_processor_kwargs"] == {
        "size": {
            "shortest_edge": 65_536,
            "longest_edge": 1_003_520,
        }
    }
    assert proof["runtime_override_path"] == ("mm_processor_kwargs.size.longest_edge")
    assert proof["vllm_012_shallow_hashable"] is True
    assert proof["nested_images_kwargs_present"] is False
    assert proof["max_pixels_kwarg_present"] is False
    assert (
        proof["continuation_expanded_prompt_token_count"]
        > proof["continuation_prompt_token_count"]
    )


def test_no_tool_static_processor_proof_uses_true1m_size_cap() -> None:
    proof = validate_no_tool_matched_processor(
        _Processor(), tokenizer_length=123, image_max_pixels=1_003_520
    )

    assert proof["configured_image_max_pixels"] == 1_003_520
    assert proof["processor_image_size"]["longest_edge"] == 16_777_216
    assert proof["effective_processor_image_size"]["longest_edge"] == 1_003_520
    assert proof["synthetic_native_source_pixel_area"] == 3_145_728
    assert proof["synthetic_native_represented_pixel_area"] == 4096
    assert proof["synthetic_native_visual_token_count"] == 4
    assert proof["runtime_mm_processor_kwargs"] == {
        "size": {
            "shortest_edge": 65_536,
            "longest_edge": 1_003_520,
        }
    }
    assert proof["runtime_override_path"] == ("mm_processor_kwargs.size.longest_edge")
    assert proof["vllm_012_shallow_hashable"] is True
    assert proof["nested_images_kwargs_present"] is False
    assert proof["max_pixels_kwarg_present"] is False
    assert proof["tool_schema_visible"] is False


def test_vllm_true1m_kwargs_use_one_level_hashable_size_mapping() -> None:
    kwargs = _official_visible_mm_processor_kwargs(_Processor(), 1_003_520)

    assert kwargs == {
        "size": {
            "shortest_edge": 65_536,
            "longest_edge": 1_003_520,
        }
    }
    assert "images_kwargs" not in kwargs
    # vLLM 0.12 shallow-wraps each top-level dict before using it as an
    # lru_cache key; nested images_kwargs would leave an unhashable dict value.
    assert hash(frozenset(kwargs["size"].items()))


def test_real_qwen3_processor_cap_changes_large_image_grid() -> None:
    if not _QWEN3_PROCESSOR_PATH.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    transformers = pytest.importorskip(
        "transformers", reason="accepted local Qwen3 processor is unavailable"
    )
    delegate = transformers.AutoProcessor.from_pretrained(
        _QWEN3_PROCESSOR_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    class _RecordingProcessor:
        def __init__(self) -> None:
            self.tokenizer = delegate.tokenizer
            self.image_processor = delegate.image_processor
            self.grids: list[list[list[int]]] = []

        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            return delegate.apply_chat_template(*args, **kwargs)

        def __call__(self, **kwargs: object) -> object:
            batch = delegate(**kwargs)
            self.grids.append(batch["image_grid_thw"].tolist())
            return batch

    processor = _RecordingProcessor()
    image = Image.new("RGB", (2048, 1536), (10, 20, 30))
    messages = list(build_visual_messages("Which detail is visible?", image="<image>"))
    try:
        _low_text, low_ids, low_counts = _render_native_prompt(
            processor,
            messages,
            images=(image,),
            image_max_pixels=512 * 512,
        )
        _high_text, high_ids, high_counts = _render_native_prompt(
            processor,
            messages,
            images=(image,),
            image_max_pixels=1_003_520,
        )
    finally:
        image.close()

    assert processor.grids == [[[1, 26, 36]], [[1, 54, 72]]]
    assert low_counts == (234,)
    assert high_counts == (972,)
    assert len(low_ids) < len(high_ids)
    patch_size = processor.image_processor.patch_size
    represented_areas = [
        grid[0][1] * grid[0][2] * patch_size**2 for grid in processor.grids
    ]
    assert represented_areas == [239_616, 995_328]
    assert represented_areas[0] <= 512 * 512
    assert represented_areas[1] <= 1_003_520


def test_official_visible_uses_paired_rng_and_preserves_legacy_seed_path() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        async def generate(self, **kwargs: object) -> object:
            assert kwargs["mm_processor_kwargs"] == {
                "size": {
                    "shortest_edge": 65_536,
                    "longest_edge": 1_003_520,
                }
            }
            self.seeds.append(kwargs["sampling_params"]["seed"])
            return SimpleNamespace(
                token_ids=[201],
                log_probs=[-0.1],
                extra_fields={"tgvf_vllm_finish_reason": "stop"},
            )

    def evaluator(
        *, step: int, namespace: str | None
    ) -> tuple[OfficialVisiblePolicyEvaluator, _Manager]:
        manager = _Manager()
        instance = OfficialVisiblePolicyEvaluator.__new__(
            OfficialVisiblePolicyEvaluator
        )
        instance.config = SimpleNamespace(
            max_model_len=32768,
            paired_seed_namespace=namespace,
        )
        instance.run = SimpleNamespace(
            policy=SimpleNamespace(
                image_max_pixels=1_003_520,
                sampling=SimpleNamespace(
                    remaining_response_tokens=lambda consumed: 100 - consumed,
                    as_vllm_parameters=lambda max_tokens: {"max_tokens": max_tokens},
                ),
            ),
            rollout_rng=SimpleNamespace(master_seed=42),
        )
        instance.manager = manager
        instance.processor = _Processor()
        instance.image_max_pixels = 1_003_520
        instance.policy_version = PolicyVersion("run", step, str(step % 10) * 64)
        instance.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "A")
        instance.evaluation_identity = {
            "sampling_rng": {
                "schema_version": PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
                "mode": "common_random_numbers_per_task_turn",
                "seed_namespace": namespace,
                "master_seed": 42,
                "task_manifest_sha256": "a" * 64,
                "protocol_sha256": "b" * 64,
            }
        }
        return instance, manager

    namespace = "coredev2511/crop/step8-step16/temp1/seed42/v1"
    step8, manager8 = evaluator(step=8, namespace=namespace)
    step16, manager16 = evaluator(step=16, namespace=namespace)
    asyncio.run(
        step8._sample_turn(
            trajectory_id="different-step8-trajectory",
            prompt_ids=(1, 2, 3),
            images=(),
            turn_index=2,
            consumed_tokens=0,
            sample_id="sample-7",
            rollout_index=0,
        )
    )
    asyncio.run(
        step16._sample_turn(
            trajectory_id="different-step16-trajectory",
            prompt_ids=(9, 8),
            images=(),
            turn_index=2,
            consumed_tokens=0,
            sample_id="sample-7",
            rollout_index=0,
        )
    )
    assert manager8.seeds == manager16.seeds

    legacy, legacy_manager = evaluator(step=8, namespace=None)
    legacy_prompt = (4, 5, 6)
    legacy_policy = legacy.policy_version
    asyncio.run(
        legacy._sample_turn(
            trajectory_id="legacy-trajectory",
            prompt_ids=legacy_prompt,
            images=(),
            turn_index=1,
            consumed_tokens=0,
            sample_id="sample-7",
            rollout_index=0,
        )
    )
    expected = ContentAddressedVLLMTurnRNG(
        master_seed=42,
        stream_identity="legacy-trajectory",
    ).for_turn(
        legacy_prompt,
        turn_index=1,
        behavior_policy=legacy_policy,
    )
    assert legacy_manager.seeds == [expected.seed]


def test_evaluator_returns_native_source_pixel_crop_audit(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 80), (10, 20, 30)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    task = CoreDevTask(
        ordinal=0,
        dataset="fixture",
        row_number=0,
        index="sample",
        sample_id="sample",
        question="Which option?",
        image_paths=(str(image_path),),
        image_sha256s=(image_sha256,),
        image_dimensions=((100, 80),),
    )
    processor = _Processor()
    processor.tokenizer.decoded = {
        201: (
            '<think>inspect</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":{"bbox_2d":[0,50,710,760]}}</tool_call>'
        ),
        202: "<think>done</think>A",
    }

    class _Manager:
        def __init__(self) -> None:
            self.calls = 0
            self.fail_context = False
            self.released: list[str] = []

        async def generate(self, **kwargs: object) -> object:
            self.calls += 1
            assert len(kwargs["image_data"]) == self.calls
            assert kwargs["mm_processor_kwargs"] == {
                "size": {
                    "shortest_edge": 65_536,
                    "longest_edge": 1_003_520,
                }
            }
            if self.fail_context and self.calls == 2:
                raise ValueError(
                    "The decoder prompt (length 32891) is longer than the maximum "
                    "model length of 32768."
                )
            token_id = 200 + self.calls
            return SimpleNamespace(
                token_ids=[token_id],
                log_probs=[-0.1],
                extra_fields={"tgvf_vllm_finish_reason": "stop"},
            )

        async def release_trajectory(self, request_id: str) -> None:
            self.released.append(request_id)

    manager = _Manager()
    policy_version = PolicyVersion("run", 0, "a" * 64)
    evaluator = OfficialVisiblePolicyEvaluator.__new__(OfficialVisiblePolicyEvaluator)
    evaluator.config = SimpleNamespace(
        evaluation_id="eval",
        max_model_len=32768,
        uses_legacy_coredev_manifest=False,
    )
    evaluator.run = SimpleNamespace(
        model=ModelIdentity(
            family="qwen3_vl",
            model_name="Qwen3-VL-8B-Instruct",
            revision_or_path="fixture",
            tokenizer_length=123,
            chat_template_sha256="b" * 64,
        ),
        policy=SimpleNamespace(
            image_max_pixels=1_003_520,
            sampling=SimpleNamespace(
                remaining_response_tokens=lambda consumed: 100 - consumed,
                as_vllm_parameters=lambda max_tokens: {
                    "max_tokens": max_tokens,
                    "logprobs": True,
                },
            ),
        ),
        rollout_rng=SimpleNamespace(master_seed=42),
    )
    evaluator.manager = manager
    evaluator.processor = processor
    evaluator.image_max_pixels = 1_003_520
    evaluator.policy_version = policy_version
    evaluator.tokenizer = processor.tokenizer

    trajectory = asyncio.run(evaluator.evaluate(task))

    assert trajectory.stop == "final_answer"
    assert trajectory.final_answer == "A"
    assert manager.calls == 2
    assert len(manager.released) == 1
    assert trajectory.tool_calls[0]["bbox_2d"] == [0, 50, 710, 760]
    assert trajectory.tool_calls[0]["source_bbox_2d"] == [0, 4, 71, 60]
    assert trajectory.tool_calls[0]["coordinate_space"] == ("qwen3_relative_0_1000")
    assert trajectory.tool_calls[0]["coordinate_reference_size"] == [1000, 1000]
    assert trajectory.tool_calls[0]["source_size"] == [100, 80]
    assert trajectory.tool_calls[0]["crop_source"] == "immutable_original_image"

    manager.calls = 0
    manager.fail_context = True
    limited = asyncio.run(evaluator.evaluate(task))
    assert limited.stop == "context_limit"
    assert limited.final_answer is None
    assert len(limited.tool_calls) == 1
    assert limited.tool_errors[-1]["code"] == "context_limit"
    assert limited.tool_errors[-1]["recoverable"] is False

    manager.calls = 0
    manager.fail_context = False
    evaluator.config.max_model_len = 1
    locally_limited = asyncio.run(evaluator.evaluate(task))
    assert locally_limited.stop == "context_limit"
    assert locally_limited.final_answer is None
    assert manager.calls == 0

    # A mixed action/final turn violates the hard </tool_call> boundary.  It
    # must be visible as a protocol failure, never accepted as a direct answer
    # with the requested Crop silently skipped.
    manager.calls = 0
    evaluator.config.max_model_len = 32768
    processor.tokenizer.decoded[201] = (
        '<think>inspect</think><tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,50,710,760]}}</tool_call> A'
    )
    mixed = asyncio.run(evaluator.evaluate(task))
    assert mixed.stop == "malformed_action"
    assert mixed.final_answer is None
    assert manager.calls == 1
    assert mixed.tool_calls == ()
    assert mixed.tool_errors[-1]["code"] == "tool_call_terminal_suffix"
    assert mixed.tool_errors[-1]["recoverable"] is False
