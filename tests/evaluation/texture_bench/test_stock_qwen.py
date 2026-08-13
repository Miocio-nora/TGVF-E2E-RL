from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

from PIL import Image
import pytest

from tgvf_rl.evaluation.texture_bench.schema import VisionPreprocessConfig
from tgvf_rl.evaluation.texture_bench.stock_qwen import (
    STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
    STOCK_QWEN_RESULT_SCHEMA,
    STOCK_QWEN_VISION,
    StockQwenVLLMRunner,
    stable_stock_qwen_seed,
)
from tgvf_rl.evaluation.texture_bench import stock_qwen


@dataclass(frozen=True, slots=True)
class _Task:
    """The exact CoreDevTask surface consumed by the stock runner."""

    ordinal: int
    dataset: str
    row_number: int
    index: str
    sample_id: str
    question: str
    image_paths: tuple[str, ...]
    image_sha256s: tuple[str, ...]
    image_dimensions: tuple[tuple[int, int], ...]

    @property
    def bound_sample_id(self) -> str:
        return self.sample_id


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        self.calls.append((messages, dict(kwargs)))
        return "<qwen-image-placeholder>\nQuestion?\n<assistant>"


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[
            tuple[list[dict[str, object]], list[dict[str, object]], bool]
        ] = []

    def generate(
        self,
        prompts: object,
        *,
        sampling_params: object,
        use_tqdm: bool,
    ) -> list[object]:
        prompt_rows = [dict(item) for item in prompts]  # type: ignore[union-attr]
        sampling_rows = [dict(item) for item in sampling_params]  # type: ignore[union-attr]
        self.calls.append((prompt_rows, sampling_rows, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"Answer is {letter}.",
                        token_ids=[100 + index, 200 + index],
                        finish_reason="stop",
                    )
                ]
            )
            for index, letter in enumerate("ABCDE"[: len(prompt_rows)])
        ]


def _write_image(
    path: Path, *, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


def _task(
    tmp_path: Path,
    *,
    ordinal: int = 0,
    sample_id: str = "mmad-fixture-0",
    question: str = "Which option describes the query texture?",
    size: tuple[int, int] = (713, 219),
) -> _Task:
    image_path = tmp_path / f"{sample_id}.png"
    _write_image(image_path, size=size, color=(17, 83, 141))
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return _Task(
        ordinal=ordinal,
        dataset="MMAD",
        row_number=ordinal,
        index=sample_id,
        sample_id=sample_id,
        question=question,
        image_paths=(str(image_path.resolve()),),
        image_sha256s=(digest,),
        image_dimensions=(size,),
    )


def test_runner_builds_image_first_qwen_prompt_without_pre_resize(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    processor = _FakeProcessor()
    engine = _FakeEngine()
    runner = StockQwenVLLMRunner(
        model_path=tmp_path / "model",
        processor=processor,
        engine=engine,
        batch_size=4,
    )

    rows = runner.run((task,))

    assert len(processor.calls) == 1
    messages, template_kwargs = processor.calls[0]
    assert template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "text"]
    assert content[0]["image"] == task.image_paths[0]
    assert content[1]["text"] == task.question

    prompts, sampling, use_tqdm = engine.calls[0]
    assert use_tqdm is False
    assert prompts[0]["prompt"] == "<qwen-image-placeholder>\nQuestion?\n<assistant>"
    assert prompts[0]["mm_processor_kwargs"] == {
        "min_pixels": 65_536,
        "max_pixels": 262_144,
    }
    image = prompts[0]["multi_modal_data"]["image"]
    assert image.size == (713, 219)
    assert STOCK_QWEN_VISION == VisionPreprocessConfig(
        min_pixels=65_536,
        max_pixels=262_144,
        preserve_aspect_ratio=True,
        pre_resize_assets=False,
    )
    assert sampling[0] == {
        "max_tokens": 2048,
        "temperature": 0.0,
        "seed": stable_stock_qwen_seed(task),
    }

    assert rows[0]["schema_version"] == STOCK_QWEN_RESULT_SCHEMA
    assert rows[0]["ordinal"] == task.ordinal
    assert rows[0]["sample_id"] == task.bound_sample_id
    assert rows[0]["final_answer"] == "Answer is A."
    assert rows[0]["model_response"] == {
        "text": "Answer is A.",
        "token_ids": [100, 200],
        "finish_reason": "stop",
    }
    vision = rows[0]["vision_identity"]
    assert vision["source_dimensions"] == [713, 219]
    assert vision["source_image_sha256"] == task.image_sha256s[0]
    assert vision["preprocess"]["pre_resize_assets"] is False
    assert vision["preprocess"]["max_pixels"] == 512 * 512
    json.dumps(rows, ensure_ascii=False, allow_nan=False)


def test_content_addressed_seed_is_stable_across_order_and_batching(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    seed_task = replace(task, image_sha256s=("a" * 64,))
    reordered = replace(seed_task, ordinal=91, row_number=47)

    seed = stable_stock_qwen_seed(seed_task)
    assert seed == 1_638_541_278
    assert stable_stock_qwen_seed(reordered) == seed
    assert stable_stock_qwen_seed(seed_task, seed_base=7) != seed
    assert (
        stable_stock_qwen_seed(replace(seed_task, question="A changed question"))
        != seed
    )

    processor = _FakeProcessor()
    engine = _FakeEngine()
    runner = StockQwenVLLMRunner(
        model_path=tmp_path / "model",
        processor=processor,
        engine=engine,
        batch_size=1,
    )
    second = _task(tmp_path, ordinal=1, sample_id="mmad-fixture-1")
    runner.run((task, second))
    assert len(engine.calls) == 2
    assert engine.calls[0][1][0]["seed"] == stable_stock_qwen_seed(task)
    assert engine.calls[1][1][0]["seed"] == stable_stock_qwen_seed(second)


def test_runner_rejects_multi_image_task_before_processor_or_engine(
    tmp_path: Path,
) -> None:
    first = _task(tmp_path)
    second_path = tmp_path / "second.png"
    _write_image(second_path, size=(41, 37), color=(1, 2, 3))
    second_digest = hashlib.sha256(second_path.read_bytes()).hexdigest()
    multi = replace(
        first,
        image_paths=(first.image_paths[0], str(second_path.resolve())),
        image_sha256s=(first.image_sha256s[0], second_digest),
        image_dimensions=(first.image_dimensions[0], (41, 37)),
    )
    processor = _FakeProcessor()
    engine = _FakeEngine()
    runner = StockQwenVLLMRunner(
        model_path=tmp_path / "model",
        processor=processor,
        engine=engine,
    )

    with pytest.raises(ValueError, match="exactly one image"):
        runner.run((multi,))

    assert processor.calls == []
    assert engine.calls == []


def test_runner_batches_in_input_order_and_emits_json_rows(tmp_path: Path) -> None:
    tasks = tuple(
        _task(tmp_path, ordinal=index, sample_id=f"last-fixture-{index}")
        for index in range(3)
    )
    engine = _FakeEngine()
    runner = StockQwenVLLMRunner(
        model_path=tmp_path / "model",
        processor=_FakeProcessor(),
        engine=engine,
        batch_size=2,
        max_tokens=73,
    )

    rows = runner.run(tasks)

    assert [row["ordinal"] for row in rows] == [0, 1, 2]
    assert [row["sample_id"] for row in rows] == [
        "last-fixture-0",
        "last-fixture-1",
        "last-fixture-2",
    ]
    assert [len(call[0]) for call in engine.calls] == [2, 1]
    assert all(
        values["max_tokens"] == 73 for call in engine.calls for values in call[1]
    )
    json.loads(json.dumps(rows, allow_nan=False))


def test_vllm_runner_owns_driver_portable_vit_attention_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="mm_encoder_attn_backend"):
        StockQwenVLLMRunner(
            model_path=tmp_path / "model",
            processor=_FakeProcessor(),
            engine_kwargs={"mm_encoder_attn_backend": "FLASH_ATTN"},
        )
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        StockQwenVLLMRunner(
            model_path=tmp_path / "model",
            processor=_FakeProcessor(),
            engine_kwargs={"tensor_parallel_size": 2},
        )

    observed: dict[str, object] = {}
    fake_vllm = ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        def generate(self, *args: object, **kwargs: object) -> list[object]:
            return []

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    lazy = stock_qwen._LazyVLLMEngine(  # noqa: SLF001
        model_path=tmp_path / "model",
        engine_kwargs={"gpu_memory_utilization": 0.5},
    )

    assert lazy.generate([], sampling_params=[], use_tqdm=False) == []
    assert (
        observed["mm_encoder_attn_backend"]
        == STOCK_QWEN_MM_ENCODER_ATTN_BACKEND
        == "TORCH_SDPA"
    )
    assert observed["tensor_parallel_size"] == 1
