"""Deterministic stock-Qwen vLLM inference for single-image texture tasks.

The original arm intentionally has no visual tools and no policy adapter.  It
uses the same immutable :class:`CoreDevTask` rows as the three tool arms and
lets Qwen's image processor enforce the pixel budget at request time.  Source
assets are decoded at their native dimensions; this module never pre-resizes
them.

Heavy optional dependencies are deliberately imported only when their
corresponding default implementation is first used.  Tests and callers may
therefore inject a processor and a synchronous engine without installing
``transformers`` or ``vllm`` and without allocating a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

from .schema import VisionPreprocessConfig, canonical_json_sha256

if TYPE_CHECKING:
    from tgvf_rl.evaluation.policy_coredev import CoreDevTask


STOCK_QWEN_RESULT_SCHEMA = "tgvf-texture-stock-qwen-result-v1"
STOCK_QWEN_REQUEST_SEED_SCHEMA = "tgvf-texture-stock-qwen-seed-v1"
STOCK_QWEN_VISION_IDENTITY_SCHEMA = "tgvf-texture-stock-qwen-vision-v1"
STOCK_QWEN_SEED_NAMESPACE = "texture-benchmark/original/stock-qwen-v1"
STOCK_QWEN_MM_ENCODER_ATTN_BACKEND = "TORCH_SDPA"
STOCK_QWEN_VISION = VisionPreprocessConfig(
    min_pixels=256 * 256,
    max_pixels=512 * 512,
    preserve_aspect_ratio=True,
    pre_resize_assets=False,
)
_LOW_31_BITS = (1 << 31) - 1


class StockQwenProcessor(Protocol):
    def apply_chat_template(self, messages: object, **kwargs: object) -> str: ...


class StockQwenEngine(Protocol):
    """Small synchronous boundary implemented by the lazy vLLM adapter/fakes."""

    def generate(
        self,
        prompts: Sequence[Mapping[str, object]],
        *,
        sampling_params: Sequence[Mapping[str, object]],
        use_tqdm: bool,
    ) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class PreparedStockQwenRequest:
    """One vLLM request plus the identities needed to audit its result."""

    prompt: Mapping[str, object]
    sampling_params: Mapping[str, object]
    request_seed: int
    vision_identity: Mapping[str, object]


def _task_sample_id(task: CoreDevTask) -> str:
    sample_id = getattr(task, "bound_sample_id", None)
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("stock Qwen task must carry a bound sample identity")
    return sample_id


def _require_single_bound_image(task: CoreDevTask) -> tuple[Path, str, tuple[int, int]]:
    image_paths = tuple(getattr(task, "image_paths", ()))
    if len(image_paths) != 1:
        raise ValueError("stock Qwen texture evaluation requires exactly one image")
    image_sha256s = tuple(getattr(task, "image_sha256s", ()))
    image_dimensions = tuple(getattr(task, "image_dimensions", ()))
    if len(image_sha256s) != 1 or len(image_dimensions) != 1:
        raise ValueError("stock Qwen task requires one bound image identity")
    path = Path(image_paths[0])
    if not path.is_absolute():
        raise ValueError("stock Qwen task image path must be absolute")
    digest = image_sha256s[0]
    dimensions = tuple(image_dimensions[0])
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("stock Qwen task image SHA256 is malformed")
    if len(dimensions) != 2 or any(
        type(value) is not int or value <= 0 for value in dimensions
    ):
        raise ValueError("stock Qwen task image dimensions are malformed")
    return path, digest, (int(dimensions[0]), int(dimensions[1]))


def build_qwen_chat_messages(task: CoreDevTask) -> list[dict[str, object]]:
    """Build the image-first Qwen user message for one manifest task."""

    path, _digest, _dimensions = _require_single_bound_image(task)
    question = getattr(task, "question", None)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("stock Qwen task question must be non-empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": question},
            ],
        }
    ]


def stable_stock_qwen_seed(
    task: CoreDevTask,
    *,
    seed_namespace: str = STOCK_QWEN_SEED_NAMESPACE,
    seed_base: int = 0,
) -> int:
    """Derive a low-31-bit seed from the exact model-visible sample content.

    Ordinals and manifest row positions are excluded, so re-sharding or
    reordering a manifest cannot perturb a sample's generation.
    """

    if (
        not isinstance(seed_namespace, str)
        or not seed_namespace
        or seed_namespace.strip() != seed_namespace
    ):
        raise ValueError("stock Qwen seed namespace must be canonical text")
    if type(seed_base) is not int or not 0 <= seed_base <= _LOW_31_BITS:
        raise ValueError("stock Qwen seed base must be a low-31-bit integer")
    _path, image_sha256, image_dimensions = _require_single_bound_image(task)
    payload = {
        "schema_version": STOCK_QWEN_REQUEST_SEED_SCHEMA,
        "seed_namespace": seed_namespace,
        "seed_base": seed_base,
        "sample_id": _task_sample_id(task),
        "dataset": getattr(task, "dataset", None),
        "index": getattr(task, "index", None),
        "question": getattr(task, "question", None),
        "image_sha256": image_sha256,
        "image_dimensions": list(image_dimensions),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big") & _LOW_31_BITS


def _load_native_rgb_image(
    task: CoreDevTask,
) -> tuple[object, dict[str, object]]:
    """Decode verified bytes into RGB while preserving native dimensions."""

    path, expected_sha256, expected_dimensions = _require_single_bound_image(task)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"stock Qwen task image is unreadable: {path}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"stock Qwen task image SHA256 changed for {_task_sample_id(task)}"
        )

    # Pillow is optional at module-import time.  Returning a detached copy also
    # closes the byte stream before a potentially long vLLM batch executes.
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as opened:
            actual_dimensions = (int(opened.width), int(opened.height))
            image = opened.convert("RGB").copy()
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("stock Qwen image decoding requires Pillow") from error
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"stock Qwen task image dimensions changed for {_task_sample_id(task)}"
        )
    return image, {
        "source_path": str(path),
        "source_image_sha256": actual_sha256,
        "source_dimensions": list(actual_dimensions),
    }


def build_stock_qwen_request(
    task: CoreDevTask,
    *,
    processor: StockQwenProcessor,
    vision: VisionPreprocessConfig = STOCK_QWEN_VISION,
    seed_namespace: str = STOCK_QWEN_SEED_NAMESPACE,
    seed_base: int = 0,
    max_tokens: int = 2048,
) -> PreparedStockQwenRequest:
    """Materialize one image-first Qwen prompt and native vLLM request."""

    if not isinstance(vision, VisionPreprocessConfig):
        raise TypeError("stock Qwen vision configuration has the wrong type")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("stock Qwen max_tokens must be positive")
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("stock Qwen processor must expose apply_chat_template")

    messages = build_qwen_chat_messages(task)
    prompt_text = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise RuntimeError("stock Qwen processor returned an invalid chat prompt")
    image, source_identity = _load_native_rgb_image(task)
    vision_content: dict[str, object] = {
        "schema_version": STOCK_QWEN_VISION_IDENTITY_SCHEMA,
        **source_identity,
        "preprocess": asdict(vision),
        "preprocess_identity_sha256": vision.identity_sha256,
    }
    vision_identity = {
        **vision_content,
        "identity_sha256": canonical_json_sha256(vision_content),
    }
    request_seed = stable_stock_qwen_seed(
        task,
        seed_namespace=seed_namespace,
        seed_base=seed_base,
    )
    return PreparedStockQwenRequest(
        prompt={
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "min_pixels": vision.min_pixels,
                "max_pixels": vision.max_pixels,
            },
        },
        sampling_params={
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": request_seed,
        },
        request_seed=request_seed,
        vision_identity=vision_identity,
    )


class _LazyVLLMEngine:
    """Adapt real vLLM to the dependency-free synchronous engine protocol."""

    def __init__(
        self, *, model_path: Path, engine_kwargs: Mapping[str, object]
    ) -> None:
        self.model_path = model_path
        self.engine_kwargs = dict(engine_kwargs)
        self._engine: object | None = None
        self._sampling_params_type: object | None = None

    def _ensure_loaded(self) -> tuple[object, object]:
        if self._engine is None:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as error:
                raise RuntimeError("stock Qwen execution requires vLLM") from error
            self._engine = LLM(
                model=str(self.model_path),
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1},
                tensor_parallel_size=1,
                # vLLM 0.12's bundled FlashAttention-2 PTX is not portable to
                # every Blackwell driver.  This dedicated multimodal override
                # affects only the ViT; decoder attention remains selected by
                # vLLM for the active GPU.
                mm_encoder_attn_backend=STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
                **self.engine_kwargs,
            )
            self._sampling_params_type = SamplingParams
        assert self._sampling_params_type is not None
        return self._engine, self._sampling_params_type

    def generate(
        self,
        prompts: Sequence[Mapping[str, object]],
        *,
        sampling_params: Sequence[Mapping[str, object]],
        use_tqdm: bool,
    ) -> Sequence[object]:
        engine, sampling_type = self._ensure_loaded()
        generate = getattr(engine, "generate", None)
        if not callable(generate) or not callable(sampling_type):
            raise RuntimeError("vLLM engine contract differs")
        concrete = [sampling_type(**dict(values)) for values in sampling_params]
        return generate(prompts, sampling_params=concrete, use_tqdm=use_tqdm)


def _load_default_processor(model_path: Path) -> StockQwenProcessor:
    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError("stock Qwen execution requires transformers") from error
    return AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)


def _response_payload(output: object) -> dict[str, object]:
    candidates = getattr(output, "outputs", None)
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) != 1
    ):
        raise RuntimeError("stock Qwen vLLM output must contain exactly one candidate")
    candidate = candidates[0]
    text = getattr(candidate, "text", None)
    if not isinstance(text, str):
        raise RuntimeError("stock Qwen vLLM output text is invalid")
    response: dict[str, object] = {"text": text}
    token_ids = getattr(candidate, "token_ids", None)
    if token_ids is not None:
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise RuntimeError("stock Qwen vLLM output token IDs are invalid")
        response["token_ids"] = [int(value) for value in token_ids]
    for name in ("finish_reason", "stop_reason"):
        value = getattr(candidate, name, None)
        if value is not None:
            response[name] = str(value)
    return response


class StockQwenVLLMRunner:
    """Execute deterministic stock-Qwen inference in synchronous batches."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        engine: StockQwenEngine | None = None,
        processor: StockQwenProcessor | None = None,
        vision: VisionPreprocessConfig = STOCK_QWEN_VISION,
        batch_size: int = 8,
        max_tokens: int = 2048,
        seed_namespace: str = STOCK_QWEN_SEED_NAMESPACE,
        seed_base: int = 0,
        engine_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(model_path, str) and not model_path.strip():
            raise ValueError("stock Qwen model_path must be non-empty")
        path = Path(model_path)
        if not isinstance(vision, VisionPreprocessConfig):
            raise TypeError("stock Qwen vision configuration has the wrong type")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("stock Qwen batch_size must be positive")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("stock Qwen max_tokens must be positive")
        # Validate seed options without requiring a task yet.
        if (
            not isinstance(seed_namespace, str)
            or not seed_namespace
            or seed_namespace.strip() != seed_namespace
        ):
            raise ValueError("stock Qwen seed namespace must be canonical text")
        if type(seed_base) is not int or not 0 <= seed_base <= _LOW_31_BITS:
            raise ValueError("stock Qwen seed base must be a low-31-bit integer")
        options = dict(engine_kwargs or {})
        forbidden = {
            "model",
            "trust_remote_code",
            "limit_mm_per_prompt",
            "mm_encoder_attn_backend",
            "tensor_parallel_size",
        } & set(options)
        if forbidden:
            raise ValueError(
                "stock Qwen engine kwargs override owned fields: "
                + ", ".join(sorted(forbidden))
            )
        self.model_path = path
        self.engine = (
            engine
            if engine is not None
            else _LazyVLLMEngine(model_path=path, engine_kwargs=options)
        )
        self.processor = processor
        self.vision = vision
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed_namespace = seed_namespace
        self.seed_base = seed_base

    def run(self, tasks: Sequence[CoreDevTask]) -> list[dict[str, object]]:
        ordered = tuple(tasks)
        if not ordered:
            raise ValueError("stock Qwen runner requires at least one task")
        # Reject multi-image rows before loading optional dependencies or images.
        for task in ordered:
            _require_single_bound_image(task)
        sample_ids = tuple(_task_sample_id(task) for task in ordered)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("stock Qwen runner received duplicate sample identities")

        processor = self.processor
        if processor is None:
            processor = _load_default_processor(self.model_path)
            self.processor = processor

        rows: list[dict[str, object]] = []
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start : start + self.batch_size]
            prepared = tuple(
                build_stock_qwen_request(
                    task,
                    processor=processor,
                    vision=self.vision,
                    seed_namespace=self.seed_namespace,
                    seed_base=self.seed_base,
                    max_tokens=self.max_tokens,
                )
                for task in batch
            )
            outputs = self.engine.generate(
                [item.prompt for item in prepared],
                sampling_params=[item.sampling_params for item in prepared],
                use_tqdm=False,
            )
            if (
                not isinstance(outputs, Sequence)
                or isinstance(outputs, (str, bytes))
                or len(outputs) != len(batch)
            ):
                raise RuntimeError(
                    "stock Qwen vLLM output count differs from request count"
                )
            for task, request, output in zip(batch, prepared, outputs, strict=True):
                response = _response_payload(output)
                rows.append(
                    {
                        "schema_version": STOCK_QWEN_RESULT_SCHEMA,
                        "ordinal": int(getattr(task, "ordinal")),
                        "sample_id": _task_sample_id(task),
                        "index": str(getattr(task, "index")),
                        "dataset": str(getattr(task, "dataset")),
                        "final_answer": response["text"],
                        "model_response": response,
                        "vision_identity": dict(request.vision_identity),
                        "request_seed": request.request_seed,
                    }
                )
        # Enforce the advertised JSON-compatible artifact boundary now, rather
        # than after a long model run has been written to disk.
        json.dumps(rows, ensure_ascii=False, allow_nan=False)
        return rows


__all__ = [
    "PreparedStockQwenRequest",
    "STOCK_QWEN_REQUEST_SEED_SCHEMA",
    "STOCK_QWEN_RESULT_SCHEMA",
    "STOCK_QWEN_SEED_NAMESPACE",
    "STOCK_QWEN_MM_ENCODER_ATTN_BACKEND",
    "STOCK_QWEN_VISION",
    "STOCK_QWEN_VISION_IDENTITY_SCHEMA",
    "StockQwenEngine",
    "StockQwenProcessor",
    "StockQwenVLLMRunner",
    "build_qwen_chat_messages",
    "build_stock_qwen_request",
    "stable_stock_qwen_seed",
]
