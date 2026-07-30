"""Qwen3/vLLM execution boundary for the accepted Policy-RL T1 canary.

The module is safe to import in CPU-only tooling.  PIL, Transformers, Torch,
and vLLM are imported only inside the image/prompt/worker paths that need them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

from .policy_selection import (
    SelectionBranch,
    SelectionCandidate,
    stable_selection_request_id,
)
from .policy_selection_runtime import (
    T1_ATTEMPTS,
    T1_MODEL_PATH_BY_REPOSITORY,
    T1_RAW_GENERATION_SCHEMA,
    T1RawGenerationEvidence,
    T1RunConfig,
    candidate_rank,
    load_resumable_chunk,
    load_t1_run_config,
    native_prompt_identity_sha256,
    native_user_message_descriptor,
    rendered_prompt_token_ids_sha256,
    sampled_token_ids_sha256,
    source_rgb_sha256,
    validate_chunk_manifest,
    write_content_addressed_chunk,
)


T1_OUTPUT_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-output-root.v1"
T1_MM_UUID_SCHEMA = "tgvf.policy-selection.t1-mm-cache-item.v1"
T1_BUDGET_CHUNK_STRIDE = 1_000_000
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_PROMPT_SUFFIX_BY_REPOSITORY = {
    "Qwen/Qwen3-VL-8B-Thinking": "<|im_start|>assistant\n<think>\n",
    "Qwen/Qwen3-VL-8B-Instruct": "<|im_start|>assistant\n",
}
if set(_EXPECTED_PROMPT_SUFFIX_BY_REPOSITORY) != set(T1_MODEL_PATH_BY_REPOSITORY):
    raise RuntimeError("T1 model and native prompt-suffix allowlists differ")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def qwen_smart_resize_dimensions(
    *,
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Mirror the Qwen smart-resize geometry without resizing any pixels."""

    for name, value in (
        ("height", height),
        ("width", width),
        ("factor", factor),
        ("min_pixels", min_pixels),
        ("max_pixels", max_pixels),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if min_pixels > max_pixels:
        raise ValueError("min_pixels must not exceed max_pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute image aspect ratio must not exceed 200")

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def load_t1_candidates(run: T1RunConfig) -> tuple[SelectionCandidate, ...]:
    """Load the already identity-checked ordered canary without normalization."""

    path = Path(str(run.selection["candidates_path"]))
    candidates: list[SelectionCandidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid selection candidate JSON at line {line_number}"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"selection candidate line {line_number} is not an object"
                )
            candidates.append(SelectionCandidate.from_record(value))
    if len(candidates) != run.selection["rows"]:
        raise ValueError("selection candidate count differs from the run identity")
    if len({candidate.identity_sha256 for candidate in candidates}) != len(candidates):
        raise ValueError("selection contains duplicate candidate identities")
    return tuple(candidates)


def rank_candidate_chunks(
    candidates: Sequence[SelectionCandidate],
    *,
    rank: int,
    world_size: int,
    chunk_candidates: int,
) -> tuple[tuple[SelectionCandidate, ...], ...]:
    if (
        type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("rank must be inside world_size")
    if type(chunk_candidates) is not int or chunk_candidates <= 0:
        raise ValueError("chunk_candidates must be positive")
    selected = tuple(
        candidate
        for candidate in candidates
        if candidate_rank(candidate.identity_sha256, world_size=world_size) == rank
    )
    return tuple(
        selected[index : index + chunk_candidates]
        for index in range(0, len(selected), chunk_candidates)
    )


def budget_chunk_index(*, budget_revision: int, local_chunk_index: int) -> int:
    if type(budget_revision) is not int or not 0 <= budget_revision <= 2:
        raise ValueError("budget_revision must be in [0, 2]")
    if type(local_chunk_index) is not int or local_chunk_index < 0:
        raise ValueError("local_chunk_index must be non-negative")
    return budget_revision * T1_BUDGET_CHUNK_STRIDE + local_chunk_index


def chunk_subshard_owns(
    local_chunk_index: int, *, subshard_count: int, subshard_index: int
) -> bool:
    """Return whether one physical worker owns an original logical chunk."""

    if type(local_chunk_index) is not int or local_chunk_index < 0:
        raise ValueError("local_chunk_index must be non-negative")
    if type(subshard_count) is not int or subshard_count <= 0:
        raise ValueError("subshard_count must be a positive integer")
    if (
        type(subshard_index) is not int
        or not 0 <= subshard_index < subshard_count
    ):
        raise ValueError("subshard_index must be inside subshard_count")
    return local_chunk_index % subshard_count == subshard_index


@dataclass(frozen=True, slots=True)
class PreparedT1Image:
    rgb: Any
    source_width: int
    source_height: int
    source_mode: str
    source_rgb_sha256: str
    processed_width: int
    processed_height: int
    mm_uuid: str

    @property
    def evidence(self) -> dict[str, object]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_mode": self.source_mode,
            "source_rgb_sha256": self.source_rgb_sha256,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
        }


@dataclass(frozen=True, slots=True)
class PreparedT1Prompt:
    candidate: SelectionCandidate
    rendered_text: str
    prompt_token_ids: tuple[int, ...]
    image: PreparedT1Image


def _mm_uuid(run: T1RunConfig, candidate: SelectionCandidate) -> str:
    value = {
        "schema": T1_MM_UUID_SCHEMA,
        "image_sha256": candidate.image["sha256"],
        "processor_identity_sha256": run.processor_identity_sha256,
        "source_color_mode": run.image["color_mode"],
        "alpha_handling": run.image["alpha_handling"],
        "min_pixels": run.image["min_pixels"],
        "max_pixels": run.image["max_pixels"],
    }
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def load_candidate_rgb(
    candidate: SelectionCandidate, *, run: T1RunConfig
) -> PreparedT1Image:
    """Verify encoded identity, decode once, then convert RGB without resize."""

    from PIL import Image

    raw_path = candidate.image.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("candidate image.path is required")
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"candidate image must be an absolute regular file: {path}")
    try:
        path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError("candidate image must remain inside the repository") from exc
    if _sha256_file(path) != candidate.image["sha256"]:
        raise ValueError(f"candidate image SHA-256 mismatch: {path}")

    with Image.open(path) as opened:
        source_mode = opened.mode
        opened.load()
        if opened.size != (candidate.image["width"], candidate.image["height"]):
            raise ValueError(f"candidate image dimensions differ: {path}")
        rgb = opened.convert("RGB").copy()
    processed_height, processed_width = qwen_smart_resize_dimensions(
        height=rgb.height,
        width=rgb.width,
        factor=int(run.image["resize_factor"]),
        min_pixels=int(run.image["min_pixels"]),
        max_pixels=int(run.image["max_pixels"]),
    )
    return PreparedT1Image(
        rgb=rgb,
        source_width=rgb.width,
        source_height=rgb.height,
        source_mode=source_mode,
        source_rgb_sha256=source_rgb_sha256(
            width=rgb.width, height=rgb.height, pixel_bytes=rgb.tobytes()
        ),
        processed_width=processed_width,
        processed_height=processed_height,
        mm_uuid=_mm_uuid(run, candidate),
    )


def prepare_candidate_prompt(
    candidate: SelectionCandidate, *, run: T1RunConfig, processor: Any
) -> PreparedT1Prompt:
    """Render and independently expand one prompt for exact runtime evidence."""

    image = load_candidate_rgb(candidate, run=run)
    image_path = str(candidate.image["path"])
    messages = native_user_message_descriptor(
        image=image_path, question=candidate.question
    )
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_repository = str(run.model["repository"])
    expected_prompt_suffix = _EXPECTED_PROMPT_SUFFIX_BY_REPOSITORY.get(
        model_repository
    )
    if expected_prompt_suffix is None:
        image.rgb.close()
        raise ValueError("T1 model has no accepted native prompt suffix")
    if not isinstance(rendered, str) or not rendered.endswith(expected_prompt_suffix):
        image.rgb.close()
        raise ValueError(
            "native Qwen prompt prefill differs from the frozen model-edition contract"
        )
    processed = processor(
        text=[rendered],
        images=[image.rgb],
        padding=False,
        return_tensors="pt",
        do_resize=True,
        min_pixels=int(run.image["min_pixels"]),
        max_pixels=int(run.image["max_pixels"]),
    )
    prompt_ids = tuple(int(value) for value in processed["input_ids"][0].tolist())
    grid = tuple(int(value) for value in processed["image_grid_thw"][0].tolist())
    patch_size = int(processor.image_processor.patch_size)
    actual_height = grid[1] * patch_size
    actual_width = grid[2] * patch_size
    if (actual_width, actual_height) != (
        image.processed_width,
        image.processed_height,
    ):
        image.rgb.close()
        raise ValueError("Qwen processor dimensions differ from smart-resize evidence")
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge_size = int(processor.image_processor.merge_size)
    expected_visual_tokens = grid[0] * grid[1] * grid[2] // (merge_size**2)
    if prompt_ids.count(image_pad_id) != expected_visual_tokens:
        image.rgb.close()
        raise ValueError("expanded prompt visual-token count differs")
    return PreparedT1Prompt(candidate, rendered, prompt_ids, image)


def prepare_output_root(config_path: str | Path) -> dict[str, object]:
    """Create or verify the immutable CPU run identity before model launch."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    identity: dict[str, object] = {
        "schema_version": T1_OUTPUT_IDENTITY_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "config_path": str(path),
        "config_file_sha256": _sha256_file(path),
        "model_identity_sha256": run.model_identity_sha256,
        "processor_identity_sha256": run.processor_identity_sha256,
        "runtime_identity_sha256": run.runtime_identity_sha256,
        "selection_candidates_sha256": run.selection["candidates_sha256"],
        "selection_rows": run.selection["rows"],
    }
    payload = _canonical_json_bytes(identity) + b"\n"
    root = run.output_root
    identity_path = root / "run-identity.json"
    if root.exists() and not identity_path.is_file():
        raise ValueError("existing output root has no matching immutable identity")
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("chunks", "manifests", "logs", "runtime/cache"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _atomic_write_immutable(identity_path, payload)
    _atomic_write_immutable(
        root / "run-config.canonical.json",
        _canonical_json_bytes(run.as_record()) + b"\n",
    )
    return identity


def _validate_prepared_output_root(run: T1RunConfig, config_path: Path) -> None:
    identity_path = run.output_root / "run-identity.json"
    if not identity_path.is_file():
        raise ValueError("run output root is not prepared")
    expected = {
        "schema_version": T1_OUTPUT_IDENTITY_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "config_path": str(config_path.resolve()),
        "config_file_sha256": _sha256_file(config_path.resolve()),
        "model_identity_sha256": run.model_identity_sha256,
        "processor_identity_sha256": run.processor_identity_sha256,
        "runtime_identity_sha256": run.runtime_identity_sha256,
        "selection_candidates_sha256": run.selection["candidates_sha256"],
        "selection_rows": run.selection["rows"],
    }
    if json.loads(identity_path.read_text(encoding="utf-8")) != expected:
        raise ValueError("prepared output-root identity differs from the run")


def _validate_runtime_versions(run: T1RunConfig) -> None:
    import torch

    actual = {
        "python": platform.python_version(),
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "pillow": importlib.metadata.version("pillow"),
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
    expected = {
        "python": run.runtime["python"],
        "vllm": run.runtime["version"],
        "torch": run.runtime["torch"],
        "transformers": run.runtime["transformers"],
        "pillow": run.runtime["pillow"],
        "flashinfer": run.runtime["flashinfer"],
    }
    if actual != expected:
        raise ValueError(f"runtime package identity differs: {actual} != {expected}")


def _raw_evidence_record(
    *,
    run: T1RunConfig,
    prepared: PreparedT1Prompt,
    attempt_index: int,
    budget_revision: int,
    token_ids: Sequence[int],
    raw_text: str,
    finish_reason: str,
    stop_reason: str | int | None,
    generation_error: str | None = None,
) -> dict[str, object]:
    candidate = prepared.candidate
    budget = run.budget(budget_revision)
    normalized_ids = [int(value) for value in token_ids]
    record: dict[str, object] = {
        "schema_version": T1_RAW_GENERATION_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "request_id": stable_selection_request_id(
            candidate_sha256=candidate.identity_sha256,
            branch=SelectionBranch.FULL_IMAGE,
            attempt_index=attempt_index,
        ),
        "sample_id": candidate.sample_id,
        "candidate_sha256": candidate.identity_sha256,
        "source": candidate.source.value,
        "branch": SelectionBranch.FULL_IMAGE.value,
        "attempt_index": attempt_index,
        "attempt_seed": run.attempt_seed(
            candidate_sha256=candidate.identity_sha256,
            attempt_index=attempt_index,
        ),
        "budget_revision": budget_revision,
        "max_model_len": budget.max_model_len,
        "max_new_tokens": budget.max_new_tokens,
        "prompt_sha256": native_prompt_identity_sha256(
            question=candidate.question,
            image_sha256=str(candidate.image["sha256"]),
            chat_template_sha256=str(run.model["chat_template_sha256"]),
        ),
        "rendered_prompt_token_ids_sha256": rendered_prompt_token_ids_sha256(
            prepared.prompt_token_ids
        ),
        "prompt_token_count": len(prepared.prompt_token_ids),
        "image_sha256": candidate.image["sha256"],
        "image_evidence": prepared.image.evidence,
        "sampled_token_ids_sha256": sampled_token_ids_sha256(normalized_ids),
        "sampled_token_count": len(normalized_ids),
        "sampled_token_ids": normalized_ids,
        "raw_text": raw_text,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "backend": {
            "name": run.runtime["backend"],
            "version": run.runtime["version"],
            "runtime_sha256": run.runtime_identity_sha256,
            "model_sha256": run.model_identity_sha256,
            "processor_sha256": run.processor_identity_sha256,
        },
    }
    if generation_error is not None:
        record["generation_error"] = generation_error
    return record


async def _generate_one(
    *,
    engine: Any,
    sampling_params_type: Any,
    output_kind: Any,
    run: T1RunConfig,
    prepared: PreparedT1Prompt,
    attempt_index: int,
    budget_revision: int,
) -> T1RawGenerationEvidence:
    candidate = prepared.candidate
    budget = run.budget(budget_revision)
    if len(prepared.prompt_token_ids) + budget.max_new_tokens > budget.max_model_len:
        raise ValueError("prompt plus response budget exceeds max_model_len")
    seed = run.attempt_seed(
        candidate_sha256=candidate.identity_sha256,
        attempt_index=attempt_index,
    )
    raw_request_id = stable_selection_request_id(
        candidate_sha256=candidate.identity_sha256,
        branch=SelectionBranch.FULL_IMAGE,
        attempt_index=attempt_index,
    )
    backend_request_id = f"{raw_request_id}:budget-{budget_revision}"
    prompt = {
        "prompt": prepared.rendered_text,
        "multi_modal_data": {"image": [prepared.image.rgb]},
        "multi_modal_uuids": {"image": [prepared.image.mm_uuid]},
        "mm_processor_kwargs": {
            "min_pixels": int(run.image["min_pixels"]),
            "max_pixels": int(run.image["max_pixels"]),
            "do_resize": True,
        },
    }
    sampling = sampling_params_type(
        n=1,
        temperature=float(run.sampling["temperature"]),
        top_p=float(run.sampling["top_p"]),
        top_k=int(run.sampling["top_k"]),
        min_p=float(run.sampling["min_p"]),
        repetition_penalty=float(run.sampling["repetition_penalty"]),
        presence_penalty=float(run.sampling["presence_penalty"]),
        frequency_penalty=float(run.sampling["frequency_penalty"]),
        seed=seed,
        stop=list(run.sampling["stop_strings"]),
        stop_token_ids=list(run.sampling["stop_token_ids"]),
        ignore_eos=False,
        max_tokens=budget.max_new_tokens,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        include_stop_str_in_output=False,
        truncate_prompt_tokens=None,
        output_kind=output_kind,
    )
    final = None
    async for output in engine.generate(prompt, sampling, backend_request_id):
        final = output
    if final is None or not final.finished or len(final.outputs) != 1:
        raise RuntimeError("vLLM request did not finish exactly once")
    completion = final.outputs[0]
    if int(completion.index) != 0:
        raise RuntimeError("vLLM completion index differs")
    prompt_ids = tuple(int(value) for value in (final.prompt_token_ids or ()))
    if prompt_ids != prepared.prompt_token_ids:
        raise RuntimeError("vLLM expanded prompt token IDs differ from preflight")
    finish_reason = str(completion.finish_reason)
    if finish_reason not in {"stop", "length"}:
        raise RuntimeError(f"unsupported vLLM finish reason: {finish_reason}")
    stop_reason = completion.stop_reason
    if stop_reason is not None and type(stop_reason) not in {str, int}:
        stop_reason = str(stop_reason)
    record = _raw_evidence_record(
        run=run,
        prepared=prepared,
        attempt_index=attempt_index,
        budget_revision=budget_revision,
        token_ids=completion.token_ids,
        raw_text=completion.text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )
    evidence = T1RawGenerationEvidence.from_record(record)
    evidence.validate_against_run(run)
    return evidence


async def run_t1_worker(
    config_path: str | Path,
    *,
    rank: int,
    cuda_visible_device: int | None = None,
    budget_revision: int = 0,
    max_chunks: int | None = None,
    chunk_subshard_count: int = 1,
    chunk_subshard_index: int = 0,
) -> dict[str, object]:
    """Run one stable single-GPU shard and publish immutable chunks."""

    if budget_revision != 0:
        raise ValueError(
            "later budgets require the length-only T1 retry worker"
        )
    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    _validate_runtime_versions(run)
    world_size = int(run.runtime["world_size"])
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be inside the configured world size")
    if cuda_visible_device is not None and (
        type(cuda_visible_device) is not int or cuda_visible_device < 0
    ):
        raise ValueError("cuda_visible_device must be a non-negative integer")
    expected_visible_device = (
        rank if cuda_visible_device is None else cuda_visible_device
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(expected_visible_device):
        raise ValueError(
            f"worker rank {rank} requires "
            f"CUDA_VISIBLE_DEVICES={expected_visible_device}, got {visible!r}"
        )
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks <= 0):
        raise ValueError("max_chunks must be positive when provided")
    if type(chunk_subshard_count) is not int or chunk_subshard_count <= 0:
        raise ValueError("chunk_subshard_count must be a positive integer")
    if (
        type(chunk_subshard_index) is not int
        or not 0 <= chunk_subshard_index < chunk_subshard_count
    ):
        raise ValueError(
            "chunk_subshard_index must be inside chunk_subshard_count"
        )

    candidates = load_t1_candidates(run)
    chunks = rank_candidate_chunks(
        candidates,
        rank=rank,
        world_size=world_size,
        chunk_candidates=int(run.runtime["chunk_candidates"]),
    )
    indexed_chunks = tuple(enumerate(chunks))
    if max_chunks is not None:
        # Keep the historical meaning of max_chunks: it bounds the original
        # logical-rank prefix before any physical-worker subdivision.
        indexed_chunks = indexed_chunks[:max_chunks]
    pending: list[tuple[int, tuple[SelectionCandidate, ...]]] = []
    resumed_records = 0
    for local_index, chunk in indexed_chunks:
        # Physical workers may subdivide one logical rank, but the original
        # local index remains the immutable manifest chunk index.  Filtering
        # before re-enumeration is essential for byte-identical resume/audit.
        if not chunk_subshard_owns(
            local_index,
            subshard_count=chunk_subshard_count,
            subshard_index=chunk_subshard_index,
        ):
            continue
        chunk_index = budget_chunk_index(
            budget_revision=budget_revision, local_chunk_index=local_index
        )
        manifest_path = (
            run.output_root
            / "manifests"
            / f"rank-{rank:02d}-chunk-{chunk_index:06d}.json"
        )
        existing = load_resumable_chunk(
            manifest_path,
            output_root=run.output_root,
            run=run,
            expected_rank=rank,
            expected_chunk_index=chunk_index,
        )
        if existing is None:
            pending.append((chunk_index, chunk))
        else:
            resumed_records += existing.record_count
    if not pending:
        return {
            "run_id": run.run_id,
            "rank": rank,
            "budget_revision": budget_revision,
            "chunks_written": 0,
            "records_written": 0,
            "records_resumed": resumed_records,
        }

    from transformers import AutoProcessor
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    processor = AutoProcessor.from_pretrained(
        str(run.model["path"]),
        trust_remote_code=True,
        min_pixels=int(run.image["min_pixels"]),
        max_pixels=int(run.image["max_pixels"]),
        use_fast=True,
    )
    if len(processor.tokenizer) != run.model["tokenizer_length"]:
        raise ValueError("worker tokenizer length differs from the run identity")
    budget = run.budget(budget_revision)
    engine_args = AsyncEngineArgs(
        model=str(run.model["path"]),
        dtype=str(run.model["dtype"]),
        trust_remote_code=True,
        quantization=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        max_model_len=budget.max_model_len,
        max_num_seqs=int(run.runtime["max_num_seqs"]),
        max_num_batched_tokens=int(run.runtime["max_num_batched_tokens"]),
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=float(run.runtime["gpu_memory_utilization"]),
        seed=int(run.runtime["engine_seed"]),
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={
            "min_pixels": int(run.image["min_pixels"]),
            "max_pixels": int(run.image["max_pixels"]),
            "do_resize": True,
        },
        mm_processor_cache_gb=float(run.runtime["mm_processor_cache_gb"]),
        mm_encoder_attn_backend=str(run.runtime["mm_encoder_attn_backend"]),
        generation_config=str(run.runtime["generation_config_mode"]),
        enforce_eager=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    chunks_written = 0
    records_written = 0
    try:
        for chunk_index, chunk in pending:
            prepared_items: list[PreparedT1Prompt] = []
            try:
                prepared_items = [
                    prepare_candidate_prompt(candidate, run=run, processor=processor)
                    for candidate in chunk
                ]
                tasks = [
                    asyncio.create_task(
                        _generate_one(
                            engine=engine,
                            sampling_params_type=SamplingParams,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            run=run,
                            prepared=prepared,
                            attempt_index=attempt_index,
                            budget_revision=budget_revision,
                        )
                    )
                    for prepared in prepared_items
                    for attempt_index in range(T1_ATTEMPTS)
                ]
                records = await asyncio.gather(*tasks)
                manifest = write_content_addressed_chunk(
                    run.output_root,
                    records,
                    run=run,
                    shard_rank=rank,
                    chunk_index=chunk_index,
                )
                chunks_written += 1
                records_written += manifest.record_count
                print(
                    json.dumps(
                        {
                            "event": "chunk_committed",
                            "rank": rank,
                            "chunk_index": chunk_index,
                            "records": manifest.record_count,
                            "manifest_sha256": manifest.manifest_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                for prepared in prepared_items:
                    prepared.image.rgb.close()
    finally:
        engine.shutdown()
    return {
        "run_id": run.run_id,
        "rank": rank,
        "budget_revision": budget_revision,
        "chunks_written": chunks_written,
        "records_written": records_written,
        "records_resumed": resumed_records,
    }


def t1_status(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    candidates = load_t1_candidates(run)
    rank_rows: list[dict[str, object]] = []
    total_records = 0
    for rank in range(int(run.runtime["world_size"])):
        chunks = rank_candidate_chunks(
            candidates,
            rank=rank,
            world_size=int(run.runtime["world_size"]),
            chunk_candidates=int(run.runtime["chunk_candidates"]),
        )
        complete = 0
        records = 0
        for manifest_path in sorted(
            (run.output_root / "manifests").glob(f"rank-{rank:02d}-chunk-*.json")
        ):
            manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
            parsed = validate_chunk_manifest(
                manifest_record, output_root=run.output_root, run=run
            )
            complete += 1
            records += parsed.record_count
        total_records += records
        rank_rows.append(
            {
                "rank": rank,
                "candidate_count": sum(len(chunk) for chunk in chunks),
                "revision0_expected_chunks": len(chunks),
                "all_revision_complete_manifests": complete,
                "records": records,
            }
        )
    return {
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "selection_rows": len(candidates),
        "logical_revision0_attempts": len(candidates) * T1_ATTEMPTS,
        "records_present": total_records,
        "ranks": rank_rows,
    }


__all__ = [
    "PreparedT1Image",
    "PreparedT1Prompt",
    "T1_BUDGET_CHUNK_STRIDE",
    "T1_OUTPUT_IDENTITY_SCHEMA",
    "budget_chunk_index",
    "chunk_subshard_owns",
    "load_candidate_rgb",
    "load_t1_candidates",
    "prepare_candidate_prompt",
    "prepare_output_root",
    "qwen_smart_resize_dimensions",
    "rank_candidate_chunks",
    "run_t1_worker",
    "t1_status",
]
