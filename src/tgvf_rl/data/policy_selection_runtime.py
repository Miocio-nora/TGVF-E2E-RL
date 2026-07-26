"""CPU-only runtime contracts for Qwen3 Policy-RL T1 scoring.

The module owns identities, validation, deterministic routing, evidence, and
resume metadata.  It deliberately imports neither Torch nor a model runtime.
GPU orchestration is expected to consume these contracts rather than recreate
their semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_ATTEMPT_SCHEMA,
    AttemptStatus,
    SelectionBranch,
    SelectionCandidate,
    SelectionSource,
    stable_selection_request_id,
)
from .policy_selection_recommended import (
    T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION,
    T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA,
    T1_RECOMMENDED_SELECTION_NAMESPACE,
    T1_RECOMMENDED_SELECTION_ROWS,
    T1_RECOMMENDED_SOURCE_QUOTAS,
)


T1_RUN_CONFIG_SCHEMA = "tgvf.policy-selection.t1-run.v1"
T1_PROMPT_SCHEMA = "qwen-native-user-image-question-v1"
T1_RAW_GENERATION_SCHEMA = "tgvf.policy-selection.t1-raw-generation.v1"
T1_CHUNK_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-chunk-manifest.v1"
T1_ATTEMPT_SEED_SCHEMA = "tgvf.policy-selection.t1-attempt-seed.v1"
T1_TOKEN_IDS_SCHEMA = "tgvf.policy-selection.sampled-token-ids.v1"
T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA = (
    "tgvf.policy-selection.rendered-prompt-token-ids.v1"
)
T1_SOURCE_RGB_SCHEMA = "tgvf.policy-selection.source-rgb-pixels.v1"
T1_PROMPT_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-native-prompt.v1"
T1_MODEL_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-model-identity.v1"
T1_PROCESSOR_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-processor-identity.v1"
T1_RUNTIME_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-runtime-identity.v1"
T1_THINKING_ANSWER_PARSER = "last-think-suffix-v1"
T1_INSTRUCT_ANSWER_PARSER = "direct-completion-v1"

T1_ATTEMPTS = 8
T1_SHARD_COUNT = 4
T1_MAX_PIXELS = 512 * 512
T1_SEED_MODULUS = 2**31 - 1
T1_MODEL_PATH_BY_REPOSITORY = MappingProxyType(
    {
        "Qwen/Qwen3-VL-8B-Thinking": (
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
        ),
        "Qwen/Qwen3-VL-8B-Instruct": (
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
        ),
    }
)

_RUN_FIELDS = {
    "schema_version",
    "run_id",
    "model",
    "prompt",
    "image",
    "sampling",
    "response_budgets",
    "runtime",
    "data",
    "selection",
    "verifier",
    "output_root",
}
_MODEL_FIELDS = {
    "repository",
    "path",
    "dtype",
    "config_sha256",
    "generation_config_sha256",
    "tokenizer_config_sha256",
    "tokenizer_json_sha256",
    "preprocessor_config_sha256",
    "chat_template_sha256",
    "tokenizer_length",
    "eos_token_id",
    "generation_eos_token_ids",
    "trust_remote_code",
    "quantization",
}
_PROMPT_FIELDS = {
    "schema",
    "user_content_order",
    "no_system",
    "no_tools",
    "add_generation_prompt",
}
_IMAGE_FIELDS = {
    "min_pixels",
    "max_pixels",
    "resize_factor",
    "resample",
    "pre_resize",
    "processor_do_resize",
    "preserve_aspect_ratio",
    "limit_image_per_prompt",
    "color_mode",
    "alpha_handling",
    "source_pixel_hash_schema",
}
_SAMPLING_FIELDS = {
    "attempts",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "logit_processors",
    "seed_root",
    "seed_namespace",
    "do_sample",
    "ignore_eos",
    "stop_token_ids",
    "effective_stop_token_ids",
    "stop_strings",
    "include_stop_str_in_output",
    "detokenize",
    "skip_special_tokens",
    "spaces_between_special_tokens",
}
_BUDGET_FIELDS = {"revision", "max_model_len", "max_new_tokens"}
_RUNTIME_FIELDS = {
    "backend",
    "version",
    "python",
    "torch",
    "transformers",
    "pillow",
    "flashinfer",
    "world_size",
    "tensor_parallel_size",
    "max_num_seqs",
    "gpu_memory_utilization",
    "mm_encoder_attn_backend",
    "decoder_attn_backend",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "mm_processor_cache_gb",
    "engine_seed",
    "chunk_candidates",
    "max_inflight",
    "retain_token_ids",
    "generation_config_mode",
}
_DATA_FIELDS = {"sources"}
_SOURCE_FIELDS = {"source", "path", "sha256", "rows"}
_SELECTION_FIELDS = {
    "kind",
    "algorithm_version",
    "candidates_path",
    "candidates_sha256",
    "rows",
    "manifest_path",
    "manifest_sha256",
}
_VERIFIER_FIELDS = {
    "schema",
    "answer_parser",
    "arxivqa_rule",
    "thinklite_rule",
    "vstar_rule",
    "semantic_judge",
}
_SEMANTIC_JUDGE_FIELDS = {
    "provider",
    "repository",
    "path",
    "served_name",
    "prompt_sha256",
    "config_sha256",
    "temperature",
    "max_tokens",
    "remote",
}
_RAW_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "run_manifest_sha256",
    "request_id",
    "sample_id",
    "candidate_sha256",
    "source",
    "branch",
    "attempt_index",
    "attempt_seed",
    "budget_revision",
    "max_model_len",
    "max_new_tokens",
    "prompt_sha256",
    "rendered_prompt_token_ids_sha256",
    "prompt_token_count",
    "image_sha256",
    "image_evidence",
    "sampled_token_ids_sha256",
    "sampled_token_count",
    "raw_text",
    "finish_reason",
    "stop_reason",
    "backend",
}
_RAW_OPTIONAL_FIELDS = {"sampled_token_ids", "generation_error"}
_IMAGE_EVIDENCE_FIELDS = {
    "source_width",
    "source_height",
    "source_mode",
    "source_rgb_sha256",
    "processed_width",
    "processed_height",
}
_BACKEND_EVIDENCE_FIELDS = {
    "name",
    "version",
    "runtime_sha256",
    "model_sha256",
    "processor_sha256",
}
_CHUNK_FIELDS = {
    "schema_version",
    "run_manifest_sha256",
    "shard_rank",
    "world_size",
    "chunk_index",
    "record_count",
    "evidence_file",
    "evidence_sha256",
    "logical_keys_sha256",
    "manifest_sha256",
}

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QWEN_TERMINAL_TEXT = re.compile(r"(?:<\|(?:im_end|endoftext)\|>\s*)+$")
_ANSWER_TAG = re.compile(r"^<answer>\s*(.*?)\s*</answer>$", re.DOTALL)
_BOXED = re.compile(r"^\\boxed\s*\{(.*)\}$", re.DOTALL)
_LATEX_FRACTION = re.compile(r"^\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}$")
_ARXIV_CANONICAL = re.compile(
    r"^\s*(?:[\(\[]\s*([A-Z])\s*[\)\]]|([A-Z])\s*[\).:]?|"
    r"(?:option|choice)\s*[\(\[]?\s*([A-Z])\s*[\)\]]?)\s*$",
    re.IGNORECASE,
)
_ARXIV_ANSWER_MARKER = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is|:|=|-)\s*"
    r"(?:(?:option|choice)\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)
_ARXIV_NAMED_OPTION = re.compile(
    r"\b(?:option|choice)\s*(?:is\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc
    return encoded.encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {constant}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field_name} fields differ; missing={missing}, unknown={unknown}"
        )


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, stripped string")
    return value


def _required_sha256(value: Any, *, field_name: str) -> str:
    value = _required_string(value, field_name=field_name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_int(
    value: Any, *, field_name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def _required_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _required_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def _absolute_normal_path(value: Any, *, field_name: str) -> Path:
    raw = _required_string(value, field_name=field_name)
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        raise ValueError(f"{field_name} must be an absolute normalized path")
    return path


def _safe_relative_path(value: Any, *, field_name: str) -> Path:
    raw = _required_string(value, field_name=field_name)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or os.path.normpath(raw) != raw:
        raise ValueError(f"{field_name} must be a safe normalized relative path")
    return path


def _json_clone(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class T1ResponseBudget:
    revision: int
    max_model_len: int
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class T1DataSource:
    source: SelectionSource
    path: Path
    sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class T1RunConfig:
    """Validated immutable identity of one T1 generation run."""

    run_id: str
    manifest_sha256: str
    model: Mapping[str, Any]
    prompt: Mapping[str, Any]
    image: Mapping[str, Any]
    sampling: Mapping[str, Any]
    response_budgets: tuple[T1ResponseBudget, ...]
    runtime: Mapping[str, Any]
    data_sources: tuple[T1DataSource, ...]
    selection: Mapping[str, Any]
    verifier: Mapping[str, Any]
    output_root: Path
    _record_bytes: bytes

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)

    @property
    def model_identity_sha256(self) -> str:
        return _sha256_json(
            {"schema": T1_MODEL_IDENTITY_SCHEMA, "model": dict(self.model)}
        )

    @property
    def processor_identity_sha256(self) -> str:
        return _sha256_json(
            {
                "schema": T1_PROCESSOR_IDENTITY_SCHEMA,
                "preprocessor_config_sha256": self.model["preprocessor_config_sha256"],
                "tokenizer_config_sha256": self.model["tokenizer_config_sha256"],
                "tokenizer_json_sha256": self.model["tokenizer_json_sha256"],
                "chat_template_sha256": self.model["chat_template_sha256"],
                "tokenizer_length": self.model["tokenizer_length"],
                "eos_token_id": self.model["eos_token_id"],
                "prompt": dict(self.prompt),
                "image": dict(self.image),
            }
        )

    @property
    def runtime_identity_sha256(self) -> str:
        return _sha256_json(
            {"schema": T1_RUNTIME_IDENTITY_SCHEMA, "runtime": dict(self.runtime)}
        )

    def attempt_seed(self, *, candidate_sha256: str, attempt_index: int) -> int:
        return derive_t1_attempt_seed(
            run_manifest_sha256=self.manifest_sha256,
            candidate_sha256=candidate_sha256,
            attempt_index=attempt_index,
            seed_root=self.sampling["seed_root"],
            seed_namespace=self.sampling["seed_namespace"],
        )

    def budget(self, revision: int) -> T1ResponseBudget:
        for budget in self.response_budgets:
            if budget.revision == revision:
                return budget
        raise ValueError(f"unknown response budget revision {revision}")


def load_t1_run_config(
    path: str | Path, *, verify_data_files: bool = False
) -> T1RunConfig:
    """Load one strict, side-effect-free JSON run configuration.

    ``verify_data_files`` additionally hashes and counts the three candidate
    JSONL files.  It remains explicit because a caller may load the identity on
    every worker while performing the expensive source check once at launch.
    """

    config_path = Path(path)
    record = _load_json_object(config_path)
    _exact_fields(record, _RUN_FIELDS, field_name="run config")
    if record["schema_version"] != T1_RUN_CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {T1_RUN_CONFIG_SCHEMA!r}")
    run_id = _required_string(record["run_id"], field_name="run_id")
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsupported characters")

    model = _mapping(record["model"], field_name="model")
    _exact_fields(model, _MODEL_FIELDS, field_name="model")
    model_repository = _required_string(
        model["repository"], field_name="model.repository"
    )
    expected_model_path = T1_MODEL_PATH_BY_REPOSITORY.get(model_repository)
    if expected_model_path is None:
        raise ValueError("model.repository is not an accepted Qwen3-VL edition")
    if (
        str(_absolute_normal_path(model["path"], field_name="model.path"))
        != expected_model_path
    ):
        raise ValueError(
            f"model.path must be {expected_model_path!r} for {model_repository!r}"
        )
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be 'bfloat16'")
    for field in (
        "config_sha256",
        "generation_config_sha256",
        "tokenizer_config_sha256",
        "tokenizer_json_sha256",
        "preprocessor_config_sha256",
        "chat_template_sha256",
    ):
        _required_sha256(model[field], field_name=f"model.{field}")
    _required_int(
        model["tokenizer_length"], field_name="model.tokenizer_length", minimum=1
    )
    _required_int(model["eos_token_id"], field_name="model.eos_token_id")
    generation_eos = model["generation_eos_token_ids"]
    if (
        not isinstance(generation_eos, Sequence)
        or isinstance(generation_eos, (str, bytes))
        or list(generation_eos) != [151_645, 151_643]
    ):
        raise ValueError("model.generation_eos_token_ids must be [151645, 151643]")
    if (
        _required_bool(model["trust_remote_code"], field_name="model.trust_remote_code")
        is not True
    ):
        raise ValueError("model.trust_remote_code must be true")
    if model["quantization"] is not None:
        raise ValueError("model.quantization must be null")

    prompt = _mapping(record["prompt"], field_name="prompt")
    _exact_fields(prompt, _PROMPT_FIELDS, field_name="prompt")
    if prompt["schema"] != T1_PROMPT_SCHEMA:
        raise ValueError(f"prompt.schema must be {T1_PROMPT_SCHEMA!r}")
    order = prompt["user_content_order"]
    if (
        not isinstance(order, Sequence)
        or isinstance(order, (str, bytes))
        or list(order)
        != [
            "image",
            "question",
        ]
    ):
        raise ValueError("prompt.user_content_order must be ['image', 'question']")
    for field in ("no_system", "no_tools", "add_generation_prompt"):
        if _required_bool(prompt[field], field_name=f"prompt.{field}") is not True:
            raise ValueError(f"prompt.{field} must be true")

    image = _mapping(record["image"], field_name="image")
    _exact_fields(image, _IMAGE_FIELDS, field_name="image")
    min_pixels = image["min_pixels"]
    if min_pixels is not None:
        _required_int(min_pixels, field_name="image.min_pixels", minimum=1)
    if image["max_pixels"] != T1_MAX_PIXELS:
        raise ValueError(f"image.max_pixels must be {T1_MAX_PIXELS}")
    if min_pixels is not None and min_pixels > T1_MAX_PIXELS:
        raise ValueError("image.min_pixels must not exceed image.max_pixels")
    if image["resize_factor"] != 32:
        raise ValueError("image.resize_factor must be 32")
    if image["resample"] != "transformers-fast-bicubic":
        raise ValueError("image.resample must be 'transformers-fast-bicubic'")
    if _required_bool(image["pre_resize"], field_name="image.pre_resize"):
        raise ValueError("image.pre_resize must be false")
    if (
        _required_bool(
            image["preserve_aspect_ratio"],
            field_name="image.preserve_aspect_ratio",
        )
        is not True
    ):
        raise ValueError("image.preserve_aspect_ratio must be true")
    if (
        _required_bool(
            image["processor_do_resize"], field_name="image.processor_do_resize"
        )
        is not True
    ):
        raise ValueError("image.processor_do_resize must be true")
    if image["limit_image_per_prompt"] != 1:
        raise ValueError("image.limit_image_per_prompt must be 1")
    if image["color_mode"] != "RGB":
        raise ValueError("image.color_mode must be 'RGB'")
    if image["alpha_handling"] != "pil-convert-rgb-discard-alpha-v1":
        raise ValueError(
            "image.alpha_handling must be 'pil-convert-rgb-discard-alpha-v1'"
        )
    if image["source_pixel_hash_schema"] != T1_SOURCE_RGB_SCHEMA:
        raise ValueError(
            f"image.source_pixel_hash_schema must be {T1_SOURCE_RGB_SCHEMA!r}"
        )

    sampling = _mapping(record["sampling"], field_name="sampling")
    _exact_fields(sampling, _SAMPLING_FIELDS, field_name="sampling")
    fixed_numbers = {
        "attempts": T1_ATTEMPTS,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    for field, expected in fixed_numbers.items():
        actual = (
            _required_int(sampling[field], field_name=f"sampling.{field}")
            if field == "attempts"
            else _required_float(sampling[field], field_name=f"sampling.{field}")
        )
        if actual != expected:
            raise ValueError(f"sampling.{field} must be {expected!r}")
    processors = sampling["logit_processors"]
    if (
        not isinstance(processors, Sequence)
        or isinstance(processors, (str, bytes))
        or list(processors)
    ):
        raise ValueError("sampling.logit_processors must be an empty list")
    _required_int(
        sampling["seed_root"],
        field_name="sampling.seed_root",
        maximum=2**63 - 1,
    )
    _required_string(sampling["seed_namespace"], field_name="sampling.seed_namespace")
    if (
        _required_bool(sampling["do_sample"], field_name="sampling.do_sample")
        is not True
    ):
        raise ValueError("sampling.do_sample must be true")
    if _required_bool(sampling["ignore_eos"], field_name="sampling.ignore_eos"):
        raise ValueError("sampling.ignore_eos must be false")
    stop_token_ids = sampling["stop_token_ids"]
    if (
        not isinstance(stop_token_ids, Sequence)
        or isinstance(stop_token_ids, (str, bytes))
        or list(stop_token_ids) != [model["eos_token_id"]]
    ):
        raise ValueError("sampling.stop_token_ids must contain only model.eos_token_id")
    effective_stop_token_ids = sampling["effective_stop_token_ids"]
    if (
        not isinstance(effective_stop_token_ids, Sequence)
        or isinstance(effective_stop_token_ids, (str, bytes))
        or list(effective_stop_token_ids) != list(model["generation_eos_token_ids"])
    ):
        raise ValueError(
            "sampling.effective_stop_token_ids must match model generation EOS IDs"
        )
    stop_strings = sampling["stop_strings"]
    if (
        not isinstance(stop_strings, Sequence)
        or isinstance(stop_strings, (str, bytes))
        or list(stop_strings)
    ):
        raise ValueError("sampling.stop_strings must be an empty list")
    if _required_bool(
        sampling["include_stop_str_in_output"],
        field_name="sampling.include_stop_str_in_output",
    ):
        raise ValueError("sampling.include_stop_str_in_output must be false")
    if (
        _required_bool(sampling["detokenize"], field_name="sampling.detokenize")
        is not True
    ):
        raise ValueError("sampling.detokenize must be true")
    for field in ("skip_special_tokens", "spaces_between_special_tokens"):
        if _required_bool(sampling[field], field_name=f"sampling.{field}"):
            raise ValueError(f"sampling.{field} must be false")

    budgets_value = record["response_budgets"]
    if not isinstance(budgets_value, Sequence) or isinstance(
        budgets_value, (str, bytes)
    ):
        raise ValueError("response_budgets must be a list")
    expected_budgets = (
        (0, 65_536, 40_960),
        (1, 131_072, 98_304),
        (2, 262_144, 196_608),
    )
    budgets: list[T1ResponseBudget] = []
    for index, value in enumerate(budgets_value):
        budget = _mapping(value, field_name=f"response_budgets[{index}]")
        _exact_fields(budget, _BUDGET_FIELDS, field_name=f"response_budgets[{index}]")
        parsed = T1ResponseBudget(
            revision=_required_int(
                budget["revision"], field_name=f"response_budgets[{index}].revision"
            ),
            max_model_len=_required_int(
                budget["max_model_len"],
                field_name=f"response_budgets[{index}].max_model_len",
                minimum=1,
            ),
            max_new_tokens=_required_int(
                budget["max_new_tokens"],
                field_name=f"response_budgets[{index}].max_new_tokens",
                minimum=1,
            ),
        )
        budgets.append(parsed)
    actual_budgets = tuple(
        (item.revision, item.max_model_len, item.max_new_tokens) for item in budgets
    )
    if actual_budgets != expected_budgets:
        raise ValueError(f"response_budgets must be exactly {expected_budgets!r}")

    runtime = _mapping(record["runtime"], field_name="runtime")
    _exact_fields(runtime, _RUNTIME_FIELDS, field_name="runtime")
    for field in (
        "backend",
        "version",
        "python",
        "torch",
        "transformers",
        "pillow",
        "flashinfer",
        "mm_encoder_attn_backend",
        "decoder_attn_backend",
    ):
        _required_string(runtime[field], field_name=f"runtime.{field}")
    if runtime["backend"] != "vllm":
        raise ValueError("runtime.backend must be 'vllm'")
    if runtime["world_size"] != T1_SHARD_COUNT:
        raise ValueError(f"runtime.world_size must be {T1_SHARD_COUNT}")
    if runtime["tensor_parallel_size"] != 1:
        raise ValueError("runtime.tensor_parallel_size must be 1")
    if runtime["max_num_seqs"] != 32:
        raise ValueError("runtime.max_num_seqs must be 32")
    utilization = _required_float(
        runtime["gpu_memory_utilization"], field_name="runtime.gpu_memory_utilization"
    )
    if not 0.0 < utilization <= 1.0:
        raise ValueError("runtime.gpu_memory_utilization must be in (0, 1]")
    if runtime["mm_encoder_attn_backend"] != "TORCH_SDPA":
        raise ValueError("runtime.mm_encoder_attn_backend must be 'TORCH_SDPA'")
    if runtime["decoder_attn_backend"] != "FLASHINFER":
        raise ValueError("runtime.decoder_attn_backend must be 'FLASHINFER'")
    if runtime["max_num_batched_tokens"] != 65_536:
        raise ValueError("runtime.max_num_batched_tokens must be 65536")
    for field in ("enable_prefix_caching", "enable_chunked_prefill"):
        if _required_bool(runtime[field], field_name=f"runtime.{field}") is not True:
            raise ValueError(f"runtime.{field} must be true")
    if (
        _required_float(
            runtime["mm_processor_cache_gb"],
            field_name="runtime.mm_processor_cache_gb",
        )
        != 4.0
    ):
        raise ValueError("runtime.mm_processor_cache_gb must be 4.0")
    if runtime["engine_seed"] != 42:
        raise ValueError("runtime.engine_seed must be 42")
    if runtime["chunk_candidates"] != 4:
        raise ValueError("runtime.chunk_candidates must be 4")
    if runtime["max_inflight"] != 64:
        raise ValueError("runtime.max_inflight must be 64")
    if (
        _required_bool(
            runtime["retain_token_ids"], field_name="runtime.retain_token_ids"
        )
        is not True
    ):
        raise ValueError("runtime.retain_token_ids must be true")
    if runtime["generation_config_mode"] != "auto":
        raise ValueError("runtime.generation_config_mode must be 'auto'")

    data = _mapping(record["data"], field_name="data")
    _exact_fields(data, _DATA_FIELDS, field_name="data")
    sources_value = data["sources"]
    if not isinstance(sources_value, Sequence) or isinstance(
        sources_value, (str, bytes)
    ):
        raise ValueError("data.sources must be a list")
    sources: list[T1DataSource] = []
    for index, value in enumerate(sources_value):
        source_record = _mapping(value, field_name=f"data.sources[{index}]")
        _exact_fields(
            source_record, _SOURCE_FIELDS, field_name=f"data.sources[{index}]"
        )
        try:
            source = SelectionSource(source_record["source"])
        except ValueError as exc:
            raise ValueError(f"data.sources[{index}].source is unsupported") from exc
        source_path = _absolute_normal_path(
            source_record["path"], field_name=f"data.sources[{index}].path"
        )
        source_sha256 = _required_sha256(
            source_record["sha256"], field_name=f"data.sources[{index}].sha256"
        )
        rows = _required_int(
            source_record["rows"], field_name=f"data.sources[{index}].rows", minimum=1
        )
        sources.append(T1DataSource(source, source_path, source_sha256, rows))
    if {item.source for item in sources} != set(SelectionSource) or len(sources) != 3:
        raise ValueError("data.sources must contain vstar, arxivqa, and thinklite once")
    if verify_data_files:
        for source in sources:
            actual_sha256 = _sha256_file(source.path)
            if actual_sha256 != source.sha256:
                raise ValueError(f"{source.source.value} candidate SHA-256 mismatch")
            with source.path.open("rb") as handle:
                actual_rows = sum(bool(line.strip()) for line in handle)
            if actual_rows != source.rows:
                raise ValueError(
                    f"{source.source.value} candidate row count {actual_rows} != {source.rows}"
                )

    selection = _mapping(record["selection"], field_name="selection")
    _exact_fields(selection, _SELECTION_FIELDS, field_name="selection")
    selection_profile = (
        selection["kind"],
        selection["algorithm_version"],
    )
    if selection_profile == (
        "stratified_canary",
        "t1-canary-content-hash-v1",
    ):
        expected_selection_rows = 192
        expected_source_counts = {source.value: 64 for source in SelectionSource}
    elif selection_profile == (
        "source_quota",
        T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION,
    ):
        expected_selection_rows = T1_RECOMMENDED_SELECTION_ROWS
        expected_source_counts = dict(T1_RECOMMENDED_SOURCE_QUOTAS)
    else:
        raise ValueError("selection kind/algorithm profile is unsupported")
    candidates_path = _absolute_normal_path(
        selection["candidates_path"], field_name="selection.candidates_path"
    )
    candidates_sha256 = _required_sha256(
        selection["candidates_sha256"], field_name="selection.candidates_sha256"
    )
    selection_rows = _required_int(
        selection["rows"], field_name="selection.rows", minimum=1
    )
    if selection_rows != expected_selection_rows:
        raise ValueError(
            f"selection.rows must be {expected_selection_rows} for this profile"
        )
    selection_manifest_path = _absolute_normal_path(
        selection["manifest_path"], field_name="selection.manifest_path"
    )
    selection_manifest_sha256 = _required_sha256(
        selection["manifest_sha256"], field_name="selection.manifest_sha256"
    )
    if verify_data_files:
        if _sha256_file(candidates_path) != candidates_sha256:
            raise ValueError("selection candidates SHA-256 mismatch")
        if _sha256_file(selection_manifest_path) != selection_manifest_sha256:
            raise ValueError("selection manifest SHA-256 mismatch")

        selected_candidates: list[SelectionCandidate] = []
        with candidates_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    candidate_record = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=lambda constant: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON number: {constant}")
                        ),
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid selection candidate JSON at line {line_number}"
                    ) from exc
                selected_candidates.append(
                    SelectionCandidate.from_record(
                        _mapping(
                            candidate_record,
                            field_name=f"selection candidate line {line_number}",
                        )
                    )
                )
        if len(selected_candidates) != selection_rows:
            raise ValueError(
                "selection candidate row count "
                f"{len(selected_candidates)} != {selection_rows}"
            )
        identities = [candidate.identity_sha256 for candidate in selected_candidates]
        sample_ids = [candidate.sample_id for candidate in selected_candidates]
        if (
            len(set(identities)) != selection_rows
            or len(set(sample_ids)) != selection_rows
        ):
            raise ValueError(
                "selection candidates must have unique identities and sample IDs"
            )
        source_counts = {
            source.value: sum(
                candidate.source is source for candidate in selected_candidates
            )
            for source in SelectionSource
        }
        if source_counts != expected_source_counts:
            raise ValueError(
                "selection candidate source counts differ from the profile"
            )

        selection_manifest = _load_json_object(selection_manifest_path)
        if (
            selection_manifest.get("selection_algorithm_version")
            != selection["algorithm_version"]
        ):
            raise ValueError("selection manifest algorithm mismatch")
        if selection_manifest.get("selection_is_outcome_independent") is not True:
            raise ValueError("selection manifest must be outcome-independent")
        if selection["kind"] == "stratified_canary":
            if (
                selection_manifest.get("schema_version")
                != "tgvf.policy-selection.t1-canary-manifest.v1"
            ):
                raise ValueError("selection manifest schema is unsupported")
            selected_manifest_rows = selection_manifest.get("selected")
            if not isinstance(selected_manifest_rows, list):
                raise ValueError("selection manifest selected must be a list")
            manifest_identities = [
                _mapping(item, field_name=f"selection manifest selected[{index}]").get(
                    "candidate_sha256"
                )
                for index, item in enumerate(selected_manifest_rows)
            ]
            if manifest_identities != identities:
                raise ValueError(
                    "selection manifest candidate order or identity mismatch"
                )
        else:
            if (
                selection_manifest.get("schema_version")
                != T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA
            ):
                raise ValueError(
                    "source-quota selection manifest schema is unsupported"
                )
            if (
                selection_manifest.get("selection_namespace")
                != T1_RECOMMENDED_SELECTION_NAMESPACE
            ):
                raise ValueError("source-quota selection namespace differs")
            if selection_manifest.get("rows") != selection_rows:
                raise ValueError("source-quota selection manifest row count differs")
            if (
                selection_manifest.get("logical_attempts")
                != selection_rows * T1_ATTEMPTS
            ):
                raise ValueError("source-quota logical attempt count differs")
            if selection_manifest.get("source_quotas") != expected_source_counts:
                raise ValueError("source-quota manifest quotas differ")
            if selection_manifest.get("source_counts") != expected_source_counts:
                raise ValueError("source-quota manifest counts differ")
            if (
                selection_manifest.get("candidates_path") != str(candidates_path)
                or selection_manifest.get("candidates_sha256") != candidates_sha256
            ):
                raise ValueError("source-quota candidate identity differs")
            manifest_sources = selection_manifest.get("sources")
            if not isinstance(manifest_sources, list) or len(manifest_sources) != 3:
                raise ValueError("source-quota manifest sources differ")
            source_bindings = {item.source.value: item for item in sources}
            for index, value in enumerate(manifest_sources):
                manifest_source = _mapping(
                    value, field_name=f"source-quota manifest sources[{index}]"
                )
                source_name = manifest_source.get("source")
                binding = source_bindings.get(source_name)
                if binding is None:
                    raise ValueError("source-quota manifest source is unsupported")
                expected_binding = (
                    str(binding.path),
                    binding.sha256,
                    binding.rows,
                    expected_source_counts[source_name],
                    expected_source_counts[source_name],
                )
                actual_binding = (
                    manifest_source.get("path"),
                    manifest_source.get("sha256"),
                    manifest_source.get("rows"),
                    manifest_source.get("quota"),
                    source_counts[source_name],
                )
                if actual_binding != expected_binding:
                    raise ValueError("source-quota manifest source binding differs")

    verifier = _mapping(record["verifier"], field_name="verifier")
    _exact_fields(verifier, _VERIFIER_FIELDS, field_name="verifier")
    for field in (
        "schema",
        "answer_parser",
        "arxivqa_rule",
        "thinklite_rule",
        "vstar_rule",
    ):
        _required_string(verifier[field], field_name=f"verifier.{field}")
    expected_answer_parser = {
        "Qwen/Qwen3-VL-8B-Thinking": T1_THINKING_ANSWER_PARSER,
        "Qwen/Qwen3-VL-8B-Instruct": T1_INSTRUCT_ANSWER_PARSER,
    }[model["repository"]]
    if verifier["answer_parser"] != expected_answer_parser:
        raise ValueError(
            "verifier.answer_parser differs from the selected model edition"
        )
    judge = _mapping(verifier["semantic_judge"], field_name="verifier.semantic_judge")
    _exact_fields(judge, _SEMANTIC_JUDGE_FIELDS, field_name="verifier.semantic_judge")
    if judge["provider"] != "local-openai-compatible":
        raise ValueError("semantic judge provider must be local-openai-compatible")
    if judge["repository"] != "Qwen/Qwen2.5-72B-Instruct":
        raise ValueError("semantic judge repository must be Qwen/Qwen2.5-72B-Instruct")
    _absolute_normal_path(judge["path"], field_name="verifier.semantic_judge.path")
    if judge["served_name"] != "Qwen2.5-72B-Instruct":
        raise ValueError("semantic judge served_name is not accepted")
    for field in ("prompt_sha256", "config_sha256"):
        _required_sha256(judge[field], field_name=f"verifier.semantic_judge.{field}")
    if (
        _required_float(
            judge["temperature"], field_name="verifier.semantic_judge.temperature"
        )
        != 0.0
    ):
        raise ValueError("semantic judge temperature must be 0.0")
    if judge["max_tokens"] != 256:
        raise ValueError("semantic judge max_tokens must be 256")
    if _required_bool(judge["remote"], field_name="verifier.semantic_judge.remote"):
        raise ValueError("semantic judge remote must be false")

    output_root = _absolute_normal_path(record["output_root"], field_name="output_root")
    normalized = _json_clone(record)
    record_bytes = _canonical_json_bytes(normalized)
    return T1RunConfig(
        run_id=run_id,
        manifest_sha256=_sha256_bytes(record_bytes),
        model=_freeze_json(dict(normalized["model"])),
        prompt=_freeze_json(dict(normalized["prompt"])),
        image=_freeze_json(dict(normalized["image"])),
        sampling=_freeze_json(dict(normalized["sampling"])),
        response_budgets=tuple(budgets),
        runtime=_freeze_json(dict(normalized["runtime"])),
        data_sources=tuple(sources),
        selection=_freeze_json(dict(normalized["selection"])),
        verifier=_freeze_json(dict(normalized["verifier"])),
        output_root=output_root,
        _record_bytes=record_bytes,
    )


def derive_t1_attempt_seed(
    *,
    run_manifest_sha256: str,
    candidate_sha256: str,
    attempt_index: int,
    seed_root: int = 0,
    seed_namespace: str = "qwen3-policy-selection-t1-v1",
) -> int:
    """Derive one batch/rank-invariant low-31-bit attempt seed."""

    _required_sha256(run_manifest_sha256, field_name="run_manifest_sha256")
    _required_sha256(candidate_sha256, field_name="candidate_sha256")
    _required_int(attempt_index, field_name="attempt_index", maximum=T1_ATTEMPTS - 1)
    _required_int(seed_root, field_name="seed_root", maximum=2**63 - 1)
    _required_string(seed_namespace, field_name="seed_namespace")
    state = {
        "schema": T1_ATTEMPT_SEED_SCHEMA,
        "run_manifest_sha256": run_manifest_sha256,
        "candidate_sha256": candidate_sha256,
        "attempt_index": attempt_index,
        "seed_root": seed_root,
        "seed_namespace": seed_namespace,
    }
    digest = hashlib.sha256(
        b"tgvf-policy-selection-t1-seed-v1\0" + _canonical_json_bytes(state)
    ).digest()
    return int.from_bytes(digest[:8], "big") % T1_SEED_MODULUS


def candidate_rank(candidate_sha256: str, *, world_size: int = T1_SHARD_COUNT) -> int:
    """Assign all attempts for one candidate to one stable rank."""

    value = _required_sha256(candidate_sha256, field_name="candidate_sha256")
    size = _required_int(world_size, field_name="world_size", minimum=1)
    return int(value, 16) % size


def native_user_message_descriptor(
    *, image: str, question: str
) -> list[dict[str, Any]]:
    """Return the sole user message accepted by the tool-free T1 prompt."""

    image_value = _required_string(image, field_name="image")
    question_value = _required_string(question, field_name="question")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_value},
                {"type": "text", "text": question_value},
            ],
        }
    ]


def native_prompt_identity_sha256(
    *, question: str, image_sha256: str, chat_template_sha256: str
) -> str:
    """Bind prompt semantics without making a relocatable image path an identity."""

    question_value = _required_string(question, field_name="question")
    image_identity = _required_sha256(image_sha256, field_name="image_sha256")
    template_identity = _required_sha256(
        chat_template_sha256, field_name="chat_template_sha256"
    )
    return _sha256_json(
        {
            "schema": T1_PROMPT_IDENTITY_SCHEMA,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_sha256": image_identity},
                        {"type": "text", "text": question_value},
                    ],
                }
            ],
            "system": None,
            "tools": None,
            "add_generation_prompt": True,
            "chat_template_sha256": template_identity,
        }
    )


def sampled_token_ids_sha256(token_ids: Sequence[int]) -> str:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("sampled token IDs must be a sequence")
    normalized = tuple(token_ids)
    if any(type(token_id) is not int or token_id < 0 for token_id in normalized):
        raise ValueError("sampled token IDs must be non-negative integers")
    return _sha256_json({"schema": T1_TOKEN_IDS_SCHEMA, "token_ids": normalized})


def rendered_prompt_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash the exact processor/vLLM-expanded prompt token sequence."""

    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("rendered prompt token IDs must be a sequence")
    normalized = tuple(token_ids)
    if not normalized or any(
        type(token_id) is not int or token_id < 0 for token_id in normalized
    ):
        raise ValueError(
            "rendered prompt token IDs must be a non-empty non-negative sequence"
        )
    return _sha256_json(
        {"schema": T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA, "token_ids": normalized}
    )


def source_rgb_sha256(*, width: int, height: int, pixel_bytes: bytes) -> str:
    """Hash exact row-major 8-bit RGB pixels given to the native processor."""

    parsed_width = _required_int(width, field_name="width", minimum=1)
    parsed_height = _required_int(height, field_name="height", minimum=1)
    if not isinstance(pixel_bytes, bytes):
        raise TypeError("pixel_bytes must be bytes")
    expected_bytes = parsed_width * parsed_height * 3
    if len(pixel_bytes) != expected_bytes:
        raise ValueError(
            f"RGB pixel byte length {len(pixel_bytes)} != {expected_bytes}"
        )
    metadata = _canonical_json_bytes(
        {
            "schema": T1_SOURCE_RGB_SCHEMA,
            "mode": "RGB",
            "width": parsed_width,
            "height": parsed_height,
            "byte_length": expected_bytes,
        }
    )
    return _sha256_bytes(b"tgvf-source-rgb-v1\0" + metadata + b"\0" + pixel_bytes)


class GenerationDisposition(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"


def classify_generation_finish(finish_reason: str) -> GenerationDisposition:
    try:
        reason = _required_string(finish_reason, field_name="finish_reason")
    except ValueError as exc:
        raise ValueError("finish_reason must be stop, length, or error") from exc
    if reason == "length":
        return GenerationDisposition.TRUNCATED
    if reason == "stop":
        return GenerationDisposition.COMPLETED
    if reason == "error":
        return GenerationDisposition.GENERATION_ERROR
    raise ValueError("finish_reason must be stop, length, or error")


@dataclass(frozen=True, slots=True)
class T1RawGenerationEvidence:
    run_id: str
    run_manifest_sha256: str
    request_id: str
    sample_id: str
    candidate_sha256: str
    source: SelectionSource
    attempt_index: int
    attempt_seed: int
    budget_revision: int
    max_model_len: int
    max_new_tokens: int
    prompt_sha256: str
    rendered_prompt_token_ids_sha256: str
    prompt_token_count: int
    image_sha256: str
    source_width: int
    source_height: int
    source_mode: str
    source_rgb_sha256: str
    processed_width: int
    processed_height: int
    sampled_token_ids_sha256: str
    sampled_token_count: int
    sampled_token_ids: tuple[int, ...] | None
    raw_text: str
    finish_reason: str
    stop_reason: str | int | None
    backend: Mapping[str, Any]
    generation_error: str | None
    _record_bytes: bytes

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "T1RawGenerationEvidence":
        value = _mapping(record, field_name="raw generation")
        actual_fields = set(value)
        if not _RAW_REQUIRED_FIELDS <= actual_fields or actual_fields - (
            _RAW_REQUIRED_FIELDS | _RAW_OPTIONAL_FIELDS
        ):
            missing = sorted(_RAW_REQUIRED_FIELDS - actual_fields)
            unknown = sorted(
                actual_fields - (_RAW_REQUIRED_FIELDS | _RAW_OPTIONAL_FIELDS)
            )
            raise ValueError(
                f"raw generation fields differ; missing={missing}, unknown={unknown}"
            )
        if value["schema_version"] != T1_RAW_GENERATION_SCHEMA:
            raise ValueError(
                f"raw generation schema_version must be {T1_RAW_GENERATION_SCHEMA!r}"
            )
        run_id = _required_string(value["run_id"], field_name="run_id")
        run_sha = _required_sha256(
            value["run_manifest_sha256"], field_name="run_manifest_sha256"
        )
        request_id = _required_string(value["request_id"], field_name="request_id")
        sample_id = _required_string(value["sample_id"], field_name="sample_id")
        candidate_sha = _required_sha256(
            value["candidate_sha256"], field_name="candidate_sha256"
        )
        try:
            source = SelectionSource(value["source"])
        except ValueError as exc:
            raise ValueError("source is unsupported") from exc
        if value["branch"] != SelectionBranch.FULL_IMAGE.value:
            raise ValueError("T1 raw generation branch must be full_image")
        attempt_index = _required_int(
            value["attempt_index"],
            field_name="attempt_index",
            maximum=T1_ATTEMPTS - 1,
        )
        expected_request_id = stable_selection_request_id(
            candidate_sha256=candidate_sha,
            branch=SelectionBranch.FULL_IMAGE,
            attempt_index=attempt_index,
        )
        if request_id != expected_request_id:
            raise ValueError("raw generation request identity mismatch")
        attempt_seed = _required_int(
            value["attempt_seed"],
            field_name="attempt_seed",
            maximum=T1_SEED_MODULUS - 1,
        )
        budget_revision = _required_int(
            value["budget_revision"], field_name="budget_revision"
        )
        max_model_len = _required_int(
            value["max_model_len"], field_name="max_model_len", minimum=1
        )
        max_new_tokens = _required_int(
            value["max_new_tokens"], field_name="max_new_tokens", minimum=1
        )
        prompt_sha = _required_sha256(
            value["prompt_sha256"], field_name="prompt_sha256"
        )
        rendered_prompt_sha = _required_sha256(
            value["rendered_prompt_token_ids_sha256"],
            field_name="rendered_prompt_token_ids_sha256",
        )
        prompt_token_count = _required_int(
            value["prompt_token_count"], field_name="prompt_token_count", minimum=1
        )
        image_sha = _required_sha256(value["image_sha256"], field_name="image_sha256")

        image_evidence = _mapping(value["image_evidence"], field_name="image_evidence")
        _exact_fields(
            image_evidence, _IMAGE_EVIDENCE_FIELDS, field_name="image_evidence"
        )
        source_width = _required_int(
            image_evidence["source_width"],
            field_name="image_evidence.source_width",
            minimum=1,
        )
        source_height = _required_int(
            image_evidence["source_height"],
            field_name="image_evidence.source_height",
            minimum=1,
        )
        source_mode = _required_string(
            image_evidence["source_mode"], field_name="image_evidence.source_mode"
        )
        source_rgb_sha = _required_sha256(
            image_evidence["source_rgb_sha256"],
            field_name="image_evidence.source_rgb_sha256",
        )
        processed_width = _required_int(
            image_evidence["processed_width"],
            field_name="image_evidence.processed_width",
            minimum=1,
        )
        processed_height = _required_int(
            image_evidence["processed_height"],
            field_name="image_evidence.processed_height",
            minimum=1,
        )

        token_sha = _required_sha256(
            value["sampled_token_ids_sha256"], field_name="sampled_token_ids_sha256"
        )
        token_count = _required_int(
            value["sampled_token_count"], field_name="sampled_token_count"
        )
        token_ids_value = value.get("sampled_token_ids")
        token_ids: tuple[int, ...] | None
        if token_ids_value is None:
            token_ids = None
        else:
            if not isinstance(token_ids_value, Sequence) or isinstance(
                token_ids_value, (str, bytes)
            ):
                raise ValueError("sampled_token_ids must be a list when present")
            token_ids = tuple(token_ids_value)
            actual_token_sha = sampled_token_ids_sha256(token_ids)
            if actual_token_sha != token_sha:
                raise ValueError("sampled token IDs SHA-256 mismatch")
            if len(token_ids) != token_count:
                raise ValueError("sampled_token_count differs from sampled_token_ids")

        raw_text = value["raw_text"]
        if not isinstance(raw_text, str):
            raise ValueError("raw_text must be a string")
        finish_reason = _required_string(
            value["finish_reason"], field_name="finish_reason"
        )
        disposition = classify_generation_finish(finish_reason)
        stop_reason = value["stop_reason"]
        if stop_reason is not None and type(stop_reason) not in {str, int}:
            raise ValueError("stop_reason must be a string, integer, or null")
        if disposition is GenerationDisposition.COMPLETED:
            if stop_reason is not None and stop_reason not in {151_645, 151_643}:
                raise ValueError("normal stop_reason is outside the effective EOS set")
        elif stop_reason is not None:
            raise ValueError("length/error evidence must have a null stop_reason")
        backend = _mapping(value["backend"], field_name="backend")
        _exact_fields(backend, _BACKEND_EVIDENCE_FIELDS, field_name="backend")
        _required_string(backend["name"], field_name="backend.name")
        _required_string(backend["version"], field_name="backend.version")
        for field in ("runtime_sha256", "model_sha256", "processor_sha256"):
            _required_sha256(backend[field], field_name=f"backend.{field}")
        generation_error = value.get("generation_error")
        if disposition is GenerationDisposition.GENERATION_ERROR:
            generation_error = _required_string(
                generation_error, field_name="generation_error"
            )
            if token_count != 0 or raw_text:
                raise ValueError(
                    "generation error evidence must have no sampled output"
                )
        elif generation_error is not None:
            raise ValueError("generation_error is only valid for finish_reason=error")

        normalized = _json_clone(value)
        record_bytes = _canonical_json_bytes(normalized)
        return cls(
            run_id=run_id,
            run_manifest_sha256=run_sha,
            request_id=request_id,
            sample_id=sample_id,
            candidate_sha256=candidate_sha,
            source=source,
            attempt_index=attempt_index,
            attempt_seed=attempt_seed,
            budget_revision=budget_revision,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            prompt_sha256=prompt_sha,
            rendered_prompt_token_ids_sha256=rendered_prompt_sha,
            prompt_token_count=prompt_token_count,
            image_sha256=image_sha,
            source_width=source_width,
            source_height=source_height,
            source_mode=source_mode,
            source_rgb_sha256=source_rgb_sha,
            processed_width=processed_width,
            processed_height=processed_height,
            sampled_token_ids_sha256=token_sha,
            sampled_token_count=token_count,
            sampled_token_ids=token_ids,
            raw_text=raw_text,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            backend=_freeze_json(dict(normalized["backend"])),
            generation_error=generation_error,
            _record_bytes=record_bytes,
        )

    @property
    def evidence_sha256(self) -> str:
        return _sha256_bytes(self._record_bytes)

    @property
    def disposition(self) -> GenerationDisposition:
        return classify_generation_finish(self.finish_reason)

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)

    def validate_against_run(self, run: T1RunConfig) -> None:
        if not isinstance(run, T1RunConfig):
            raise TypeError("run must be T1RunConfig")
        if self.run_id != run.run_id or self.run_manifest_sha256 != run.manifest_sha256:
            raise ValueError("raw generation run identity mismatch")
        if self.attempt_seed != run.attempt_seed(
            candidate_sha256=self.candidate_sha256,
            attempt_index=self.attempt_index,
        ):
            raise ValueError("raw generation attempt seed mismatch")
        budget = run.budget(self.budget_revision)
        if (self.max_model_len, self.max_new_tokens) != (
            budget.max_model_len,
            budget.max_new_tokens,
        ):
            raise ValueError("raw generation response budget mismatch")
        expected_backend = {
            "name": run.runtime["backend"],
            "version": run.runtime["version"],
            "runtime_sha256": run.runtime_identity_sha256,
            "model_sha256": run.model_identity_sha256,
            "processor_sha256": run.processor_identity_sha256,
        }
        if dict(self.backend) != expected_backend:
            raise ValueError("raw generation backend identity mismatch")
        if self.processed_width * self.processed_height > run.image["max_pixels"]:
            raise ValueError("processed image exceeds the run max_pixels")


def extract_final_answer(raw_text: str) -> str | None:
    """Return the non-empty suffix after the last sampled ``</think>``."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    marker = "</think>"
    index = raw_text.rfind(marker)
    if index < 0:
        return None
    suffix = raw_text[index + len(marker) :].strip()
    suffix = _QWEN_TERMINAL_TEXT.sub("", suffix).strip()
    return suffix or None


def extract_direct_completion(raw_text: str) -> str | None:
    """Return an Instruct completion without inventing a reasoning boundary."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    answer = _QWEN_TERMINAL_TEXT.sub("", raw_text.strip()).strip()
    return answer or None


def parse_t1_answer(raw_text: str, *, answer_parser: str) -> str | None:
    """Dispatch final-answer extraction by the run-bound native dialect."""

    if answer_parser == T1_THINKING_ANSWER_PARSER:
        return extract_final_answer(raw_text)
    if answer_parser == T1_INSTRUCT_ANSWER_PARSER:
        return extract_direct_completion(raw_text)
    raise ValueError(f"unsupported T1 answer parser: {answer_parser!r}")


class VerificationOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SEMANTIC_REQUIRED = "semantic_required"


@dataclass(frozen=True, slots=True)
class DeterministicVerification:
    outcome: VerificationOutcome
    route: str
    evidence: str

    @property
    def correct(self) -> bool | None:
        if self.outcome is VerificationOutcome.CORRECT:
            return True
        if self.outcome is VerificationOutcome.INCORRECT:
            return False
        return None


def _unwrap_answer(text: str) -> str:
    value = _QWEN_TERMINAL_TEXT.sub("", text.strip()).strip()
    answer = _ANSWER_TAG.fullmatch(value)
    if answer is not None:
        value = answer.group(1).strip()
    boxed = _BOXED.fullmatch(value)
    if boxed is not None:
        value = boxed.group(1).strip()
    return value


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", _unwrap_answer(text).casefold()).strip()


def _reference_answers(expected_answer: Any) -> tuple[str, ...]:
    if isinstance(expected_answer, str) and expected_answer.strip():
        return (expected_answer,)
    if isinstance(expected_answer, Sequence) and not isinstance(
        expected_answer, (str, bytes)
    ):
        values = tuple(expected_answer)
        if values and all(isinstance(value, str) and value.strip() for value in values):
            return values
    raise ValueError("expected_answer must be a non-empty string or string list")


def _parse_number(value: str) -> Fraction | None:
    compact = _unwrap_answer(value).strip().replace(",", "")
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    latex = _LATEX_FRACTION.fullmatch(compact)
    try:
        if latex is not None:
            denominator = int(latex.group(2))
            if denominator == 0:
                return None
            result = Fraction(int(latex.group(1)), denominator)
        elif "/" in compact and compact.count("/") == 1:
            numerator, denominator = compact.split("/", 1)
            result = Fraction(int(numerator.strip()), int(denominator.strip()))
        else:
            result = Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def _arxiv_letter(value: str) -> str | None:
    text = re.sub(r"[*_`]", "", _unwrap_answer(value))
    canonical = _ARXIV_CANONICAL.fullmatch(text)
    if canonical is not None:
        return next(group.upper() for group in canonical.groups() if group is not None)
    markers = tuple(_ARXIV_ANSWER_MARKER.finditer(text))
    if markers:
        return markers[-1].group(1).upper()
    named = tuple(_ARXIV_NAMED_OPTION.finditer(text))
    if named:
        return named[-1].group(1).upper()
    return None


def verify_arxivqa_answer(
    candidate_answer: str | None, expected_answer: Any, *, option_count: int
) -> DeterministicVerification:
    """Verify one ArxivQA answer inside that row's canonical A--Z range."""

    count = _required_int(
        option_count, field_name="option_count", minimum=1, maximum=26
    )
    references = _reference_answers(expected_answer)
    if len(references) != 1:
        raise ValueError("ArxivQA requires one canonical ground-truth label")
    expected = _arxiv_letter(references[0])
    upper_bound = chr(ord("A") + count - 1)
    if expected is None or not "A" <= expected <= upper_bound:
        raise ValueError("ArxivQA ground truth is outside the row option range")
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "arxivqa_missing_final_answer",
            f"expected={expected}; range=A-{upper_bound}",
        )
    candidate = _arxiv_letter(candidate_answer)
    if candidate is None or not "A" <= candidate <= upper_bound:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "arxivqa_rule",
            f"candidate=unparsed_or_out_of_range; expected={expected}; range=A-{upper_bound}",
        )
    correct = candidate == expected
    return DeterministicVerification(
        VerificationOutcome.CORRECT if correct else VerificationOutcome.INCORRECT,
        "arxivqa_rule",
        f"candidate={candidate}; expected={expected}; range=A-{upper_bound}",
    )


def verify_thinklite_answer(
    candidate_answer: str | None, expected_answer: Any
) -> DeterministicVerification:
    references = _reference_answers(expected_answer)
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "thinklite_missing_final_answer",
            "no non-empty suffix follows the last </think>",
        )
    normalized_candidate = _normalize_answer(candidate_answer)
    normalized_references = tuple(_normalize_answer(value) for value in references)
    if normalized_candidate in normalized_references:
        return DeterministicVerification(
            VerificationOutcome.CORRECT,
            "thinklite_normalized_exact",
            "normalized answer matches a reference",
        )
    candidate_number = _parse_number(candidate_answer)
    reference_numbers = tuple(_parse_number(value) for value in references)
    if candidate_number is not None and all(
        value is not None for value in reference_numbers
    ):
        correct = candidate_number in reference_numbers
        return DeterministicVerification(
            VerificationOutcome.CORRECT if correct else VerificationOutcome.INCORRECT,
            "thinklite_numeric",
            f"numeric_equivalence={correct}",
        )
    return DeterministicVerification(
        VerificationOutcome.SEMANTIC_REQUIRED,
        "thinklite_semantic_required",
        "deterministic exact/numeric rules are inconclusive",
    )


def verify_vstar_answer(
    candidate_answer: str | None, expected_answer: Any
) -> DeterministicVerification:
    references = _reference_answers(expected_answer)
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "vstar_missing_final_answer",
            "no non-empty suffix follows the last </think>",
        )
    candidate = _normalize_answer(candidate_answer)
    if candidate in {_normalize_answer(value) for value in references}:
        return DeterministicVerification(
            VerificationOutcome.CORRECT,
            "vstar_normalized_exact",
            "normalized answer matches a reference",
        )
    return DeterministicVerification(
        VerificationOutcome.SEMANTIC_REQUIRED,
        "vstar_semantic_required",
        "deterministic exact rule is inconclusive",
    )


def verify_t1_answer(
    *,
    source: SelectionSource | str,
    candidate_answer: str | None,
    expected_answer: Any,
    option_count: int | None = None,
) -> DeterministicVerification:
    try:
        normalized_source = SelectionSource(source)
    except ValueError as exc:
        raise ValueError("source is unsupported") from exc
    if normalized_source is SelectionSource.ARXIVQA:
        if option_count is None:
            raise ValueError("ArxivQA verification requires option_count")
        return verify_arxivqa_answer(
            candidate_answer, expected_answer, option_count=option_count
        )
    if option_count is not None:
        raise ValueError("option_count is only valid for ArxivQA")
    if normalized_source is SelectionSource.THINKLITE:
        return verify_thinklite_answer(candidate_answer, expected_answer)
    return verify_vstar_answer(candidate_answer, expected_answer)


def evidence_to_attempt_record(
    evidence: T1RawGenerationEvidence | Mapping[str, Any],
    *,
    expected_answer: Any,
    option_count: int | None = None,
    budget_exhausted: bool = False,
    semantic_verdict: bool | None = None,
    semantic_judge_evidence_sha256: str | None = None,
    answer_parser: str = T1_THINKING_ANSWER_PARSER,
) -> dict[str, Any] | None:
    """Convert raw evidence to the reducer schema without losing uncertainty.

    A non-final length finish returns ``None`` so the caller schedules the next
    budget revision.  A semantic-required rule result becomes verifier_error
    until identified judge evidence is supplied.
    """

    raw = (
        evidence
        if isinstance(evidence, T1RawGenerationEvidence)
        else T1RawGenerationEvidence.from_record(evidence)
    )
    if type(budget_exhausted) is not bool:
        raise TypeError("budget_exhausted must be boolean")
    if semantic_verdict is not None and type(semantic_verdict) is not bool:
        raise TypeError("semantic_verdict must be boolean or None")
    if semantic_judge_evidence_sha256 is not None:
        _required_sha256(
            semantic_judge_evidence_sha256,
            field_name="semantic_judge_evidence_sha256",
        )

    base: dict[str, Any] = {
        "schema_version": POLICY_SELECTION_ATTEMPT_SCHEMA,
        "request_id": raw.request_id,
        "sample_id": raw.sample_id,
        "candidate_sha256": raw.candidate_sha256,
        "source": raw.source.value,
        "branch": SelectionBranch.FULL_IMAGE.value,
        "attempt_index": raw.attempt_index,
        "run_id": raw.run_id,
        "run_manifest_sha256": raw.run_manifest_sha256,
        "raw_generation_sha256": raw.evidence_sha256,
        "budget_revision": raw.budget_revision,
    }
    if raw.disposition is GenerationDisposition.TRUNCATED:
        if not budget_exhausted:
            return None
        return {
            **base,
            "status": AttemptStatus.TRUNCATED.value,
            "correct": None,
            "answer": None,
            "verification_route": "response_budgets_exhausted",
        }
    if raw.disposition is GenerationDisposition.GENERATION_ERROR:
        return {
            **base,
            "status": AttemptStatus.GENERATION_ERROR.value,
            "correct": None,
            "answer": None,
            "verification_route": "generation_error",
            "generation_error": raw.generation_error,
        }

    answer = parse_t1_answer(raw.raw_text, answer_parser=answer_parser)
    verification = verify_t1_answer(
        source=raw.source,
        candidate_answer=answer,
        expected_answer=expected_answer,
        option_count=option_count,
    )
    if verification.outcome is VerificationOutcome.SEMANTIC_REQUIRED:
        if semantic_verdict is None:
            return {
                **base,
                "status": AttemptStatus.VERIFIER_ERROR.value,
                "correct": None,
                "answer": answer,
                "verification_route": verification.route,
                "verification_evidence": verification.evidence,
                "semantic_required": True,
            }
        if semantic_judge_evidence_sha256 is None:
            raise ValueError(
                "a semantic verdict requires semantic_judge_evidence_sha256"
            )
        correct = semantic_verdict
        route = "local_qwen25_72b_semantic_judge"
    else:
        if semantic_verdict is not None or semantic_judge_evidence_sha256 is not None:
            raise ValueError("semantic judge evidence is forbidden for a rule decision")
        assert verification.correct is not None
        correct = verification.correct
        route = verification.route
    return {
        **base,
        "status": AttemptStatus.SCORED.value,
        "correct": correct,
        "answer": answer,
        "verification_route": route,
        "verification_evidence": verification.evidence,
        "semantic_judge_evidence_sha256": semantic_judge_evidence_sha256,
    }


@dataclass(frozen=True, slots=True)
class T1ChunkManifest:
    run_manifest_sha256: str
    shard_rank: int
    world_size: int
    chunk_index: int
    record_count: int
    evidence_file: Path
    evidence_sha256: str
    logical_keys_sha256: str
    manifest_sha256: str
    _record_bytes: bytes

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "T1ChunkManifest":
        value = _mapping(record, field_name="chunk manifest")
        _exact_fields(value, _CHUNK_FIELDS, field_name="chunk manifest")
        if value["schema_version"] != T1_CHUNK_MANIFEST_SCHEMA:
            raise ValueError(
                f"chunk manifest schema_version must be {T1_CHUNK_MANIFEST_SCHEMA!r}"
            )
        run_sha = _required_sha256(
            value["run_manifest_sha256"], field_name="run_manifest_sha256"
        )
        world_size = _required_int(
            value["world_size"], field_name="world_size", minimum=1
        )
        shard_rank = _required_int(
            value["shard_rank"],
            field_name="shard_rank",
            maximum=world_size - 1,
        )
        chunk_index = _required_int(value["chunk_index"], field_name="chunk_index")
        record_count = _required_int(
            value["record_count"], field_name="record_count", minimum=1
        )
        evidence_file = _safe_relative_path(
            value["evidence_file"], field_name="evidence_file"
        )
        evidence_sha = _required_sha256(
            value["evidence_sha256"], field_name="evidence_sha256"
        )
        expected_file = Path("chunks") / f"{evidence_sha}.jsonl"
        if evidence_file != expected_file:
            raise ValueError("evidence_file is not named by evidence_sha256")
        logical_sha = _required_sha256(
            value["logical_keys_sha256"], field_name="logical_keys_sha256"
        )
        manifest_sha = _required_sha256(
            value["manifest_sha256"], field_name="manifest_sha256"
        )
        identity = dict(_json_clone(value))
        del identity["manifest_sha256"]
        if _sha256_json(identity) != manifest_sha:
            raise ValueError("chunk manifest SHA-256 mismatch")
        normalized = _json_clone(value)
        return cls(
            run_manifest_sha256=run_sha,
            shard_rank=shard_rank,
            world_size=world_size,
            chunk_index=chunk_index,
            record_count=record_count,
            evidence_file=evidence_file,
            evidence_sha256=evidence_sha,
            logical_keys_sha256=logical_sha,
            manifest_sha256=manifest_sha,
            _record_bytes=_canonical_json_bytes(normalized),
        )

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)


def _logical_evidence_key(evidence: T1RawGenerationEvidence) -> tuple[Any, ...]:
    return (
        evidence.candidate_sha256,
        evidence.attempt_index,
        evidence.budget_revision,
    )


def _logical_keys_sha256(evidences: Sequence[T1RawGenerationEvidence]) -> str:
    return _sha256_json(
        {
            "schema": "tgvf.policy-selection.t1-chunk-logical-keys.v1",
            "keys": [
                {
                    "candidate_sha256": evidence.candidate_sha256,
                    "attempt_index": evidence.attempt_index,
                    "budget_revision": evidence.budget_revision,
                }
                for evidence in evidences
            ],
        }
    )


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
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"existing immutable artifact differs: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_chunk_manifest(
    path: str | Path, manifest: T1ChunkManifest | Mapping[str, Any]
) -> T1ChunkManifest:
    parsed = (
        manifest
        if isinstance(manifest, T1ChunkManifest)
        else T1ChunkManifest.from_record(manifest)
    )
    _atomic_write_immutable(Path(path), parsed._record_bytes + b"\n")
    return parsed


def write_content_addressed_chunk(
    output_root: str | Path,
    records: Sequence[T1RawGenerationEvidence | Mapping[str, Any]],
    *,
    run: T1RunConfig,
    shard_rank: int,
    chunk_index: int,
) -> T1ChunkManifest:
    """Atomically publish one immutable evidence chunk and logical manifest."""

    root = Path(output_root)
    if root.absolute() != run.output_root:
        raise ValueError("chunk output_root differs from the run configuration")
    rank = _required_int(
        shard_rank,
        field_name="shard_rank",
        maximum=int(run.runtime["world_size"]) - 1,
    )
    index = _required_int(chunk_index, field_name="chunk_index")
    evidences = [
        item
        if isinstance(item, T1RawGenerationEvidence)
        else T1RawGenerationEvidence.from_record(item)
        for item in records
    ]
    if not evidences:
        raise ValueError("a chunk must contain at least one evidence record")
    for evidence in evidences:
        evidence.validate_against_run(run)
        if (
            candidate_rank(
                evidence.candidate_sha256, world_size=int(run.runtime["world_size"])
            )
            != rank
        ):
            raise ValueError("evidence candidate belongs to a different shard")
    evidences.sort(key=_logical_evidence_key)
    logical_keys = [_logical_evidence_key(evidence) for evidence in evidences]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("chunk contains duplicate logical evidence keys")
    evidence_bytes = b"".join(
        _canonical_json_bytes(evidence.as_record()) + b"\n" for evidence in evidences
    )
    evidence_sha = _sha256_bytes(evidence_bytes)
    evidence_relative = Path("chunks") / f"{evidence_sha}.jsonl"
    _atomic_write_immutable(root / evidence_relative, evidence_bytes)
    identity = {
        "schema_version": T1_CHUNK_MANIFEST_SCHEMA,
        "run_manifest_sha256": run.manifest_sha256,
        "shard_rank": rank,
        "world_size": int(run.runtime["world_size"]),
        "chunk_index": index,
        "record_count": len(evidences),
        "evidence_file": evidence_relative.as_posix(),
        "evidence_sha256": evidence_sha,
        "logical_keys_sha256": _logical_keys_sha256(evidences),
    }
    manifest_record = {**identity, "manifest_sha256": _sha256_json(identity)}
    manifest = T1ChunkManifest.from_record(manifest_record)
    manifest_path = root / "manifests" / f"rank-{rank:02d}-chunk-{index:06d}.json"
    atomic_write_chunk_manifest(manifest_path, manifest)
    return manifest


def validate_chunk_manifest(
    manifest: T1ChunkManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    run: T1RunConfig,
    expected_rank: int | None = None,
    expected_chunk_index: int | None = None,
) -> T1ChunkManifest:
    """Validate a manifest, its canonical JSONL, and every raw record."""

    parsed = (
        manifest
        if isinstance(manifest, T1ChunkManifest)
        else T1ChunkManifest.from_record(manifest)
    )
    root = Path(output_root)
    if root.absolute() != run.output_root:
        raise ValueError("chunk output_root differs from the run configuration")
    if parsed.run_manifest_sha256 != run.manifest_sha256:
        raise ValueError("chunk run identity mismatch")
    if parsed.world_size != run.runtime["world_size"]:
        raise ValueError("chunk world_size mismatch")
    if expected_rank is not None and parsed.shard_rank != expected_rank:
        raise ValueError("chunk shard rank mismatch")
    if expected_chunk_index is not None and parsed.chunk_index != expected_chunk_index:
        raise ValueError("chunk index mismatch")
    evidence_path = root / parsed.evidence_file
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    payload = evidence_path.read_bytes()
    if _sha256_bytes(payload) != parsed.evidence_sha256:
        raise ValueError("chunk evidence SHA-256 mismatch")
    lines = payload.splitlines()
    if len(lines) != parsed.record_count or any(not line for line in lines):
        raise ValueError("chunk evidence record count mismatch")
    evidences: list[T1RawGenerationEvidence] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {constant}")
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid evidence JSON at line {index + 1}") from exc
        evidence = T1RawGenerationEvidence.from_record(record)
        evidence.validate_against_run(run)
        if (
            candidate_rank(evidence.candidate_sha256, world_size=parsed.world_size)
            != parsed.shard_rank
        ):
            raise ValueError("chunk contains an evidence record for another shard")
        if _canonical_json_bytes(record) != line:
            raise ValueError("chunk evidence JSONL is not canonical")
        evidences.append(evidence)
    if evidences != sorted(evidences, key=_logical_evidence_key):
        raise ValueError("chunk evidence records are not in canonical logical order")
    logical_keys = [_logical_evidence_key(evidence) for evidence in evidences]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("chunk contains duplicate logical evidence keys")
    if _logical_keys_sha256(evidences) != parsed.logical_keys_sha256:
        raise ValueError("chunk logical-key SHA-256 mismatch")
    return parsed


def load_resumable_chunk(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    run: T1RunConfig,
    expected_rank: int,
    expected_chunk_index: int,
) -> T1ChunkManifest | None:
    """Return an already complete chunk, or ``None`` if it was never committed."""

    path = Path(manifest_path)
    if not path.exists():
        return None
    record = _load_json_object(path)
    return validate_chunk_manifest(
        record,
        output_root=output_root,
        run=run,
        expected_rank=expected_rank,
        expected_chunk_index=expected_chunk_index,
    )


__all__ = [
    "DeterministicVerification",
    "GenerationDisposition",
    "T1_ATTEMPTS",
    "T1_ATTEMPT_SEED_SCHEMA",
    "T1_CHUNK_MANIFEST_SCHEMA",
    "T1_MAX_PIXELS",
    "T1_INSTRUCT_ANSWER_PARSER",
    "T1_MODEL_PATH_BY_REPOSITORY",
    "T1_PROMPT_SCHEMA",
    "T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA",
    "T1_RAW_GENERATION_SCHEMA",
    "T1_RUN_CONFIG_SCHEMA",
    "T1_SHARD_COUNT",
    "T1_SOURCE_RGB_SCHEMA",
    "T1_THINKING_ANSWER_PARSER",
    "T1ChunkManifest",
    "T1DataSource",
    "T1RawGenerationEvidence",
    "T1ResponseBudget",
    "T1RunConfig",
    "VerificationOutcome",
    "atomic_write_chunk_manifest",
    "candidate_rank",
    "classify_generation_finish",
    "derive_t1_attempt_seed",
    "evidence_to_attempt_record",
    "extract_final_answer",
    "extract_direct_completion",
    "parse_t1_answer",
    "load_resumable_chunk",
    "load_t1_run_config",
    "native_prompt_identity_sha256",
    "native_user_message_descriptor",
    "rendered_prompt_token_ids_sha256",
    "sampled_token_ids_sha256",
    "source_rgb_sha256",
    "validate_chunk_manifest",
    "verify_arxivqa_answer",
    "verify_t1_answer",
    "verify_thinklite_answer",
    "verify_vstar_answer",
    "write_content_addressed_chunk",
]
