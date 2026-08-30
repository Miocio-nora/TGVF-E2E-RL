"""Strict, CPU-only loader for T1 policy-selection run configuration."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import re

from tgvf_rl.public_api_compat import rebind_public_function

from .policy_selection import SelectionCandidate, SelectionSource
from .policy_selection_config_schema import (
    T1_INSTRUCT_ANSWER_PARSER as T1_INSTRUCT_ANSWER_PARSER,
    T1_MAX_PIXELS as T1_MAX_PIXELS,
    T1_MODEL_IDENTITY_SCHEMA as T1_MODEL_IDENTITY_SCHEMA,
    T1_MODEL_PATH_BY_REPOSITORY as T1_MODEL_PATH_BY_REPOSITORY,
    T1_PROCESSOR_IDENTITY_SCHEMA as T1_PROCESSOR_IDENTITY_SCHEMA,
    T1_PROMPT_IDENTITY_SCHEMA as T1_PROMPT_IDENTITY_SCHEMA,
    T1_PROMPT_SCHEMA as T1_PROMPT_SCHEMA,
    T1_RUN_CONFIG_SCHEMA as T1_RUN_CONFIG_SCHEMA,
    T1_RUNTIME_IDENTITY_SCHEMA as T1_RUNTIME_IDENTITY_SCHEMA,
    T1_SOURCE_RGB_SCHEMA as T1_SOURCE_RGB_SCHEMA,
    T1_THINKING_ANSWER_PARSER as T1_THINKING_ANSWER_PARSER,
    T1DataSource as T1DataSource,
    T1ResponseBudget as T1ResponseBudget,
    T1RunConfig as T1RunConfig,
)
from .policy_selection_config_values import (
    T1_ATTEMPTS as T1_ATTEMPTS,
    T1_ATTEMPT_SEED_SCHEMA as T1_ATTEMPT_SEED_SCHEMA,
    T1_SEED_MODULUS as T1_SEED_MODULUS,
    T1_SHARD_COUNT as T1_SHARD_COUNT,
    _absolute_normal_path as _absolute_normal_path,
    _canonical_json_bytes as _canonical_json_bytes,
    _exact_fields as _exact_fields,
    _freeze_json as _freeze_json,
    _json_clone as _json_clone,
    _load_json_object as _load_json_object,
    _mapping as _mapping,
    _reject_duplicate_keys as _reject_duplicate_keys,
    _required_bool as _required_bool,
    _required_float as _required_float,
    _required_int as _required_int,
    _required_sha256 as _required_sha256,
    _required_string as _required_string,
    _safe_relative_path as _safe_relative_path,
    _sha256_bytes as _sha256_bytes,
    _sha256_file as _sha256_file,
    _sha256_json as _sha256_json,
    candidate_rank as candidate_rank,
    derive_t1_attempt_seed as derive_t1_attempt_seed,
)
from .policy_selection_recommended import (
    T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION,
    T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA,
    T1_RECOMMENDED_SELECTION_NAMESPACE,
    T1_RECOMMENDED_SELECTION_ROWS,
    T1_RECOMMENDED_SOURCE_QUOTAS,
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

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
rebind_public_function(
    load_t1_run_config,
    implementation_module=__name__,
    public_module=_PUBLIC_RUNTIME_MODULE,
)

__all__ = [
    "T1_ATTEMPTS",
    "T1_ATTEMPT_SEED_SCHEMA",
    "T1_INSTRUCT_ANSWER_PARSER",
    "T1_MAX_PIXELS",
    "T1_MODEL_PATH_BY_REPOSITORY",
    "T1_PROMPT_SCHEMA",
    "T1_RUN_CONFIG_SCHEMA",
    "T1_SHARD_COUNT",
    "T1_SOURCE_RGB_SCHEMA",
    "T1_THINKING_ANSWER_PARSER",
    "T1DataSource",
    "T1ResponseBudget",
    "T1RunConfig",
    "candidate_rank",
    "derive_t1_attempt_seed",
    "load_t1_run_config",
]
