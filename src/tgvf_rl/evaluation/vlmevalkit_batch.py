"""Deterministic batched inference bridge for the pinned VLMEvalKit Qwen3 path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any


_BATCH_CONFIG_ATTR = "_tgvf_coredev_batch_inference_config"
_BATCH_OPTION_NAMES = (
    "inference_batch_size",
    "request_seed_base",
    "request_seed_namespace",
)
_LOW_31_BITS = (1 << 31) - 1


@dataclass(frozen=True, slots=True)
class CoreDevBatchInferenceConfig:
    """Frozen request-batching identity attached to one local model instance."""

    inference_batch_size: int
    request_seed_base: int
    request_seed_namespace: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.inference_batch_size, bool)
            or not isinstance(self.inference_batch_size, int)
            or self.inference_batch_size <= 0
        ):
            raise ValueError("inference_batch_size must be a positive integer")
        if (
            isinstance(self.request_seed_base, bool)
            or not isinstance(self.request_seed_base, int)
            or not 0 <= self.request_seed_base <= _LOW_31_BITS
        ):
            raise ValueError("request_seed_base must be a low-31-bit integer")
        if (
            not isinstance(self.request_seed_namespace, str)
            or not self.request_seed_namespace
            or self.request_seed_namespace.strip() != self.request_seed_namespace
        ):
            raise ValueError("request_seed_namespace must be a non-empty canonical string")


def stable_coredev_request_seed(
    *,
    seed_namespace: str,
    seed_base: int,
    dataset_name: str,
    canonical_index: object,
) -> int:
    """Return the low 31 bits of the accepted canonical row-identity hash."""

    config = CoreDevBatchInferenceConfig(
        inference_batch_size=1,
        request_seed_base=seed_base,
        request_seed_namespace=seed_namespace,
    )
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    canonical_json = json.dumps(
        [
            config.request_seed_namespace,
            config.request_seed_base,
            dataset_name,
            str(canonical_index),
        ],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return int.from_bytes(sha256(canonical_json.encode("utf-8")).digest(), "big") & _LOW_31_BITS


def _validate_batch_model(model: Any) -> None:
    if getattr(model, "use_vllm", None) is not True:
        raise RuntimeError("CoreDev batched inference requires use_vllm=true")
    if getattr(model, "post_process", None) is not False:
        raise RuntimeError("CoreDev batched inference requires post_process=false")
    if not callable(getattr(model, "generate", None)):
        raise RuntimeError("CoreDev batched inference requires model.generate")
    if not callable(getattr(getattr(model, "llm", None), "generate", None)):
        raise RuntimeError("CoreDev batched inference requires model.llm.generate")


def attach_coredev_batch_options_from_factory_kwargs(factory: Any) -> Any:
    """Pop the accepted batch fields and attach their frozen identity to a model."""

    @wraps(factory)
    def configured_factory(*args: Any, **kwargs: Any) -> Any:
        present = {name for name in _BATCH_OPTION_NAMES if name in kwargs}
        if not present:
            return factory(*args, **kwargs)
        if present != set(_BATCH_OPTION_NAMES):
            missing = ", ".join(sorted(set(_BATCH_OPTION_NAMES) - present))
            raise ValueError(f"incomplete CoreDev batch configuration; missing: {missing}")
        config = CoreDevBatchInferenceConfig(
            inference_batch_size=kwargs.pop("inference_batch_size"),
            request_seed_base=kwargs.pop("request_seed_base"),
            request_seed_namespace=kwargs.pop("request_seed_namespace"),
        )
        model = factory(*args, **kwargs)
        _validate_batch_model(model)
        if hasattr(model, _BATCH_CONFIG_ATTR):
            raise RuntimeError("CoreDev batch configuration is already attached")
        setattr(model, _BATCH_CONFIG_ATTR, config)
        return model

    return configured_factory


def _capture_vllm_requests(
    *,
    model: Any,
    messages: Sequence[Any],
    dataset_name: str,
) -> tuple[list[Any], list[Any], Callable[..., Any]]:
    """Run the pinned scalar preparation path while intercepting its engine call."""

    _validate_batch_model(model)
    real_generate = model.llm.generate
    requests: list[Any] = []
    captured_params: list[Any] = []

    def capture_generate(
        prompts: Any,
        *args: Any,
        sampling_params: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        if args or kwargs:
            raise RuntimeError("pinned vLLM request call contract drifted")
        if (
            not isinstance(prompts, Sequence)
            or isinstance(prompts, (str, bytes))
            or len(prompts) != 1
            or not isinstance(prompts[0], Mapping)
        ):
            raise RuntimeError("pinned Qwen wrapper must submit exactly one request during capture")
        if sampling_params is None or not hasattr(sampling_params, "seed"):
            raise RuntimeError("captured SamplingParams has no seed field")
        requests.append(prompts[0])
        captured_params.append(sampling_params)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="")])]

    replaced = False
    try:
        model.llm.generate = capture_generate
        replaced = True
        for message in messages:
            before = len(requests)
            model.generate(message=message, dataset=dataset_name)
            if len(requests) != before + 1:
                raise RuntimeError("one row must produce exactly one captured vLLM request")
    finally:
        if replaced:
            model.llm.generate = real_generate

    if len(requests) != len(messages) or len(captured_params) != len(messages):
        raise RuntimeError("captured request count does not match row count")
    if len({id(item) for item in requests}) != len(requests):
        raise RuntimeError("captured vLLM requests must be distinct objects")
    if len({id(item) for item in captured_params}) != len(captured_params):
        raise RuntimeError("captured SamplingParams must be distinct objects")
    return requests, captured_params, real_generate


def _generate_batch(
    *,
    model: Any,
    messages: Sequence[Any],
    canonical_indices: Sequence[object],
    dataset_name: str,
    config: CoreDevBatchInferenceConfig,
) -> list[str]:
    if not messages or len(messages) != len(canonical_indices):
        raise ValueError("messages and canonical_indices must be non-empty and aligned")
    if len(messages) > config.inference_batch_size:
        raise ValueError("physical batch exceeds the frozen inference_batch_size")
    requests, params, real_generate = _capture_vllm_requests(
        model=model,
        messages=messages,
        dataset_name=dataset_name,
    )
    for index, sampling in zip(canonical_indices, params, strict=True):
        if sampling.seed is not None:
            raise RuntimeError("pinned scalar SamplingParams unexpectedly carried a request seed")
        sampling.seed = stable_coredev_request_seed(
            seed_namespace=config.request_seed_namespace,
            seed_base=config.request_seed_base,
            dataset_name=dataset_name,
            canonical_index=index,
        )
    outputs = real_generate(requests, sampling_params=params, use_tqdm=False)
    if not isinstance(outputs, Sequence) or len(outputs) != len(requests):
        raise RuntimeError("vLLM batched output count or type drifted")
    texts: list[str] = []
    for output in outputs:
        candidates = getattr(output, "outputs", None)
        if not isinstance(candidates, Sequence) or len(candidates) < 1:
            raise RuntimeError("vLLM request output contract drifted")
        text = getattr(candidates[0], "text", None)
        if not isinstance(text, str):
            raise RuntimeError("vLLM generated text must be a string")
        texts.append(text)
    return texts


def _atomic_dump(inference_module: Any, data: Any, destination: str) -> None:
    path = Path(destination)
    if path.suffix != ".pkl":
        raise RuntimeError("CoreDev auxiliary inference checkpoint must be a pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.pkl", dir=path.parent
    )
    os.close(descriptor)
    try:
        inference_module.dump(data, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_coredev_batched_inference(inference_module: Any) -> Any:
    """Install a narrow batched replacement for pinned ``infer_data``."""

    original = inference_module.infer_data

    @wraps(original)
    def batched_infer_data(
        model: Any,
        model_name: str,
        work_dir: str,
        dataset: Any,
        out_file: str,
        verbose: bool = False,
        api_nproc: int = 4,
        use_vllm: bool = False,
        retry_failed: bool = True,
    ) -> Any:
        config = getattr(model, _BATCH_CONFIG_ATTR, None) if not isinstance(model, str) else None
        if config is None:
            return original(
                model=model,
                model_name=model_name,
                work_dir=work_dir,
                dataset=dataset,
                out_file=out_file,
                verbose=verbose,
                api_nproc=api_nproc,
                use_vllm=use_vllm,
                retry_failed=retry_failed,
            )
        if not isinstance(config, CoreDevBatchInferenceConfig):
            raise RuntimeError("invalid attached CoreDev batch configuration")
        _validate_batch_model(model)
        rank, world_size = inference_module.get_rank_and_world_size()
        if rank != 0 or world_size != 1:
            raise RuntimeError("CoreDev batched inference requires world size 1")
        if getattr(model, "is_api", False):
            raise RuntimeError("CoreDev batched inference requires a local model")
        if os.environ.get("SKIP_ERR") == "1":
            raise RuntimeError("SKIP_ERR is incompatible with exact batched inference")

        dataset_name = dataset.dataset_name
        prev_file = f"{work_dir}/{model_name}_{dataset_name}_PREV.pkl"
        res = inference_module.load(prev_file) if inference_module.osp.exists(prev_file) else {}
        if inference_module.osp.exists(out_file):
            res.update(inference_module.load(out_file))

        data = dataset.data
        data_indices = list(data["index"])
        if all(index in res for index in data_indices):
            _atomic_dump(inference_module, {index: res[index] for index in data_indices}, out_file)
            return model

        data = data[~data["index"].isin(res)]
        model.set_dump_image(dataset.dump_image)
        progress = inference_module.tqdm(
            total=len(data),
            desc=f"Infer {model_name}/{dataset_name}, Rank 0/1 (batch={config.inference_batch_size})",
        )
        try:
            for start in range(0, len(data), config.inference_batch_size):
                batch = data.iloc[start : start + config.inference_batch_size]
                indices = list(batch["index"])
                messages = []
                for offset in range(len(batch)):
                    row = batch.iloc[offset]
                    if getattr(dataset, "force_use_dataset_prompt", False):
                        struct = dataset.build_prompt(row)
                    elif hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
                        struct = model.build_prompt(row, dataset=dataset_name)
                    else:
                        struct = dataset.build_prompt(row)
                    messages.append(struct)
                responses = _generate_batch(
                    model=model,
                    messages=messages,
                    canonical_indices=indices,
                    dataset_name=dataset_name,
                    config=config,
                )
                for index, response in zip(indices, responses, strict=True):
                    res[index] = response
                    if verbose:
                        print(response, flush=True)
                _atomic_dump(inference_module, res, out_file)
                progress.update(len(indices))
        finally:
            progress.close()

        ordered = {index: res[index] for index in data_indices}
        _atomic_dump(inference_module, ordered, out_file)
        return model

    inference_module.infer_data = batched_infer_data
    return original


__all__ = [
    "CoreDevBatchInferenceConfig",
    "attach_coredev_batch_options_from_factory_kwargs",
    "install_coredev_batched_inference",
    "stable_coredev_request_seed",
]
