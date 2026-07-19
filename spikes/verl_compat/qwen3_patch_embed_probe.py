"""Bounded Qwen3-VL patch-embedding parity and performance probe.

The probe isolates the projection at the front of the accepted local Qwen3
vision tower.  It reads only the projection weight and bias from their declared
safetensors shard; it does not instantiate or load the full 8B model.

The synthetic input represents one already-patchified 512 by 512 image:
``image_grid_thw=[1, 32, 32]``, hence 1024 flattened patches with
``3 * 2 * 16 * 16 == 1536`` values each.  The two compared expressions are::

    conv3d(x.view(N, 3, 2, 16, 16), W, b).view(N, 1152)
    linear(x, W.view(1152, 1536), b)

GPU execution is deliberately fail-closed.  For the Torch 2.11 candidate on
physical GPU 3, the exact invocation is::

    CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
      .venv-torch211-cu129/bin/python \
      spikes/verl_compat/qwen3_patch_embed_probe.py \
      --run-id <ledger-run-id> --runtime candidate --physical-gpu 3 \
      --output artifacts/compatibility/<ledger-cell>.json

The command may be launched only after that exact output identity has a
complete ``PLANNED`` experiment-ledger entry.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys
from time import perf_counter_ns
from typing import Any

import torch
from torch.nn import functional as F

from tgvf_rl.experiment_identity import validate_run_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
MODEL_INDEX_NAME = "model.safetensors.index.json"
MODEL_CONFIG_NAME = "config.json"
PATCH_WEIGHT_KEY = "model.visual.patch_embed.proj.weight"
PATCH_BIAS_KEY = "model.visual.patch_embed.proj.bias"
RESULT_SCHEMA_VERSION = "qwen3-patch-embed-compatibility-v1"
ALLOWED_PHYSICAL_GPUS = frozenset({2, 3})
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REQUIRED_PYTHONHASHSEED = "0"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    torch_version: str
    torch_cuda_version: str


RUNTIME_IDENTITIES: Mapping[str, RuntimeIdentity] = {
    "control": RuntimeIdentity(
        torch_version="2.9.0+cu128",
        torch_cuda_version="12.8",
    ),
    "candidate": RuntimeIdentity(
        torch_version="2.11.0+cu129",
        torch_cuda_version="12.9",
    ),
}


@dataclass(frozen=True, slots=True)
class PatchEmbedGeometry:
    image_height: int = 512
    image_width: int = 512
    temporal_grid: int = 1
    height_grid: int = 32
    width_grid: int = 32
    in_channels: int = 3
    temporal_patch_size: int = 2
    patch_size: int = 16
    output_width: int = 1152

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.height_grid * self.patch_size != self.image_height:
            raise ValueError("height grid does not describe the declared image")
        if self.width_grid * self.patch_size != self.image_width:
            raise ValueError("width grid does not describe the declared image")

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (self.temporal_grid, self.height_grid, self.width_grid)

    @property
    def patch_count(self) -> int:
        return math.prod(self.grid_thw)

    @property
    def flattened_patch_width(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def flattened_input_shape(self) -> tuple[int, int]:
        return (self.patch_count, self.flattened_patch_width)

    @property
    def convolution_input_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.patch_count,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

    @property
    def weight_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.output_width,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.patch_count, self.output_width)


@dataclass(frozen=True, slots=True)
class ParityTolerance:
    absolute: float
    relative: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} tolerance must be a finite non-negative float"
                )


GEOMETRY = PatchEmbedGeometry()
SEED = 20260720
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10
PARITY_TOLERANCES: Mapping[str, ParityTolerance] = {
    "float32": ParityTolerance(absolute=1.0e-4, relative=1.0e-4),
    "bfloat16": ParityTolerance(absolute=1.5625e-2, relative=1.5625e-2),
}


def bounded_output_path(raw: Path) -> Path:
    """Resolve a new JSON result strictly below artifacts/compatibility."""

    path = raw if raw.is_absolute() else REPOSITORY_ROOT / raw
    path = path.resolve()
    allowed = (REPOSITORY_ROOT / "artifacts" / "compatibility").resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"output must be a child of {allowed}")
    if path.suffix != ".json":
        raise ValueError("patch-embed probe output must have a .json suffix")
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    return path


def validate_environment(
    *,
    environment: Mapping[str, str],
    physical_gpu: int,
    runtime_name: str,
    torch_version: str,
    torch_cuda_version: str | None,
) -> RuntimeIdentity:
    """Validate the predeclared runtime and single-GPU visibility contract."""

    if physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError("physical GPU must be one of the project-authorized devices")
    if environment.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the declared physical GPU"
        )
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != (REQUIRED_CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{REQUIRED_CUBLAS_WORKSPACE_CONFIG!r}"
        )
    if environment.get("PYTHONHASHSEED") != REQUIRED_PYTHONHASHSEED:
        raise RuntimeError("PYTHONHASHSEED must be exactly '0'")
    try:
        expected = RUNTIME_IDENTITIES[runtime_name]
    except KeyError as error:
        raise ValueError(f"unknown runtime identity: {runtime_name!r}") from error
    if torch_version != expected.torch_version:
        raise RuntimeError(
            f"torch version differs from {runtime_name!r} runtime identity"
        )
    if torch_cuda_version != expected.torch_cuda_version:
        raise RuntimeError(
            f"torch CUDA version differs from {runtime_name!r} runtime identity"
        )
    return expected


def validate_model_metadata(
    *,
    config: Mapping[str, Any],
    weight_map: Mapping[str, Any],
) -> Mapping[str, str]:
    """Validate the fixed model geometry and resolve exactly two tensor shards."""

    if config.get("model_type") != "qwen3_vl":
        raise ValueError("accepted model config must have model_type='qwen3_vl'")
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError("accepted model architecture identity changed")
    vision = config.get("vision_config")
    if not isinstance(vision, Mapping):
        raise TypeError("accepted model config is missing vision_config")
    expected_vision = {
        "in_channels": GEOMETRY.in_channels,
        "temporal_patch_size": GEOMETRY.temporal_patch_size,
        "patch_size": GEOMETRY.patch_size,
        "hidden_size": GEOMETRY.output_width,
    }
    actual_vision = {name: vision.get(name) for name in expected_vision}
    if actual_vision != expected_vision:
        raise ValueError("accepted Qwen3 patch-embedding geometry changed")

    resolved: dict[str, str] = {}
    for key in (PATCH_WEIGHT_KEY, PATCH_BIAS_KEY):
        shard = weight_map.get(key)
        if not isinstance(shard, str) or not shard.endswith(".safetensors"):
            raise ValueError(f"weight map does not resolve {key!r} to safetensors")
        resolved[key] = shard
    if len(set(resolved.values())) != 1:
        raise ValueError("patch-embedding weight and bias must share one shard")
    return resolved


def native_conv3d_patch_embed(
    flattened_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply the native full-patch Conv3D expression."""

    _validate_patch_tensors(flattened_input, weight, bias)
    convolution_input = flattened_input.reshape(
        flattened_input.shape[0], *weight.shape[1:]
    )
    output = F.conv3d(
        convolution_input,
        weight,
        bias,
        stride=tuple(int(value) for value in weight.shape[2:]),
    )
    return output.reshape(flattened_input.shape[0], weight.shape[0])


def linear_patch_embed(
    flattened_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply the mathematically equivalent reshape-plus-Linear expression."""

    _validate_patch_tensors(flattened_input, weight, bias)
    return F.linear(flattened_input, weight.reshape(weight.shape[0], -1), bias)


def parity_metrics(
    native: torch.Tensor,
    linear: torch.Tensor,
    *,
    tolerance: ParityTolerance,
) -> dict[str, float | bool]:
    """Return finite error metrics and the predeclared allclose decision."""

    if native.shape != linear.shape:
        raise ValueError("patch-embed parity tensors must have the same shape")
    if native.dtype != linear.dtype:
        raise ValueError("patch-embed parity tensors must have the same dtype")
    if native.numel() == 0:
        raise ValueError("patch-embed parity tensors cannot be empty")
    native_float = native.detach().to(dtype=torch.float32)
    linear_float = linear.detach().to(dtype=torch.float32)
    if not torch.isfinite(native_float).all() or not torch.isfinite(linear_float).all():
        raise ValueError("patch-embed parity tensors must be finite")
    absolute = torch.abs(native_float - linear_float)
    denominator = torch.maximum(
        torch.abs(linear_float),
        torch.tensor(1.0e-12, device=linear_float.device),
    )
    relative = absolute / denominator
    metrics: dict[str, float | bool] = {
        "passed": bool(
            torch.allclose(
                native,
                linear,
                atol=tolerance.absolute,
                rtol=tolerance.relative,
            )
        ),
        "max_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "root_mean_square_error": float(
            torch.sqrt(torch.mean(absolute.square())).item()
        ),
        "max_relative_error_with_1e-12_floor": float(relative.max().item()),
    }
    if any(
        not math.isfinite(value) for name, value in metrics.items() if name != "passed"
    ):
        raise ValueError("patch-embed parity metrics must be finite")
    return metrics


def _validate_patch_tensors(
    flattened_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    if not all(
        isinstance(tensor, torch.Tensor) for tensor in (flattened_input, weight, bias)
    ):
        raise TypeError("patch-embed inputs must be tensors")
    if flattened_input.ndim != 2 or weight.ndim != 5 or bias.ndim != 1:
        raise ValueError("patch-embed tensor ranks are invalid")
    if flattened_input.shape[1] != math.prod(weight.shape[1:]):
        raise ValueError("flattened input width differs from Conv3D kernel volume")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("patch-embed bias width differs from output width")
    if not (flattened_input.dtype == weight.dtype == bias.dtype):
        raise ValueError("patch-embed input, weight, and bias dtypes must match")
    if not flattened_input.device == weight.device == bias.device:
        raise ValueError("patch-embed input, weight, and bias devices must match")


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON mapping: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return value


def _load_projection_weights() -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "safetensors is required for the patch-embed probe"
        ) from error

    config_path = ACCEPTED_MODEL_PATH / MODEL_CONFIG_NAME
    index_path = ACCEPTED_MODEL_PATH / MODEL_INDEX_NAME
    config = _read_json_mapping(config_path)
    index = _read_json_mapping(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("model safetensors index is missing weight_map")
    shards = validate_model_metadata(config=config, weight_map=weight_map)
    model_root = ACCEPTED_MODEL_PATH.resolve()
    shard_path = (ACCEPTED_MODEL_PATH / shards[PATCH_WEIGHT_KEY]).resolve()
    if model_root not in shard_path.parents:
        raise ValueError("patch-embedding shard resolves outside the accepted model")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        available = frozenset(handle.keys())
        missing = {PATCH_WEIGHT_KEY, PATCH_BIAS_KEY} - available
        if missing:
            raise ValueError(
                f"patch-embedding shard is missing keys: {sorted(missing)}"
            )
        weight = handle.get_tensor(PATCH_WEIGHT_KEY)
        bias = handle.get_tensor(PATCH_BIAS_KEY)
    if tuple(weight.shape) != GEOMETRY.weight_shape:
        raise ValueError("loaded patch-embedding weight shape changed")
    if tuple(bias.shape) != (GEOMETRY.output_width,):
        raise ValueError("loaded patch-embedding bias shape changed")
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        raise ValueError("accepted patch-embedding tensors must be bfloat16")
    details = {
        "loader": "safetensors.safe_open/get_tensor",
        "loaded_tensor_count": 2,
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "index_path": str(index_path),
        "index_sha256": _file_sha256(index_path),
        "shard_path": str(shard_path),
        "keys": {
            PATCH_WEIGHT_KEY: {
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "storage_sha256": _tensor_storage_sha256(weight),
            },
            PATCH_BIAS_KEY: {
                "shape": list(bias.shape),
                "dtype": str(bias.dtype),
                "storage_sha256": _tensor_storage_sha256(bias),
            },
        },
    }
    return weight, bias, details


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_storage_sha256(tensor: torch.Tensor) -> str:
    storage_bytes = (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return sha256(storage_bytes).hexdigest()


def _synchronized_benchmark(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    output: torch.Tensor | None = None
    for _ in range(WARMUP_ITERATIONS):
        output = operation()
        torch.cuda.synchronize(device)

    elapsed_ms: list[float] = []
    for _ in range(TIMED_ITERATIONS):
        torch.cuda.synchronize(device)
        started = perf_counter_ns()
        output = operation()
        torch.cuda.synchronize(device)
        elapsed_ms.append((perf_counter_ns() - started) / 1_000_000)
    if output is None or any(
        not math.isfinite(value) or value <= 0 for value in elapsed_ms
    ):
        raise RuntimeError("synchronized patch-embed benchmark produced invalid timing")
    summary = {
        "synchronization": (
            "torch.cuda.synchronize before and after every timed iteration"
        ),
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "elapsed_ms": elapsed_ms,
        "mean_ms": mean(elapsed_ms),
        "median_ms": median(elapsed_ms),
        "minimum_ms": min(elapsed_ms),
        "maximum_ms": max(elapsed_ms),
    }
    return summary, output


def _configure_determinism() -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    return {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
    }


def _run_dtype(
    *,
    dtype: torch.dtype,
    dtype_name: str,
    flattened_input_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    bias_cpu: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    tolerance = PARITY_TOLERANCES[dtype_name]
    flattened_input = flattened_input_cpu.to(device=device, dtype=dtype)
    weight = weight_cpu.to(device=device, dtype=dtype)
    bias = bias_cpu.to(device=device, dtype=dtype)

    native_operation = lambda: native_conv3d_patch_embed(  # noqa: E731
        flattened_input, weight, bias
    )
    linear_operation = lambda: linear_patch_embed(  # noqa: E731
        flattened_input, weight, bias
    )
    with torch.inference_mode():
        native_parity = native_operation()
        linear_parity = linear_operation()
        torch.cuda.synchronize(device)
        parity = parity_metrics(
            native_parity,
            linear_parity,
            tolerance=tolerance,
        )
        native_timing, native_output = _synchronized_benchmark(
            native_operation,
            device=device,
        )
        linear_timing, linear_output = _synchronized_benchmark(
            linear_operation,
            device=device,
        )

    native_mean = float(native_timing["mean_ms"])
    linear_mean = float(linear_timing["mean_ms"])
    return {
        "dtype": str(dtype),
        "input_storage_sha256": _tensor_storage_sha256(flattened_input),
        "tolerance": asdict(tolerance),
        "parity": parity,
        "native_conv3d": {
            **native_timing,
            "output_storage_sha256": _tensor_storage_sha256(native_output),
        },
        "reshape_linear": {
            **linear_timing,
            "output_storage_sha256": _tensor_storage_sha256(linear_output),
        },
        "linear_speedup_over_native_mean": native_mean / linear_mean,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=_run_id_argument,
        required=True,
        help="explicit experiment identity; never inferred from --output",
    )
    parser.add_argument("--runtime", choices=tuple(RUNTIME_IDENTITIES), required=True)
    parser.add_argument(
        "--physical-gpu",
        choices=tuple(sorted(ALLOWED_PHYSICAL_GPUS)),
        type=int,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = bounded_output_path(args.output)
    expected_runtime = validate_environment(
        environment=os.environ,
        physical_gpu=args.physical_gpu,
        runtime_name=args.runtime,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("patch-embed probe requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible CUDA device must support bfloat16")
    logical_device = 0
    torch.cuda.set_device(logical_device)
    device = torch.device("cuda", logical_device)
    deterministic_settings = _configure_determinism()

    weight_cpu, bias_cpu, weight_details = _load_projection_weights()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    flattened_input_cpu = torch.randn(
        GEOMETRY.flattened_input_shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    dtype_results = {
        "float32": _run_dtype(
            dtype=torch.float32,
            dtype_name="float32",
            flattened_input_cpu=flattened_input_cpu,
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            device=device,
        ),
        "bfloat16": _run_dtype(
            dtype=torch.bfloat16,
            dtype_name="bfloat16",
            flattened_input_cpu=flattened_input_cpu,
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            device=device,
        ),
    }
    passed = all(bool(result["parity"]["passed"]) for result in dtype_results.values())
    properties = torch.cuda.get_device_properties(logical_device)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "result": "PASS" if passed else "FAIL",
        "scope": "isolated_qwen3_patch_embed_projection_only",
        "invocation": {
            "argv": list(sys.argv if argv is None else argv),
            "runtime_name": args.runtime,
            "physical_gpu": args.physical_gpu,
            "logical_device": logical_device,
            "required_environment": {
                "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
                "CUBLAS_WORKSPACE_CONFIG": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
                "PYTHONHASHSEED": REQUIRED_PYTHONHASHSEED,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "expected": asdict(expected_runtime),
            "cudnn_runtime": torch.backends.cudnn.version(),
            "cudnn_package": metadata.version("nvidia-cudnn-cu12"),
            "safetensors": metadata.version("safetensors"),
            "device": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            },
            "determinism": deterministic_settings,
        },
        "model": {
            "configured_path": str(ACCEPTED_MODEL_PATH),
            "resolved_path": str(ACCEPTED_MODEL_PATH.resolve()),
            "projection": weight_details,
        },
        "input": {
            "description": "one synthetic already-patchified 512x512 image",
            "seed": SEED,
            "generator": "torch.Generator(device='cpu').manual_seed then torch.randn(float32)",
            "float32_storage_sha256": _tensor_storage_sha256(flattened_input_cpu),
            "geometry": {
                **asdict(GEOMETRY),
                "image_grid_thw": list(GEOMETRY.grid_thw),
                "patch_count": GEOMETRY.patch_count,
                "flattened_patch_width": GEOMETRY.flattened_patch_width,
                "flattened_input_shape": list(GEOMETRY.flattened_input_shape),
                "convolution_input_shape": list(GEOMETRY.convolution_input_shape),
                "output_shape": list(GEOMETRY.output_shape),
            },
        },
        "results": dtype_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
