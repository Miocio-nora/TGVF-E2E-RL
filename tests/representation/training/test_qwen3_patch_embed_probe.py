from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = REPOSITORY_ROOT / "spikes/verl_compat/qwen3_patch_embed_probe.py"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tgvf_qwen3_patch_embed_probe_test",
        PROBE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Qwen3 patch-embed probe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_512_geometry_is_exact() -> None:
    probe = _load_probe()

    assert probe.GEOMETRY.grid_thw == (1, 32, 32)
    assert probe.GEOMETRY.patch_count == 1024
    assert probe.GEOMETRY.flattened_patch_width == 1536
    assert probe.GEOMETRY.flattened_input_shape == (1024, 1536)
    assert probe.GEOMETRY.weight_shape == (1152, 3, 2, 16, 16)
    assert probe.GEOMETRY.output_shape == (1024, 1152)


def test_conv3d_and_linear_helpers_are_mathematically_equivalent_on_cpu() -> None:
    probe = _load_probe()
    generator = torch.Generator(device="cpu").manual_seed(17)
    values = torch.randn(7, 3 * 2 * 2 * 2, generator=generator)
    weight = torch.randn(5, 3, 2, 2, 2, generator=generator)
    bias = torch.randn(5, generator=generator)

    native = probe.native_conv3d_patch_embed(values, weight, bias)
    linear = probe.linear_patch_embed(values, weight, bias)
    metrics = probe.parity_metrics(
        native,
        linear,
        tolerance=probe.ParityTolerance(absolute=1.0e-5, relative=1.0e-5),
    )

    assert native.shape == (7, 5)
    assert metrics["passed"] is True
    assert metrics["max_absolute_error"] <= 1.0e-5


def test_runtime_and_single_gpu_scope_are_fail_closed() -> None:
    probe = _load_probe()
    environment = {
        "CUDA_VISIBLE_DEVICES": "3",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }

    expected = probe.validate_environment(
        environment=environment,
        physical_gpu=3,
        runtime_name="candidate",
        torch_version="2.11.0+cu129",
        torch_cuda_version="12.9",
    )
    assert expected.torch_version == "2.11.0+cu129"

    with pytest.raises(RuntimeError, match="exactly the declared"):
        probe.validate_environment(
            environment={**environment, "CUDA_VISIBLE_DEVICES": "2,3"},
            physical_gpu=3,
            runtime_name="candidate",
            torch_version="2.11.0+cu129",
            torch_cuda_version="12.9",
        )
    with pytest.raises(RuntimeError, match="torch version differs"):
        probe.validate_environment(
            environment=environment,
            physical_gpu=3,
            runtime_name="candidate",
            torch_version="2.9.0+cu128",
            torch_cuda_version="12.8",
        )


def test_model_metadata_contract_resolves_only_projection_tensors() -> None:
    probe = _load_probe()
    config = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "vision_config": {
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "hidden_size": 1152,
        },
    }
    weight_map = {
        probe.PATCH_WEIGHT_KEY: "model-00004-of-00004.safetensors",
        probe.PATCH_BIAS_KEY: "model-00004-of-00004.safetensors",
        "unrelated": "model-00001-of-00004.safetensors",
    }

    resolved = probe.validate_model_metadata(config=config, weight_map=weight_map)

    assert resolved == {
        probe.PATCH_WEIGHT_KEY: "model-00004-of-00004.safetensors",
        probe.PATCH_BIAS_KEY: "model-00004-of-00004.safetensors",
    }
    with pytest.raises(ValueError, match="geometry changed"):
        probe.validate_model_metadata(
            config={
                **config,
                "vision_config": {**config["vision_config"], "patch_size": 14},
            },
            weight_map=weight_map,
        )


def test_output_path_is_bounded_json_and_non_overwriting(tmp_path: Path) -> None:
    probe = _load_probe()
    allowed = probe.REPOSITORY_ROOT / "artifacts" / "compatibility"

    assert (
        probe.bounded_output_path(
            Path("artifacts/compatibility/proposed-patch-probe.json")
        )
        == (allowed / "proposed-patch-probe.json").resolve()
    )
    with pytest.raises(ValueError, match="child"):
        probe.bounded_output_path(tmp_path / "outside.json")
    with pytest.raises(ValueError, match=".json suffix"):
        probe.bounded_output_path(
            Path("artifacts/compatibility/proposed-patch-probe.txt")
        )

    existing = allowed / "test-existing-patch-probe.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.touch()
    try:
        with pytest.raises(FileExistsError, match="already exists"):
            probe.bounded_output_path(existing)
    finally:
        existing.unlink()


def test_cli_requires_safe_explicit_run_id() -> None:
    probe = _load_probe()
    common = [
        "--runtime",
        "candidate",
        "--physical-gpu",
        "3",
        "--output",
        "artifacts/compatibility/proposed-patch-probe.json",
    ]
    with pytest.raises(SystemExit):
        probe._parse_args(common)
    with pytest.raises(SystemExit):
        probe._parse_args(["--run-id", "../unsafe", *common])
    args = probe._parse_args(["--run-id", "RP-15P-T211-PATCH", *common])
    assert args.run_id == "RP-15P-T211-PATCH"
