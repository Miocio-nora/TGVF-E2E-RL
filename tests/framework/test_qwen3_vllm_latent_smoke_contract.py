from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPOSITORY_ROOT / "spikes/verl_compat/qwen3_vllm_latent_smoke.py"
RUN_ID = "SC-20-T211-QWEN3-VLLM-LATENT-TEST"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tgvf_qwen3_latent_contract", SMOKE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load latent smoke")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_requires_run_id_and_explicit_candidate_stack() -> None:
    smoke = _load()
    output = "artifacts/compatibility/proposed-latent.json"
    with pytest.raises(SystemExit):
        smoke._parse_args(["--output", output])
    args = smoke._parse_args(
        [
            "--run-id",
            RUN_ID,
            "--stack",
            "torch211-cu129",
            "--output",
            output,
        ]
    )
    assert args.run_id == RUN_ID
    assert args.stack == "torch211-cu129"


def test_candidate_runtime_identity_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    smoke = _load()
    versions = {
        "torch": "2.11.0+cu129",
        "transformers": "4.57.6",
        "vllm": "0.23.0+cu129",
        "verl": "0.9.0.dev0",
    }
    monkeypatch.setattr(smoke.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(smoke.torch, "__version__", "2.11.0+cu129")
    monkeypatch.setattr(
        smoke,
        "verify_verl_distribution_identity",
        lambda **_kwargs: SimpleNamespace(
            commit="638b8ff84f279e054982f1f4633a546f3c6ced68",
            source_url="https://github.com/verl-project/verl.git",
            source_kind="vcs",
            source_clean=None,
        ),
    )

    identity = smoke._validate_candidate_runtime("torch211-cu129")
    assert identity["compatibility_stack"]["selector"] == "torch211-cu129"
    assert all(identity["checks"].values())

    versions["vllm"] = "0.23.1+cu129"
    with pytest.raises(RuntimeError, match="runtime identity failed"):
        smoke._validate_candidate_runtime("torch211-cu129")
