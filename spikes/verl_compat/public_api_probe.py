#!/usr/bin/env python3
"""CPU-only JSON probe for the pinned veRL/vLLM public extension surface."""

from __future__ import annotations

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any
from urllib.parse import unquote, urlparse


# This is set before torch, vLLM, or veRL can be imported.  The probe inspects
# Python APIs only and is not an experiment or a GPU workload.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


EXPECTED = {
    "vllm": "0.12.0",
    "verl_commit": "e003163181731412595257a72ec173071efb125f",
}


PUBLIC_SYMBOLS = (
    ("vllm", "ModelRegistry.register_model"),
    ("vllm.multimodal", "MULTIMODAL_REGISTRY.register_processor"),
    ("vllm.plugins", "DEFAULT_PLUGINS_GROUP"),
    ("vllm.plugins", "load_general_plugins"),
    (
        "vllm.model_executor.models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
    ),
    ("vllm.model_executor.models.qwen3_vl", "Qwen3VLMultiModalProcessor"),
    ("vllm.multimodal.parse", "DictEmbeddingItems"),
    ("verl.experimental.agent_loop", "AgentLoopOutput"),
    ("verl.experimental.agent_loop", "AgentLoopManager"),
    ("verl.protocol", "DataProto"),
    ("verl.trainer.ppo.core_algos", "register_policy_loss"),
    ("verl.workers.config", "FSDPEngineConfig"),
    ("verl.utils.checkpoint", "CheckpointHandler"),
)


def _distribution(name: str) -> dict[str, Any]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"installed": False}
    raw_direct_url = distribution.read_text("direct_url.json")
    direct_url: dict[str, Any] | None = None
    direct_url_error: str | None = None
    if raw_direct_url:
        try:
            parsed = json.loads(raw_direct_url)
            if isinstance(parsed, dict):
                direct_url = parsed
            else:
                direct_url_error = "direct_url.json is not an object"
        except json.JSONDecodeError as error:
            direct_url_error = f"{type(error).__name__}: {error}"

    result: dict[str, Any] = {
        "installed": True,
        "version": distribution.version,
        "direct_url": direct_url,
    }
    if direct_url_error:
        result["direct_url_error"] = direct_url_error
    if direct_url:
        result["commit_identity"] = _commit_identity(direct_url)
    return result


def _commit_identity(direct_url: dict[str, Any]) -> dict[str, Any] | None:
    vcs = direct_url.get("vcs_info")
    if isinstance(vcs, dict):
        return {
            "kind": "vcs",
            "commit": vcs.get("commit_id"),
            "requested_revision": vcs.get("requested_revision"),
            "vcs": vcs.get("vcs"),
        }
    url = direct_url.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return {"kind": "non_file", "url": url}
    path = Path(unquote(parsed.path)).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "kind": "local_unresolved",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "kind": "local_git",
        "path": str(path),
        "commit": commit,
        "clean": not bool(status),
    }


def _symbol(module_name: str, dotted_name: str) -> dict[str, Any]:
    identity = f"{module_name}.{dotted_name}"
    try:
        value: Any = importlib.import_module(module_name)
        for part in dotted_name.split("."):
            value = getattr(value, part)
    except Exception as error:  # the exception itself is compatibility evidence
        return {
            "identity": identity,
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "identity": identity,
        "available": True,
        "callable": callable(value),
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def main() -> int:
    distributions = {
        name: _distribution(name) for name in ("torch", "transformers", "vllm", "verl")
    }
    try:
        import torch

        torch_state: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count(),
        }
    except Exception as error:
        torch_state = {"error": f"{type(error).__name__}: {error}"}

    symbols = tuple(_symbol(module, name) for module, name in PUBLIC_SYMBOLS)
    general_plugins = tuple(
        {"name": entry.name, "value": entry.value}
        for entry in metadata.entry_points(group="vllm.general_plugins")
    )
    verl_commit = distributions.get("verl", {}).get("commit_identity", {}).get("commit")
    result = {
        "schema_version": "tgvf-public-api-probe-v1",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "expected": EXPECTED,
        "distributions": distributions,
        "torch_runtime": torch_state,
        "public_symbols": symbols,
        "vllm_general_plugins": general_plugins,
        "checks": {
            "cuda_hidden": (
                not torch_state.get("cuda_available", True)
                and torch_state.get("visible_device_count", -1) == 0
            ),
            "vllm_version": distributions.get("vllm", {}).get("version")
            == EXPECTED["vllm"],
            "verl_commit": verl_commit == EXPECTED["verl_commit"],
            "all_public_symbols": all(item["available"] for item in symbols),
            "tgvf_general_plugin": any(
                entry["name"] == "tgvf_qwen3_precomputed"
                and entry["value"]
                == "tgvf_rl.framework.vllm:register_tgvf_qwen3_vllm_plugin"
                for entry in general_plugins
            ),
        },
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
