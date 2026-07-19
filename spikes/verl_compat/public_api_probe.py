#!/usr/bin/env python3
"""CPU-only JSON probe for the pinned veRL/vLLM public extension surface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse


# This is set before torch, vLLM, or veRL can be imported.  The probe inspects
# Python APIs only and is not an experiment or a GPU workload.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tgvf_rl.compatibility_stack import (  # noqa: E402
    CONTROL_COMPATIBILITY_STACK,
    TORCH211_CU129_COMPATIBILITY_STACK,
    audited_compatibility_stack,
    audited_stack_for_framework_pair,
)


_CONTROL_STACK = audited_compatibility_stack(CONTROL_COMPATIBILITY_STACK)
_TORCH211_STACK = audited_compatibility_stack(TORCH211_CU129_COMPATIBILITY_STACK)
DEFAULT_EXPECTED = {
    "vllm": _CONTROL_STACK.vllm_distribution_version,
    "verl_commit": _CONTROL_STACK.verl_commit,
}


PUBLIC_SYMBOLS = (
    ("vllm", "ModelRegistry.register_model", True),
    ("vllm.multimodal", "MULTIMODAL_REGISTRY.register_processor", True),
    ("vllm.plugins", "DEFAULT_PLUGINS_GROUP", False),
    ("vllm.plugins", "load_general_plugins", True),
    (
        "vllm.model_executor.models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        True,
    ),
    ("vllm.model_executor.models.qwen3_vl", "Qwen3VLMultiModalProcessor", True),
    ("vllm.multimodal.parse", "DictEmbeddingItems", True),
    ("verl.experimental.agent_loop", "AgentLoopOutput", True),
    ("verl.experimental.agent_loop", "AgentLoopManager", True),
    ("verl.protocol", "DataProto", True),
    ("verl.trainer.ppo.core_algos", "register_policy_loss", True),
    ("verl.workers.config", "FSDPEngineConfig", True),
    ("verl.utils.checkpoint", "CheckpointHandler", True),
)

CANDIDATE_V1_TRANSFER_QUEUE_SYMBOLS = (
    ("transfer_queue", "KVBatchMeta", True),
    ("transfer_queue", "init", True),
    ("verl.trainer.ppo.v1.agent_loop_tq", "AgentLoopManagerTQ", True),
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


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-vllm-version",
        type=_nonempty,
        default=DEFAULT_EXPECTED["vllm"],
        help="exact installed vLLM distribution version",
    )
    parser.add_argument(
        "--expected-verl-commit",
        type=_nonempty,
        default=DEFAULT_EXPECTED["verl_commit"],
        help="exact installed veRL VCS or local-checkout commit",
    )
    return parser.parse_args(argv)


def _archive_identity_matches(
    distribution: dict[str, Any],
    *,
    expected_url: str | None,
    expected_sha256: str | None,
) -> bool:
    if expected_url is None or expected_sha256 is None:
        return (
            expected_url is None
            and expected_sha256 is None
            and distribution.get("direct_url") is None
        )
    direct_url = distribution.get("direct_url")
    if not isinstance(direct_url, dict) or direct_url.get("url") != expected_url:
        return False
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        return False
    hashes = archive_info.get("hashes")
    return (
        isinstance(hashes, dict)
        and hashes.get("sha256") == expected_sha256
        and archive_info.get("hash") == f"sha256={expected_sha256}"
    )


def _symbols_match_contract(
    symbols: tuple[dict[str, Any], ...],
    expected_symbols: tuple[tuple[str, str, bool], ...],
) -> bool:
    if len(symbols) != len(expected_symbols):
        return False
    return all(
        item.get("identity") == f"{module}.{name}"
        and item.get("available") is True
        and item.get("callable") is expected_callable
        for item, (module, name, expected_callable) in zip(
            symbols, expected_symbols, strict=True
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected = {
        "vllm": args.expected_vllm_version,
        "verl_commit": args.expected_verl_commit,
    }
    try:
        expected_stack = audited_stack_for_framework_pair(
            vllm_distribution_version=args.expected_vllm_version,
            verl_commit=args.expected_verl_commit,
        )
    except ValueError:
        expected_stack = None
    public_symbol_contract = PUBLIC_SYMBOLS
    if (
        expected_stack is not None
        and expected_stack.selector == _TORCH211_STACK.selector
    ):
        public_symbol_contract += CANDIDATE_V1_TRANSFER_QUEUE_SYMBOLS
    import_stdout = io.StringIO()
    with redirect_stdout(import_stdout):
        distributions = {
            name: _distribution(name)
            for name in ("torch", "transformers", "vllm", "verl", "TransferQueue")
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

        symbols = tuple(
            _symbol(module, name) for module, name, _ in public_symbol_contract
        )
    general_plugins = tuple(
        {"name": entry.name, "value": entry.value}
        for entry in metadata.entry_points(group="vllm.general_plugins")
    )
    verl_commit = distributions.get("verl", {}).get("commit_identity", {}).get("commit")
    verl_source_clean = (
        distributions.get("verl", {}).get("commit_identity", {}).get("clean")
    )
    transfer_queue = distributions.get("TransferQueue", {})
    result = {
        "schema_version": "tgvf-public-api-probe-v1",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "expected": {
            **expected,
            "stack": expected_stack.selector if expected_stack is not None else None,
        },
        "distributions": distributions,
        "torch_runtime": torch_state,
        "public_symbols": symbols,
        "captured_import_stdout": import_stdout.getvalue(),
        "vllm_general_plugins": general_plugins,
        "checks": {
            "audited_framework_pair": expected_stack is not None,
            "cuda_hidden": (
                not torch_state.get("cuda_available", True)
                and torch_state.get("visible_device_count", -1) == 0
            ),
            "python_identity": (
                expected_stack is not None
                and platform.python_implementation() == "CPython"
                and sys.version_info[:2] == expected_stack.python_major_minor
            ),
            "torch_identity": (
                expected_stack is not None
                and distributions.get("torch", {}).get("version")
                == expected_stack.torch_distribution_version
                and torch_state.get("version") == expected_stack.torch_runtime_version
            ),
            "transformers_version": (
                expected_stack is not None
                and distributions.get("transformers", {}).get("version")
                == expected_stack.transformers_distribution_version
            ),
            "vllm_version": distributions.get("vllm", {}).get("version")
            == expected["vllm"],
            "vllm_archive_identity": (
                expected_stack is not None
                and _archive_identity_matches(
                    distributions.get("vllm", {}),
                    expected_url=expected_stack.vllm_archive_url,
                    expected_sha256=expected_stack.vllm_archive_sha256,
                )
            ),
            "verl_commit": verl_commit == expected["verl_commit"],
            "verl_source_clean": verl_source_clean is not False,
            "transfer_queue_identity": (
                expected_stack is not None
                and (
                    expected_stack.transfer_queue_distribution_version is None
                    or (
                        transfer_queue.get("version")
                        == expected_stack.transfer_queue_distribution_version
                        and _archive_identity_matches(
                            transfer_queue,
                            expected_url=expected_stack.transfer_queue_archive_url,
                            expected_sha256=(
                                expected_stack.transfer_queue_archive_sha256
                            ),
                        )
                    )
                )
            ),
            "public_symbol_identity_and_callable": _symbols_match_contract(
                symbols, public_symbol_contract
            ),
            "tgvf_general_plugin": any(
                entry["name"] == "tgvf_qwen3_precomputed"
                and entry["value"]
                == "tgvf_rl.framework.vllm:register_tgvf_qwen3_vllm_plugin"
                for entry in general_plugins
            ),
        },
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
