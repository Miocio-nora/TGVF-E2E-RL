from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPOSITORY_ROOT / "spikes/verl_compat/public_api_probe.py"
CONTROL_VLLM_VERSION = "0.12.0"
CONTROL_VERL_COMMIT = "e003163181731412595257a72ec173071efb125f"
CANDIDATE_VLLM_VERSION = "0.23.0+cu129"
CANDIDATE_VERL_COMMIT = "638b8ff84f279e054982f1f4633a546f3c6ced68"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tgvf_public_api_probe_test", PROBE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load public API probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_runtime(
    monkeypatch,
    probe: Any,
    *,
    vllm_version: str,
    verl_commit: str,
) -> None:
    runtime_stack = probe.audited_stack_for_framework_pair(
        vllm_distribution_version=vllm_version,
        verl_commit=verl_commit,
    )
    versions = {
        "torch": runtime_stack.torch_distribution_version,
        "transformers": runtime_stack.transformers_distribution_version,
        "vllm": vllm_version,
        "verl": "test-verl",
        "TransferQueue": runtime_stack.transfer_queue_distribution_version,
    }

    def distribution(name: str) -> dict[str, Any]:
        version = versions[name]
        if version is None:
            return {"installed": False}
        result: dict[str, Any] = {
            "installed": True,
            "version": version,
            "direct_url": None,
        }
        if name == "verl":
            result["commit_identity"] = {"kind": "vcs", "commit": verl_commit}
        archive_identity = None
        if name == "vllm" and runtime_stack.vllm_archive_url is not None:
            archive_identity = (
                runtime_stack.vllm_archive_url,
                runtime_stack.vllm_archive_sha256,
            )
        elif (
            name == "TransferQueue"
            and runtime_stack.transfer_queue_archive_url is not None
        ):
            archive_identity = (
                runtime_stack.transfer_queue_archive_url,
                runtime_stack.transfer_queue_archive_sha256,
            )
        if archive_identity is not None:
            url, sha256 = archive_identity
            result["direct_url"] = {
                "url": url,
                "archive_info": {
                    "hash": f"sha256={sha256}",
                    "hashes": {"sha256": sha256},
                },
            }
        return result

    monkeypatch.setattr(probe, "_distribution", distribution)
    monkeypatch.setattr(
        probe,
        "_symbol",
        lambda module, name: {
            "identity": f"{module}.{name}",
            "available": True,
            "callable": not (
                module == "vllm.plugins" and name == "DEFAULT_PLUGINS_GROUP"
            ),
            "type": "test.stub",
        },
    )
    monkeypatch.setattr(
        probe.metadata,
        "entry_points",
        lambda *, group: (
            SimpleNamespace(
                name="tgvf_qwen3_precomputed",
                value="tgvf_rl.framework.vllm:register_tgvf_qwen3_vllm_plugin",
            ),
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(
            __version__=runtime_stack.torch_runtime_version,
            cuda=SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
            ),
        ),
    )


def test_probe_defaults_preserve_control_stack(monkeypatch, capsys) -> None:
    probe = _load_probe()
    _stub_runtime(
        monkeypatch,
        probe,
        vllm_version=CONTROL_VLLM_VERSION,
        verl_commit=CONTROL_VERL_COMMIT,
    )

    assert probe.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["expected"] == {
        "vllm": CONTROL_VLLM_VERSION,
        "verl_commit": CONTROL_VERL_COMMIT,
        "stack": "control",
    }
    assert all(payload["checks"].values())


def test_probe_accepts_explicit_candidate_and_fails_closed(monkeypatch, capsys) -> None:
    probe = _load_probe()
    _stub_runtime(
        monkeypatch,
        probe,
        vllm_version=CANDIDATE_VLLM_VERSION,
        verl_commit=CANDIDATE_VERL_COMMIT,
    )
    candidate_args = [
        "--expected-vllm-version",
        CANDIDATE_VLLM_VERSION,
        "--expected-verl-commit",
        CANDIDATE_VERL_COMMIT,
    ]

    assert probe.main(candidate_args) == 0
    candidate_payload = json.loads(capsys.readouterr().out)
    assert all(candidate_payload["checks"].values())

    assert probe.main([]) == 1
    control_payload = json.loads(capsys.readouterr().out)
    assert control_payload["checks"]["vllm_version"] is False
    assert control_payload["checks"]["verl_commit"] is False


def test_probe_rejects_every_cross_stack_or_unknown_pair(monkeypatch, capsys) -> None:
    probe = _load_probe()
    _stub_runtime(
        monkeypatch,
        probe,
        vllm_version=CANDIDATE_VLLM_VERSION,
        verl_commit=CANDIDATE_VERL_COMMIT,
    )
    for vllm_version, verl_commit in (
        (CONTROL_VLLM_VERSION, CANDIDATE_VERL_COMMIT),
        (CANDIDATE_VLLM_VERSION, CONTROL_VERL_COMMIT),
        ("0.23.1+cu129", CANDIDATE_VERL_COMMIT),
    ):
        assert (
            probe.main(
                [
                    "--expected-vllm-version",
                    vllm_version,
                    "--expected-verl-commit",
                    verl_commit,
                ]
            )
            == 1
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["expected"]["stack"] is None
        assert payload["checks"]["audited_framework_pair"] is False


def test_probe_rejects_candidate_archive_hash_and_symbol_contract_drift(
    monkeypatch, capsys
) -> None:
    probe = _load_probe()
    _stub_runtime(
        monkeypatch,
        probe,
        vllm_version=CANDIDATE_VLLM_VERSION,
        verl_commit=CANDIDATE_VERL_COMMIT,
    )
    original_distribution = probe._distribution

    def bad_distribution(name: str) -> dict[str, Any]:
        result = original_distribution(name)
        if name == "TransferQueue":
            result["direct_url"]["archive_info"]["hashes"]["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(probe, "_distribution", bad_distribution)
    candidate_args = [
        "--expected-vllm-version",
        CANDIDATE_VLLM_VERSION,
        "--expected-verl-commit",
        CANDIDATE_VERL_COMMIT,
    ]
    assert probe.main(candidate_args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["transfer_queue_identity"] is False

    _stub_runtime(
        monkeypatch,
        probe,
        vllm_version=CANDIDATE_VLLM_VERSION,
        verl_commit=CANDIDATE_VERL_COMMIT,
    )
    original_symbol = probe._symbol

    def bad_symbol(module: str, name: str) -> dict[str, Any]:
        result = original_symbol(module, name)
        if name == "AgentLoopManagerTQ":
            result["identity"] = "wrong.identity"
        return result

    monkeypatch.setattr(probe, "_symbol", bad_symbol)
    assert probe.main(candidate_args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["public_symbol_identity_and_callable"] is False
