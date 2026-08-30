#!/usr/bin/python3 -I
"""Validate the pinned external VLMEvalKit CPU/CLI deployment without downloads."""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/validate_vlmevalkit_deployment.py",
        ),
    )

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOYMENT = (
    REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
)
PROBE_PREFIX = "TGVF_VLMEVALKIT_PROBE="


def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        **kwargs,
    )


def _require_ok(result: subprocess.CompletedProcess[str], label: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{label} failed: {detail}")
    return result.stdout.strip()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def validate(deployment_path: Path) -> dict[str, Any]:
    config = json.loads(deployment_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported deployment schema_version")

    checkout = _resolve_repo_path(config["checkout"])
    python = _resolve_repo_path(config["python"])
    overlay = _resolve_repo_path(config["overlay"])
    benchmark_root = _resolve_repo_path(config["shared_benchmark_root"])
    example_config = _resolve_repo_path(config["example_config"])
    for label, path in (
        ("checkout", checkout),
        ("python", python),
        ("overlay", overlay),
        ("shared benchmark root", benchmark_root),
        ("example config", example_config),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    commit = _require_ok(
        _run(["git", "-C", str(checkout), "rev-parse", "HEAD"]),
        "checkout commit probe",
    )
    if commit != config["commit"]:
        raise RuntimeError(f"checkout commit mismatch: {commit}")
    dirty = _require_ok(
        _run(["git", "-C", str(checkout), "status", "--porcelain"]),
        "checkout cleanliness probe",
    )
    if dirty:
        raise RuntimeError("external VLMEvalKit checkout is dirty")

    run_py = checkout / "run.py"
    run_sha256 = sha256(run_py.read_bytes()).hexdigest()
    if run_sha256 != config["run_py_sha256"]:
        raise RuntimeError("external VLMEvalKit run.py SHA256 mismatch")

    example = json.loads(example_config.read_text(encoding="utf-8"))
    required_model = config["required_model"]
    if required_model not in example.get("model", {}):
        raise ValueError("example config does not select the required model")
    if not example.get("data"):
        raise ValueError("example config does not select a dataset")

    environment = os.environ.copy()
    python_path = [str(checkout), str(overlay)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["LMUData"] = str(benchmark_root)

    probe_source = """
import json
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS

required_model = __import__('os').environ['TGVF_REQUIRED_MODEL']
required_datasets = json.loads(__import__('os').environ['TGVF_REQUIRED_DATASETS'])
payload = {
    'model_registered': required_model in supported_VLM,
    'datasets_registered': [name for name in required_datasets if name in SUPPORTED_DATASETS],
}
print('TGVF_VLMEVALKIT_PROBE=' + json.dumps(payload, sort_keys=True))
"""
    environment["TGVF_REQUIRED_MODEL"] = required_model
    environment["TGVF_REQUIRED_DATASETS"] = json.dumps(config["required_datasets"])
    probe = _run([str(python), "-c", probe_source], env=environment)
    _require_ok(probe, "VLMEvalKit import/registry probe")
    probe_lines = [
        line for line in probe.stdout.splitlines() if line.startswith(PROBE_PREFIX)
    ]
    if len(probe_lines) != 1:
        raise RuntimeError("VLMEvalKit registry probe did not emit its result")
    registry = json.loads(probe_lines[0][len(PROBE_PREFIX) :])
    if not registry["model_registered"]:
        raise RuntimeError(f"VLMEvalKit does not register {required_model}")
    if registry["datasets_registered"] != config["required_datasets"]:
        raise RuntimeError("one or more required benchmark aliases are not registered")

    help_result = _run([str(python), str(run_py), "--help"], env=environment)
    help_text = _require_ok(help_result, "VLMEvalKit CLI probe")
    if "--config CONFIG" not in help_text or "--mode {all,infer,eval}" not in help_text:
        raise RuntimeError("VLMEvalKit CLI surface differs from the pinned contract")

    version = _require_ok(_run([str(python), "--version"]), "Python version probe")
    return {
        "identity": config["identity"],
        "checkout_clean": True,
        "commit": commit,
        "run_py_sha256": run_sha256,
        "python": version,
        "import_and_cli": "ok",
        **registry,
    }


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/validate_vlmevalkit_deployment.py"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment",
        type=Path,
        default=DEFAULT_DEPLOYMENT,
        help="deployment identity JSON",
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.deployment.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
