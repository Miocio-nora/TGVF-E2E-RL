"""Control and canary the local Qwen3-VL-32B TGVF visual judge.

``preflight`` is CPU-only.  ``launch`` starts one TP1 vLLM service on the
selected physical GPU.  ``wait`` validates both health and served identity,
and ``canary`` sends one request through the exact production provider.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from tgvf_rl.judges.tgvf_visual_quality import (
    TGVFVisualQualityJudgeRequest,
    load_tgvf_visual_quality_judge,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/policy/judges/local_qwen3_vl_32b_tgvf_visual_quality_v1.json"
)
DEFAULT_MODEL = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-32B-Thinking")
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "artifacts/policy-control/LOCAL-QWEN3-VL-32B-TGVF-JUDGE"
)
DEFAULT_CANARY_IMAGE = REPOSITORY_ROOT / "reports/figures/chart_2017.png"
SERVED_MODEL = "Qwen3-VL-32B-Thinking"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bound_config(path: Path):
    config_sha256 = _file_sha256(path)
    return (
        load_tgvf_visual_quality_judge(
            path,
            expected_file_sha256=config_sha256,
        ),
        config_sha256,
    )


def _server_command(model_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8013",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.80",
        "--max-num-batched-tokens",
        "32768",
        "--max-num-seqs",
        "64",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--mm-encoder-attn-backend",
        "TORCH_SDPA",
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
        "--mm-processor-kwargs",
        '{"max_pixels":262144}',
    ]


def _server_environment(physical_gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    include_root = REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CPATH": f"{include_root}:{include_root / 'python3.12'}",
            "PATH": (
                f"{REPOSITORY_ROOT / '.venv312/bin'}:"
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "VLLM_USE_V1": "1",
            # An unset VLLM_PLUGINS loads every discovered plugin.  This judge
            # uses upstream Qwen3VLForConditionalGeneration, not the repo's
            # separate precomputed-latent rollout architecture.
            "VLLM_PLUGINS": "",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
        }
    )
    return environment


def _health_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _read_json_url(url: str, *, timeout_seconds: float) -> Any:
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.load(response)


def _validate_models(base_url: str, *, timeout_seconds: float) -> list[str]:
    payload = _read_json_url(
        base_url.rstrip("/") + "/models",
        timeout_seconds=timeout_seconds,
    )
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("local visual judge /models response has no data list")
    identities = [row.get("id") for row in rows if isinstance(row, dict)]
    if identities != [SERVED_MODEL]:
        raise RuntimeError(
            f"local visual judge served identities differ: {identities!r}"
        )
    return identities


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    bound, config_sha256 = _bound_config(config_path)
    if bound.config.base_url != "http://127.0.0.1:8013/v1":
        raise RuntimeError("local visual judge base URL differs from controller")
    if bound.config.model_name != SERVED_MODEL:
        raise RuntimeError("local visual judge served model differs from controller")
    if not model_path.is_dir():
        raise RuntimeError(f"local visual judge model is missing: {model_path}")
    shards = sorted(model_path.glob("model-*-of-*.safetensors"))
    if len(shards) != 14:
        raise RuntimeError(
            f"local visual judge expected 14 weight shards, found {len(shards)}"
        )
    required = (
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "tokenizer_config.json",
        DEFAULT_CANARY_IMAGE,
        REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include/python3.12/Python.h",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"local visual judge prerequisites are missing: {missing}")
    versions = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "vllm")
    }
    if versions["vllm"] != "0.12.0":
        raise RuntimeError(f"local visual judge vLLM differs: {versions['vllm']}")
    return {
        "status": "preflight_passed",
        "config_path": str(config_path),
        "config_file_sha256": config_sha256,
        "config_identity_sha256": bound.config_identity.sha256,
        "model_path": str(model_path),
        "weight_shards": len(shards),
        "weight_bytes": sum(path.stat().st_size for path in shards),
        "versions": versions,
        "physical_gpu": args.physical_gpu,
        "command": _server_command(model_path),
        "environment": {
            name: _server_environment(args.physical_gpu)[name]
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CPATH",
                "VLLM_USE_V1",
                "VLLM_PLUGINS",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_ATTENTION_BACKEND",
            )
        },
    }


def _launch(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "server.log"
    command = _server_command(args.model.resolve())
    environment = _server_environment(args.physical_gpu)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        **preflight,
        "status": "server_starting",
        "pid": process.pid,
        "process_group": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
    }
    _write_json_atomic(output_dir / "server-process.json", result)
    return result


def _wait(args: argparse.Namespace) -> dict[str, object]:
    bound, config_sha256 = _bound_config(args.config.resolve())
    deadline = time.monotonic() + args.wait_timeout_seconds
    attempts = 0
    last_error = "not attempted"
    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urlopen(
                _health_url(bound.config.base_url),
                timeout=min(5.0, args.wait_timeout_seconds),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"health status {response.status}")
            identities = _validate_models(
                bound.config.base_url,
                timeout_seconds=5.0,
            )
            return {
                "status": "ready",
                "attempts": attempts,
                "config_file_sha256": config_sha256,
                "base_url": bound.config.base_url,
                "served_models": identities,
            }
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(2.0)
    raise RuntimeError(
        "local visual judge did not become ready before timeout; "
        f"last error: {last_error}"
    )


def _canary(args: argparse.Namespace) -> dict[str, object]:
    bound, config_sha256 = _bound_config(args.config.resolve())
    identities = _validate_models(
        bound.config.base_url,
        timeout_seconds=10.0,
    )
    image_path = args.canary_image.resolve()
    image_sha256 = _file_sha256(image_path)
    judge_request = TGVFVisualQualityJudgeRequest(
        request_id="local-qwen3-vl-32b-tgvf-visual-quality-canary-v1",
        image_path=image_path,
        image_sha256=image_sha256,
        question="Which year does the highlighted chart evidence refer to?",
        tool_target="the visible year labels and corresponding chart marks",
        post_tool_reasoning=(
            "The chart visibly contains the relevant year label and plotted evidence."
        ),
        final_answer="2017",
        prompt_identity=bound.config.prompt_identity,
    )
    started = time.perf_counter()
    result = bound.provider.judge(judge_request)
    elapsed_seconds = time.perf_counter() - started
    payload: dict[str, object] = {
        "status": "passed" if result.ok else "failed",
        "elapsed_seconds": elapsed_seconds,
        "config_file_sha256": config_sha256,
        "config_identity_sha256": bound.config_identity.sha256,
        "served_models": identities,
        "request_id": result.request_id,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "ok": result.ok,
        "focus_score": result.focus_score,
        "grounding_score": result.grounding_score,
        "failure_kind": (
            result.failure_kind.value if result.failure_kind is not None else None
        ),
        "failure_reason": result.failure_reason,
        "usage": asdict(result.usage) if result.usage is not None else None,
    }
    _write_json_atomic(args.output_dir.resolve() / "canary.json", payload)
    if not result.ok:
        raise RuntimeError(
            "local visual judge canary failed: "
            f"{result.failure_kind} / {result.failure_reason}"
        )
    return payload


def _status(args: argparse.Namespace) -> dict[str, object]:
    bound, config_sha256 = _bound_config(args.config.resolve())
    try:
        with urlopen(_health_url(bound.config.base_url), timeout=2.0) as response:
            healthy = response.status == 200
        identities = _validate_models(bound.config.base_url, timeout_seconds=2.0)
        error = None
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as caught:
        healthy = False
        identities = []
        error = f"{type(caught).__name__}: {caught}"
    return {
        "status": "ready" if healthy else "not_ready",
        "healthy": healthy,
        "served_models": identities,
        "error": error,
        "config_file_sha256": config_sha256,
        "base_url": bound.config.base_url,
    }


def _stop(args: argparse.Namespace) -> dict[str, object]:
    process_path = args.output_dir.resolve() / "server-process.json"
    metadata = json.loads(process_path.read_text(encoding="utf-8"))
    pid = metadata.get("pid")
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("local visual judge process metadata has an invalid PID")
    proc_root = Path(f"/proc/{pid}")
    if not proc_root.exists():
        return {"status": "already_stopped", "pid": pid}
    if proc_root.stat().st_uid != os.getuid():
        raise RuntimeError("local visual judge PID belongs to another user")
    command_line = (proc_root / "cmdline").read_bytes().replace(b"\0", b" ")
    required_fragments = (
        b"vllm.entrypoints.openai.api_server",
        str(DEFAULT_MODEL).encode("utf-8"),
        b"--port 8013",
    )
    if any(fragment not in command_line for fragment in required_fragments):
        raise RuntimeError("refusing to stop a PID with a different command line")
    if os.getpgid(pid) != pid:
        raise RuntimeError("refusing to stop a PID that is not its process-group leader")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    while proc_root.exists() and time.monotonic() < deadline:
        time.sleep(0.25)
    if proc_root.exists():
        raise RuntimeError(
            "local visual judge did not stop after SIGTERM; inspect before escalation"
        )
    return {"status": "stopped", "pid": pid}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("preflight", "launch", "wait", "canary", "status", "stop"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--wait-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--canary-image", type=Path, default=DEFAULT_CANARY_IMAGE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.physical_gpu < 0:
        raise ValueError("physical GPU must be non-negative")
    handlers = {
        "preflight": _preflight,
        "launch": _launch,
        "wait": _wait,
        "canary": _canary,
        "status": _status,
        "stop": _stop,
    }
    payload = handlers[args.action](args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
