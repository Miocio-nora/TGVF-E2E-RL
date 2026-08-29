#!/usr/bin/env python3
"""Run the PRL27 real-processor Crop replay gate and write one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = Path("tests/environment/test_prl27_real_processor_crop_replay.py")
MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
SCHEMA_VERSION = "tgvf.prl27-real-processor-double-crop-final-replay-canary.v1"
IMAGE_MAX_PIXELS = 262_144
EXPECTED_ENVIRONMENT_TEXT_SHA256 = (
    "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
)
POSITIVE_TEST = (
    "test_real_qwen_processor_two_matched_crops_and_final_replay_are_identical"
)
NEGATIVE_TEST = "test_real_qwen_processor_detects_one_token_layout_appender_drift"
ISOLATED_ACCELERATOR_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("git HEAD is not one exact commit")
    return head


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _environment_token_proof() -> dict[str, object]:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("real Crop replay receipt requires transformers") from error
    from tgvf_rl.environment.native_appender import (
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
    )

    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"local Qwen processor is unavailable: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = tuple(
        tokenizer.encode(
            QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
            add_special_tokens=False,
        )
    )
    if (
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
        != EXPECTED_ENVIRONMENT_TEXT_SHA256
        or len(token_ids) != 60
        or token_ids.count(tokenizer.convert_tokens_to_ids("<|image_pad|>")) != 1
    ):
        raise RuntimeError("matched Crop environment token contract differs")
    return {
        "model_path": str(MODEL_PATH),
        "tokenizer_length": len(tokenizer),
        "success_environment_text_sha256": (
            QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
        ),
        "success_environment_token_count": len(token_ids),
        "success_environment_token_ids_sha256": _canonical_sha256(token_ids),
        "native_image_placeholder_count": 1,
    }


def _run_canary() -> dict[str, object]:
    test = REPOSITORY_ROOT / TEST_PATH
    if not test.is_file():
        raise RuntimeError(f"Crop replay canary test is missing: {test}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        "--tb=short",
        str(TEST_PATH),
    ]
    environment = dict(os.environ)
    environment.update(ISOLATED_ACCELERATOR_ENVIRONMENT)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    transcript = result.stdout + result.stderr
    positive_passed = f"{TEST_PATH}::{POSITIVE_TEST} PASSED" in transcript
    negative_passed = f"{TEST_PATH}::{NEGATIVE_TEST} PASSED" in transcript
    if (
        result.returncode != 0
        or not positive_passed
        or not negative_passed
        or " skipped" in transcript.lower()
        or "2 passed" not in transcript.lower()
    ):
        sys.stderr.write(transcript)
        raise RuntimeError("real-processor Crop replay canary did not pass exactly twice")
    return {
        "command": command,
        "return_code": result.returncode,
        "pytest_stdout_sha256": hashlib.sha256(
            result.stdout.encode("utf-8")
        ).hexdigest(),
        "pytest_stderr_sha256": hashlib.sha256(
            result.stderr.encode("utf-8")
        ).hexdigest(),
        "pytest_exact_pass_count": 2,
        "pytest_skipped_count": 0,
        "two_crop_runtime_layout_appender_passed": positive_passed,
        "final_recorded_visual_replay_passed": positive_passed,
        "one_token_negative_drift_rejected": negative_passed,
        "model_weights_loaded": False,
        "vllm_or_ray_started": False,
        "accelerator_environment": dict(ISOLATED_ACCELERATOR_ENVIRONMENT),
        "accelerator_environment_sha256": _canonical_sha256(
            ISOLATED_ACCELERATOR_ENVIRONMENT
        ),
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite Crop replay receipt: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    output = arguments.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite Crop replay receipt: {output}")
    head = _git_head()
    if arguments.expected_head is not None and arguments.expected_head != head:
        raise RuntimeError(
            f"Crop replay canary HEAD differs: expected={arguments.expected_head} actual={head}"
        )
    if _git_dirty():
        raise RuntimeError("Crop replay canary requires a clean tracked worktree")

    token_proof = _environment_token_proof()
    execution = _run_canary()
    # Re-read HEAD and tracked state after execution so a concurrent code change
    # cannot be hidden behind a successful receipt.
    if _git_head() != head or _git_dirty():
        raise RuntimeError("repository changed while Crop replay canary was running")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
        "repository_root": str(REPOSITORY_ROOT),
        "git_head": head,
        "tracked_worktree_clean": True,
        "test_path": str(TEST_PATH),
        "test_file_sha256": _sha256(REPOSITORY_ROOT / TEST_PATH),
        "validator_file_sha256": _sha256(Path(__file__).resolve()),
        "image_max_pixels": IMAGE_MAX_PIXELS,
        **token_proof,
        **execution,
    }
    payload["receipt_identity_sha256"] = _canonical_sha256(payload)
    _write_exclusive(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
