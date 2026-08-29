#!/usr/bin/env python3
"""PRL-27-B identity binding over the audited PRL-27-A handoff gates."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import validate_prl27_a_crop_training_handoff as _base  # noqa: E402


_BASE_WRITE_RESULT = _base._write_result


EXPECTED_TARGET_RUN_ID = (
    "PRL-27-B-TRAIN512-S32-CROP-REPLAY-BYTE-PARITY-"
    "QWEN3-INSTRUCT-BS16-N16-TEACHER25-WS8"
)
EXPECTED_TARGET_CONFIG_SHA256 = (
    "20d7abd1f3c98d00e848e26fc7560cbcd66e88d2e4af8a57e926623ee0bc9637"
)
EXPECTED_TARGET_IDENTITY_SHA256 = (
    "45d66b2e830a18869cb8e1fef82c47ccd95b3f85fb99f436a1ab735869db9de1"
)
EXPECTED_CORE_FIX_COMMIT = "c448e583887e4e49b79fe52fefb4b42934cd787e"
EXPECTED_TARGET_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-27-B-train512-s32-crop-replay-byte-parity-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
CANARY_SCHEMA = "tgvf.prl27-real-processor-double-crop-final-replay-canary.v1"
CANARY_TEST_PATH = Path("tests/environment/test_prl27_real_processor_crop_replay.py")
CANARY_DRIVER_PATH = Path("tools/validate_prl27_real_processor_crop_replay.py")
CANARY_TOKEN_IDS_SHA256 = (
    "a95e7dfe36e35f6fc23d7459b307fca558d5a5407887285660be1790b15753d5"
)
CANARY_ACCELERATOR_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
}


def _bind_variant() -> None:
    _base.EXPECTED_TARGET_RUN_ID = EXPECTED_TARGET_RUN_ID
    _base.EXPECTED_TARGET_CONFIG_SHA256 = EXPECTED_TARGET_CONFIG_SHA256
    _base.EXPECTED_TARGET_IDENTITY_SHA256 = EXPECTED_TARGET_IDENTITY_SHA256
    _base.EXPECTED_CORE_FIX_COMMIT = EXPECTED_CORE_FIX_COMMIT
    _base.EXPECTED_TARGET_ROOT = EXPECTED_TARGET_ROOT
    _base._AUTHORIZATION_SCHEMA = "tgvf.prl27-b-crop-fresh-authorization.v1"
    _base._write_result = _write_variant_result


def _write_variant_result(value: Any) -> None:
    if isinstance(value, dict):
        value = dict(value)
        value["schema_version"] = {
            "tgvf.prl27-a-training-contract-audit.v1": (
                "tgvf.prl27-b-training-contract-audit.v1"
            ),
            "tgvf.prl27-a-crop-launch-readiness.v1": (
                "tgvf.prl27-b-crop-launch-readiness.v1"
            ),
        }.get(value.get("schema_version"), value.get("schema_version"))
    _BASE_WRITE_RESULT(value)


def _read_canary(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
        raise RuntimeError("PRL-27-B real-processor canary receipt is absent or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("PRL-27-B real-processor canary receipt is malformed")
    return value


def validate_canary_receipt(*, path: Path, admitted_head: str) -> dict[str, Any]:
    value = _read_canary(path)
    identity = value.get("receipt_identity_sha256")
    content = {key: item for key, item in value.items() if key != "receipt_identity_sha256"}
    expected_identity = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    test_sha256 = hashlib.sha256(
        (REPOSITORY_ROOT / CANARY_TEST_PATH).read_bytes()
    ).hexdigest()
    driver_sha256 = hashlib.sha256(
        (REPOSITORY_ROOT / CANARY_DRIVER_PATH).read_bytes()
    ).hexdigest()
    accelerator_sha256 = hashlib.sha256(
        json.dumps(
            CANARY_ACCELERATOR_ENVIRONMENT,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if value.get("schema_version") != CANARY_SCHEMA:
        raise RuntimeError("PRL-27-B real-processor canary schema differs")
    if value.get("status") != "accepted":
        raise RuntimeError("PRL-27-B real-processor canary did not pass")
    if value.get("git_head") != admitted_head:
        raise RuntimeError("PRL-27-B real-processor canary HEAD differs")
    if (
        value.get("tracked_worktree_clean") is not True
        or value.get("repository_root") != str(REPOSITORY_ROOT)
        or value.get("test_path") != str(CANARY_TEST_PATH)
        or value.get("test_file_sha256") != test_sha256
        or value.get("validator_file_sha256") != driver_sha256
        or value.get("image_max_pixels") != 262_144
        or value.get("model_path")
        != "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
        or value.get("tokenizer_length") != 151_669
        or value.get("success_environment_text_sha256")
        != _base.EXPECTED_CONTINUATION_SHA256
        or value.get("success_environment_token_count") != 60
        or value.get("success_environment_token_ids_sha256")
        != CANARY_TOKEN_IDS_SHA256
        or value.get("native_image_placeholder_count") != 1
        or value.get("return_code") != 0
        or value.get("pytest_exact_pass_count") != 2
        or value.get("pytest_skipped_count") != 0
        or value.get("two_crop_runtime_layout_appender_passed") is not True
        or value.get("final_recorded_visual_replay_passed") is not True
        or value.get("one_token_negative_drift_rejected") is not True
        or value.get("model_weights_loaded") is not False
        or value.get("vllm_or_ray_started") is not False
        or value.get("accelerator_environment") != CANARY_ACCELERATOR_ENVIRONMENT
        or value.get("accelerator_environment_sha256") != accelerator_sha256
        or identity != expected_identity
    ):
        raise RuntimeError("PRL-27-B real-processor canary proof differs")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _bind_variant()
    if arguments[:1] != ["canary-complete"]:
        return _base.main(arguments)
    if len(arguments) != 4 or arguments[1] != "--receipt" or arguments[3] == "":
        print(
            "usage: validate_prl27_b_crop_training_handoff.py "
            "canary-complete --receipt PATH ADMITTED_HEAD",
            file=sys.stderr,
        )
        return 2
    try:
        result = validate_canary_receipt(
            path=Path(arguments[2]), admitted_head=arguments[3]
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"PRL-27-B canary rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_canary_receipt"]
