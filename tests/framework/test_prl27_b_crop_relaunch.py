from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_27_b_qwen3_instruct_full_crop_train512_replay_byte_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
A_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_27_a_qwen3_instruct_full_crop_train512_exact_continuation_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
SOURCE_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
VALIDATOR = TOOLS / "validate_prl27_b_crop_training_handoff.py"
B_HANDOFF = TOOLS / "handoff_prl26_c_to_prl27_b_crop_train512_s32.sh"
B_LAUNCHER = TOOLS / "launch_prl27_b_crop_train_tmux.sh"


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_prl27_b_contract_is_fresh_disjoint_and_core_fix_bound() -> None:
    validator = _module(VALIDATOR, "prl27_b_validator_under_test")
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validator._bind_variant()
    result = validator._base.validate_contracts(
        repository=ROOT,
        source_config_path=SOURCE_CONFIG,
        target_config_path=CONFIG,
        admitted_head=head,
        require_clean=False,
    )

    assert result["target_run_id"] == validator.EXPECTED_TARGET_RUN_ID
    assert result["target_identity_sha256"] == (
        validator.EXPECTED_TARGET_IDENTITY_SHA256
    )
    assert result["target_output_root"] == str(validator.EXPECTED_TARGET_ROOT)
    assert "PRL-27-B" in result["target_output_root"]
    assert "PRL-27-A" not in result["target_output_root"]

    with A_CONFIG.open("rb") as handle:
        historical = tomllib.load(handle)
    with CONFIG.open("rb") as handle:
        recovered = tomllib.load(handle)
    recovered["run_id"] = historical["run_id"]
    recovered["code"]["commit"] = historical["code"]["commit"]
    recovered["reward"]["judge_reason"] = historical["reward"]["judge_reason"]
    recovered["output"] = historical["output"]
    assert recovered == historical


def test_prl27_b_launch_orders_real_canary_before_fresh_root_and_training() -> None:
    for script in (B_HANDOFF, B_LAUNCHER):
        subprocess.run(["bash", "-n", str(script)], check=True)
    handoff = B_HANDOFF.read_text(encoding="utf-8")
    source = handoff.index("source-complete")
    resources = handoff.index("resources-free", source)
    canary = handoff.index("canary-complete", resources)
    target = handoff.index("target-ready", canary)
    training = handoff.rindex("trainable_tgvf_supervisor")
    assert source < resources < canary < target < training
    assert canary < target < training
    assert "source_live_pane_required=false" in handoff
    assert "release_stable_polls < 3" in handoff
    assert "validate_prl27_real_processor_crop_replay.py" in handoff
    assert "PRL-27-A-crop-train512-s32-20260829" not in handoff
    assert "PRL-27-B-crop-train512-s32-20260830" in handoff
    assert "prl27b_train512_crop_replay_byte_parity_s32" in handoff

    launcher = B_LAUNCHER.read_text(encoding="utf-8")
    assert "prl27-b-crop-train512-s32" in launcher
    assert "PRL27_B_ADMITTED_HEAD" in launcher
    assert "PRL-27-B-crop-train512-s32-20260830" in launcher


def test_prl27_b_canary_receipt_is_head_and_self_identity_bound(tmp_path: Path) -> None:
    validator = _module(VALIDATOR, "prl27_b_canary_validator_under_test")
    validator._bind_variant()
    head = "a" * 40
    payload = {
        "schema_version": validator.CANARY_SCHEMA,
        "status": "accepted",
        "repository_root": str(ROOT),
        "git_head": head,
        "tracked_worktree_clean": True,
        "test_path": str(validator.CANARY_TEST_PATH),
        "test_file_sha256": hashlib.sha256(
            (ROOT / validator.CANARY_TEST_PATH).read_bytes()
        ).hexdigest(),
        "validator_file_sha256": hashlib.sha256(
            (ROOT / validator.CANARY_DRIVER_PATH).read_bytes()
        ).hexdigest(),
        "image_max_pixels": 262_144,
        "model_path": "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
        "tokenizer_length": 151_669,
        "success_environment_text_sha256": (
            validator._base.EXPECTED_CONTINUATION_SHA256
        ),
        "success_environment_token_count": 60,
        "success_environment_token_ids_sha256": (
            validator.CANARY_TOKEN_IDS_SHA256
        ),
        "native_image_placeholder_count": 1,
        "return_code": 0,
        "pytest_exact_pass_count": 2,
        "pytest_skipped_count": 0,
        "two_crop_runtime_layout_appender_passed": True,
        "final_recorded_visual_replay_passed": True,
        "one_token_negative_drift_rejected": True,
        "model_weights_loaded": False,
        "vllm_or_ray_started": False,
        "accelerator_environment": validator.CANARY_ACCELERATOR_ENVIRONMENT,
        "accelerator_environment_sha256": _canonical_sha256(
            validator.CANARY_ACCELERATOR_ENVIRONMENT
        ),
    }
    payload["receipt_identity_sha256"] = _canonical_sha256(payload)
    receipt = tmp_path / "canary.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    assert validator.validate_canary_receipt(path=receipt, admitted_head=head) == payload
    payload["test_file_sha256"] = "0" * 64
    payload["receipt_identity_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_identity_sha256"
        }
    )
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validator.validate_canary_receipt(path=receipt, admitted_head=head)
    except RuntimeError as error:
        assert "proof differs" in str(error)
    else:
        raise AssertionError("tampered canary receipt was accepted")

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    try:
        validator.validate_canary_receipt(path=symlink, admitted_head=head)
    except RuntimeError as error:
        assert "absent or unsafe" in str(error)
    else:
        raise AssertionError("symlink canary receipt was accepted")
