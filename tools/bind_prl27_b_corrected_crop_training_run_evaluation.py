#!/usr/bin/env python3
"""Bind the independent PRL-27-B corrected Crop S32 evaluation.

PRL-27-A is immutable failed history.  This binder deliberately gives the
fresh-S0 byte-parity repair its own checkpoint owner, evaluation identity,
artifact root, and RNG namespace while reusing the already-audited single-arm
training-run plan construction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def _load_a_binder() -> ModuleType:
    path = REPOSITORY_ROOT / (
        "tools/bind_prl27_a_corrected_crop_training_run_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_prl27_a_corrected_crop_binder_base", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audited PRL-27-A binder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_a_binder()

MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
CROP_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_27_b_qwen3_instruct_full_crop_train512_replay_byte_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
MATERIALIZATION_CONFIG = _BASE.MATERIALIZATION_CONFIG
TASKS = _BASE.TASKS
SOURCE_ROOT = _BASE.SOURCE_ROOT
MATHVERSE_SOURCE = _BASE.MATHVERSE_SOURCE
PIXEL512 = 262_144
OPTIMIZER_STEP = 32
EXPECTED_CODE_COMMIT = "c448e583887e4e49b79fe52fefb4b42934cd787e"
CROP_EVALUATION_ID = (
    "PRL27-B-CROP-REPLAY-BYTE-PARITY-TRAIN512-S32-TRAINING-RUN-COREDEV2511-PIXEL512-V1"
)
PAIRED_SEED_NAMESPACE = (
    "coredev2511/prl27-b/crop-replay-byte-parity/training-run/"
    "train512-eval512/s32/temp1/seed42/v1"
)
CROP_OWNER_ROOT = MAIN_ROOT / (
    "artifacts/policy/"
    "PRL-27-B-train512-s32-crop-replay-byte-parity-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
EVALUATION_ROOT = CROP_OWNER_ROOT / "evaluation" / CROP_EVALUATION_ID

# The audited builder resolves these names in its own module globals.  Bind all
# owner-specific values once, with no environment-variable redirection surface.
for _name, _value in {
    "CROP_CONFIG": CROP_CONFIG,
    "EXPECTED_CODE_COMMIT": EXPECTED_CODE_COMMIT,
    "CROP_EVALUATION_ID": CROP_EVALUATION_ID,
    "PAIRED_SEED_NAMESPACE": PAIRED_SEED_NAMESPACE,
    "CROP_OWNER_ROOT": CROP_OWNER_ROOT,
    "EVALUATION_ROOT": EVALUATION_ROOT,
}.items():
    setattr(_BASE, _name, _value)

_sha256 = _BASE._sha256
_canonical_sha256 = _BASE._canonical_sha256
_write_immutable_json = _BASE._write_immutable_json
_validate_crop_owner_contract = _BASE._validate_crop_owner_contract
_validate_formal_crop_owner = _BASE._validate_formal_crop_owner
_corrected_protocol = _BASE._corrected_protocol
_BASE_CROP_PLAN = _BASE._crop_plan
load_deepeyes_native_run_contract = _BASE.load_deepeyes_native_run_contract
TRAINING_RUN_EVALUATION_PROTOCOL = _BASE.TRAINING_RUN_EVALUATION_PROTOCOL


def _crop_plan(
    *, run: Any, completion_path: Path, materialization_contract: Any
) -> dict[str, object]:
    """Build the audited plan and close its last PRL-27-A literal."""

    plan = _BASE_CROP_PLAN(
        run=run,
        completion_path=completion_path,
        materialization_contract=materialization_contract,
    )
    scoring = plan.get("scoring")
    if not isinstance(scoring, dict):
        raise RuntimeError("PRL-27-B scoring plan is absent")
    scoring["run_id_prefix"] = "T20260830-PRL27-B-CROP-REPLAY-BYTE-PARITY-S32-PIXEL512"
    return plan


def bind(*, crop_plan_output: Path, handoff_output: Path) -> dict[str, object]:
    crop, crop_receipt, crop_receipt_path = _validate_formal_crop_owner()
    materialization_contract = load_deepeyes_native_run_contract(
        MATERIALIZATION_CONFIG.resolve()
    )
    plan = _crop_plan(
        run=crop,
        completion_path=crop_receipt_path,
        materialization_contract=materialization_contract,
    )
    _write_immutable_json(crop_plan_output, plan)
    content: dict[str, object] = {
        "schema_version": (
            "tgvf.prl27-b-crop-replay-byte-parity-training-run-evaluation-handoff.v1"
        ),
        "status": "ready",
        "evaluation_id": CROP_EVALUATION_ID,
        "evaluation_root": str(EVALUATION_ROOT),
        "train_image_max_pixels": PIXEL512,
        "evaluation_image_max_pixels": PIXEL512,
        "optimizer_step": OPTIMIZER_STEP,
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "datasets": 7,
        },
        "crop": {
            "run_id": crop.run_id,
            "run_identity_sha256": crop.identity_sha256,
            "config_path": str(CROP_CONFIG.resolve()),
            "config_file_sha256": _sha256(CROP_CONFIG),
            "completion_path": str(crop_receipt_path),
            "completion_file_sha256": _sha256(crop_receipt_path),
            "checkpoint_pair_integrity_sha256": crop_receipt.get(
                "pair_integrity_sha256"
            ),
            "bound_plan_path": str(crop_plan_output.resolve()),
            "bound_plan_file_sha256": _sha256(crop_plan_output),
            "evaluation_protocol": TRAINING_RUN_EVALUATION_PROTOCOL,
            "protocol_sha256": plan["paired_rng"]["protocol_sha256"],
            "paired_seed_namespace": PAIRED_SEED_NAMESPACE,
        },
    }
    payload = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_immutable_json(handoff_output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-plan-output",
        type=Path,
        default=EVALUATION_ROOT / "runtime/bound-crop-plan.json",
    )
    parser.add_argument(
        "--handoff-output",
        type=Path,
        default=EVALUATION_ROOT / "runtime/bound-handoff.json",
    )
    args = parser.parse_args()
    result = bind(
        crop_plan_output=args.crop_plan_output.resolve(),
        handoff_output=args.handoff_output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
