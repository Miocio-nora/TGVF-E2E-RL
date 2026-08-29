#!/usr/bin/env python3
"""Bind the corrected PRL-27-A Crop S32 training-run evaluation.

This binder runs only after the PRL-27-A exact-continuation checkpoint is
complete, validates its exact Train@512 owner, and writes a new single-arm
evaluation plan whose prompt,
post-tool continuation, error/cap behavior, response budget, and action
boundary come from the checkpoint owner's training-run contract.  The older
official-visible evaluation remains a separate historical control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_prl15_paired_evaluation import (  # noqa: E402
    _policy_checkpoint_receipt,
    _training_run_crop_plan_protocol,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    TRAINING_RUN_EVALUATION_PROTOCOL,
)
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
CROP_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_27_a_qwen3_instruct_full_crop_train512_exact_continuation_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
MATERIALIZATION_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_24_d_base_qwen3_instruct_full_crop_teacher25_native_prl13.toml"
)
TASKS = MAIN_ROOT / "artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"
SOURCE_ROOT = Path(
    "/nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1"
)
MATHVERSE_SOURCE = Path(
    "/nvmesv/dredvpn009/datasets/benchmarks/mathverse/snapshot/testmini.json"
)
PIXEL512 = 262_144
OPTIMIZER_STEP = 32
EXPECTED_CODE_COMMIT = "ecddc379d392d154c91783d7651528b20d40afba"
CROP_EVALUATION_ID = (
    "PRL27-A-CROP-EXACT-CONTINUATION-TRAIN512-S32-MATCHED-"
    "COREDEV2511-PIXEL512-V1"
)
PAIRED_SEED_NAMESPACE = (
    "coredev2511/prl27-a/corrected-crop/training-run/"
    "train512-eval512/s32/temp1/seed42/v1"
)
CROP_OWNER_ROOT = MAIN_ROOT / (
    "artifacts/policy/"
    "PRL-27-A-train512-s32-crop-exact-continuation-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
EVALUATION_ROOT = CROP_OWNER_ROOT / "evaluation" / CROP_EVALUATION_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable PRL-27-A evaluation handoff differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(
                    f"immutable PRL-27-A evaluation handoff differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _validate_crop_owner_contract(run: Any) -> None:
    if (
        run.schema_version
        != POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        or run.code.commit != EXPECTED_CODE_COMMIT
        or run.policy.image_max_pixels != PIXEL512
        or run.policy.sampling.temperature != 1.0
        or run.policy.sampling.do_sample is not True
        or run.rollout_rng.master_seed != 42
        or run.protocol.tool_profile.value != "crop_only"
        or run.protocol.enabled_tool_names != ("image_zoom_in_tool",)
        or run.protocol.maximum_tool_calls != 6
        or run.training.maximum_optimizer_steps != OPTIMIZER_STEP
        or OPTIMIZER_STEP not in run.training.permanent_checkpoint_steps
        or run.distributed.world_size != 8
        or run.output.root.resolve() != CROP_OWNER_ROOT.resolve()
    ):
        raise RuntimeError("PRL-27-A corrected Crop owner contract differs")


def _validate_formal_crop_owner() -> tuple[Any, dict[str, Any], Path]:
    run = load_policy_e2e_smoke_run_config(
        CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    _validate_crop_owner_contract(run)
    checkpoint = run.output.root / f"permanent-checkpoints/global_step_{OPTIMIZER_STEP}"
    receipt, receipt_path = _policy_checkpoint_receipt(
        run, checkpoint=checkpoint, optimizer_step=OPTIMIZER_STEP
    )
    expected_receipt = checkpoint / "tgvf_permanent_checkpoint_receipt.json"
    if receipt_path != expected_receipt.resolve():
        raise RuntimeError("PRL-27-A Crop S32 completion receipt path differs")
    metrics = [
        json.loads(line)
        for line in run.output.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row.get("optimizer_step") for row in metrics[:OPTIMIZER_STEP]] != list(
        range(1, OPTIMIZER_STEP + 1)
    ):
        raise RuntimeError("PRL-27-A Crop S32 metrics are incomplete or non-contiguous")
    return run, receipt, receipt_path


def _corrected_protocol(run: Any) -> dict[str, object]:
    protocol = _training_run_crop_plan_protocol(run)
    identity = protocol.get("training_run_identity")
    if (
        protocol.get("evaluation_protocol") != TRAINING_RUN_EVALUATION_PROTOCOL
        or not isinstance(identity, dict)
        or identity.get("profile") != TRAINING_RUN_EVALUATION_PROTOCOL
        or identity.get("tool_profile") != "crop_only"
        or identity.get("enabled_tool_names") != ["image_zoom_in_tool"]
        or identity.get("maximum_tool_calls") != 6
        or identity.get("native_pixels") is not False
        or identity.get("precomputed_image_embeds") is not True
        or identity.get("success_environment_renderer")
        != "render_qwen_native_matched_crop_success_environment_text"
        or identity.get("cap_error_behavior") != "one_final_answer_turn"
        or identity.get("response_budget_scope") != "total_response_tokens"
        or identity.get("single_response_max_tokens") != 10_240
    ):
        raise RuntimeError("PRL-27-A corrected Crop training-run protocol differs")
    return protocol


def _crop_plan(
    *, run: Any, completion_path: Path, materialization_contract: Any
) -> dict[str, object]:
    task_sha256 = _sha256(TASKS)
    protocol = _corrected_protocol(run)
    protocol_identity = protocol["training_run_identity"]
    return {
        "schema_version": "tgvf.paired-policy-benchmark-plan.v3",
        "evaluation_id": CROP_EVALUATION_ID,
        "status": "ready",
        "checkpoint_owner": {
            "contract_type": (
                "policy_e2e_crop_exact_pixel512_parity_run_config_v1"
            ),
            "config_path": str(CROP_CONFIG.relative_to(REPOSITORY_ROOT)),
            "config_sha256": _sha256(CROP_CONFIG),
            "run_id": run.run_id,
            "run_identity_sha256": run.identity_sha256,
            "output_root": str(run.output.root.resolve()),
            "checkpoint_world_size": run.distributed.world_size,
            "completion_path": str(completion_path.resolve()),
            "completion_sha256": _sha256(completion_path),
        },
        "protocol_contract": {
            "contract_type": "deepeyes_native_crop_v1",
            "config_path": str(MATERIALIZATION_CONFIG.relative_to(REPOSITORY_ROOT)),
            "config_sha256": _sha256(MATERIALIZATION_CONFIG),
            "run_id": materialization_contract.run_id,
            "run_identity_sha256": materialization_contract.identity_sha256,
        },
        "snapshot": {
            "backend": "full_model",
            "inference_concurrency_per_gpu": 8,
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "enable_chunked_prefill": False,
            "gpu_memory_utilization": 0.8,
        },
        "task_manifest_path": str(TASKS),
        "task_manifest_sha256": task_sha256,
        "expected_task_count": 2511,
        "expected_single_image_count": 2240,
        "unsupported_multi_image_count": 271,
        "evaluation_image_max_pixels": PIXEL512,
        "paired_rng": {
            "schema_version": "tgvf.policy-paired-evaluation-rng-plan.v1",
            "mode": "common_random_numbers_per_task_turn",
            "seed_namespace": PAIRED_SEED_NAMESPACE,
            "master_seed": 42,
            "task_manifest_sha256": task_sha256,
            "protocol_sha256": _canonical_sha256(protocol_identity),
            "temperature": 1.0,
            "do_sample": True,
            "excluded_arm_components": [
                "evaluation_id",
                "arm_name",
                "optimizer_step",
                "checkpoint_hash",
                "policy_weights_sha256",
                "prompt_token_ids_sha256",
            ],
        },
        "arms": [
            {
                "name": "step32",
                "optimizer_step": OPTIMIZER_STEP,
                "evaluation_id": CROP_EVALUATION_ID,
                "source": {
                    "kind": "owner_checkpoint",
                    "relative_path": "permanent-checkpoints/global_step_32",
                },
            }
        ],
        "protocol": protocol,
        "scoring": {
            "datasets": [
                "VStarBench",
                "HRBench4K",
                "BLINK",
                "OCRBench_v2",
                "MMMU_Pro_10c",
                "MathVista_MINI",
                "MathVerse_MINI",
            ],
            "evaluated_model": "Qwen3-VL-8B-Instruct",
            "execution": {"mode": "eval", "reuse": True, "reuse_aux": "infer"},
            "gpt_fallback": False,
            "judge_api_nproc": 4,
            "judge_retry": 6,
            "judge_timeout_seconds": 600,
            "judge_config_path": "configs/evaluation/qwen25_72b_judge_service_v1.json",
            "judge_config_sha256": _sha256(
                REPOSITORY_ROOT / "configs/evaluation/qwen25_72b_judge_service_v1.json"
            ),
            "pinned_artifacts_config_path": (
                "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
            ),
            "pinned_artifacts_config_sha256": _sha256(
                REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
            ),
            "vlmevalkit_deployment_config_path": (
                "configs/evaluation/vlmevalkit_deployment_v1.json"
            ),
            "vlmevalkit_deployment_config_sha256": _sha256(
                REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
            ),
            "source_root": str(SOURCE_ROOT),
            "view_name": "coredev-official-v1",
            "mathverse_source_json": str(MATHVERSE_SOURCE),
            "mathverse_source_sha256": _sha256(MATHVERSE_SOURCE),
            "run_id_prefix": (
                "T20260829-PRL27-A-CROP-EXACT-CONTINUATION-S32-PIXEL512"
            ),
        },
    }


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
            "tgvf.prl27-a-corrected-crop-training-run-evaluation-handoff.v1"
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
    parser = argparse.ArgumentParser()
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
