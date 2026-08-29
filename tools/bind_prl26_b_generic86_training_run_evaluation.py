#!/usr/bin/env python3
"""Bind PRL-26-B S32 to its owner-native generic Crop continuation."""

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
    PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT,
    PRL26_B_GENERIC_CROP_OWNER_CODE_COMMIT,
    PRL26_B_GENERIC_CROP_OWNER_RUN_ID,
    PRL26_B_GENERIC_CROP_TRAINING_LAUNCH_COMMIT,
    PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT,
    TRAINING_RUN_EVALUATION_PROTOCOL,
)
from tgvf_rl.environment.native_appender import (  # noqa: E402
    QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256,
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
    "prl_26_b_qwen3_instruct_full_crop_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
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
EVALUATION_ID = (
    "PRL26-B-S32-OWNER-GENERIC86-TRAINING-RUN-"
    "COREDEV2511-PIXEL512-V1"
)
PAIRED_SEED_NAMESPACE = (
    "coredev2511/prl26-b/owner-generic86/training-run/"
    "train512-eval512/s32/temp1/seed42/v1"
)
CROP_OWNER_ROOT = MAIN_ROOT / (
    "artifacts/policy/"
    "PRL-26-B-train512-s32-parity-crop-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
EVALUATION_ROOT = CROP_OWNER_ROOT / "evaluation" / EVALUATION_ID
LAUNCH_PROVENANCE = CROP_OWNER_ROOT / "launch-provenance.jsonl"
LAUNCH_PROVENANCE_SHA256 = (
    "2e53d0323606b5f18125272783dad024337c173135e8ce339de1fee7085785d7"
)
EXPECTED_GENERIC86_ENVIRONMENT_TEXT_SHA256 = (
    "72a2caecb47a2b775a4497e5846c244061d9455fbb4b9690d3501cbc2521e187"
)


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
            raise RuntimeError(f"immutable generic86 evaluation handoff differs: {path}")
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
                    f"immutable generic86 evaluation handoff differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _validate_launch_provenance() -> dict[str, Any]:
    if (
        LAUNCH_PROVENANCE.is_symlink()
        or not LAUNCH_PROVENANCE.is_file()
        or _sha256(LAUNCH_PROVENANCE) != LAUNCH_PROVENANCE_SHA256
    ):
        raise RuntimeError("PRL-26-B launch provenance bytes differ")
    rows = [
        json.loads(line)
        for line in LAUNCH_PROVENANCE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 1:
        raise RuntimeError("PRL-26-B launch provenance row count differs")
    row = rows[0]
    project = row.get("project")
    if (
        row.get("schema_version") != "tgvf.prl15-launch-provenance.v1"
        or row.get("mode") != "formal"
        or row.get("run_id") != PRL26_B_GENERIC_CROP_OWNER_RUN_ID
        or row.get("target_step") != OPTIMIZER_STEP
        or row.get("run_config_file_sha256") != _sha256(CROP_CONFIG)
        or not isinstance(project, dict)
        or project.get("commit") != PRL26_B_GENERIC_CROP_TRAINING_LAUNCH_COMMIT
        or project.get("clean") is not True
        or project.get("changes") != []
    ):
        raise RuntimeError("PRL-26-B launch provenance identity differs")
    return row


def _validate_formal_owner() -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    run = load_policy_e2e_smoke_run_config(
        CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    provenance = _validate_launch_provenance()
    if (
        run.schema_version
        != POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        or run.run_id != PRL26_B_GENERIC_CROP_OWNER_RUN_ID
        or run.code.commit != PRL26_B_GENERIC_CROP_OWNER_CODE_COMMIT
        or run.policy.image_max_pixels != PIXEL512
        or run.policy.sampling.temperature != 1.0
        or run.policy.sampling.do_sample is not True
        or run.rollout_rng.master_seed != 42
        or run.protocol.tool_profile.value != "crop_only"
        or run.protocol.enabled_tool_names != ("image_zoom_in_tool",)
        or run.protocol.maximum_tool_calls != 6
        or tuple(run.policy.sampling.stop_strings) != ("</tool_call>",)
        or run.policy.sampling.include_stop_str_in_output is not True
        or run.training.maximum_optimizer_steps != OPTIMIZER_STEP
        or OPTIMIZER_STEP not in run.training.permanent_checkpoint_steps
        or run.distributed.world_size != 8
        or run.output.root.resolve() != CROP_OWNER_ROOT.resolve()
        or provenance.get("run_identity_sha256") != run.identity_sha256
    ):
        raise RuntimeError("PRL-26-B generic86 checkpoint owner contract differs")
    checkpoint = run.output.root / f"permanent-checkpoints/global_step_{OPTIMIZER_STEP}"
    receipt, receipt_path = _policy_checkpoint_receipt(
        run, checkpoint=checkpoint, optimizer_step=OPTIMIZER_STEP
    )
    if receipt_path != (
        checkpoint / "tgvf_permanent_checkpoint_receipt.json"
    ).resolve():
        raise RuntimeError("PRL-26-B S32 receipt path differs")
    metrics = [
        json.loads(line)
        for line in run.output.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row.get("optimizer_step") for row in metrics[:OPTIMIZER_STEP]] != list(
        range(1, OPTIMIZER_STEP + 1)
    ):
        raise RuntimeError("PRL-26-B S32 metrics are incomplete or non-contiguous")
    return run, receipt, receipt_path, provenance


def _owner_generic86_protocol(run: Any) -> dict[str, object]:
    protocol = _training_run_crop_plan_protocol(
        run, training_run_variant=PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT
    )
    identity = protocol.get("training_run_identity")
    if (
        protocol.get("evaluation_protocol") != TRAINING_RUN_EVALUATION_PROTOCOL
        or protocol.get("training_run_variant")
        != PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT
        or not isinstance(identity, dict)
        or identity.get("profile") != TRAINING_RUN_EVALUATION_PROTOCOL
        or identity.get("tool_profile") != "crop_only"
        or identity.get("success_environment_renderer")
        != "render_qwen_native_success_environment_text"
        or identity.get("success_environment_text_sha256")
        != QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256
        or QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256
        != EXPECTED_GENERIC86_ENVIRONMENT_TEXT_SHA256
        or identity.get("success_environment_token_count")
        != PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT
        or identity.get("training_launch_project_commit")
        != PRL26_B_GENERIC_CROP_TRAINING_LAUNCH_COMMIT
        or identity.get("response_budget_scope") != "total_response_tokens"
        or identity.get("single_response_max_tokens") != 10_240
    ):
        raise RuntimeError("PRL-26-B owner-native generic86 protocol differs")
    return protocol


def _plan(
    *, run: Any, completion_path: Path, materialization_contract: Any
) -> dict[str, object]:
    task_sha256 = _sha256(TASKS)
    protocol = _owner_generic86_protocol(run)
    protocol_identity = protocol["training_run_identity"]
    return {
        "schema_version": "tgvf.paired-policy-benchmark-plan.v3",
        "evaluation_id": EVALUATION_ID,
        "status": "ready",
        "checkpoint_owner": {
            "contract_type": "policy_e2e_crop_exact_pixel512_parity_run_config_v1",
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
                "evaluation_id": EVALUATION_ID,
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
            "run_id_prefix": "T20260830-PRL26-B-S32-OWNER-GENERIC86-PIXEL512",
        },
    }


def bind(*, plan_output: Path, handoff_output: Path) -> dict[str, object]:
    run, receipt, receipt_path, provenance = _validate_formal_owner()
    materialization = load_deepeyes_native_run_contract(
        MATERIALIZATION_CONFIG.resolve()
    )
    plan = _plan(
        run=run,
        completion_path=receipt_path,
        materialization_contract=materialization,
    )
    _write_immutable_json(plan_output, plan)
    content: dict[str, object] = {
        "schema_version": "tgvf.prl26-b-generic86-training-run-evaluation-handoff.v1",
        "status": "ready",
        "evaluation_id": EVALUATION_ID,
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
            "run_id": run.run_id,
            "run_identity_sha256": run.identity_sha256,
            "config_path": str(CROP_CONFIG.resolve()),
            "config_file_sha256": _sha256(CROP_CONFIG),
            "completion_path": str(receipt_path),
            "completion_file_sha256": _sha256(receipt_path),
            "checkpoint_pair_integrity_sha256": receipt.get(
                "pair_integrity_sha256"
            ),
            "launch_provenance_path": str(LAUNCH_PROVENANCE),
            "launch_provenance_file_sha256": _sha256(LAUNCH_PROVENANCE),
            "training_launch_project_commit": provenance["project"]["commit"],
            "bound_plan_path": str(plan_output.resolve()),
            "bound_plan_file_sha256": _sha256(plan_output),
            "evaluation_protocol": TRAINING_RUN_EVALUATION_PROTOCOL,
            "training_run_variant": PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT,
            "environment_text_sha256": (
                QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256
            ),
            "environment_token_count": (
                PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT
            ),
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
        "--plan-output",
        type=Path,
        default=EVALUATION_ROOT / "runtime/bound-crop-plan.json",
    )
    parser.add_argument(
        "--handoff-output",
        type=Path,
        default=EVALUATION_ROOT / "runtime/bound-handoff.json",
    )
    args = parser.parse_args()
    payload = bind(
        plan_output=args.plan_output.resolve(),
        handoff_output=args.handoff_output.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
