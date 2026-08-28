#!/usr/bin/env python3
"""Fail-closed handoff binder for the two PRL-26 Train@512 S32 evaluations.

The permanent checkpoint receipts do not exist when the repository-side
evaluation code is authored.  This command therefore runs only after both S32
checkpoints close, validates the complete Policy-E2E checkpoint pairs, and
writes the immutable Crop evaluator plan plus a two-arm handoff receipt.  No
checkpoint hash is guessed or represented by a repository placeholder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_prl15_paired_evaluation import _policy_checkpoint_receipt  # noqa: E402
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    paired_evaluation_rng_contract,
)
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
NOTOOL_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_26_a_qwen3_instruct_full_no_tool_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
)
CROP_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_26_b_qwen3_instruct_full_crop_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
)
PROTOCOL_CONFIG = REPOSITORY_ROOT / (
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
CROP_EVALUATION_ID = (
    "PRL26-B-TRAIN512-S32-CROP-MATCHED-COREDEV2511-PIXEL512-BOUNDARYFIX-V1"
)
NOTOOL_EVALUATION_ID = "PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1"
CONTROL_ROOT = MAIN_ROOT / (
    "artifacts/evaluation/PRL26-TRAIN512-S32-PIXEL512-COREDEV2511-V1"
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
            raise RuntimeError(f"immutable PRL-26 evaluation handoff differs: {path}")
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
                    f"immutable PRL-26 evaluation handoff differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _validate_formal_owner(
    config_path: Path, *, schema: str, expected_tool_profile: str
) -> tuple[Any, dict[str, Any], Path]:
    run = load_policy_e2e_smoke_run_config(
        config_path.resolve(), allow_external_agent_loop_config=True
    )
    if (
        run.schema_version != schema
        or run.policy.image_max_pixels != PIXEL512
        or run.policy.sampling.temperature != 1.0
        or run.policy.sampling.do_sample is not True
        or run.rollout_rng.master_seed != 42
        or run.protocol.tool_profile.value != expected_tool_profile
        or run.training.maximum_optimizer_steps != OPTIMIZER_STEP
        or OPTIMIZER_STEP not in run.training.permanent_checkpoint_steps
        or run.distributed.world_size != 8
    ):
        raise RuntimeError("PRL-26 Train@512 S32 owner contract differs")
    checkpoint = run.output.root / f"permanent-checkpoints/global_step_{OPTIMIZER_STEP}"
    receipt, receipt_path = _policy_checkpoint_receipt(
        run, checkpoint=checkpoint, optimizer_step=OPTIMIZER_STEP
    )
    expected_receipt = checkpoint / "tgvf_permanent_checkpoint_receipt.json"
    if receipt_path != expected_receipt.resolve():
        raise RuntimeError("PRL-26 S32 completion receipt path differs")
    metrics = [
        json.loads(line)
        for line in run.output.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row.get("optimizer_step") for row in metrics[:OPTIMIZER_STEP]] != list(
        range(1, OPTIMIZER_STEP + 1)
    ):
        raise RuntimeError("PRL-26 S32 metrics are incomplete or non-contiguous")
    return run, receipt, receipt_path


def _crop_protocol_sha256(run: Any, *, task_manifest_sha256: str) -> str:
    namespace = (
        "coredev2511-official-v1/prl26-b-train512-s32-crop/pixel262144/temp1/seed42/v1"
    )
    config = SimpleNamespace(
        evaluation_protocol=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        paired_seed_namespace=namespace,
        paired_rng_protocol_projection=None,
        evaluation_image_max_pixels=PIXEL512,
    )
    snapshot = SimpleNamespace(
        run=SimpleNamespace(
            model=run.model,
            policy=run.policy,
            rollout_rng=run.rollout_rng,
        ),
        policy_version=SimpleNamespace(optimizer_step=OPTIMIZER_STEP),
    )
    contract = paired_evaluation_rng_contract(
        config, snapshot, task_manifest_sha256=task_manifest_sha256
    )
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != "tgvf-policy-paired-evaluation-rng-v1"
        or contract.get("master_seed") != 42
    ):
        raise RuntimeError("PRL-26 Crop paired RNG contract differs")
    return str(contract["protocol_sha256"])


def _crop_plan(*, run: Any, completion_path: Path, protocol: Any) -> dict[str, object]:
    task_sha256 = _sha256(TASKS)
    protocol_payload = protocol.payload["protocol"]
    namespace = (
        "coredev2511-official-v1/prl26-b-train512-s32-crop/pixel262144/temp1/seed42/v1"
    )
    return {
        "schema_version": "tgvf.paired-policy-benchmark-plan.v3",
        "evaluation_id": CROP_EVALUATION_ID,
        "status": "ready",
        "checkpoint_owner": {
            "contract_type": ("policy_e2e_crop_exact_pixel512_parity_run_config_v1"),
            "config_path": str(CROP_CONFIG.relative_to(REPOSITORY_ROOT)),
            "config_sha256": _sha256(CROP_CONFIG),
            "run_id": run.run_id,
            "run_identity_sha256": run.identity_sha256,
            "output_root": str(run.output.root.resolve()),
            "checkpoint_world_size": run.distributed.world_size,
            "completion_path": str(completion_path),
            "completion_sha256": _sha256(completion_path),
        },
        "protocol_contract": {
            "contract_type": "deepeyes_native_crop_v1",
            "config_path": str(PROTOCOL_CONFIG.relative_to(REPOSITORY_ROOT)),
            "config_sha256": _sha256(PROTOCOL_CONFIG),
            "run_id": protocol.run_id,
            "run_identity_sha256": protocol.identity_sha256,
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
            "seed_namespace": namespace,
            "master_seed": 42,
            "task_manifest_sha256": task_sha256,
            "protocol_sha256": _crop_protocol_sha256(
                run, task_manifest_sha256=task_sha256
            ),
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
        "protocol": {
            "evaluation_protocol": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
            "visual_prompt_bundle_sha256": protocol_payload[
                "visual_prompt_bundle_sha256"
            ],
            "tool_name": protocol_payload["tool_name"],
            "tool_parser": protocol_payload["tool_parser"],
            "maximum_tool_calls": protocol_payload["max_active_perception"],
            "native_pixels": True,
            "sampling_source": "bound_protocol_contract",
            "same_tasks_and_rank_partition": True,
        },
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
            "run_id_prefix": "T20260829-PRL26-B-TRAIN512-S32-CROP-PIXEL512",
        },
    }


def bind(*, crop_plan_output: Path, handoff_output: Path) -> dict[str, object]:
    no_tool, no_tool_receipt, no_tool_receipt_path = _validate_formal_owner(
        NOTOOL_CONFIG,
        schema=POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        expected_tool_profile="no_tool",
    )
    crop, crop_receipt, crop_receipt_path = _validate_formal_owner(
        CROP_CONFIG,
        schema=POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        expected_tool_profile="crop_only",
    )
    protocol = load_deepeyes_native_run_contract(PROTOCOL_CONFIG.resolve())
    plan = _crop_plan(run=crop, completion_path=crop_receipt_path, protocol=protocol)
    _write_immutable_json(crop_plan_output, plan)
    content: dict[str, object] = {
        "schema_version": "tgvf.prl26-train512-s32-evaluation-handoff.v1",
        "status": "ready",
        "train_image_max_pixels": PIXEL512,
        "evaluation_image_max_pixels": PIXEL512,
        "optimizer_step": OPTIMIZER_STEP,
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "datasets": 7,
        },
        "no_tool": {
            "evaluation_id": NOTOOL_EVALUATION_ID,
            "run_id": no_tool.run_id,
            "run_identity_sha256": no_tool.identity_sha256,
            "config_path": str(NOTOOL_CONFIG.resolve()),
            "config_file_sha256": _sha256(NOTOOL_CONFIG),
            "completion_path": str(no_tool_receipt_path),
            "completion_file_sha256": _sha256(no_tool_receipt_path),
            "checkpoint_pair_integrity_sha256": no_tool_receipt.get(
                "pair_integrity_sha256"
            ),
        },
        "crop": {
            "evaluation_id": CROP_EVALUATION_ID,
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
            "tool_action_boundary": {
                "stop_strings": list(crop.policy.sampling.stop_strings),
                "include_stop_str_in_output": (
                    crop.policy.sampling.include_stop_str_in_output
                ),
            },
        },
    }
    payload = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_immutable_json(handoff_output, payload)
    return payload


def main() -> int:
    crop_root = Path(
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-26-B-train512-s32-parity-crop-qwen3-instruct-bs16-n16-teacher25-"
        f"ws8/evaluation/{CROP_EVALUATION_ID}"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop-plan-output",
        type=Path,
        default=crop_root / "runtime/bound-crop-plan.json",
    )
    parser.add_argument(
        "--handoff-output",
        type=Path,
        default=CONTROL_ROOT / "runtime/bound-handoff.json",
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
