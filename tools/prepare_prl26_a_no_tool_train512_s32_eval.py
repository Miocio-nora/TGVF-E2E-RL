#!/usr/bin/env python3
"""Materialize the matched NoTool Train@512 S32 CoreDev evaluator arm."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_prl15_paired_evaluation import _policy_checkpoint_receipt  # noqa: E402
from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_full_model_policy_benchmark_config,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    TRAINING_RUN_EVALUATION_PROTOCOL,
)
from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    FullModelCheckpointOwner,
    build_full_model_snapshot_manifest,
    load_full_model_snapshot_manifest,
    materialize_full_model_snapshot,
    write_full_model_materialization_receipt,
    write_full_model_snapshot_manifest,
)
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
OWNER_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_26_a_qwen3_instruct_full_no_tool_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
)
PROTOCOL_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_24_d_base_qwen3_instruct_full_crop_teacher25_native_prl13.toml"
)
TRAINING_ROOT = MAIN_ROOT / (
    "artifacts/policy/PRL-26-A-train512-s32-parity-notool-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
EVALUATION_ID = "PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1"
EVALUATION_ROOT = TRAINING_ROOT / f"evaluation/{EVALUATION_ID}"
TASKS = MAIN_ROOT / "artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"
OPTIMIZER_STEP = 32
IMAGE_MAX_PIXELS = 262_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(*, gpu_ids: tuple[int, int, int, int]) -> Path:
    if len(set(gpu_ids)) != 4 or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("NoTool Train@512 evaluation requires four distinct GPUs")
    run = load_policy_e2e_smoke_run_config(
        OWNER_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    if (
        run.schema_version != POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        or run.output.root.resolve() != TRAINING_ROOT.resolve()
        or run.policy.image_max_pixels != IMAGE_MAX_PIXELS
        or run.protocol.tool_profile.value != "no_tool"
        or run.protocol.enabled_tool_names
        or run.training.maximum_optimizer_steps != OPTIMIZER_STEP
        or OPTIMIZER_STEP not in run.training.permanent_checkpoint_steps
        or run.distributed.world_size != 8
    ):
        raise RuntimeError("PRL-26-A Train@512 S32 owner contract differs")
    checkpoint = TRAINING_ROOT / "permanent-checkpoints/global_step_32"
    _receipt, completion = _policy_checkpoint_receipt(
        run, checkpoint=checkpoint, optimizer_step=OPTIMIZER_STEP
    )
    owner = FullModelCheckpointOwner(
        run_id=run.run_id,
        run_identity_sha256=run.identity_sha256,
        config_path=str(OWNER_CONFIG.resolve()),
        config_file_sha256=_sha256(OWNER_CONFIG),
        completion_path=str(completion),
        completion_file_sha256=_sha256(completion),
    )
    protocol = load_deepeyes_native_run_contract(PROTOCOL_CONFIG.resolve())
    shared = EVALUATION_ROOT / "shared/step32"
    manifest_path = shared / "full-model-snapshot.json"
    receipt_path = shared / "full-model-materialization.json"
    model_path = shared / "model"
    expected_manifest = build_full_model_snapshot_manifest(
        protocol,
        source_path=checkpoint,
        optimizer_step=OPTIMIZER_STEP,
        runtime_fsdp_world_size=run.distributed.world_size,
        checkpoint_owner=owner,
    )
    if manifest_path.is_file():
        if load_full_model_snapshot_manifest(manifest_path) != expected_manifest:
            raise RuntimeError("existing PRL-26-A S32 snapshot manifest differs")
    else:
        write_full_model_snapshot_manifest(manifest_path, expected_manifest)
    if not receipt_path.is_file():
        materialization = materialize_full_model_snapshot(
            expected_manifest, target_dir=model_path
        )
        write_full_model_materialization_receipt(receipt_path, materialization)

    arm_root = EVALUATION_ROOT / "matched/step32"
    config_path = arm_root / "config.json"
    payload = materialize_full_model_policy_benchmark_config(
        evaluation_id=EVALUATION_ID,
        policy_config_path=OWNER_CONFIG,
        snapshot_manifest_path=manifest_path,
        materialization_receipt_path=receipt_path,
        expected_optimizer_step=OPTIMIZER_STEP,
        task_manifest_path=TASKS,
        expected_task_count=2511,
        expected_single_image_count=2240,
        output_root=arm_root,
        config_path=config_path,
        inference_concurrency_per_gpu=8,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.8,
        gpu_ids=gpu_ids,
        paired_seed_namespace=(
            "coredev2511/no-tool-rl-matched/prl26-a-train512-s32/"
            "pixel262144/temp1/seed42/v1"
        ),
        evaluation_image_max_pixels=IMAGE_MAX_PIXELS,
        evaluation_protocol=TRAINING_RUN_EVALUATION_PROTOCOL,
    )
    if (
        payload.get("evaluation_id") != EVALUATION_ID
        or payload.get("evaluation_image_max_pixels") != IMAGE_MAX_PIXELS
        or payload.get("gpu_ids") != list(gpu_ids)
    ):
        raise RuntimeError("materialized PRL-26-A evaluator config differs")
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=int, nargs=4, required=True)
    args = parser.parse_args()
    config = prepare(gpu_ids=tuple(args.gpu_ids))
    print(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
