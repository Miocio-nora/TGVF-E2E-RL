#!/usr/bin/env python3
"""Materialize one immutable PRL25-F matched no-tool CoreDev arm."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

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
    load_policy_e2e_smoke_run_config,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
TRAINING_ROOT = MAIN_ROOT / (
    "artifacts/policy/"
    "PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-"
    "32step-ws8"
)
EVALUATION_ROOT = TRAINING_ROOT / (
    "evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1"
)
OWNER_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_25_f_qwen3_instruct_full_no_tool_rl_bs16_n16_tfree_teacher25_"
    "32step_ws8.toml"
)
PROTOCOL_CONFIG = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl25-crop-aligned/"
    "configs/policy/runs/"
    "prl_24_d_base_qwen3_instruct_full_crop_teacher25_native_prl13.toml"
)
COMPLETION = TRAINING_ROOT / (
    "permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
)
TASKS = MAIN_ROOT / "artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(0, 8, 16, 32), required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=4, required=True)
    args = parser.parse_args()

    run = load_policy_e2e_smoke_run_config(
        OWNER_CONFIG, allow_external_agent_loop_config=True
    )
    protocol = load_deepeyes_native_run_contract(PROTOCOL_CONFIG)
    owner = FullModelCheckpointOwner(
        run_id=run.run_id,
        run_identity_sha256=run.identity_sha256,
        config_path=str(OWNER_CONFIG.resolve()),
        config_file_sha256=_sha256(OWNER_CONFIG),
        completion_path=str(COMPLETION.resolve()),
        completion_file_sha256=_sha256(COMPLETION),
    )
    source = (
        Path(run.model.revision_or_path)
        if args.step == 0
        else TRAINING_ROOT / f"permanent-checkpoints/global_step_{args.step}"
    )
    shared = EVALUATION_ROOT / f"shared/step{args.step}"
    manifest_path = shared / "full-model-snapshot.json"
    receipt_path = shared / "full-model-materialization.json"
    model_path = shared / "model"
    if manifest_path.is_file():
        manifest = load_full_model_snapshot_manifest(manifest_path)
    else:
        manifest = build_full_model_snapshot_manifest(
            protocol,
            source_path=source,
            optimizer_step=args.step,
            runtime_fsdp_world_size=8 if args.step else None,
            checkpoint_owner=owner,
        )
        write_full_model_snapshot_manifest(manifest_path, manifest)
    if not receipt_path.is_file():
        receipt = materialize_full_model_snapshot(
            manifest,
            target_dir=model_path if args.step else None,
        )
        write_full_model_materialization_receipt(receipt_path, receipt)

    arm_root = EVALUATION_ROOT / f"matched/step{args.step}"
    config_path = arm_root / "config.json"
    materialize_full_model_policy_benchmark_config(
        evaluation_id=f"PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S{args.step}-V1",
        policy_config_path=OWNER_CONFIG,
        snapshot_manifest_path=manifest_path,
        materialization_receipt_path=receipt_path,
        expected_optimizer_step=args.step,
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
        gpu_ids=tuple(args.gpu_ids),
        paired_seed_namespace=(
            "coredev2511/no-tool-rl-matched/s0-s8-s16-s32/temp1/seed42/v1"
        ),
        evaluation_protocol=TRAINING_RUN_EVALUATION_PROTOCOL,
    )
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
