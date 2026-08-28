#!/usr/bin/env python3
"""Prepare one immutable PRL25-F No-Tool corrected true1M V2 arm.

The V1 evaluation accidentally let the Qwen fast processor retain its
16,777,216-pixel default during decode.  This preparer deliberately creates a
new evaluation root and identity.  It may reference the already verified
full-model snapshot and materialization receipt (weight-only artifacts), but
it never copies or admits V1 inference/scoring rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_full_model_policy_benchmark_config,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    TRAINING_RUN_EVALUATION_PROTOCOL,
)
from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    FullModelSourceKind,
    load_full_model_materialization_receipt,
    load_full_model_snapshot_manifest,
)


PLAN_SCHEMA = "tgvf.prl25-f-no-tool-true1m-evaluation-plan.v2"
PLAN_PATH = REPOSITORY_ROOT / (
    "configs/evaluation/prl25_f_no_tool_s0_s8_s16_s32_true1m_v2_coredev2511_plan.json"
)
TRUE1M_MAX_PIXELS = 1_003_520
STEPS = (0, 8, 16, 32)
WORLD_SIZE_PER_ARM = 4
MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
PYTHON_ENVIRONMENT_ROOT = MAIN_ROOT / ".venv312"
PYTHON_HEADER_ROOT = MAIN_ROOT / ".deps/python312-dev/root/usr/include"
TOOLCHAIN_ENVIRONMENT = {
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "CPATH": os.pathsep.join(
        (str(PYTHON_HEADER_ROOT), str(PYTHON_HEADER_ROOT / "python3.12"))
    ),
    "LIBRARY_PATH": str(PYTHON_ENVIRONMENT_ROOT / "lib"),
    "PATH": os.pathsep.join(
        (
            str(PYTHON_ENVIRONMENT_ROOT / "bin"),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        )
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable output differs: {path}")
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
                raise RuntimeError(f"immutable output differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def load_true1m_v2_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "evaluation_id",
        "status",
        "checkpoint_owner",
        "evaluation_root",
        "weight_snapshot_reuse",
        "benchmark",
        "protocol",
        "image_preprocessing",
        "paired_rng",
        "arms",
        "execution",
        "scoring",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema_version") != PLAN_SCHEMA
        or payload.get("status") != "ready"
    ):
        raise ValueError("PRL25-F true1M V2 plan schema/status differs")

    owner = payload["checkpoint_owner"]
    owner_path = REPOSITORY_ROOT / owner["config_path"]
    if (
        set(owner)
        != {
            "config_path",
            "config_file_sha256",
            "run_id",
            "run_identity_sha256",
        }
        or not owner_path.is_file()
        or _sha256(owner_path) != owner["config_file_sha256"]
    ):
        raise RuntimeError("PRL25-F true1M V2 checkpoint owner differs")
    _require_sha256(owner["config_file_sha256"], name="owner config")
    _require_sha256(owner["run_identity_sha256"], name="owner run identity")

    reuse = payload["weight_snapshot_reuse"]
    reuse_root = Path(reuse["source_root"])
    if (
        set(reuse)
        != {
            "source_root",
            "allowed_files",
            "allowed_payload",
            "reuse_inference_rows",
            "reuse_scoring_rows",
        }
        or reuse["allowed_files"]
        != ["full-model-snapshot.json", "full-model-materialization.json"]
        or reuse["allowed_payload"]
        != "materialized_full_model_tree_referenced_by_receipt"
        or reuse["reuse_inference_rows"] is not False
        or reuse["reuse_scoring_rows"] is not False
        or reuse_root.name != "shared"
        or "inference" in reuse_root.parts
        or "scoring" in reuse_root.parts
    ):
        raise ValueError("PRL25-F true1M V2 weight-only reuse policy differs")

    benchmark = payload["benchmark"]
    tasks = Path(benchmark["task_manifest_path"])
    if (
        benchmark["expected_task_count"] != 2511
        or benchmark["expected_single_image_count"] != 2240
        or benchmark["unsupported_multi_image_count"] != 271
        or benchmark["datasets"]
        != [
            "VStarBench",
            "HRBench4K",
            "BLINK",
            "OCRBench_v2",
            "MMMU_Pro_10c",
            "MathVista_MINI",
            "MathVerse_MINI",
        ]
        or not tasks.is_file()
        or _sha256(tasks) != benchmark["task_manifest_sha256"]
    ):
        raise RuntimeError("PRL25-F true1M V2 benchmark closure differs")
    _require_sha256(benchmark["task_manifest_sha256"], name="task manifest")

    protocol = payload["protocol"]
    if (
        protocol.get("evaluation_protocol") != TRAINING_RUN_EVALUATION_PROTOCOL
        or protocol.get("prompt_contract") != "prl25-f-training-matched-no-tool"
        or protocol.get("tool_profile") != "no_tool"
        or protocol.get("enabled_tool_names") != []
        or protocol.get("system_tool_schema_visible") is not False
        or protocol.get("template_tools_argument") != []
        or protocol.get("native_pixels") is not True
    ):
        raise ValueError("PRL25-F true1M V2 protocol differs")
    _require_sha256(protocol.get("prompt_bundle_sha256"), name="prompt bundle")

    image = payload["image_preprocessing"]
    probe = image.get("prepare_validate_probe")
    if (
        image.get("configured_image_max_pixels") != TRUE1M_MAX_PIXELS
        or image.get("qwen_fast_processor_default_max_pixels") != 16_777_216
        or image.get("runtime_override_path") != "mm_processor_kwargs.size.longest_edge"
        or image.get("forbid_nested_images_kwargs") is not True
        or image.get("forbid_max_pixels_kwarg") is not True
        or probe
        != {
            "source_width": 2048,
            "source_height": 1536,
            "source_pixel_area": 3_145_728,
            "expected_represented_pixel_area": 995_328,
            "expected_visual_token_count": 972,
        }
    ):
        raise ValueError("PRL25-F true1M V2 processor contract differs")

    paired = payload["paired_rng"]
    if (
        paired.get("mode") != "common_random_numbers_per_task_turn"
        or paired.get("seed_namespace")
        != ("coredev2511/no-tool-rl-matched/s0-s8-s16-s32/temp1/seed42/true1m-v2")
        or paired.get("master_seed") != 42
        or paired.get("temperature") != 1.0
        or paired.get("do_sample") is not True
        or paired.get("rollout_index") != 0
    ):
        raise ValueError("PRL25-F true1M V2 paired RNG differs")

    expected_arms = [
        (0, 0, "left"),
        (8, 0, "right"),
        (16, 1, "left"),
        (32, 1, "right"),
    ]
    observed_arms = []
    for arm in payload["arms"]:
        step = arm.get("optimizer_step")
        if set(arm) != {
            "name",
            "optimizer_step",
            "evaluation_id",
            "concurrency_round",
            "gpu_group",
        }:
            raise ValueError("PRL25-F true1M V2 arm fields differ")
        if (
            arm["name"] != f"step{step}"
            or arm["evaluation_id"]
            != f"PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S{step}-TRUE1M-V2"
        ):
            raise ValueError("PRL25-F true1M V2 arm identity differs")
        observed_arms.append((step, arm["concurrency_round"], arm["gpu_group"]))
    if observed_arms != expected_arms:
        raise ValueError("PRL25-F true1M V2 arm schedule differs")

    execution = payload["execution"]
    if (
        execution.get("physical_gpu_count") != 8
        or execution.get("world_size_per_arm") != WORLD_SIZE_PER_ARM
        or execution.get("concurrent_arm_count") != 2
        or execution.get("inference_concurrency_per_gpu") != 8
        or execution.get("max_model_len") != 32768
        or execution.get("max_num_batched_tokens") != 32768
        or execution.get("enable_chunked_prefill") is not False
        or execution.get("gpu_memory_utilization") != 0.8
        or execution.get("rank_cache_environment")
        != ["VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
        or execution.get("toolchain_environment") != TOOLCHAIN_ENVIRONMENT
        or execution.get("resume_policy") != "identity_validated_rank_jsonl"
        or execution.get("failure_policy")
        != "terminate_active_process_groups_preserve_durable_rows"
    ):
        raise ValueError("PRL25-F true1M V2 execution contract differs")
    if payload["scoring"].get("entrypoint") != (
        "tools/supervise_prl25_f_no_tool_true1m_v2_scoring.sh"
    ):
        raise ValueError("PRL25-F true1M V2 scoring entrypoint differs")
    return payload


@dataclass(frozen=True, slots=True)
class ArmPaths:
    step: int
    evaluation_id: str
    output_root: Path
    config_path: Path
    source_shared_root: Path
    snapshot_manifest_path: Path
    materialization_receipt_path: Path
    reuse_receipt_path: Path


def arm_paths(plan: dict[str, Any], step: int) -> ArmPaths:
    if step not in STEPS:
        raise ValueError("PRL25-F true1M V2 step differs")
    arm = next(item for item in plan["arms"] if item["optimizer_step"] == step)
    evaluation_root = Path(plan["evaluation_root"])
    source_shared = Path(plan["weight_snapshot_reuse"]["source_root"]) / (f"step{step}")
    output_root = evaluation_root / f"matched/step{step}"
    return ArmPaths(
        step=step,
        evaluation_id=arm["evaluation_id"],
        output_root=output_root,
        config_path=output_root / "config.json",
        source_shared_root=source_shared,
        snapshot_manifest_path=source_shared / "full-model-snapshot.json",
        materialization_receipt_path=(
            source_shared / "full-model-materialization.json"
        ),
        reuse_receipt_path=output_root / "runtime/weight-snapshot-reuse.json",
    )


def _validate_gpu_ids(gpu_ids: tuple[int, ...]) -> None:
    if (
        len(gpu_ids) != WORLD_SIZE_PER_ARM
        or len(set(gpu_ids)) != WORLD_SIZE_PER_ARM
        or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
    ):
        raise ValueError("one PRL25-F true1M V2 arm requires four distinct GPU IDs")


def _weight_reuse_receipt(plan: dict[str, Any], paths: ArmPaths) -> dict[str, object]:
    manifest = load_full_model_snapshot_manifest(paths.snapshot_manifest_path)
    receipt = load_full_model_materialization_receipt(
        paths.materialization_receipt_path
    )
    owner = manifest.checkpoint_owner
    if (
        manifest.optimizer_step != paths.step
        or owner is None
        or owner.run_id != plan["checkpoint_owner"]["run_id"]
        or owner.run_identity_sha256 != plan["checkpoint_owner"]["run_identity_sha256"]
        or receipt.snapshot_identity_sha256 != manifest.identity_sha256
    ):
        raise RuntimeError("reused PRL25-F full-model snapshot identity differs")
    model_path = Path(receipt.model_path).resolve(strict=True)
    if paths.step == 0:
        if manifest.source_kind is not FullModelSourceKind.BASE_HF:
            raise RuntimeError("PRL25-F S0 reuse is not the bound base model")
    elif (
        manifest.source_kind is not FullModelSourceKind.VERL_FSDP
        or not model_path.is_relative_to(paths.source_shared_root.resolve())
    ):
        raise RuntimeError("PRL25-F trained-step reuse escapes old shared weights")
    for path in (
        paths.snapshot_manifest_path,
        paths.materialization_receipt_path,
        model_path,
    ):
        if "inference" in path.parts or "scoring" in path.parts:
            raise RuntimeError("V1 inference/scoring artifact was offered for reuse")
    return {
        "schema_version": "tgvf.prl25-f-weight-only-reuse-receipt.v1",
        "destination_evaluation_id": paths.evaluation_id,
        "destination_output_root": str(paths.output_root),
        "optimizer_step": paths.step,
        "reuse_kind": "full_model_snapshot_and_materialized_weight_tree_only",
        "source_snapshot_manifest_path": str(paths.snapshot_manifest_path),
        "source_snapshot_manifest_file_sha256": _sha256(paths.snapshot_manifest_path),
        "source_snapshot_identity_sha256": manifest.identity_sha256,
        "source_materialization_receipt_path": str(paths.materialization_receipt_path),
        "source_materialization_receipt_file_sha256": _sha256(
            paths.materialization_receipt_path
        ),
        "source_materialization_identity_sha256": receipt.identity_sha256,
        "source_model_path": str(model_path),
        "source_model_tree_sha256": receipt.model_tree_sha256,
        "inference_rows_reused": False,
        "scoring_rows_reused": False,
    }


def prepare_arm(*, step: int, gpu_ids: tuple[int, ...]) -> Path:
    _validate_gpu_ids(gpu_ids)
    plan = load_true1m_v2_plan()
    paths = arm_paths(plan, step)
    reuse_receipt = _weight_reuse_receipt(plan, paths)
    benchmark = plan["benchmark"]
    execution = plan["execution"]
    owner_config = REPOSITORY_ROOT / plan["checkpoint_owner"]["config_path"]
    payload = materialize_full_model_policy_benchmark_config(
        evaluation_id=paths.evaluation_id,
        policy_config_path=owner_config,
        snapshot_manifest_path=paths.snapshot_manifest_path,
        materialization_receipt_path=paths.materialization_receipt_path,
        expected_optimizer_step=step,
        task_manifest_path=benchmark["task_manifest_path"],
        expected_task_count=benchmark["expected_task_count"],
        expected_single_image_count=benchmark["expected_single_image_count"],
        output_root=paths.output_root,
        config_path=paths.config_path,
        inference_concurrency_per_gpu=execution["inference_concurrency_per_gpu"],
        max_model_len=execution["max_model_len"],
        max_num_batched_tokens=execution["max_num_batched_tokens"],
        enable_chunked_prefill=execution["enable_chunked_prefill"],
        gpu_memory_utilization=execution["gpu_memory_utilization"],
        gpu_ids=gpu_ids,
        paired_seed_namespace=plan["paired_rng"]["seed_namespace"],
        evaluation_image_max_pixels=TRUE1M_MAX_PIXELS,
        evaluation_protocol=TRAINING_RUN_EVALUATION_PROTOCOL,
    )
    if (
        payload.get("evaluation_image_max_pixels") != TRUE1M_MAX_PIXELS
        or payload.get("output_root") != str(paths.output_root.resolve())
        or payload.get("evaluation_id") != paths.evaluation_id
    ):
        raise RuntimeError("materialized PRL25-F true1M V2 config differs")
    _write_immutable_json(paths.reuse_receipt_path, reuse_receipt)
    return paths.config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=WORLD_SIZE_PER_ARM, required=True)
    args = parser.parse_args()
    config = prepare_arm(step=args.step, gpu_ids=tuple(args.gpu_ids))
    print(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
