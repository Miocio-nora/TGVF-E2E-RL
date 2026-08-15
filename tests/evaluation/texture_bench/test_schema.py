from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.evaluation.texture_bench.schema import (
    PipelineArm,
    PipelineBackend,
    PipelineKind,
    TextureBenchmarkMatrix,
    VisionPreprocessConfig,
    load_texture_benchmark_matrix,
)


def _arms(tmp_path: Path) -> tuple[PipelineArm, ...]:
    model = tmp_path / "model"
    model.mkdir()
    pointer = tmp_path / "pointer.json"
    pointer.write_text("{}\n", encoding="utf-8")
    tool_arms = []
    for kind in (PipelineKind.CROP, PipelineKind.TGVF, PipelineKind.TGVF_CROP):
        policy = tmp_path / f"policy-{kind.value}.toml"
        policy.write_text(
            f"[protocol]\ntool_profile='{kind.policy_tool_profile}'\n",
            encoding="utf-8",
        )
        tool_arms.append(
            PipelineArm(
                kind.value,
                kind,
                PipelineBackend.POLICY_BENCHMARK,
                policy_config_path=policy,
                lora_pointer_path=pointer,
                expected_optimizer_step=1,
                evaluation_protocol="training_run",
            )
        )
    return (
        PipelineArm(
            "original", PipelineKind.ORIGINAL, PipelineBackend.STOCK_QWEN_VLLM, model
        ),
        *tool_arms,
    )


def test_complete_matrix_has_every_pipeline_and_shared_512_area_cap(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    matrix = TextureBenchmarkMatrix(
        matrix_id="paired",
        task_manifest_path=tasks.resolve(),
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=1,
        output_root=(tmp_path / "out").resolve(),
        arms=_arms(tmp_path),
    )
    assert matrix.vision == VisionPreprocessConfig(
        min_pixels=65536,
        max_pixels=262144,
        preserve_aspect_ratio=True,
        pre_resize_assets=False,
        qwen_patch_size=16,
        qwen_merge_size=2,
    )
    assert matrix.vision.qwen_resize_factor == 32
    assert matrix.complete_four_arm_matrix is True
    assert len(matrix.identity_sha256) == 64
    matrix.validate_files()
    resolved = matrix.resolved_arm_identities()
    assert len(resolved) == 4
    assert resolved[0]["directory_identity"] == {
        "model_path": "deferred_to_backend_tree_identity"
    }
    assert len(resolved[1]["file_sha256s"]["policy_config_path"]) == 64


def test_matrix_loader_round_trips_json(tmp_path: Path) -> None:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    tasks.write_text("{}\n", encoding="utf-8")
    matrix = TextureBenchmarkMatrix(
        matrix_id="paired",
        task_manifest_path=tasks,
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=1,
        output_root=(tmp_path / "out").resolve(),
        arms=_arms(tmp_path),
    )
    config = tmp_path / "matrix.json"
    config.write_text(
        json.dumps(matrix.identity_payload(), indent=2) + "\n", encoding="utf-8"
    )
    loaded = load_texture_benchmark_matrix(config)
    assert loaded == matrix


def test_matrix_supports_staged_subset_but_rejects_duplicate_pipeline_kind(
    tmp_path: Path,
) -> None:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    tasks.write_text("{}\n", encoding="utf-8")
    arms = _arms(tmp_path)
    partial = TextureBenchmarkMatrix(
        matrix_id="partial",
        task_manifest_path=tasks,
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=1,
        output_root=(tmp_path / "partial-out").resolve(),
        arms=arms[:-1],
    )
    assert partial.missing_pipeline_kinds == (PipelineKind.TGVF_CROP,)
    with pytest.raises(ValueError, match="complete texture comparison"):
        partial.require_complete_arms()

    with pytest.raises(ValueError, match="at most one"):
        TextureBenchmarkMatrix(
            matrix_id="bad",
            task_manifest_path=tasks,
            task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
            task_count=1,
            output_root=(tmp_path / "out").resolve(),
            arms=arms[:-1] + (arms[-2],),
        )


def test_vision_contract_rejects_square_prerender_semantics() -> None:
    with pytest.raises(ValueError, match="native resolution"):
        VisionPreprocessConfig(pre_resize_assets=True)
    with pytest.raises(ValueError, match="preserve aspect ratio"):
        VisionPreprocessConfig(preserve_aspect_ratio=False)


def test_matrix_rejects_invalid_physical_gpu_map(tmp_path: Path) -> None:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    tasks.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="four distinct"):
        TextureBenchmarkMatrix(
            matrix_id="bad-gpus",
            task_manifest_path=tasks,
            task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
            task_count=1,
            output_root=(tmp_path / "out").resolve(),
            arms=_arms(tmp_path),
            gpu_ids=(0, 0, 1, 2),
        )


def test_tool_arm_accepts_current_paired_qwen_rp66_snapshot(tmp_path: Path) -> None:
    policy = (tmp_path / "policy.toml").resolve()
    qwen = (tmp_path / "qwen").resolve()
    pointer = (tmp_path / "pointer.json").resolve()
    policy.write_text("[protocol]\ntool_profile='tgvf_only'\n", encoding="utf-8")
    qwen.mkdir()
    pointer.write_text("{}\n", encoding="utf-8")

    arm = PipelineArm(
        "tgvf-step8",
        PipelineKind.TGVF,
        PipelineBackend.POLICY_BENCHMARK,
        policy_config_path=policy,
        paired_qwen_model_path=qwen,
        paired_rp66_pointer_path=pointer,
        paired_snapshot_receipt_path=(tmp_path / "receipt.json").resolve(),
        expected_optimizer_step=8,
        evaluation_protocol="training_run",
    )

    assert arm.paired_qwen_model_path == qwen
    with pytest.raises(ValueError, match="requires an RP66 pointer"):
        PipelineArm(
            "bad",
            PipelineKind.TGVF,
            PipelineBackend.POLICY_BENCHMARK,
            policy_config_path=policy,
            paired_qwen_model_path=qwen,
            paired_snapshot_receipt_path=(tmp_path / "bad-receipt.json").resolve(),
            expected_optimizer_step=8,
            evaluation_protocol="training_run",
        )


def test_matrix_rejects_policy_tool_profile_mismatch(tmp_path: Path) -> None:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    tasks.write_text("{}\n", encoding="utf-8")
    arms = list(_arms(tmp_path))
    arms[1] = PipelineArm(
        "crop-wrong-policy",
        PipelineKind.CROP,
        PipelineBackend.POLICY_BENCHMARK,
        policy_config_path=arms[2].policy_config_path,
        lora_pointer_path=arms[2].lora_pointer_path,
        expected_optimizer_step=1,
        evaluation_protocol="training_run",
    )
    matrix = TextureBenchmarkMatrix(
        matrix_id="bad-profile",
        task_manifest_path=tasks,
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=1,
        output_root=(tmp_path / "out").resolve(),
        arms=tuple(arms),
    )

    with pytest.raises(ValueError, match="policy tool profile differs"):
        matrix.validate_files()


def test_full_model_arm_cannot_silently_ignore_declared_protocol(
    tmp_path: Path,
) -> None:
    policy = (tmp_path / "crop.toml").resolve()
    manifest = (tmp_path / "snapshot.json").resolve()
    receipt = (tmp_path / "receipt.json").resolve()

    with pytest.raises(ValueError, match="crop official-visible protocol"):
        PipelineArm(
            "crop",
            PipelineKind.CROP,
            PipelineBackend.POLICY_BENCHMARK,
            policy_config_path=policy,
            full_model_snapshot_manifest_path=manifest,
            full_model_materialization_receipt_path=receipt,
            expected_optimizer_step=8,
            evaluation_protocol="training_run",
        )


def test_full_model_v2_validates_owner_and_protocol_configs_separately(
    tmp_path: Path,
) -> None:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    owner = (tmp_path / "owner.toml").resolve()
    protocol = (tmp_path / "protocol.toml").resolve()
    manifest = (tmp_path / "snapshot.json").resolve()
    receipt = (tmp_path / "receipt.json").resolve()
    tasks.write_text("{}\n", encoding="utf-8")
    owner.write_text("run_id='PRL21'\n", encoding="utf-8")
    protocol.write_text(
        "[protocol]\ntool_name='image_zoom_in_tool'\n", encoding="utf-8"
    )
    receipt.write_text("{}\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "tgvf-full-model-snapshot-v2",
                "run_contract_path": str(protocol),
                "run_contract_file_sha256": hashlib.sha256(
                    protocol.read_bytes()
                ).hexdigest(),
                "checkpoint_owner": {
                    "config_path": str(owner),
                    "config_file_sha256": hashlib.sha256(
                        owner.read_bytes()
                    ).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = TextureBenchmarkMatrix(
        matrix_id="owner-aware-full-model",
        task_manifest_path=tasks,
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=1,
        output_root=(tmp_path / "out").resolve(),
        arms=(
            PipelineArm(
                "crop-prl21-step16",
                PipelineKind.CROP,
                PipelineBackend.POLICY_BENCHMARK,
                policy_config_path=owner,
                full_model_snapshot_manifest_path=manifest,
                full_model_materialization_receipt_path=receipt,
                expected_optimizer_step=16,
                evaluation_protocol="deepeyes_official_visible_native_crop_v1",
            ),
        ),
    )

    matrix.validate_files()

    owner.write_text("run_id='TAMPERED'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint owner config differs"):
        matrix.validate_files()
