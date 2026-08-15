from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.evaluation import policy_benchmark_config as implementation
from tgvf_rl.evaluation.policy_coredev import load_policy_coredev_config


def test_config_materializer_binds_exact_pointer_and_task_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = tmp_path / "policy-state/latest-lora-snapshot.json"
    pointer.parent.mkdir()
    pointer.write_text("pointer\n", encoding="utf-8")
    policy_config = tmp_path / "policy.toml"
    policy_config.write_text("fixture\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    run = SimpleNamespace(run_id="PRL-11", identity_sha256="a" * 64)
    snapshot = SimpleNamespace(
        pointer_file_sha256="b" * 64,
        run_identity_sha256="a" * 64,
        policy_version=PolicyVersion("PRL-11", 8, "c" * 64),
    )
    monkeypatch.setattr(
        implementation,
        "load_policy_e2e_smoke_run_config",
        lambda path, **kwargs: run,
    )
    monkeypatch.setattr(
        implementation, "load_lora_snapshot_pointer", lambda *args, **kwargs: snapshot
    )
    observed_task_binding: dict[str, object] = {}

    def load_tasks(path: Path, **kwargs: object) -> tuple[object, ...]:
        observed_task_binding.update(kwargs)
        return (object(),)

    monkeypatch.setattr(implementation, "load_benchmark_tasks", load_tasks)
    destination = tmp_path / "configs/step8.json"

    payload = implementation.materialize_policy_benchmark_config(
        evaluation_id="DEEPEYES-STEP8",
        policy_config_path=policy_config,
        lora_pointer_path=pointer,
        expected_optimizer_step=8,
        task_manifest_path=tasks,
        expected_task_count=591,
        expected_single_image_count=591,
        output_root=tmp_path / "evaluation-step8",
        config_path=destination,
        image_max_pixels=512 * 512,
        gpu_ids=(4, 5, 6, 7),
    )

    assert payload["expected_optimizer_step"] == 8
    assert payload["expected_policy_run_id"] == "PRL-11"
    assert payload["expected_policy_run_identity_sha256"] == "a" * 64
    assert payload["lora_pointer_sha256"] == "b" * 64
    assert payload["expected_policy_weights_sha256"] == "c" * 64
    assert payload["image_max_pixels"] == 512 * 512
    assert payload["gpu_ids"] == [4, 5, 6, 7]
    loaded = load_policy_coredev_config(destination)
    assert loaded.effective_image_max_pixels(SimpleNamespace()) == 512 * 512
    assert (
        payload["task_manifest_sha256"]
        == hashlib.sha256(tasks.read_bytes()).hexdigest()
    )
    assert observed_task_binding["verify_image_contents"] is True
    assert json.loads(destination.read_text()) == payload
    frozen_policy_config = (
        tmp_path / "evaluation-step8/runtime/frozen-policy-config.toml"
    )
    assert payload["policy_config_path"] == str(frozen_policy_config)
    assert frozen_policy_config.read_bytes() == policy_config.read_bytes()

    with pytest.raises(RuntimeError, match="config differs"):
        implementation.materialize_policy_benchmark_config(
            evaluation_id="DIFFERENT",
            policy_config_path=policy_config,
            lora_pointer_path=pointer,
            expected_optimizer_step=8,
            task_manifest_path=tasks,
            expected_task_count=591,
            expected_single_image_count=591,
            output_root=tmp_path / "evaluation-step8",
            config_path=destination,
        )


def test_full_model_config_materializer_binds_manifest_receipt_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_config = tmp_path / "policy.toml"
    policy_config.write_text("full-model-run\n", encoding="utf-8")
    snapshot_manifest = tmp_path / "snapshot.json"
    snapshot_manifest.write_text("manifest\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("receipt\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    policy_config_sha256 = hashlib.sha256(policy_config.read_bytes()).hexdigest()
    snapshot = SimpleNamespace(
        manifest=SimpleNamespace(
            identity_sha256="d" * 64,
            run_contract_file_sha256=policy_config_sha256,
        ),
        run_identity_sha256="a" * 64,
        policy_version=PolicyVersion("PRL-13-A", 8, "c" * 64),
    )
    monkeypatch.setattr(
        implementation,
        "load_full_model_evaluation_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    observed_task_binding: dict[str, object] = {}

    def load_tasks(path: Path, **kwargs: object) -> tuple[object, ...]:
        observed_task_binding.update(kwargs)
        return (object(),)

    monkeypatch.setattr(implementation, "load_benchmark_tasks", load_tasks)
    output_root = tmp_path / "evaluation-step8"
    destination = tmp_path / "configs/step8.json"

    payload = implementation.materialize_full_model_policy_benchmark_config(
        evaluation_id="PRL13-A-COREDEV-STEP8",
        policy_config_path=policy_config,
        snapshot_manifest_path=snapshot_manifest,
        materialization_receipt_path=receipt,
        expected_optimizer_step=8,
        task_manifest_path=tasks,
        expected_task_count=2511,
        expected_single_image_count=2240,
        output_root=output_root,
        config_path=destination,
        gpu_ids=(4, 5, 6, 7),
        image_max_pixels=512 * 512,
    )

    assert payload["snapshot_backend"] == "full_model"
    assert payload["lora_pointer_path"] is None
    assert payload["expected_optimizer_step"] == 8
    assert payload["expected_policy_run_id"] == "PRL-13-A"
    assert payload["required_snapshot_identity_sha256"] == "d" * 64
    assert payload["gpu_ids"] == [4, 5, 6, 7]
    assert payload["image_max_pixels"] == 512 * 512
    assert "paired_seed_namespace" not in payload
    assert (
        payload["full_model_snapshot_manifest_sha256"]
        == hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest()
    )
    assert (
        payload["full_model_materialization_receipt_sha256"]
        == hashlib.sha256(receipt.read_bytes()).hexdigest()
    )
    assert observed_task_binding["require_explicit_sample_ids"] is True
    assert observed_task_binding["require_image_identities"] is True
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert (
        output_root / "runtime/frozen-policy-config.toml"
    ).read_bytes() == policy_config.read_bytes()

    paired_payload = implementation.materialize_full_model_policy_benchmark_config(
        evaluation_id="PRL13-A-COREDEV-STEP8-PAIRED",
        policy_config_path=policy_config,
        snapshot_manifest_path=snapshot_manifest,
        materialization_receipt_path=receipt,
        expected_optimizer_step=8,
        task_manifest_path=tasks,
        expected_task_count=2511,
        expected_single_image_count=2240,
        output_root=tmp_path / "evaluation-step8-paired",
        config_path=tmp_path / "configs/step8-paired.json",
        gpu_ids=(4, 5, 6, 7),
        paired_seed_namespace="coredev2511/crop/step8-step16/temp1/seed42/v1",
    )
    assert paired_payload["paired_seed_namespace"] == (
        "coredev2511/crop/step8-step16/temp1/seed42/v1"
    )

    with pytest.raises(ValueError, match="optimizer step"):
        implementation.materialize_full_model_policy_benchmark_config(
            evaluation_id="PRL13-A-COREDEV-STEP20",
            policy_config_path=policy_config,
            snapshot_manifest_path=snapshot_manifest,
            materialization_receipt_path=receipt,
            expected_optimizer_step=20,
            task_manifest_path=tasks,
            expected_task_count=2511,
            expected_single_image_count=2240,
            output_root=tmp_path / "evaluation-step20",
            config_path=tmp_path / "configs/step20.json",
        )


def test_paired_config_materializer_binds_texture_pixel_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_config = tmp_path / "policy.toml"
    policy_config.write_text("fixture\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "runtime/paired-receipt.json"

    def materialize_pair(**kwargs: object) -> SimpleNamespace:
        assert kwargs["optimizer_step"] == 8
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            run_id="PRL-18",
            run_identity_sha256="a" * 64,
            optimizer_step=8,
            combined_weights_sha256="b" * 64,
        )

    monkeypatch.setattr(
        implementation, "materialize_paired_tgvf_snapshot", materialize_pair
    )
    monkeypatch.setattr(implementation, "load_benchmark_tasks", lambda *a, **k: ())
    destination = tmp_path / "paired-config.json"

    payload = implementation.materialize_paired_tgvf_policy_benchmark_config(
        evaluation_id="texture-tgvf-step8",
        policy_config_path=policy_config,
        optimizer_step=8,
        qwen_model_path=tmp_path / "qwen",
        rp66_pointer_path=tmp_path / "rp66-pointer.json",
        paired_snapshot_receipt_path=receipt_path,
        task_manifest_path=tasks,
        expected_task_count=42_870,
        expected_single_image_count=42_870,
        output_root=tmp_path / "evaluation",
        config_path=destination,
        image_max_pixels=512 * 512,
        paired_seed_namespace="texture-suite-v1",
    )

    assert payload["snapshot_backend"] == "full_model_trainable_rp66"
    assert payload["image_max_pixels"] == 262_144
    assert payload["paired_seed_namespace"] == "texture-suite-v1"
    loaded = load_policy_coredev_config(destination)
    assert loaded.effective_image_max_pixels(SimpleNamespace()) == 262_144
