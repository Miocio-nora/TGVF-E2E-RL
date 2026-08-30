from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tgvf_rl.evaluation.policy_coredev as policy_coredev
import tgvf_rl.evaluation.policy_full_model_snapshot as full_model_snapshot
from tgvf_rl.evaluation.policy_full_model_snapshot import (
    FULL_MODEL_EVALUATION_BACKEND,
    FullModelMaterializationMode,
    FullModelSourceKind,
    build_full_model_snapshot_manifest,
    build_full_model_standalone_manager,
    full_model_materialization_preflight,
    full_model_policy_evaluation_identity,
    full_model_snapshot_identity_record,
    full_model_vllm_engine_kwargs,
    load_full_model_evaluation_snapshot,
    materialize_full_model_snapshot,
    write_full_model_materialization_receipt,
    write_full_model_snapshot_manifest,
)
from tgvf_rl.evaluation.policy_coredev import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
)
from tgvf_rl.evaluation.policy_official_visible import OfficialVisiblePolicyEvaluator
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)
from tgvf_rl.protocol import (
    NativeActionBoundaryProtocolId,
    NativeSuccessObservationProtocolId,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = (
    REPOSITORY_ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_80step_gpu0123.template.toml"
)


def _write_hf_model(path: Path, *, scale: float = 1.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "model.visual.patch_embed.weight": torch.full((2, 2), scale),
            "model.language_model.layers.0.weight": torch.full((2, 2), scale + 1),
        },
        path / "pytorch_model.bin",
    )


def _write_hf_safetensors_model(path: Path, *, scale: float = 1.0) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.visual.patch_embed.weight": torch.full((2, 2), scale),
            "model.language_model.layers.0.weight": torch.full((2, 2), scale + 1),
        },
        path / "model.safetensors",
    )


def _write_run_contract(tmp_path: Path, model_path: Path) -> Path:
    source = _TEMPLATE.read_text(encoding="utf-8")
    source = source.replace(
        "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
        str(model_path),
    )
    path = tmp_path / "run.toml"
    path.write_text(source, encoding="utf-8")
    return path


def _contract(tmp_path: Path, model_path: Path):
    return load_deepeyes_native_run_contract(
        _write_run_contract(tmp_path, model_path), allow_template=True
    )


def _write_fake_fsdp_checkpoint(
    path: Path,
    *,
    missing_rank: int | None = None,
    world_size: int = 4,
    complete_embedded_hf: bool = False,
) -> None:
    actor = path / "actor"
    huggingface = actor / "huggingface"
    huggingface.mkdir(parents=True)
    (huggingface / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
            }
        ),
        encoding="utf-8",
    )
    (actor / "fsdp_config.json").write_text(
        json.dumps({"FSDP_version": 2, "world_size": world_size}), encoding="utf-8"
    )
    if complete_embedded_hf:
        _write_hf_model(huggingface, scale=8.0)
    for rank in range(world_size):
        if rank != missing_rank:
            torch.save(
                {
                    "model.visual.patch_embed.weight": torch.full((1, 2), float(rank)),
                    "model.language_model.layers.0.weight": torch.full(
                        (1, 2), float(rank + 10)
                    ),
                },
                actor / f"model_world_size_{world_size}_rank_{rank}.pt",
            )
        torch.save(
            {"state": rank}, actor / f"optim_world_size_{world_size}_rank_{rank}.pt"
        )
        torch.save(
            {"rng": rank},
            actor / f"extra_state_world_size_{world_size}_rank_{rank}.pt",
        )
    torch.save({"cursor": 8}, path / "data.pt")


def test_step_zero_full_model_snapshot_round_trip_and_tamper(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)

    manifest = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )
    receipt = materialize_full_model_snapshot(manifest)

    assert manifest.source_kind is FullModelSourceKind.BASE_HF
    assert manifest.policy_version.optimizer_step == 0
    assert manifest.vision_parameter_key_count == 1
    assert manifest.language_parameter_key_count == 1
    assert receipt.mode is FullModelMaterializationMode.BASE_HF
    assert receipt.command == ()

    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path, receipt_path, require_launchable_run=False
    )
    identity = full_model_snapshot_identity_record(snapshot)
    assert identity["snapshot_backend"] == "full_model"
    assert identity["lora_request"] is None
    assert snapshot.model_path == base.resolve()

    torch.save({"changed": torch.ones(1)}, base / "pytorch_model.bin")
    with pytest.raises(ValueError, match="full-model"):
        load_full_model_evaluation_snapshot(
            manifest_path, receipt_path, require_launchable_run=False
        )


def test_coredev_full_model_record_freeze_rehashes_same_size_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    manifest = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )
    receipt = materialize_full_model_snapshot(manifest)
    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)

    runtime_modes: list[bool] = []

    def load_template_snapshot(
        bound_manifest_path: str | Path,
        bound_receipt_path: str | Path,
        *,
        runtime_lightweight: bool,
    ):
        runtime_modes.append(runtime_lightweight)
        return load_full_model_evaluation_snapshot(
            bound_manifest_path,
            bound_receipt_path,
            require_launchable_run=False,
            runtime_lightweight=runtime_lightweight,
        )

    monkeypatch.setattr(
        policy_coredev,
        "load_full_model_evaluation_snapshot",
        load_template_snapshot,
    )
    config = SimpleNamespace(
        snapshot_backend=FULL_MODEL_EVALUATION_BACKEND,
        policy_config_path=contract.source_path,
        output_root=tmp_path / "evaluation",
        full_model_snapshot_manifest_path=manifest_path,
        full_model_snapshot_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        full_model_materialization_receipt_path=receipt_path,
        full_model_materialization_receipt_sha256=hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        expected_policy_run_id=manifest.run_id,
        expected_policy_run_identity_sha256=manifest.run_identity_sha256,
        expected_optimizer_step=manifest.optimizer_step,
        expected_policy_weights_sha256=manifest.weights_sha256,
        required_snapshot_identity_sha256=manifest.identity_sha256,
    )

    source_snapshot = policy_coredev.load_policy_evaluation_snapshot(config)
    frozen_snapshot = policy_coredev.freeze_policy_evaluation_snapshot(
        config, source_snapshot
    )
    assert frozen_snapshot.policy_version == source_snapshot.policy_version
    assert runtime_modes == [False, False]

    weight_path = base / "pytorch_model.bin"
    original = weight_path.read_bytes()
    _write_hf_model(base, scale=9.0)
    mutated = weight_path.read_bytes()
    assert len(mutated) == len(original)
    assert mutated != original

    with pytest.raises(ValueError, match="checkpoint closure changed"):
        policy_coredev.load_frozen_policy_evaluation_snapshot(config)
    assert runtime_modes == [False, False, False]


def test_full_model_builder_rehashes_same_size_safetensors_before_vllm_use(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-model"
    _write_hf_safetensors_model(base)
    contract = _contract(tmp_path, base)
    manifest = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )
    receipt = materialize_full_model_snapshot(manifest)
    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path,
        receipt_path,
        require_launchable_run=False,
    )

    weight_path = base / "model.safetensors"
    original = weight_path.read_bytes()
    _write_hf_safetensors_model(base, scale=7.0)
    mutated = weight_path.read_bytes()
    assert len(mutated) == len(original)
    assert mutated != original

    # Integrity verification is the builder's first action, before importing
    # vLLM or constructing an engine over the receipt's external model path.
    with pytest.raises(ValueError, match="checkpoint closure changed"):
        asyncio.run(build_full_model_standalone_manager(SimpleNamespace(), snapshot))


def test_full_model_snapshot_rejects_adapter_artifacts(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    (base / "adapter_config.json").write_text("{}", encoding="utf-8")
    contract = _contract(tmp_path, base)

    with pytest.raises(ValueError, match="LoRA/adapter"):
        build_full_model_snapshot_manifest(contract, source_path=base, optimizer_step=0)


def test_step_zero_snapshot_rejects_wrong_hf_architecture(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2_vl",
            }
        ),
        encoding="utf-8",
    )
    contract = _contract(tmp_path, base)

    with pytest.raises(ValueError, match="PRL13 Qwen3-VL"):
        build_full_model_snapshot_manifest(contract, source_path=base, optimizer_step=0)


def test_snapshot_identity_ignores_huggingface_cache_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    metadata = base / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("first", encoding="utf-8")
    contract = _contract(tmp_path, base)

    first = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )
    metadata.write_text("changed cache bookkeeping", encoding="utf-8")
    second = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )

    assert first.identity_sha256 == second.identity_sha256
    assert all(
        not item.relative_path.startswith(".cache/") for item in first.source_files
    )


def test_fake_four_rank_fsdp_materializes_without_loading_8b(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    checkpoint = tmp_path / "global_step_8"
    _write_fake_fsdp_checkpoint(checkpoint)

    manifest = build_full_model_snapshot_manifest(
        contract, source_path=checkpoint, optimizer_step=8
    )
    target = tmp_path / "merged-model"
    preflight = full_model_materialization_preflight(manifest, target_dir=target)
    commands: list[tuple[str, ...]] = []

    def fake_merger(command: tuple[str, ...]) -> None:
        commands.append(command)
        assert command[command.index("--backend") + 1] == "fsdp"
        assert Path(command[command.index("--local_dir") + 1]) == checkpoint / "actor"
        assert Path(command[command.index("--target_dir") + 1]) == target
        _write_hf_model(target, scale=8.0)

    receipt = materialize_full_model_snapshot(
        manifest, target_dir=target, command_runner=fake_merger
    )

    assert manifest.source_kind is FullModelSourceKind.VERL_FSDP
    assert manifest.fsdp_world_size == 4
    assert preflight["gpu_required"] is False
    assert preflight["distributed_launch_required"] is False
    assert preflight["materialization_required"] is True
    assert preflight["estimated_peak_cpu_bytes"] == preflight["source_weight_bytes"] * 4
    assert len(commands) == 1
    assert receipt.mode is FullModelMaterializationMode.VERL_FSDP_MERGE

    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path, receipt_path, require_launchable_run=False
    )
    kwargs = full_model_vllm_engine_kwargs(
        snapshot,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        inference_concurrency_per_gpu=8,
        gpu_memory_utilization=0.8,
        enable_chunked_prefill=False,
    )
    assert kwargs["model"] == str(target.resolve())
    assert kwargs["tokenizer"] == str(base.resolve())
    assert kwargs["max_model_len"] == 33792
    assert kwargs["max_num_batched_tokens"] == 33792
    assert kwargs["mm_encoder_attn_backend"] == "TORCH_SDPA"
    assert kwargs["enable_lora"] is False
    assert "hf_overrides" not in kwargs
    assert "worker_extension_cls" not in kwargs


def test_complete_embedded_hf_binds_runtime_ws8_without_hashing_fsdp_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    checkpoint = tmp_path / "global_step_8"
    _write_fake_fsdp_checkpoint(checkpoint, world_size=8, complete_embedded_hf=True)

    with pytest.raises(ValueError, match="bind runtime_fsdp_world_size explicitly"):
        build_full_model_snapshot_manifest(
            contract, source_path=checkpoint, optimizer_step=8
        )

    original_sha256_file = full_model_snapshot._sha256_file
    hashed: list[Path] = []

    def recording_sha256(path: Path) -> str:
        hashed.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(full_model_snapshot, "_sha256_file", recording_sha256)
    manifest = build_full_model_snapshot_manifest(
        contract,
        source_path=checkpoint,
        optimizer_step=8,
        runtime_fsdp_world_size=8,
    )
    receipt = materialize_full_model_snapshot(manifest)

    assert manifest.fsdp_world_size == 8
    assert manifest.as_record()["fsdp_world_size"] == 8
    assert receipt.mode is FullModelMaterializationMode.EMBEDDED_HF
    assert not any(
        item.relative_path.startswith("actor/model_world_size_")
        or item.relative_path.startswith("actor/optim_world_size_")
        or item.relative_path.startswith("actor/extra_state_world_size_")
        for item in manifest.source_files
    )
    assert not any(
        path.name.startswith(
            ("model_world_size_", "optim_world_size_", "extra_state_world_size_")
        )
        for path in hashed
    )


def test_runtime_lightweight_uses_receipts_headers_and_rechecks_rank_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    checkpoint = tmp_path / "global_step_8"
    _write_fake_fsdp_checkpoint(checkpoint, world_size=8, complete_embedded_hf=True)
    manifest = build_full_model_snapshot_manifest(
        contract,
        source_path=checkpoint,
        optimizer_step=8,
        runtime_fsdp_world_size=8,
    )
    receipt = materialize_full_model_snapshot(manifest)
    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)

    original_sha256_file = full_model_snapshot._sha256_file

    def forbid_payload_hash(path: Path) -> str:
        if path.suffix in {".bin", ".safetensors"} or path.name.startswith(
            ("model_world_size_", "optim_world_size_", "extra_state_world_size_")
        ):
            raise AssertionError(f"runtime hashed model payload: {path}")
        return original_sha256_file(path)

    monkeypatch.setattr(full_model_snapshot, "_sha256_file", forbid_payload_hash)
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path,
        receipt_path,
        require_launchable_run=False,
        runtime_lightweight=True,
    )
    assert snapshot.model_path == (checkpoint / "actor/huggingface").resolve()

    (checkpoint / "actor/model_world_size_8_rank_7.pt").unlink()
    with pytest.raises(ValueError, match="rank shard set is incomplete"):
        load_full_model_evaluation_snapshot(
            manifest_path,
            receipt_path,
            require_launchable_run=False,
            runtime_lightweight=True,
        )


def test_fsdp_snapshot_rejects_missing_rank_and_lora_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    missing = tmp_path / "global_step_8"
    _write_fake_fsdp_checkpoint(missing, missing_rank=3)

    with pytest.raises(ValueError, match="rank shard set is incomplete"):
        build_full_model_snapshot_manifest(
            contract, source_path=missing, optimizer_step=8
        )

    complete = tmp_path / "global_step_20"
    _write_fake_fsdp_checkpoint(complete)
    (complete / "actor/lora_train_meta.json").write_text(
        json.dumps({"r": 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="LoRA/adapter"):
        build_full_model_snapshot_manifest(
            contract, source_path=complete, optimizer_step=20
        )


def test_official_visible_constructor_accepts_only_adapter_free_full_snapshot(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    manifest = build_full_model_snapshot_manifest(
        contract, source_path=base, optimizer_step=0
    )
    receipt = materialize_full_model_snapshot(manifest)
    manifest_path = tmp_path / "snapshot.json"
    receipt_path = tmp_path / "receipt.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path, receipt_path, require_launchable_run=False
    )

    output_root = tmp_path / "evaluation"
    task_path = output_root / "runtime/policy-benchmark-tasks.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
    config = SimpleNamespace(
        evaluation_protocol=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        evaluation_id="PRL13-FULL-STEP0",
        schema_version="fixture-evaluation-v1",
        output_root=output_root,
        uses_legacy_coredev_manifest=False,
        task_manifest_sha256=None,
        expected_task_count=1,
        expected_single_image_count=1,
        gpu_ids=(0,),
        max_model_len=32768,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=False,
        inference_concurrency_per_gpu=1,
        declared_image_max_pixels=snapshot.run.policy.image_max_pixels,
        success_observation_protocol_id=(
            NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
        ),
        action_boundary_protocol_id=(
            NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
        ),
    )
    strict_config = SimpleNamespace(**vars(config))
    strict_config.action_boundary_protocol_id = (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    )
    strict_identity = full_model_policy_evaluation_identity(strict_config, snapshot)

    identity = full_model_policy_evaluation_identity(config, snapshot)
    assert identity["eval_contract"]["pixels"] == {
        "training_image_max_pixels": snapshot.run.policy.image_max_pixels,
        "declared_image_max_pixels": snapshot.run.policy.image_max_pixels,
        "effective_image_max_pixels": snapshot.run.policy.image_max_pixels,
    }
    assert identity["eval_contract"]["success_observation"]["protocol_id"] == (
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1.value
    )
    assert identity["eval_contract"]["parser"]["protocol_id"] == (
        "deepeyes-hermes-last-complete-crop-call-v1"
    )
    assert identity["eval_contract"]["action_boundary"]["protocol_id"] == (
        NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1.value
    )
    assert strict_identity["eval_contract"]["action_boundary"]["protocol_id"] == (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value
    )
    assert (
        strict_identity["eval_contract"]["action_boundary"]["identity_sha256"]
        != identity["eval_contract"]["action_boundary"]["identity_sha256"]
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_a, **_k: ""))
    manager = SimpleNamespace(
        native_pixels=True,
        capture_hidden=False,
        lora_request=None,
    )

    evaluator = OfficialVisiblePolicyEvaluator(
        config=config,
        run=snapshot.run,
        manager=manager,
        processor=processor,
        snapshot=snapshot,
        evaluation_identity=identity,
    )
    assert evaluator.policy_version == snapshot.policy_version
    assert evaluator.full_model is True
    assert identity["policy_snapshot"]["snapshot_backend"] == "full_model"

    strict_evaluator = OfficialVisiblePolicyEvaluator(
        config=strict_config,
        run=snapshot.run,
        manager=manager,
        processor=processor,
        snapshot=snapshot,
        evaluation_identity=strict_identity,
    )
    assert strict_evaluator.action_boundary_protocol_id is (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    )

    manager.lora_request = object()
    with pytest.raises(ValueError, match="forbids LoRARequest"):
        OfficialVisiblePolicyEvaluator(
            config=config,
            run=snapshot.run,
            manager=manager,
            processor=processor,
            snapshot=snapshot,
            evaluation_identity=identity,
        )
