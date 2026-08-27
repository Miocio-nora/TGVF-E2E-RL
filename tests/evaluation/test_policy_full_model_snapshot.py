from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tgvf_rl.evaluation.policy_full_model_snapshot as full_model_snapshot
from tgvf_rl.contracts.errors import PolicyOutputContractError
from tgvf_rl.trajectories.schema import TrajectoryIdentity
from tgvf_rl.evaluation.policy_full_model_snapshot import (
    FULL_MODEL_SNAPSHOT_SCHEMA_V2,
    FullModelCheckpointOwner,
    FullModelMaterializationMode,
    FullModelSourceKind,
    build_full_model_snapshot_manifest,
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
    CoreDevTask,
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    load_policy_benchmark_results,
    policy_output_contract_failure_audit_payload,
    validate_policy_benchmark_result,
)
from tgvf_rl.evaluation.policy_official_visible import (
    OfficialVisiblePolicyEvaluator,
    OfficialVisibleTrajectory,
    official_visible_trajectory_audit_payload,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = (
    REPOSITORY_ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_80step_gpu0123.template.toml"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash_evaluation_identity(value: dict[str, object]) -> None:
    value.pop("identity_sha256", None)
    value["identity_sha256"] = _canonical_sha256(value)


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
    assert "protocol_run_id" not in identity
    assert snapshot.model_path == base.resolve()
    assert identity["evaluation_sampling"]["stop_strings"] == ["</tool_call>"]
    assert identity["evaluation_sampling"]["include_stop_str_in_output"] is True
    assert dict(
        snapshot.run.policy.sampling.as_vllm_parameters(max_tokens=1024)
    ) == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_token_ids": [151645],
        "stop": ["</tool_call>"],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
        "max_tokens": 1024,
        "logprobs": True,
    }

    torch.save({"changed": torch.ones(1)}, base / "pytorch_model.bin")
    with pytest.raises(ValueError, match="full-model"):
        load_full_model_evaluation_snapshot(
            manifest_path, receipt_path, require_launchable_run=False
        )


def test_v1_full_model_result_writer_does_not_require_v2_protocol_fields(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    config_path = _write_run_contract(tmp_path, base)
    contract = load_deepeyes_native_run_contract(config_path, allow_template=True)
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
        evaluation_id="V1-FULL-STEP0",
        schema_version="fixture-evaluation-v1",
        policy_config_path=config_path.resolve(),
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
    )
    evaluation_identity = full_model_policy_evaluation_identity(config, snapshot)
    assert "protocol_contract" not in evaluation_identity
    assert "protocol_run_id" not in evaluation_identity["policy_snapshot"]
    assert "sampling_rng" not in evaluation_identity

    task = CoreDevTask(
        ordinal=0,
        dataset="fixture",
        row_number=0,
        index="row-0",
        sample_id="sample-0",
        question="Which option is correct?",
        image_paths=("/immutable/image.png",),
    )
    trajectory = OfficialVisibleTrajectory(
        identity=TrajectoryIdentity(
            config.evaluation_id, task.bound_sample_id, 0, "benchmark:0"
        ),
        model=snapshot.run.model,
        behavior_policy=snapshot.policy_version,
        stop="final_answer",
        final_answer="A",
        assistant_turns=(),
        tool_calls=(),
        tool_errors=(),
        native_image_sha256s=("6" * 64,),
    )
    result = official_visible_trajectory_audit_payload(
        task,
        trajectory,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )
    assert result["policy_snapshot_backend"] == "full_model"
    assert "protocol_run_id" not in result
    assert "sampling_rng" not in result
    validate_policy_benchmark_result(
        result,
        task=task,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )


def test_v2_snapshot_binds_distinct_checkpoint_owner_and_protocol(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-model"
    _write_hf_model(base)
    contract = _contract(tmp_path, base)
    owner_config = tmp_path / "crop-owner.toml"
    owner_config.write_text("run_id = 'PRL21-OWNER'\n", encoding="utf-8")
    owner_completion = tmp_path / "completion.json"
    owner_completion.write_text('{"status":"complete"}\n', encoding="utf-8")
    owner = FullModelCheckpointOwner(
        run_id="PRL21-OWNER",
        run_identity_sha256="1" * 64,
        config_path=str(owner_config.resolve()),
        config_file_sha256=hashlib.sha256(owner_config.read_bytes()).hexdigest(),
        completion_path=str(owner_completion.resolve()),
        completion_file_sha256=hashlib.sha256(
            owner_completion.read_bytes()
        ).hexdigest(),
    )
    manifest = build_full_model_snapshot_manifest(
        contract,
        source_path=base,
        optimizer_step=0,
        checkpoint_owner=owner,
    )
    receipt = materialize_full_model_snapshot(manifest)
    manifest_path = tmp_path / "snapshot-v2.json"
    receipt_path = tmp_path / "receipt-v2.json"
    write_full_model_snapshot_manifest(manifest_path, manifest)
    write_full_model_materialization_receipt(receipt_path, receipt)

    snapshot = load_full_model_evaluation_snapshot(
        manifest_path, receipt_path, require_launchable_run=False
    )
    identity_record = full_model_snapshot_identity_record(snapshot)
    assert manifest.schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2
    assert snapshot.policy_version.run_id == "PRL21-OWNER"
    assert snapshot.run_identity_sha256 == "1" * 64
    assert snapshot.run.run_id == "PRL21-OWNER"
    assert snapshot.run.protocol_run_id == contract.run_id
    assert identity_record["run_id"] == "PRL21-OWNER"
    assert identity_record["protocol_run_id"] == contract.run_id
    assert identity_record["checkpoint_owner_config_path"] == str(
        owner_config.resolve()
    )

    output_root = tmp_path / "evaluation-v2"
    task_path = output_root / "runtime/policy-benchmark-tasks.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
    config = SimpleNamespace(
        evaluation_protocol=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        evaluation_id="PRL21-FULL-STEP0",
        schema_version="fixture-evaluation-v1",
        policy_config_path=owner_config.resolve(),
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
        paired_seed_namespace="fixture/full-model/step0/temp1/seed42/v1",
    )
    evaluation_identity = full_model_policy_evaluation_identity(config, snapshot)
    assert evaluation_identity["policy_snapshot"]["run_id"] == "PRL21-OWNER"
    assert evaluation_identity["protocol_contract"]["run_id"] == contract.run_id
    assert evaluation_identity["sampling_rng"]["seed_namespace"] == (
        "fixture/full-model/step0/temp1/seed42/v1"
    )
    rng_protocol = dict(evaluation_identity["protocol"])
    assert "base_equivalence" in rng_protocol
    rng_protocol.pop("base_equivalence")
    assert evaluation_identity["sampling_rng"]["protocol_sha256"] == (
        _canonical_sha256(rng_protocol)
    )

    task = CoreDevTask(
        ordinal=0,
        dataset="fixture",
        row_number=0,
        index="row-0",
        sample_id="sample-0",
        question="Which option is correct?",
        image_paths=("/immutable/image.png",),
    )
    trajectory = OfficialVisibleTrajectory(
        identity=TrajectoryIdentity(
            config.evaluation_id, task.bound_sample_id, 0, "benchmark:0"
        ),
        model=snapshot.run.model,
        behavior_policy=snapshot.policy_version,
        stop="final_answer",
        final_answer="A",
        assistant_turns=(),
        tool_calls=(),
        tool_errors=(),
        native_image_sha256s=("6" * 64,),
    )
    official_result = official_visible_trajectory_audit_payload(
        task,
        trajectory,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )
    assert official_result["policy_run_id"] == "PRL21-OWNER"
    assert official_result["protocol_run_id"] == contract.run_id
    assert official_result["sampling_rng"] == evaluation_identity["sampling_rng"]
    assert len(official_result["paired_rng_stream_identity_sha256"]) == 64
    validate_policy_benchmark_result(
        official_result,
        task=task,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )

    missing_protocol = json.loads(json.dumps(evaluation_identity))
    del missing_protocol["policy_snapshot"]["protocol_run_id"]
    del missing_protocol["policy_snapshot"]["protocol_run_identity_sha256"]
    _rehash_evaluation_identity(missing_protocol)
    with pytest.raises(ValueError, match="v2 protocol snapshot binding differs"):
        official_visible_trajectory_audit_payload(
            task,
            trajectory,
            evaluation_identity=missing_protocol,
            rank=0,
            world_size=1,
        )

    wrong_protocol = json.loads(json.dumps(evaluation_identity))
    wrong_protocol["policy_snapshot"]["protocol_run_id"] = "WRONG-PROTOCOL"
    wrong_protocol["policy_snapshot"]["protocol_run_identity_sha256"] = "9" * 64
    _rehash_evaluation_identity(wrong_protocol)
    with pytest.raises(ValueError, match="v2 protocol snapshot binding differs"):
        official_visible_trajectory_audit_payload(
            task,
            trajectory,
            evaluation_identity=wrong_protocol,
            rank=0,
            world_size=1,
        )
    wrong_result = dict(official_result)
    wrong_result["evaluation_identity_sha256"] = wrong_protocol["identity_sha256"]
    wrong_result["protocol_run_id"] = "WRONG-PROTOCOL"
    wrong_result["protocol_run_identity_sha256"] = "9" * 64
    wrong_result.pop("result_identity_sha256")
    wrong_result["result_identity_sha256"] = _canonical_sha256(wrong_result)
    with pytest.raises(ValueError, match="v2 protocol snapshot binding differs"):
        validate_policy_benchmark_result(
            wrong_result,
            task=task,
            evaluation_identity=wrong_protocol,
            rank=0,
            world_size=1,
        )

    error = PolicyOutputContractError(
        "vLLM emitted a tool-call suffix outside the run-bound contract",
        code="tool_call_terminal_suffix",
        diagnostic={
            "response_text_sha256": "2" * 64,
            "suffix_sha256": "3" * 64,
            "suffix_char_count": 1,
            "suffix_utf8_byte_count": 1,
            "finish_reason": "stop",
            "stop_reason": "</tool_call>",
            "backend_request_sha256": "4" * 64,
            "backend_response_sha256": "5" * 64,
        },
    )
    result = policy_output_contract_failure_audit_payload(
        task,
        error,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )
    assert result["policy_run_id"] == "PRL21-OWNER"
    assert result["protocol_run_id"] == contract.run_id
    validate_policy_benchmark_result(
        result,
        task=task,
        evaluation_identity=evaluation_identity,
        rank=0,
        world_size=1,
    )
    inference = output_root / "inference"
    inference.mkdir()
    (inference / "rank-0.jsonl").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_policy_benchmark_results(
        inference,
        tasks=(task,),
        evaluation_identity=evaluation_identity,
        require_complete=True,
    )
    assert loaded[0]["policy_run_id"] == "PRL21-OWNER"
    assert loaded[0]["protocol_run_id"] == contract.run_id

    owner_config.write_text("run_id = 'TAMPERED'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="owner config bytes changed"):
        load_full_model_evaluation_snapshot(
            manifest_path, receipt_path, require_launchable_run=False
        )


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
    )
    identity = full_model_policy_evaluation_identity(config, snapshot)
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
    assert "checkpoint_owner" not in identity
    assert "protocol_contract" not in identity

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
