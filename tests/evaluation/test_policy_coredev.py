from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import threading
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
import tgvf_rl.evaluation.policy_coredev as policy_coredev
import tgvf_rl.evaluation.policy_evaluation_config as policy_evaluation_config
from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    LORA_ADAPTER_EVALUATION_BACKEND,
    POLICY_BENCHMARK_SCHEMA,
    POLICY_COREDEV_LEGACY_SCHEMA_V1,
    POLICY_COREDEV_SCHEMA,
    PolicyCoreDevConfig,
    PolicyEvaluationSnapshot,
    StandaloneTGVFVLLMManager,
    VLLM_LORA_ADAPTER_CONFIG_FILENAME,
    VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
    VLLM_LORA_ADAPTER_MODEL_FILENAME,
    VLLM_LORA_ENGINE_ATTESTATION,
    VLLM_LORA_RESIDUAL_RACE,
    _termination_contract,
    build_policy_eval_contract,
    build_vllm_lora_adapter_integrity_verifier,
    freeze_policy_evaluation_snapshot,
    load_coredev_tasks,
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
    materialize_vllm_lora_adapter,
)
from tgvf_rl.framework.vllm import VLLMTerminationOutcome
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
    publish_policy_weight_sync_request,
    wrap_lora_parameter_stream_for_snapshot,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.protocol import (
    NativeActionBoundaryProtocolId,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_policy_coredev_reexports_policy_evaluation_config_contract() -> None:
    assert (
        policy_coredev.PolicyCoreDevConfig
        is policy_evaluation_config.PolicyCoreDevConfig
    )
    assert (
        policy_coredev.load_policy_coredev_config
        is policy_evaluation_config.load_policy_coredev_config
    )
    assert (
        policy_coredev.POLICY_COREDEV_SCHEMA
        == policy_evaluation_config.POLICY_COREDEV_SCHEMA
    )
    assert (
        policy_coredev.POLICY_BENCHMARK_SCHEMA
        == policy_evaluation_config.POLICY_BENCHMARK_SCHEMA
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lora_evaluation_fixture(
    tmp_path: Path,
) -> tuple[SimpleNamespace, PolicyEvaluationSnapshot]:
    tensor_bytes = b"exact fixture safetensors bytes"
    pointer_bytes = b'{"fixture":"pointer"}\n'
    manifest_bytes = b'{"fixture":"manifest"}\n'
    run_id = "fixture-policy-run"
    run_identity_sha256 = "b" * 64
    snapshot = PolicyEvaluationSnapshot(
        run=SimpleNamespace(
            run_id=run_id,
            identity_sha256=run_identity_sha256,
            model=SimpleNamespace(revision_or_path="/models/Qwen3-VL-8B-Instruct"),
        ),
        lora=PolicyLoRASnapshot(
            policy_version=PolicyVersion(
                run_id,
                32,
                _sha256(tensor_bytes),
            ),
            run_identity_sha256=run_identity_sha256,
            request_sha256="c" * 64,
            pointer_file=(tmp_path / "source/latest.json").resolve(),
            pointer_file_sha256=_sha256(pointer_bytes),
            pointer_bytes=pointer_bytes,
            tensor_file=(tmp_path / "source/adapter.safetensors").resolve(),
            tensor_file_sha256=_sha256(tensor_bytes),
            tensor_bytes=tensor_bytes,
            manifest_file=(tmp_path / "source/manifest.json").resolve(),
            manifest_file_sha256=_sha256(manifest_bytes),
            manifest_bytes=manifest_bytes,
            tensors={},
        ),
    )
    config = SimpleNamespace(
        output_root=(tmp_path / "evaluation").resolve(),
        evaluation_id="fixture-evaluation",
    )
    return config, snapshot


def test_policy_evaluation_snapshot_preserves_legacy_pickle_path(
    tmp_path: Path,
) -> None:
    _config, snapshot = _lora_evaluation_fixture(tmp_path)

    restored = pickle.loads(pickle.dumps(snapshot))

    assert type(restored) is PolicyEvaluationSnapshot
    assert restored == snapshot


def _contract_run(
    *,
    tool_profile: NativeToolCapabilityProfile,
    model_name: str,
    prompt_sha256: str = "a" * 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(model_name=model_name),
        protocol=SimpleNamespace(
            tool_profile=tool_profile,
            enabled_tool_names=tool_profile.tool_names,
            tool_schema_sha256=tool_profile.tool_set_sha256,
            prompt_sha256=prompt_sha256,
        ),
        policy=SimpleNamespace(
            image_max_pixels=262144,
            sampling=SimpleNamespace(
                stop_strings=("</tool_call>",),
                stop_token_ids=(151645,),
                include_stop_str_in_output=True,
            ),
        ),
    )


def _explicit_coredev_v2_payload(
    *,
    success_observation_protocol_id: NativeSuccessObservationProtocolId,
    policy_config_path: str = "/fixtures/policy.toml",
) -> dict[str, object]:
    return {
        "schema_version": POLICY_COREDEV_SCHEMA,
        "evaluation_id": "fixture-coredev-v2",
        "policy_config_path": policy_config_path,
        "lora_pointer_path": "/fixtures/latest-lora-snapshot.json",
        "lora_pointer_sha256": "a" * 64,
        "expected_policy_run_id": "fixture-policy-run",
        "expected_policy_run_identity_sha256": "b" * 64,
        "expected_optimizer_step": 80,
        "expected_policy_weights_sha256": "c" * 64,
        "output_root": "/fixtures/evaluation",
        "gpu_ids": [0, 1, 2, 3],
        "declared_image_max_pixels": 262144,
        "success_observation_protocol_id": success_observation_protocol_id.value,
        "action_boundary_protocol_id": (
            NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value
        ),
        "evaluation_protocol": "training_run",
        "inference_concurrency_per_gpu": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
    }


def test_historical_v1_policy_configs_are_immutable_evidence_only() -> None:
    expected = {
        "coredev_2511_tgvf_step80_v1.json": (
            "dc7cf50c905c5f06935f0b28949790279b8f3d0b82b8afde63f3128e699eb273"
        ),
        "coredev_2511_crop_step80_v1.json": (
            "b961103f6e333083b2268df0a671aaacea756e0d30319d32e35188e642753a46"
        ),
    }
    for name, expected_sha256 in expected.items():
        path = REPOSITORY_ROOT / "configs/evaluation" / name
        assert _sha256(path.read_bytes()) == expected_sha256
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == (
            POLICY_COREDEV_LEGACY_SCHEMA_V1
        )
        with pytest.raises(ValueError, match="immutable evidence only"):
            load_policy_coredev_config(path)


def test_canonical_evaluation_schema_tracks_runtime_v2_constants() -> None:
    schema = json.loads(
        (
            REPOSITORY_ROOT
            / "configs/canonical/evaluation/policy_evaluation_v2.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert schema["properties"]["schema_version"]["enum"] == [
        POLICY_COREDEV_SCHEMA,
        POLICY_BENCHMARK_SCHEMA,
    ]
    required = set(schema["required"])
    assert {
        "declared_image_max_pixels",
        "success_observation_protocol_id",
        "action_boundary_protocol_id",
        "expected_policy_run_identity_sha256",
        "expected_policy_weights_sha256",
    } <= required


def test_policy_evaluation_accepts_native_vllm_eos_identity() -> None:
    run = load_policy_e2e_smoke_run_config(
        REPOSITORY_ROOT
        / "configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml",
        allow_external_agent_loop_config=True,
        allow_historical_reward_contract=True,
    )

    assert (
        VLLMTerminationOutcome("stop", None)
        in _termination_contract(run).final_turn_outcomes
    )


def test_training_eval_contracts_bind_tgvf_and_historical_crop_explicitly() -> None:
    cases = {
        NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1: (
            NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            "tgvf-native-run-prompt-v1",
        ),
        NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1: (
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1,
            "tgvf-native-run-prompt-v1",
        ),
    }
    for fixture_protocol_id, (
        observation_protocol_id,
        prompt_protocol_id,
    ) in cases.items():
        config = PolicyCoreDevConfig(
            **_explicit_coredev_v2_payload(
                success_observation_protocol_id=fixture_protocol_id
            )
        )
        run = _contract_run(
            tool_profile=(
                NativeToolCapabilityProfile.TGVF_ONLY
                if observation_protocol_id
                is NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1
                else NativeToolCapabilityProfile.CROP_ONLY
            ),
            model_name="Qwen3-VL-8B-Thinking",
        )
        contract = build_policy_eval_contract(config, SimpleNamespace(run=run))

        assert contract.success_observation_protocol_id is observation_protocol_id
        assert contract.prompt_protocol_id == prompt_protocol_id
        assert contract.parser_protocol_id == "strict-native-single-tool-call-v1"
        assert (
            contract.action_boundary_protocol_id
            is NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        )
        assert contract.training_image_max_pixels == 262144
        assert contract.declared_image_max_pixels == 262144
        assert contract.effective_image_max_pixels == 262144
        assert len(contract.identity_sha256) == 64


def test_eval_config_without_observation_protocol_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _explicit_coredev_v2_payload(
        success_observation_protocol_id=(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1
        )
    )
    del payload["success_observation_protocol_id"]
    path = tmp_path / "old-crop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config fields differ"):
        load_policy_coredev_config(path)


def test_crop_eval_never_guesses_generic_native_renderer(tmp_path: Path) -> None:
    payload = _explicit_coredev_v2_payload(
        success_observation_protocol_id=(
            NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1
        )
    )
    path = tmp_path / "wrong-crop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_policy_coredev_config(path)
    run = _contract_run(
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        model_name="Qwen3-VL-8B-Thinking",
    )

    with pytest.raises(ValueError, match="explicit matched or legacy Crop"):
        build_policy_eval_contract(config, SimpleNamespace(run=run))


def test_historical_instruct_crop_exact_runtime_accepts_only_explicit_generic86(
    tmp_path: Path,
) -> None:
    payload = _explicit_coredev_v2_payload(
        success_observation_protocol_id=(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
        ),
        policy_config_path=str(
            REPOSITORY_ROOT / "configs/policy/runs/"
            "prl_04_r1_qwen3_instruct_grpo_bs16_crop_t1full_80step_gpu0123.toml"
        ),
    )
    path = tmp_path / "historical-instruct-crop-exact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_policy_coredev_config(path)
    run = _contract_run(
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        model_name="Qwen3-VL-8B-Instruct",
    )

    contract = build_policy_eval_contract(config, SimpleNamespace(run=run))

    assert (
        contract.success_observation_protocol_id
        is NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
    )


def test_historical_reward_snapshot_freeze_and_reload_use_same_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "historical-reward-fixture"
    run_identity_sha256 = "d" * 64
    source_root = (tmp_path / "source-policy-state").resolve()
    environment = {
        "TGVF_POLICY_STATE_DIR": str(source_root),
        "TGVF_POLICY_RUN_ID": run_id,
        "TGVF_POLICY_RUN_IDENTITY_SHA256": run_identity_sha256,
        "RANK": "0",
        "WORLD_SIZE": "4",
    }
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, 32, nonce="historical-reward")
    list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(
                [
                    (
                        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                        torch.ones((1, 2), dtype=torch.bfloat16),
                    )
                ]
            ),
            base_sync_done=True,
            rank=0,
            world_size=4,
            global_steps=32,
            environment=environment,
        )
    )
    pointer_sha256 = _sha256(state.latest_path.read_bytes())
    historical_run = SimpleNamespace(
        run_id=run_id,
        identity_sha256=run_identity_sha256,
    )
    calls: list[tuple[bool, bool]] = []

    def load_historical_run(
        path: Path,
        *,
        allow_external_agent_loop_config: bool = False,
        allow_historical_reward_contract: bool = False,
    ) -> SimpleNamespace:
        del path
        calls.append(
            (
                allow_external_agent_loop_config,
                allow_historical_reward_contract,
            )
        )
        if not allow_historical_reward_contract:
            raise ValueError("historical reward contract was not explicitly allowed")
        return historical_run

    monkeypatch.setattr(
        policy_coredev,
        "load_policy_e2e_smoke_run_config",
        load_historical_run,
    )
    config = SimpleNamespace(
        snapshot_backend=LORA_ADAPTER_EVALUATION_BACKEND,
        policy_config_path=tmp_path / "historical-policy.toml",
        lora_pointer_path=state.latest_path,
        lora_pointer_sha256=pointer_sha256,
        expected_policy_run_id=run_id,
        expected_policy_run_identity_sha256=run_identity_sha256,
        expected_optimizer_step=32,
        expected_policy_weights_sha256=None,
        output_root=(tmp_path / "evaluation").resolve(),
    )

    loaded = load_policy_evaluation_snapshot(config)
    frozen = freeze_policy_evaluation_snapshot(config, loaded)

    assert frozen.policy_version == loaded.policy_version
    assert frozen.lora.tensor_bytes == loaded.lora.tensor_bytes
    assert calls == [(True, True), (True, True)]


def test_coredev_task_loader_keeps_order_and_single_image_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "ordinal": index,
            "dataset": "fixture",
            "row_number": index,
            "index": str(index),
            "question": "question",
            "image_paths": ["a.jpg"] if index != 3 else ["a.jpg", "b.jpg"],
        }
        for index in range(2511)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tasks = load_coredev_tasks(path)
    assert isinstance(tasks[0], CoreDevTask)
    assert tasks[0].single_image is True
    assert tasks[3].single_image is False
    assert tasks[-1].ordinal == 2510


def test_vllm_lora_materialization_is_content_addressed_private_and_exact(
    tmp_path: Path,
) -> None:
    config, snapshot = _lora_evaluation_fixture(tmp_path)

    adapter_root = materialize_vllm_lora_adapter(config, snapshot)
    verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    )
    verifier.verify(phase="normal regression test")

    assert adapter_root.name == verifier.materialization_identity_sha256
    assert adapter_root.parent.name == "lora-adapters"
    assert adapter_root.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in adapter_root.iterdir()} == {
        VLLM_LORA_ADAPTER_MODEL_FILENAME,
        VLLM_LORA_ADAPTER_CONFIG_FILENAME,
        VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
    }
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in adapter_root.iterdir())
    assert (adapter_root / VLLM_LORA_ADAPTER_MODEL_FILENAME).read_bytes() == (
        snapshot.lora.tensor_bytes
    )
    identity = json.loads(
        (adapter_root / VLLM_LORA_ADAPTER_IDENTITY_FILENAME).read_text(encoding="utf-8")
    )
    assert identity["engine_loaded_identity_attestation"] == (
        VLLM_LORA_ENGINE_ATTESTATION
    )
    assert identity["residual_race"] == VLLM_LORA_RESIDUAL_RACE


def test_vllm_lora_content_identity_is_portable_across_source_paths(
    tmp_path: Path,
) -> None:
    config, snapshot = _lora_evaluation_fixture(tmp_path)
    relocated = PolicyEvaluationSnapshot(
        run=snapshot.run,
        lora=replace(
            snapshot.lora,
            pointer_file=(tmp_path / "relocated/latest.json").resolve(),
            manifest_file=(tmp_path / "relocated/manifest.json").resolve(),
            tensor_file=(tmp_path / "relocated/adapter.safetensors").resolve(),
        ),
    )

    original_root = materialize_vllm_lora_adapter(config, snapshot)
    relocated_root = materialize_vllm_lora_adapter(config, relocated)

    assert relocated_root == original_root
    original_verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        original_root,
    )
    relocated_verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        relocated,
        relocated_root,
    )
    assert (
        relocated_verifier.materialization_identity_sha256
        == original_verifier.materialization_identity_sha256
    )
    assert relocated_verifier.identity_bytes == original_verifier.identity_bytes


@pytest.mark.parametrize(
    ("competing_payload", "raises_mismatch"),
    [(b"different concurrent payload", True), (b"expected payload", False)],
)
def test_vllm_lora_publish_never_replaces_a_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    competing_payload: bytes,
    raises_mismatch: bool,
) -> None:
    filename = "adapter_model.safetensors"
    destination = tmp_path / filename
    expected_payload = b"expected payload"
    link_reached = threading.Event()
    competitor_done = threading.Event()
    original_link = os.link

    def pause_before_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        link_reached.set()
        assert competitor_done.wait(timeout=5)
        original_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def publish_competitor() -> None:
        assert link_reached.wait(timeout=5)
        destination.write_bytes(competing_payload)
        destination.chmod(0o600)
        competitor_done.set()

    monkeypatch.setattr(policy_coredev.os, "link", pause_before_link)
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            competitor = executor.submit(publish_competitor)
            if raises_mismatch:
                with pytest.raises(
                    ReplayMismatchError,
                    match="content-addressed vLLM LoRA output.*differs",
                ):
                    policy_coredev._write_private_vllm_lora_file_at(
                        root_descriptor,
                        filename,
                        expected_payload,
                    )
            else:
                policy_coredev._write_private_vllm_lora_file_at(
                    root_descriptor,
                    filename,
                    expected_payload,
                )
            competitor.result(timeout=5)
    finally:
        os.close(root_descriptor)

    assert destination.read_bytes() == competing_payload
    assert destination.stat().st_mode & 0o777 == 0o600
    assert {path.name for path in tmp_path.iterdir()} == {filename}


def test_vllm_lora_existing_hardlink_is_rejected_then_recoverable(
    tmp_path: Path,
) -> None:
    filename = "adapter_model.safetensors"
    destination = tmp_path / filename
    payload = b"exact adapter payload"
    destination.write_bytes(payload)
    destination.chmod(0o600)
    hardlink = tmp_path / "adapter-model-hardlink.safetensors"
    os.link(destination, hardlink)
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(ReplayMismatchError, match="unexpected hardlink"):
            policy_coredev._write_private_vllm_lora_file_at(
                root_descriptor,
                filename,
                payload,
            )

        hardlink.unlink()
        policy_coredev._write_private_vllm_lora_file_at(
            root_descriptor,
            filename,
            payload,
        )
    finally:
        os.close(root_descriptor)

    assert destination.read_bytes() == payload
    assert destination.stat().st_nlink == 1


def test_private_directory_open_closes_descriptor_when_fchmod_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    original_open = os.open
    original_fchmod = os.fchmod
    opened_descriptors: list[int] = []

    def record_open(
        path: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "adapter" and dir_fd == parent_descriptor:
            opened_descriptors.append(descriptor)
        return descriptor

    def fail_child_fchmod(descriptor: int, mode: int) -> None:
        if descriptor in opened_descriptors:
            raise OSError("forced fchmod failure")
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(policy_coredev.os, "open", record_open)
    monkeypatch.setattr(policy_coredev.os, "fchmod", fail_child_fchmod)
    try:
        with pytest.raises(ReplayMismatchError, match="unreadable"):
            policy_coredev._open_or_create_private_directory_at(
                parent_descriptor,
                "adapter",
                owner="test adapter directory",
            )
    finally:
        os.close(parent_descriptor)

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


@pytest.mark.parametrize("symlinked_component", ["runtime", "lora-adapters"])
def test_vllm_lora_verifier_rejects_symlinked_adapter_ancestor(
    tmp_path: Path,
    symlinked_component: str,
) -> None:
    config, snapshot = _lora_evaluation_fixture(tmp_path)
    adapter_root = materialize_vllm_lora_adapter(config, snapshot)
    verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    )
    ancestor = (
        config.output_root / "runtime"
        if symlinked_component == "runtime"
        else config.output_root / "runtime" / "lora-adapters"
    )
    moved = tmp_path / f"moved-{symlinked_component}"
    ancestor.rename(moved)
    ancestor.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ReplayMismatchError, match="contains a symlink"):
        verifier.verify(phase="ancestor symlink regression test")


@pytest.mark.parametrize("symlinked_component", ["runtime", "lora-adapters"])
def test_vllm_lora_materializer_rejects_preexisting_symlinked_ancestor(
    tmp_path: Path,
    symlinked_component: str,
) -> None:
    config, snapshot = _lora_evaluation_fixture(tmp_path)
    config.output_root.mkdir(parents=True)
    outside = tmp_path / f"outside-{symlinked_component}"
    outside.mkdir()
    if symlinked_component == "runtime":
        (config.output_root / "runtime").symlink_to(
            outside,
            target_is_directory=True,
        )
    else:
        runtime = config.output_root / "runtime"
        runtime.mkdir()
        (runtime / "lora-adapters").symlink_to(
            outside,
            target_is_directory=True,
        )

    with pytest.raises(ReplayMismatchError, match="contains a symlink"):
        materialize_vllm_lora_adapter(config, snapshot)
    assert tuple(outside.iterdir()) == ()


@pytest.mark.parametrize(
    "filename",
    [
        VLLM_LORA_ADAPTER_MODEL_FILENAME,
        VLLM_LORA_ADAPTER_CONFIG_FILENAME,
        VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
    ],
)
def test_vllm_lora_mutation_before_generate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    config, snapshot = _lora_evaluation_fixture(tmp_path)
    adapter_root = materialize_vllm_lora_adapter(config, snapshot)
    verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    )

    vllm = ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    class Engine:
        calls = 0

        async def generate(self, *args: object, **kwargs: object):
            del args, kwargs
            self.calls += 1
            completion = SimpleNamespace(
                token_ids=[7],
                logprobs=[{7: SimpleNamespace(logprob=-0.25)}],
                finish_reason="stop",
                stop_reason=None,
            )
            yield SimpleNamespace(finished=True, outputs=[completion])

    engine = Engine()
    manager = StandaloneTGVFVLLMManager(
        engine,
        SimpleNamespace(lora_path=str(adapter_root)),
        capture_hidden=False,
        adapter_integrity_verifier=verifier,
    )
    first = asyncio.run(
        manager.generate(
            "fixture-request-1",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 1, "logprobs": True},
            tgvf_expected_step=32,
        )
    )
    assert first.token_ids == [7]
    assert engine.calls == 1

    materialized = adapter_root / filename
    materialized.write_bytes(materialized.read_bytes() + b"tampered")

    with pytest.raises(ReplayMismatchError, match="before engine.generate"):
        asyncio.run(
            manager.generate(
                "fixture-request-2",
                prompt_ids=[1, 2],
                sampling_params={"max_tokens": 1, "logprobs": True},
                tgvf_expected_step=32,
            )
        )
    assert engine.calls == 1
