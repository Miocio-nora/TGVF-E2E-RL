from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

import pytest
import torch

import tgvf_rl.framework.verl.policy_weight_sync as policy_weight_sync
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_weight_sync import (
    POLICY_LORA_LATEST_FILENAME,
    PolicyWeightSyncState,
    TGVFPolicyCheckpointEngineManager,
    load_latest_lora_snapshot,
    load_latest_policy_version,
    load_lora_snapshot_pointer,
    load_policy_weight_sync_request,
    lora_parameter_mapping_sha256,
    publish_policy_weight_sync_request,
    wrap_lora_parameter_stream_for_snapshot,
)


RUN_IDENTITY = "7" * 64


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "TGVF_POLICY_STATE_DIR": str((tmp_path / "policy-state").resolve()),
        "TGVF_POLICY_RUN_ID": "policy-pilot-test",
        "TGVF_POLICY_RUN_IDENTITY_SHA256": RUN_IDENTITY,
        "RANK": "0",
        "WORLD_SIZE": "4",
    }


def _stream(step: int = 0) -> list[tuple[str, torch.Tensor]]:
    return [
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.tensor([[1.0 + step, 2.0]], dtype=torch.bfloat16),
        ),
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
            torch.tensor([[3.0], [4.0 + step]], dtype=torch.bfloat16),
        ),
    ]


def _publish(
    tmp_path: Path,
    *,
    step: int,
    rank: int = 0,
) -> tuple[PolicyWeightSyncState, list[tuple[str, torch.Tensor]]]:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, step, nonce=f"request-{step}")
    source = _stream(step)
    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=True,
            rank=rank,
            world_size=4,
            global_steps=step,
            environment=environment,
        )
    )
    return state, observed


def test_rank_zero_publishes_exact_snapshot_without_changing_stream(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    request = publish_policy_weight_sync_request(state, 5, nonce="fixed-request")
    source = _stream(5)
    wrapped = wrap_lora_parameter_stream_for_snapshot(
        iter(source),
        base_sync_done=True,
        rank=0,
        world_size=4,
        global_steps=5,
        environment=environment,
    )

    assert not state.latest_path.exists()
    observed = list(wrapped)

    assert len(observed) == len(source)
    assert all(
        actual is expected for actual, expected in zip(observed, source, strict=True)
    )
    assert all(
        actual[1] is expected[1]
        for actual, expected in zip(observed, source, strict=True)
    )
    snapshot = load_latest_lora_snapshot(
        state,
        expected_optimizer_step=5,
        expected_request_sha256=request.request_sha256,
    )
    expected_mapping = {name: tensor for name, tensor in source}
    assert snapshot.policy_version == PolicyVersion(
        state.run_id,
        5,
        lora_parameter_mapping_sha256(expected_mapping),
    )
    assert tuple(sorted(snapshot.tensors)) == tuple(sorted(expected_mapping))
    assert hashlib.sha256(snapshot.pointer_bytes).hexdigest() == (
        snapshot.pointer_file_sha256
    )
    assert hashlib.sha256(snapshot.manifest_bytes).hexdigest() == (
        snapshot.manifest_file_sha256
    )
    assert hashlib.sha256(snapshot.tensor_bytes).hexdigest() == (
        snapshot.tensor_file_sha256
    )
    for name, tensor in expected_mapping.items():
        torch.testing.assert_close(snapshot.tensors[name], tensor.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_nonzero_ranks_leave_lora_stream_and_state_unwritten(
    tmp_path: Path, rank: int
) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, 2, nonce="non-writer")
    source = _stream(2)

    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=True,
            rank=rank,
            world_size=4,
            global_steps=2,
            environment=environment,
        )
    )

    assert all(
        actual is expected for actual, expected in zip(observed, source, strict=True)
    )
    assert not state.latest_path.exists()


def test_base_model_stream_passes_without_snapshot_state(tmp_path: Path) -> None:
    source = _stream()
    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=False,
            environment={},
        )
    )

    assert all(
        actual is expected for actual, expected in zip(observed, source, strict=True)
    )
    assert not (tmp_path / POLICY_LORA_LATEST_FILENAME).exists()


@pytest.mark.parametrize(
    ("competing_payload", "raises_mismatch"),
    [(b"different concurrent payload", True), (b"expected payload", False)],
)
def test_immutable_snapshot_publish_never_replaces_a_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    competing_payload: bytes,
    raises_mismatch: bool,
) -> None:
    destination = (tmp_path / "lora-snapshots" / "immutable.bin").resolve()
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

    monkeypatch.setattr(policy_weight_sync.os, "link", pause_before_link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        competitor = executor.submit(publish_competitor)
        if raises_mismatch:
            with pytest.raises(
                ReplayMismatchError,
                match="existing content-addressed LoRA test artifact differs",
            ):
                policy_weight_sync._write_immutable_bytes(
                    destination,
                    expected_payload,
                    owner="LoRA test artifact",
                )
        else:
            policy_weight_sync._write_immutable_bytes(
                destination,
                expected_payload,
                owner="LoRA test artifact",
            )
        competitor.result(timeout=5)

    assert destination.read_bytes() == competing_payload
    assert {path.name for path in destination.parent.iterdir()} == {destination.name}


def test_immutable_snapshot_publish_forces_private_mode_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "lora-snapshots" / "private.bin").resolve()
    destination.parent.mkdir()
    previous_umask = os.umask(0o777)
    try:
        policy_weight_sync._write_immutable_bytes(
            destination,
            b"private payload",
            owner="LoRA private artifact",
        )
    finally:
        os.umask(previous_umask)

    assert destination.stat().st_mode & 0o777 == 0o600


def test_immutable_snapshot_existing_inode_metadata_is_strict_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "lora-snapshots" / "strict.bin").resolve()
    destination.parent.mkdir(parents=True)
    payload = b"strict payload"
    destination.write_bytes(payload)
    destination.chmod(0o644)

    with pytest.raises(ReplayMismatchError, match="unsafe inode metadata"):
        policy_weight_sync._write_immutable_bytes(
            destination,
            payload,
            owner="LoRA strict artifact",
        )

    destination.chmod(0o600)
    hardlink = destination.with_name("strict-hardlink.bin")
    os.link(destination, hardlink)
    with pytest.raises(ReplayMismatchError, match="unsafe inode metadata"):
        policy_weight_sync._write_immutable_bytes(
            destination,
            payload,
            owner="LoRA strict artifact",
        )

    hardlink.unlink()
    target_identity = (destination.stat().st_dev, destination.stat().st_ino)
    winner_fsyncs: list[int] = []
    original_fsync = os.fsync

    def record_winner_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == target_identity:
            winner_fsyncs.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(policy_weight_sync.os, "fsync", record_winner_fsync)
    policy_weight_sync._write_immutable_bytes(
        destination,
        payload,
        owner="LoRA strict artifact",
    )

    assert winner_fsyncs
    assert destination.read_bytes() == payload


def test_immutable_snapshot_publish_rejects_symlinked_parent_ancestor(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    destination = Path(
        os.path.abspath(os.fspath(linked / "lora-snapshots" / "immutable.bin"))
    )

    with pytest.raises(ReplayMismatchError, match="contains a symlink"):
        policy_weight_sync._write_immutable_bytes(
            destination,
            b"must not escape",
            owner="LoRA symlink-bound artifact",
        )

    assert tuple(outside.iterdir()) == ()


def test_immutable_snapshot_cleanup_failure_releases_lock_and_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "lora-snapshots" / "cleanup.bin").resolve()
    original_link = os.link
    original_unlink = os.unlink
    unlink_failed = False

    def fail_publication_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("forced publication link failure")

    def fail_first_temporary_unlink(
        path: str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal unlink_failed
        if path.startswith(".cleanup.bin.") and not unlink_failed:
            unlink_failed = True
            raise PermissionError("forced temporary unlink denial")
        original_unlink(path, dir_fd=dir_fd)

    descriptor_count_before = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(policy_weight_sync.os, "link", fail_publication_link)
    monkeypatch.setattr(policy_weight_sync.os, "unlink", fail_first_temporary_unlink)
    with pytest.raises(ReplayMismatchError) as captured:
        policy_weight_sync._write_immutable_bytes(
            destination,
            b"cleanup payload",
            owner="LoRA cleanup artifact",
        )

    notes = getattr(captured.value, "__notes__", ())
    assert any("unlink temporary file" in note for note in notes)
    assert len(os.listdir("/proc/self/fd")) == descriptor_count_before

    monkeypatch.setattr(policy_weight_sync.os, "link", original_link)
    monkeypatch.setattr(policy_weight_sync.os, "unlink", original_unlink)
    second_errors: list[BaseException] = []

    def publish_again() -> None:
        try:
            policy_weight_sync._write_immutable_bytes(
                destination,
                b"cleanup payload",
                owner="LoRA cleanup artifact",
            )
        except BaseException as error:
            second_errors.append(error)

    second = threading.Thread(target=publish_again, daemon=True)
    second.start()
    second.join(timeout=5)

    assert not second.is_alive(), "second publication deadlocked on the leaked lock"
    assert second_errors == []
    assert destination.read_bytes() == b"cleanup payload"
    assert len(os.listdir("/proc/self/fd")) == descriptor_count_before
    for stale in destination.parent.glob(".cleanup.bin.*.tmp"):
        stale.unlink()


def test_strict_latest_load_rejects_safetensors_tampering(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=3)
    snapshot = load_latest_lora_snapshot(state, expected_optimizer_step=3)
    value = bytearray(snapshot.tensor_file.read_bytes())
    value[-1] ^= 1
    snapshot.tensor_file.write_bytes(value)

    with pytest.raises(ReplayMismatchError, match="safetensors file digest"):
        load_latest_policy_version(state, expected_optimizer_step=3)


def test_strict_latest_load_rejects_pointer_tampering(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=4)
    payload = json.loads(state.latest_path.read_text(encoding="utf-8"))
    payload["optimizer_step"] = 9
    state.latest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayMismatchError, match="integrity mismatch"):
        load_latest_policy_version(state)


def test_bound_historical_pointer_is_not_redirected_by_latest(
    tmp_path: Path,
) -> None:
    state, _ = _publish(tmp_path, step=0)
    historical = state.directory / "step-0-pointer.json"
    historical.write_bytes(state.latest_path.read_bytes())
    historical_sha256 = hashlib.sha256(historical.read_bytes()).hexdigest()
    _publish(tmp_path, step=8)

    snapshot = load_lora_snapshot_pointer(
        state,
        pointer_path=historical,
        expected_pointer_file_sha256=historical_sha256,
        expected_optimizer_step=0,
    )

    assert snapshot.policy_version.optimizer_step == 0
    assert snapshot.pointer_file == historical
    with pytest.raises(ReplayMismatchError, match="pointer file digest"):
        load_lora_snapshot_pointer(
            state,
            pointer_path=historical,
            expected_pointer_file_sha256="0" * 64,
        )


def test_bound_pointer_rejects_symlink(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=1)
    symlink = state.directory / "pointer-symlink.json"
    symlink.symlink_to(state.latest_path.name)

    with pytest.raises(ReplayMismatchError, match="missing or unreadable"):
        load_lora_snapshot_pointer(state, pointer_path=symlink)


def test_snapshot_closure_rejects_symlinked_manifest_parent(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=2)
    manifest_parent = state.directory / "lora-manifests"
    moved_parent = tmp_path / "moved-manifests"
    manifest_parent.rename(moved_parent)
    manifest_parent.symlink_to(moved_parent, target_is_directory=True)

    with pytest.raises(ReplayMismatchError, match="manifest.*symlink"):
        load_latest_lora_snapshot(state, expected_optimizer_step=2)


def test_snapshot_closure_rejects_manifest_path_escape(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=3)
    pointer = json.loads(state.latest_path.read_text(encoding="utf-8"))
    pointer.pop("integrity_sha256")
    pointer["manifest_file"] = "../outside-manifest.json"
    pointer["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            pointer,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    state.latest_path.write_bytes(
        json.dumps(
            pointer,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(ReplayMismatchError, match="unsafe path"):
        load_latest_lora_snapshot(state, expected_optimizer_step=3)


def test_snapshot_closure_rejects_root_replacement_without_mixing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = _publish(tmp_path, step=4)
    original_reader = policy_weight_sync._read_relative_file_bytes_at
    moved_root = tmp_path / "original-policy-state"
    replaced = False

    def replace_root_after_pointer(
        root_descriptor: int,
        relative_path: str,
        owner: str,
    ) -> bytes:
        nonlocal replaced
        payload = original_reader(root_descriptor, relative_path, owner)
        if owner == "latest LoRA pointer" and not replaced:
            state.directory.rename(moved_root)
            state.directory.mkdir()
            (state.directory / "lora-manifests").mkdir()
            (state.directory / "lora-snapshots").mkdir()
            replaced = True
        return payload

    monkeypatch.setattr(
        policy_weight_sync,
        "_read_relative_file_bytes_at",
        replace_root_after_pointer,
    )

    with pytest.raises(ReplayMismatchError, match="root changed"):
        load_latest_lora_snapshot(state, expected_optimizer_step=4)

    assert replaced is True


def test_snapshot_closure_rejects_symlinked_state_root(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=5)
    linked_root = tmp_path / "linked-policy-state"
    linked_root.symlink_to(state.directory, target_is_directory=True)
    linked_state = PolicyWeightSyncState(
        directory=linked_root,
        run_id=state.run_id,
        run_identity_sha256=state.run_identity_sha256,
    )

    with pytest.raises(ReplayMismatchError, match="root.*symlink"):
        load_lora_snapshot_pointer(
            linked_state,
            pointer_path=linked_state.latest_path,
            expected_optimizer_step=5,
        )


def test_snapshot_closure_rejects_symlinked_state_root_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    state, _ = _publish(real_parent, step=6)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_state = PolicyWeightSyncState(
        directory=linked_parent / state.directory.name,
        run_id=state.run_id,
        run_identity_sha256=state.run_identity_sha256,
    )

    with pytest.raises(ReplayMismatchError, match="contains a symlink"):
        load_lora_snapshot_pointer(
            linked_state,
            pointer_path=linked_state.latest_path,
            expected_optimizer_step=6,
        )


def test_step_mismatch_fails_before_lora_stream_is_consumed(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, 6, nonce="step-six")
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield from _stream(6)

    with pytest.raises(IdentityMismatchError, match="optimizer step"):
        wrap_lora_parameter_stream_for_snapshot(
            source(),
            base_sync_done=True,
            rank=0,
            world_size=4,
            global_steps=7,
            environment=environment,
        )
    assert consumed is False
    assert not state.latest_path.exists()


class _PublishingUpstreamManager:
    def __init__(self, *, environment: dict[str, str], **kwargs: object) -> None:
        self.environment = environment
        self.constructor_kwargs = kwargs
        self.calls: list[int] = []

    async def update_weights(self, global_steps: int) -> dict[str, int]:
        await asyncio.sleep(0)
        state = PolicyWeightSyncState.from_environment(self.environment)
        request = load_policy_weight_sync_request(state)
        assert request.optimizer_step == global_steps
        list(
            wrap_lora_parameter_stream_for_snapshot(
                iter(_stream(global_steps)),
                base_sync_done=True,
                rank=0,
                world_size=4,
                global_steps=global_steps,
                environment=self.environment,
            )
        )
        self.calls.append(global_steps)
        return {"upstream_step": global_steps}

    def sleep_replicas(self) -> str:
        return "slept"


def test_manager_preserves_pinned_sync_and_async_surface(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    created: list[_PublishingUpstreamManager] = []

    def factory(**kwargs: object) -> _PublishingUpstreamManager:
        manager = _PublishingUpstreamManager(environment=environment, **kwargs)
        created.append(manager)
        return manager

    wrapper = TGVFPolicyCheckpointEngineManager(
        config="config",
        actor_wg="actor",
        replicas=["replica"],
        upstream_manager_factory=factory,
        environment=environment,
    )

    assert wrapper.sleep_replicas() == "slept"
    assert wrapper.update_weights(1) == {"upstream_step": 1}
    assert wrapper.last_policy_version == load_latest_policy_version(
        PolicyWeightSyncState.from_environment(environment),
        expected_optimizer_step=1,
    )

    async def update_inside_loop() -> object:
        pending = wrapper.update_weights(2)
        assert inspect.isawaitable(pending)
        return await pending

    assert asyncio.run(update_inside_loop()) == {"upstream_step": 2}
    assert created[0].calls == [1, 2]
    assert wrapper.last_policy_version is not None
    assert wrapper.last_policy_version.optimizer_step == 2


def test_manager_rejects_upstream_sync_without_exact_snapshot(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    class MissingSnapshotManager:
        async def update_weights(self, global_steps: int) -> dict[str, int]:
            return {"upstream_step": global_steps}

    wrapper = TGVFPolicyCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=[],
        upstream_manager_factory=lambda **kwargs: MissingSnapshotManager(),
        environment=environment,
    )

    with pytest.raises(ReplayMismatchError, match="latest LoRA pointer"):
        wrapper.update_weights(0)
    assert wrapper.last_policy_version is None
