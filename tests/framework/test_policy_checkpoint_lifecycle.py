from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.compatibility import FSDP2BridgeConfig
from tgvf_rl.framework.verl.checkpoint_bridge import (
    POLICY_PILOT_CHECKPOINT_PAIR_FILENAME,
    POLICY_PILOT_PROJECT_STATE_FILENAME,
    PolicyPilotVerlCheckpointPair,
)
from tgvf_rl.framework.verl.policy_checkpoint_lifecycle import (
    POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
    POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME,
    PolicyCheckpointLifecycle,
    policy_checkpoint_lifecycle_from_runtime,
)
from tgvf_rl.policy.checkpoint import (
    DATA_CURSOR_OWNER,
    ROLLOUT_RNG_OWNER,
    ROLLOUT_SAMPLER_OWNER,
    OpaqueProjectState,
    PilotOptimizerDataCursor,
    PilotProjectCheckpointState,
    PilotRolloutBarrier,
    PilotRunIdentityHashes,
)
from tgvf_rl.policy.metrics import PilotMetricsCheckpointState


_SHA0 = "0" * 64
_SHA1 = "1" * 64
_SHA2 = "2" * 64


def _identity(run_id: str = "prl15-test") -> PilotRunIdentityHashes:
    return PilotRunIdentityHashes.from_hashes(
        run_id,
        {"config": _SHA0, "dataset": _SHA1, "prompt": _SHA2},
    )


def _materialize_generation(
    root: Path,
    step: int,
    *,
    identity: PilotRunIdentityHashes,
    world_size: int = 2,
) -> Path:
    generation = root / f"global_step_{step}"
    actor = generation / "actor"
    actor.mkdir(parents=True)
    state = PilotProjectCheckpointState(
        run_identity=identity,
        progress=PilotOptimizerDataCursor(
            step,
            OpaqueProjectState(
                DATA_CURSOR_OWNER,
                "json-v1",
                json.dumps({"step": step}).encode(),
            ),
        ),
        rollout_sampler_state=OpaqueProjectState(
            ROLLOUT_SAMPLER_OWNER, "json-v1", f"sampler-{step}".encode()
        ),
        rollout_rng_state=OpaqueProjectState(
            ROLLOUT_RNG_OWNER, "json-v1", f"rng-{step}".encode()
        ),
        metrics_state=PilotMetricsCheckpointState(
            optimizer_steps=step,
            prompts=step,
            trajectories=16 * step,
            generated_policy_tokens=16 * step,
            reasoning_tokens=8 * step,
            original_visual_tokens=16 * step,
            total_visual_tokens=16 * step,
            step_time_seconds_total=float(step),
        ),
        policy_version=PolicyVersion(identity.run_id, step, _SHA1),
        reference_version=PolicyVersion("frozen-reference", 0, _SHA2),
        rollout_barrier=PilotRolloutBarrier(),
    )
    pair = PolicyPilotVerlCheckpointPair(
        run_id=identity.run_id,
        optimizer_step=step,
        project_state_sha256=state.integrity_sha256,
        upstream_save_contents=("model", "optimizer", "extra"),
        upstream_load_contents=("model", "optimizer", "extra"),
    )
    (actor / POLICY_PILOT_PROJECT_STATE_FILENAME).write_text(
        json.dumps(state.to_checkpoint_mapping()), encoding="utf-8"
    )
    (actor / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME).write_text(
        json.dumps(pair.to_checkpoint_mapping()), encoding="utf-8"
    )
    (actor / "fsdp_config.json").write_text("{}\n", encoding="utf-8")
    for rank in range(world_size):
        for prefix in ("model", "optim", "extra_state"):
            (actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                f"{prefix}-{step}-{rank}".encode()
            )
    (generation / "data.pt").write_bytes(f"data-{step}".encode())
    return generation


def _lifecycle(
    root: Path,
    identity: PilotRunIdentityHashes,
    *,
    permanent_steps: tuple[int, ...] = (8,),
) -> PolicyCheckpointLifecycle:
    return PolicyCheckpointLifecycle(
        checkpoint_root=root,
        maximum_checkpoints_to_keep=2,
        checkpoint_steps=tuple(range(9)),
        every_completed_step=True,
        permanent_steps=permanent_steps,
        permanent_root=(root.parent / "permanent-checkpoints")
        if permanent_steps
        else None,
        world_size=2,
        run_identity=identity,
        fsdp2=FSDP2BridgeConfig(),
    )


def _write_tracker(root: Path, step: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")


def test_restart_scan_prunes_before_save_and_keeps_latest_tracker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    identity = _identity()
    _materialize_generation(root, 1, identity=identity)
    _materialize_generation(root, 2, identity=identity)
    _write_tracker(root, 2)

    # A fresh lifecycle object models a new process: no in-memory path list is
    # carried over from the process that wrote steps 1 and 2.
    _lifecycle(root, identity).prepare_for_save(3)
    assert not (root / "global_step_1").exists()
    assert (root / "global_step_2/actor/model_world_size_2_rank_0.pt").is_file()
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "2"

    _materialize_generation(root, 3, identity=identity)
    _write_tracker(root, 3)
    restarted = _lifecycle(root, identity)
    restarted.finalize_saved_checkpoint(3)
    assert [item.optimizer_step for item in restarted.scan_committed()] == [2, 3]


def test_step8_is_hard_linked_permanently_and_finalization_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    identity = _identity()
    _materialize_generation(root, 7, identity=identity)
    source = _materialize_generation(root, 8, identity=identity)
    _write_tracker(root, 8)
    lifecycle = _lifecycle(root, identity)

    lifecycle.finalize_saved_checkpoint(8)
    lifecycle.finalize_saved_checkpoint(8)

    permanent = tmp_path / "permanent-checkpoints/global_step_8"
    assert (permanent / POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME).is_file()
    source_shard = source / "actor/model_world_size_2_rank_0.pt"
    retained_shard = permanent / "actor/model_world_size_2_rank_0.pt"
    assert os.stat(source_shard).st_ino == os.stat(retained_shard).st_ino
    assert os.stat(source_shard).st_dev == os.stat(retained_shard).st_dev
    expected_payload = retained_shard.read_bytes()
    source_shard.unlink()
    assert retained_shard.read_bytes() == expected_payload


def test_multiple_permanent_steps_survive_rolling_prune_across_restarts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    identity = _identity()
    permanent_steps = (4, 5, 6, 8)

    for step in range(1, 9):
        # Reconstructing the lifecycle each step proves that retention does not
        # depend on an in-memory list from the process that saved the source.
        lifecycle = _lifecycle(root, identity, permanent_steps=permanent_steps)
        lifecycle.prepare_for_save(step)
        _materialize_generation(root, step, identity=identity)
        _write_tracker(root, step)
        lifecycle.finalize_saved_checkpoint(step)

    restarted = _lifecycle(root, identity, permanent_steps=permanent_steps)
    assert [item.optimizer_step for item in restarted.scan_committed()] == [7, 8]
    for step in permanent_steps:
        permanent = tmp_path / f"permanent-checkpoints/global_step_{step}"
        receipt = json.loads(
            (permanent / POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert receipt["optimizer_step"] == step
        assert (
            permanent / "actor/model_world_size_2_rank_0.pt"
        ).read_bytes() == f"model-{step}-0".encode()


def test_prepare_backfills_permanent_step_after_crash_before_finalize(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    identity = _identity()
    permanent_steps = (4, 8)
    _materialize_generation(root, 3, identity=identity)
    step4 = _materialize_generation(root, 4, identity=identity)
    _write_tracker(root, 4)

    # Model a process death after upstream committed the checkpoint and tracker
    # but before lifecycle.finalize_saved_checkpoint(4) could retain it.
    restarted = _lifecycle(root, identity, permanent_steps=permanent_steps)
    restarted.prepare_for_save(5)

    permanent = tmp_path / "permanent-checkpoints/global_step_4"
    source_shard = step4 / "actor/model_world_size_2_rank_0.pt"
    retained_shard = permanent / "actor/model_world_size_2_rank_0.pt"
    assert (permanent / POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME).is_file()
    assert os.stat(source_shard).st_ino == os.stat(retained_shard).st_ino

    _materialize_generation(root, 5, identity=identity)
    _write_tracker(root, 5)
    restarted.finalize_saved_checkpoint(5)
    _lifecycle(root, identity, permanent_steps=permanent_steps).prepare_for_save(6)

    assert not step4.exists()
    assert retained_shard.read_bytes() == b"model-4-0"


def test_foreign_or_partial_generation_is_never_a_retention_deletion_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    identity = _identity()
    foreign = _materialize_generation(root, 1, identity=_identity("another-run"))
    partial = root / "global_step_2"
    partial.mkdir(parents=True)
    exact = _materialize_generation(root, 3, identity=identity)
    _write_tracker(root, 3)

    _lifecycle(root, identity).prepare_for_save(4)
    assert foreign.is_dir()
    assert partial.is_dir()
    assert exact.is_dir()


def test_runtime_record_preserves_low_cost_nonformal_checkpoint_behavior(
    tmp_path: Path,
) -> None:
    root = tmp_path / "smoke/checkpoints"
    record = {
        "schema_version": POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
        "checkpoint_steps": [0, 1],
        "every_completed_step": False,
        "rolling_retention_across_restarts": True,
        "rolling_max_checkpoints": 2,
        "permanent_steps": [],
        "permanent_directory": "",
    }
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                custom={"checkpoint_steps": [0, 1], "checkpoint_lifecycle": record}
            )
        ),
        trainer=SimpleNamespace(
            max_actor_ckpt_to_keep=2,
            default_local_dir=str(root),
            total_training_steps=1,
        ),
    )
    lifecycle = policy_checkpoint_lifecycle_from_runtime(
        config, run_identity=_identity(), world_size=2
    )

    assert lifecycle is not None
    assert lifecycle.checkpoint_steps == (0, 1)
    assert lifecycle.every_completed_step is False
    assert lifecycle.permanent_steps == ()
    assert lifecycle.permanent_root is None
