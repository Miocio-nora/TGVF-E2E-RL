from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.policy.checkpoint import (
    DATA_CURSOR_OWNER,
    POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA,
    ROLLOUT_RNG_OWNER,
    ROLLOUT_SAMPLER_OWNER,
    CheckpointIdentityHash,
    OpaqueProjectState,
    PilotOptimizerDataCursor,
    PilotProjectCheckpointState,
    PilotRolloutBarrier,
    PilotRunIdentityHashes,
    capture_pilot_project_checkpoint,
    restore_pilot_project_checkpoint,
)
from tgvf_rl.policy.metrics import (
    PilotMetricsAccumulator,
    PilotMetricsCheckpointState,
)


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
SHA4 = "4" * 64


def _run_identity(*, prompt_sha256: str = SHA1) -> PilotRunIdentityHashes:
    return PilotRunIdentityHashes.from_hashes(
        "pilot-run-a",
        {
            "data_manifest": SHA0,
            "policy_config": SHA2,
            "prompt": prompt_sha256,
        },
    )


def _metrics(step: int = 1) -> PilotMetricsAccumulator:
    accumulator = PilotMetricsAccumulator()
    accumulator.restore_checkpoint_state(
        PilotMetricsCheckpointState(
            optimizer_steps=step,
            prompts=step,
            trajectories=8 * step,
            generated_policy_tokens=80 * step,
            reasoning_tokens=64 * step,
            original_visual_tokens=128 * step,
            total_visual_tokens=128 * step,
            step_time_seconds_total=2.5 * step,
        )
    )
    return accumulator


def _checkpoint() -> PilotProjectCheckpointState:
    return capture_pilot_project_checkpoint(
        run_identity=_run_identity(),
        progress=PilotOptimizerDataCursor(
            optimizer_step=1,
            data_cursor=OpaqueProjectState(
                DATA_CURSOR_OWNER, "json-v1", b'{"epoch":0,"next_prompt":4}'
            ),
        ),
        rollout_sampler_state=OpaqueProjectState(
            ROLLOUT_SAMPLER_OWNER, "sampler-v1", b"sampler-state\x00"
        ),
        rollout_rng_state=OpaqueProjectState(
            ROLLOUT_RNG_OWNER, "vllm-rng-v1", b"rng-state\xff"
        ),
        metrics_accumulator=_metrics(),
        policy_version=PolicyVersion("pilot-run-a", 1, SHA3),
        reference_version=PolicyVersion("frozen-qwen3-reference", 0, SHA4),
        rollout_barrier=PilotRolloutBarrier(),
    )


def test_project_checkpoint_json_round_trip_restores_every_owned_state() -> None:
    checkpoint = _checkpoint()
    payload = json.loads(json.dumps(checkpoint.to_checkpoint_mapping()))

    assert payload["schema_version"] == POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA
    assert set(payload) == {
        "schema_version",
        "run_identity",
        "progress",
        "rollout_sampler_state",
        "rollout_rng_state",
        "metrics_state",
        "policy_version",
        "reference_version",
        "rollout_barrier",
        "integrity_sha256",
    }
    assert not {
        "model_state",
        "lora_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
    }.intersection(payload)

    restored = PilotProjectCheckpointState.from_checkpoint_mapping(payload)
    assert restored == checkpoint
    assert restored.to_checkpoint_mapping() == payload
    assert restored.progress.data_cursor.payload == b'{"epoch":0,"next_prompt":4}'
    assert restored.rollout_sampler_state.payload == b"sampler-state\x00"
    assert restored.rollout_rng_state.payload == b"rng-state\xff"

    destination = PilotMetricsAccumulator()
    result = restore_pilot_project_checkpoint(
        payload,
        expected_run_identity=checkpoint.run_identity,
        loaded_policy_version=checkpoint.policy_version,
        loaded_reference_version=checkpoint.reference_version,
        metrics_accumulator=destination,
    )
    assert result == checkpoint
    assert destination.state == checkpoint.metrics_state

    payload["metrics_state"]["optimizer_steps"] = 99
    assert destination.state == checkpoint.metrics_state


def test_restore_fails_closed_on_run_or_framework_weight_identity_mismatch() -> None:
    checkpoint = _checkpoint()
    destination = PilotMetricsAccumulator()
    before = destination.state

    with pytest.raises(IdentityMismatchError, match="run_identity"):
        restore_pilot_project_checkpoint(
            checkpoint,
            expected_run_identity=_run_identity(prompt_sha256=SHA0),
            loaded_policy_version=checkpoint.policy_version,
            loaded_reference_version=checkpoint.reference_version,
            metrics_accumulator=destination,
        )
    assert destination.state == before

    with pytest.raises(IdentityMismatchError, match="policy_version"):
        restore_pilot_project_checkpoint(
            checkpoint,
            expected_run_identity=checkpoint.run_identity,
            loaded_policy_version=replace(checkpoint.policy_version, weights_sha256=SHA2),
            loaded_reference_version=checkpoint.reference_version,
            metrics_accumulator=destination,
        )
    assert destination.state == before

    with pytest.raises(IdentityMismatchError, match="reference_version"):
        restore_pilot_project_checkpoint(
            checkpoint,
            expected_run_identity=checkpoint.run_identity,
            loaded_policy_version=checkpoint.policy_version,
            loaded_reference_version=replace(
                checkpoint.reference_version, weights_sha256=SHA2
            ),
            metrics_accumulator=destination,
        )
    assert destination.state == before


def test_corruption_and_unknown_fields_are_rejected_before_metrics_mutation() -> None:
    checkpoint = _checkpoint()
    destination = PilotMetricsAccumulator()
    before = destination.state

    corrupt = deepcopy(checkpoint.to_checkpoint_mapping())
    corrupt["rollout_rng_state"]["payload_base64"] = "bm90LXRoZS1ybmc="
    with pytest.raises(ReplayMismatchError, match="integrity mismatch"):
        restore_pilot_project_checkpoint(
            corrupt,
            expected_run_identity=checkpoint.run_identity,
            loaded_policy_version=checkpoint.policy_version,
            loaded_reference_version=checkpoint.reference_version,
            metrics_accumulator=destination,
        )
    assert destination.state == before

    missing = deepcopy(checkpoint.to_checkpoint_mapping())
    del missing["progress"]
    with pytest.raises(ReplayMismatchError, match="fields differ"):
        PilotProjectCheckpointState.from_checkpoint_mapping(missing)

    extra = deepcopy(checkpoint.to_checkpoint_mapping())
    extra["optimizer_state"] = {"forbidden": True}
    with pytest.raises(ReplayMismatchError, match="extra=.*optimizer_state"):
        PilotProjectCheckpointState.from_checkpoint_mapping(extra)

    opaque = checkpoint.rollout_rng_state.to_checkpoint_mapping()
    opaque["payload_sha256"] = SHA0
    with pytest.raises(ReplayMismatchError, match="digest mismatch"):
        OpaqueProjectState.from_checkpoint_mapping(opaque)


def test_quiescent_barrier_and_cross_state_step_invariants_are_mandatory() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="zero rollout staleness"):
        PilotRolloutBarrier(asynchronous_staleness_steps=1)
    with pytest.raises(ValueError, match="no outstanding rollouts"):
        PilotRolloutBarrier(outstanding_rollout_count=1)

    next_cursor = replace(checkpoint.progress, optimizer_step=2)
    with pytest.raises(ValueError, match="metrics optimizer step"):
        replace(checkpoint, progress=next_cursor)
    with pytest.raises(ValueError, match="policy version"):
        replace(
            checkpoint,
            progress=next_cursor,
            metrics_state=_metrics(2).state,
        )
    with pytest.raises(ValueError, match="frozen Pilot reference"):
        replace(
            checkpoint,
            reference_version=replace(checkpoint.reference_version, optimizer_step=1),
        )

    with pytest.raises(ValueError, match="rollout sampler owner"):
        replace(
            checkpoint,
            rollout_sampler_state=OpaqueProjectState(
                ROLLOUT_RNG_OWNER, "sampler-v1", b"wrong-owner"
            ),
        )
    with pytest.raises(TypeError, match="immutable bytes"):
        OpaqueProjectState(ROLLOUT_RNG_OWNER, "rng-v1", bytearray(b"mutable"))
    with pytest.raises(FrozenInstanceError):
        checkpoint.progress.optimizer_step = 4  # type: ignore[misc]

    with pytest.raises(ValueError, match="unique and sorted"):
        PilotRunIdentityHashes(
            "pilot-run-a",
            (
                CheckpointIdentityHash("prompt", SHA1),
                CheckpointIdentityHash("data_manifest", SHA0),
            ),
        )
