from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
    POLICY_BENCHMARK_SCHEMA,
    POLICY_EVALUATION_IDENTITY_SCHEMA,
    paired_evaluation_rng_for_task,
    trajectory_audit_payload,
    validate_policy_benchmark_result,
)
from tgvf_rl.trajectories.schema import (
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)


SHA = "a" * 64
TASK_SHA = "b" * 64
PROTOCOL_SHA = "c" * 64


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sampling_rng(namespace: str = "coredev2511/paired/temp1/v1") -> dict[str, object]:
    return {
        "schema_version": PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
        "mode": "common_random_numbers_per_task_turn",
        "seed_namespace": namespace,
        "master_seed": 42,
        "task_manifest_sha256": TASK_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "seed_components": [
            "master_seed",
            "seed_namespace",
            "task_manifest_sha256",
            "protocol_sha256",
            "sample_id",
            "rollout_index",
            "assistant_turn_index",
        ],
        "excluded_arm_components": [
            "evaluation_id",
            "arm_name",
            "optimizer_step",
            "checkpoint_hash",
            "policy_weights_sha256",
            "prompt_token_ids_sha256",
        ],
    }


def _identity(
    *,
    step: int,
    weights: str,
    include_rng: bool,
    evaluation_id: str = "PAIRED-EVAL-ARM",
) -> dict[str, object]:
    model = ModelIdentity(
        family="qwen3_vl",
        model_name="fixture",
        revision_or_path="fixture",
        tokenizer_length=1,
        chat_template_sha256=SHA,
    )
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": evaluation_id,
        "evaluation_schema_version": POLICY_BENCHMARK_SCHEMA,
        "policy_config_path": "/immutable/policy.toml",
        "policy_config_file_sha256": SHA,
        "policy_run_config_identity_sha256": SHA,
        "model_identity": asdict(model),
        "policy_snapshot": {
            "run_id": "PRL-17-R2",
            "run_identity_sha256": SHA,
            "optimizer_step": step,
            "weights_sha256": weights,
            "pointer_file_sha256": SHA,
            "manifest_file_sha256": SHA,
            "tensor_file_sha256": SHA,
            "request_sha256": SHA,
        },
        "task_manifest": {
            "path": "/immutable/tasks.jsonl",
            "sha256": TASK_SHA,
            "task_count": 1,
            "single_image_count": 1,
        },
        "execution": {
            "world_size": 4,
            "gpu_ids": [0, 1, 2, 3],
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "enable_chunked_prefill": False,
            "inference_concurrency_per_gpu": 8,
        },
    }
    if include_rng:
        content["sampling_rng"] = _sampling_rng()
    return {**content, "identity_sha256": _canonical_sha256(content)}


def test_same_sample_and_turn_receive_same_seed_across_policy_arms() -> None:
    step0 = paired_evaluation_rng_for_task(
        _identity(
            step=0,
            weights="0" * 64,
            include_rng=True,
            evaluation_id="PAIRED-EVAL-STEP0",
        ),
        sample_id="sample-7",
        rollout_index=0,
    )
    step16 = paired_evaluation_rng_for_task(
        _identity(
            step=16,
            weights="f" * 64,
            include_rng=True,
            evaluation_id="PAIRED-EVAL-STEP16",
        ),
        sample_id="sample-7",
        rollout_index=0,
    )

    first = step0.for_turn(
        (1, 2, 3),
        turn_index=2,
        behavior_policy=PolicyVersion("run-step0", 0, "0" * 64),
    )
    second = step16.for_turn(
        (9, 8, 7, 6),
        turn_index=2,
        behavior_policy=PolicyVersion("run-step16", 16, "f" * 64),
    )

    assert first == second
    assert step0.stream_identity_sha256 == step16.stream_identity_sha256


def test_sample_turn_and_namespace_each_partition_the_rng_stream() -> None:
    identity = _identity(step=8, weights="8" * 64, include_rng=True)
    sample_a = paired_evaluation_rng_for_task(
        identity, sample_id="sample-a", rollout_index=0
    )
    sample_b = paired_evaluation_rng_for_task(
        identity, sample_id="sample-b", rollout_index=0
    )
    other_contract = dict(identity)
    other_contract["sampling_rng"] = _sampling_rng("coredev2511/paired/temp1/v2")
    other_namespace = paired_evaluation_rng_for_task(
        other_contract, sample_id="sample-a", rollout_index=0
    )
    policy = PolicyVersion("run", 8, "8" * 64)

    turn0 = sample_a.for_turn((1,), turn_index=0, behavior_policy=policy)
    turn1 = sample_a.for_turn((1,), turn_index=1, behavior_policy=policy)
    assert turn0 != turn1
    assert turn0 != sample_b.for_turn((1,), turn_index=0, behavior_policy=policy)
    assert turn0 != other_namespace.for_turn((1,), turn_index=0, behavior_policy=policy)


def test_legacy_random_stream_result_cannot_resume_into_paired_identity() -> None:
    task = CoreDevTask(
        ordinal=0,
        dataset="VStarBench",
        row_number=0,
        index="sample-0",
        sample_id="sample-0",
        question="question",
        image_paths=("/immutable/image.png",),
        image_sha256s=(SHA,),
        image_dimensions=((32, 24),),
    )
    model = ModelIdentity(
        family="qwen3_vl",
        model_name="fixture",
        revision_or_path="fixture",
        tokenizer_length=1,
        chat_template_sha256=SHA,
    )
    trajectory = TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=TrajectoryIdentity("PAIRED-EVAL-ARM", "sample-0", 0, "benchmark:0"),
        model=model,
        behavior_policy=PolicyVersion("PRL-17-R2", 8, "8" * 64),
        assistant_turns=(),
        tool_calls=(),
        observations=(),
        final_answer="answer",
        stop=TrajectoryStop.DIRECT_ANSWER,
    )
    legacy_identity = _identity(step=8, weights="8" * 64, include_rng=False)
    paired_identity = _identity(step=8, weights="8" * 64, include_rng=True)
    legacy_payload = trajectory_audit_payload(
        task,
        trajectory,
        evaluation_identity=legacy_identity,
        rank=0,
        world_size=4,
    )

    with pytest.raises(RuntimeError, match="evaluation_identity_sha256 differs"):
        validate_policy_benchmark_result(
            legacy_payload,
            task=task,
            evaluation_identity=paired_identity,
            rank=0,
            world_size=4,
        )

    paired_payload = trajectory_audit_payload(
        task,
        trajectory,
        evaluation_identity=paired_identity,
        rank=0,
        world_size=4,
    )
    assert paired_payload["sampling_rng"] == paired_identity["sampling_rng"]
    assert len(paired_payload["paired_rng_stream_identity_sha256"]) == 64
    validate_policy_benchmark_result(
        paired_payload,
        task=task,
        evaluation_identity=paired_identity,
        rank=0,
        world_size=4,
    )
