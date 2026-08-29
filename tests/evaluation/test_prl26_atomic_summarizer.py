from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/summarize_prl26_e_atomic_s32_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("prl26_atomic_summarizer", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_TASK_MANIFEST = "a" * 64
_EVALUATION_ID = "TEST-ATOMIC-STEP32"
_EXCLUSIONS = [
    "evaluation_id",
    "arm_name",
    "optimizer_step",
    "checkpoint_hash",
    "policy_weights_sha256",
    "prompt_token_ids_sha256",
]
_PLANNED_RNG = {
    "schema_version": "tgvf.policy-paired-evaluation-rng-plan.v1",
    "mode": "common_random_numbers_per_task_turn",
    "seed_namespace": "test/atomic/seed42/v1",
    "master_seed": 42,
    "task_manifest_sha256": _TASK_MANIFEST,
    "protocol_sha256": "1" * 64,
    "temperature": 1.0,
    "do_sample": True,
    "excluded_arm_components": _EXCLUSIONS,
}
_RUNTIME_RNG = {
    "schema_version": "tgvf-policy-paired-evaluation-rng-v1",
    "mode": "common_random_numbers_per_task_turn",
    "seed_namespace": "test/atomic/seed42/v1",
    "master_seed": 42,
    "task_manifest_sha256": _TASK_MANIFEST,
    "protocol_sha256": "1" * 64,
    "excluded_arm_components": _EXCLUSIONS,
    "seed_components": [
        "master_seed",
        "seed_namespace",
        "task_manifest_sha256",
        "protocol_sha256",
        "sample_id",
        "rollout_index",
        "assistant_turn_index",
    ],
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    result.pop("result_identity_sha256", None)
    hash_payload = dict(result)
    hash_payload.pop("wall_seconds", None)
    result["result_identity_sha256"] = _MODULE._canonical_sha256(hash_payload)
    return result


def _identity_row(ordinal: int) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "tgvf-policy-benchmark-trajectory-audit-v1",
        "selection_reasons": ["representative_rollout_zero"],
        "evaluation_identity_sha256": "2" * 64,
        "policy_run_identity_sha256": "3" * 64,
        "policy_paired_snapshot_identity_sha256": "4" * 64,
        "policy_qwen_tree_sha256": "5" * 64,
        "policy_rp66_state_sha256": "6" * 64,
        "policy_rp66_storage_sha256": "7" * 64,
        "policy_config_identity_sha256": "3" * 64,
        "task_manifest_sha256": _TASK_MANIFEST,
        "model_identity": {"model_name": "Qwen3-VL-8B-Instruct"},
        "rank": ordinal % 4,
        "world_size": 4,
        "evaluation_id": _EVALUATION_ID,
        "sample_id": f"sample-{ordinal}",
        "group_uid": f"benchmark:{ordinal}",
        "rollout_index": 0,
        "ordinal": ordinal,
        "dataset": "VStarBench",
        "row_number": ordinal,
        "index": ordinal,
        "question": "question",
        "image_paths": ["image.png"],
        "image_sha256s": ["8" * 64],
        "image_dimensions": [[512, 512]],
        "trajectory_id": f"trajectory-{ordinal}",
        "policy_run_id": "TEST-ATOMIC-RUN",
        "optimizer_step": 32,
        "policy_weights_sha256": "9" * 64,
        "policy_snapshot_backend": _MODULE.POLICY_SNAPSHOT_BACKEND,
        "sampling_rng": dict(_RUNTIME_RNG),
        "paired_rng_stream_identity_sha256": _digest(f"stream-{ordinal}"),
        "trajectory_sha256": _digest(f"trajectory-{ordinal}"),
        "stop": "direct_answer",
        "final_answer": "answer",
        "assistant_turns": [
            {
                "turn_index": 0,
                "raw_text": "answer",
                "sampled_token_count": 1,
                "is_tool_call": False,
                "stop_reason": "stop",
            }
        ],
        "tool_calls": [],
        "tool_errors": [],
        "successful_observation_count": 0,
        "wall_seconds": 0.5,
    }
    return _rehash(row)


def _turn(index: int, tokens: int, *, tool: bool) -> dict[str, object]:
    return {
        "turn_index": index,
        "raw_text": "tool" if tool else "answer",
        "sampled_token_count": tokens,
        "is_tool_call": tool,
        "stop_reason": "stop",
    }


def test_usage_distinguishes_attempts_successes_and_errors() -> None:
    direct = {
        "assistant_turns": [_turn(0, 5, tool=False)],
        "tool_calls": [],
        "tool_errors": [],
        "successful_observation_count": 0,
        "stop": "direct_answer",
    }
    successful = {
        "assistant_turns": [_turn(0, 7, tool=True), _turn(1, 3, tool=False)],
        "tool_calls": [
            {
                "call_index": 0,
                "assistant_turn_index": 0,
                "function_name": "tgvf_crop_tool",
            }
        ],
        "tool_errors": [],
        "successful_observation_count": 1,
        "stop": "final_answer",
    }
    error_only = {
        "assistant_turns": [_turn(0, 4, tool=True), _turn(1, 6, tool=False)],
        "tool_calls": [],
        "tool_errors": [
            {
                "attempt_index": 0,
                "assistant_turn_index": 0,
                "function_name": None,
                "code": "tool_parse.invalid_tool_name",
                "payload_json": "{}",
                "recoverable": True,
            }
        ],
        "successful_observation_count": 0,
        "stop": "final_answer",
    }

    usage = _MODULE._usage([direct, successful, error_only])

    assert usage["trajectory_count"] == 3
    assert usage["no_tool_trajectory_count"] == 1
    assert usage["trajectories_attempting_tool"] == 2
    assert usage["tool_attempt_trajectory_rate"] == pytest.approx(2 / 3)
    assert usage["total_tool_attempts"] == 2
    assert usage["trajectories_with_successful_tool_call"] == 1
    assert usage["successful_tool_call_trajectory_rate"] == pytest.approx(1 / 3)
    assert usage["successful_tool_call_count"] == 1
    assert usage["trajectories_with_repeat_successful_tool_call"] == 0
    assert usage["repeat_successful_tool_call_count"] == 0
    assert usage["successful_tool_attempt_rate"] == 0.5
    assert usage["trajectories_with_tool_error"] == 1
    assert usage["tool_error_count"] == 1
    assert usage["tool_error_code_counts"] == {
        "tool_parse.invalid_tool_name": 1
    }
    assert usage["tool_errors_without_function_name"] == 1
    assert usage["successful_observation_count"] == 1
    assert usage["generated_token_mean"] == pytest.approx(25 / 3)


def test_usage_rejects_success_without_an_observation() -> None:
    row = {
        "assistant_turns": [_turn(0, 2, tool=True)],
        "tool_calls": [
            {
                "call_index": 0,
                "assistant_turn_index": 0,
                "function_name": "tgvf_crop_tool",
            }
        ],
        "tool_errors": [],
        "successful_observation_count": 0,
        "stop": "max_tokens",
    }

    with pytest.raises(RuntimeError, match="success/attempt counts"):
        _MODULE._usage([row])


def test_identity_closure_is_order_stable_and_binds_all_sequences() -> None:
    rows = [_identity_row(5), _identity_row(0)]

    closure = _MODULE._identity_closure(
        rows,
        expected_evaluation_id=_EVALUATION_ID,
        expected_task_manifest_sha256=_TASK_MANIFEST,
        planned_paired_rng=_PLANNED_RNG,
    )
    reversed_closure = _MODULE._identity_closure(
        list(reversed(rows)),
        expected_evaluation_id=_EVALUATION_ID,
        expected_task_manifest_sha256=_TASK_MANIFEST,
        planned_paired_rng=_PLANNED_RNG,
    )

    assert closure == reversed_closure
    assert closure["result_row_count"] == 2
    assert closure["ordinal_count"] == 2
    assert closure["sample_id_count"] == 2
    assert closure["result_identity_count"] == 2
    assert closure["paired_rng_stream_count"] == 2
    for field in (
        "ordinal_sequence_sha256",
        "sample_id_sequence_sha256",
        "result_identity_sequence_sha256",
        "paired_rng_stream_sequence_sha256",
    ):
        assert len(closure[field]) == 64


def test_identity_closure_rejects_tampering_and_duplicate_rng_streams() -> None:
    first = _identity_row(0)
    tampered = dict(first)
    tampered["final_answer"] = "changed after publication"
    with pytest.raises(RuntimeError, match="result identity digest"):
        _MODULE._identity_closure(
            [tampered],
            expected_evaluation_id=_EVALUATION_ID,
            expected_task_manifest_sha256=_TASK_MANIFEST,
        )

    second = _identity_row(4)
    second["paired_rng_stream_identity_sha256"] = first[
        "paired_rng_stream_identity_sha256"
    ]
    second = _rehash(second)
    with pytest.raises(RuntimeError, match="RNG stream identity is duplicated"):
        _MODULE._identity_closure(
            [first, second],
            expected_evaluation_id=_EVALUATION_ID,
            expected_task_manifest_sha256=_TASK_MANIFEST,
        )


def test_identity_closure_binds_exact_manifest_task_selection() -> None:
    row = _identity_row(0)
    expected_task = {
        key: row[key]
        for key in (
            "sample_id",
            "dataset",
            "row_number",
            "index",
            "question",
            "image_paths",
            "image_sha256s",
            "image_dimensions",
        )
    }
    expected_sequence_sha256 = _MODULE._canonical_sha256(
        [{"ordinal": 0, **expected_task}]
    )
    closure = _MODULE._identity_closure(
        [row],
        expected_evaluation_id=_EVALUATION_ID,
        expected_task_manifest_sha256=_TASK_MANIFEST,
        expected_tasks={0: expected_task},
        expected_task_sequence_sha256=expected_sequence_sha256,
    )
    assert (
        closure["manifest_single_image_task_sequence_sha256"]
        == expected_sequence_sha256
    )

    row["question"] = "a different task"
    row = _rehash(row)
    with pytest.raises(RuntimeError, match="differs from bound manifest"):
        _MODULE._identity_closure(
            [row],
            expected_evaluation_id=_EVALUATION_ID,
            expected_task_manifest_sha256=_TASK_MANIFEST,
            expected_tasks={0: expected_task},
            expected_task_sequence_sha256=expected_sequence_sha256,
        )


def test_repository_file_resolution_rejects_original_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.toml"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.toml"
    link.symlink_to(target)
    monkeypatch.setattr(_MODULE, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="cannot be a symlink"):
        _MODULE._resolve_repository_file("link.toml", name="test config")


def test_paired_summary_must_bind_official_summary_and_row_identity() -> None:
    official = {"status": "pass", "sample_count": 2511}
    evaluation_identity = "b" * 64
    expected_sampling = {"source": "bound_policy_run_config"}
    paired = {
        "schema_version": "tgvf.prl15-paired-coredev-summary.v1",
        "evaluation_id": _MODULE.EVALUATION_ID,
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "multi_image_policy": "unsupported_explicit_hold",
        },
        "materialization": {_MODULE.ARM_NAME: {"status": "complete"}},
        "sampling": expected_sampling,
        "arms": {
            _MODULE.ARM_NAME: {
                "optimizer_step": 32,
                "evaluation_identity_sha256": evaluation_identity,
                "official_summary": official,
            }
        },
        _MODULE.ARM_NAME: official,
    }

    _MODULE._validate_paired_summary(
        paired,
        summary=official,
        closure={"evaluation_identity_sha256": evaluation_identity},
        plan_contract={"expected_sampling": expected_sampling},
    )
    paired["arms"][_MODULE.ARM_NAME]["evaluation_identity_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="paired arm identity"):
        _MODULE._validate_paired_summary(
            paired,
            summary=official,
            closure={"evaluation_identity_sha256": evaluation_identity},
            plan_contract={"expected_sampling": expected_sampling},
        )
