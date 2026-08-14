from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    POLICY_BENCHMARK_SCHEMA,
    POLICY_EVALUATION_IDENTITY_SCHEMA,
    trajectory_audit_payload,
)
from tgvf_rl.trajectories.schema import (
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/audit_texture_policy_completion.py"
SHA = "a" * 64


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_texture_policy_completion", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_result_identity(row: dict[str, object]) -> None:
    content = dict(row)
    content.pop("result_identity_sha256", None)
    content.pop("wall_seconds", None)
    row["result_identity_sha256"] = _canonical_sha256(content)


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    output_root = (tmp_path / "evaluation").resolve()
    runtime_root = output_root / "runtime"
    inference_root = output_root / "inference"
    runtime_root.mkdir(parents=True)
    inference_root.mkdir()
    image = (tmp_path / "image.bin").resolve()
    image.write_bytes(b"bound-image")

    task_rows = [
        {
            "ordinal": ordinal,
            "dataset": "LAST_2D_Texture_Retrieval" if ordinal < 2 else "MMAD",
            "row_number": ordinal,
            "index": f"texture-{ordinal}",
            "sample_id": f"texture-{ordinal}",
            "question": "Which option is correct?",
            "image_paths": [str(image)],
            "image_sha256s": [hashlib.sha256(image.read_bytes()).hexdigest()],
            "image_dimensions": [[8, 6]],
            "answer": "A",
            "options": [["A", "yes"], ["B", "no"]],
        }
        for ordinal in range(4)
    ]
    tasks_path = runtime_root / "policy-benchmark-tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in task_rows
        ),
        encoding="utf-8",
    )
    tasks_sha256 = hashlib.sha256(tasks_path.read_bytes()).hexdigest()

    policy_config = (tmp_path / "frozen-policy-config.toml").resolve()
    policy_config.write_text("schema_version = 'fixture'\n", encoding="utf-8")
    pointer = (tmp_path / "pointer.json").resolve()
    pointer.write_text("{}\n", encoding="utf-8")
    pointer_sha256 = hashlib.sha256(pointer.read_bytes()).hexdigest()
    config = {
        "schema_version": POLICY_BENCHMARK_SCHEMA,
        "evaluation_id": "TEXTURE-STEP16",
        "policy_config_path": str(policy_config),
        "lora_pointer_path": str(pointer),
        "lora_pointer_sha256": pointer_sha256,
        "output_root": str(output_root),
        "gpu_ids": [0, 1, 2, 3],
        "inference_concurrency_per_gpu": 8,
        "max_model_len": 32768,
        "max_num_batched_tokens": 32768,
        "enable_chunked_prefill": False,
        "gpu_memory_utilization": 0.9,
        "image_max_pixels": 262144,
        "task_manifest_path": str(tasks_path),
        "task_manifest_sha256": tasks_sha256,
        "expected_task_count": 4,
        "expected_single_image_count": 4,
        "expected_policy_run_id": "PRL-FIXTURE",
        "expected_policy_run_identity_sha256": SHA,
        "expected_optimizer_step": 16,
        "expected_policy_weights_sha256": SHA,
        "evaluation_protocol": "training_run",
    }
    config_path = (tmp_path / "policy-benchmark-config.json").resolve()
    _write_json(config_path, config)

    model = ModelIdentity(
        family="qwen3_vl",
        model_name="fixture",
        revision_or_path="fixture",
        tokenizer_length=1,
        chat_template_sha256=SHA,
    )
    identity_content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": "TEXTURE-STEP16",
        "evaluation_schema_version": POLICY_BENCHMARK_SCHEMA,
        "policy_config_path": str(policy_config),
        "policy_config_file_sha256": hashlib.sha256(
            policy_config.read_bytes()
        ).hexdigest(),
        "policy_run_config_identity_sha256": SHA,
        "model_identity": asdict(model),
        "policy_snapshot": {
            "run_id": "PRL-FIXTURE",
            "run_identity_sha256": SHA,
            "optimizer_step": 16,
            "weights_sha256": SHA,
            "pointer_file_sha256": pointer_sha256,
            "manifest_file_sha256": "b" * 64,
            "tensor_file_sha256": "c" * 64,
            "request_sha256": "d" * 64,
        },
        "task_manifest": {
            "path": str(tasks_path),
            "sha256": tasks_sha256,
            "task_count": 4,
            "single_image_count": 4,
        },
        "execution": {
            "world_size": 4,
            "gpu_ids": [0, 1, 2, 3],
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "enable_chunked_prefill": False,
            "inference_concurrency_per_gpu": 8,
            "image_max_pixels": 262144,
        },
    }
    identity = {
        **identity_content,
        "identity_sha256": _canonical_sha256(identity_content),
    }
    _write_json(runtime_root / "evaluation-identity.json", identity)

    payloads: list[dict[str, object]] = []
    for task_row in task_rows:
        task = CoreDevTask(**task_row)
        trajectory = TrajectoryRecord(
            schema_version="trajectory-v1",
            identity=TrajectoryIdentity(
                "TEXTURE-STEP16",
                task.bound_sample_id,
                0,
                f"benchmark:{task.ordinal}",
            ),
            model=model,
            behavior_policy=PolicyVersion("PRL-FIXTURE", 16, SHA),
            assistant_turns=(),
            tool_calls=(),
            observations=(),
            final_answer="A",
            stop=TrajectoryStop.DIRECT_ANSWER,
        )
        payload = trajectory_audit_payload(
            task,
            trajectory,
            evaluation_identity=identity,
            rank=task.ordinal % 4,
            world_size=4,
        )
        payload["wall_seconds"] = float(task.ordinal + 1)
        payload["assistant_turns"] = [
            {"turn_index": turn_index}
            for turn_index in range((2, 1, 3, 2)[task.ordinal])
        ]
        _rewrite_result_identity(payload)
        payloads.append(payload)

    payloads[0]["tool_calls"] = [{"function_name": "tgvf_focus_tool", "call_index": 0}]
    payloads[0]["tool_errors"] = [{"code": "invalid_bbox", "recoverable": True}]
    payloads[0]["stop"] = "tool_error"
    _rewrite_result_identity(payloads[0])
    for rank, payload in enumerate(payloads):
        path = inference_root / f"rank-{rank}.jsonl"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return config_path, output_root, payloads


def test_completion_audit_binds_full_coverage_and_writes_idempotently(
    tmp_path: Path,
) -> None:
    tool = _tool()
    config_path, output_root, _ = _fixture(tmp_path)
    report_path = tmp_path / "completion-audit.json"

    first = tool.materialize_texture_policy_completion_audit(
        config_path,
        output=report_path,
        expected_task_count=4,
    )
    second = tool.materialize_texture_policy_completion_audit(
        config_path,
        output=report_path,
        expected_task_count=4,
    )

    assert first == second
    assert first["complete"] is True
    assert first["coverage"] == {
        "expected_task_count": 4,
        "observed_task_count": 4,
        "missing_count": 0,
        "duplicate_count": 0,
        "first_ordinal": 0,
        "last_ordinal": 3,
    }
    assert [item["row_count"] for item in first["inputs"]["rank_results"]] == [
        1,
        1,
        1,
        1,
    ]
    assert first["summary"]["stop"] == {"direct_answer": 3, "tool_error": 1}
    assert first["summary"]["tool_calls"]["total"] == 1
    assert first["summary"]["assistant_turns"] == {
        "total": 8,
        "mean": 2.0,
        "per_row_count_histogram": {"1": 1, "2": 2, "3": 1},
    }
    assert first["summary"]["tool_errors"] == {
        "total": 1,
        "rows_with_errors": 1,
        "recoverable": 1,
        "non_recoverable": 0,
        "by_code": {"invalid_bbox": 1},
    }
    assert first["summary"]["runtime"]["wall_seconds_sum"] == 10.0
    assert first["summary"]["by_dataset"]["LAST_2D_Texture_Retrieval"] == {
        "row_count": 2,
        "result_kind": {"trajectory": 2},
        "stop": {"direct_answer": 1, "tool_error": 1},
        "assistant_turns": {
            "total": 3,
            "mean": 1.5,
            "per_row_count_histogram": {"1": 1, "2": 1},
        },
        "tool_calls": {
            "total": 1,
            "rows_with_calls": 1,
            "by_function_name": {"tgvf_focus_tool": 1},
            "per_row_count_histogram": {"0": 1, "1": 1},
        },
        "tool_errors": {
            "total": 1,
            "rows_with_errors": 1,
            "recoverable": 1,
            "non_recoverable": 0,
            "by_code": {"invalid_bbox": 1},
        },
        "runtime": {
            "row_count": 2,
            "wall_seconds_sum": 3.0,
            "wall_seconds_mean": 1.5,
            "wall_seconds_min": 1.0,
            "wall_seconds_max": 2.0,
            "wall_seconds_p50_nearest_rank": 1.0,
            "wall_seconds_p95_nearest_rank": 2.0,
        },
    }
    assert first["summary"]["by_dataset"]["MMAD"]["assistant_turns"] == {
        "total": 5,
        "mean": 2.5,
        "per_row_count_histogram": {"2": 1, "3": 1},
    }
    assert first["summary"]["by_dataset"]["MMAD"]["stop"] == {"direct_answer": 2}
    assert first["summary"]["by_dataset"]["MMAD"]["tool_calls"] == {
        "total": 0,
        "rows_with_calls": 0,
        "by_function_name": {},
        "per_row_count_histogram": {"0": 2},
    }
    assert first["summary"]["by_dataset"]["MMAD"]["tool_errors"] == {
        "total": 0,
        "rows_with_errors": 0,
        "recoverable": 0,
        "non_recoverable": 0,
        "by_code": {},
    }
    assert first["summary"]["by_dataset"]["MMAD"]["runtime"] == {
        "row_count": 2,
        "wall_seconds_sum": 7.0,
        "wall_seconds_mean": 3.5,
        "wall_seconds_min": 3.0,
        "wall_seconds_max": 4.0,
        "wall_seconds_p50_nearest_rank": 3.0,
        "wall_seconds_p95_nearest_rank": 4.0,
    }
    assert first["task_manifest"]["task_count"] == 4
    assert json.loads(report_path.read_text(encoding="utf-8")) == first
    assert first["evaluation"]["execution"]["image_max_pixels"] == 262144
    assert Path(first["inputs"]["rank_results"][0]["path"]).parent == (
        output_root / "inference"
    )

    changed = dict(first)
    changed["complete"] = False
    _write_json(report_path, changed)
    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        tool.materialize_texture_policy_completion_audit(
            config_path,
            output=report_path,
            expected_task_count=4,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "incomplete"),
        ("duplicate", "duplicate/invalid"),
        ("wrong_rank", "result rank differs"),
        ("identity_drift", "evaluation_identity_sha256 differs"),
        ("unexpected_rank", "rank file set differs"),
        ("malformed_assistant_turns", "assistant turns are malformed"),
    ),
)
def test_completion_audit_rejects_incomplete_or_drifting_results(
    tmp_path: Path, corruption: str, message: str
) -> None:
    tool = _tool()
    config_path, output_root, payloads = _fixture(tmp_path)
    inference = output_root / "inference"
    if corruption == "missing":
        (inference / "rank-3.jsonl").write_text("", encoding="utf-8")
    elif corruption == "duplicate":
        with (inference / "rank-0.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payloads[0], sort_keys=True) + "\n")
    elif corruption == "wrong_rank":
        (inference / "rank-0.jsonl").write_text("", encoding="utf-8")
        with (inference / "rank-1.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payloads[0], sort_keys=True) + "\n")
    elif corruption == "identity_drift":
        payloads[0]["evaluation_identity_sha256"] = "f" * 64
        _rewrite_result_identity(payloads[0])
        (inference / "rank-0.jsonl").write_text(
            json.dumps(payloads[0], sort_keys=True) + "\n", encoding="utf-8"
        )
    elif corruption == "unexpected_rank":
        (inference / "rank-4.jsonl").write_text("", encoding="utf-8")
    else:
        payloads[0]["assistant_turns"] = 2
        _rewrite_result_identity(payloads[0])
        (inference / "rank-0.jsonl").write_text(
            json.dumps(payloads[0], sort_keys=True) + "\n", encoding="utf-8"
        )

    with pytest.raises((RuntimeError, FileNotFoundError), match=message):
        tool.audit_texture_policy_completion(config_path, expected_task_count=4)


def test_cli_contract_keeps_the_production_population_fixed(tmp_path: Path) -> None:
    tool = _tool()
    config_path, _, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly 42870 tasks"):
        tool.audit_texture_policy_completion(config_path)
