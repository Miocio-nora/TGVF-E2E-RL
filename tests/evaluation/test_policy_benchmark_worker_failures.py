from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tgvf_rl.contracts.errors import PolicyOutputContractError, ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.evaluation.policy_benchmark_scoring import (
    materialize_policy_benchmark_mcq_scoring,
)
from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    POLICY_BENCHMARK_SCHEMA,
    POLICY_EVALUATION_IDENTITY_SCHEMA,
    POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA,
    load_policy_benchmark_results,
    policy_output_contract_failure_audit_payload,
    validate_policy_benchmark_result,
)


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_policy_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_policy_benchmark", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_SHA = "a" * 64


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _task(ordinal: int) -> CoreDevTask:
    return CoreDevTask(
        ordinal=ordinal,
        dataset="fixture",
        row_number=ordinal,
        index=f"row-{ordinal}",
        sample_id=f"sample-{ordinal}",
        question="Which option is correct?",
        image_paths=(f"/immutable/image-{ordinal}.png",),
    )


def _evaluation_identity(
    *, task_manifest_sha256: str = _SHA, task_count: int = 1
) -> dict[str, object]:
    model = ModelIdentity(
        family="qwen3_vl",
        model_name="fixture",
        revision_or_path="/immutable/model",
        tokenizer_length=1,
        chat_template_sha256=_SHA,
    )
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": "FIXTURE-EVALUATION",
        "evaluation_schema_version": POLICY_BENCHMARK_SCHEMA,
        "policy_config_path": "/immutable/policy.toml",
        "policy_config_file_sha256": _SHA,
        "policy_run_config_identity_sha256": _SHA,
        "model_identity": asdict(model),
        "policy_snapshot": {
            "run_id": "FIXTURE-RUN",
            "run_identity_sha256": _SHA,
            "optimizer_step": 8,
            "weights_sha256": _SHA,
            "pointer_file_sha256": _SHA,
            "manifest_file_sha256": _SHA,
            "tensor_file_sha256": _SHA,
            "request_sha256": _SHA,
        },
        "task_manifest": {
            "path": "/immutable/tasks.jsonl",
            "sha256": task_manifest_sha256,
            "task_count": task_count,
            "single_image_count": task_count,
        },
        "execution": {
            "world_size": 1,
            "gpu_ids": [0],
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "enable_chunked_prefill": False,
            "inference_concurrency_per_gpu": 2,
        },
    }
    return {**content, "identity_sha256": _canonical_sha256(content)}


def _policy_output_error() -> PolicyOutputContractError:
    return PolicyOutputContractError(
        "vLLM emitted a tool-call suffix outside the run-bound contract",
        code="tool_call_terminal_suffix",
        diagnostic={
            "response_text_sha256": "1" * 64,
            "suffix_sha256": "2" * 64,
            "suffix_char_count": 3,
            "suffix_utf8_byte_count": 3,
            "finish_reason": "stop",
            "stop_reason": "</tool_call>",
            "backend_request_sha256": "3" * 64,
            "backend_response_sha256": "4" * 64,
        },
    )


class _Engine:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


def _patch_worker_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tasks: tuple[CoreDevTask, ...],
    evaluator_type: type,
) -> tuple[argparse.Namespace, SimpleNamespace, _Engine]:
    config = SimpleNamespace(
        gpu_ids=(0,),
        output_root=tmp_path,
        evaluation_protocol="training_run",
        inference_concurrency_per_gpu=max(1, len(tasks)),
    )
    args = argparse.Namespace(rank=0, world_size=1, max_tasks=-1)
    engine = _Engine()
    run = SimpleNamespace(model=SimpleNamespace(revision_or_path="/immutable/model"))

    async def build_manager(_config, _snapshot):
        return object(), engine, run

    fake_transformers = ModuleType("transformers")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    fake_transformers.AutoProcessor = AutoProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(_MODULE, "load_bound_policy_benchmark_tasks", lambda _: tasks)
    monkeypatch.setattr(
        _MODULE, "load_frozen_policy_evaluation_snapshot", lambda _: object()
    )
    monkeypatch.setattr(
        _MODULE,
        "write_policy_evaluation_identity",
        lambda *_args: {"identity_sha256": "a" * 64},
    )
    monkeypatch.setattr(_MODULE, "load_policy_benchmark_results", lambda *_a, **_k: {})
    monkeypatch.setattr(_MODULE, "build_standalone_manager", build_manager)
    monkeypatch.setattr(_MODULE, "PolicyCoreDevEvaluator", evaluator_type)
    monkeypatch.setattr(
        _MODULE,
        "trajectory_audit_payload",
        lambda task, _trajectory, **_kwargs: {
            "ordinal": task.ordinal,
            "result_kind": "trajectory",
            "final_answer": "A",
        },
    )
    return args, config, engine


def test_policy_output_failure_row_is_identity_bound_and_resume_complete(
    tmp_path: Path,
) -> None:
    task = _task(0)
    identity = _evaluation_identity()
    payload = policy_output_contract_failure_audit_payload(
        task,
        _policy_output_error(),
        evaluation_identity=identity,
        rank=0,
        world_size=1,
    )

    assert payload["result_kind"] == "sample_local_failure"
    assert payload["trajectory_available"] is False
    assert payload["stop"] == "invalid_format"
    assert payload["final_answer"] is None
    assert "trajectory_sha256" not in payload
    assert payload["failure"]["schema_version"] == (
        POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA
    )
    assert payload["failure"]["code"] == "tool_call_terminal_suffix"
    assert "trailing" not in json.dumps(payload, sort_keys=True)

    validate_policy_benchmark_result(
        payload,
        task=task,
        evaluation_identity=identity,
        rank=0,
        world_size=1,
    )
    inference = tmp_path / "inference"
    inference.mkdir()
    (inference / "rank-0.jsonl").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert (
        load_policy_benchmark_results(
            inference,
            tasks=(task,),
            evaluation_identity=identity,
            require_complete=True,
        )[0]["result_kind"]
        == "sample_local_failure"
    )


def test_policy_output_failure_stays_in_mcq_denominator_as_incorrect(
    tmp_path: Path,
) -> None:
    row = {
        "ordinal": 0,
        "dataset": "fixture",
        "row_number": 0,
        "index": "row-0",
        "sample_id": "row-0",
        "question": "Which option is correct?",
        "image_paths": ["/immutable/image-0.png"],
        "image_sha256s": ["5" * 64],
        "image_dimensions": [[32, 24]],
        "answer": "A",
        "options": [["A", "yes"], ["B", "no"]],
    }
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    tasks_sha256 = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    task = CoreDevTask(**row)
    identity = _evaluation_identity(
        task_manifest_sha256=tasks_sha256,
        task_count=1,
    )
    identity_path = tmp_path / "evaluation-identity.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inference = tmp_path / "inference"
    inference.mkdir()
    payload = policy_output_contract_failure_audit_payload(
        task,
        _policy_output_error(),
        evaluation_identity=identity,
        rank=0,
        world_size=1,
    )
    (inference / "rank-0.jsonl").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = materialize_policy_benchmark_mcq_scoring(
        inference_root=inference,
        tasks_path=tasks_path,
        tasks_sha256=tasks_sha256,
        evaluation_identity_path=identity_path,
        evaluation_identity_file_sha256=hashlib.sha256(
            identity_path.read_bytes()
        ).hexdigest(),
        output_root=tmp_path / "scoring",
    )

    assert report["sample_count"] == 1
    assert report["sample_local_failure_count"] == 1
    assert report["correct_count"] == 0
    assert report["micro_accuracy"] == 0.0
    scored = json.loads(
        (tmp_path / "scoring/scored-results.jsonl").read_text(encoding="utf-8")
    )
    assert scored["prediction"] == ""
    assert scored["correct"] is False
    assert scored["result_kind"] == "sample_local_failure"


def test_worker_persists_policy_output_contract_failure_and_continues_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Evaluator:
        def __init__(self, **_kwargs) -> None:
            pass

        async def evaluate(self, task: CoreDevTask):
            if task.ordinal == 0:
                raise _policy_output_error()
            await asyncio.sleep(0)
            return SimpleNamespace(tool_calls=(), stop="direct_answer")

    args, config, engine = _patch_worker_runtime(
        monkeypatch,
        tmp_path,
        tasks=(_task(0), _task(1)),
        evaluator_type=Evaluator,
    )
    identity = _evaluation_identity(task_count=2)
    monkeypatch.setattr(
        _MODULE, "write_policy_evaluation_identity", lambda *_args: identity
    )

    assert asyncio.run(_MODULE._worker(args, config)) == 0

    rows = [
        json.loads(line)
        for line in (tmp_path / "inference/rank-0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["ordinal"] for row in rows} == {0, 1}
    failure = next(row for row in rows if row["ordinal"] == 0)
    assert failure["result_kind"] == "sample_local_failure"
    assert failure["trajectory_available"] is False
    assert failure["final_answer"] is None
    assert next(row for row in rows if row["ordinal"] == 1)["result_kind"] == (
        "trajectory"
    )
    assert engine.shutdown_called is True


def test_worker_does_not_downgrade_generic_replay_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Evaluator:
        def __init__(self, **_kwargs) -> None:
            pass

        async def evaluate(self, _task: CoreDevTask):
            raise ReplayMismatchError("trajectory source identity changed")

    args, config, engine = _patch_worker_runtime(
        monkeypatch,
        tmp_path,
        tasks=(_task(0),),
        evaluator_type=Evaluator,
    )
    with pytest.raises(ReplayMismatchError, match="trajectory source identity"):
        asyncio.run(_MODULE._worker(args, config))

    assert not (tmp_path / "inference/rank-0.jsonl").exists()
    assert engine.shutdown_called is True
