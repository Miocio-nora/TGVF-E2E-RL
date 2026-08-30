from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.evaluation import policy_benchmark_scoring as implementation
from tgvf_rl.evaluation.policy_benchmark_scoring import (
    infer_mcq_option,
    materialize_policy_benchmark_mcq_scoring,
)
from tgvf_rl.immutable_publication import ImmutablePublicationRaceError
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


SHA = "a" * 64


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_no_judge_mcq_extraction_is_deterministic() -> None:
    options = {"A": "North", "B": "South", "C": "East", "D": "West"}

    assert infer_mcq_option("After checking, the answer is B.", options) == (
        "B",
        "terminal_option_token",
    )
    assert infer_mcq_option("West", options) == ("D", "unique_option_text")
    assert infer_mcq_option("North or South", options) == (
        None,
        "ambiguous_or_unmatched",
    )


def test_immutable_scoring_writer_preserves_retry_and_collision_contract(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested/results.jsonl"

    implementation._write_immutable(destination, b"stable")  # noqa: SLF001
    implementation._write_immutable(destination, b"stable")  # noqa: SLF001

    with pytest.raises(RuntimeError, match="immutable scoring output differs"):
        implementation._write_immutable(destination, b"different")  # noqa: SLF001
    assert destination.read_bytes() == b"stable"


def test_immutable_scoring_writer_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"protected")
    destination = tmp_path / "results.jsonl"
    destination.symlink_to(target)

    with pytest.raises(RuntimeError, match="immutable scoring output differs"):
        implementation._write_immutable(destination, b"protected")  # noqa: SLF001

    assert destination.is_symlink()
    assert target.read_bytes() == b"protected"


def test_immutable_scoring_writer_translates_publication_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(_path: Path, _payload: bytes) -> None:
        raise ImmutablePublicationRaceError("unstable destination")

    monkeypatch.setattr(implementation, "publish_bytes_content_consistent", reject)

    destination = tmp_path / "results.jsonl"
    with pytest.raises(RuntimeError, match=f"immutable scoring output differs: {destination}"):
        implementation._write_immutable(destination, b"payload")  # noqa: SLF001


def test_generic_scorer_binds_identity_and_materializes_dataset_tsvs(
    tmp_path: Path,
) -> None:
    task_rows = [
        {
            "ordinal": ordinal,
            "dataset": dataset,
            "row_number": 0,
            "index": f"sample-{ordinal}",
            "sample_id": f"sample-{ordinal}",
            "question": "Question: Where?",
            "image_paths": [f"/immutable/image-{ordinal}.png"],
            "image_sha256s": [str(ordinal + 1) * 64],
            "image_dimensions": [[32, 24]],
            "answer": answer,
            "options": [
                ["A", "North"],
                ["B", "South"],
                ["C", "East"],
                ["D", "West"],
            ],
            "metadata": [["category", "single"]],
        }
        for ordinal, (dataset, answer) in enumerate(
            (("VStarBench", "A"), ("HRBench8K", "D"))
        )
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    tasks_sha256 = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    model = ModelIdentity(
        family="qwen3_vl",
        model_name="fixture",
        revision_or_path="fixture",
        tokenizer_length=1,
        chat_template_sha256=SHA,
    )
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": "DEEPEYES-STEP8",
        "evaluation_schema_version": POLICY_BENCHMARK_SCHEMA,
        "policy_config_path": "/immutable/policy.toml",
        "policy_config_file_sha256": SHA,
        "policy_run_config_identity_sha256": SHA,
        "model_identity": asdict(model),
        "policy_snapshot": {
            "run_id": "PRL-11",
            "run_identity_sha256": SHA,
            "optimizer_step": 8,
            "weights_sha256": SHA,
            "pointer_file_sha256": SHA,
            "manifest_file_sha256": SHA,
            "tensor_file_sha256": SHA,
            "request_sha256": SHA,
        },
        "task_manifest": {
            "path": str(tasks_path),
            "sha256": tasks_sha256,
            "task_count": 2,
            "single_image_count": 2,
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
    identity = {**content, "identity_sha256": _canonical_sha256(content)}
    identity_path = tmp_path / "evaluation-identity.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    identity_file_sha256 = hashlib.sha256(identity_path.read_bytes()).hexdigest()

    inference_root = tmp_path / "inference"
    inference_root.mkdir()
    for rank in range(4):
        (inference_root / f"rank-{rank}.jsonl").write_text("", encoding="utf-8")
    final_answers = ("The answer is A.", "I choose A.")
    for row, final_answer in zip(task_rows, final_answers, strict=True):
        task = CoreDevTask(**row)
        trajectory = TrajectoryRecord(
            schema_version="trajectory-v1",
            identity=TrajectoryIdentity(
                "DEEPEYES-STEP8",
                task.bound_sample_id,
                0,
                f"benchmark:{task.ordinal}",
            ),
            model=model,
            behavior_policy=PolicyVersion("PRL-11", 8, SHA),
            assistant_turns=(),
            tool_calls=(),
            observations=(),
            final_answer=final_answer,
            stop=TrajectoryStop.DIRECT_ANSWER,
        )
        rank = task.ordinal % 4
        payload = trajectory_audit_payload(
            task,
            trajectory,
            evaluation_identity=identity,
            rank=rank,
            world_size=4,
        )
        with (inference_root / f"rank-{rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    output_root = tmp_path / "scoring"
    report = materialize_policy_benchmark_mcq_scoring(
        inference_root=inference_root,
        tasks_path=tasks_path,
        tasks_sha256=tasks_sha256,
        evaluation_identity_path=identity_path,
        evaluation_identity_file_sha256=identity_file_sha256,
        output_root=output_root,
    )

    assert report["sample_count"] == 2
    assert report["correct_count"] == 1
    assert report["micro_accuracy"] == 0.5
    assert report["macro_dataset_accuracy"] == 0.5
    assert report["policy_snapshot"]["optimizer_step"] == 8
    assert (output_root / "datasets/VStarBench.tsv").is_file()
    assert (output_root / "datasets/HRBench8K.tsv").is_file()
    assert len((output_root / "scored-results.jsonl").read_text().splitlines()) == 2
