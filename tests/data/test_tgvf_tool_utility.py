from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.tgvf_tool_utility import (
    TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
    TGVFToolUtilityError,
    load_tgvf_tool_utility_runtime_binding,
    materialize_indexed_tgvf_tool_utility_schedule,
    materialize_tgvf_tool_utility_schedule,
    materialize_tgvf_tool_utility_sidecar,
)


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _shuffle_key(sample_id: str, seed: int = 42) -> tuple[str, str]:
    payload = f"sha256-sort-v1\0{seed}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest(), sample_id


def _dataset(tmp_path: Path, full_correct: list[int]) -> tuple[Path, list[str]]:
    root = tmp_path / "mixed-v2"
    root.mkdir()
    sample_ids = sorted(
        (f"sample:{index}" for index in range(len(full_correct))),
        key=_shuffle_key,
    )
    correct_by_id = {
        f"sample:{index}": correct for index, correct in enumerate(full_correct)
    }
    rows = []
    for sample_id in sample_ids:
        correct = correct_by_id[sample_id]
        rows.append(
            {
                "schema_version": "tgvf.policy-t1-mixed-rl.sample.v2",
                "sample_id": sample_id,
                "candidate_sha256": _sha256(sample_id.encode()),
                "decision_sha256": _sha256(f"decision:{sample_id}".encode()),
                "image": {
                    "path": str((tmp_path / f"{sample_id}.png").resolve()),
                    "sha256": _sha256(f"image:{sample_id}".encode()),
                    "width": 10,
                    "height": 10,
                },
                "extra_info": {"question": f"question for {sample_id}"},
                "reward_model": {"ground_truth": f"answer for {sample_id}"},
                "data_source": "vstar",
                "task_kind": "open",
                "selection": {
                    "decision_stage": "final",
                    "t1": {
                        "decision": "retain",
                        "full_image": {
                            "accuracy": correct / 8,
                            "complete": True,
                            "correct_count": correct,
                            "expected_attempts": 8,
                            "missing_indices": [],
                            "observed_attempts": 8,
                            "scoreable_attempts": 8,
                            "status_counts": {"scored": 8},
                        },
                        "reason": "between_one_and_seven_of_eight_correct",
                    },
                },
            }
        )
    samples_payload = b"".join(_canonical_line(row) for row in rows)
    (root / "samples.jsonl").write_bytes(samples_payload)
    manifest = {
        "schema_version": "tgvf.policy-t1-mixed-rl.manifest.v2",
        "dataset_kind": "policy_t1_retained_mixed",
        "content_sha256": "c" * 64,
        "retained_count": len(rows),
        "shuffle": {"algorithm": "sha256-sort-v1", "seed": 42},
        "samples": {
            "path": "samples.jsonl",
            "rows": len(rows),
            "sha256": _sha256(samples_payload),
        },
    }
    (root / "manifest.json").write_bytes(_canonical_line(manifest))
    return root, sample_ids


def _attempts(
    path: Path,
    sample_ids: list[str],
    correct_counts: list[int],
    *,
    attempts_per_sample: int = 4,
) -> None:
    records = []
    for sample_id, correct_count in zip(sample_ids, correct_counts, strict=True):
        for attempt_index in range(attempts_per_sample):
            records.append(
                {
                    "schema_version": TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
                    "run_id": "forced-tgvf-fixture",
                    "run_identity_sha256": "a" * 64,
                    "sample_id": sample_id,
                    "attempt_index": attempt_index,
                    "status": "scored",
                    "correct": attempt_index < correct_count,
                }
            )
    path.write_bytes(b"".join(_canonical_line(record) for record in records))


def test_schedule_is_exact_sequential_training_prefix(tmp_path: Path) -> None:
    dataset, ordered_ids = _dataset(tmp_path, [1, 2, 3, 4, 5, 6])
    result = materialize_tgvf_tool_utility_schedule(
        dataset,
        tmp_path / "schedule",
        global_prompt_batch_size=2,
        optimizer_steps=2,
        canary_sample_count=2,
    )

    rows = [json.loads(line) for line in result.schedule_path.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ordered_ids[:4]
    assert [row["training_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["optimizer_step"] for row in rows] == [1, 1, 2, 2]
    assert [row["prompt_index_in_step"] for row in rows] == [0, 1, 0, 1]
    assert [row["is_canary"] for row in rows] == [True, True, False, False]
    assert result.sample_count == 4
    assert result.canary_sample_count == 2

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schedule"] == {
        "selection": "sequential-prefix-v1",
        "global_prompt_batch_size": 2,
        "optimizer_steps": 2,
        "sample_count": 4,
        "canary_sample_count": 2,
        "canary_optimizer_steps": 1,
    }
    assert manifest["files"]["schedule"]["sha256"] == _sha256(
        result.schedule_path.read_bytes()
    )


def test_indexed_schedule_preserves_external_training_order(tmp_path: Path) -> None:
    dataset, ordered_ids = _dataset(tmp_path, [1, 2, 3, 4])
    external_order = [ordered_ids[2], ordered_ids[0], ordered_ids[3], ordered_ids[1]]
    result = materialize_indexed_tgvf_tool_utility_schedule(
        dataset,
        tmp_path / "indexed-schedule",
        external_order,
        source_selection="deepeyes-stratified-prefix-v1",
        source_schedule_identity_sha256="d" * 64,
        global_prompt_batch_size=2,
        optimizer_steps=2,
        canary_sample_count=2,
    )

    rows = [json.loads(line) for line in result.schedule_path.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == external_order
    assert [row["training_index"] for row in rows] == [0, 1, 2, 3]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schedule"]["selection"] == "deepeyes-stratified-prefix-v1"
    assert manifest["schedule"]["source_schedule_identity_sha256"] == "d" * 64


def test_sidecar_labels_real_complete_counterfactual_attempts(tmp_path: Path) -> None:
    dataset, ordered_ids = _dataset(tmp_path, [2, 4, 6])
    schedule = materialize_tgvf_tool_utility_schedule(
        dataset,
        tmp_path / "schedule",
        global_prompt_batch_size=1,
        optimizer_steps=3,
        canary_sample_count=1,
    )
    p_full_by_id = {
        f"sample:{index}": correct / 8 for index, correct in enumerate([2, 4, 6])
    }
    desired_p_tgvf = [
        p_full_by_id[sample_id] + delta
        for sample_id, delta in zip(ordered_ids, (0.25, 0.0, -0.25), strict=True)
    ]
    correct_counts = [int(value * 4) for value in desired_p_tgvf]
    attempts_path = tmp_path / "attempts.jsonl"
    _attempts(attempts_path, ordered_ids, correct_counts)

    result = materialize_tgvf_tool_utility_sidecar(
        schedule.output_root,
        attempts_path,
        tmp_path / "sidecar",
        run_id="forced-tgvf-fixture",
        run_identity_sha256="a" * 64,
        attempts_per_sample=4,
    )

    rows = [json.loads(line) for line in result.sidecar_path.read_text().splitlines()]
    assert [row["utility_label"] for row in rows] == [
        "needed",
        "optional",
        "unnecessary",
    ]
    assert [row["delta"] for row in rows] == [0.25, 0.0, -0.25]
    assert all(row["confidence"] == 0.5 for row in rows)
    assert all(row["attempt_counts"]["full"]["expected"] == 8 for row in rows)
    assert all(row["attempt_counts"]["tgvf"]["expected"] == 4 for row in rows)
    assert result.label_counts == {"needed": 1, "optional": 1, "unnecessary": 1}


def test_sidecar_never_invents_missing_tgvf_probability(tmp_path: Path) -> None:
    dataset, ordered_ids = _dataset(tmp_path, [4])
    schedule = materialize_tgvf_tool_utility_schedule(
        dataset,
        tmp_path / "schedule",
        global_prompt_batch_size=1,
        optimizer_steps=1,
        canary_sample_count=1,
    )
    attempts_path = tmp_path / "attempts.jsonl"
    _attempts(attempts_path, ordered_ids, [2])
    records = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    attempts_path.write_bytes(b"".join(_canonical_line(row) for row in records[:-1]))

    with pytest.raises(TGVFToolUtilityError, match="lacks 4 scored"):
        materialize_tgvf_tool_utility_sidecar(
            schedule.output_root,
            attempts_path,
            tmp_path / "sidecar",
            run_id="forced-tgvf-fixture",
            run_identity_sha256="a" * 64,
            attempts_per_sample=4,
        )


def test_runtime_binding_proves_sidecar_manifest_dataset_and_label_rows(
    tmp_path: Path,
) -> None:
    dataset, ordered_ids = _dataset(tmp_path, [2, 4])
    schedule = materialize_tgvf_tool_utility_schedule(
        dataset,
        tmp_path / "schedule",
        global_prompt_batch_size=1,
        optimizer_steps=2,
        canary_sample_count=1,
    )
    attempts_path = tmp_path / "attempts.jsonl"
    _attempts(attempts_path, ordered_ids, [2, 2])
    sidecar = materialize_tgvf_tool_utility_sidecar(
        schedule.output_root,
        attempts_path,
        tmp_path / "sidecar",
        run_id="forced-tgvf-fixture",
        run_identity_sha256="a" * 64,
        attempts_per_sample=4,
    )
    schedule_manifest = json.loads(schedule.manifest_path.read_text())
    dataset_iteration = schedule_manifest["dataset"]["iteration_identity_sha256"]

    binding = load_tgvf_tool_utility_runtime_binding(
        sidecar.sidecar_path,
        expected_sidecar_sha256=sidecar.sidecar_sha256,
        manifest_path=sidecar.manifest_path,
        expected_manifest_sha256=sidecar.manifest_sha256,
        expected_dataset_iteration_identity_sha256=dataset_iteration,
    )

    assert binding.label_for_sample(ordered_ids[0]).confidence == 0.5
    assert binding.label_for_sample(ordered_ids[0]).row_sha256
    with pytest.raises(TGVFToolUtilityError, match="no label"):
        binding.label_for_sample("missing")
    with pytest.raises(TGVFToolUtilityError, match="dataset identity"):
        load_tgvf_tool_utility_runtime_binding(
            sidecar.sidecar_path,
            expected_sidecar_sha256=sidecar.sidecar_sha256,
            manifest_path=sidecar.manifest_path,
            expected_manifest_sha256=sidecar.manifest_sha256,
            expected_dataset_iteration_identity_sha256="f" * 64,
        )


def test_schedule_rejects_dataset_order_drift(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path, [2, 4])
    rows = [
        json.loads(line)
        for line in (dataset / "samples.jsonl").read_text().splitlines()
    ]
    payload = b"".join(_canonical_line(row) for row in reversed(rows))
    (dataset / "samples.jsonl").write_bytes(payload)
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["samples"]["sha256"] = _sha256(payload)
    (dataset / "manifest.json").write_bytes(_canonical_line(manifest))

    with pytest.raises(TGVFToolUtilityError, match="sequential hash order"):
        materialize_tgvf_tool_utility_schedule(
            dataset,
            tmp_path / "schedule",
            global_prompt_batch_size=1,
            optimizer_steps=2,
            canary_sample_count=1,
        )
