from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.data.policy_selection import (
    SelectionCandidate,
    canonical_json_line,
)
from tgvf_rl.data.policy_t1_mixed_rl_dataset import (
    POLICY_T1_MIXED_DATASET_KIND,
    POLICY_T1_MIXED_SAMPLE_SCHEMA,
    PolicyT1MixedMaterializationError,
    PolicyT1MixedRuntimeBinding,
    PolicyT1MixedRuntimeValidationError,
    load_policy_t1_mixed_runtime,
    materialize_policy_t1_mixed_retained_pool,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _candidate(tmp_path: Path, source: str, index: int) -> dict[str, object]:
    image_path = (tmp_path / f"{source}-{index}.bin").resolve()
    image_path.write_bytes(f"{source}-image-{index}".encode())
    questions = {
        "vstar": "What color is the marked object?",
        "arxivqa": "Which answer is correct?\nChoices:\nA. no\nB. yes",
        "thinklite": "What is 6 times 7?",
    }
    answers = {"vstar": "blue", "arxivqa": "B", "thinklite": "42"}
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": f"candidate:{source}:{index}",
        "source": source,
        "question": questions[source],
        "ground_truth": answers[source],
        "image": {
            "path": str(image_path),
            "sha256": _sha256(image_path.read_bytes()),
            "width": 8,
            "height": 8,
        },
        "gt_regions": [[0, 0, 1, 1]] if source == "vstar" else [],
        "provenance": {"fixture": True, "index": index},
        "selection_metadata": {"option_count": 2} if source == "arxivqa" else {},
    }


def _decision(
    candidate: dict[str, object],
    decision: str,
    *,
    t2: object,
) -> dict[str, object]:
    parsed = SelectionCandidate.from_record(candidate)
    correct_counts = {
        "retain": 4,
        "exclude_too_easy": 8,
        "exclude_too_hard": 0,
    }
    correct = correct_counts.get(decision, 0)
    complete = decision != "unresolved"
    return {
        "schema_version": "tgvf.policy-selection.decision.v1",
        "sample_id": parsed.sample_id,
        "candidate_sha256": parsed.identity_sha256,
        "source": parsed.source.value,
        "t1": {
            "decision": decision,
            "full_image": {
                "accuracy": correct / 8 if complete else None,
                "complete": complete,
                "correct_count": correct,
                "expected_attempts": 8,
                "missing_indices": [] if complete else [7],
                "observed_attempts": 8 if complete else 7,
                "scoreable_attempts": 8 if complete else 7,
                "status_counts": {"scored": 8}
                if complete
                else {"scored": 7, "truncated": 1},
            },
            "reason": "between_one_and_seven_of_eight_correct"
            if decision == "retain"
            else "fixture_nonretain",
        },
        "t2": t2,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_line(record) for record in records))


def _write_final_bundle(
    root: Path, decisions: list[dict[str, object]]
) -> tuple[Path, Path]:
    root.mkdir()
    decisions_path = root / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)
    identity = {
        "schema_version": "tgvf.policy-selection.t1-final-scoring-manifest.v1",
        "run_id": "T1-fixture",
        "run_manifest_sha256": "1" * 64,
        "scoring_manifest_sha256": "2" * 64,
        "judge_manifest_sha256": "3" * 64,
        "files": {
            "attempts": {
                "path": "attempts.jsonl",
                "rows": len(decisions) * 8,
                "sha256": "4" * 64,
            },
            "decisions": {
                "path": "decisions.jsonl",
                "rows": len(decisions),
                "sha256": _sha256(decisions_path.read_bytes()),
            },
            "report": {"path": "report.json", "sha256": "5" * 64},
        },
    }
    manifest = {**identity, "manifest_sha256": _sha256(_canonical_json_bytes(identity))}
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    return manifest_path, decisions_path


def _three_source_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, int]]:
    candidates = [
        _candidate(tmp_path, "vstar", 0),
        _candidate(tmp_path, "arxivqa", 1),
        _candidate(tmp_path, "thinklite", 2),
        _candidate(tmp_path, "vstar", 3),
    ]
    decisions = [
        _decision(candidates[0], "retain", t2={"decision": "unresolved"}),
        _decision(candidates[1], "retain", t2=None),
        _decision(candidates[2], "retain", t2={"ignored": True}),
        _decision(
            candidates[3],
            "exclude_too_easy",
            t2={"decision": "unresolved", "must_not_affect_membership": True},
        ),
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates_path, candidates)
    manifest_path, _ = _write_final_bundle(tmp_path / "final-v1", decisions)
    return (
        candidates_path,
        manifest_path,
        {
            "vstar": 2,
            "arxivqa": 1,
            "thinklite": 1,
        },
    )


def test_materializes_all_final_t1_retains_without_t2_or_balancing(
    tmp_path: Path,
) -> None:
    candidates_path, final_manifest_path, expected = _three_source_fixture(tmp_path)
    result = materialize_policy_t1_mixed_retained_pool(
        candidates_path,
        final_manifest_path,
        tmp_path / "artifact",
        expected_source_counts=expected,
    )

    assert result.sample_count == 3
    records = [
        json.loads(line)
        for line in (result.output_root / "samples.jsonl").read_text().splitlines()
    ]
    by_source = {record["data_source"]: record for record in records}
    assert {source: row["task_kind"] for source, row in by_source.items()} == {
        "vstar": "open",
        "arxivqa": "mcq",
        "thinklite": "math",
    }
    for row in records:
        assert row["schema_version"] == POLICY_T1_MIXED_SAMPLE_SCHEMA
        assert set(row) == {
            "schema_version",
            "sample_id",
            "candidate_sha256",
            "decision_sha256",
            "image",
            "extra_info",
            "reward_model",
            "data_source",
            "task_kind",
            "selection",
        }
        assert set(row["selection"]) == {"decision_stage", "t1"}
        assert row["selection"]["decision_stage"] == "final"
        assert row["selection"]["t1"]["decision"] == "retain"

    manifest = json.loads((result.output_root / "manifest.json").read_bytes())
    assert manifest["dataset_kind"] == POLICY_T1_MIXED_DATASET_KIND
    assert manifest["selection_policy"] == {
        "t1": "retain",
        "t2": "ignored",
        "post_t1_balancing": "none",
    }
    assert manifest["retained_count"] == 3
    assert manifest["candidate_count"] == manifest["decision_count"] == 4
    assert {
        source: report["retained_count"]
        for source, report in manifest["sources"].items()
    } == {"vstar": 1, "arxivqa": 1, "thinklite": 1}
    assert all(
        report["retained_share"] == pytest.approx(1 / 3)
        for report in manifest["sources"].values()
    )
    assert manifest["inputs"]["final_scoring_manifest"]["manifest_sha256"]
    assert manifest["inputs"]["final_scoring_manifest"]["file_sha256"]
    assert manifest["inputs"]["decisions"]["rows"] == 4

    runtime = load_policy_t1_mixed_runtime(
        result.output_root,
        binding=PolicyT1MixedRuntimeBinding(
            manifest_file_sha256=result.manifest_file_sha256,
            content_sha256=result.content_sha256,
            shuffle_seed=result.shuffle_seed,
            expected_sample_count=result.sample_count,
        ),
    )
    assert runtime.samples_sha256 == result.samples_sha256
    assert runtime.iteration_identity_sha256 == result.iteration_identity_sha256
    assert {
        sample.data_source: sample.task_kind.value for sample in runtime.samples
    } == {"vstar": "open", "arxivqa": "mcq", "thinklite": "math"}
    assert all(sample.image_path.is_absolute() for sample in runtime.samples)


def test_runtime_rejects_changed_absolute_image_bytes(tmp_path: Path) -> None:
    candidates_path, final_manifest_path, expected = _three_source_fixture(tmp_path)
    result = materialize_policy_t1_mixed_retained_pool(
        candidates_path,
        final_manifest_path,
        tmp_path / "artifact",
        expected_source_counts=expected,
    )
    first = json.loads(
        (result.output_root / "samples.jsonl").read_text().splitlines()[0]
    )
    Path(first["image"]["path"]).write_bytes(b"changed after materialization")

    with pytest.raises(
        PolicyT1MixedRuntimeValidationError,
        match="source image SHA-256 differs",
    ):
        load_policy_t1_mixed_runtime(
            result.output_root,
            binding=PolicyT1MixedRuntimeBinding(
                manifest_file_sha256=result.manifest_file_sha256,
                content_sha256=result.content_sha256,
                shuffle_seed=result.shuffle_seed,
                expected_sample_count=result.sample_count,
            ),
        )


def test_runtime_fails_closed_on_samples_file_hash_drift(tmp_path: Path) -> None:
    candidates_path, final_manifest_path, expected = _three_source_fixture(tmp_path)
    result = materialize_policy_t1_mixed_retained_pool(
        candidates_path,
        final_manifest_path,
        tmp_path / "artifact",
        expected_source_counts=expected,
    )
    samples_path = result.output_root / "samples.jsonl"
    samples_path.write_bytes(samples_path.read_bytes() + b"\n")

    with pytest.raises(
        PolicyT1MixedRuntimeValidationError,
        match="samples file hash differs",
    ):
        load_policy_t1_mixed_runtime(
            result.output_root,
            binding=PolicyT1MixedRuntimeBinding(
                manifest_file_sha256=result.manifest_file_sha256,
                content_sha256=result.content_sha256,
                shuffle_seed=result.shuffle_seed,
                expected_sample_count=result.sample_count,
            ),
        )


def test_rejects_final_decisions_hash_drift(tmp_path: Path) -> None:
    candidates_path, final_manifest_path, expected = _three_source_fixture(tmp_path)
    decisions_path = final_manifest_path.parent / "decisions.jsonl"
    decisions_path.write_bytes(decisions_path.read_bytes() + b"\n")

    with pytest.raises(
        PolicyT1MixedMaterializationError,
        match="decisions file SHA-256 differs",
    ):
        materialize_policy_t1_mixed_retained_pool(
            candidates_path,
            final_manifest_path,
            tmp_path / "artifact",
            expected_source_counts=expected,
        )


def test_rejects_incomplete_final_decision_population(tmp_path: Path) -> None:
    candidates = [
        _candidate(tmp_path, "vstar", 0),
        _candidate(tmp_path, "arxivqa", 1),
        _candidate(tmp_path, "thinklite", 2),
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates_path, candidates)
    manifest_path, _ = _write_final_bundle(
        tmp_path / "final-v1",
        [
            _decision(candidates[0], "retain", t2=None),
            _decision(candidates[1], "retain", t2=None),
        ],
    )

    with pytest.raises(
        PolicyT1MixedMaterializationError,
        match="decision row count differs",
    ):
        materialize_policy_t1_mixed_retained_pool(
            candidates_path,
            manifest_path,
            tmp_path / "artifact",
            expected_source_counts={"vstar": 1, "arxivqa": 1, "thinklite": 1},
        )


def test_cli_materializes_fixture(tmp_path: Path) -> None:
    candidates_path, final_manifest_path, _ = _three_source_fixture(tmp_path)
    output_root = tmp_path / "cli-artifact"
    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools/materialize_policy_t1_mixed_retained_pool.py"),
            "--candidates",
            str(candidates_path),
            "--final-manifest",
            str(final_manifest_path),
            "--output-root",
            str(output_root),
            "--expected-vstar-count",
            "2",
            "--expected-arxivqa-count",
            "1",
            "--expected-thinklite-count",
            "1",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    assert output["dataset_kind"] == POLICY_T1_MIXED_DATASET_KIND
    assert output["sample_count"] == 3
    assert output_root.is_dir()
