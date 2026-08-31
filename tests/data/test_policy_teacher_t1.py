from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tgvf_rl.data import policy_teacher_t1 as teacher
from tgvf_rl.data.policy_selection import (
    SelectionBranch,
    SelectionCandidate,
    SelectionSource,
    canonical_json_line,
    stable_selection_request_id,
)
from tgvf_rl.data.policy_selection_runtime import (
    VerificationOutcome,
    load_t1_run_config,
    verify_t1_answer,
)
from tgvf_rl.data.policy_selection_t1_scoring import _option_count
from tgvf_rl.data.policy_teacher_t1_retained import (
    _load_decisions,
    materialize_teacher_t1_retained,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_line(row) for row in rows))


def _row(
    *,
    uid: str,
    source_uid: str,
    image: Path,
    question: str,
    answer_format: str,
    answer: str,
    short_answer: str,
    choices: list[dict[str, str]] | None = None,
    focus_step_index: int = 1,
) -> dict[str, object]:
    return {
        "uid": uid,
        "source_uid": source_uid,
        "focus_step_index": focus_step_index,
        "image": str(image),
        "question": question,
        "answer_format": answer_format,
        "answer": answer,
        "short_answer": short_answer,
        "choices": choices or [],
        "source_dataset": "fixture",
        "stable_image_uid": f"image:{image.name}",
        "target": "must not enter the candidate",
        "evidence_description": "must not enter the candidate",
        "source_trace": [{"type": "focus", "focus_text": "secret"}],
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _retained_candidate(tmp_path: Path, index: int) -> dict[str, object]:
    image_path = (tmp_path / f"retained-{index}.png").resolve()
    Image.new("RGB", (8, 6), (index * 20, 40, 60)).save(image_path)
    if index == 0:
        question = "What color?\nChoices:\nA. red\nB. green"
        ground_truth = "B"
        metadata: dict[str, object] = {
            "task_kind": "mcq",
            "answer_format": "multiple_choice",
            "option_count": 2,
            "choices": ["A. red", "B. green"],
            "answer_text": "green",
        }
    else:
        question = "Read the word."
        ground_truth = "hello"
        metadata = {"task_kind": "open", "answer_format": "open"}
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": f"policy-teacher-candidate:{index}",
        "source": "teacher",
        "question": question,
        "ground_truth": ground_truth,
        "image": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "width": 8,
            "height": 6,
        },
        "gt_regions": [],
        "provenance": {
            "source_uid": f"teacher:{index}",
            "source_dataset": "fixture",
        },
        "selection_metadata": metadata,
    }


def _teacher_run_fixture(
    tmp_path: Path,
    *,
    mutate_candidates: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
    candidates = [_retained_candidate(tmp_path, 0), _retained_candidate(tmp_path, 1)]
    if mutate_candidates:
        candidates[1]["selection_metadata"] = {
            "task_kind": "open",
            "answer_format": "multiple_choice",
        }
    candidates_path = (tmp_path / "teacher-candidates.jsonl").resolve()
    _write_jsonl(candidates_path, candidates)
    candidates_sha256 = _sha256(candidates_path)
    selection_manifest = {
        "schema_version": "tgvf.policy-selection.teacher-t1-candidates-manifest.v1",
        "selection_algorithm_version": "teacher-train-source-uid-full-v1",
        "selection_is_outcome_independent": True,
        "candidates": {
            "path": str(candidates_path),
            "sha256": candidates_sha256,
            "rows": len(candidates),
        },
        "logical_attempts": len(candidates) * 8,
        "source_counts": {"teacher": len(candidates)},
    }
    selection_manifest_path = (tmp_path / "teacher-manifest.json").resolve()
    selection_manifest_path.write_bytes(canonical_json_line(selection_manifest))

    repository_root = Path(__file__).resolve().parents[2]
    template_path = (
        repository_root / "configs/policy/data_selection/"
        "qwen3_instruct_t1_512_vstar170k_arxiv32k_thinklite69842_v1.json"
    )
    config = json.loads(template_path.read_text(encoding="utf-8"))
    config["run_id"] = "T1-TEACHER-RETAINED-FIXTURE"
    config["data"] = {
        "sources": [
            {
                "source": "teacher",
                "path": str(candidates_path),
                "sha256": candidates_sha256,
                "rows": len(candidates),
            }
        ]
    }
    config["selection"] = {
        "kind": "teacher_full",
        "algorithm_version": "teacher-train-source-uid-full-v1",
        "candidates_path": str(candidates_path),
        "candidates_sha256": candidates_sha256,
        "rows": len(candidates),
        "manifest_path": str(selection_manifest_path),
        "manifest_sha256": _sha256(selection_manifest_path),
    }
    config["verifier"]["schema"] = "t1-source-verifier-v2"
    config["verifier"]["teacher_rule"] = (
        "mcq-bounded-label-else-normalized-exact-numeric-semantic-v1"
    )
    config["output_root"] = str((tmp_path / "unused-run-output").resolve())
    config_path = (tmp_path / "teacher-run.json").resolve()
    config_path.write_bytes(canonical_json_line(config))
    load_t1_run_config(config_path, verify_data_files=True)
    return config_path, candidates


def _decision_for(
    candidate_record: dict[str, object], correct_count: int
) -> dict[str, object]:
    candidate = SelectionCandidate.from_record(candidate_record)
    if correct_count == 0:
        decision = "exclude_too_hard"
        reason = "zero_of_eight_correct"
    elif correct_count == 8:
        decision = "exclude_too_easy"
        reason = "eight_of_eight_correct"
    else:
        decision = "retain"
        reason = "between_one_and_seven_of_eight_correct"
    return {
        "schema_version": "tgvf.policy-selection.decision.v1",
        "sample_id": candidate.sample_id,
        "candidate_sha256": candidate.identity_sha256,
        "source": "teacher",
        "t1": {
            "decision": decision,
            "reason": reason,
            "full_image": {
                "expected_attempts": 8,
                "observed_attempts": 8,
                "scoreable_attempts": 8,
                "correct_count": correct_count,
                "accuracy": correct_count / 8,
                "status_counts": {"scored": 8},
                "missing_indices": [],
                "complete": True,
            },
        },
        "t2": {
            "decision": "not_applicable_preserve_t1",
            "reason": "deepeyes_perception_utility_is_vstar_only",
            "gt_region": None,
        },
    }


def _write_final_bundle(
    tmp_path: Path,
    config_path: Path,
    candidates: list[dict[str, object]],
    *,
    correct_counts: list[int],
) -> Path:
    run = load_t1_run_config(config_path, verify_data_files=True)
    root = (tmp_path / "final-v2").resolve()
    root.mkdir()
    attempts: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for candidate_record, correct_count in zip(candidates, correct_counts, strict=True):
        candidate = SelectionCandidate.from_record(candidate_record)
        decisions.append(_decision_for(candidate_record, correct_count))
        for attempt_index in range(8):
            attempts.append(
                {
                    "schema_version": "tgvf.policy-selection.attempt.v1",
                    "request_id": stable_selection_request_id(
                        candidate_sha256=candidate.identity_sha256,
                        branch=SelectionBranch.FULL_IMAGE,
                        attempt_index=attempt_index,
                    ),
                    "sample_id": candidate.sample_id,
                    "candidate_sha256": candidate.identity_sha256,
                    "source": "teacher",
                    "branch": "full_image",
                    "attempt_index": attempt_index,
                    "status": "scored",
                    "correct": attempt_index < correct_count,
                    "run_id": run.run_id,
                    "run_manifest_sha256": run.manifest_sha256,
                    "raw_generation_sha256": _sha256_bytes(
                        f"{candidate.sample_id}:{attempt_index}".encode()
                    ),
                    "budget_revision": 0,
                    "answer": "fixture",
                    "verification_route": "fixture",
                }
            )
    attempts_path = root / "attempts.jsonl"
    decisions_path = root / "decisions.jsonl"
    report_path = root / "report.json"
    _write_jsonl(attempts_path, attempts)
    _write_jsonl(decisions_path, decisions)
    report_path.write_bytes(canonical_json_line({"fixture": True}))
    identity = {
        "schema_version": "tgvf.policy-selection.t1-final-scoring-manifest.v2",
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "scoring_manifest_sha256": "2" * 64,
        "judge_manifest_sha256": "3" * 64,
        "files": {
            "attempts": {
                "path": "attempts.jsonl",
                "rows": len(attempts),
                "sha256": _sha256(attempts_path),
            },
            "decisions": {
                "path": "decisions.jsonl",
                "rows": len(decisions),
                "sha256": _sha256(decisions_path),
            },
            "report": {"path": "report.json", "sha256": _sha256(report_path)},
        },
    }
    manifest = {
        **identity,
        "manifest_sha256": _sha256_bytes(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return manifest_path


def _rebind_final_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempts_path = manifest_path.parent / "attempts.jsonl"
    manifest["files"]["attempts"]["sha256"] = _sha256(attempts_path)
    identity = dict(manifest)
    identity.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha256_bytes(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    manifest_path.write_bytes(canonical_json_line(manifest))


def test_teacher_candidate_materialization_is_independent_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(teacher, "_REPO_ROOT", tmp_path)
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    heldout_image = tmp_path / "heldout.png"
    coredev_image = tmp_path / "coredev.png"
    for path, color in (
        (image_a, (255, 0, 0)),
        (image_b, (0, 255, 0)),
        (heldout_image, (0, 0, 255)),
        (coredev_image, (255, 255, 0)),
    ):
        Image.new("RGB", (8, 6), color).save(path)

    choices = [
        {"label": "A", "text": "red"},
        {"label": "B", "text": "green"},
    ]
    first = _row(
        uid="teacher:a::focus1",
        source_uid="teacher:a",
        image=image_a,
        question="What color?",
        answer_format="multiple_choice",
        answer="B. green",
        short_answer="green",
        choices=choices,
    )
    second_focus = dict(first)
    second_focus.update(uid="teacher:a::focus2", focus_step_index=2)
    train_rows = [
        first,
        second_focus,
        _row(
            uid="teacher:b::focus1",
            source_uid="teacher:b",
            image=image_b,
            question="Read the word.",
            answer_format="open",
            answer="hello",
            short_answer="hello",
        ),
        _row(
            uid="teacher:c::focus1",
            source_uid="teacher:c",
            image=heldout_image,
            question="Held out?",
            answer_format="open",
            answer="yes",
            short_answer="yes",
        ),
    ]
    test_rows = [
        _row(
            uid="teacher:test::focus1",
            source_uid="teacher:test",
            image=heldout_image,
            question="test",
            answer_format="open",
            answer="test",
            short_answer="test",
        )
    ]
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    coredev_path = tmp_path / "coredev.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(
        coredev_path,
        [{"dataset": "fixture", "image_paths": [str(coredev_image)]}],
    )
    result = teacher.materialize_teacher_t1_candidates(
        train_path,
        test_path,
        coredev_path,
        tmp_path / "output",
        expected_train_sha256=_sha256(train_path),
        expected_test_sha256=_sha256(test_path),
        expected_train_rows=4,
        expected_test_rows=1,
        expected_candidates=2,
    )
    assert result.unique_source_uids == 3
    assert result.candidate_rows == 2
    records = [
        json.loads(line)
        for line in (result.output_root / "candidates.jsonl").read_text().splitlines()
    ]
    mcq = next(
        row for row in records if row["selection_metadata"]["task_kind"] == "mcq"
    )
    assert mcq["question"].endswith("Choices:\nA. red\nB. green")
    assert mcq["ground_truth"] == "B"
    assert "target" not in mcq and "evidence_description" not in mcq
    assert Path(mcq["image"]["path"]).is_file()
    assert not Path(mcq["image"]["path"]).is_symlink()
    exclusions = [
        json.loads(line)
        for line in (result.output_root / "exclusions.jsonl").read_text().splitlines()
    ]
    assert exclusions == [
        {
            "image_sha256": _sha256(heldout_image),
            "reasons": ["teacher_test_exact_image_sha256"],
            "schema_version": teacher.TEACHER_T1_EXCLUSION_SCHEMA,
            "source_dataset": "fixture",
            "source_uid": "teacher:c",
        }
    ]


def test_teacher_verifier_routes_mcq_and_open_without_legacy_masquerade() -> None:
    mcq = verify_t1_answer(
        source=SelectionSource.TEACHER,
        candidate_answer="The answer is option B.",
        expected_answer="B",
        option_count=4,
    )
    assert mcq.outcome is VerificationOutcome.CORRECT
    assert mcq.route == "teacher_mcq_rule"

    exact = verify_t1_answer(
        source="teacher",
        candidate_answer=" 42 ",
        expected_answer="42",
    )
    assert exact.outcome is VerificationOutcome.CORRECT
    assert exact.route == "teacher_open_normalized_exact"

    semantic = verify_t1_answer(
        source="teacher",
        candidate_answer="a canine",
        expected_answer="a dog",
    )
    assert semantic.outcome is VerificationOutcome.SEMANTIC_REQUIRED
    assert semantic.route == "teacher_open_semantic_required"


def test_teacher_option_count_is_bound_to_candidate_task_kind() -> None:
    base = {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": "teacher:mcq",
        "source": "teacher",
        "question": "Question?\nChoices:\nA. one\nB. two",
        "ground_truth": "A",
        "image": {
            "path": "/tmp/image.png",
            "sha256": "a" * 64,
            "width": 8,
            "height": 6,
        },
        "gt_regions": [],
        "provenance": {"source_uid": "teacher:mcq"},
        "selection_metadata": {"task_kind": "mcq", "option_count": 2},
    }
    assert _option_count(SelectionCandidate.from_record(base)) == 2
    open_record = {
        **base,
        "sample_id": "teacher:open",
        "question": "Question?",
        "ground_truth": "answer",
        "selection_metadata": {"task_kind": "open"},
    }
    assert _option_count(SelectionCandidate.from_record(open_record)) is None


def test_teacher_retained_join_requires_complete_one_to_seven_of_eight(
    tmp_path: Path,
) -> None:
    candidate_record = {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": "teacher:retained",
        "source": "teacher",
        "question": "Question?",
        "ground_truth": "answer",
        "image": {
            "path": "/tmp/image.png",
            "sha256": "b" * 64,
            "width": 8,
            "height": 6,
        },
        "gt_regions": [],
        "provenance": {"source_uid": "teacher:retained"},
        "selection_metadata": {"task_kind": "open"},
    }
    candidate = SelectionCandidate.from_record(candidate_record)
    full_image = {
        "expected_attempts": 8,
        "observed_attempts": 8,
        "scoreable_attempts": 8,
        "correct_count": 3,
        "accuracy": 3 / 8,
        "status_counts": {"scored": 8},
        "missing_indices": [],
        "complete": True,
    }
    decision = {
        "schema_version": "tgvf.policy-selection.decision.v1",
        "sample_id": candidate.sample_id,
        "candidate_sha256": candidate.identity_sha256,
        "source": "teacher",
        "t1": {
            "decision": "retain",
            "reason": "between_one_and_seven_of_eight_correct",
            "full_image": full_image,
        },
        "t2": {
            "decision": "not_applicable_preserve_t1",
            "reason": "deepeyes_perception_utility_is_vstar_only",
            "gt_region": None,
        },
    }
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, [decision])
    parsed, counts = _load_decisions(
        decisions_path, candidates={candidate.sample_id: candidate}
    )
    assert parsed[candidate.sample_id]["t1"]["full_image"]["correct_count"] == 3
    assert counts == {"retain": 1}

    invalid = dict(decision)
    invalid["t1"] = {
        **decision["t1"],
        "full_image": {**full_image, "correct_count": 8, "accuracy": 1.0},
    }
    _write_jsonl(decisions_path, [invalid])
    try:
        _load_decisions(decisions_path, candidates={candidate.sample_id: candidate})
    except ValueError as error:
        assert "1--7/8" in str(error)
    else:
        raise AssertionError("8/8 teacher row must not enter the retained dataset")

    unexpected = {**decision, "unexpected": True}
    _write_jsonl(decisions_path, [unexpected])
    with pytest.raises(ValueError, match="schema differs"):
        _load_decisions(decisions_path, candidates={candidate.sample_id: candidate})

    decisions_path.write_text(
        json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _load_decisions(decisions_path, candidates={candidate.sample_id: candidate})


def test_teacher_retained_materializes_end_to_end_with_exact_coverage(
    tmp_path: Path,
) -> None:
    config_path, candidates = _teacher_run_fixture(tmp_path)
    final_manifest = _write_final_bundle(
        tmp_path,
        config_path,
        candidates,
        correct_counts=[3, 8],
    )
    result = materialize_teacher_t1_retained(
        config_path,
        final_manifest,
        tmp_path / "teacher-retained",
    )
    assert result.candidate_count == 2
    assert result.retained_count == 1
    sample = json.loads(
        (result.output_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    assert sample["data_source"] == "teacher"
    assert sample["reward_model"] == {"ground_truth": "B"}
    assert len(sample["decision_sha256"]) == 64
    assert sample["selection"]["t1"]["full_image"]["correct_count"] == 3

    manifest = json.loads(
        (result.output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_kind"] == "policy_teacher_t1_retained"
    assert manifest["candidate_count"] == manifest["decision_count"] == 2
    assert manifest["retained_count"] == 1
    assert manifest["t1_decision_counts"] == {
        "retain": 1,
        "exclude_too_hard": 0,
        "exclude_too_easy": 1,
        "unresolved": 0,
    }
    assert manifest["inputs"]["attempts"]["rows"] == 16
    assert manifest["inputs"]["attempts"]["coverage"] == (
        "each-candidate-full-image-attempt-indices-0-through-7"
    )
    assert manifest["inputs"]["decisions"]["rows"] == 2
    assert manifest["selection_policy"] == {
        "t1": "retain",
        "t2": "ignored",
        "post_t1_balancing": "none",
    }


def test_teacher_retained_rejects_rehashed_duplicate_attempt_coverage(
    tmp_path: Path,
) -> None:
    config_path, candidates = _teacher_run_fixture(tmp_path)
    final_manifest = _write_final_bundle(
        tmp_path,
        config_path,
        candidates,
        correct_counts=[3, 8],
    )
    attempts_path = final_manifest.parent / "attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    attempts[7] = dict(attempts[6])
    _write_jsonl(attempts_path, attempts)
    _rebind_final_manifest(final_manifest)

    with pytest.raises(ValueError, match="duplicate teacher T1 attempt"):
        materialize_teacher_t1_retained(
            config_path,
            final_manifest,
            tmp_path / "teacher-retained",
        )


def test_teacher_retained_rejects_empty_retained_population(tmp_path: Path) -> None:
    config_path, candidates = _teacher_run_fixture(tmp_path)
    final_manifest = _write_final_bundle(
        tmp_path,
        config_path,
        candidates,
        correct_counts=[0, 8],
    )
    with pytest.raises(ValueError, match="retained no rows"):
        materialize_teacher_t1_retained(
            config_path,
            final_manifest,
            tmp_path / "teacher-retained",
        )


def test_teacher_retained_rejects_invalid_candidate_metadata(tmp_path: Path) -> None:
    config_path, candidates = _teacher_run_fixture(tmp_path, mutate_candidates=True)
    final_manifest = _write_final_bundle(
        tmp_path,
        config_path,
        candidates,
        correct_counts=[3, 8],
    )
    with pytest.raises(ValueError, match="open answer_format differs"):
        materialize_teacher_t1_retained(
            config_path,
            final_manifest,
            tmp_path / "teacher-retained",
        )


def test_teacher_retained_rejects_noncanonical_final_manifest(tmp_path: Path) -> None:
    config_path, candidates = _teacher_run_fixture(tmp_path)
    final_manifest = _write_final_bundle(
        tmp_path,
        config_path,
        candidates,
        correct_counts=[3, 8],
    )
    value = json.loads(final_manifest.read_text(encoding="utf-8"))
    final_manifest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        materialize_teacher_t1_retained(
            config_path,
            final_manifest,
            tmp_path / "teacher-retained",
        )
