from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.data import (
    POLICY_SELECTION_ATTEMPT_SCHEMA,
    POLICY_SELECTION_CANDIDATE_SCHEMA,
    AttemptStatus,
    DeepEyesTaskKind,
    SelectionBranch,
    SelectionSource,
    T1Decision,
    T2Decision,
    build_selection_requests,
    classify_policy_selection_task_kind,
    policy_selection_semantic_judge_task_kind,
    records_sha256,
    reduce_selection_attempts,
    summarize_selection_decisions,
)


@pytest.mark.parametrize(
    ("source", "question", "ground_truth", "expected", "judge_route"),
    [
        (
            SelectionSource.VSTAR,
            r"Compute \\frac{1}{2}+\\frac{1}{2}.",
            "1",
            DeepEyesTaskKind.OPEN,
            "open_vqa",
        ),
        (
            SelectionSource.ARXIVQA,
            "Which answer is correct?",
            "B",
            DeepEyesTaskKind.MCQ,
            "open_vqa",
        ),
        (
            SelectionSource.THINKLITE,
            "What establishment is serving this food?",
            "food truck",
            DeepEyesTaskKind.OPEN,
            "open_vqa",
        ),
        (
            SelectionSource.THINKLITE,
            r"Find \\angle ABC in the diagram.",
            "45 degrees",
            DeepEyesTaskKind.MATH,
            "math",
        ),
        (
            SelectionSource.THINKLITE,
            "Adriana wants to buy 3 pounds of silver confetti. How much?",
            "36",
            DeepEyesTaskKind.MATH,
            "math",
        ),
        (
            SelectionSource.THINKLITE,
            "What fraction of the fruit were plums?",
            "36/89",
            DeepEyesTaskKind.MATH,
            "math",
        ),
        (
            SelectionSource.THINKLITE,
            "Find the requested ratio.",
            r"\\frac { 4 } { 5 }",
            DeepEyesTaskKind.MATH,
            "math",
        ),
        (
            SelectionSource.THINKLITE,
            "How long does the trip take?",
            "45 minutes",
            DeepEyesTaskKind.MATH,
            "math",
        ),
        (
            SelectionSource.THINKLITE,
            "Choose one:\n(A) cat\n(B) dog",
            "B",
            DeepEyesTaskKind.MCQ,
            "open_vqa",
        ),
        (
            SelectionSource.THINKLITE,
            "Choose one:\n(A) cat\n(B) dog",
            "dog",
            DeepEyesTaskKind.OPEN,
            "open_vqa",
        ),
    ],
)
def test_policy_selection_task_kind_is_sample_specific_for_thinklite(
    source: SelectionSource,
    question: str,
    ground_truth: str,
    expected: DeepEyesTaskKind,
    judge_route: str,
) -> None:
    assert (
        classify_policy_selection_task_kind(
            source=source,
            question=question,
            ground_truth=ground_truth,
        )
        is expected
    )
    assert (
        policy_selection_semantic_judge_task_kind(
            source=source,
            question=question,
            ground_truth=ground_truth,
        )
        == judge_route
    )


def _candidate(
    sample_id: str, source: str, *, regions: bool = True
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
        "sample_id": sample_id,
        "source": source,
        "question": f"Question for {sample_id}?",
        "ground_truth": "answer",
        "image": {
            "path": f"images/{sample_id}.png",
            "sha256": (sample_id[0] if sample_id[0] in "abcdef" else "a") * 64,
            "width": 1000,
            "height": 800,
        },
        "provenance": {"dataset": source, "row_id": sample_id},
    }
    if regions:
        record["gt_regions"] = [[10, 20, 110, 220]]
    return record


def _attempts(
    requests: Iterable[Mapping[str, object]],
    *,
    full_correct: int,
    oracle_correct: int = 0,
    full_status_override: Mapping[int, AttemptStatus] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    full_status_override = full_status_override or {}
    for request in requests:
        branch = str(request["branch"])
        attempt_index = int(request["attempt_index"])
        if branch == SelectionBranch.FULL_IMAGE.value:
            status = full_status_override.get(attempt_index, AttemptStatus.SCORED)
            correct = (
                attempt_index < full_correct if status is AttemptStatus.SCORED else None
            )
        else:
            status = AttemptStatus.SCORED
            correct = attempt_index < oracle_correct
        records.append(
            {
                "schema_version": POLICY_SELECTION_ATTEMPT_SCHEMA,
                "request_id": request["request_id"],
                "sample_id": request["sample_id"],
                "source": request["source"],
                "branch": branch,
                "attempt_index": attempt_index,
                "status": status.value,
                "correct": correct,
            }
        )
    return records


@pytest.mark.parametrize(
    ("correct_count", "expected"),
    [
        (0, T1Decision.EXCLUDE_TOO_HARD),
        (1, T1Decision.RETAIN),
        (4, T1Decision.RETAIN),
        (7, T1Decision.RETAIN),
        (8, T1Decision.EXCLUDE_TOO_EASY),
    ],
)
def test_t1_matches_the_deepeyes_eight_attempt_truth_table(
    correct_count: int, expected: T1Decision
) -> None:
    candidate = _candidate("a-sample", "vstar")
    requests = build_selection_requests([candidate])
    decisions = reduce_selection_attempts(
        [candidate], _attempts(requests, full_correct=correct_count)
    )

    assert decisions[0]["t1"]["decision"] == expected.value
    assert decisions[0]["t1"]["full_image"]["correct_count"] == correct_count


def test_truncation_and_missing_attempt_are_unresolved_not_incorrect() -> None:
    candidate = _candidate("a-sample", "vstar")
    requests = build_selection_requests([candidate])
    truncated = _attempts(
        requests,
        full_correct=3,
        full_status_override={7: AttemptStatus.TRUNCATED},
    )
    truncated_decision = reduce_selection_attempts([candidate], truncated)[0]
    missing_decision = reduce_selection_attempts([candidate], truncated[:-1])[0]

    assert truncated_decision["t1"]["decision"] == T1Decision.UNRESOLVED.value
    assert truncated_decision["t1"]["full_image"]["scoreable_attempts"] == 7
    assert missing_decision["t1"]["decision"] == T1Decision.UNRESOLVED.value
    assert missing_decision["t1"]["full_image"]["missing_indices"] == [7]


def test_request_generation_is_deterministic_and_keeps_answers_out_of_model_input() -> (
    None
):
    candidates = [
        _candidate("b-chart", "arxivqa"),
        _candidate("a-vstar", "vstar"),
    ]
    first = build_selection_requests(candidates, oracle_attempts=3)
    second = build_selection_requests(reversed(candidates), oracle_attempts=3)

    assert first == second
    assert len(first) == 8 + 8 + 3
    assert records_sha256(first) == records_sha256(second)
    assert len({record["request_id"] for record in first}) == len(first)
    assert all("ground_truth" not in record["model_input"] for record in first)
    assert sum(record["branch"] == "gt_region" for record in first) == 3


def test_vstar_t2_preserves_oracle_counts_but_refuses_to_guess_membership() -> None:
    candidate = _candidate("a-vstar", "vstar")
    requests = build_selection_requests([candidate], oracle_attempts=4)
    attempts = _attempts(requests, full_correct=3, oracle_correct=4)
    decision = reduce_selection_attempts(
        [candidate], attempts, expected_oracle_attempts=4
    )[0]

    assert decision["t1"]["decision"] == T1Decision.RETAIN.value
    assert decision["t2"]["decision"] == T2Decision.UNRESOLVED.value
    assert decision["t2"]["reason"] == "perception_utility_membership_rule_not_accepted"
    assert decision["t2"]["gt_region"]["correct_count"] == 4


def test_non_vstar_t2_preserves_t1_and_distribution_is_only_reported() -> None:
    candidates = [
        _candidate("a-vstar", "vstar"),
        _candidate("b-chart", "arxivqa"),
        _candidate("c-reasoning", "thinklite"),
    ]
    attempts: list[dict[str, object]] = []
    for candidate in candidates:
        requests = build_selection_requests([candidate])
        attempts.extend(_attempts(requests, full_correct=4))
    decisions = reduce_selection_attempts(candidates, attempts)
    by_source = {decision["source"]: decision for decision in decisions}
    summary = summarize_selection_decisions(decisions)

    assert by_source["arxivqa"]["t2"]["decision"] == (
        T2Decision.NOT_APPLICABLE_PRESERVE_T1.value
    )
    assert by_source["thinklite"]["t2"]["decision"] == (
        T2Decision.NOT_APPLICABLE_PRESERVE_T1.value
    )
    assert summary["t1_retained_total"] == 3
    assert summary["distribution_tolerance"] is None
    assert summary["distribution_membership_enforced"] is False
    assert summary["sources"]["vstar"]["retained_share"] == pytest.approx(1 / 3)


def test_invalid_gt_region_and_duplicate_attempt_fail_closed() -> None:
    invalid = _candidate("a-vstar", "vstar")
    invalid["gt_regions"] = [[0, 0, 1001, 100]]
    with pytest.raises(ValueError, match="source-pixel box"):
        build_selection_requests([invalid], oracle_attempts=1)

    candidate = _candidate("a-vstar", "vstar")
    requests = build_selection_requests([candidate])
    attempts = _attempts(requests, full_correct=4)
    with pytest.raises(ValueError, match="duplicate request_id"):
        reduce_selection_attempts([candidate], [*attempts, attempts[0]])

    attempts[0]["request_id"] = "qwen3-selection:" + "0" * 64
    with pytest.raises(ValueError, match="request_id identity mismatch"):
        reduce_selection_attempts([candidate], attempts)


def test_cpu_cli_build_and_reduce_dry_run(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    candidates_path = tmp_path / "candidates.jsonl"
    requests_path = tmp_path / "requests.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    summary_path = tmp_path / "summary.json"
    candidate = _candidate("a-vstar", "vstar")
    candidates_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "tools/prepare_policy_data_selection.py",
            "build-requests",
            "--candidates",
            str(candidates_path),
            "--output",
            str(requests_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    attempts = _attempts(requests, full_correct=4)
    attempts_path.write_text(
        "".join(json.dumps(record) + "\n" for record in attempts),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "tools/prepare_policy_data_selection.py",
            "reduce",
            "--candidates",
            str(candidates_path),
            "--attempts",
            str(attempts_path),
            "--output",
            str(decisions_path),
            "--summary-output",
            str(summary_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    decision = json.loads(decisions_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert decision["t1"]["decision"] == T1Decision.RETAIN.value
    assert summary["t1_retained_total"] == 1
