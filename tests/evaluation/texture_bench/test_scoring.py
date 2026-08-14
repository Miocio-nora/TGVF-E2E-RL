from __future__ import annotations

import pytest

from tgvf_rl.evaluation.texture_bench.scoring import (
    LAST_DATASET,
    MMAD_DATASET,
    MMAD_PARSER_UPSTREAM_LIKE_PERMISSIVE,
    parse_strict_choice,
    score_texture_benchmark,
)
from tgvf_rl.evaluation.texture_bench.task import TextureTask


TASK_MANIFEST_SHA256 = "0" * 64


def _task(
    ordinal: int,
    *,
    dataset: str,
    answer: str,
    metadata: dict[str, str],
    labels: str = "ABCD",
) -> TextureTask:
    sample_id = f"{dataset}:{ordinal}"
    return TextureTask(
        ordinal=ordinal,
        dataset=dataset,
        row_number=ordinal,
        index=sample_id,
        sample_id=sample_id,
        question="question",
        image_paths=(f"/image/{ordinal}.png",),
        answer=answer,
        options=tuple((label, f"option {label}") for label in labels),
        metadata=tuple(metadata.items()),
        image_sha256s=("0" * 64,),
        image_dimensions=((1, 1),),
    )


def test_strict_parser_accepts_decisive_answer_and_rejects_conflicts() -> None:
    allowed = ("B", "C", "D")
    for raw, expected in (
        ("B", "B"),
        ("(c).", "C"),
        ("**D**", "D"),
        ("Panel B", "B"),
        ("Answer is C.", "C"),
        ("reasoning mentions B and C\n\\boxed{D}", "D"),
    ):
        assert parse_strict_choice(raw, allowed=allowed).choice == expected
    for raw in (None, "", "A", "N", "B or C", "Answer: B, answer: C"):
        assert parse_strict_choice(raw, allowed=allowed).choice is None


def test_last_primary_is_equal_weight_condition_macro_and_invalid_is_wrong() -> None:
    tasks = (
        _task(
            0,
            dataset=LAST_DATASET,
            answer="B",
            labels="BCD",
            metadata={"source_dir": "r1", "condition_id": "c1"},
        ),
        _task(
            1,
            dataset=LAST_DATASET,
            answer="C",
            labels="BCD",
            metadata={"source_dir": "r1", "condition_id": "c1"},
        ),
        _task(
            2,
            dataset=LAST_DATASET,
            answer="D",
            labels="BCD",
            metadata={"source_dir": "r2", "condition_id": "c2"},
        ),
    )
    records = {
        0: {
            "ordinal": 0,
            "sample_id": tasks[0].sample_id,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "final_answer": "B",
        },
        1: {
            "ordinal": 1,
            "sample_id": tasks[1].sample_id,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "final_answer": "B or C",
        },
        2: {
            "ordinal": 2,
            "sample_id": tasks[2].sample_id,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "final_answer": "D",
        },
    }
    score = score_texture_benchmark(
        tasks, records, task_manifest_sha256=TASK_MANIFEST_SHA256
    )["last"]
    assert score["conditions"]["c1"]["accuracy"] == 0.5
    assert score["conditions"]["c1"]["invalid_count"] == 1
    assert score["conditions"]["c2"]["accuracy"] == 1.0
    assert score["primary_metric"] == "four_condition_macro_accuracy"
    assert score["four_condition_macro_accuracy"] == 0.75
    assert score["micro"]["accuracy"] == 2 / 3


def test_scoring_rejects_unbound_or_cross_manifest_result() -> None:
    task = _task(
        0,
        dataset=LAST_DATASET,
        answer="B",
        labels="BCD",
        metadata={"source_dir": "r1", "condition_id": "c1"},
    )
    with pytest.raises(ValueError, match="sample identity differs"):
        score_texture_benchmark(
            (task,),
            {
                0: {
                    "ordinal": 0,
                    "task_manifest_sha256": TASK_MANIFEST_SHA256,
                    "final_answer": "B",
                }
            },
            task_manifest_sha256=TASK_MANIFEST_SHA256,
        )
    with pytest.raises(ValueError, match="task manifest identity differs"):
        score_texture_benchmark(
            (task,),
            {
                0: {
                    "ordinal": 0,
                    "sample_id": task.sample_id,
                    "task_manifest_sha256": "1" * 64,
                    "final_answer": "B",
                }
            },
            task_manifest_sha256=TASK_MANIFEST_SHA256,
        )


def test_mmad_detection_uses_balanced_accuracy_then_task_macro() -> None:
    tasks = (
        _task(
            0,
            dataset=MMAD_DATASET,
            answer="A",
            labels="AB",
            metadata={
                "score_dataset": "VisA",
                "question_type_score": "Anomaly Detection",
                "is_normal": "false",
            },
        ),
        _task(
            1,
            dataset=MMAD_DATASET,
            answer="B",
            labels="AB",
            metadata={
                "score_dataset": "VisA",
                "question_type_score": "Anomaly Detection",
                "is_normal": "true",
            },
        ),
        _task(
            2,
            dataset=MMAD_DATASET,
            answer="B",
            labels="AB",
            metadata={
                "score_dataset": "VisA",
                "question_type_score": "Anomaly Detection",
                "is_normal": "true",
            },
        ),
        _task(
            3,
            dataset=MMAD_DATASET,
            answer="C",
            metadata={
                "score_dataset": "VisA",
                "question_type_score": "Defect Description",
                "is_normal": "false",
            },
        ),
    )
    records = {
        ordinal: {
            "ordinal": ordinal,
            "sample_id": tasks[ordinal].sample_id,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "final_answer": answer,
        }
        for ordinal, answer in enumerate(("A", "B", "A", "C"))
    }
    score = score_texture_benchmark(
        tasks, records, task_manifest_sha256=TASK_MANIFEST_SHA256
    )["mmad"]
    visa = score["datasets"]["VisA"]
    detection = visa["tasks"]["Anomaly Detection"]
    assert detection["accuracy"] == 2 / 3
    assert detection["official_balanced_accuracy"] == 0.75
    assert visa["official_task_macro_accuracy"] == (0.75 + 1.0) / 2
    assert score["official_dataset_task_macro_accuracy"] == 0.875


def test_mmad_permissive_reparse_is_explicit_and_keeps_full_denominator() -> None:
    task = _task(
        0,
        dataset=MMAD_DATASET,
        answer="B",
        labels="AB",
        metadata={
            "score_dataset": "VisA",
            "question_type_score": "Defect Description",
            "is_normal": "false",
        },
    )
    records = {
        0: {
            "ordinal": 0,
            "sample_id": task.sample_id,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "final_answer": "B. Yes.",
        }
    }

    strict = score_texture_benchmark(
        (task,), records, task_manifest_sha256=TASK_MANIFEST_SHA256
    )
    permissive = score_texture_benchmark(
        (task,),
        records,
        task_manifest_sha256=TASK_MANIFEST_SHA256,
        mmad_parser=MMAD_PARSER_UPSTREAM_LIKE_PERMISSIVE,
    )

    assert strict["mmad"]["micro"]["invalid_count"] == 1
    assert permissive["mmad"]["micro"]["accuracy"] == 1.0
    contract = permissive["parser_contract"]
    assert contract["mmad_parser"] == MMAD_PARSER_UPSTREAM_LIKE_PERMISSIVE
    assert contract["denominator"] == "fixed_complete_task_manifest"
    assert contract["invalid_answer_policy"] == "count_as_incorrect"
    assert contract["drop_invalid_rows"] is False
    assert contract["exact_upstream_evaluator"] is False
    assert len(contract["identity_sha256"]) == 64
    assert contract["identity_sha256"] != strict["parser_contract"]["identity_sha256"]
