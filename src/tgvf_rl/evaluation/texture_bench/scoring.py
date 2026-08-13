"""Strict local scoring for LAS&T and MMAD texture benchmarks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from .schema import require_sha256
from .task import TextureTask


TEXTURE_SCORING_SCHEMA = "tgvf-texture-benchmark-scoring-v1"
LAST_DATASET = "LAST_2D_Texture_Retrieval"
MMAD_DATASET = "MMAD"
MMAD_TASK_ORDER = (
    "Anomaly Detection",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
    "Object Classification",
    "Object Analysis",
)

_TERMINAL_TAG_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)+\s*$")
_WRAPPED_RE = re.compile(
    r"(?is)^\s*(?:<answer>\s*)?\\?boxed\s*\{?\s*([A-E])\s*\}?\s*(?:</answer>)?\s*$"
)
_ANSWER_MARKER_RE = re.compile(
    r"(?i)\b(?:answer|option|panel|choice)\s*(?:is|:|=)?\s*\(?\s*([A-E])\s*\)?"
)


@dataclass(frozen=True, slots=True)
class ParsedChoice:
    choice: str | None
    status: str
    method: str

    @property
    def valid(self) -> bool:
        return self.choice is not None


def _normalize_answer(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _TERMINAL_TAG_RE.sub("", normalized).strip()
    return normalized or None


def parse_strict_choice(answer: object, *, allowed: Sequence[str]) -> ParsedChoice:
    """Parse an explicit final MCQ answer without looking through tool traces."""

    options = tuple(str(item).upper() for item in allowed)
    if (
        not options
        or len(options) != len(set(options))
        or any(len(item) != 1 or not item.isalpha() for item in options)
    ):
        raise ValueError("allowed choices must be unique letters")
    text = _normalize_answer(answer)
    if text is None:
        return ParsedChoice(None, "invalid", "missing")
    wrapped = _WRAPPED_RE.fullmatch(text)
    if wrapped and wrapped.group(1).upper() in options:
        return ParsedChoice(wrapped.group(1).upper(), "valid", "whole_wrapper")
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    final = lines[-1] if lines else text
    plain_patterns = (
        re.compile(r"(?i)^\s*(?:\*\*)?\(?\s*([A-E])\s*\)?[.!]?\s*(?:\*\*)?$"),
        re.compile(
            r"(?i)^\s*(?:answer|option|panel|choice)\s*(?:is|:|=)?\s*\(?\s*([A-E])\s*\)?[.!]?\s*$"
        ),
        re.compile(r"(?i)^\s*\\boxed\s*\{\s*([A-E])\s*\}\s*[.!]?\s*$"),
        re.compile(r"(?is)^\s*<answer>\s*([A-E])\s*</answer>\s*$"),
    )
    for pattern in plain_patterns:
        match = pattern.fullmatch(final)
        if match and match.group(1).upper() in options:
            return ParsedChoice(match.group(1).upper(), "valid", "decisive_final_line")
    markers = tuple(
        match.group(1).upper()
        for match in _ANSWER_MARKER_RE.finditer(text)
        if match.group(1).upper() in options
    )
    if markers and len(set(markers)) == 1:
        return ParsedChoice(markers[0], "valid", "unique_answer_marker")
    return ParsedChoice(None, "invalid", "ambiguous_or_unmatched")


def parse_mmad_official_legacy(
    answer: object, options: Mapping[str, str]
) -> ParsedChoice:
    """Reproduce MMAD's last-uppercase-letter/fuzzy fallback for diagnostics."""

    text = _normalize_answer(answer)
    if text is None:
        return ParsedChoice(None, "invalid", "missing")
    allowed = tuple(options)
    matches = [item for item in re.findall(r"\b([A-E])\b", text) if item in allowed]
    if matches:
        return ParsedChoice(matches[-1], "valid", "official_last_letter")
    closest = difflib.get_close_matches(text, list(options.values()), n=1, cutoff=0.0)
    if closest:
        for label, option_text in options.items():
            if option_text == closest[0]:
                return ParsedChoice(label, "valid", "official_fuzzy_option")
    return ParsedChoice(None, "invalid", "official_unmatched")


def load_result_rows(paths: Iterable[str | Path]) -> dict[int, dict[str, object]]:
    """Load one or more JSONLs and reject incomplete/ambiguous row identities."""

    records: dict[int, dict[str, object]] = {}
    for source in paths:
        path = Path(source)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"texture result file is unreadable: {path}") from error
        for line_number, line in enumerate(lines, 1):
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"result row is not an object at {path}:{line_number}")
            ordinal = row.get("ordinal")
            if type(ordinal) is not int or ordinal < 0:
                raise ValueError(f"result ordinal is invalid at {path}:{line_number}")
            if ordinal in records:
                raise ValueError(f"duplicate result ordinal: {ordinal}")
            records[ordinal] = row
    return records


def _metadata(task: TextureTask) -> dict[str, str]:
    return dict(task.metadata)


def _score_rows(
    tasks: Sequence[TextureTask],
    records: Mapping[int, Mapping[str, object]],
    *,
    task_manifest_sha256: str,
) -> list[dict[str, object]]:
    require_sha256(task_manifest_sha256, name="task manifest SHA256")
    expected = {task.ordinal for task in tasks}
    observed = set(records)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"texture results are not complete (missing={missing[:8]}, extra={extra[:8]})"
        )
    rows: list[dict[str, object]] = []
    for task in tasks:
        if task.answer is None or not task.options:
            raise ValueError(
                f"texture task has no gold/options: {task.bound_sample_id}"
            )
        result = records[task.ordinal]
        result_sample = result.get("sample_id")
        if result_sample != task.bound_sample_id:
            raise ValueError(
                f"result sample identity differs at ordinal {task.ordinal}"
            )
        if result.get("task_manifest_sha256") != task_manifest_sha256:
            raise ValueError(
                f"result task manifest identity differs at ordinal {task.ordinal}"
            )
        options = dict(task.options)
        parsed = parse_strict_choice(result.get("final_answer"), allowed=tuple(options))
        rows.append(
            {
                "ordinal": task.ordinal,
                "sample_id": task.bound_sample_id,
                "dataset": task.dataset,
                "gold": task.answer,
                "prediction": parsed.choice,
                "parse_status": parsed.status,
                "parse_method": parsed.method,
                "correct": parsed.choice == task.answer,
                "metadata": _metadata(task),
            }
        )
    return rows


def _accuracy(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    valid = sum(row["prediction"] is not None for row in rows)
    valid_correct = sum(
        bool(row["correct"]) for row in rows if row["prediction"] is not None
    )
    return {
        "sample_count": total,
        "correct_count": correct,
        "valid_count": valid,
        "invalid_count": total - valid,
        "accuracy": correct / total if total else None,
        "invalid_rate": (total - valid) / total if total else None,
        "valid_only_accuracy": valid_correct / valid if valid else None,
    }


def score_last(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if row["dataset"] == LAST_DATASET]
    if not selected:
        raise ValueError("results contain no LAS&T tasks")
    by_source: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_condition: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        metadata = row["metadata"]
        assert isinstance(metadata, Mapping)
        source_dir = metadata.get("source_dir")
        condition = metadata.get("condition_id", metadata.get("category"))
        if not source_dir or not condition:
            raise ValueError("LAS&T task metadata lacks source_dir/condition_id")
        by_source[str(source_dir)].append(row)
        by_condition[str(condition)].append(row)
    source_scores = {
        name: _accuracy(values) for name, values in sorted(by_source.items())
    }
    condition_scores = {
        name: _accuracy(values) for name, values in sorted(by_condition.items())
    }
    condition_accuracies = [
        float(value["accuracy"])
        for value in condition_scores.values()
        if value["accuracy"] is not None
    ]
    return {
        "primary_metric": "four_condition_macro_accuracy",
        "four_condition_macro_accuracy": (
            sum(condition_accuracies) / len(condition_accuracies)
        ),
        "micro": _accuracy(selected),
        "conditions": condition_scores,
        "physical_directories": source_scores,
    }


def _binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def score_mmad(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if row["dataset"] == MMAD_DATASET]
    if not selected:
        raise ValueError("results contain no MMAD tasks")
    by_dataset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        metadata = row["metadata"]
        assert isinstance(metadata, Mapping)
        dataset = metadata.get("score_dataset", metadata.get("cycle_category"))
        question_type = metadata.get("question_type_score", metadata.get("category"))
        if not dataset or not question_type:
            raise ValueError("MMAD task metadata lacks score dataset/question type")
        by_dataset[str(dataset)].append(row)

    datasets: dict[str, object] = {}
    dataset_macro_values: list[float] = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        by_type: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in dataset_rows:
            metadata = row["metadata"]
            assert isinstance(metadata, Mapping)
            by_type[
                str(metadata.get("question_type_score", metadata.get("category")))
            ].append(row)
        task_scores: dict[str, dict[str, object]] = {}
        for question_type in MMAD_TASK_ORDER:
            values = by_type.get(question_type, [])
            if not values:
                continue
            cell = _accuracy(values)
            if question_type == "Anomaly Detection":
                normal = []
                abnormal = []
                for row in values:
                    metadata = row["metadata"]
                    assert isinstance(metadata, Mapping)
                    (
                        normal if metadata.get("is_normal") == "true" else abnormal
                    ).append(row)
                normal_score = _accuracy(normal)
                abnormal_score = _accuracy(abnormal)
                normal_accuracy = normal_score["accuracy"]
                abnormal_accuracy = abnormal_score["accuracy"]
                balanced = (
                    (float(normal_accuracy) + float(abnormal_accuracy)) / 2
                    if normal_accuracy is not None and abnormal_accuracy is not None
                    else None
                )
                cell.update(
                    {
                        "official_balanced_accuracy": balanced,
                        "normal": normal_score,
                        "abnormal": abnormal_score,
                    }
                )
            task_scores[question_type] = cell
        official_cells = []
        for question_type, cell in task_scores.items():
            value = (
                cell.get("official_balanced_accuracy")
                if question_type == "Anomaly Detection"
                else cell.get("accuracy")
            )
            if value is not None:
                official_cells.append(float(value))
        dataset_average = (
            sum(official_cells) / len(official_cells) if official_cells else None
        )
        detection = by_type.get("Anomaly Detection", [])
        abnormal_rows = []
        normal_rows = []
        for row in detection:
            metadata = row["metadata"]
            assert isinstance(metadata, Mapping)
            (
                normal_rows if metadata.get("is_normal") == "true" else abnormal_rows
            ).append(row)
        tp = sum(bool(row["correct"]) for row in abnormal_rows)
        fn = len(abnormal_rows) - tp
        tn = sum(bool(row["correct"]) for row in normal_rows)
        fp = len(normal_rows) - tn
        binary = _binary_metrics(tp, fp, fn)
        binary.update(
            {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "overkill": fp / len(normal_rows) if normal_rows else None,
                "miss": fn / len(abnormal_rows) if abnormal_rows else None,
            }
        )
        datasets[dataset] = {
            "sample_count": len(dataset_rows),
            "official_task_macro_accuracy": dataset_average,
            "tasks": task_scores,
            "anomaly_detection": binary,
        }
        if dataset_average is not None:
            dataset_macro_values.append(dataset_average)
    return {
        "primary_metric": "official_dataset_task_macro_accuracy",
        "official_dataset_task_macro_accuracy": (
            sum(dataset_macro_values) / len(dataset_macro_values)
            if dataset_macro_values
            else None
        ),
        "micro": _accuracy(selected),
        "datasets": datasets,
    }


def score_texture_benchmark(
    tasks: Sequence[TextureTask],
    records: Mapping[int, Mapping[str, object]],
    *,
    task_manifest_sha256: str,
) -> dict[str, object]:
    rows = _score_rows(
        tasks,
        records,
        task_manifest_sha256=task_manifest_sha256,
    )
    datasets = {str(row["dataset"]) for row in rows}
    summary: dict[str, object] = {
        "schema_version": TEXTURE_SCORING_SCHEMA,
        "task_count": len(rows),
        "micro": _accuracy(rows),
    }
    if LAST_DATASET in datasets:
        summary["last"] = score_last(rows)
    if MMAD_DATASET in datasets:
        summary["mmad"] = score_mmad(rows)
    unknown = datasets - {LAST_DATASET, MMAD_DATASET}
    if unknown:
        raise ValueError(f"unknown texture benchmark datasets: {sorted(unknown)}")
    return summary


__all__ = [
    "LAST_DATASET",
    "MMAD_DATASET",
    "MMAD_TASK_ORDER",
    "ParsedChoice",
    "TEXTURE_SCORING_SCHEMA",
    "load_result_rows",
    "parse_mmad_official_legacy",
    "parse_strict_choice",
    "score_last",
    "score_mmad",
    "score_texture_benchmark",
]
