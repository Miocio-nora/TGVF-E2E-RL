"""Generic, identity-bound MCQ scoring for policy benchmark manifests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

from tgvf_rl.immutable_publication import (
    ImmutablePublicationError,
    publish_bytes_content_consistent,
)

from .policy_coredev import (
    POLICY_EVALUATION_IDENTITY_SCHEMA,
    load_benchmark_tasks,
    load_policy_benchmark_results,
)
from .policy_coredev_scoring import normalize_policy_final_answer


POLICY_BENCHMARK_SCORING_SCHEMA = "tgvf-policy-benchmark-mcq-scoring-v1"
_TERMINAL_OPTION_RE = re.compile(
    r"(?i)(?:correct\s+)?answer\s+(?:is|:)\s*\**\(?([A-Z])\)?\**"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def load_policy_evaluation_identity(
    path: str | Path, *, expected_file_sha256: str
) -> tuple[dict[str, Any], str]:
    """Load one immutable evaluation identity and verify its internal digest."""

    identity_path = Path(path).resolve()
    expected = _require_sha256(
        expected_file_sha256, name="evaluation identity file SHA256"
    )
    observed = _file_sha256(identity_path)
    if observed != expected:
        raise ValueError("evaluation identity file SHA256 differs")
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation identity is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("evaluation identity must be an object")
    if payload.get("schema_version") != POLICY_EVALUATION_IDENTITY_SCHEMA:
        raise ValueError("evaluation identity schema differs")
    declared = _require_sha256(
        payload.get("identity_sha256"), name="evaluation identity SHA256"
    )
    content = dict(payload)
    content.pop("identity_sha256")
    if _canonical_sha256(content) != declared:
        raise ValueError("evaluation identity internal digest differs")
    return payload, observed


def infer_mcq_option(
    answer: object, options: Mapping[str, str]
) -> tuple[str | None, str]:
    """Apply a deterministic VLMEvalKit-compatible no-judge MCQ heuristic."""

    normalized = normalize_policy_final_answer(answer)
    if normalized is None:
        return None, "missing_final_answer"
    allowed = tuple(options)
    if not allowed or any(
        len(letter) != 1 or not letter.isupper() for letter in allowed
    ):
        raise ValueError("MCQ options must use unique uppercase letters")
    answer_mod = normalized
    for character in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(character, " ")
    tokens = [token.strip() for token in answer_mod.split()]
    mentioned = [letter for letter in allowed if letter in tokens]
    if len(mentioned) == 1 and tokens.index(mentioned[0]) > len(tokens) - 5:
        return mentioned[0], "terminal_option_token"
    match = _TERMINAL_OPTION_RE.search(normalized)
    if match and match.group(1).upper() in options:
        return match.group(1).upper(), "answer_is_pattern"
    lowered = normalized.casefold()
    option_text_length = sum(len(str(value)) for value in options.values())
    if len(normalized) <= 2 * option_text_length:
        text_matches = [
            letter
            for letter, value in options.items()
            if str(value).casefold() in lowered
        ]
        if len(text_matches) == 1:
            return text_matches[0], "unique_option_text"
    return None, "ambiguous_or_unmatched"


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_content_consistent(path, payload)
    except ImmutablePublicationError as error:
        raise RuntimeError(f"immutable scoring output differs: {path}") from error


def _tsv_bytes(rows: list[dict[str, object]]) -> bytes:
    fields = (
        "ordinal",
        "dataset",
        "index",
        "sample_id",
        "answer",
        "prediction",
        "extracted_option",
        "correct",
        "extraction_method",
        "category",
        "cycle_category",
        "trajectory_id",
        "trajectory_sha256",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _inference_tree_identity(root: Path, *, world_size: int) -> dict[str, object]:
    files = []
    for rank in range(world_size):
        path = root / f"rank-{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing policy benchmark rank result: {path}")
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {"files": files, "identity_sha256": _canonical_sha256(files)}


def materialize_policy_benchmark_mcq_scoring(
    *,
    inference_root: str | Path,
    tasks_path: str | Path,
    tasks_sha256: str,
    evaluation_identity_path: str | Path,
    evaluation_identity_file_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Score a complete arbitrary MCQ task manifest and materialize TSV views."""

    identity, identity_file_sha256 = load_policy_evaluation_identity(
        evaluation_identity_path,
        expected_file_sha256=evaluation_identity_file_sha256,
    )
    task_binding = identity.get("task_manifest")
    execution = identity.get("execution")
    if not isinstance(task_binding, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("evaluation identity task/execution binding is malformed")
    expected_tasks_sha256 = _require_sha256(tasks_sha256, name="tasks manifest SHA256")
    if task_binding.get("sha256") != expected_tasks_sha256:
        raise ValueError("tasks manifest differs from evaluation identity")
    task_count = task_binding.get("task_count")
    single_image_count = task_binding.get("single_image_count")
    if type(task_count) is not int or type(single_image_count) is not int:
        raise ValueError("evaluation identity task counts are malformed")
    tasks = load_benchmark_tasks(
        tasks_path,
        expected_task_count=task_count,
        expected_single_image_count=single_image_count,
        expected_sha256=expected_tasks_sha256,
        verify_image_paths=False,
        verify_image_contents=False,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    if any(not task.single_image for task in tasks):
        raise ValueError(
            "generic MCQ scoring does not accept unsupported multi-image rows"
        )
    records = load_policy_benchmark_results(
        inference_root,
        tasks=tasks,
        evaluation_identity=identity,
        require_complete=True,
    )
    scored: list[dict[str, object]] = []
    dataset_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        if task.answer is None or not task.options:
            raise ValueError(
                f"MCQ task {task.bound_sample_id} lacks retained gold answer/options"
            )
        result = records[task.ordinal]
        options = dict(task.options)
        extracted, method = infer_mcq_option(result.get("final_answer"), options)
        metadata = dict(task.metadata)
        row: dict[str, object] = {
            "ordinal": task.ordinal,
            "dataset": task.dataset,
            "index": task.index,
            "sample_id": task.bound_sample_id,
            "answer": task.answer,
            "prediction": normalize_policy_final_answer(result.get("final_answer"))
            or "",
            "extracted_option": extracted or "",
            "correct": extracted == task.answer,
            "extraction_method": method,
            "category": metadata.get("category", ""),
            "cycle_category": metadata.get("cycle_category", ""),
            "trajectory_id": result["trajectory_id"],
            "trajectory_sha256": result["trajectory_sha256"],
        }
        scored.append(row)
        dataset_rows[task.dataset].append(row)
    dataset_summaries: dict[str, dict[str, object]] = {}
    output = Path(output_root).resolve()
    for dataset, rows in sorted(dataset_rows.items()):
        correct_count = sum(bool(row["correct"]) for row in rows)
        summary: dict[str, object] = {
            "sample_count": len(rows),
            "correct_count": correct_count,
            "accuracy": correct_count / len(rows),
        }
        categories: dict[str, dict[str, object]] = {}
        for category in sorted(
            {str(row["category"]) for row in rows if row["category"]}
        ):
            subset = [row for row in rows if row["category"] == category]
            category_correct = sum(bool(row["correct"]) for row in subset)
            categories[category] = {
                "sample_count": len(subset),
                "correct_count": category_correct,
                "accuracy": category_correct / len(subset),
            }
        if categories:
            summary["categories"] = categories
        dataset_summaries[dataset] = summary
        _write_immutable(output / "datasets" / f"{dataset}.tsv", _tsv_bytes(rows))
    total_correct = sum(bool(row["correct"]) for row in scored)
    world_size = execution.get("world_size")
    if type(world_size) is not int:
        raise ValueError("evaluation identity world_size is malformed")
    inference_identity = _inference_tree_identity(
        Path(inference_root).resolve(), world_size=world_size
    )
    result: dict[str, Any] = {
        "schema_version": POLICY_BENCHMARK_SCORING_SCHEMA,
        "evaluation_id": identity["evaluation_id"],
        "evaluation_identity_sha256": identity["identity_sha256"],
        "policy_snapshot": identity["policy_snapshot"],
        "inputs": {
            "evaluation_identity_file_sha256": identity_file_sha256,
            "task_manifest_sha256": expected_tasks_sha256,
            "inference": inference_identity,
        },
        "sample_count": len(scored),
        "correct_count": total_correct,
        "micro_accuracy": total_correct / len(scored),
        "macro_dataset_accuracy": sum(
            float(summary["accuracy"]) for summary in dataset_summaries.values()
        )
        / len(dataset_summaries),
        "datasets": dataset_summaries,
    }
    result["report_identity_sha256"] = _canonical_sha256(result)
    scored_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored
    ).encode("utf-8")
    _write_immutable(output / "scored-results.jsonl", scored_bytes)
    _write_immutable(
        output / "summary.json",
        (
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return result


__all__ = [
    "POLICY_BENCHMARK_SCORING_SCHEMA",
    "infer_mcq_option",
    "load_policy_evaluation_identity",
    "materialize_policy_benchmark_mcq_scoring",
]
