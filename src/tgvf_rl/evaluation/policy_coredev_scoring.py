"""Materialize immutable VLMEvalKit inputs from policy CoreDev trajectories."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .final_answer_view import materialize_final_answer_view
from .policy_coredev import load_coredev_tasks


POLICY_SCORING_VIEW_SCHEMA = "tgvf-policy-coredev-scoring-view-v1"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
DATASETS = (
    "VStarBench",
    "HRBench4K",
    "BLINK",
    "OCRBench_v2",
    "MMMU_Pro_10c",
    "MathVista_MINI",
    "MathVerse_MINI",
)
_TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")


def normalize_policy_final_answer(value: object) -> str | None:
    """Remove only terminal Qwen markers from the agent-loop final answer."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("policy final_answer must be text or null")
    answer = value.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _TERMINAL_MARKERS:
            if answer.endswith(marker):
                answer = answer[: -len(marker)].rstrip()
                changed = True
    return answer if answer.strip() else None


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    previous = csv.field_size_limit()
    csv.field_size_limit(1024 * 1024 * 1024)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "index" not in reader.fieldnames:
                raise RuntimeError(f"CoreDev TSV has no index: {path}")
            rows = [dict(row) for row in reader]
    finally:
        csv.field_size_limit(previous)
    if len({row["index"] for row in rows}) != len(rows):
        raise RuntimeError(f"CoreDev TSV indices are not unique: {path}")
    return list(reader.fieldnames), rows


def _write_tsv_exclusive(
    path: Path, fields: list[str], rows: list[Mapping[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_trajectories(
    inference_root: Path, *, tasks_path: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    tasks = load_coredev_tasks(tasks_path)
    task_by_ordinal = {task.ordinal: task for task in tasks}
    expected = {task.ordinal for task in tasks if task.single_image}
    records: dict[int, dict[str, Any]] = {}
    for rank in range(4):
        path = inference_root / f"rank-{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing policy inference rank: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = json.loads(line)
            ordinal = raw.get("ordinal")
            if type(ordinal) is not int or ordinal in records:
                raise RuntimeError(f"invalid/duplicate ordinal in {path}:{line_number}")
            task = task_by_ordinal.get(ordinal)
            if task is None or not task.single_image:
                raise RuntimeError("policy result is outside the single-image tranche")
            if raw.get("dataset") != task.dataset or raw.get("index") != task.index:
                raise RuntimeError("policy result identity differs from task materialization")
            records[ordinal] = raw
    if set(records) != expected:
        raise RuntimeError("policy scoring requires all 2,240 single-image trajectories")
    by_index = {
        (task_by_ordinal[ordinal].dataset, task_by_ordinal[ordinal].index): raw
        for ordinal, raw in records.items()
    }
    return by_index, len(tasks) - len(expected)


def materialize_policy_coredev_scoring_views(
    *,
    inference_root: str | Path,
    tasks_path: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    evaluation_id: str,
    run_id: str,
    mathverse_source_json: str | Path,
) -> dict[str, Any]:
    """Build seven full-count scoring TSVs with unsupported rows fail-closed."""

    inference = Path(inference_root).resolve()
    tasks = Path(tasks_path).resolve()
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if not evaluation_id or not run_id:
        raise ValueError("evaluation_id and run_id must be non-empty")
    trajectories, unsupported_count = _load_trajectories(
        inference, tasks_path=tasks
    )
    created_at = datetime.now().astimezone().isoformat()
    slices: list[dict[str, Any]] = []
    observed_total = 0
    for dataset in DATASETS:
        fields, source_rows = _read_tsv(source / f"{dataset}.tsv")
        if "prediction" in fields or "extra_records" in fields:
            raise RuntimeError("official source TSV unexpectedly owns result columns")
        raw_fields = fields + ["prediction", "extra_records"]
        raw_rows: list[dict[str, str]] = []
        observed = 0
        for row in source_rows:
            trajectory = trajectories.get((dataset, row["index"]))
            materialized = dict(row)
            if trajectory is None:
                answer = None
                extra = {
                    "schema_version": POLICY_SCORING_VIEW_SCHEMA,
                    "evaluation_id": evaluation_id,
                    "coverage": "unsupported_multi_image",
                }
            else:
                observed += 1
                answer = normalize_policy_final_answer(trajectory.get("final_answer"))
                extra = {
                    "schema_version": POLICY_SCORING_VIEW_SCHEMA,
                    "evaluation_id": evaluation_id,
                    "coverage": "single_image_evaluated",
                    "ordinal": trajectory["ordinal"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "trajectory_sha256": trajectory["trajectory_sha256"],
                    "policy_run_id": trajectory["policy_run_id"],
                    "policy_weights_sha256": trajectory["policy_weights_sha256"],
                    "stop": trajectory["stop"],
                    "tool_call_count": len(trajectory["tool_calls"]),
                    "successful_observation_count": trajectory[
                        "successful_observation_count"
                    ],
                }
            # Reuse the audited final-answer view's fail-closed invalid-row
            # handling by presenting the already-separated answer as a native
            # suffix. Empty/unsupported rows deliberately lack the closer.
            materialized["prediction"] = "" if answer is None else f"</think>{answer}"
            materialized["extra_records"] = json.dumps(
                extra, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            raw_rows.append(materialized)
        observed_total += observed
        work_dir = output / dataset
        run_dir = work_dir / MODEL_NAME / run_id
        raw_path = output / "raw" / f"{dataset}.tsv"
        derived_path = run_dir / f"{MODEL_NAME}_{dataset}.tsv"
        manifest_path = run_dir / "final-answer-view-manifest.json"
        _write_tsv_exclusive(raw_path, raw_fields, raw_rows)
        materialized = materialize_final_answer_view(
            source_tsv=raw_path,
            derived_tsv=derived_path,
            manifest_path=manifest_path,
            mathverse_source_json=(
                Path(mathverse_source_json).resolve()
                if dataset == "MathVerse_MINI"
                else None
            ),
        )
        status = {
            "schema_version": "1.0",
            "eval_id": run_id,
            "created_at": created_at,
            "datasets": {
                dataset: {
                    "status": "done",
                    "prediction_file": str(derived_path),
                    "updated_at": created_at,
                    "judge_model": "Qwen2.5-72B-Instruct",
                    "source_run": evaluation_id,
                    "reuse_aux": "infer",
                    "skip_reason": "mode_infer",
                    "scoring_view_contract": POLICY_SCORING_VIEW_SCHEMA,
                    "observed_single_image_count": observed,
                    "unsupported_multi_image_count": len(source_rows) - observed,
                }
            },
            "model_name": MODEL_NAME,
            "commit": "7055d301",
            "argv": ["synthetic-policy-coredev-scoring-view", evaluation_id],
            "api_mode": False,
            "world_size": 1,
            "pred_format": "tsv",
            "eval_format": "json",
            "mode": "infer",
            "reuse": False,
            "reuse_aux": "infer",
            "updated_at": created_at,
        }
        _write_json_exclusive(run_dir / "status.json", status)
        _write_json_exclusive(run_dir / "materializer-output.json", materialized)
        slices.append(
            {
                "dataset": dataset,
                "official_row_count": len(source_rows),
                "observed_single_image_count": observed,
                "unsupported_multi_image_count": len(source_rows) - observed,
                "work_dir": str(work_dir),
                "prediction_file": str(derived_path),
                "manifest": str(manifest_path),
            }
        )
    if observed_total != 2240 or unsupported_count != 271:
        raise RuntimeError("CoreDev coverage boundary differs from 2,240 + 271")
    result = {
        "schema_version": POLICY_SCORING_VIEW_SCHEMA,
        "evaluation_id": evaluation_id,
        "run_id": run_id,
        "observed_single_image_count": observed_total,
        "unsupported_multi_image_count": unsupported_count,
        "official_row_count": observed_total + unsupported_count,
        "slices": slices,
    }
    _write_json_exclusive(output / "materialization-summary.json", result)
    return result


__all__ = [
    "POLICY_SCORING_VIEW_SCHEMA",
    "materialize_policy_coredev_scoring_views",
    "normalize_policy_final_answer",
]
