"""Fail-closed judge and result acceptance for the seven CoreDev slices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
import csv
import json
from pathlib import Path
import time
from typing import Any, Literal
from urllib import request

from .coredev_materialize import (
    COREDEV_JUDGE_CONTRACTS,
    COREDEV_LLM_JUDGE_MODEL,
)
from .vlmevalkit import COREDEV_2511, VLMEVALKIT_REVIEW_COMMIT


COREDEV_BASELINE_MODEL = "Qwen3-VL-8B-Thinking"
JUDGE_READY_TEXT = "TGVF_JUDGE_READY"
JUDGE_FAILURE_MARKERS = (
    "Failed to obtain answer via API",
    "Failed in Prefetch, no GPT-based answer matching",
    "Failed to predict, thus randomly generate one",
    "All 5 retries failed",
)


def _json_request(
    url: str,
    payload: Mapping[str, object] | None,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    with opener(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("judge service returned a non-object JSON response")
    return result


def check_qwen25_72b_judge(
    *,
    base_url: str,
    expected_model: str = COREDEV_LLM_JUDGE_MODEL,
    timeout: float = 120,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Require both model discovery and one deterministic judge completion."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("judge base_url must be non-empty text")
    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ValueError("expected judge model must be non-empty text")
    if timeout <= 0:
        raise ValueError("judge timeout must be positive")
    resolved_opener = request.urlopen if opener is None else opener
    endpoint = base_url.rstrip("/")
    started = time.monotonic()
    models = _json_request(
        f"{endpoint}/models",
        None,
        timeout=timeout,
        opener=resolved_opener,
    )
    data = models.get("data")
    if not isinstance(data, list):
        raise RuntimeError("judge service /models response lacks data array")
    served = [item.get("id") for item in data if isinstance(item, Mapping)]
    if expected_model not in served:
        raise RuntimeError(
            f"expected judge {expected_model}, service returned {served}"
        )

    completion = _json_request(
        f"{endpoint}/chat/completions",
        {
            "model": expected_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with only the text {JUDGE_READY_TEXT}.",
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
            "seed": 42,
        },
        timeout=timeout,
        opener=resolved_opener,
    )
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("judge service returned no completion choice")
    first = choices[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
        raise RuntimeError("judge service returned a malformed completion choice")
    content = first["message"].get("content")
    if not isinstance(content, str) or content.strip() != JUDGE_READY_TEXT:
        raise RuntimeError(f"unexpected deterministic judge response: {content!r}")

    return {
        "schema_version": 1,
        "status": "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "base_url": endpoint,
        "model": expected_model,
        "served_models": served,
        "response": content,
        "finish_reason": first.get("finish_reason"),
        "usage": completion.get("usage"),
        "elapsed_seconds": time.monotonic() - started,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class FailClosedJudge:
    """Reject unavailable or exhausted API calls instead of returning sentinels."""

    __slots__ = ("_delegate",)

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        delegate = object.__getattribute__(self, "_delegate")
        return getattr(delegate, name)

    def __getstate__(self) -> Any:
        return object.__getattribute__(self, "_delegate")

    def __setstate__(self, delegate: Any) -> None:
        object.__setattr__(self, "_delegate", delegate)

    def working(self) -> bool:
        try:
            ready = self._delegate.working()
        except Exception as error:
            raise RuntimeError("CoreDev judge readiness check failed") from error
        if ready is not True:
            raise RuntimeError(
                "CoreDev judge is unavailable; exact-matching fallback is forbidden"
            )
        return True

    def generate(self, *args: Any, **kwargs: Any) -> str:
        try:
            result = self._delegate.generate(*args, **kwargs)
        except Exception as error:
            raise RuntimeError("CoreDev judge request failed") from error
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("CoreDev judge returned an empty response")
        if any(marker in result for marker in JUDGE_FAILURE_MARKERS):
            raise RuntimeError("CoreDev judge exhausted its API retries")
        return result


def install_fail_closed_judge_builders(modules: Iterable[Any]) -> None:
    """Wrap VLMEvalKit module-local ``build_judge`` bindings in place."""

    for module in modules:
        original = getattr(module, "build_judge")
        if getattr(original, "_tgvf_fail_closed", False):
            continue

        @wraps(original)
        def build_fail_closed(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            return FailClosedJudge(_original(*args, **kwargs))

        build_fail_closed._tgvf_fail_closed = True  # type: ignore[attr-defined]
        module.build_judge = build_fail_closed


def _read_tsv(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"index", "prediction"}.issubset(
            reader.fieldnames
        ):
            raise RuntimeError(f"prediction TSV schema is incomplete: {path}")
        indices: list[str] = []
        predictions: list[str] = []
        for row in reader:
            indices.append(str(row["index"]))
            predictions.append(str(row["prediction"] or ""))
    return indices, predictions


def _resolve_prediction_path(
    raw_path: object,
    *,
    repository_root: Path,
    status_path: Path,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"status lacks prediction_file: {status_path}")
    candidate = Path(raw_path)
    candidates = (
        candidate,
        repository_root / candidate,
        status_path.parent / candidate,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"prediction file does not exist: {raw_path}")


def _load_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"status is not a JSON object: {path}")
    return payload


def _status_matches_phase(payload: Mapping[str, Any], phase: str) -> bool:
    mode = payload.get("mode")
    if phase == "infer":
        return mode == "infer"
    return mode in {"eval", "all"}


def _latest_status(
    dataset_root: Path,
    *,
    dataset_name: str,
    phase: Literal["infer", "eval"],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    status_paths = sorted(
        (dataset_root / COREDEV_BASELINE_MODEL).glob("T*/status.json"), reverse=True
    )
    for status_path in status_paths:
        payload = _load_status(status_path)
        if not _status_matches_phase(payload, phase):
            continue
        datasets = payload.get("datasets")
        entry = datasets.get(dataset_name) if isinstance(datasets, Mapping) else None
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"latest {phase} status lacks {dataset_name}: {status_path}"
            )
        if entry.get("status") != "done":
            raise RuntimeError(
                f"latest {phase} run is incomplete for {dataset_name}: "
                f"status={entry.get('status')}"
            )
        if entry.get("error_message"):
            raise RuntimeError(
                f"latest {phase} run failed for {dataset_name}: "
                f"{entry['error_message']}"
            )
        if phase == "infer" and entry.get("skip_reason") != "mode_infer":
            raise RuntimeError(
                f"latest inference did not complete for {dataset_name}: "
                f"skip={entry.get('skip_reason')}"
            )
        if phase == "eval" and entry.get("skip_reason"):
            raise RuntimeError(
                f"latest evaluation was skipped for {dataset_name}: "
                f"{entry.get('skip_reason')}"
            )
        return status_path, payload, entry
    raise RuntimeError(f"no {phase} status for {dataset_name}")


def _validate_health_record(
    path: Path,
    *,
    expected_model: str,
    expected_base_url: str | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing judge health record: {path}")
    payload = _load_status(path)
    if payload.get("status") != "pass" or payload.get("model") != expected_model:
        raise RuntimeError(f"invalid judge health record: {path}")
    if expected_base_url is not None and payload.get("base_url") != expected_base_url:
        raise RuntimeError(f"judge health endpoint mismatch: {path}")
    return payload


def _judge_artifacts(run_dir: Path, prediction_path: Path) -> list[Path]:
    candidates = []
    for path in run_dir.iterdir():
        if not path.is_file() or path.resolve() == prediction_path:
            continue
        if path.name in {"status.json", "status.json.lock"}:
            continue
        if path.name.endswith(("_checkpoint.pkl", "_PREV.pkl", "_structs.pkl")):
            continue
        if (
            COREDEV_LLM_JUDGE_MODEL in path.name
            or "_result" in path.stem
            or "_extract" in path.stem
        ):
            candidates.append(path)
    return sorted(candidates)


def _reject_judge_failure_markers(paths: Iterable[Path]) -> None:
    encoded = tuple(marker.encode("utf-8") for marker in JUDGE_FAILURE_MARKERS)
    for path in paths:
        payload = path.read_bytes()
        matches = [
            marker.decode("utf-8") for marker in encoded if marker in payload
        ]
        if matches:
            raise RuntimeError(
                f"judge fallback/failure marker in {path.name}: {matches[0]}"
            )


def summarize_coredev_results(
    *,
    work_dir: Path,
    repository_root: Path,
    phase: Literal["infer", "eval"],
    datasets: tuple[str, ...] | None = None,
    expected_judge_base_url: str | None = None,
) -> dict[str, Any]:
    """Validate statuses, exact row counts, judge evidence, and aggregate metrics."""

    if not work_dir.is_absolute() or not repository_root.is_absolute():
        raise ValueError("work_dir and repository_root must be absolute")
    canonical = tuple(spec.vlmeval_dataset for spec in COREDEV_2511.slices)
    selected = canonical if datasets is None else datasets
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("CoreDev result datasets must be non-empty and unique")
    if any(name not in canonical for name in selected):
        raise ValueError("CoreDev results contain an unknown dataset")
    if tuple(name for name in canonical if name in selected) != selected:
        raise ValueError("CoreDev result datasets must follow canonical order")

    expected_counts = {
        spec.vlmeval_dataset: spec.sample_count for spec in COREDEV_2511.slices
    }
    slices = []
    for dataset_name in selected:
        nested_root = work_dir / dataset_name
        if (nested_root / COREDEV_BASELINE_MODEL).is_dir():
            dataset_root = nested_root
        else:
            dataset_root = work_dir
        status_path, status, entry = _latest_status(
            dataset_root,
            dataset_name=dataset_name,
            phase=phase,
        )
        if status.get("model_name") != COREDEV_BASELINE_MODEL:
            raise RuntimeError(f"evaluated model identity mismatch: {dataset_name}")
        commit = status.get("commit")
        if not isinstance(commit, str) or not VLMEVALKIT_REVIEW_COMMIT.startswith(commit):
            raise RuntimeError(f"VLMEvalKit commit mismatch: {dataset_name}")
        prediction_path = _resolve_prediction_path(
            entry.get("prediction_file"),
            repository_root=repository_root,
            status_path=status_path,
        )
        indices, predictions = _read_tsv(prediction_path)
        expected_count = expected_counts[dataset_name]
        if len(indices) != expected_count or len(set(indices)) != expected_count:
            raise RuntimeError(f"prediction row identity mismatch: {dataset_name}")
        if any(not prediction.strip() for prediction in predictions):
            raise RuntimeError(f"empty prediction in {dataset_name}")

        judge_contract = COREDEV_JUDGE_CONTRACTS[dataset_name]
        metrics = entry.get("metrics")
        judge_artifacts: list[Path] = []
        if phase == "eval":
            if not isinstance(metrics, Mapping) or not metrics:
                raise RuntimeError(f"evaluation metrics are missing: {dataset_name}")
            if judge_contract != "none_rule_based":
                if entry.get("judge_model") != COREDEV_LLM_JUDGE_MODEL:
                    raise RuntimeError(f"judge model identity mismatch: {dataset_name}")
                _validate_health_record(
                    dataset_root / "judge-health-pre.json",
                    expected_model=COREDEV_LLM_JUDGE_MODEL,
                    expected_base_url=expected_judge_base_url,
                )
                _validate_health_record(
                    dataset_root / "judge-health-post.json",
                    expected_model=COREDEV_LLM_JUDGE_MODEL,
                    expected_base_url=expected_judge_base_url,
                )
                judge_artifacts = _judge_artifacts(status_path.parent, prediction_path)
                if not judge_artifacts:
                    raise RuntimeError(f"judge evidence is missing: {dataset_name}")
                _reject_judge_failure_markers(judge_artifacts)

        slices.append(
            {
                "dataset": dataset_name,
                "sample_count": expected_count,
                "judge_contract": judge_contract,
                "status_path": str(status_path),
                "eval_id": status.get("eval_id"),
                "prediction_file": str(prediction_path),
                "prediction_sha256": sha256(prediction_path.read_bytes()).hexdigest(),
                "judge_model": entry.get("judge_model"),
                "judge_artifacts": [str(path) for path in judge_artifacts],
                "primary_metric": entry.get("primary_metric"),
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            }
        )

    return {
        "schema_version": 1,
        "suite": "coredev-2511-vlmevalkit-7055d301-v1",
        "phase": phase,
        "status": "pass",
        "model": COREDEV_BASELINE_MODEL,
        "vlmevalkit_commit": VLMEVALKIT_REVIEW_COMMIT,
        "sample_count": sum(item["sample_count"] for item in slices),
        "slice_count": len(slices),
        "slices": slices,
    }
