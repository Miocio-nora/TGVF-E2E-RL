"""Fail-closed judge and result acceptance for the seven CoreDev slices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from hashlib import sha256
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
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
RANDOM_JUDGE_FALLBACK_MARKER = "Failed to predict, thus randomly generate one"
DETERMINISTIC_JUDGE_PARSE_FAILURE_MARKER = (
    "TGVF deterministic judge parse failure; scored incorrect"
)
MAX_JUDGE_PARSE_FAILURE_RATE = 0.05
MIN_JUDGE_PARSE_FAILURE_LIMIT = 3
COREDEV_MACRO_STAR_SCHEMA = "tgvf.coredev-2511-macro-star.v1"
COREDEV_ACCEPTED_SUITE = "coredev-2511-vlmevalkit-7055d301-v1"
COREDEV_MACRO_STAR_COMPONENTS = (
    "vstar",
    "hr_average_all",
    "blink_single_180",
    "ocr_mean",
    "mmmu_single_269",
    "mathvista",
    "mathverse_five_version_macro",
)
_SINGLE_IMAGE_HEADLINE_COUNTS = {
    "BLINK": 180,
    "MMMU_Pro_10c": 269,
}
_MATHVERSE_VERSIONS = (
    "Text Dominant",
    "Vision Only",
    "Text Lite",
    "Vision Intensive",
    "Vision Dominant",
)
JUDGE_FAILURE_MARKERS = (
    "Failed to obtain answer via API",
    "Failed in Prefetch, no GPT-based answer matching",
    RANDOM_JUDGE_FALLBACK_MARKER,
    "All 5 retries failed",
)
_COREDEV_VERBOSE_OPTION = re.compile(
    r"(?i)(?:correct\s+)?(?:answer|output)\s+is\s+\**([A-Z])\**"
)
_MATHVERSE_SCORE_PROMPT_PREFIX = (
    "Below are two answers to a math question. Question is [Question], "
    "[Standard Answer] is the standard answer to the question, and [Model_answer] "
    "is the answer extracted from a model's output to this question."
)
_MATHVERSE_SCORE_PROMPT_SUFFIX = "Judgement:"
_MATHVERSE_LEADING_VERDICT = re.compile(r"\A\s*([01])(?=\s|\Z)")


def _is_mathverse_score_prompt(prompt: object) -> bool:
    return (
        isinstance(prompt, str)
        and prompt.lstrip().startswith(_MATHVERSE_SCORE_PROMPT_PREFIX)
        and prompt.rstrip().endswith(_MATHVERSE_SCORE_PROMPT_SUFFIX)
    )


def _canonicalize_mathverse_score_response(
    prompt: object,
    response: str,
) -> str:
    if not _is_mathverse_score_prompt(prompt):
        return response
    match = _MATHVERSE_LEADING_VERDICT.match(response)
    return response if match is None else match.group(1)


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
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        prompt = args[0] if args else kwargs.get("prompt")
        return _canonicalize_mathverse_score_response(prompt, result)


def install_fail_closed_judge_builders(modules: Iterable[Any]) -> None:
    """Wrap VLMEvalKit module-local ``build_judge`` bindings in place."""

    for module in modules:
        original = getattr(module, "build_judge")
        if getattr(original, "_tgvf_fail_closed", False):
            continue

        @wraps(original)
        def build_fail_closed(
            *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            return FailClosedJudge(_original(*args, **kwargs))

        build_fail_closed._tgvf_fail_closed = True  # type: ignore[attr-defined]
        module.build_judge = build_fail_closed


class _DeterministicChoiceProxy:
    """Replace ValKit's MCQ random fallback with an invalid sentinel."""

    __slots__ = ("_delegate",)

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def choice(self, _population: object) -> str:
        return "Z"


def install_deterministic_mcq_answer_policy(module: Any) -> None:
    """Parse A--J responses and score exhausted matching as incorrect.

    The pinned ValKit matcher randomly chooses an option after three failed
    parses. CoreDev instead emits ``Z``, which cannot match an A--J gold label,
    and preserves a dedicated row-local audit marker.
    """

    original_can_infer = getattr(module, "can_infer")
    if not getattr(original_can_infer, "_tgvf_coredev_answer_policy", False):

        @wraps(original_can_infer)
        def can_infer_coredev(answer: object, choices: Mapping[str, object]) -> Any:
            inferred = original_can_infer(answer, choices)
            if inferred:
                return inferred
            match = _COREDEV_VERBOSE_OPTION.search(str(answer))
            if match is None:
                return False
            candidate = match.group(1).upper()
            return candidate if candidate in choices or candidate == "Z" else False

        can_infer_coredev._tgvf_coredev_answer_policy = True  # type: ignore[attr-defined]
        module.can_infer = can_infer_coredev

    if not isinstance(getattr(module, "rd"), _DeterministicChoiceProxy):
        module.rd = _DeterministicChoiceProxy(module.rd)

    original_extract = getattr(module, "extract_answer_from_item")
    if getattr(original_extract, "_tgvf_coredev_answer_policy", False):
        return

    @wraps(original_extract)
    def extract_answer_coredev(*args: Any, **kwargs: Any) -> Any:
        result = original_extract(*args, **kwargs)
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "ValKit MCQ answer matcher returned a non-mapping result"
            )
        if RANDOM_JUDGE_FALLBACK_MARKER not in str(result.get("log", "")):
            return result
        if result.get("opt") != "Z":
            raise RuntimeError(
                "CoreDev deterministic MCQ fallback was not bound to invalid option Z"
            )
        deterministic = dict(result)
        deterministic["opt"] = "Z"
        deterministic["log"] = DETERMINISTIC_JUDGE_PARSE_FAILURE_MARKER
        return deterministic

    extract_answer_coredev._tgvf_coredev_answer_policy = True  # type: ignore[attr-defined]
    module.extract_answer_from_item = extract_answer_coredev


def _read_tsv(path: Path) -> tuple[list[str], list[str]]:
    # ``csv`` defaults to an arbitrary 128 KiB field ceiling.  A valid policy
    # response (or its structured trajectory metadata) can exceed that ceiling,
    # while still being bounded by the already-materialized TSV artifact.  Raise
    # the parser ceiling only for this read and restore the process-global value
    # afterwards so unrelated CSV consumers retain their own contract.
    previous_field_limit = csv.field_size_limit()
    artifact_bound = max(previous_field_limit, path.stat().st_size)
    field_limit_changed = artifact_bound != previous_field_limit
    if field_limit_changed:
        csv.field_size_limit(artifact_bound)
    try:
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
    finally:
        if field_limit_changed:
            csv.field_size_limit(previous_field_limit)


def _read_dict_rows(
    path: Path, *, delimiter: str
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read one scorer table without imposing Python's small CSV field cap."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"CoreDev metric artifact is not a regular file: {path}")
    previous_field_limit = csv.field_size_limit()
    artifact_bound = max(previous_field_limit, path.stat().st_size)
    field_limit_changed = artifact_bound != previous_field_limit
    if field_limit_changed:
        csv.field_size_limit(artifact_bound)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise RuntimeError(f"CoreDev metric artifact has no header: {path}")
            fields = tuple(reader.fieldnames)
            if any(not field.strip() for field in fields) or len(set(fields)) != len(
                fields
            ):
                raise RuntimeError(
                    f"CoreDev metric artifact has blank/duplicate headers: {path}"
                )
            rows = []
            for row in reader:
                if None in row or any(row.get(field) is None for field in fields):
                    raise RuntimeError(
                        f"CoreDev metric artifact has a ragged row: {path}"
                    )
                rows.append(dict(row))
        return fields, rows
    finally:
        if field_limit_changed:
            csv.field_size_limit(previous_field_limit)


def _finite_decimal(value: object, *, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RuntimeError(f"CoreDev headline {name} is not numeric") from error
    if not result.is_finite():
        raise RuntimeError(f"CoreDev headline {name} is not finite")
    return result


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("CoreDev headline JSON artifact has duplicate keys")
        result[key] = value
    return result


def _one_metric_artifact(run_dir: Path, pattern: str) -> Path:
    matches = tuple(sorted(run_dir.glob(pattern)))
    if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
        raise RuntimeError(
            f"CoreDev headline requires exactly one {pattern} artifact in {run_dir}"
        )
    return matches[0]


def _slice_run_dirs(
    summary: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, frozenset[str]]]:
    if (
        summary.get("schema_version") != 1
        or summary.get("suite") != COREDEV_ACCEPTED_SUITE
        or summary.get("status") != "pass"
        or summary.get("phase") != "eval"
        or summary.get("vlmevalkit_commit") != VLMEVALKIT_REVIEW_COMMIT
        or summary.get("sample_count") != COREDEV_2511.sample_count
        or summary.get("slice_count") != len(COREDEV_2511.slices)
    ):
        raise RuntimeError("CoreDev headline requires an accepted eval summary")
    slices = summary.get("slices")
    if not isinstance(slices, list) or len(slices) != len(COREDEV_2511.slices):
        raise RuntimeError("CoreDev headline summary lacks slices")
    expected = tuple(spec.vlmeval_dataset for spec in COREDEV_2511.slices)
    if (
        tuple(item.get("dataset") for item in slices if isinstance(item, Mapping))
        != expected
    ):
        raise RuntimeError("CoreDev headline slice order differs")
    result: dict[str, Path] = {}
    expected_indices: dict[str, frozenset[str]] = {}
    status_paths: set[Path] = set()
    for item, spec in zip(slices, COREDEV_2511.slices, strict=True):
        assert isinstance(item, Mapping)
        if (
            item.get("sample_count") != spec.sample_count
            or item.get("judge_contract")
            != COREDEV_JUDGE_CONTRACTS[spec.vlmeval_dataset]
        ):
            raise RuntimeError(
                f"CoreDev headline {spec.vlmeval_dataset} slice identity differs"
            )
        status_path = Path(str(item.get("status_path", "")))
        if status_path.name != "status.json" or not status_path.is_file():
            raise RuntimeError(
                f"CoreDev headline status artifact is absent: {status_path}"
            )
        status_path = status_path.resolve()
        if status_path in status_paths:
            raise RuntimeError("CoreDev headline status paths are not unique")
        status_paths.add(status_path)
        prediction_path = Path(str(item.get("prediction_file", "")))
        if prediction_path.is_symlink() or not prediction_path.is_file():
            raise RuntimeError(
                f"CoreDev headline prediction artifact is absent: {prediction_path}"
            )
        prediction_indices, _ = _read_tsv(prediction_path)
        if (
            len(prediction_indices) != spec.sample_count
            or any(not index.strip() for index in prediction_indices)
            or len(set(prediction_indices)) != spec.sample_count
        ):
            raise RuntimeError(
                f"CoreDev headline {spec.vlmeval_dataset} prediction identity differs"
            )
        result[str(item["dataset"])] = status_path.parent
        expected_indices[spec.vlmeval_dataset] = frozenset(prediction_indices)
    return result, expected_indices


def _fraction_csv_value(
    path: Path,
    *,
    row_selector: Mapping[str, str],
    value_column: str,
    name: str,
) -> Decimal:
    fields, rows = _read_dict_rows(path, delimiter=",")
    required = {*row_selector, value_column}
    if not required.issubset(fields):
        raise RuntimeError(f"CoreDev headline {name} CSV schema differs")
    selected = [
        row
        for row in rows
        if all(str(row.get(key, "")) == value for key, value in row_selector.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"CoreDev headline {name} row is not unique")
    fraction = _finite_decimal(selected[0][value_column], name=name)
    if not Decimal(0) <= fraction <= Decimal(1):
        raise RuntimeError(f"CoreDev headline {name} fraction lies outside [0,1]")
    return fraction * Decimal(100)


def _single_image_mcq_accuracy(
    path: Path, *, dataset: str, expected_indices: frozenset[str]
) -> tuple[Decimal, dict[str, int]]:
    fields, rows = _read_dict_rows(path, delimiter="\t")
    if not {"index", "extra_records", "hit"}.issubset(fields):
        raise RuntimeError(f"CoreDev headline {dataset} result schema differs")
    expected_total = next(
        spec.sample_count
        for spec in COREDEV_2511.slices
        if spec.vlmeval_dataset == dataset
    )
    observed_indices = tuple(str(row["index"]).strip() for row in rows)
    if (
        len(rows) != expected_total
        or any(not index for index in observed_indices)
        or len(set(observed_indices)) != expected_total
        or frozenset(observed_indices) != expected_indices
    ):
        raise RuntimeError(f"CoreDev headline {dataset} result coverage differs")
    single_hits: list[int] = []
    coverage_counts: dict[str, int] = {}
    for row in rows:
        try:
            extra = json.loads(
                row["extra_records"], object_pairs_hook=_reject_duplicate_json_pairs
            )
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(
                f"CoreDev headline {dataset} coverage metadata is malformed"
            ) from error
        if not isinstance(extra, Mapping):
            raise RuntimeError(
                f"CoreDev headline {dataset} coverage metadata is not an object"
            )
        coverage = extra.get("coverage")
        if coverage not in {
            "single_image_evaluated",
            "single_image_policy_output_contract_failure",
            "unsupported_multi_image",
        }:
            raise RuntimeError(f"CoreDev headline {dataset} coverage label differs")
        coverage_counts[str(coverage)] = coverage_counts.get(str(coverage), 0) + 1
        hit = _finite_decimal(row["hit"], name=f"{dataset} hit")
        if hit not in {Decimal(0), Decimal(1)}:
            raise RuntimeError(f"CoreDev headline {dataset} hit is not binary")
        if coverage == "unsupported_multi_image":
            if hit != 0:
                raise RuntimeError(
                    f"CoreDev headline {dataset} unsupported sample scored correct"
                )
            continue
        if coverage == "single_image_policy_output_contract_failure" and hit != 0:
            raise RuntimeError(
                f"CoreDev headline {dataset} contract failure scored correct"
            )
        single_hits.append(int(hit))
    expected_single = _SINGLE_IMAGE_HEADLINE_COUNTS[dataset]
    if len(single_hits) != expected_single:
        raise RuntimeError(
            f"CoreDev headline {dataset} single-image coverage differs: "
            f"{len(single_hits)} != {expected_single}"
        )
    if sum(coverage_counts.values()) != expected_total:
        raise RuntimeError(f"CoreDev headline {dataset} coverage total differs")
    correct = sum(single_hits)
    return (
        Decimal(100) * Decimal(correct) / Decimal(expected_single),
        {"correct": correct, "count": expected_single},
    )


def extract_coredev_macro_star(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the frozen seven-component CoreDev headline from scorer files.

    This intentionally does not consume ``status.primary_metric``.  In
    particular, HRBench4K's status primary is cycle 0, while the frozen
    headline is the ``cycle=Average,type=all`` row.  BLINK and MMMU are scored
    over their explicitly evaluated single-image populations only.
    """

    run_dirs, expected_indices = _slice_run_dirs(summary)
    model = summary.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("CoreDev headline model identity is absent")

    vstar_path = _one_metric_artifact(
        run_dirs["VStarBench"], f"{model}_VStarBench_acc.csv"
    )
    vstar_fields, vstar_rows = _read_dict_rows(vstar_path, delimiter=",")
    if len(vstar_rows) != 1 or set(vstar_fields) != {
        "split",
        "Overall",
        "direct_attributes",
        "relative_position",
    }:
        raise RuntimeError("CoreDev headline VStarBench score schema differs")
    vstar = _fraction_csv_value(
        vstar_path,
        row_selector={"split": "none"},
        value_column="Overall",
        name="VStarBench Overall",
    )
    hr_path = _one_metric_artifact(run_dirs["HRBench4K"], f"{model}_HRBench4K_acc.csv")
    hr_fields, hr_rows = _read_dict_rows(hr_path, delimiter=",")
    expected_hr_selectors = {
        (cycle, row_type)
        for cycle in ("0", "1", "2", "3", "Average")
        for row_type in ("all", "cross", "single")
    }
    if (
        set(hr_fields) != {"cycle", "type", "accuracy"}
        or {(row.get("cycle"), row.get("type")) for row in hr_rows}
        != expected_hr_selectors
        or len(hr_rows) != len(expected_hr_selectors)
    ):
        raise RuntimeError("CoreDev headline HRBench4K score rows differ")
    hr = _fraction_csv_value(
        hr_path,
        row_selector={"cycle": "Average", "type": "all"},
        value_column="accuracy",
        name="HRBench4K Average/all",
    )
    blink, blink_counts = _single_image_mcq_accuracy(
        _one_metric_artifact(run_dirs["BLINK"], f"{model}_BLINK_*_result.tsv"),
        dataset="BLINK",
        expected_indices=expected_indices["BLINK"],
    )

    ocr_path = _one_metric_artifact(
        run_dirs["OCRBench_v2"], f"{model}_OCRBench_v2_score.json"
    )
    ocr = json.loads(
        ocr_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_pairs,
    )
    if not isinstance(ocr, Mapping):
        raise RuntimeError("CoreDev headline OCR score is not an object")
    ocr_english = Decimal(100) * _finite_decimal(
        ocr.get("English Overall Score"), name="OCR English Overall"
    )
    ocr_chinese = Decimal(100) * _finite_decimal(
        ocr.get("Chinese Overall Score"), name="OCR Chinese Overall"
    )
    if any(
        not Decimal(0) <= value <= Decimal(100) for value in (ocr_english, ocr_chinese)
    ):
        raise RuntimeError("CoreDev headline OCR score lies outside [0,100]")
    ocr_mean = (ocr_english + ocr_chinese) / Decimal(2)

    mmmu, mmmu_counts = _single_image_mcq_accuracy(
        _one_metric_artifact(
            run_dirs["MMMU_Pro_10c"], f"{model}_MMMU_Pro_10c_*_result.tsv"
        ),
        dataset="MMMU_Pro_10c",
        expected_indices=expected_indices["MMMU_Pro_10c"],
    )

    mathvista_path = _one_metric_artifact(
        run_dirs["MathVista_MINI"], f"{model}_MathVista_MINI_*_score.csv"
    )
    mathvista_fields, mathvista_rows = _read_dict_rows(mathvista_path, delimiter=",")
    if not {"Task&Skill", "acc"}.issubset(mathvista_fields):
        raise RuntimeError("CoreDev headline MathVista score schema differs")
    expected_mathvista_rows = {
        "Overall",
        "geometry reasoning",
        "scientific reasoning",
        "textbook question answering",
        "algebraic reasoning",
        "statistical reasoning",
        "figure question answering",
        "numeric commonsense",
        "arithmetic reasoning",
        "visual question answering",
        "geometry problem solving",
        "math word problem",
        "logical reasoning",
    }
    if (
        len(mathvista_rows) != len(expected_mathvista_rows)
        or {row.get("Task&Skill") for row in mathvista_rows} != expected_mathvista_rows
    ):
        raise RuntimeError("CoreDev headline MathVista score rows differ")
    mathvista_selected = [
        row for row in mathvista_rows if row.get("Task&Skill") == "Overall"
    ]
    if len(mathvista_selected) != 1:
        raise RuntimeError("CoreDev headline MathVista Overall row is not unique")
    mathvista = _finite_decimal(mathvista_selected[0]["acc"], name="MathVista Overall")
    if not Decimal(0) <= mathvista <= Decimal(100):
        raise RuntimeError("CoreDev headline MathVista score lies outside [0,100]")

    mathverse_path = _one_metric_artifact(
        run_dirs["MathVerse_MINI"], f"{model}_MathVerse_MINI_*_score.csv"
    )
    mathverse_fields, mathverse_rows = _read_dict_rows(mathverse_path, delimiter=",")
    if not {"split", "Overall"}.issubset(mathverse_fields):
        raise RuntimeError("CoreDev headline MathVerse score schema differs")
    by_split = {row.get("split"): row for row in mathverse_rows}
    if len(by_split) != len(mathverse_rows) or set(by_split) != set(
        _MATHVERSE_VERSIONS
    ):
        raise RuntimeError("CoreDev headline MathVerse five-version rows differ")
    mathverse_values = [
        _finite_decimal(by_split[version]["Overall"], name=f"MathVerse {version}")
        for version in _MATHVERSE_VERSIONS
    ]
    if any(not Decimal(0) <= value <= Decimal(100) for value in mathverse_values):
        raise RuntimeError("CoreDev headline MathVerse score lies outside [0,100]")
    mathverse = sum(mathverse_values, Decimal(0)) / Decimal(len(mathverse_values))

    components = {
        "vstar": vstar,
        "hr_average_all": hr,
        "blink_single_180": blink,
        "ocr_mean": ocr_mean,
        "mmmu_single_269": mmmu,
        "mathvista": mathvista,
        "mathverse_five_version_macro": mathverse,
    }
    if tuple(components) != COREDEV_MACRO_STAR_COMPONENTS:
        raise AssertionError("CoreDev Macro* component order changed")
    return {
        "schema_version": COREDEV_MACRO_STAR_SCHEMA,
        "aggregation": "unweighted_mean_of_seven_percent_components",
        "components_percent": {key: float(value) for key, value in components.items()},
        "ocr_language_components_percent": {
            "english": float(ocr_english),
            "chinese": float(ocr_chinese),
        },
        "mathverse_version_components_percent": {
            version: float(value)
            for version, value in zip(
                _MATHVERSE_VERSIONS, mathverse_values, strict=True
            )
        },
        "single_image_counts": {
            "blink": blink_counts,
            "mmmu": mmmu_counts,
        },
        "macro_star_percent": float(
            sum(components.values(), Decimal(0)) / Decimal(len(components))
        ),
    }


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
    expected_model: str,
    expected_eval_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if expected_eval_id is not None:
        if (
            not isinstance(expected_eval_id, str)
            or not expected_eval_id
            or Path(expected_eval_id).name != expected_eval_id
        ):
            raise ValueError("expected_eval_id must be one safe run-directory name")
        status_paths = [
            dataset_root / expected_model / expected_eval_id / "status.json"
        ]
    else:
        status_paths = sorted(
            (dataset_root / expected_model).glob("T*/status.json"), reverse=True
        )
    for status_path in status_paths:
        if not status_path.is_file() or status_path.is_symlink():
            continue
        payload = _load_status(status_path)
        if expected_eval_id is not None and payload.get("eval_id") != expected_eval_id:
            raise RuntimeError(
                f"exact {phase} status eval_id differs for {dataset_name}: "
                f"{status_path}"
            )
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
        matches = [marker.decode("utf-8") for marker in encoded if marker in payload]
        if matches:
            raise RuntimeError(
                f"judge fallback/failure marker in {path.name}: {matches[0]}"
            )


def _deterministic_judge_parse_failures(
    paths: Iterable[Path],
    *,
    expected_indices: Iterable[str],
    require_mcq_result: bool,
) -> list[str]:
    """Audit result coverage and return deterministic parse-failure IDs."""

    marker = DETERMINISTIC_JUDGE_PARSE_FAILURE_MARKER
    marker_bytes = marker.encode("utf-8")
    marker_present = False
    sample_ids: set[str] = set()
    expected = tuple(str(index) for index in expected_indices)
    expected_set = set(expected)
    structured_result_count = 0
    for path in paths:
        payload = path.read_bytes()
        marker_present = marker_present or marker_bytes in payload
        if path.suffix.lower() != ".tsv":
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or ())
            if not {"index", "hit", "log"}.issubset(fields):
                continue
            structured_result_count += 1
            rows = list(reader)
            observed = [str(row.get("index") or "").strip() for row in rows]
            if (
                len(observed) != len(expected)
                or len(set(observed)) != len(expected)
                or set(observed) != expected_set
            ):
                raise RuntimeError(f"judge result row identity mismatch: {path}")
            for row in rows:
                if marker not in str(row.get("log") or ""):
                    continue
                if str(row.get("hit", "")).strip() not in {
                    "0",
                    "0.0",
                    "False",
                    "false",
                }:
                    raise RuntimeError(
                        "deterministic judge parse failure was not scored incorrect: "
                        f"{path}"
                    )
                sample_id = str(row.get("index") or "").strip()
                if not sample_id:
                    raise RuntimeError(
                        f"deterministic judge parse failure lacks sample id: {path}"
                    )
                sample_ids.add(sample_id)
    if require_mcq_result and structured_result_count != 1:
        raise RuntimeError(
            "CoreDev MCQ scoring requires exactly one complete structured result TSV"
        )
    if marker_present and not sample_ids:
        raise RuntimeError(
            "deterministic judge parse failure lacks structured TSV audit evidence"
        )
    return sorted(sample_ids)


def _judge_parse_failure_limit(sample_count: int) -> int:
    """Allow isolated bad rows while rejecting systemic scorer failure."""

    return max(
        MIN_JUDGE_PARSE_FAILURE_LIMIT,
        math.ceil(sample_count * MAX_JUDGE_PARSE_FAILURE_RATE),
    )


def summarize_coredev_results(
    *,
    work_dir: Path,
    repository_root: Path,
    phase: Literal["infer", "eval"],
    datasets: tuple[str, ...] | None = None,
    expected_judge_base_url: str | None = None,
    expected_model: str = COREDEV_BASELINE_MODEL,
    expected_eval_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate statuses, exact row counts, judge evidence, and aggregate metrics."""

    if not work_dir.is_absolute() or not repository_root.is_absolute():
        raise ValueError("work_dir and repository_root must be absolute")
    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ValueError("expected_model must be non-empty text")
    canonical = tuple(spec.vlmeval_dataset for spec in COREDEV_2511.slices)
    selected = canonical if datasets is None else datasets
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("CoreDev result datasets must be non-empty and unique")
    if any(name not in canonical for name in selected):
        raise ValueError("CoreDev results contain an unknown dataset")
    if tuple(name for name in canonical if name in selected) != selected:
        raise ValueError("CoreDev result datasets must follow canonical order")
    if expected_eval_ids is not None and set(expected_eval_ids) != set(selected):
        raise ValueError("expected_eval_ids must exactly bind every selected dataset")

    expected_counts = {
        spec.vlmeval_dataset: spec.sample_count for spec in COREDEV_2511.slices
    }
    slices = []
    for dataset_name in selected:
        nested_root = work_dir / dataset_name
        if (nested_root / expected_model).is_dir():
            dataset_root = nested_root
        else:
            dataset_root = work_dir
        status_path, status, entry = _latest_status(
            dataset_root,
            dataset_name=dataset_name,
            phase=phase,
            expected_model=expected_model,
            expected_eval_id=(
                None if expected_eval_ids is None else expected_eval_ids[dataset_name]
            ),
        )
        if status.get("model_name") != expected_model:
            raise RuntimeError(f"evaluated model identity mismatch: {dataset_name}")
        commit = status.get("commit")
        if not isinstance(commit, str) or not VLMEVALKIT_REVIEW_COMMIT.startswith(
            commit
        ):
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
        parse_failure_ids: list[str] = []
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
                parse_failure_ids = _deterministic_judge_parse_failures(
                    judge_artifacts,
                    expected_indices=indices,
                    require_mcq_result=(
                        judge_contract == "qwen2_5_72b_fallback_or_exact_matching"
                    ),
                )
                parse_failure_limit = _judge_parse_failure_limit(expected_count)
                if len(parse_failure_ids) > parse_failure_limit:
                    raise RuntimeError(
                        "systemic judge parse failure rate: "
                        f"{dataset_name} has {len(parse_failure_ids)}/{expected_count}, "
                        f"limit={parse_failure_limit}"
                    )

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
                "judge_parse_failure_count": len(parse_failure_ids),
                "judge_parse_failure_sample_ids": parse_failure_ids,
                "judge_parse_failure_limit": _judge_parse_failure_limit(expected_count),
                "primary_metric": entry.get("primary_metric"),
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            }
        )

    return {
        "schema_version": 1,
        "suite": "coredev-2511-vlmevalkit-7055d301-v1",
        "phase": phase,
        "status": "pass",
        "model": expected_model,
        "vlmevalkit_commit": VLMEVALKIT_REVIEW_COMMIT,
        "sample_count": sum(item["sample_count"] for item in slices),
        "slice_count": len(slices),
        "judge_parse_failure_policy": "deterministic_incorrect",
        "judge_parse_failure_rate_limit": MAX_JUDGE_PARSE_FAILURE_RATE,
        "judge_parse_failure_count": sum(
            item["judge_parse_failure_count"] for item in slices
        ),
        "slices": slices,
    }
