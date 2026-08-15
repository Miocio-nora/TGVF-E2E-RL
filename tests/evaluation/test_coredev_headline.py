from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import pytest

from tgvf_rl.evaluation.coredev_materialize import COREDEV_JUDGE_CONTRACTS
from tgvf_rl.evaluation.coredev_results import extract_coredev_macro_star
from tgvf_rl.evaluation.vlmevalkit import (
    COREDEV_2511,
    VLMEVALKIT_REVIEW_COMMIT,
)


_MODEL = "Qwen3-VL-8B-Instruct"
_JUDGE = "Qwen2.5-72B-Instruct"
_MATHVISTA_ROWS = (
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
)
_MATHVERSE_VALUES = {
    "Text Dominant": "10.004",
    "Text Lite": "20.004",
    "Vision Dominant": "30.004",
    "Vision Intensive": "40.004",
    "Vision Only": "50.004",
}
_OCR_KEYS = (
    "en_text_recognition",
    "en_text_detection",
    "en_text_spotting",
    "en_relationship_extraction",
    "en_element_parsing",
    "en_mathematical_calculation",
    "en_visual_text_understanding",
    "en_knowledge_reasoning",
    "cn_text_recognition",
    "cn_relationship_extraction",
    "cn_element_parsing",
    "cn_visual_text_understanding",
    "cn_knowledge_reasoning",
)
_LEGACY_PRL21_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8/evaluation/"
    "PRL21-R0-CROP-TFREE-COREDEV2511-STEP8-STEP16-TEMP1-SEED42-V1"
)


def _write_table(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    *,
    delimiter: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)


def _read_table(path: Path, *, delimiter: str) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    return rows[0], rows[1:]


def _extra(coverage: str) -> str:
    return json.dumps(
        {
            "schema_version": "tgvf-policy-coredev-scoring-view-v1",
            "evaluation_id": "TEST-COREDEV-HEADLINE-V1",
            "coverage": coverage,
        },
        separators=(",", ":"),
    )


def _write_prediction(path: Path, dataset: str, count: int) -> list[str]:
    indices = [f"{dataset}-{index:04d}" for index in range(count)]
    _write_table(
        path,
        ["index", "prediction"],
        [[index, "answer"] for index in indices],
        delimiter="\t",
    )
    return indices


def _accepted_summary_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths: dict[str, Path] = {}
    slices: list[dict[str, Any]] = []
    for spec in COREDEV_2511.slices:
        dataset = spec.vlmeval_dataset
        run_dir = tmp_path / dataset / _MODEL / "T20260815-000000"
        run_dir.mkdir(parents=True)
        prediction = run_dir / f"{_MODEL}_{dataset}.tsv"
        indices = _write_prediction(prediction, dataset, spec.sample_count)
        status_path = run_dir / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "eval_id": "T20260815-000000",
                    "datasets": {
                        dataset: {
                            "status": "done",
                            "prediction_file": prediction.name,
                            "judge_model": _JUDGE,
                            "metrics": {"fixture": 1},
                        }
                    },
                    "model_name": _MODEL,
                    "commit": VLMEVALKIT_REVIEW_COMMIT[:8],
                    "mode": "eval",
                    "reuse": True,
                    "reuse_aux": "infer",
                }
            ),
            encoding="utf-8",
        )
        paths[f"{dataset}.run_dir"] = run_dir
        paths[f"{dataset}.prediction"] = prediction

        judge_artifacts: list[str] = []
        if dataset == "VStarBench":
            score = run_dir / f"{_MODEL}_VStarBench_acc.csv"
            _write_table(
                score,
                ["split", "Overall", "direct_attributes", "relative_position"],
                [["none", "0.50125", "0.5", "0.5025"]],
                delimiter=",",
            )
            paths["vstar"] = score
        elif dataset == "HRBench4K":
            score = run_dir / f"{_MODEL}_HRBench4K_acc.csv"
            rows = []
            for cycle in ("0", "1", "2", "3", "Average"):
                for row_type in ("all", "cross", "single"):
                    accuracy = "0.5"
                    if (cycle, row_type) == ("0", "all"):
                        accuracy = "0.01125"
                    elif (cycle, row_type) == ("Average", "all"):
                        accuracy = "0.80125"
                    rows.append([cycle, row_type, accuracy])
            _write_table(score, ["cycle", "type", "accuracy"], rows, delimiter=",")
            paths["hr"] = score
        elif dataset == "BLINK":
            score = run_dir / f"{_MODEL}_BLINK_{_JUDGE}_result.tsv"
            rows = []
            for ordinal, index in enumerate(indices):
                is_single = ordinal < 180
                coverage = (
                    "single_image_evaluated" if is_single else "unsupported_multi_image"
                )
                hit = "1" if is_single and ordinal < 91 else "0"
                rows.append([index, _extra(coverage), hit, "Succeed"])
            _write_table(
                score,
                ["index", "extra_records", "hit", "log"],
                rows,
                delimiter="\t",
            )
            paths["blink"] = score
            judge_artifacts.append(str(score))
        elif dataset == "OCRBench_v2":
            score = run_dir / f"{_MODEL}_OCRBench_v2_score.json"
            payload = {key: 0 for key in _OCR_KEYS}
            payload["English Overall Score"] = 0.4995
            payload["Chinese Overall Score"] = 0.5007
            score.write_text(json.dumps(payload), encoding="utf-8")
            paths["ocr"] = score
        elif dataset == "MMMU_Pro_10c":
            score = run_dir / f"{_MODEL}_MMMU_Pro_10c_{_JUDGE}_result.tsv"
            rows = []
            for ordinal, index in enumerate(indices):
                is_single = ordinal < 269
                coverage = (
                    "single_image_evaluated" if is_single else "unsupported_multi_image"
                )
                hit = "1" if is_single and ordinal < 136 else "0"
                rows.append([index, _extra(coverage), hit, "Succeed"])
            _write_table(
                score,
                ["index", "extra_records", "hit", "log"],
                rows,
                delimiter="\t",
            )
            paths["mmmu"] = score
            judge_artifacts.append(str(score))
        elif dataset == "MathVista_MINI":
            score = run_dir / f"{_MODEL}_MathVista_MINI_{_JUDGE}_score.csv"
            rows = [
                [
                    name,
                    "300" if name == "Overall" else "1",
                    "0",
                    "0",
                    "0",
                    "70.005" if name == "Overall" else "0",
                ]
                for name in _MATHVISTA_ROWS
            ]
            _write_table(
                score,
                ["Task&Skill", "tot", "prefetch", "hit", "prefetch_rate", "acc"],
                rows,
                delimiter=",",
            )
            paths["mathvista"] = score
            judge_artifacts.append(str(score))
        elif dataset == "MathVerse_MINI":
            score = run_dir / f"{_MODEL}_MathVerse_MINI_{_JUDGE}_score.csv"
            _write_table(
                score,
                ["split", "Overall"],
                [[name, value] for name, value in _MATHVERSE_VALUES.items()],
                delimiter=",",
            )
            paths["mathverse"] = score
            judge_artifacts.append(str(score))
        else:  # pragma: no cover - guarded by the canonical CoreDev contract
            raise AssertionError(dataset)

        slices.append(
            {
                "dataset": dataset,
                "sample_count": spec.sample_count,
                "judge_contract": COREDEV_JUDGE_CONTRACTS[dataset],
                "status_path": str(status_path),
                "prediction_file": str(prediction),
                "judge_model": _JUDGE,
                "judge_artifacts": judge_artifacts,
                "metrics": {"split=none|Overall": 0.99},
            }
        )

    return (
        {
            "schema_version": 1,
            "suite": "coredev-2511-vlmevalkit-7055d301-v1",
            "phase": "eval",
            "status": "pass",
            "model": _MODEL,
            "vlmevalkit_commit": VLMEVALKIT_REVIEW_COMMIT,
            "sample_count": 2511,
            "slice_count": 7,
            "slices": slices,
        },
        paths,
    )


def test_frozen_headline_uses_average_supported_sets_and_unrounded_macro(
    tmp_path: Path,
) -> None:
    summary, _ = _accepted_summary_fixture(tmp_path)

    result = extract_coredev_macro_star(summary)

    expected_components = {
        "vstar": Decimal("50.125"),
        "hr_average_all": Decimal("80.125"),
        "blink_single_180": Decimal(91) * 100 / Decimal(180),
        "ocr_mean": (Decimal("49.95") + Decimal("50.07")) / 2,
        "mmmu_single_269": Decimal(136) * 100 / Decimal(269),
        "mathvista": Decimal("70.005"),
        "mathverse_five_version_macro": Decimal("30.004"),
    }
    expected_macro = sum(expected_components.values()) / Decimal(7)
    assert result["components_percent"] == pytest.approx(
        {key: float(value) for key, value in expected_components.items()}
    )
    assert result["components_percent"]["hr_average_all"] != pytest.approx(1.125)
    assert result["components_percent"]["blink_single_180"] != pytest.approx(99.0)
    assert result["components_percent"]["mmmu_single_269"] != pytest.approx(99.0)
    assert result["ocr_language_components_percent"] == pytest.approx(
        {"english": 49.95, "chinese": 50.07}
    )
    assert result["macro_star_percent"] == pytest.approx(float(expected_macro))

    early_rounded_macro = sum(
        value.quantize(Decimal("0.01")) for value in expected_components.values()
    ) / Decimal(7)
    assert early_rounded_macro != expected_macro
    assert f"{result['macro_star_percent']:.2f}" == f"{expected_macro:.2f}"


def test_policy_output_failure_stays_in_fixed_single_image_denominator(
    tmp_path: Path,
) -> None:
    summary, paths = _accepted_summary_fixture(tmp_path)
    header, rows = _read_table(paths["blink"], delimiter="\t")
    rows[0][header.index("extra_records")] = _extra(
        "single_image_policy_output_contract_failure"
    )
    rows[0][header.index("hit")] = "0"
    _write_table(paths["blink"], header, rows, delimiter="\t")

    result = extract_coredev_macro_star(summary)

    assert result["components_percent"]["blink_single_180"] == pytest.approx(50.0)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_row",
        "blank_index",
        "duplicate_index",
        "missing_expected_index",
        "nonnumeric_hit",
        "nonfinite_hit",
        "nonbinary_hit",
        "unknown_coverage",
        "duplicate_coverage_field",
        "unsupported_hit",
    ),
)
def test_single_image_result_artifact_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    summary, paths = _accepted_summary_fixture(tmp_path)
    path = paths["blink"]
    header, rows = _read_table(path, delimiter="\t")
    index_column = header.index("index")
    hit_column = header.index("hit")
    extra_column = header.index("extra_records")
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "blank_index":
        rows[0][index_column] = ""
    elif mutation == "duplicate_index":
        rows[0][index_column] = rows[1][index_column]
    elif mutation == "missing_expected_index":
        rows[0][index_column] = "BLINK-unexpected-but-unique"
    elif mutation == "nonnumeric_hit":
        rows[0][hit_column] = "wrong"
    elif mutation == "nonfinite_hit":
        rows[0][hit_column] = "NaN"
    elif mutation == "nonbinary_hit":
        rows[0][hit_column] = "2"
    elif mutation == "unknown_coverage":
        rows[0][extra_column] = _extra("single_image_guessed")
    elif mutation == "duplicate_coverage_field":
        rows[100][extra_column] = (
            '{"coverage":"single_image_evaluated","coverage":"single_image_evaluated"}'
        )
    elif mutation == "unsupported_hit":
        rows[-1][hit_column] = "1"
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(mutation)
    _write_table(path, header, rows, delimiter="\t")

    with pytest.raises(RuntimeError):
        extract_coredev_macro_star(summary)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    (
        ("vstar", "extra_row"),
        ("hr", "missing_row"),
        ("hr", "duplicate_selector"),
        ("mathvista", "missing_row"),
        ("mathverse", "duplicate_selector"),
        ("hr", "duplicate_header"),
        ("hr", "blank_header"),
        ("hr", "ragged_row"),
    ),
)
def test_score_tables_reject_row_and_field_ambiguity(
    tmp_path: Path,
    artifact: str,
    mutation: str,
) -> None:
    summary, paths = _accepted_summary_fixture(tmp_path)
    path = paths[artifact]
    header, rows = _read_table(path, delimiter=",")
    if mutation == "extra_row":
        rows.append(list(rows[0]))
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "duplicate_selector":
        rows[-1] = list(rows[0])
    elif mutation == "duplicate_header":
        header.append("accuracy")
        for row in rows:
            row.append(row[-1])
    elif mutation == "blank_header":
        header.append("")
        for row in rows:
            row.append("")
    elif mutation == "ragged_row":
        rows[0].pop()
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(mutation)
    _write_table(path, header, rows, delimiter=",")

    with pytest.raises(RuntimeError):
        extract_coredev_macro_star(summary)


def test_duplicate_ocr_json_field_is_ambiguous(tmp_path: Path) -> None:
    summary, paths = _accepted_summary_fixture(tmp_path)
    paths["ocr"].write_text(
        '{"English Overall Score":0.4,"English Overall Score":0.5,'
        '"Chinese Overall Score":0.6}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        extract_coredev_macro_star(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("suite", "coredev-lookalike"),
        ("sample_count", 2510),
        ("slice_count", 6),
        ("vlmevalkit_commit", "0" * 40),
    ),
)
def test_headline_requires_exact_accepted_summary_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    summary, _ = _accepted_summary_fixture(tmp_path)
    summary[field] = value

    with pytest.raises(RuntimeError):
        extract_coredev_macro_star(summary)


def test_headline_rejects_slice_count_identity_drift(tmp_path: Path) -> None:
    summary, _ = _accepted_summary_fixture(tmp_path)
    summary["slices"][2]["sample_count"] = 419

    with pytest.raises(RuntimeError):
        extract_coredev_macro_star(summary)


@pytest.mark.skipif(
    not _LEGACY_PRL21_ROOT.is_dir(),
    reason="legacy PRL21 golden artifacts are not installed",
)
@pytest.mark.parametrize(
    ("arm", "expected"),
    (("step8", 61.10317146456491), ("step16", 61.086167936836276)),
)
def test_legacy_prl21_summary_is_recomputed_in_place_without_identity_migration(
    arm: str, expected: float
) -> None:
    summary_path = (
        _LEGACY_PRL21_ROOT
        / arm
        / "scoring/coredev-official-v1-recovery2/coredev-2511-eval-summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = extract_coredev_macro_star(summary)

    assert result["macro_star_percent"] == pytest.approx(expected)
