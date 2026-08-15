from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("paired_scorer_receipts", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_MODEL = _MODULE.EVALUATED_MODEL
_SOURCE_RUN = "T20260815_Gdeadbeef"
_SOURCE_EVALUATION = "SEMANTIC-EVALUATION"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _materialize_receipts(scoring_root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for index, dataset in enumerate(_MODULE.COREDEV_DATASETS, 1):
        dataset_root = scoring_root / dataset
        source_run_dir = dataset_root / _MODEL / _SOURCE_RUN
        source_prediction = source_run_dir / f"{_MODEL}_{dataset}.tsv"
        source_prediction.parent.mkdir(parents=True)
        source_prediction.write_text(
            f"index\tprediction\n{index}\tanswer\n", encoding="utf-8"
        )
        source_manifest = source_run_dir / "final-answer-view-manifest.json"
        _write_json(
            source_manifest,
            {
                "derived": {
                    "path": str(source_prediction.resolve()),
                    "sha256": _sha(source_prediction),
                }
            },
        )
        _write_json(
            source_run_dir / "status.json",
            {
                "eval_id": _SOURCE_RUN,
                "mode": "infer",
                "reuse_aux": "infer",
                "datasets": {
                    dataset: {
                        "status": "done",
                        "source_run": _SOURCE_EVALUATION,
                        "prediction_file": str(source_prediction.resolve()),
                    }
                },
            },
        )

        destination_run_id = f"T20260815_G{index:08x}"
        destination_run_dir = dataset_root / _MODEL / destination_run_id
        destination_prediction = destination_run_dir / f"{_MODEL}_{dataset}.tsv"
        destination_prediction.parent.mkdir(parents=True)
        destination_prediction.write_bytes(source_prediction.read_bytes())
        destination_status = destination_run_dir / "status.json"
        _write_json(
            destination_status,
            {
                "eval_id": destination_run_id,
                "mode": "eval",
                "reuse": True,
                "reuse_aux": "infer",
                "datasets": {
                    dataset: {
                        "status": "done",
                        "source_run": _SOURCE_RUN,
                        "prediction_file": destination_prediction.name,
                    }
                },
            },
        )
        _write_json(
            dataset_root / "pinned-reuse-receipt.json",
            {
                "schema_version": "tgvf.vlmevalkit-pinned-reuse-receipt.v1",
                "dataset": dataset,
                "model": _MODEL,
                "source_evaluation_id": _SOURCE_EVALUATION,
                "source_run_id": _SOURCE_RUN,
                "source_manifest_path": str(source_manifest.resolve()),
                "source_manifest_sha256": _sha(source_manifest),
                "source_prediction_path": str(source_prediction.resolve()),
                "source_prediction_sha256": _sha(source_prediction),
                "destination_run_id": destination_run_id,
                "destination_status_path": str(destination_status.resolve()),
                "destination_status_sha256": _sha(destination_status),
                "destination_prediction_path": str(destination_prediction.resolve()),
                "destination_prediction_sha256": _sha(destination_prediction),
            },
        )
        expected[dataset] = destination_run_id
    return expected


def test_receipts_bind_every_dataset_to_its_exact_destination(tmp_path: Path) -> None:
    expected = _materialize_receipts(tmp_path)
    # A lexically newer poisoned sibling must not affect receipt resolution.
    poisoned = tmp_path / "VStarBench" / _MODEL / "T20260815_Gffffffff"
    poisoned.mkdir()
    _write_json(poisoned / "status.json", {"mode": "eval", "poison": True})

    observed = _MODULE._load_pinned_scoring_receipts(
        tmp_path,
        source_run_id=_SOURCE_RUN,
        source_evaluation_id=_SOURCE_EVALUATION,
    )

    assert observed == expected


@pytest.mark.parametrize(
    "tamper",
    [
        "source_prediction",
        "source_status",
        "destination_status",
        "destination_prediction",
    ],
)
def test_receipts_reject_source_or_destination_identity_drift(
    tmp_path: Path, tamper: str
) -> None:
    expected = _materialize_receipts(tmp_path)
    source_run_dir = tmp_path / "VStarBench" / _MODEL / _SOURCE_RUN
    run_dir = tmp_path / "VStarBench" / _MODEL / expected["VStarBench"]
    if tamper == "source_prediction":
        (source_run_dir / f"{_MODEL}_VStarBench.tsv").write_text(
            "tampered", encoding="utf-8"
        )
        expected_error = "source bytes"
    elif tamper == "source_status":
        status_path = source_run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["datasets"]["VStarBench"]["source_run"] = "WRONG"
        _write_json(status_path, status)
        expected_error = "source proof"
    elif tamper == "destination_status":
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["datasets"]["VStarBench"]["source_run"] = "WRONG"
        _write_json(status_path, status)
        expected_error = "destination bytes"
    else:
        (run_dir / f"{_MODEL}_VStarBench.tsv").write_text("tampered", encoding="utf-8")
        expected_error = "destination bytes"

    with pytest.raises(RuntimeError, match=expected_error):
        _MODULE._load_pinned_scoring_receipts(
            tmp_path,
            source_run_id=_SOURCE_RUN,
            source_evaluation_id=_SOURCE_EVALUATION,
        )


def test_accepted_summary_passes_receipt_eval_ids_to_summarizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _materialize_receipts(tmp_path)
    (tmp_path / "coredev-2511-eval-summary.json").write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def summarize(**kwargs):
        captured.update(kwargs)
        return {"status": "pass", "sample_count": 2511}

    monkeypatch.setattr(_MODULE, "summarize_coredev_results", summarize)
    result = _MODULE._accepted_official_summary(
        tmp_path,
        {"server": {"base_url": "http://judge.invalid/v1"}},
        source_run_id=_SOURCE_RUN,
        source_evaluation_id=_SOURCE_EVALUATION,
    )

    assert result == {"status": "pass", "sample_count": 2511}
    assert captured["expected_eval_ids"] == expected
