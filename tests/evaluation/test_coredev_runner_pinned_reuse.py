from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_coredev_2511_vlmevalkit.py"
_SPEC = importlib.util.spec_from_file_location("coredev_pinned_reuse", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_MODEL = "Qwen3-VL-8B-Instruct"
_DATASET = "VStarBench"
_SOURCE_RUN = "T20260815_Gdeadbeef"
_DESTINATION_RUN = "T20260815_G01234567"


@pytest.mark.parametrize("port", [8012, 8013, 8014, 8015])
def test_judge_base_url_accepts_evaluator_owned_local_ports(port: int) -> None:
    expected = "http://127.0.0.1:8012/v1"

    assert (
        _MODULE._validate_judge_base_url(f"http://127.0.0.1:{port}/v1/", expected)
        == f"http://127.0.0.1:{port}/v1"
    )


@pytest.mark.parametrize(
    "observed",
    [
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8016/v1",
        "http://localhost:8013/v1",
        "http://127.0.0.2:8013/v1",
        "https://127.0.0.1:8013/v1",
        "http://127.0.0.1:8013/other",
        "http://user@127.0.0.1:8013/v1",
    ],
)
def test_judge_base_url_rejects_non_pinned_deployment(observed: str) -> None:
    with pytest.raises(RuntimeError, match="pinned local deployment"):
        _MODULE._validate_judge_base_url(
            observed,
            "http://127.0.0.1:8012/v1",
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluated_model_is_derived_from_single_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _write_json(config, {"model": {_MODEL: {"model_path": "/model"}}})
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(config)])

    assert _MODULE._evaluated_model_from_cli_or_config() == _MODEL


def test_evaluated_model_rejects_ambiguous_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _write_json(config, {"model": {"model-a": {}, "model-b": {}}})
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(config)])

    with pytest.raises(RuntimeError, match="exactly one model"):
        _MODULE._evaluated_model_from_cli_or_config()


def _source_fixture(tmp_path: Path):
    work_dir = tmp_path / "work"
    source_run_dir = work_dir / _MODEL / _SOURCE_RUN
    prediction = source_run_dir / f"{_MODEL}_{_DATASET}.tsv"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b'"index"\t"prediction"\n"1"\t"A"\n')
    prediction_sha256 = sha256(prediction.read_bytes()).hexdigest()
    manifest = source_run_dir / "final-answer-view-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "derived": {
                "path": str(prediction.resolve()),
                "sha256": prediction_sha256,
                "row_count": 1,
            },
        },
    )
    _write_json(
        source_run_dir / "status.json",
        {
            "eval_id": _SOURCE_RUN,
            "mode": "infer",
            "reuse_aux": "infer",
            "datasets": {
                _DATASET: {
                    "status": "done",
                    "prediction_file": str(prediction.resolve()),
                    "source_run": "SEMANTIC-POLICY-EVALUATION",
                }
            },
        },
    )
    contract = _MODULE._load_pinned_reuse_contract(
        work_dir=work_dir,
        model_name=_MODEL,
        dataset_name=_DATASET,
        source_run_id=_SOURCE_RUN,
        manifest_path=manifest,
    )
    return contract


def _write_destination(contract, *, source_run: str = _SOURCE_RUN) -> Path:
    destination = contract.model_root / _DESTINATION_RUN
    prediction = destination / f"{_MODEL}_{_DATASET}.tsv"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(contract.prediction_path.read_bytes())
    _write_json(
        destination / "status.json",
        {
            "eval_id": _DESTINATION_RUN,
            "mode": "eval",
            "reuse": True,
            "reuse_aux": "infer",
            "datasets": {
                _DATASET: {
                    "status": "done",
                    "prediction_file": prediction.name,
                    "source_run": source_run,
                }
            },
        },
    )
    return destination


def test_pinned_selector_ignores_newer_sibling_and_returns_exact_source(
    tmp_path: Path,
) -> None:
    contract = _source_fixture(tmp_path)
    newer = contract.model_root / "T20260815_Gffffffff"
    newer.mkdir()
    (newer / f"{_MODEL}_{_DATASET}.tsv").write_text("wrong", encoding="utf-8")
    smp = SimpleNamespace()
    smp_file = SimpleNamespace()

    selector = _MODULE._install_pinned_reuse_selector(
        contract,
        destination_run_id=_DESTINATION_RUN,
        smp_module=smp,
        smp_file_module=smp_file,
    )
    result_file = contract.model_root / _DESTINATION_RUN / f"{_MODEL}_{_DATASET}.tsv"
    selected = selector(
        str(contract.model_root),
        _DESTINATION_RUN,
        _MODEL,
        _DATASET,
        str(result_file),
        set(),
    )

    assert smp.build_eval_id() == _DESTINATION_RUN
    assert smp_file.select_reuse_run is selector
    assert Path(selected["run_dir"]) == contract.source_run_dir
    assert Path(selected["prediction_file"]) == contract.prediction_path
    assert Path(selected["run_dir"]) != newer


def test_pinned_selector_rejects_a_different_destination_request(
    tmp_path: Path,
) -> None:
    contract = _source_fixture(tmp_path)
    selector = _MODULE._install_pinned_reuse_selector(
        contract,
        destination_run_id=_DESTINATION_RUN,
        smp_module=SimpleNamespace(),
        smp_file_module=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="differs from pinned contract"):
        selector(
            str(contract.model_root),
            "T20260815_Gbad0",
            _MODEL,
            _DATASET,
            str(contract.model_root / "T20260815_Gbad0" / "wrong.tsv"),
            set(),
        )


def test_pinned_postflight_binds_unique_destination_source_run_and_sha(
    tmp_path: Path,
) -> None:
    contract = _source_fixture(tmp_path)
    preexisting = _MODULE._eval_run_ids(contract.model_root)
    _write_destination(contract)

    receipt = _MODULE._verify_pinned_reuse_postflight(
        contract,
        destination_run_id=_DESTINATION_RUN,
        preexisting_run_ids=preexisting,
    )
    assert receipt["source_evaluation_id"] == "SEMANTIC-POLICY-EVALUATION"
    assert receipt["source_run_id"] == _SOURCE_RUN
    assert receipt["destination_run_id"] == _DESTINATION_RUN
    assert (
        receipt["source_prediction_sha256"] == receipt["destination_prediction_sha256"]
    )


@pytest.mark.parametrize("failure", ["source_run", "prediction_sha", "extra_run"])
def test_pinned_postflight_rejects_wrong_identity(tmp_path: Path, failure: str) -> None:
    contract = _source_fixture(tmp_path)
    preexisting = _MODULE._eval_run_ids(contract.model_root)
    destination = _write_destination(
        contract,
        source_run="T20260815_Gffffffff" if failure == "source_run" else _SOURCE_RUN,
    )
    expected_error = "status/source_run"
    if failure == "prediction_sha":
        (destination / f"{_MODEL}_{_DATASET}.tsv").write_text(
            "tampered", encoding="utf-8"
        )
        expected_error = "prediction identity"
    elif failure == "extra_run":
        (contract.model_root / "T20260815_Geeeeeeee").mkdir()
        expected_error = "unique destination"

    with pytest.raises(RuntimeError, match=expected_error):
        _MODULE._verify_pinned_reuse_postflight(
            contract,
            destination_run_id=_DESTINATION_RUN,
            preexisting_run_ids=preexisting,
        )


def test_pinned_contract_rejects_manifest_prediction_sha_drift(tmp_path: Path) -> None:
    contract = _source_fixture(tmp_path)
    contract.prediction_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable manifest"):
        _MODULE._load_pinned_reuse_contract(
            work_dir=contract.work_dir,
            model_name=_MODEL,
            dataset_name=_DATASET,
            source_run_id=_SOURCE_RUN,
            manifest_path=contract.manifest_path,
        )
