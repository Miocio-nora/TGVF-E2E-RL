from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_true1m_v1.json"
)
SUPERVISOR_PATH = (
    REPOSITORY_ROOT
    / "tools/supervise_original_qwen3_instruct_raw_direct_true1m_inference.py"
)


def _load_supervisor() -> ModuleType:
    name = "original_qwen3_instruct_true1m_supervisor_under_test"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_true1m_config_matches_historical_prl04_except_pixel_cap() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = payload["model"]["Qwen3-VL-8B-Instruct"]

    historical_prl04_at_512 = {
        "model_path": "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
        "use_custom_prompt": False,
        "use_vllm": True,
        "min_pixels": None,
        "max_pixels": 512 * 512,
        "total_pixels": None,
        "max_new_tokens": 8192,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "do_sample": True,
        "post_process": False,
        "system_prompt": None,
        "gpu_utils": 0.9,
        "inference_batch_size": 8,
        "request_seed_base": 42,
        "request_seed_namespace": (
            "coredev-2511-qwen3-instruct-direct-prl04-comparable-v1"
        ),
        "limit_mm_per_prompt": {"image": 24, "video": 0},
        "max_model_len": 65536,
        "mm_encoder_attn_backend": "TORCH_SDPA",
    }
    projected_to_512 = {**model, "max_pixels": 512 * 512}

    assert model["max_pixels"] == 1_003_520
    assert projected_to_512 == historical_prl04_at_512
    assert model["request_seed_base"] == 42
    assert model["use_custom_prompt"] is False
    assert model["system_prompt"] is None
    assert "tools" not in payload
    assert "tools" not in model


def test_true1m_config_freezes_all_seven_official_coredev_slices() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert tuple(payload["data"]) == (
        "VStarBench",
        "HRBench4K",
        "BLINK",
        "OCRBench_v2",
        "MMMU_Pro_10c",
        "MathVista_MINI",
        "MathVerse_MINI",
    )
    assert sum((191, 200, 420, 600, 300, 300, 500)) == 2511


def test_true1m_supervisor_materializes_one_slice_per_gpu_and_scorer_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor()
    python_bin = Path("/nonexistent/test-python")
    launches, spare_gpus = supervisor._materialize_launches(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        python_bin=python_bin,
        gpu_ids=tuple(range(8)),
    )
    base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setenv("CC", "/foreign/conda-cc")
    monkeypatch.setenv("CXX", "/foreign/conda-cxx")
    monkeypatch.setenv("CFLAGS", "-isystem /foreign/conda/include")
    monkeypatch.setenv("CMAKE_ARGS", "-DCMAKE_AR=/foreign/conda-ar")
    monkeypatch.setenv("CONDA_PREFIX", "/foreign/conda")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-ccbin=/foreign/conda-cxx")

    assert tuple(item.dataset for item in launches) == tuple(base["data"])
    assert tuple(item.gpu_id for item in launches) == tuple(range(7))
    assert spare_gpus == (7,)
    assert sum(item.expected_rows for item in launches) == 2511
    for item in launches:
        resolved = json.loads(item.resolved_config.read_text(encoding="utf-8"))
        assert resolved["model"] == base["model"]
        assert resolved["data"] == {item.dataset: base["data"][item.dataset]}
        assert (
            item.work_dir == (tmp_path / "inference" / item.dataset / "work").resolve()
        )
        assert item.command[2:] == (
            "--config",
            str(item.resolved_config),
            "--work-dir",
            str(item.work_dir),
            "--mode",
            "infer",
        )
        assert item.command[0] == str(python_bin)
        environment = supervisor._launch_environment(item, output_root=tmp_path)
        assert environment["CUDA_VISIBLE_DEVICES"] == str(item.gpu_id)
        assert environment["CC"] == "/usr/bin/gcc"
        assert environment["CXX"] == "/usr/bin/g++"
        assert environment["PATH"] == supervisor.WORKER_PATH
        assert environment["CPATH"] == (
            f"{supervisor.PYTHON_HEADER_ROOT}:"
            f"{supervisor.PYTHON_HEADER_ROOT / 'python3.12'}"
        )
        assert environment["PYTHONPATH"] == str(supervisor.REPOSITORY_ROOT / "src")
        for polluted in (
            "CFLAGS",
            "CMAKE_ARGS",
            "CONDA_PREFIX",
            "NVCC_PREPEND_FLAGS",
        ):
            assert polluted not in environment
        controlled_names = set(
            supervisor._controlled_worker_environment(item, output_root=tmp_path)
        )
        assert all(
            name not in environment
            for name in supervisor._PURGED_TOOLCHAIN_ENVIRONMENT
            if name not in controlled_names
        )
        assert not any(
            name.startswith(supervisor._PURGED_ENVIRONMENT_PREFIXES)
            for name in environment
        )
        for variable, family in (
            ("VLLM_CACHE_ROOT", "vllm"),
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ):
            cache = Path(environment[variable])
            assert cache == tmp_path / "runtime/cache" / family / item.dataset
            assert cache.is_dir()

    receipt = supervisor._contract_receipt(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        launches=launches,
        spare_gpus=spare_gpus,
    )
    assert receipt["max_pixels"] == 1_003_520
    assert receipt["config_sha256"] == supervisor.DEFAULT_CONFIG_SHA256
    assert receipt["request_seed_base"] == 42
    assert receipt["prompt_contract"] == (
        "official-dataset-raw-prompt-no-system-no-tools"
    )
    assert receipt["sample_count"] == 2511
    assert receipt["slice_count"] == 7
    worker_environment = receipt["worker_environment"]
    assert worker_environment["schema_version"] == (
        "tgvf.original-raw-direct-worker-environment.v2"
    )
    assert receipt["worker_environment_sha256"] == supervisor._canonical_sha256(
        worker_environment
    )
    assert worker_environment["workers"][0]["controlled"]["CC"] == "/usr/bin/gcc"
    assert worker_environment["workers"][0]["controlled"]["CXX"] == "/usr/bin/g++"
    proof = receipt["pixel_preprocessing_proof"]
    assert proof["configured_max_pixels"] == 1_003_520
    assert proof["image_item_has_max_pixels"] is True
    assert proof["processor_default_size"]["longest_edge"] == 16_777_216
    assert proof["source_dimensions"] == [2048, 1536]
    assert proof["source_pixel_area"] == 3_145_728
    assert proof["process_vision_info_dimensions"] == [1152, 864]
    assert proof["image_grid_thw"] == [1, 54, 72]
    assert proof["represented_pixel_area"] == 995_328
    assert proof["represented_pixel_area"] <= proof["configured_max_pixels"]
    assert proof["visual_token_count"] == 972

    contract_path = tmp_path / "runtime/inference-supervisor/contract.json"
    supervisor._write_json_atomic(contract_path, receipt)
    completed_receipts = tuple(
        {"dataset": item.dataset, "rows": item.expected_rows} for item in launches
    )
    receipt_by_dataset = {receipt["dataset"]: receipt for receipt in completed_receipts}
    monkeypatch.setattr(
        supervisor,
        "_completed_receipt",
        lambda launch: receipt_by_dataset[launch.dataset],
    )
    completion = supervisor._inference_completion_receipt(
        contract_path=contract_path,
        contract=receipt,
        launches=launches,
        receipts=completed_receipts,
    )
    assert completion["inference_contract_path"] == str(contract_path.resolve())
    assert completion["inference_contract_sha256"] == supervisor._sha256_file(
        contract_path
    )
    assert completion["config_sha256"] == supervisor.DEFAULT_CONFIG_SHA256
    assert completion["request_seed_namespace"] == supervisor.SEED_NAMESPACE
    assert completion["prompt_contract"] == (
        "official-dataset-raw-prompt-no-system-no-tools"
    )
    assert completion["worker_environment"] == worker_environment
    assert (
        completion["worker_environment_sha256"] == receipt["worker_environment_sha256"]
    )

    persisted = json.loads(contract_path.read_text(encoding="utf-8"))
    persisted["prompt_contract"] = "foreign-prompt-contract"
    supervisor._write_json_atomic(contract_path, persisted)
    with pytest.raises(RuntimeError, match="completion inputs differ"):
        supervisor._inference_completion_receipt(
            contract_path=contract_path,
            contract=receipt,
            launches=launches,
            receipts=completed_receipts,
        )


def test_completion_scan_binds_exact_status_argv_and_prediction_layout(
    tmp_path: Path,
) -> None:
    supervisor = _load_supervisor()
    resolved_config = (tmp_path / "resolved-configs/vstar.json").resolve()
    resolved_config.parent.mkdir(parents=True)
    resolved_config.write_text("{}\n", encoding="utf-8")
    work_dir = (tmp_path / "work").resolve()
    launch = supervisor.DatasetLaunch(
        dataset="VStarBench",
        gpu_id=0,
        expected_rows=1,
        resolved_config=resolved_config,
        work_dir=work_dir,
        cwd=(tmp_path / "cwd").resolve(),
        log_path=(tmp_path / "infer.log").resolve(),
        command=(),
    )
    run_dir = work_dir / supervisor.MODEL_NAME / "T20260828_Gabc123"
    prediction = run_dir / f"{supervisor.MODEL_NAME}_VStarBench.tsv"
    prediction.parent.mkdir(parents=True)
    with prediction.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("index", "prediction"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerow({"index": "vstar-0", "prediction": "answer"})
    status_path = run_dir / "status.json"
    status = {
        "eval_id": run_dir.name,
        "mode": "infer",
        "model_name": supervisor.MODEL_NAME,
        "argv": [
            "runner",
            "--config",
            str(resolved_config),
            "--work-dir",
            str(work_dir),
            "--mode",
            "infer",
        ],
        "datasets": {
            "VStarBench": {
                "status": "done",
                "skip_reason": "mode_infer",
                "prediction_file": str(prediction),
            }
        },
    }
    supervisor._write_json_atomic(status_path, status)

    receipt = supervisor._completed_receipt(launch)
    assert receipt is not None
    assert receipt["prediction_path"] == str(prediction)

    status["argv"].extend(("--config", str(resolved_config)))
    supervisor._write_json_atomic(status_path, status)
    assert supervisor._completed_receipt(launch) is None

    status["argv"] = status["argv"][:-2]
    status["argv"][status["argv"].index("--work-dir") + 1] = str(
        tmp_path / "foreign-work"
    )
    supervisor._write_json_atomic(status_path, status)
    assert supervisor._completed_receipt(launch) is None


def test_failed_v1_contract_is_not_overwritten_or_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _load_supervisor()
    contract_path = tmp_path / "runtime/inference-supervisor/contract.json"
    legacy = {
        "schema_version": "tgvf.original-raw-direct-inference-contract.v1",
        "evaluation_contract": supervisor.EVALUATION_CONTRACT,
    }
    supervisor._write_json_atomic(contract_path, legacy)
    monkeypatch.setattr(
        supervisor,
        "_wait_for_gpus",
        lambda *_args, **_kwargs: pytest.fail("contract mismatch queried GPUs"),
    )

    with pytest.raises(RuntimeError, match="quarantine the complete failed output"):
        supervisor._run_inference(
            output_root=tmp_path,
            launches=(),
            contract={
                "schema_version": supervisor.INFERENCE_CONTRACT_SCHEMA,
                "evaluation_contract": supervisor.EVALUATION_CONTRACT,
            },
            wait_for_gpus=True,
        )

    assert json.loads(contract_path.read_text(encoding="utf-8")) == legacy


@pytest.mark.parametrize(
    "gpu_ids",
    [tuple(range(6)), (0, 1, 2, 3, 4, 5, 5), tuple(range(9))],
)
def test_true1m_supervisor_rejects_ambiguous_gpu_assignments(
    gpu_ids: tuple[int, ...],
) -> None:
    supervisor = _load_supervisor()

    with pytest.raises(ValueError):
        supervisor._validate_gpu_ids(gpu_ids)


def test_true1m_supervisor_preserves_virtualenv_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supervisor = _load_supervisor()
    captured: dict[str, Path] = {}

    def materialize_launches(
        **kwargs: object,
    ) -> tuple[tuple[object, ...], tuple[int, ...]]:
        python_bin = kwargs["python_bin"]
        assert isinstance(python_bin, Path)
        captured["python_bin"] = python_bin
        return (), ()

    monkeypatch.setattr(supervisor, "_materialize_launches", materialize_launches)
    monkeypatch.setattr(
        supervisor,
        "_contract_receipt",
        lambda **_kwargs: {"status": "validated"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SUPERVISOR_PATH), "--validate-only"],
    )

    assert supervisor.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "validated"}
    assert captured["python_bin"] == supervisor.DEFAULT_PYTHON
    assert (
        supervisor._absolute_path_without_resolving_symlinks(supervisor.DEFAULT_PYTHON)
        == supervisor.DEFAULT_PYTHON
    )
