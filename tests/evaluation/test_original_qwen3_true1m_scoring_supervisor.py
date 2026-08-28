from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_PATH = (
    REPOSITORY_ROOT
    / "tools/supervise_original_qwen3_instruct_raw_direct_true1m_scoring.py"
)


def _load_supervisor() -> ModuleType:
    name = "original_qwen3_true1m_scoring_supervisor_under_test"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_prediction(
    path: Path,
    *,
    dataset: str,
    rows: int,
    extra_fields: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("index", "prediction", *extra_fields)
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(rows):
            row = {
                "index": f"{dataset}-{index}",
                "prediction": "answer",
            }
            if "source_row_index" in extra_fields:
                row["source_row_index"] = str(index)
            if "metadata" in extra_fields:
                row["metadata"] = "{}"
            writer.writerow(row)


def _materialize_inference_completion(module: ModuleType, root: Path) -> None:
    base_config = json.loads(module.INFERENCE_CONFIG_PATH.read_text(encoding="utf-8"))
    receipts = []
    workers = []
    for ordinal, (dataset, rows) in enumerate(module.DATASETS):
        dataset_root = root / f"inference/{dataset}"
        work_dir = (dataset_root / "work").resolve()
        resolved_config = (
            dataset_root / f"resolved-configs/coredev-subset-{ordinal}.json"
        ).resolve()
        _write_json(
            resolved_config,
            {
                "model": base_config["model"],
                "data": {dataset: base_config["data"][dataset]},
            },
        )
        run_id = f"T20260828_G{ordinal + 1:064x}"
        run_dir = work_dir / module.MODEL_NAME / run_id
        prediction = run_dir / f"{module.MODEL_NAME}_{dataset}.tsv"
        _write_prediction(prediction, dataset=dataset, rows=rows)
        status = run_dir / "status.json"
        _write_json(
            status,
            {
                "eval_id": run_id,
                "mode": "infer",
                "model_name": module.MODEL_NAME,
                "commit": module.VLMEVALKIT_REVIEW_COMMIT[:8],
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
                    dataset: {
                        "status": "done",
                        "skip_reason": "mode_infer",
                        "prediction_file": str(prediction),
                    }
                },
            },
        )
        receipts.append(
            {
                "dataset": dataset,
                "rows": rows,
                "status_path": str(status),
                "status_sha256": _sha(status),
                "prediction_path": str(prediction),
                "prediction_sha256": _sha(prediction),
            }
        )
        workers.append(
            {
                "dataset": dataset,
                "expected_rows": rows,
                "gpu_id": ordinal,
                "resolved_config": str(resolved_config),
                "resolved_config_sha256": _sha(resolved_config),
                "work_dir": str(work_dir),
            }
        )
    typed_workers = tuple(
        module.InferenceWorkerContract(
            dataset=str(worker["dataset"]),
            expected_rows=int(worker["expected_rows"]),
            gpu_id=int(worker["gpu_id"]),
            resolved_config=Path(str(worker["resolved_config"])),
            resolved_config_sha256=str(worker["resolved_config_sha256"]),
            work_dir=Path(str(worker["work_dir"])),
        )
        for worker in workers
    )
    worker_environment = module._expected_worker_environment_contract(
        typed_workers, output_root=root
    )
    contract_path = root / "runtime/inference-supervisor/contract.json"
    _write_json(
        contract_path,
        {
            "schema_version": module.INFERENCE_CONTRACT_SCHEMA,
            "evaluation_contract": module.EVALUATION_CONTRACT,
            "config_path": str(module.INFERENCE_CONFIG_PATH.resolve()),
            "config_sha256": module.INFERENCE_CONFIG_SHA256,
            "output_root": str(root.resolve()),
            "model": module.MODEL_NAME,
            "prompt_contract": module.RAW_PROMPT_CONTRACT,
            "max_pixels": module.TRUE1M_MAX_PIXELS,
            "request_seed_base": 42,
            "request_seed_namespace": module.REQUEST_SEED_NAMESPACE,
            "worker_environment": worker_environment,
            "worker_environment_sha256": module._canonical_sha256(worker_environment),
            "sample_count": 2511,
            "slice_count": 7,
            "workers": workers,
            "spare_gpu_ids": [7],
        },
    )
    _write_json(
        root / "runtime/inference-supervisor/original-true1m-inference-complete.json",
        {
            "schema_version": module.INFERENCE_COMPLETION_SCHEMA,
            "status": "complete",
            "sample_count": 2511,
            "slice_count": 7,
            "max_pixels": 1_003_520,
            "inference_contract_path": str(contract_path.resolve()),
            "inference_contract_sha256": _sha(contract_path),
            "config_path": str(module.INFERENCE_CONFIG_PATH.resolve()),
            "config_sha256": module.INFERENCE_CONFIG_SHA256,
            "request_seed_namespace": module.REQUEST_SEED_NAMESPACE,
            "prompt_contract": module.RAW_PROMPT_CONTRACT,
            "worker_environment": worker_environment,
            "worker_environment_sha256": module._canonical_sha256(worker_environment),
            "receipts": receipts,
        },
    )


def test_static_contract_reuses_exact_judge_and_macro_star(tmp_path: Path) -> None:
    module = _load_supervisor()
    contract = module._static_contract(
        output_root=tmp_path,
        python_bin=Path("/accepted/python3.12"),
        task_manifest=Path("/accepted/tasks.jsonl"),
        mathverse_source_json=Path("/accepted/mathverse.json"),
        judge_port=8012,
        judge_gpu_ids=(0, 1),
    )

    assert contract["evaluation_contract"] == (
        "original-qwen3-instruct-raw-direct-true1m-v1"
    )
    assert contract["max_pixels"] == 1_003_520
    assert contract["sample_count"] == 2511
    assert contract["slice_count"] == 7
    assert contract["inference_config_sha256"] == module.INFERENCE_CONFIG_SHA256
    assert contract["raw_prompt_contract"] == module.RAW_PROMPT_CONTRACT
    assert contract["request_seed_namespace"] == module.REQUEST_SEED_NAMESPACE
    assert contract["source_views"]["mathverse_source_sha256"] == (
        "f4ce9b18d111b23d5950dcbc8f377c6a05955a458db6a3103aec706fa63b0e9b"
    )
    assert contract["judge"] == {
        "model": "Qwen2.5-72B-Instruct",
        "model_path": "/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct",
        "base_url": "http://127.0.0.1:8012/v1",
        "gpu_ids": [0, 1],
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 64,
        "seed": 42,
        "generation_config": "vllm",
        "prefix_caching": True,
        "attention_backend": "TRITON_ATTN",
        "toolchain_environment": module.controlled_toolchain_contract(
            module.JUDGE_TOOLCHAIN_ENVIRONMENT
        ),
        "toolchain_verification": module.controlled_toolchain_verification(
            module._judge_environment((0, 1)),
            controlled=module.JUDGE_TOOLCHAIN_ENVIRONMENT,
        ),
    }
    assert contract["headline"] == {
        "components": [
            "vstar",
            "hr_average_all",
            "blink_single_180",
            "ocr_mean",
            "mmmu_single_269",
            "mathvista",
            "mathverse_five_version_macro",
        ],
        "aggregation": "unweighted_mean_of_seven_percent_components",
        "blink_single_image_count": 180,
        "mmmu_single_image_count": 269,
        "mathverse_versions": [
            "Text Dominant",
            "Vision Only",
            "Text Lite",
            "Vision Intensive",
            "Vision Dominant",
        ],
        "task_manifest": "/accepted/tasks.jsonl",
        "task_manifest_sha256": (
            "3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc"
        ),
    }


def test_completed_inference_is_receipt_bound_not_latest_glob(tmp_path: Path) -> None:
    module = _load_supervisor()
    _materialize_inference_completion(module, tmp_path)
    poisoned = (
        tmp_path / f"inference/VStarBench/work/{module.MODEL_NAME}/"
        "T20260828_Gffffffffffffffff/status.json"
    )
    _write_json(poisoned, {"mode": "infer", "poison": True})

    sources = module._load_completed_inference(tmp_path)

    assert tuple(source.dataset for source in sources) == tuple(
        dataset for dataset, _rows in module.DATASETS
    )
    assert sum(source.expected_rows for source in sources) == 2511
    assert sources[0].source_run_id.endswith(f"{1:064x}")

    sources[0].prediction_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes changed"):
        module._load_completed_inference(tmp_path)


def test_inference_provenance_rejects_contract_config_and_argv_attacks(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()

    contract_root = tmp_path / "contract-attack"
    _materialize_inference_completion(module, contract_root)
    contract_path = contract_root / "runtime/inference-supervisor/contract.json"
    completion_path = (
        contract_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["request_seed_namespace"] = "wrong-seed-namespace"
    _write_json(contract_path, contract)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["inference_contract_sha256"] = _sha(contract_path)
    _write_json(completion_path, completion)
    with pytest.raises(RuntimeError, match="contract identity differs"):
        module._load_completed_inference(contract_root)

    config_sha_root = tmp_path / "config-sha-attack"
    _materialize_inference_completion(module, config_sha_root)
    config_sha_completion_path = (
        config_sha_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    config_sha_completion = json.loads(
        config_sha_completion_path.read_text(encoding="utf-8")
    )
    config_sha_completion["config_sha256"] = "0" * 64
    _write_json(config_sha_completion_path, config_sha_completion)
    with pytest.raises(RuntimeError, match="completion provenance differs"):
        module._load_completed_inference(config_sha_root)

    environment_root = tmp_path / "environment-attack"
    _materialize_inference_completion(module, environment_root)
    environment_contract_path = (
        environment_root / "runtime/inference-supervisor/contract.json"
    )
    environment_completion_path = (
        environment_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    environment_contract = json.loads(
        environment_contract_path.read_text(encoding="utf-8")
    )
    environment_contract["worker_environment"]["workers"][0]["controlled"]["CC"] = (
        "/foreign/conda-cc"
    )
    environment_contract["worker_environment_sha256"] = module._canonical_sha256(
        environment_contract["worker_environment"]
    )
    _write_json(environment_contract_path, environment_contract)
    environment_completion = json.loads(
        environment_completion_path.read_text(encoding="utf-8")
    )
    environment_completion["inference_contract_sha256"] = _sha(
        environment_contract_path
    )
    environment_completion["worker_environment"] = environment_contract[
        "worker_environment"
    ]
    environment_completion["worker_environment_sha256"] = environment_contract[
        "worker_environment_sha256"
    ]
    _write_json(environment_completion_path, environment_completion)
    with pytest.raises(RuntimeError, match="worker environment differs"):
        module._load_completed_inference(environment_root)

    config_root = tmp_path / "config-attack"
    _materialize_inference_completion(module, config_root)
    config_contract_path = config_root / "runtime/inference-supervisor/contract.json"
    config_completion_path = (
        config_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    config_contract = json.loads(config_contract_path.read_text(encoding="utf-8"))
    resolved_config = Path(config_contract["workers"][0]["resolved_config"])
    resolved_payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    resolved_payload["data"] = {
        "VStarBench": {"class": "WrongSlice", "dataset": "VStarBench"}
    }
    _write_json(resolved_config, resolved_payload)
    config_contract["workers"][0]["resolved_config_sha256"] = _sha(resolved_config)
    _write_json(config_contract_path, config_contract)
    config_completion = json.loads(config_completion_path.read_text(encoding="utf-8"))
    config_completion["inference_contract_sha256"] = _sha(config_contract_path)
    _write_json(config_completion_path, config_completion)
    with pytest.raises(RuntimeError, match="resolved inference config differs"):
        module._load_completed_inference(config_root)

    argv_root = tmp_path / "argv-attack"
    _materialize_inference_completion(module, argv_root)
    argv_completion_path = (
        argv_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    argv_completion = json.loads(argv_completion_path.read_text(encoding="utf-8"))
    first_receipt = argv_completion["receipts"][0]
    status_path = Path(first_receipt["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    work_dir_index = status["argv"].index("--work-dir") + 1
    status["argv"][work_dir_index] = str(argv_root / "wrong-work-dir")
    _write_json(status_path, status)
    first_receipt["status_sha256"] = _sha(status_path)
    _write_json(argv_completion_path, argv_completion)
    with pytest.raises(RuntimeError, match="inference status differs"):
        module._load_completed_inference(argv_root)


def test_source_views_are_pinned_and_preserve_raw_predictions(tmp_path: Path) -> None:
    module = _load_supervisor()
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "artifact"
    mathverse_source = tmp_path / "mathverse.json"
    _write_json(
        mathverse_source,
        [
            {"problem_version": module._MATHVERSE_VERSIONS[index % 5]}
            for index in range(500)
        ],
    )

    vstar_prediction = raw_root / "vstar.tsv"
    _write_prediction(vstar_prediction, dataset="VStarBench", rows=1)
    vstar_raw = module.InferenceSource(
        dataset="VStarBench",
        expected_rows=1,
        source_run_id="T20260828_Gabc1",
        status_path=raw_root / "status.json",
        prediction_path=vstar_prediction.resolve(),
        prediction_sha256=_sha(vstar_prediction),
    )
    vstar = module._materialize_source_view(
        vstar_raw,
        output_root=output_root,
        mathverse_source_json=mathverse_source,
    )
    assert vstar.prediction_path.read_bytes() == vstar_prediction.read_bytes()
    assert vstar.source_evaluation_id == vstar_raw.source_run_id
    vstar_status = json.loads(
        (vstar.prediction_path.parent / "status.json").read_text(encoding="utf-8")
    )
    assert vstar_status["datasets"]["VStarBench"]["source_run"] == (
        vstar_raw.source_run_id
    )
    vstar_manifest = json.loads(vstar.manifest_path.read_text(encoding="utf-8"))
    vstar.prediction_path.write_bytes(
        vstar.prediction_path.read_bytes().replace(b"answer", b"tampered", 1)
    )
    vstar_manifest["derived"]["sha256"] = _sha(vstar.prediction_path)
    _write_json(vstar.manifest_path, vstar_manifest)
    with pytest.raises(RuntimeError, match="raw source view bytes differ"):
        module._validate_source_view(
            vstar_raw,
            vstar,
            mathverse_source_json=mathverse_source,
        )

    mathverse_prediction = raw_root / "mathverse.tsv"
    _write_prediction(
        mathverse_prediction,
        dataset="MathVerse_MINI",
        rows=500,
        extra_fields=("source_row_index", "metadata"),
    )
    mathverse_raw = module.InferenceSource(
        dataset="MathVerse_MINI",
        expected_rows=500,
        source_run_id="T20260828_Gabc2",
        status_path=raw_root / "status-mathverse.json",
        prediction_path=mathverse_prediction.resolve(),
        prediction_sha256=_sha(mathverse_prediction),
    )
    mathverse = module._materialize_source_view(
        mathverse_raw,
        output_root=output_root,
        mathverse_source_json=mathverse_source,
    )
    with mathverse.prediction_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["prediction"] == "answer"
    assert json.loads(row["metadata"])["problem_version"] == "Text Dominant"
    manifest = json.loads(mathverse.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract"] == "vlmevalkit-mathverse-metadata-view-v1"
    assert manifest["prediction_values_identical"] is True

    command = module._score_command(
        mathverse,
        python_bin=Path("/accepted/python"),
        judge_base_url="http://127.0.0.1:8012/v1",
    )
    assert command[command.index("--tgvf-reuse-source-run-id") + 1] == (
        mathverse.source_run_id
    )
    assert command[command.index("--tgvf-reuse-manifest") + 1] == str(
        mathverse.manifest_path
    )
    assert command[command.index("--judge") + 1] == "Qwen2.5-72B-Instruct"
    assert module._scoring_environment()["CUDA_VISIBLE_DEVICES"] == ""

    fields, rows = module._read_tsv_rows(mathverse.prediction_path)
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["prediction"] = "tampered"
    with mathverse.prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(changed_rows)
    manifest = json.loads(mathverse.manifest_path.read_text(encoding="utf-8"))
    manifest["derived"]["sha256"] = _sha(mathverse.prediction_path)
    _write_json(mathverse.manifest_path, manifest)
    with pytest.raises(RuntimeError, match="changed source row"):
        module._validate_source_view(
            mathverse_raw,
            mathverse,
            mathverse_source_json=mathverse_source,
        )


@pytest.mark.parametrize(
    ("dataset", "row_count", "single_image_count"),
    (("BLINK", 420, 180), ("MMMU_Pro_10c", 300, 269)),
)
def test_coverage_views_revalidate_rows_and_frozen_task_identity(
    tmp_path: Path,
    dataset: str,
    row_count: int,
    single_image_count: int,
) -> None:
    module = _load_supervisor()
    source_run = tmp_path / f"source/{dataset}"
    source_status = source_run / "status.json"
    _write_json(source_status, {"dataset": dataset})
    source_result = source_run / f"{module.MODEL_NAME}_{dataset}_judge_result.tsv"
    source_result.parent.mkdir(parents=True, exist_ok=True)
    with source_result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("index", "prediction", "hit"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            {
                "index": f"{dataset}-{index}",
                "prediction": "answer",
                "hit": "0",
            }
            for index in range(row_count)
        )
    task_manifest = tmp_path / f"tasks-{dataset}.jsonl"
    task_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": dataset,
                    "index": f"{dataset}-{index}",
                    "image_paths": (
                        ["one.jpg"]
                        if index < single_image_count
                        else ["one.jpg", "two.jpg"]
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(row_count)
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "artifact"
    item = {"dataset": dataset, "status_path": str(source_status)}
    module._coverage_view(
        item,
        output_root=output_root,
        task_manifest=task_manifest,
    )
    coverage_dir = output_root / f"scoring/headline-coverage-views/{dataset}"
    manifest_path = coverage_dir / "coverage-view-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "single_image_evaluated": single_image_count,
        "excluded_multi_image_reference": row_count - single_image_count,
    }

    # An unchanged immutable view remains reusable.
    module._coverage_view(
        {"dataset": dataset, "status_path": str(source_status)},
        output_root=output_root,
        task_manifest=task_manifest,
    )

    derived_result = Path(manifest["derived"]["path"])
    fields, rows = module._read_tsv_rows(derived_result)
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["hit"] = "1"
    with derived_result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(changed_rows)
    manifest["derived"]["sha256"] = _sha(derived_result)
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="changed source row"):
        module._coverage_view(
            {"dataset": dataset, "status_path": str(source_status)},
            output_root=output_root,
            task_manifest=task_manifest,
        )


def test_aggregate_binds_receipts_coverage_views_and_frozen_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_supervisor()
    sources = tuple(
        module.ScoringSource(
            dataset=dataset,
            expected_rows=rows,
            source_evaluation_id=f"raw-{dataset}",
            source_run_id=f"T20260828_G{ordinal + 1:x}",
            work_dir=tmp_path / f"scoring/datasets/{dataset}",
            cwd=tmp_path / f"cwd/{dataset}",
            manifest_path=tmp_path / f"manifests/{dataset}.json",
            prediction_path=tmp_path / f"predictions/{dataset}.tsv",
        )
        for ordinal, (dataset, rows) in enumerate(module.DATASETS)
    )
    captured: dict[str, object] = {"coverage": []}

    monkeypatch.setattr(
        module,
        "_completed_scoring_destination",
        lambda source: f"T20260828_Gdest{source.dataset}",
    )

    def summarize(**kwargs: object) -> dict[str, object]:
        captured["summarize"] = kwargs
        return {
            "schema_version": 1,
            "sample_count": 2511,
            "slice_count": 7,
            "slices": [
                {"dataset": dataset, "status_path": f"/{dataset}/status.json"}
                for dataset, _rows in module.DATASETS
            ],
        }

    def coverage(item: dict[str, object], **kwargs: object) -> None:
        captured["coverage"].append((item["dataset"], kwargs))

    headline = {
        "components_percent": {
            component: 1.0 for component in module.COREDEV_MACRO_STAR_COMPONENTS
        },
        "single_image_counts": {
            "blink": {"count": 180},
            "mmmu": {"count": 269},
        },
        "mathverse_version_components_percent": {
            version: 1.0 for version in module._MATHVERSE_VERSIONS
        },
        "aggregation": "unweighted_mean_of_seven_percent_components",
        "macro_star_percent": 1.0,
    }
    monkeypatch.setattr(module, "summarize_coredev_results", summarize)
    monkeypatch.setattr(module, "_coverage_view", coverage)
    monkeypatch.setattr(module, "extract_coredev_macro_star", lambda summary: headline)
    monkeypatch.setattr(
        module,
        "_validate_frozen_scoring_inputs",
        lambda **kwargs: captured.setdefault("frozen_inputs", kwargs),
    )
    monkeypatch.setattr(
        module,
        "_revalidate_aggregate_provenance",
        lambda revalidated_sources, **kwargs: captured.setdefault(
            "aggregate_provenance", (revalidated_sources, kwargs)
        ),
    )

    summary = module._aggregate(
        sources,
        output_root=tmp_path,
        task_manifest=Path("/accepted/tasks.jsonl"),
        mathverse_source_json=Path("/accepted/mathverse.json"),
        judge_base_url="http://127.0.0.1:8012/v1",
    )

    summarize_kwargs = captured["summarize"]
    assert summarize_kwargs["expected_model"] == "Qwen3-VL-8B-Instruct"
    assert summarize_kwargs["expected_judge_base_url"] == ("http://127.0.0.1:8012/v1")
    assert tuple(summarize_kwargs["expected_eval_ids"]) == tuple(
        dataset for dataset, _rows in module.DATASETS
    )
    assert len(captured["coverage"]) == 7
    assert captured["frozen_inputs"] == {
        "task_manifest": Path("/accepted/tasks.jsonl"),
        "mathverse_source_json": Path("/accepted/mathverse.json"),
    }
    assert captured["aggregate_provenance"] == (
        sources,
        {
            "output_root": tmp_path,
            "mathverse_source_json": Path("/accepted/mathverse.json"),
        },
    )
    assert summary["max_pixels"] == 1_003_520
    assert summary["raw_inference_run_ids"] == {
        source.dataset: source.source_evaluation_id for source in sources
    }
    assert summary["headline"] is headline


def test_frozen_task_and_mathverse_hashes_reject_content_drift(tmp_path: Path) -> None:
    module = _load_supervisor()
    bad_task_manifest = tmp_path / "tasks.jsonl"
    bad_task_manifest.write_bytes(module.DEFAULT_TASK_MANIFEST.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="task manifest SHA256 differs"):
        module._validate_frozen_scoring_inputs(
            task_manifest=bad_task_manifest,
            mathverse_source_json=module.DEFAULT_MATHVERSE_SOURCE,
        )

    bad_mathverse_source = tmp_path / "mathverse.json"
    bad_mathverse_source.write_bytes(
        module.DEFAULT_MATHVERSE_SOURCE.read_bytes() + b"\n"
    )
    with pytest.raises(RuntimeError, match="MathVerse source SHA256 differs"):
        module._validate_frozen_scoring_inputs(
            task_manifest=module.DEFAULT_TASK_MANIFEST,
            mathverse_source_json=bad_mathverse_source,
        )


def test_judge_environment_purges_parent_toolchain_pollution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_supervisor()
    poisoned = {
        "CFLAGS": "-I/poison/revisit-vlm/include",
        "CPPFLAGS": "-I/poison/revisit-vlm/include",
        "CXXFLAGS": "-I/poison/revisit-vlm/include",
        "LD": "/poison/revisit-vlm/bin/ld",
        "LDFLAGS": "-L/poison/revisit-vlm/lib",
        "CONDA_PREFIX": "/poison/conda",
        "_CONDA_ROOT": "/poison/conda-root",
    }
    for name, value in poisoned.items():
        monkeypatch.setenv(name, value)

    environment = module._judge_environment((0, 1))

    assert (
        module.controlled_toolchain_verification(
            environment,
            controlled=module.JUDGE_TOOLCHAIN_ENVIRONMENT,
        )["verified"]
        is True
    )
    assert all(name not in environment for name in poisoned)


def test_validate_only_never_queries_or_launches_gpus(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_supervisor()
    monkeypatch.setattr(module, "_validate_static_dependencies", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_busy_gpus",
        lambda *_args, **_kwargs: pytest.fail("validate-only queried GPUs"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("validate-only launched a process"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SUPERVISOR_PATH), "--validate-only"],
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == (
        "tgvf.original-raw-direct-true1m-scoring-supervisor.v2"
    )
    assert payload["scorer"]["python"] == str(module.DEFAULT_PYTHON)
    assert module._judge_command(
        python_bin=module.DEFAULT_PYTHON, port=module.DEFAULT_JUDGE_PORT
    )[0] == str(module.DEFAULT_PYTHON)
