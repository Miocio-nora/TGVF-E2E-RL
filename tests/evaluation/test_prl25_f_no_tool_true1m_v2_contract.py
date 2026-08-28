from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tgvf_rl.evaluation.policy_no_tool_matched import (
    validate_no_tool_matched_processor,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREPARER_PATH = REPOSITORY_ROOT / "tools/prepare_prl25_f_no_tool_true1m_v2_arm.py"
SUPERVISOR_PATH = (
    REPOSITORY_ROOT / "tools/supervise_prl25_f_no_tool_true1m_v2_inference.py"
)
SCORING_FINALIZER_PATH = (
    REPOSITORY_ROOT / "tools/finalize_prl25_f_no_tool_true1m_v2_scoring.py"
)
SCORING_WRAPPER_PATH = (
    REPOSITORY_ROOT / "tools/supervise_prl25_f_no_tool_true1m_v2_scoring.sh"
)
MATCHED_SCORING_PATH = (
    REPOSITORY_ROOT / "tools/supervise_prl25_f_no_tool_matched_scoring.sh"
)
CONTROLLED_EXEC_PATH = REPOSITORY_ROOT / "tools/exec_with_controlled_toolchain.py"
QWEN3_PROCESSOR_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
HISTORIC_DUAL_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-"
    "32step-ws8/evaluation/"
    "PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preparer() -> ModuleType:
    return _load_module("prl25_f_true1m_v2_preparer_under_test", PREPARER_PATH)


@pytest.fixture(scope="module")
def supervisor() -> ModuleType:
    return _load_module("prl25_f_true1m_v2_supervisor_under_test", SUPERVISOR_PATH)


@pytest.fixture(scope="module")
def scoring_finalizer() -> ModuleType:
    return _load_module(
        "prl25_f_true1m_v2_scoring_finalizer_under_test",
        SCORING_FINALIZER_PATH,
    )


def _valid_headline(finalizer: ModuleType) -> dict[str, object]:
    components = {
        "vstar": 10.0,
        "hr_average_all": 20.0,
        "blink_single_180": 30.0,
        "ocr_mean": 40.0,
        "mmmu_single_269": 0.0,
        "mathvista": 60.0,
        "mathverse_five_version_macro": 70.0,
    }
    return {
        "schema_version": "tgvf.coredev-2511-macro-star.v1",
        "aggregation": "unweighted_mean_of_seven_percent_components",
        "components_percent": components,
        "ocr_language_components_percent": {
            "english": 30.0,
            "chinese": 50.0,
        },
        "mathverse_version_components_percent": {
            version: value
            for version, value in zip(
                finalizer.MATHVERSE_VERSIONS,
                (50.0, 60.0, 70.0, 80.0, 90.0),
                strict=True,
            )
        },
        "single_image_counts": {
            "blink": {"correct": 54, "count": 180},
            "mmmu": {"correct": 0, "count": 269},
        },
        "macro_star_percent": math.fsum(components.values()) / len(components),
    }


def _write_raw_summary(finalizer: ModuleType, root: Path, step: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = finalizer._summary_path(root.resolve(), step)
    path.parent.mkdir(parents=True, exist_ok=True)
    scoring_root = path.parent
    slices: list[dict[str, object]] = []
    for ordinal, dataset in enumerate(finalizer.DATASETS):
        dataset_root = scoring_root / dataset
        source_run_id = finalizer.SOURCE_RUN_IDS[step]
        source_run = dataset_root / finalizer.MODEL / source_run_id
        source_run.mkdir(parents=True)
        prediction_name = f"{finalizer.MODEL}_{dataset}.tsv"
        source_prediction = source_run / prediction_name
        source_prediction.write_text(
            f"index\tprediction\n{ordinal}\tstep-{step}-{dataset}\n",
            encoding="utf-8",
        )
        source_prediction_sha256 = hashlib.sha256(
            source_prediction.read_bytes()
        ).hexdigest()
        source_manifest = source_run / "final-answer-view-manifest.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract": "vlmevalkit-final-answer-view-v1",
                    "derived": {
                        "path": str(source_prediction),
                        "sha256": source_prediction_sha256,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source_status = source_run / "status.json"
        source_status.write_text(
            json.dumps(
                {
                    "eval_id": source_run_id,
                    "model_name": finalizer.MODEL,
                    "mode": "infer",
                    "reuse": False,
                    "reuse_aux": "infer",
                    "datasets": {
                        dataset: {
                            "status": "done",
                            "source_run": finalizer._evaluation_id(step),
                            "prediction_file": str(source_prediction),
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        destination_run_id = f"T20260828_G{step:02x}{ordinal:02x}"
        destination_run = dataset_root / finalizer.MODEL / destination_run_id
        destination_run.mkdir(parents=True)
        destination_prediction = destination_run / prediction_name
        destination_prediction.write_bytes(source_prediction.read_bytes())
        metrics = {"accepted": float(ordinal + 1)}
        primary_metric = "accepted"
        destination_status = destination_run / "status.json"
        destination_status.write_text(
            json.dumps(
                {
                    "eval_id": destination_run_id,
                    "model_name": finalizer.MODEL,
                    "mode": "eval",
                    "reuse": True,
                    "reuse_aux": "infer",
                    "datasets": {
                        dataset: {
                            "status": "done",
                            "source_run": source_run_id,
                            "prediction_file": prediction_name,
                            "primary_metric": primary_metric,
                            "metrics": metrics,
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        judge_artifacts: list[str] = []
        if dataset != "OCRBench_v2":
            judge = destination_run / f"{dataset}-judge-result.tsv"
            judge.write_text("index\thit\n0\t1\n", encoding="utf-8")
            judge_artifacts.append(str(judge))

        receipt = {
            "schema_version": finalizer.PINNED_RECEIPT_SCHEMA,
            "dataset": dataset,
            "model": finalizer.MODEL,
            "source_evaluation_id": finalizer._evaluation_id(step),
            "source_run_id": source_run_id,
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
            "source_prediction_path": str(source_prediction),
            "source_prediction_sha256": source_prediction_sha256,
            "destination_run_id": destination_run_id,
            "destination_status_path": str(destination_status),
            "destination_status_sha256": hashlib.sha256(
                destination_status.read_bytes()
            ).hexdigest(),
            "destination_prediction_path": str(destination_prediction),
            "destination_prediction_sha256": hashlib.sha256(
                destination_prediction.read_bytes()
            ).hexdigest(),
        }
        (dataset_root / "pinned-reuse-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )
        slices.append(
            {
                "dataset": dataset,
                "eval_id": destination_run_id,
                "status_path": str(destination_status),
                "prediction_file": str(destination_prediction),
                "prediction_sha256": receipt["destination_prediction_sha256"],
                "judge_artifacts": judge_artifacts,
                "metrics": metrics,
                "primary_metric": primary_metric,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "eval",
                "model": "Qwen3-VL-8B-Instruct",
                "sample_count": 2511,
                "slice_count": 7,
                "slices": slices,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _formal_root(finalizer: ModuleType, parent: Path) -> Path:
    return parent / finalizer.TOTAL_EVALUATION_ID


def _identity_payload(
    finalizer: ModuleType, schema: str, **fields: object
) -> dict[str, object]:
    unsigned = {"schema_version": schema, **fields}
    return {
        **unsigned,
        "identity_sha256": finalizer._canonical_sha256(unsigned),
    }


def _identity_binding(
    finalizer: ModuleType, path: Path, payload: dict[str, object]
) -> dict[str, object]:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity_sha256": payload["identity_sha256"],
    }


def _write_inference_source_fixture(
    finalizer: ModuleType, eval_root: Path
) -> tuple[dict[str, object], Path]:
    control = eval_root / "runtime/supervisor"
    control.mkdir(parents=True, exist_ok=True)
    runtime = _identity_payload(
        finalizer,
        "tgvf.prl25-f-runtime-environment-contract.v2",
        evaluation_id=finalizer.TOTAL_EVALUATION_ID,
    )
    runtime_path = control / "runtime-environment-contract.json"
    runtime_binding = _identity_binding(finalizer, runtime_path, runtime)
    launch = _identity_payload(
        finalizer,
        "tgvf.prl25-f-worker-launch-contract.v2",
        evaluation_id=finalizer.TOTAL_EVALUATION_ID,
    )
    launch_path = control / "worker-launch-contract.json"
    launch_binding = _identity_binding(finalizer, launch_path, launch)

    aggregate_arms: dict[str, object] = {}
    first_rank_path: Path | None = None
    for step in finalizer.STEPS:
        arm_root = eval_root / f"matched/step{step}"
        inference_root = arm_root / "inference"
        runtime_root = arm_root / "runtime"
        inference_root.mkdir(parents=True, exist_ok=True)
        runtime_root.mkdir(parents=True, exist_ok=True)
        evaluation_id = finalizer._evaluation_id(step)
        files: list[dict[str, object]] = []
        global_identities: list[dict[str, object]] = []
        for rank in range(4):
            rank_path = inference_root / f"rank-{rank}.jsonl"
            if first_rank_path is None:
                first_rank_path = rank_path
            ordinals = list(range(rank, 2240, 4))
            identities: list[dict[str, object]] = []
            rows: list[str] = []
            for ordinal in ordinals:
                unsigned_row = {
                    "evaluation_id": evaluation_id,
                    "ordinal": ordinal,
                    "dataset": "Fixture",
                    "index": str(ordinal),
                    "final_answer": f"answer-{step}-{ordinal}",
                }
                result_identity = finalizer._canonical_sha256(unsigned_row)
                row = {
                    **unsigned_row,
                    "result_identity_sha256": result_identity,
                }
                rows.append(json.dumps(row, sort_keys=True))
                identities.append(
                    {
                        "ordinal": ordinal,
                        "result_identity_sha256": result_identity,
                    }
                )
            rank_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            global_identities.extend(identities)
            files.append(
                {
                    "path": str(rank_path.resolve()),
                    "resolved_path": str(rank_path.resolve()),
                    "sha256": hashlib.sha256(rank_path.read_bytes()).hexdigest(),
                    "size_bytes": rank_path.stat().st_size,
                    "rank": rank,
                    "line_count": len(ordinals),
                    "ordinal_sequence_sha256": finalizer._canonical_sha256(ordinals),
                    "result_identity_sequence_sha256": (
                        finalizer._canonical_sha256(identities)
                    ),
                }
            )
        global_identities.sort(key=lambda item: int(item["ordinal"]))
        rank_tree = _identity_payload(
            finalizer,
            finalizer.RANK_TREE_SCHEMA,
            evaluation_id=evaluation_id,
            optimizer_step=step,
            evaluation_identity_sha256=f"{step + 1:064x}",
            task_manifest_sha256="a" * 64,
            world_size=4,
            row_count=2240,
            files=files,
            result_identity_sequence_sha256=finalizer._canonical_sha256(
                global_identities
            ),
        )
        status = _identity_payload(
            finalizer,
            finalizer.INFERENCE_STATUS_SCHEMA,
            evaluation_id=evaluation_id,
            optimizer_step=step,
            rank_tree=rank_tree,
        )
        status_path = runtime_root / "inference-status.json"
        status_binding = _identity_binding(finalizer, status_path, status)
        completion = _identity_payload(
            finalizer,
            finalizer.INFERENCE_COMPLETION_SCHEMA,
            evaluation_id=evaluation_id,
            optimizer_step=step,
            completed_single_image=2240,
            unsupported_multi_image=271,
            rank_tree_identity_sha256=rank_tree["identity_sha256"],
            result_identity_sequence_sha256=rank_tree[
                "result_identity_sequence_sha256"
            ],
            status_receipt=status_binding,
        )
        completion_path = runtime_root / "inference-complete.json"
        aggregate_arms[str(step)] = _identity_binding(
            finalizer, completion_path, completion
        )

        config_path = arm_root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "evaluation_image_max_pixels": 1_003_520,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        proof = {
            "configured_image_max_pixels": 1_003_520,
            "synthetic_native_source_pixel_area": 3_145_728,
            "synthetic_native_represented_pixel_area": 995_328,
            "synthetic_native_visual_token_count": 972,
            "runtime_mm_processor_kwargs": {
                "size": {"shortest_edge": 65_536, "longest_edge": 1_003_520}
            },
            "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
            "vllm_012_shallow_hashable": True,
            "nested_images_kwargs_present": False,
            "max_pixels_kwarg_present": False,
            "tool_schema_visible": False,
            "system_prompt_present": False,
        }
        proof_path = runtime_root / "true1m-processor-proof.json"
        proof_path.write_text(
            json.dumps(
                {
                    "schema_version": ("tgvf.prl25-f-true1m-processor-acceptance.v2"),
                    "evaluation_id": evaluation_id,
                    "optimizer_step": step,
                    "gpu_or_api_used": False,
                    "vllm_engine_constructed": False,
                    "proof": proof,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    aggregate = _identity_payload(
        finalizer,
        finalizer.MATCHED_INFERENCE_COMPLETION_SCHEMA,
        evaluation_id=finalizer.TOTAL_EVALUATION_ID,
        optimizer_steps=list(finalizer.STEPS),
        completed_single_image_per_step=2240,
        unsupported_multi_image_per_step=271,
        runtime_environment_contract=runtime_binding,
        worker_launch_contract=launch_binding,
        arms=aggregate_arms,
    )
    aggregate_path = control / "matched-inference-complete.json"
    _identity_binding(finalizer, aggregate_path, aggregate)
    assert first_rank_path is not None
    return aggregate, first_rank_path


def test_true1m_v2_plan_has_new_root_ids_and_weight_only_reuse(
    preparer: ModuleType,
) -> None:
    plan = preparer.load_true1m_v2_plan()

    assert plan["evaluation_id"].endswith("S0-S8-S16-S32-TRUE1M-V2")
    assert Path(plan["evaluation_root"]).name.endswith("TRUE1M-V2")
    assert "DUAL-V1" not in plan["evaluation_root"]
    assert plan["image_preprocessing"]["configured_image_max_pixels"] == 1_003_520
    assert (
        plan["image_preprocessing"]["qwen_fast_processor_default_max_pixels"]
        == 16_777_216
    )
    image = plan["image_preprocessing"]
    assert image["runtime_override_path"] == ("mm_processor_kwargs.size.longest_edge")
    assert image["forbid_nested_images_kwargs"] is True
    assert image["forbid_max_pixels_kwarg"] is True
    reuse = plan["weight_snapshot_reuse"]
    assert reuse["reuse_inference_rows"] is False
    assert reuse["reuse_scoring_rows"] is False
    assert reuse["allowed_files"] == [
        "full-model-snapshot.json",
        "full-model-materialization.json",
    ]
    assert [arm["optimizer_step"] for arm in plan["arms"]] == [0, 8, 16, 32]
    assert len({arm["evaluation_id"] for arm in plan["arms"]}) == 4
    assert all("TRUE1M-V2" in arm["evaluation_id"] for arm in plan["arms"])
    assert plan["paired_rng"]["master_seed"] == 42
    assert plan["paired_rng"]["mode"] == ("common_random_numbers_per_task_turn")
    assert plan["execution"]["toolchain_environment"] == (
        preparer.TOOLCHAIN_ENVIRONMENT
    )


def test_no_tool_checkpoint_owner_exposes_full_model_sampling_identity(
    preparer: ModuleType,
) -> None:
    plan = preparer.load_true1m_v2_plan()
    owner = REPOSITORY_ROOT / plan["checkpoint_owner"]["config_path"]
    run = load_policy_e2e_smoke_run_config(owner, allow_external_agent_loop_config=True)

    identity = dict(run.policy.sampling.identity_record())
    assert identity["max_response_length"] == 20_480
    assert identity["temperature"] == 1.0
    assert identity["stop_token_ids"] == [151_645]
    assert identity["stop_strings"] == []
    assert identity["include_stop_str_in_output"] is True
    assert identity["ignore_eos"] is False


def test_true1m_v2_plan_rejects_poisoned_toolchain_contract(
    tmp_path: Path, preparer: ModuleType
) -> None:
    payload = json.loads(preparer.PLAN_PATH.read_text(encoding="utf-8"))
    payload["execution"]["toolchain_environment"]["CC"] = "/poison/conda/bin/cc"
    attacked = tmp_path / "attacked-plan.json"
    attacked.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution contract"):
        preparer.load_true1m_v2_plan(attacked)


@pytest.mark.parametrize("step", (0, 8, 16, 32))
def test_true1m_v2_reuses_only_bound_shared_model_closure(
    preparer: ModuleType, step: int
) -> None:
    plan = preparer.load_true1m_v2_plan()
    paths = preparer.arm_paths(plan, step)
    receipt = preparer._weight_reuse_receipt(plan, paths)

    assert receipt["optimizer_step"] == step
    assert receipt["inference_rows_reused"] is False
    assert receipt["scoring_rows_reused"] is False
    assert receipt["reuse_kind"] == (
        "full_model_snapshot_and_materialized_weight_tree_only"
    )
    for field in (
        "source_snapshot_manifest_path",
        "source_materialization_receipt_path",
        "source_model_path",
    ):
        parts = Path(receipt[field]).parts
        assert "inference" not in parts
        assert "scoring" not in parts
    assert paths.output_root != paths.source_shared_root
    assert "TRUE1M-V2" in str(paths.output_root)


def test_true1m_v2_supervisor_schedules_two_four_rank_arms_per_round(
    tmp_path: Path, supervisor: ModuleType
) -> None:
    plan = supervisor.load_true1m_v2_plan()
    launches = supervisor._worker_launches(
        plan,
        gpu_ids=tuple(range(8)),
        evaluation_root=tmp_path,
    )
    rounds = supervisor._pair_schedule(launches)

    assert len(launches) == 16
    assert [len(round_launches) for round_launches in rounds] == [8, 8]
    assert [{item.step for item in round_launches} for round_launches in rounds] == [
        {0, 8},
        {16, 32},
    ]
    for round_launches in rounds:
        assert {item.gpu_id for item in round_launches} == set(range(8))
    for launch in launches:
        assert launch.command[-4:] == (
            "--rank",
            str(launch.rank),
            "--world-size",
            "4",
        )
        assert launch.environment["CUDA_VISIBLE_DEVICES"] == str(launch.gpu_id)
        for name, expected in supervisor.FIXED_RUNTIME_ENVIRONMENT.items():
            assert launch.environment[name] == expected
        for variable, family in (
            ("VLLM_CACHE_ROOT", "vllm"),
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ):
            cache = Path(launch.environment[variable])
            assert cache == (
                tmp_path
                / f"matched/step{launch.step}/runtime/cache/{family}"
                / f"rank-{launch.rank}"
            )
            assert cache.is_dir()


def test_true1m_v2_runtime_environment_rejects_polluted_parent_shell(
    monkeypatch: pytest.MonkeyPatch, supervisor: ModuleType
) -> None:
    poisoned = {
        "CC": "/poison/conda-cc",
        "CXX": "/poison/conda-cxx",
        "CFLAGS": "-I/poison/revisit-vlm/include",
        "CPPFLAGS": "-I/poison/revisit-vlm/include",
        "CXXFLAGS": "-I/poison/revisit-vlm/include",
        "LD": "/poison/revisit-vlm/bin/ld",
        "LDFLAGS": "-L/poison/revisit-vlm/lib",
        "CMAKE_ARGS": "-DCMAKE_PREFIX_PATH=/poison/revisit-vlm",
        "CPATH": "/poison/include",
        "LIBRARY_PATH": "/poison/lib",
        "PATH": "/poison/bin",
        "PYTHONPATH": "/poison/python",
        "CONDA_PREFIX": "/poison/conda",
        "_CONDA_ROOT": "/poison/conda-root",
    }
    for name, value in poisoned.items():
        monkeypatch.setenv(name, value)

    environment = supervisor._base_environment()

    for name, expected in supervisor.FIXED_RUNTIME_ENVIRONMENT.items():
        assert environment[name] == expected
    assert all(
        name not in environment for name in supervisor.PURGED_ONLY_TOOLCHAIN_ENVIRONMENT
    )
    assert all(
        not any(
            name.startswith(prefix) for prefix in supervisor.PURGED_ENVIRONMENT_PREFIXES
        )
        for name in environment
    )
    assert all(value not in environment.values() for value in poisoned.values())
    assert "revisit-vlm" not in environment["CC"]
    assert "revisit-vlm" not in environment["CXX"]


def test_true1m_v2_worker_contract_rejects_unrecorded_toolchain_pollution(
    tmp_path: Path, supervisor: ModuleType
) -> None:
    plan = supervisor.load_true1m_v2_plan()
    launch = supervisor._worker_launches(
        plan,
        gpu_ids=tuple(range(8)),
        evaluation_root=tmp_path,
    )[0]
    clean = supervisor._worker_environment_record(launch)
    unsigned = {
        key: value
        for key, value in clean.items()
        if key != "environment_identity_sha256"
    }
    assert clean["environment_identity_sha256"] == supervisor._canonical_sha256(
        unsigned
    )
    assert clean["environment_purge_verification"] == {
        "controlled_names": list(supervisor.TOOLCHAIN_ENVIRONMENT),
        "purged_exact_absent": list(supervisor.PURGED_ONLY_TOOLCHAIN_ENVIRONMENT),
        "purged_prefixes_absent": list(supervisor.PURGED_ENVIRONMENT_PREFIXES),
        "verified": True,
    }

    launch.environment["CFLAGS"] = "-I/poison/revisit-vlm/include"
    with pytest.raises(RuntimeError, match="retained forbidden environment"):
        supervisor._worker_environment_record(launch)


def test_true1m_v2_runtime_environment_contract_binds_toolchain_artifacts(
    supervisor: ModuleType,
) -> None:
    plan = supervisor.load_true1m_v2_plan()
    contract = supervisor._runtime_environment_contract(plan)

    assert contract["schema_version"] == supervisor.RUNTIME_ENVIRONMENT_SCHEMA
    assert contract["environment"] == supervisor.FIXED_RUNTIME_ENVIRONMENT
    assert contract[
        "environment_purge_policy"
    ] == supervisor.controlled_toolchain_contract(supervisor.TOOLCHAIN_ENVIRONMENT)
    assert contract["environment_purge_verification"] == {
        "controlled_names": list(supervisor.TOOLCHAIN_ENVIRONMENT),
        "purged_exact_absent": list(supervisor.PURGED_ONLY_TOOLCHAIN_ENVIRONMENT),
        "purged_prefixes_absent": list(supervisor.PURGED_ENVIRONMENT_PREFIXES),
        "verified": True,
    }
    assert contract["identity_sha256"] == supervisor._canonical_sha256(
        {key: value for key, value in contract.items() if key != "identity_sha256"}
    )
    artifacts = contract["artifacts"]
    assert set(artifacts) == {
        "python",
        "cc",
        "cxx",
        "python_h",
        "python_pyconfig_h",
        "platform_pyconfig_h",
        "runner",
        "preparer",
        "supervisor",
        "plan",
    }
    assert artifacts["cc"]["path"] == "/usr/bin/gcc"
    assert artifacts["cxx"]["path"] == "/usr/bin/g++"
    assert all(len(item["sha256"]) == 64 for item in artifacts.values())


def test_true1m_v2_formal_status_closes_and_rejects_resigned_env_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervisor: ModuleType,
) -> None:
    plan = deepcopy(supervisor.load_true1m_v2_plan())
    plan["evaluation_root"] = str(tmp_path / "PRL25-F-NO-TOOL-RL-MATCHED-TRUE1M-V2")
    control = Path(plan["evaluation_root"]) / "runtime/supervisor"
    control.mkdir(parents=True)
    runtime_path = control / supervisor.RUNTIME_ENVIRONMENT_FILENAME
    runtime = supervisor._runtime_environment_contract(plan)
    supervisor._write_immutable_json(runtime_path, runtime)
    launches = supervisor._worker_launches(plan, gpu_ids=tuple(range(8)))
    launch = supervisor._worker_launch_contract(
        plan,
        launches,
        runtime_environment_identity_sha256=runtime["identity_sha256"],
    )
    launch_path = control / supervisor.WORKER_LAUNCH_FILENAME
    supervisor._write_immutable_json(launch_path, launch)
    workers = launch["workers"]

    def rank_tree(_plan: dict[str, object], *, step: int) -> dict[str, object]:
        return supervisor._with_identity(
            {
                "schema_version": supervisor.RANK_TREE_SCHEMA,
                "evaluation_id": supervisor.arm_paths(plan, step).evaluation_id,
                "optimizer_step": step,
                "evaluation_identity_sha256": f"{step:064x}",
                "task_manifest_sha256": "a" * 64,
                "world_size": 4,
                "row_count": 2240,
                "files": [],
                "result_identity_sequence_sha256": "b" * 64,
            }
        )

    monkeypatch.setattr(supervisor, "_rank_tree_evidence", rank_tree)

    for step in supervisor.STEPS:
        paths = supervisor.arm_paths(plan, step)
        step_rank_tree = rank_tree(plan, step=step)
        status = supervisor._with_identity(
            {
                "schema_version": supervisor.INFERENCE_STATUS_SCHEMA,
                "evaluation_id": paths.evaluation_id,
                "optimizer_step": step,
                "runner_status": {
                    "evaluation_id": paths.evaluation_id,
                    "completed_single_image": 2240,
                    "total_single_image": 2240,
                    "remaining_single_image": 0,
                    "multi_image_pending_protocol_decision": 271,
                },
                "runtime_environment_identity_sha256": runtime["identity_sha256"],
                "worker_launch_identity_sha256": launch["identity_sha256"],
                "worker_environment_identity_sha256": [
                    worker["environment_identity_sha256"]
                    for worker in workers
                    if worker["optimizer_step"] == step
                ],
                "rank_tree": step_rank_tree,
            }
        )
        status_path = paths.output_root / "runtime/inference-status.json"
        supervisor._write_immutable_json(status_path, status)
        completion = supervisor._with_identity(
            {
                "schema_version": supervisor.INFERENCE_COMPLETION_SCHEMA,
                "evaluation_id": paths.evaluation_id,
                "optimizer_step": step,
                "completed_single_image": 2240,
                "unsupported_multi_image": 271,
                "runtime_environment_identity_sha256": runtime["identity_sha256"],
                "worker_launch_identity_sha256": launch["identity_sha256"],
                "rank_tree_identity_sha256": step_rank_tree["identity_sha256"],
                "result_identity_sequence_sha256": step_rank_tree[
                    "result_identity_sequence_sha256"
                ],
                "status_receipt": supervisor._file_binding(
                    status_path,
                    identity_sha256=status["identity_sha256"],
                ),
            }
        )
        supervisor._write_immutable_json(
            paths.output_root / "runtime/inference-complete.json", completion
        )

    aggregate = supervisor._matched_completion_receipt(
        plan,
        runtime_contract_path=runtime_path,
        runtime_contract=runtime,
        launch_contract_path=launch_path,
        launch_contract=launch,
    )
    aggregate_path = control / "matched-inference-complete.json"
    supervisor._write_immutable_json(aggregate_path, aggregate)
    assert supervisor._validate_matched_inference_completion(plan) == aggregate

    attacked_path = (
        supervisor.arm_paths(plan, 8).output_root / "runtime/inference-status.json"
    )
    attacked = json.loads(attacked_path.read_text(encoding="utf-8"))
    attacked.pop("identity_sha256")
    attacked["runtime_environment_identity_sha256"] = "0" * 64
    attacked = supervisor._with_identity(attacked)
    attacked_path.write_text(
        json.dumps(attacked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="formal inference status"):
        supervisor._validate_matched_inference_completion(plan)


@pytest.mark.parametrize(
    "gpu_ids",
    [tuple(range(7)), (0, 1, 2, 3, 4, 5, 6, 6), tuple(range(9))],
)
def test_true1m_v2_supervisor_rejects_ambiguous_gpu_sets(
    supervisor: ModuleType, gpu_ids: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        supervisor._validate_gpu_ids(gpu_ids)


def test_prepare_only_supersedes_stale_failure_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervisor: ModuleType
) -> None:
    plan = {
        "evaluation_id": "test-no-tool-true1m",
        "evaluation_root": str(tmp_path),
        "execution": {"toolchain_environment": dict(supervisor.TOOLCHAIN_ENVIRONMENT)},
        "arms": [
            {"optimizer_step": 0, "gpu_group": "left"},
            {"optimizer_step": 8, "gpu_group": "right"},
            {"optimizer_step": 16, "gpu_group": "left"},
            {"optimizer_step": 32, "gpu_group": "right"},
        ],
    }
    failure = tmp_path / "runtime/supervisor/failed.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"message":"obsolete"}\n', encoding="utf-8")
    prepared: list[tuple[int, tuple[int, ...]]] = []
    monkeypatch.setattr(supervisor, "load_true1m_v2_plan", lambda _path: plan)
    monkeypatch.setattr(
        supervisor,
        "_prepare_arm",
        lambda _plan, *, step, gpu_ids: prepared.append((step, gpu_ids)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SUPERVISOR_PATH), "--prepare-only", "--gpu-ids", *map(str, range(8))],
    )

    assert supervisor.main() == 0
    assert prepared == [
        (0, (0, 1, 2, 3)),
        (8, (4, 5, 6, 7)),
        (16, (0, 1, 2, 3)),
        (32, (4, 5, 6, 7)),
    ]
    assert not failure.exists()


def test_true1m_v2_validate_gate_requires_exact_real_processor_geometry(
    supervisor: ModuleType,
) -> None:
    plan = supervisor.load_true1m_v2_plan()
    proof = {
        "configured_image_max_pixels": 1_003_520,
        "processor_image_size": {
            "shortest_edge": 65_536,
            "longest_edge": 16_777_216,
        },
        "effective_processor_image_size": {
            "shortest_edge": 65_536,
            "longest_edge": 1_003_520,
        },
        "synthetic_native_source_pixel_area": 3_145_728,
        "synthetic_native_represented_pixel_area": 995_328,
        "synthetic_native_visual_token_count": 972,
        "runtime_mm_processor_kwargs": {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": 1_003_520,
            }
        },
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "vllm_012_shallow_hashable": True,
        "nested_images_kwargs_present": False,
        "max_pixels_kwarg_present": False,
        "tool_schema_visible": False,
        "system_prompt_present": False,
    }
    validation = {
        "evaluation_id": plan["arms"][0]["evaluation_id"],
        "optimizer_step": 0,
        "gpu_or_api_used": False,
        "vllm_engine_constructed": False,
        "no_tool_matched_processor_proof": proof,
    }

    accepted = supervisor._validate_processor_proof(validation, plan, step=0)
    assert accepted["proof"]["synthetic_native_represented_pixel_area"] == 995_328

    proof["synthetic_native_represented_pixel_area"] = 3_145_728
    with pytest.raises(RuntimeError, match="real-Qwen true1M processor proof"):
        supervisor._validate_processor_proof(validation, plan, step=0)


def test_real_qwen_no_tool_true1m_probe_is_995328_pixels() -> None:
    if not QWEN3_PROCESSOR_PATH.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    transformers = pytest.importorskip(
        "transformers", reason="accepted local Qwen3 processor is unavailable"
    )
    processor = transformers.AutoProcessor.from_pretrained(
        QWEN3_PROCESSOR_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    proof = validate_no_tool_matched_processor(
        processor,
        tokenizer_length=151_669,
        image_max_pixels=1_003_520,
    )

    assert proof["synthetic_native_source_pixel_area"] == 3_145_728
    assert proof["synthetic_native_represented_pixel_area"] == 995_328
    assert proof["synthetic_native_visual_token_count"] == 972
    assert proof["synthetic_native_represented_pixel_area"] <= 1_003_520
    assert proof["processor_image_size"]["longest_edge"] == 16_777_216
    assert proof["effective_processor_image_size"]["longest_edge"] == 1_003_520
    assert proof["runtime_mm_processor_kwargs"] == {
        "size": {"shortest_edge": 65_536, "longest_edge": 1_003_520}
    }
    assert proof["vllm_012_shallow_hashable"] is True
    assert proof["nested_images_kwargs_present"] is False
    assert proof["max_pixels_kwarg_present"] is False


def test_real_vllm_012_qwen_processor_accepts_flat_size_true1m() -> None:
    if not QWEN3_PROCESSOR_PATH.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    vllm_config = pytest.importorskip(
        "vllm.config", reason="accepted vLLM 0.12 runtime is unavailable"
    )
    vllm_multimodal = pytest.importorskip(
        "vllm.multimodal", reason="accepted vLLM 0.12 runtime is unavailable"
    )
    model = str(QWEN3_PROCESSOR_PATH)
    config = vllm_config.ModelConfig(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        max_model_len=32_768,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_cache_gb=0,
    )
    processor = vllm_multimodal.MULTIMODAL_REGISTRY.create_processor(config, cache=None)
    image = pytest.importorskip("PIL.Image").new("RGB", (2048, 1536), (10, 20, 30))
    try:
        output = processor.apply(
            "<|vision_start|><|image_pad|><|vision_end|>",
            {"image": image},
            {
                "size": {
                    "shortest_edge": 65_536,
                    "longest_edge": 1_003_520,
                }
            },
        )
    finally:
        image.close()

    grid = output["mm_kwargs"]["image"][0]["image_grid_thw"].data.tolist()
    assert grid == [1, 54, 72]
    assert grid[1] * grid[2] * 16**2 == 995_328
    assert output["mm_placeholders"]["image"][0].length == 972
    assert len(output["mm_hashes"]["image"]) == 1


def test_true1m_v2_scoring_wrapper_enables_formal_headline_closure() -> None:
    wrapper = SCORING_WRAPPER_PATH.read_text(encoding="utf-8")
    matched = MATCHED_SCORING_PATH.read_text(encoding="utf-8")

    assert "PRL25_F_MATCHED_REQUIRED_IMAGE_MAX_PIXELS=1003520" in wrapper
    assert "finalize_prl25_f_no_tool_true1m_v2_scoring.py" in matched
    assert '--eval-root "$eval_root" --step "$step"' in matched
    assert '--eval-root "$eval_root" --finalize-all' in matched
    assert 'complete_receipt="$control_root/matched-scoring-complete.json"' in matched
    assert "--verify-completion-only" in matched
    assert "exec_with_controlled_toolchain.py" in matched
    assert '--contract-out "$control_root/judge-toolchain-contract.json"' in matched


def test_no_tool_judge_wrapper_records_clean_shared_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executor = _load_module(
        "controlled_toolchain_exec_under_test", CONTROLLED_EXEC_PATH
    )
    for name in ("CFLAGS", "CPPFLAGS", "CXXFLAGS", "LD", "LDFLAGS"):
        monkeypatch.setenv(name, f"/poison/revisit-vlm/{name}")
    monkeypatch.setenv("CONDA_PREFIX", "/poison/conda")
    contract_path = tmp_path / "judge-toolchain-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(CONTROLLED_EXEC_PATH),
            "--python-environment-root",
            "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312",
            "--python-header-root",
            (
                "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/"
                ".deps/python312-dev/root/usr/include"
            ),
            "--environment",
            "CUDA_VISIBLE_DEVICES=0,1",
            "--contract-out",
            str(contract_path),
            "--validate-only",
            "--",
            "/bin/true",
        ],
    )

    assert executor.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload["verification"]["verified"] is True
    assert all(
        name in payload["toolchain"]["purged_exact"]
        for name in ("CFLAGS", "CPPFLAGS", "CXXFLAGS", "LD", "LDFLAGS")
    )


def test_inference_source_closure_binds_rank_bytes_and_processor_proofs(
    tmp_path: Path,
    scoring_finalizer: ModuleType,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    aggregate, first_rank_path = _write_inference_source_fixture(
        scoring_finalizer, eval_root
    )

    closure = scoring_finalizer.validate_inference_source_closure(eval_root)

    assert (
        closure["matched_inference_completion_identity_sha256"]
        == aggregate["identity_sha256"]
    )
    assert len(closure["arms"]) == 4
    assert all(
        arm["processor_proof_sha256"] and arm["rank_tree_identity_sha256"]
        for arm in closure["arms"]
    )

    original_rank_bytes = first_rank_path.read_bytes()
    first_rank_path.write_bytes(
        original_rank_bytes.replace(b"answer-0-0", b"answer-X-0", 1)
    )
    with pytest.raises(RuntimeError, match="rank0 byte binding"):
        scoring_finalizer.validate_inference_source_closure(eval_root)
    first_rank_path.write_bytes(original_rank_bytes)

    proof_path = eval_root / "matched/step16/runtime/true1m-processor-proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["proof"]["synthetic_native_represented_pixel_area"] = 3_145_728
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="S16 processor proof differs"):
        scoring_finalizer.validate_inference_source_closure(eval_root)


@pytest.mark.parametrize(
    "attack",
    (
        "missing_component",
        "wrong_blink_count",
        "wrong_mmmu_count",
        "missing_mathverse_version",
        "wrong_ocr_mean",
        "wrong_mathverse_macro",
        "wrong_macro",
        "non_finite_macro",
    ),
)
def test_true1m_v2_headline_rejects_population_and_metric_attacks(
    scoring_finalizer: ModuleType, attack: str
) -> None:
    headline = deepcopy(_valid_headline(scoring_finalizer))
    if attack == "missing_component":
        headline["components_percent"].pop("mathvista")  # type: ignore[union-attr]
    elif attack == "wrong_blink_count":
        headline["single_image_counts"]["blink"]["count"] = 179  # type: ignore[index]
    elif attack == "wrong_mmmu_count":
        headline["single_image_counts"]["mmmu"]["count"] = 268  # type: ignore[index]
    elif attack == "missing_mathverse_version":
        headline["mathverse_version_components_percent"].pop("Vision Dominant")  # type: ignore[union-attr]
    elif attack == "wrong_ocr_mean":
        headline["components_percent"]["ocr_mean"] = 41.0  # type: ignore[index]
    elif attack == "wrong_mathverse_macro":
        headline["components_percent"]["mathverse_five_version_macro"] = 71.0  # type: ignore[index]
    elif attack == "wrong_macro":
        headline["macro_star_percent"] = 99.0
    elif attack == "non_finite_macro":
        headline["macro_star_percent"] = float("nan")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(attack)

    with pytest.raises(RuntimeError):
        scoring_finalizer.validate_headline(headline)


def test_true1m_v2_scoring_writes_and_revalidates_all_completion_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    headline = _valid_headline(scoring_finalizer)
    inference_closure = {
        "schema_version": scoring_finalizer.INFERENCE_SOURCE_CLOSURE_SCHEMA,
        "evaluation_contract": scoring_finalizer.TOTAL_EVALUATION_ID,
        "matched_inference_completion_path": str(
            eval_root / "runtime/supervisor/matched-inference-complete.json"
        ),
        "matched_inference_completion_sha256": "a" * 64,
        "matched_inference_completion_identity_sha256": "b" * 64,
        "arms": [],
        "identity_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: deepcopy(headline),
    )
    monkeypatch.setattr(
        scoring_finalizer,
        "validate_inference_source_closure",
        lambda _root: deepcopy(inference_closure),
    )
    for step in scoring_finalizer.STEPS:
        _write_raw_summary(scoring_finalizer, eval_root, step)
        completion = scoring_finalizer.finalize_step(eval_root, step)
        assert completion["status"] == "complete"
        assert completion["max_pixels"] == 1_003_520
        assert completion["blink_single_image_count"] == 180
        assert completion["mmmu_single_image_count"] == 269
        assert completion["mathverse_versions"] == list(
            scoring_finalizer.MATHVERSE_VERSIONS
        )

    aggregate = scoring_finalizer.finalize_all(eval_root)
    assert aggregate["status"] == "complete"
    assert aggregate["optimizer_steps"] == [0, 8, 16, 32]
    assert len(aggregate["arms"]) == 4
    assert aggregate["inference_source_closure"] == inference_closure
    aggregate_path = (
        eval_root / "runtime/scoring-supervisor/matched-scoring-complete.json"
    )
    assert aggregate_path.is_file()

    step8 = scoring_finalizer._step_completion_path(eval_root.resolve(), 8)
    tampered = json.loads(step8.read_text(encoding="utf-8"))
    tampered["summary_sha256"] = "0" * 64
    step8.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        scoring_finalizer.validate_aggregate_completion(aggregate, eval_root=eval_root)


def test_true1m_v2_scoring_rejects_incomplete_four_arm_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: deepcopy(_valid_headline(scoring_finalizer)),
    )
    for step in (0, 8, 32):
        _write_raw_summary(scoring_finalizer, eval_root, step)
        scoring_finalizer.finalize_step(eval_root, step)

    with pytest.raises(RuntimeError):
        scoring_finalizer.finalize_all(eval_root)


def test_true1m_v2_finalizer_rejects_foreign_evaluation_root_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
) -> None:
    foreign_root = tmp_path / "PRL25-F-NO-TOOL-RL-COREDEV2511-DUAL-V1"
    summary = _write_raw_summary(scoring_finalizer, foreign_root, 0)
    before = hashlib.sha256(summary.read_bytes()).hexdigest()
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: pytest.fail("foreign root reached headline extraction"),
    )

    with pytest.raises(RuntimeError, match="evaluation-root basename"):
        scoring_finalizer.finalize_step(foreign_root, 0)
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == before


def test_true1m_v2_finalizer_read_only_rejects_historic_dual_v1(
    scoring_finalizer: ModuleType,
) -> None:
    summary = (
        HISTORIC_DUAL_ROOT / "matched/step0/scoring/coredev-official-v1/"
        "coredev-2511-eval-summary.json"
    )
    if not summary.is_file():
        pytest.skip("historic DUAL-V1 attack fixture is unavailable")
    before = hashlib.sha256(summary.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="evaluation-root basename"):
        scoring_finalizer.finalize_step(HISTORIC_DUAL_ROOT, 0)
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == before


def test_true1m_v2_finalizer_rejects_foreign_dual_summary_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    accepted_summary = _write_raw_summary(scoring_finalizer, eval_root, 0)
    foreign_summary = _write_raw_summary(
        scoring_finalizer, tmp_path / "foreign-DUAL-V1", 0
    )
    accepted_summary.write_bytes(foreign_summary.read_bytes())
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: pytest.fail("foreign summary reached headline extraction"),
    )

    with pytest.raises(RuntimeError, match="escapes|binding"):
        scoring_finalizer.finalize_step(eval_root, 0)


def test_true1m_v2_finalizer_rejects_s0_s8_summary_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    step0_summary = _write_raw_summary(scoring_finalizer, eval_root, 0)
    step8_summary = _write_raw_summary(scoring_finalizer, eval_root, 8)
    step0_summary.write_bytes(step8_summary.read_bytes())
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: pytest.fail("swapped summary reached headline extraction"),
    )

    with pytest.raises(RuntimeError, match="escapes|binding"):
        scoring_finalizer.finalize_step(eval_root, 0)


@pytest.mark.parametrize(
    "attack",
    (
        "receipt_step_swap",
        "destination_prediction_bytes",
        "foreign_judge_path",
        "summary_destination_path",
    ),
)
def test_true1m_v2_finalizer_rejects_receipt_and_path_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_finalizer: ModuleType,
    attack: str,
) -> None:
    eval_root = _formal_root(scoring_finalizer, tmp_path)
    summary_path = _write_raw_summary(scoring_finalizer, eval_root, 0)
    scoring_root = summary_path.parent
    dataset = scoring_finalizer.DATASETS[0]
    receipt_path = scoring_root / dataset / "pinned-reuse-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if attack == "receipt_step_swap":
        receipt["source_evaluation_id"] = scoring_finalizer._evaluation_id(8)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif attack == "destination_prediction_bytes":
        Path(receipt["destination_prediction_path"]).write_text(
            "tampered\n", encoding="utf-8"
        )
    elif attack == "foreign_judge_path":
        foreign = tmp_path / "foreign-judge.tsv"
        foreign.write_text("index\thit\n0\t1\n", encoding="utf-8")
        summary["slices"][0]["judge_artifacts"] = [str(foreign)]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif attack == "summary_destination_path":
        summary["slices"][0]["prediction_file"] = receipt["source_prediction_path"]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(attack)
    monkeypatch.setattr(
        scoring_finalizer,
        "extract_coredev_macro_star",
        lambda _summary: pytest.fail("source attack reached headline extraction"),
    )

    with pytest.raises(RuntimeError):
        scoring_finalizer.finalize_step(eval_root, 0)
