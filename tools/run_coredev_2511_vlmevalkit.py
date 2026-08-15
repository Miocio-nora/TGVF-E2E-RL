"""Run pinned VLMEvalKit with official aliases bound to immutable CoreDev slices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
PINNED_ARTIFACTS = (
    REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
)
DIRECT_BASELINE_CONFIG = (
    REPOSITORY_ROOT / "configs/evaluation/coredev_2511_qwen3_direct_v1.json"
)
JUDGE_SERVICE_CONFIG = (
    REPOSITORY_ROOT / "configs/evaluation/qwen25_72b_judge_service_v1.json"
)
_EVAL_RUN_ID = re.compile(r"(?:T\d{8}-\d{6}|T\d{8}_G[0-9a-fA-F]+)")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _PinnedReuseContract:
    work_dir: Path
    model_name: str
    dataset_name: str
    source_evaluation_id: str
    source_run_id: str
    source_run_dir: Path
    source_status_path: Path
    manifest_path: Path
    prediction_path: Path
    prediction_sha256: str

    @property
    def model_root(self) -> Path:
        return self.work_dir / self.model_name


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _status_prediction_path(value: object, *, run_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("VLMEvalKit status has no prediction_file")
    path = Path(value)
    return (path if path.is_absolute() else run_dir / path).resolve()


def _load_pinned_reuse_contract(
    *,
    work_dir: str | Path,
    model_name: str,
    dataset_name: str,
    source_run_id: str,
    manifest_path: str | Path,
) -> _PinnedReuseContract:
    """Bind scoring reuse to one immutable synthetic inference view."""

    if _EVAL_RUN_ID.fullmatch(source_run_id) is None:
        raise RuntimeError("pinned reuse source_run_id is not a VLMEvalKit run ID")
    work = Path(work_dir).resolve()
    source_run_dir = work / model_name / source_run_id
    expected_manifest = source_run_dir / "final-answer-view-manifest.json"
    manifest = Path(manifest_path)
    manifest = (manifest if manifest.is_absolute() else Path.cwd() / manifest).resolve()
    if manifest != expected_manifest.resolve():
        raise RuntimeError("pinned reuse manifest is outside its exact source run")
    expected_prediction = source_run_dir / f"{model_name}_{dataset_name}.tsv"
    status_path = source_run_dir / "status.json"
    for label, path in (
        ("source run", source_run_dir),
        ("manifest", manifest),
        ("prediction", expected_prediction),
        ("status", status_path),
    ):
        if path.is_symlink() or not path.exists():
            raise RuntimeError(f"pinned reuse {label} is missing or is a symlink")
    if not source_run_dir.is_dir() or not all(
        path.is_file() for path in (manifest, expected_prediction, status_path)
    ):
        raise RuntimeError("pinned reuse source record types differ")

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    derived = (
        manifest_payload.get("derived") if isinstance(manifest_payload, dict) else None
    )
    if not isinstance(derived, dict):
        raise RuntimeError("pinned reuse manifest has no derived record")
    derived_sha256 = derived.get("sha256")
    if (
        Path(str(derived.get("path", ""))).resolve() != expected_prediction.resolve()
        or not isinstance(derived_sha256, str)
        or _SHA256.fullmatch(derived_sha256) is None
        or _sha256_file(expected_prediction) != derived_sha256
    ):
        raise RuntimeError("pinned reuse prediction differs from immutable manifest")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    dataset_status = (
        status.get("datasets", {}).get(dataset_name)
        if isinstance(status, dict)
        else None
    )
    if (
        not isinstance(dataset_status, dict)
        or status.get("eval_id") != source_run_id
        or status.get("mode") != "infer"
        or status.get("reuse_aux") != "infer"
        or dataset_status.get("status") != "done"
        or not isinstance(dataset_status.get("source_run"), str)
        or not dataset_status["source_run"]
        or _status_prediction_path(
            dataset_status.get("prediction_file"), run_dir=source_run_dir
        )
        != expected_prediction.resolve()
    ):
        raise RuntimeError("pinned reuse synthetic source status differs")
    source_evaluation_id = dataset_status["source_run"]

    return _PinnedReuseContract(
        work_dir=work,
        model_name=model_name,
        dataset_name=dataset_name,
        source_evaluation_id=source_evaluation_id,
        source_run_id=source_run_id,
        source_run_dir=source_run_dir,
        source_status_path=status_path,
        manifest_path=manifest,
        prediction_path=expected_prediction,
        prediction_sha256=derived_sha256,
    )


def _eval_run_ids(model_root: Path) -> set[str]:
    if not model_root.is_dir():
        return set()
    return {
        candidate.name
        for candidate in model_root.iterdir()
        if candidate.is_dir() and _EVAL_RUN_ID.fullmatch(candidate.name) is not None
    }


def _allocate_destination_run_id(contract: _PinnedReuseContract) -> str:
    """Choose and later patch in one exact destination ID for this invocation."""

    date = datetime.now().strftime("%Y%m%d")
    for counter in range(1024):
        nonce = (
            f"{contract.source_run_id}\0{contract.dataset_name}\0"
            f"{os.getpid()}\0{time.time_ns()}\0{counter}"
        )
        candidate = f"T{date}_G{sha256(nonce.encode('utf-8')).hexdigest()}"
        if not (contract.model_root / candidate).exists():
            return candidate
    raise RuntimeError("could not allocate a unique pinned scoring destination")


def _install_pinned_reuse_selector(
    contract: _PinnedReuseContract,
    *,
    destination_run_id: str,
    smp_module: Any,
    smp_file_module: Any,
) -> Callable[..., dict[str, Any]]:
    """Force VLMEvalKit to reuse exactly the bound synthetic run."""

    def select_pinned_reuse_run(
        pred_root_meta: str,
        eval_id: str,
        model_name: str,
        dataset_name: str,
        result_file: str,
        infer_aux_file_names: set[str],
    ) -> dict[str, Any]:
        destination = contract.model_root / destination_run_id
        expected_result = destination / f"{model_name}_{dataset_name}.tsv"
        if (
            Path(pred_root_meta).resolve() != contract.model_root.resolve()
            or eval_id != destination_run_id
            or model_name != contract.model_name
            or dataset_name != contract.dataset_name
            or Path(result_file).resolve() != expected_result.resolve()
        ):
            raise RuntimeError("VLMEvalKit reuse request differs from pinned contract")
        # Recheck immutable source bytes immediately before they are copied.
        if _sha256_file(contract.prediction_path) != contract.prediction_sha256:
            raise RuntimeError("pinned reuse source prediction changed before copy")
        status = json.loads(contract.source_status_path.read_text(encoding="utf-8"))
        infer_aux_files = [
            str(contract.source_run_dir / name)
            for name in infer_aux_file_names
            if (contract.source_run_dir / name).is_file()
        ]
        return {
            "run_dir": str(contract.source_run_dir),
            "status": status,
            "prediction_file": str(contract.prediction_path),
            "infer_aux_files": infer_aux_files,
        }

    smp_module.build_eval_id = lambda: destination_run_id
    smp_file_module.select_reuse_run = select_pinned_reuse_run
    return select_pinned_reuse_run


def _verify_pinned_reuse_postflight(
    contract: _PinnedReuseContract,
    *,
    destination_run_id: str,
    preexisting_run_ids: set[str],
) -> dict[str, Any]:
    """Accept only this invocation's exact destination and copied prediction."""

    newly_created = _eval_run_ids(contract.model_root) - preexisting_run_ids
    if newly_created != {destination_run_id}:
        raise RuntimeError(
            "pinned scoring did not create its unique destination: "
            f"expected {destination_run_id}, observed {sorted(newly_created)}"
        )
    destination = contract.model_root / destination_run_id
    prediction = destination / f"{contract.model_name}_{contract.dataset_name}.tsv"
    status_path = destination / "status.json"
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or prediction.is_symlink()
        or not prediction.is_file()
        or status_path.is_symlink()
        or not status_path.is_file()
        or _sha256_file(prediction) != contract.prediction_sha256
        or _sha256_file(contract.prediction_path) != contract.prediction_sha256
    ):
        raise RuntimeError("pinned scoring destination prediction identity differs")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    dataset_status = (
        status.get("datasets", {}).get(contract.dataset_name)
        if isinstance(status, dict)
        else None
    )
    if (
        not isinstance(dataset_status, dict)
        or status.get("eval_id") != destination_run_id
        or status.get("mode") != "eval"
        or status.get("reuse") is not True
        or status.get("reuse_aux") != "infer"
        or dataset_status.get("source_run") != contract.source_run_id
        or dataset_status.get("status") != "done"
        or _status_prediction_path(
            dataset_status.get("prediction_file"), run_dir=destination
        )
        != prediction.resolve()
    ):
        raise RuntimeError("pinned scoring destination status/source_run differs")
    return {
        "schema_version": "tgvf.vlmevalkit-pinned-reuse-receipt.v1",
        "dataset": contract.dataset_name,
        "model": contract.model_name,
        "source_evaluation_id": contract.source_evaluation_id,
        "source_run_id": contract.source_run_id,
        "source_manifest_path": str(contract.manifest_path.resolve()),
        "source_manifest_sha256": _sha256_file(contract.manifest_path),
        "source_prediction_path": str(contract.prediction_path.resolve()),
        "source_prediction_sha256": contract.prediction_sha256,
        "destination_run_id": destination_run_id,
        "destination_status_path": str(status_path.resolve()),
        "destination_status_sha256": _sha256_file(status_path),
        "destination_prediction_path": str(prediction.resolve()),
        "destination_prediction_sha256": _sha256_file(prediction),
    }


def _pop_option(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise RuntimeError(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def _required_option(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise RuntimeError(f"CoreDev runner requires {name}")
    return sys.argv[sys.argv.index(name) + 1]


def _option_values(name: str) -> tuple[str, ...]:
    if name not in sys.argv:
        return ()
    start = sys.argv.index(name) + 1
    values = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return tuple(values)


def main() -> int:
    pinned_source_run_id = _pop_option("--tgvf-reuse-source-run-id")
    pinned_manifest_path = _pop_option("--tgvf-reuse-manifest")
    if "--tgvf-reuse-source-run-id" in sys.argv or "--tgvf-reuse-manifest" in sys.argv:
        raise RuntimeError("pinned reuse options may be provided only once")
    if (pinned_source_run_id is None) != (pinned_manifest_path is None):
        raise RuntimeError("pinned reuse requires both source run ID and manifest")

    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_ARTIFACTS.read_text(encoding="utf-8"))
    judge_service = json.loads(JUDGE_SERVICE_CONFIG.read_text(encoding="utf-8"))
    checkout = Path(deployment["checkout"])
    overlay = Path(deployment["overlay"])
    artifact_root = Path(pinned["artifact_root"])
    artifact_identity = artifact_root / pinned["artifact_manifest"]
    sys.path[:0] = [str(checkout), str(overlay), str(REPOSITORY_ROOT / "src")]

    from tgvf_rl.evaluation.coredev_materialize import (  # noqa: PLC0415
        COREDEV_LLM_JUDGE_MODEL,
        COREDEV_LLM_JUDGE_REPOSITORY,
        register_coredev_vlmevalkit_slices,
        verify_coredev_2511_artifacts,
    )
    from tgvf_rl.evaluation.vlmevalkit import (  # noqa: PLC0415
        inject_vllm_engine_options_from_factory_kwargs,
        isolate_torchrun_environment_for_spawned_factory,
        materialize_coredev_subset_config,
    )
    from tgvf_rl.evaluation.coredev_results import (  # noqa: PLC0415
        check_qwen25_72b_judge,
        install_deterministic_mcq_answer_policy,
        install_fail_closed_judge_builders,
        summarize_coredev_results,
        write_json_atomic,
    )
    from tgvf_rl.evaluation.vlmevalkit_batch import (  # noqa: PLC0415
        attach_coredev_batch_options_from_factory_kwargs,
        install_coredev_batched_inference,
    )

    canonical_datasets = tuple(item["dataset"] for item in pinned["slices"])
    selected = _pop_option("--coredev-data")
    selected_datasets: tuple[str, ...]
    if selected is not None:
        config_path = Path(_required_option("--config")).resolve()
        if config_path != DIRECT_BASELINE_CONFIG:
            raise RuntimeError(
                "--coredev-data requires the pinned direct baseline config"
            )
        datasets = tuple(item.strip() for item in selected.split(",") if item.strip())
        selected_datasets = datasets
        work_dir = Path(_required_option("--work-dir")).resolve()
        resolved = materialize_coredev_subset_config(
            base_config_path=config_path,
            output_dir=work_dir / "resolved-configs",
            datasets=datasets,
        )
        sys.argv[sys.argv.index("--config") + 1] = str(resolved)
    elif _option_values("--data"):
        selected_datasets = _option_values("--data")
    else:
        selected_datasets = canonical_datasets

    if any(name not in canonical_datasets for name in selected_datasets):
        raise RuntimeError("CoreDev runner received an unknown dataset")
    if (
        tuple(name for name in canonical_datasets if name in selected_datasets)
        != selected_datasets
    ):
        raise RuntimeError("CoreDev datasets must follow canonical suite order")

    if pinned["llm_judge_model"] != COREDEV_LLM_JUDGE_MODEL:
        raise RuntimeError("CoreDev served judge identity mismatch")
    if pinned["llm_judge_repository"] != COREDEV_LLM_JUDGE_REPOSITORY:
        raise RuntimeError("CoreDev judge repository identity mismatch")
    artifacts = verify_coredev_2511_artifacts(artifact_identity)
    pinned_slices = {item["dataset"]: item for item in pinned["slices"]}
    actual_slices = {item["dataset"]: item for item in artifacts["slices"]}
    for dataset_name, expected in pinned_slices.items():
        actual = actual_slices.get(dataset_name)
        fields = (
            "dataset_class",
            "judge_contract",
            "sample_count",
            "tsv_md5",
            "tsv_sha256",
        )
        if actual is None or any(actual[field] != expected[field] for field in fields):
            raise RuntimeError(f"CoreDev pinned artifact mismatch: {dataset_name}")
    os.environ["LMUData"] = str(artifact_root)
    os.environ.setdefault("PRED_FORMAT", "tsv")
    os.environ.setdefault("EVAL_FORMAT", "json")

    import vlmeval.config as model_config_module  # noqa: PLC0415
    import vlmeval.dataset as dataset_module  # noqa: PLC0415
    import vlmeval.inference as inference_module  # noqa: PLC0415

    register_coredev_vlmevalkit_slices(dataset_module, artifacts)
    model_name = "Qwen3-VL-8B-Thinking"
    model_factory = attach_coredev_batch_options_from_factory_kwargs(
        inject_vllm_engine_options_from_factory_kwargs(
            model_config_module.supported_VLM[model_name]
        )
    )
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:
        model_factory = isolate_torchrun_environment_for_spawned_factory(model_factory)
    model_config_module.supported_VLM[model_name] = model_factory
    install_coredev_batched_inference(inference_module)

    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "all"
    evaluated_model = (
        _required_option("--model") if "--help" not in sys.argv else model_name
    )
    pinned_reuse: _PinnedReuseContract | None = None
    pinned_destination_run_id: str | None = None
    preexisting_run_ids: set[str] | None = None
    if pinned_source_run_id is not None:
        assert pinned_manifest_path is not None
        if mode != "eval" or "--reuse" not in sys.argv:
            raise RuntimeError("pinned reuse requires --mode eval --reuse")
        if _required_option("--reuse-aux") != "infer":
            raise RuntimeError("pinned reuse requires --reuse-aux infer")
        if len(selected_datasets) != 1:
            raise RuntimeError("pinned reuse requires exactly one dataset")
        pinned_reuse = _load_pinned_reuse_contract(
            work_dir=_required_option("--work-dir"),
            model_name=evaluated_model,
            dataset_name=selected_datasets[0],
            source_run_id=pinned_source_run_id,
            manifest_path=pinned_manifest_path,
        )
        preexisting_run_ids = _eval_run_ids(pinned_reuse.model_root)
        if pinned_reuse.source_run_id not in preexisting_run_ids:
            raise RuntimeError("pinned reuse source run is not discoverable")
        pinned_destination_run_id = _allocate_destination_run_id(pinned_reuse)
        smp_module = importlib.import_module("vlmeval.smp")
        smp_file_module = importlib.import_module("vlmeval.smp.file")
        _install_pinned_reuse_selector(
            pinned_reuse,
            destination_run_id=pinned_destination_run_id,
            smp_module=smp_module,
            smp_file_module=smp_file_module,
        )
    judge_base_url = None
    if "--help" not in sys.argv and mode in {"all", "eval"}:
        if "--judge" not in sys.argv:
            raise RuntimeError(
                f"CoreDev evaluation requires --judge {COREDEV_LLM_JUDGE_MODEL}"
            )
        judge = sys.argv[sys.argv.index("--judge") + 1]
        if judge != COREDEV_LLM_JUDGE_MODEL:
            raise RuntimeError(f"CoreDev LLM judge must be {COREDEV_LLM_JUDGE_MODEL}")
        if "--judge-base-url" not in sys.argv:
            raise RuntimeError("CoreDev Qwen2.5-72B judge requires --judge-base-url")
        judge_base_url = _required_option("--judge-base-url").rstrip("/")
        expected_base_url = judge_service["server"]["base_url"].rstrip("/")
        if judge_base_url != expected_base_url:
            raise RuntimeError(
                f"CoreDev judge service must use the pinned endpoint {expected_base_url}"
            )
        if judge_service["model"]["served_name"] != COREDEV_LLM_JUDGE_MODEL:
            raise RuntimeError("CoreDev judge service config model mismatch")
        if not judge_service["scope"]["allows_vlmevalkit_benchmark_judging"]:
            raise RuntimeError(
                "CoreDev judge service is not authorized for benchmark scoring"
            )

        work_dir = Path(_required_option("--work-dir")).resolve()
        preflight = check_qwen25_72b_judge(base_url=judge_base_url)
        write_json_atomic(work_dir / "judge-health-pre.json", preflight)
        image_mcq_module = importlib.import_module("vlmeval.dataset.image_mcq")
        image_vqa_module = importlib.import_module("vlmeval.dataset.image_vqa")
        multiple_choice_module = importlib.import_module(
            "vlmeval.dataset.utils.multiple_choice"
        )
        install_fail_closed_judge_builders((image_mcq_module, image_vqa_module))
        install_deterministic_mcq_answer_policy(multiple_choice_module)

    run_path = checkout / "run.py"
    sys.argv[0] = str(run_path)
    runpy.run_path(str(run_path), run_name="__main__")
    pinned_receipt: dict[str, Any] | None = None
    if pinned_reuse is not None:
        assert pinned_destination_run_id is not None
        assert preexisting_run_ids is not None
        pinned_receipt = _verify_pinned_reuse_postflight(
            pinned_reuse,
            destination_run_id=pinned_destination_run_id,
            preexisting_run_ids=preexisting_run_ids,
        )
    if "--help" not in sys.argv and mode in {"all", "eval"}:
        assert judge_base_url is not None
        postflight = check_qwen25_72b_judge(base_url=judge_base_url)
        write_json_atomic(work_dir / "judge-health-post.json", postflight)
        summarize_coredev_results(
            work_dir=work_dir,
            repository_root=REPOSITORY_ROOT,
            phase="eval",
            datasets=selected_datasets,
            expected_judge_base_url=judge_base_url,
            expected_model=evaluated_model,
            expected_eval_ids=(
                None
                if pinned_destination_run_id is None
                else {selected_datasets[0]: pinned_destination_run_id}
            ),
        )
        if pinned_receipt is not None:
            write_json_atomic(work_dir / "pinned-reuse-receipt.json", pinned_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
