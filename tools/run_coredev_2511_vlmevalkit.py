#!/usr/bin/python3 -I
"""Run pinned VLMEvalKit with official aliases bound to immutable CoreDev slices."""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/run_coredev_2511_vlmevalkit.py",
        ),
    )

import importlib
import json
import os
from pathlib import Path
import runpy
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)

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
    assert_legacy_standalone_execution_quarantined(
        "tools/run_coredev_2511_vlmevalkit.py"
    )
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
        install_fail_closed_judge_builders((image_mcq_module, image_vqa_module))

    run_path = checkout / "run.py"
    sys.argv[0] = str(run_path)
    runpy.run_path(str(run_path), run_name="__main__")
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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
