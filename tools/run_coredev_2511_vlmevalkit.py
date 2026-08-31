"""Run pinned VLMEvalKit with official aliases bound to immutable CoreDev slices."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import runpy
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
PINNED_ARTIFACTS = (
    REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
)
DIRECT_BASELINE_CONFIG = (
    REPOSITORY_ROOT / "configs/evaluation/coredev_2511_qwen3_direct_v1.json"
)
INSTRUCT_DIRECT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_v1.json"
)
JUDGE_SERVICE_CONFIG = (
    REPOSITORY_ROOT / "configs/evaluation/qwen25_72b_judge_service_v1.json"
)
BASELINE_MODEL = "Qwen3-VL-8B-Thinking"
POLICY_SCORING_MODEL = "Qwen3-VL-8B-Instruct"


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
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_ARTIFACTS.read_text(encoding="utf-8"))
    judge_service = json.loads(JUDGE_SERVICE_CONFIG.read_text(encoding="utf-8"))
    checkout = Path(deployment["checkout"])
    overlay = Path(deployment["overlay"])
    artifact_root = Path(pinned["artifact_root"])
    artifact_identity = artifact_root / pinned["artifact_manifest"]
    # The overlay supplies dependencies missing from the project environment,
    # but must remain a fallback.  In particular it contains antlr4 4.11,
    # whereas the project's OmegaConf and compatibility lock require 4.9.3.
    # Keeping the normal environment ahead of the overlay also lets
    # latex2sympy2 select its bundled 4.9.3 grammar consistently.
    injected_paths = (str(checkout), str(overlay), str(REPOSITORY_ROOT / "src"))
    sys.path[:] = [entry for entry in sys.path if entry not in injected_paths]
    sys.path[:0] = [str(checkout), str(REPOSITORY_ROOT / "src")]
    sys.path.append(str(overlay))

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

    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "all"
    requested_models = _option_values("--model")
    if len(requested_models) > 1:
        raise RuntimeError("CoreDev runner requires exactly one model")
    requested_config = Path(_required_option("--config")).resolve()
    configured_models = tuple(
        json.loads(requested_config.read_text(encoding="utf-8")).get("model", {})
    )
    expected_model = (
        requested_models[0]
        if requested_models
        else configured_models[0]
        if len(configured_models) == 1
        else BASELINE_MODEL
    )
    if expected_model not in {BASELINE_MODEL, POLICY_SCORING_MODEL}:
        raise RuntimeError(f"unsupported CoreDev evaluated model: {expected_model}")
    is_instruct_direct = (
        expected_model == POLICY_SCORING_MODEL
        and requested_config == INSTRUCT_DIRECT_CONFIG
    )
    if (
        expected_model == POLICY_SCORING_MODEL
        and mode != "eval"
        and not is_instruct_direct
    ):
        raise RuntimeError(
            f"{POLICY_SCORING_MODEL} inference requires the pinned Instruct direct config"
        )

    canonical_datasets = tuple(item["dataset"] for item in pinned["slices"])
    selected = _pop_option("--coredev-data")
    selected_datasets: tuple[str, ...]
    if selected is not None:
        config_path = Path(_required_option("--config")).resolve()
        expected_config = (
            INSTRUCT_DIRECT_CONFIG
            if expected_model == POLICY_SCORING_MODEL
            else DIRECT_BASELINE_CONFIG
        )
        if config_path != expected_config:
            raise RuntimeError(
                "--coredev-data requires the matching pinned direct config"
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
    model_name = expected_model
    model_factory = attach_coredev_batch_options_from_factory_kwargs(
        inject_vllm_engine_options_from_factory_kwargs(
            model_config_module.supported_VLM[model_name]
        )
    )
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:
        model_factory = isolate_torchrun_environment_for_spawned_factory(model_factory)
    model_config_module.supported_VLM[model_name] = model_factory
    install_coredev_batched_inference(inference_module)

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
    # Eval-only runs reuse an already materialized prediction directory.  The
    # external CLI's config mode constructs the model even under ``--mode
    # eval``; switch to its model/data selection form so scoring never reserves
    # policy GPUs or instantiates Qwen3.
    if mode == "eval" and requested_models:
        config_index = sys.argv.index("--config")
        del sys.argv[config_index : config_index + 2]
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
            expected_model=expected_model,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
