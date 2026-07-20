"""Run pinned VLMEvalKit with official aliases bound to immutable CoreDev slices."""

from __future__ import annotations

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


def main() -> int:
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_ARTIFACTS.read_text(encoding="utf-8"))
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

    import vlmeval.dataset as dataset_module  # noqa: PLC0415

    register_coredev_vlmevalkit_slices(dataset_module, artifacts)

    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "all"
    if "--help" not in sys.argv and mode in {"all", "eval"}:
        if "--judge" not in sys.argv:
            raise RuntimeError(f"CoreDev evaluation requires --judge {COREDEV_LLM_JUDGE_MODEL}")
        judge = sys.argv[sys.argv.index("--judge") + 1]
        if judge != COREDEV_LLM_JUDGE_MODEL:
            raise RuntimeError(f"CoreDev LLM judge must be {COREDEV_LLM_JUDGE_MODEL}")
        if "--judge-base-url" not in sys.argv:
            raise RuntimeError("CoreDev Qwen2.5-72B judge requires --judge-base-url")

    run_path = checkout / "run.py"
    sys.argv[0] = str(run_path)
    runpy.run_path(str(run_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
