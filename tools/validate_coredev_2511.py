"""Validate CoreDev artifacts against the pinned VLMEvalKit dataset classes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
PINNED_ARTIFACTS = REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"


def main() -> int:
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_ARTIFACTS.read_text(encoding="utf-8"))
    checkout = Path(deployment["checkout"])
    overlay = Path(deployment["overlay"])
    artifact_root = Path(pinned["artifact_root"])
    sys.path[:0] = [str(checkout), str(overlay), str(REPOSITORY_ROOT / "src")]
    os.environ["LMUData"] = str(artifact_root)

    from tgvf_rl.evaluation.coredev_materialize import (  # noqa: PLC0415
        COREDEV_DATASET_CLASSES,
        register_coredev_vlmevalkit_slices,
        verify_coredev_2511_artifacts,
    )
    import vlmeval.dataset as dataset_module  # noqa: PLC0415

    artifacts = verify_coredev_2511_artifacts(
        artifact_root / pinned["artifact_manifest"]
    )
    pinned_slices = {item["dataset"]: item for item in pinned["slices"]}
    actual_slices = {item["dataset"]: item for item in artifacts["slices"]}
    if set(pinned_slices) != set(actual_slices):
        raise RuntimeError("pinned and materialized CoreDev slice sets differ")
    fields = (
        "dataset_class",
        "judge_contract",
        "sample_count",
        "tsv_md5",
        "tsv_sha256",
    )
    for name, expected in pinned_slices.items():
        if any(actual_slices[name][field] != expected[field] for field in fields):
            raise RuntimeError(f"pinned CoreDev identity mismatch: {name}")

    runtime_classes = register_coredev_vlmevalkit_slices(dataset_module, artifacts)
    validated = []
    for artifact in artifacts["slices"]:
        dataset_name = artifact["dataset"]
        wrapper_class = getattr(dataset_module, runtime_classes[dataset_name])
        base_class = getattr(dataset_module, COREDEV_DATASET_CLASSES[dataset_name])
        wrapper_evaluate = getattr(wrapper_class.evaluate, "__func__", wrapper_class.evaluate)
        base_evaluate = getattr(base_class.evaluate, "__func__", base_class.evaluate)
        if wrapper_evaluate is not base_evaluate:
            raise RuntimeError(f"{dataset_name} scorer was overridden")
        dataset = wrapper_class(dataset=dataset_name)
        if len(dataset) != artifact["sample_count"]:
            raise RuntimeError(f"{dataset_name} loaded row count differs")
        for row_index in (0, len(dataset) - 1):
            prompt = dataset.build_prompt(row_index)
            image_paths = [part["value"] for part in prompt if part["type"] == "image"]
            if not image_paths or not all(Path(path).is_file() for path in image_paths):
                raise RuntimeError(f"{dataset_name} prompt image path is invalid")
        validated.append(
            {
                "dataset": dataset_name,
                "dataset_class": artifact["dataset_class"],
                "runtime_subclass": runtime_classes[dataset_name],
                "rows": len(dataset),
                "official_scorer_inherited": True,
            }
        )
    print(json.dumps({"identity": artifacts["identity"], "validated": validated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
