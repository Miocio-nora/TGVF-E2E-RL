"""Read-only external-file binding for representation-training configs."""

from __future__ import annotations

from hashlib import sha256

from .config_run_schema import RepresentationTrainingConfig
from .config_values import (
    _nearest_existing_parent,
    _optional_existing_file_probe,
    _read_existing_file_bytes,
    _require_existing_file_probe,
)


def _verify_external_files(
    config: RepresentationTrainingConfig,
    *,
    allow_existing_post_training_report: bool = False,
) -> None:
    if not config.model.local_path.is_dir():
        raise ValueError(
            f"accepted Qwen3 model directory is unavailable: {config.model.local_path}"
        )
    required_model_files = (
        "config.json",
        "tokenizer.json",
        "chat_template.json",
        "model.safetensors.index.json",
    )
    missing_names: list[str] = []
    for name in required_model_files:
        try:
            _require_existing_file_probe(
                config.model.local_path / name,
                field_name=f"model.local_path/{name}",
            )
        except (TypeError, ValueError):
            missing_names.append(name)
    missing = tuple(missing_names)
    if missing:
        raise ValueError(f"accepted Qwen3 directory is incomplete: {missing}")
    for name, split in (
        ("train", config.data.train),
        ("validation", config.data.validation),
    ):
        _, payload = _read_existing_file_bytes(
            split.jsonl_path, field_name=f"data.{name}.jsonl_path"
        )
        actual = sha256(payload).hexdigest()
        if actual != split.source_sha256:
            raise ValueError(
                f"data.{name}.source_sha256 mismatch: expected "
                f"{split.source_sha256}, got {actual}"
            )
    for name, path in (
        ("output.final_artifact_path", config.output.final_artifact_path),
        ("output.metrics_jsonl_path", config.output.metrics_jsonl_path),
        ("checkpoint.directory", config.checkpoint.directory),
    ):
        parent = path if name == "checkpoint.directory" else path.parent
        existing_parent = _nearest_existing_parent(parent)
        if not existing_parent.is_dir():
            raise ValueError(f"{name} has no usable directory ancestor")
    if config.resume.enabled:
        assert config.resume.checkpoint_path is not None
        if not config.resume.checkpoint_path.is_dir():
            raise ValueError(
                "resume.checkpoint_path must be an existing distributed "
                f"checkpoint directory: {config.resume.checkpoint_path}"
            )
    evaluation = config.post_training_internal_evaluation
    if evaluation is not None and evaluation.enabled:
        for name, path, expected_sha256 in (
            (
                "ordered_group_manifest_path",
                evaluation.ordered_group_manifest_path,
                evaluation.ordered_group_manifest_sha256,
            ),
            (
                "counterfactual_manifest_path",
                evaluation.counterfactual_manifest_path,
                evaluation.counterfactual_manifest_sha256,
            ),
            *(
                (
                    (
                        "grounding_manifest_path",
                        evaluation.grounding_manifest_path,
                        evaluation.grounding_manifest_sha256,
                    ),
                )
                if evaluation.grounding_manifest_path is not None
                else ()
            ),
        ):
            assert path is not None and expected_sha256 is not None
            _, payload = _read_existing_file_bytes(
                path,
                field_name=f"post_training_internal_evaluation.{name}",
            )
            actual_sha256 = sha256(payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"post_training_internal_evaluation.{name} SHA256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        assert evaluation.report_path is not None
        report_parent = _nearest_existing_parent(evaluation.report_path.parent)
        if not report_parent.is_dir():
            raise ValueError(
                "post_training_internal_evaluation.report_path has no usable parent"
            )
        existing_report = _optional_existing_file_probe(
            evaluation.report_path,
            field_name="post_training_internal_evaluation.report_path",
        )
        if existing_report is not None and not allow_existing_post_training_report:
            raise ValueError(
                "post_training_internal_evaluation.report_path already exists"
            )
