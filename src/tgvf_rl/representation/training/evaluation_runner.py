"""Single-GPU, evaluation-only runner for a completed representation artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import random
import subprocess
import tomllib
from typing import Any, Mapping

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter

from .config import (
    RepresentationPostTrainingInternalEvaluationConfig,
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from .data import load_retained_representation_jsonl
from .distributed_checkpoint import load_rank_zero_adapter_owned_state_export
from .native_pipeline import Qwen3NativeRepresentationGroupBuilder
from .post_training_evaluation import run_post_training_internal_evaluation
from .runtime import create_qwen3_representation_runtime


REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION = (
    "representation-internal-evaluation-run-v2"
)
REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_LEGACY_SCHEMA_VERSION = (
    "representation-internal-evaluation-run-v1"
)
_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_SHA256_CHARS = frozenset("0123456789abcdef")
_CODE_IDENTITY_PATHS = (
    "src/tgvf_rl",
    "pyproject.toml",
    "requirements/compatibility.lock",
    "requirements/compatibility-torch211-cu129.lock",
    "uv.lock",
)


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationRunConfig:
    run_id: str
    code_repository: str
    code_commit: str
    training_config_path: Path
    training_config_sha256: str
    artifact_path: Path
    artifact_file_sha256: str
    artifact_manifest_sha256: str
    expected_run_identity_sha256: str
    expected_global_step: int
    physical_gpu_id: int
    evaluation_data_path: Path | None
    evaluation_data_source_sha256: str | None
    evaluation: RepresentationPostTrainingInternalEvaluationConfig
    source_path: Path
    source_sha256: str
    schema_version: str = REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION


def load_representation_internal_evaluation_run_config(
    path: str | Path,
) -> RepresentationInternalEvaluationRunConfig:
    """Load one exact post-hoc evaluation identity without hidden defaults."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("internal-evaluation run config fields differ")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION,
        REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_LEGACY_SCHEMA_VERSION,
    }:
        raise ValueError("internal-evaluation run config schema mismatch")
    expected_fields = {
        "schema_version",
        "run_id",
        "code",
        "source",
        "artifact",
        "execution",
        "evaluation",
    }
    if schema_version == REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION:
        expected_fields.add("evaluation_data")
    if set(payload) != expected_fields:
        raise ValueError("internal-evaluation run config fields differ")
    _require_text(payload["run_id"], name="run_id")
    code = _table(payload, "code", {"repository", "commit", "dirty"})
    if code["repository"] != "Miocio-nora/TGVF-E2E-RL":
        raise ValueError("code.repository differs from the accepted repository")
    code_commit = _git_commit(code["commit"], name="code.commit")
    if code["dirty"] is not False:
        raise ValueError("formal internal evaluation requires dirty=false")
    source = _table(
        payload, "source", {"training_config_path", "training_config_sha256"}
    )
    artifact = _table(
        payload,
        "artifact",
        {
            "path",
            "file_sha256",
            "manifest_sha256",
            "expected_run_identity_sha256",
            "expected_global_step",
        },
    )
    execution = _table(payload, "execution", {"physical_gpu_id"})
    evaluation_data_path: Path | None = None
    evaluation_data_source_sha256: str | None = None
    if schema_version == REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION:
        evaluation_data = _table(
            payload,
            "evaluation_data",
            {"jsonl_path", "source_sha256"},
        )
        evaluation_data_path = _absolute_path(
            evaluation_data["jsonl_path"], name="evaluation_data.jsonl_path"
        )
        evaluation_data_source_sha256 = _sha256(
            evaluation_data["source_sha256"],
            name="evaluation_data.source_sha256",
        )
    evaluation = _table(
        payload,
        "evaluation",
        {
            "evaluation_id",
            "ordered_group_manifest_path",
            "ordered_group_manifest_sha256",
            "counterfactual_manifest_path",
            "counterfactual_manifest_sha256",
            "report_path",
            "random_seed",
            "max_new_tokens",
            "eos_token_ids",
        },
    )
    training_config_path = _absolute_path(
        source["training_config_path"], name="source.training_config_path"
    )
    artifact_path = _absolute_path(artifact["path"], name="artifact.path")
    expected_global_step = _positive_int(
        artifact["expected_global_step"], name="artifact.expected_global_step"
    )
    physical_gpu_id = _nonnegative_int(
        execution["physical_gpu_id"], name="execution.physical_gpu_id"
    )
    eval_config = RepresentationPostTrainingInternalEvaluationConfig(
        enabled=True,
        evaluation_id=evaluation["evaluation_id"],
        ordered_group_manifest_path=_absolute_path(
            evaluation["ordered_group_manifest_path"],
            name="evaluation.ordered_group_manifest_path",
        ),
        ordered_group_manifest_sha256=_sha256(
            evaluation["ordered_group_manifest_sha256"],
            name="evaluation.ordered_group_manifest_sha256",
        ),
        counterfactual_manifest_path=_absolute_path(
            evaluation["counterfactual_manifest_path"],
            name="evaluation.counterfactual_manifest_path",
        ),
        counterfactual_manifest_sha256=_sha256(
            evaluation["counterfactual_manifest_sha256"],
            name="evaluation.counterfactual_manifest_sha256",
        ),
        report_path=_absolute_path(
            evaluation["report_path"], name="evaluation.report_path"
        ),
        random_seed=_nonnegative_int(
            evaluation["random_seed"], name="evaluation.random_seed"
        ),
        max_new_tokens=_positive_int(
            evaluation["max_new_tokens"], name="evaluation.max_new_tokens"
        ),
        eos_token_ids=tuple(evaluation["eos_token_ids"]),
    )
    return RepresentationInternalEvaluationRunConfig(
        run_id=payload["run_id"],
        code_repository=code["repository"],
        code_commit=code_commit,
        training_config_path=training_config_path,
        training_config_sha256=_sha256(
            source["training_config_sha256"], name="source.training_config_sha256"
        ),
        artifact_path=artifact_path,
        artifact_file_sha256=_sha256(
            artifact["file_sha256"], name="artifact.file_sha256"
        ),
        artifact_manifest_sha256=_sha256(
            artifact["manifest_sha256"], name="artifact.manifest_sha256"
        ),
        expected_run_identity_sha256=_sha256(
            artifact["expected_run_identity_sha256"],
            name="artifact.expected_run_identity_sha256",
        ),
        expected_global_step=expected_global_step,
        physical_gpu_id=physical_gpu_id,
        evaluation_data_path=evaluation_data_path,
        evaluation_data_source_sha256=evaluation_data_source_sha256,
        evaluation=eval_config,
        source_path=source_path,
        source_sha256=sha256(raw).hexdigest(),
        schema_version=schema_version,
    )


def run_representation_internal_evaluation_from_artifact(
    config_path: str | Path,
) -> dict[str, object]:
    """Evaluate one immutable Adapter export without resuming its training run."""

    config = load_representation_internal_evaluation_run_config(config_path)
    _require_launch_environment(config)
    _verify_live_code_identity(config)
    training = load_representation_training_config(config.training_config_path)
    _require_file_sha256(
        config.training_config_path,
        config.training_config_sha256,
        name="training config",
    )
    _require_file_sha256(
        config.artifact_path,
        config.artifact_file_sha256,
        name="Adapter artifact",
    )
    for name, path, digest in (
        (
            "ordered-group manifest",
            config.evaluation.ordered_group_manifest_path,
            config.evaluation.ordered_group_manifest_sha256,
        ),
        (
            "counterfactual manifest",
            config.evaluation.counterfactual_manifest_path,
            config.evaluation.counterfactual_manifest_sha256,
        ),
    ):
        assert path is not None and digest is not None
        _require_file_sha256(path, digest, name=name)
    report_path = config.evaluation.report_path
    assert report_path is not None
    if report_path.exists():
        raise FileExistsError(
            f"internal-evaluation report already exists: {report_path}"
        )
    if not report_path.parent.is_dir():
        raise FileNotFoundError(
            f"internal-evaluation report parent does not exist: {report_path.parent}"
        )

    export = load_rank_zero_adapter_owned_state_export(config.artifact_path)
    manifest = export.manifest
    run_identity = manifest.run_identity
    if state_digest(manifest) != config.artifact_manifest_sha256:
        raise ValueError("Adapter artifact manifest SHA256 mismatch")
    if (
        manifest.run_identity_sha256 != config.expected_run_identity_sha256
        or run_identity.identity_sha256 != config.expected_run_identity_sha256
    ):
        raise ValueError("Adapter artifact run identity mismatch")
    if manifest.global_step != config.expected_global_step:
        raise ValueError("Adapter artifact global step mismatch")
    _validate_training_artifact_binding(training, run_identity)

    evaluation_data_path = (
        config.evaluation_data_path
        if config.evaluation_data_path is not None
        else training.data.validation.jsonl_path
    )
    evaluation_data_source_sha256 = (
        config.evaluation_data_source_sha256
        if config.evaluation_data_source_sha256 is not None
        else training.data.validation.source_sha256
    )
    evaluation_data = load_retained_representation_jsonl(
        evaluation_data_path,
        expected_source_sha256=evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    if config.evaluation_data_path is None:
        validation_identity = getattr(run_identity, "validation_identity", None)
        if validation_identity is None or (
            validation_identity.validation_retained_manifest_sha256
            != evaluation_data.manifest.manifest_sha256
        ):
            raise ValueError("validation retained-manifest identity mismatch")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _enable_determinism()
    _seed_current_process(config.evaluation.random_seed)
    processor, model = _load_qwen(training, device=device)
    tokenizer_length_before = len(processor.tokenizer)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=processor,
        model_identity=training.model_identity,
        conditioning_config=training.provider,
        adapter_dtype=_torch_dtype(training.model.dtype),
        adapter_variant=training.adapter_variant,
        fixture_mode=False,
    )
    if len(processor.tokenizer) != tokenizer_length_before:
        raise RuntimeError("internal evaluation changed tokenizer length")
    run_identity.adapter_contract.assert_matches(runtime.adapter)
    if export.state is None:
        raise RuntimeError("Adapter export has no tensor state")
    runtime.adapter.load_artifact_state_dict(export.state)
    runtime.adapter.requires_grad_(False)
    runtime.adapter.eval()
    model.requires_grad_(False)
    model.eval()
    family_adapter = Qwen3VLAdapter()
    group_builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family_adapter,
        prompt=training.prompt,
        image_loader=_load_rgb_image,
        image_max_pixels=training.model.image_max_pixels,
    )
    artifact = run_post_training_internal_evaluation(
        config=config.evaluation,
        runtime=runtime,
        qwen_model=model,
        family_adapter=family_adapter,
        validation_samples=evaluation_data.samples,
        validation_manifest_sha256=evaluation_data.manifest.manifest_sha256,
        group_builder=group_builder,
        model_identity=state_digest(asdict(run_identity.model)),
        checkpoint_identity=config.artifact_manifest_sha256,
        prompt_identity=f"{training.prompt.identity}:{training.prompt.sha256}",
    )
    return {
        "schema_version": "representation-internal-evaluation-run-result-v1",
        "status": "complete",
        "run_id": config.run_id,
        "source_config_sha256": config.source_sha256,
        "training_run_id": run_identity.run_id,
        "training_run_identity_sha256": run_identity.identity_sha256,
        "artifact_manifest_sha256": config.artifact_manifest_sha256,
        "evaluation_report_path": artifact.path,
        "evaluation_report_sha256": artifact.payload_sha256,
        "evaluation_report_byte_count": artifact.byte_count,
        "evaluation_data_manifest_sha256": (evaluation_data.manifest.manifest_sha256),
        "conditioning_provider": training.provider.provider.value,
        "physical_gpu_id": config.physical_gpu_id,
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": len(processor.tokenizer),
    }


def _validate_training_artifact_binding(
    training: RepresentationTrainingConfig, run_identity: Any
) -> None:
    comparisons = {
        "training run ID": (run_identity.run_id, training.run_id),
        "code identity": (run_identity.code, training.code_identity),
        "model identity": (run_identity.model, training.model_identity),
        "conditioning provider": (run_identity.provider, training.provider),
        "prompt SHA256": (run_identity.prompt_sha256, training.prompt.sha256),
        "objective": (run_identity.objective, training.objective.objective),
        "planned optimizer steps": (
            getattr(run_identity, "planned_target_optimizer_steps", None),
            training.training.target_optimizer_steps,
        ),
    }
    mismatches = [
        name
        for name, (observed, expected) in comparisons.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "training config differs from Adapter artifact: " + ", ".join(mismatches)
        )


def _load_qwen(
    training: RepresentationTrainingConfig, *, device: torch.device
) -> tuple[Any, torch.nn.Module]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Transformers Qwen runtime is unavailable") from error
    processor = AutoProcessor.from_pretrained(
        training.model.local_path,
        local_files_only=training.model.local_files_only,
        trust_remote_code=training.model.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        training.model.local_path,
        local_files_only=training.model.local_files_only,
        trust_remote_code=training.model.trust_remote_code,
        dtype=_torch_dtype(training.model.dtype),
        attn_implementation=training.model.attention_backend,
        low_cpu_mem_usage=True,
    ).to(device=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Qwen loader did not return an nn.Module")
    return processor, model


def _require_launch_environment(
    config: RepresentationInternalEvaluationRunConfig,
) -> None:
    required = {
        "CUDA_VISIBLE_DEVICES": str(config.physical_gpu_id),
        "CUBLAS_WORKSPACE_CONFIG": _REQUIRED_CUBLAS_WORKSPACE,
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    mismatches = {
        name: (expected, os.environ.get(name))
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"internal-evaluation launch environment mismatch: {mismatches}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("internal evaluation requires exactly one visible CUDA GPU")


def _verify_live_code_identity(
    config: RepresentationInternalEvaluationRunConfig,
) -> None:
    root = Path(__file__).resolve().parents[4]
    _run_git(root, "cat-file", "-e", f"{config.code_commit}^{{commit}}")
    changed = _run_git(
        root,
        "diff",
        "--name-only",
        config.code_commit,
        "HEAD",
        "--",
        *_CODE_IDENTITY_PATHS,
    ).strip()
    if changed:
        raise ValueError("evaluation code changed after configured commit: " + changed)
    local_patch = _run_git(
        root, "diff", "--name-only", "HEAD", "--", *_CODE_IDENTITY_PATHS
    ).strip()
    untracked = _run_git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *_CODE_IDENTITY_PATHS,
    ).strip()
    if local_patch or untracked:
        raise ValueError("formal internal evaluation requires clean live code paths")


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def _seed_current_process(seed: int | None) -> None:
    if seed is None:
        raise ValueError("internal evaluation requires an explicit seed")
    random.seed(seed)
    torch.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _enable_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_rgb_image(path: str) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for representation images") from error
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _torch_dtype(value: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported internal-evaluation dtype: {value}") from error


def _table(
    payload: Mapping[str, Any], name: str, fields: set[str]
) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"[{name}] fields differ")
    return value


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _require_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_commit(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase full Git commit")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_file_sha256(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {observed}")


__all__ = [
    "REPRESENTATION_INTERNAL_EVALUATION_RUN_CONFIG_SCHEMA_VERSION",
    "RepresentationInternalEvaluationRunConfig",
    "load_representation_internal_evaluation_run_config",
    "run_representation_internal_evaluation_from_artifact",
]
