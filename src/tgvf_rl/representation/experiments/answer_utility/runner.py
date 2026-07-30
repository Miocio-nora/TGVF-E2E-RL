"""Single-GPU executable for the isolated answer-utility smoke experiment.

This runner intentionally writes an experiment-private checkpoint/artifact
format.  An ineffective experiment can therefore be removed without teaching
the production Stage1 or RL loaders about its schema.  Checkpoints are exact at
optimizer boundaries and restore Adapter, optimizer, sampler, and RNG state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.checkpoint import (
    capture_representation_rng_state,
    restore_representation_rng_state,
)
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigV3,
)
from tgvf_rl.representation.training.runtime import (
    create_qwen3_representation_runtime,
)
from .config import load_answer_utility_experiment_config
from .controls import _normalized_answer_identity
from .native_pipeline import Qwen3AnswerUtilityGroupBuilder
from .run_config import AnswerUtilityRunConfig, load_answer_utility_run_config
from .sampling import AnswerSafeSameImageBatchSampler
from .trainer import AnswerUtilityTrainer


ANSWER_UTILITY_CHECKPOINT_SCHEMA_VERSION = "answer-utility-checkpoint-v1"
ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION = "answer-utility-adapter-artifact-v1"
ANSWER_UTILITY_METRICS_SCHEMA_VERSION = "answer-utility-metrics-v1"
_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_CHECKPOINT_NAME = re.compile(r"answer-utility-step-(?P<step>[0-9]{8})[.]pt\Z")
_IMPLEMENTATION_DEPENDENCY_GLOBS = (
    "tgvf_rl/checkpoint/coordinator.py",
    "tgvf_rl/conditioning/*.py",
    "tgvf_rl/contracts/*.py",
    "tgvf_rl/objectives/base.py",
    "tgvf_rl/observations/store.py",
    "tgvf_rl/protocol/*.py",
    "tgvf_rl/qwen/*.py",
    "tgvf_rl/representation/*.py",
    "tgvf_rl/representation/training/*.py",
)


@dataclass(frozen=True, slots=True)
class AnswerUtilityCheckpoint:
    run_identity_sha256: str
    global_step: int
    adapter_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    sampler_state: dict[str, object]
    rng_state: dict[str, object]
    adapter_state_sha256: str
    optimizer_state_sha256: str
    sampler_state_sha256: str
    rng_state_sha256: str
    schema_version: str = ANSWER_UTILITY_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha(self.run_identity_sha256, name="checkpoint run identity")
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
        ):
            raise ValueError("checkpoint global_step must be a non-negative integer")
        for value, name in (
            (self.adapter_state, "adapter_state"),
            (self.optimizer_state, "optimizer_state"),
            (self.sampler_state, "sampler_state"),
            (self.rng_state, "rng_state"),
        ):
            if not isinstance(value, dict):
                raise TypeError(f"checkpoint {name} must be a dict")
        if (
            _answer_utility_state_digest(self.adapter_state)
            != self.adapter_state_sha256
        ):
            raise ValueError("checkpoint Adapter state digest mismatch")
        if (
            _answer_utility_state_digest(self.optimizer_state)
            != self.optimizer_state_sha256
        ):
            raise ValueError("checkpoint optimizer state digest mismatch")
        if (
            _answer_utility_state_digest(self.sampler_state)
            != self.sampler_state_sha256
        ):
            raise ValueError("checkpoint sampler state digest mismatch")
        if _answer_utility_state_digest(self.rng_state) != self.rng_state_sha256:
            raise ValueError("checkpoint RNG state digest mismatch")
        if self.schema_version != ANSWER_UTILITY_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("answer-utility checkpoint schema mismatch")


@dataclass(frozen=True, slots=True)
class _ValidatedAnswerUtilityInputs:
    run: AnswerUtilityRunConfig
    experiment: Any
    training: Any
    export: Any
    train_data: Any
    run_identity_sha256: str
    planned_group_count: int
    skipped_planned_group_count: int
    resume_checkpoint: AnswerUtilityCheckpoint | None
    resume_metrics_high_water: int | None


def validate_answer_utility_experiment(
    run_config_path: str | Path,
    *,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate all byte-bound inputs without loading Qwen or launching GPU work."""

    inputs = _load_validated_inputs(
        run_config_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    run = inputs.run
    experiment = inputs.experiment
    training = inputs.training
    return {
        "schema_version": "answer-utility-validation-result-v1",
        "run_id": run.run_id,
        "variant": experiment.variant.value,
        "answer_supervision_view": (experiment.profile.answer_supervision_view.value),
        "train_adapter": experiment.profile.train_adapter,
        "zero_control": experiment.profile.requires_zero_control,
        "wrong_control": (
            "wrong_same_image_target"
            if experiment.profile.requires_wrong_control
            else None
        ),
        "run_identity_sha256": inputs.run_identity_sha256,
        "implementation_sha256": _implementation_sha256(),
        "environment_identity": _environment_identity(),
        "experiment_config_sha256": run.experiment_config_sha256,
        "base_training_config_sha256": (experiment.base_training_config_sha256),
        "source_artifact_sha256": run.source_artifact.file_sha256,
        "source_artifact_manifest_sha256": (run.source_artifact.manifest_sha256),
        "source_run_identity_sha256": (
            run.source_artifact.expected_run_identity_sha256
        ),
        "source_global_step": run.source_artifact.expected_global_step,
        "model_name": training.model.model_name,
        "target_optimizer_steps": run.target_optimizer_steps,
        "learning_rate": run.learning_rate,
        "rows_per_optimizer_step": (
            training.data.train.batch_size
            * training.training.gradient_accumulation_steps
        ),
        "planned_same_image_group_count": inputs.planned_group_count,
        "skipped_unsafe_same_image_group_count": (inputs.skipped_planned_group_count),
        "output_directory": str(run.output_directory),
        "resume_global_step": (
            None
            if inputs.resume_checkpoint is None
            else inputs.resume_checkpoint.global_step
        ),
        "gpu_work_launched": False,
    }


def run_answer_utility_experiment(
    run_config_path: str | Path,
    *,
    resume_checkpoint_path: str | Path | None = None,
    stop_after_global_step: int | None = None,
) -> dict[str, object]:
    """Run or exactly resume one isolated single-GPU trainable experiment."""

    inputs = _load_validated_inputs(
        run_config_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    run = inputs.run
    experiment = inputs.experiment
    training = inputs.training
    export = inputs.export
    train_data = inputs.train_data
    run_identity_sha256 = inputs.run_identity_sha256
    if not experiment.profile.train_adapter:
        raise ValueError("E0 is evaluation-only; use the existing RP66 evaluator")
    _require_launch_environment(run)
    invocation_target = _invocation_target(
        run,
        stop_after_global_step=stop_after_global_step,
    )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _enable_determinism()
    _seed_current_process(run.seed)
    processor, model = _load_qwen(training, device=device)
    model.requires_grad_(False)
    model.eval()
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
        raise RuntimeError("answer utility changed tokenizer length")
    export.manifest.run_identity.adapter_contract.assert_matches(runtime.adapter)
    if export.state is None:
        raise RuntimeError("source Adapter export has no tensor state")
    runtime.adapter.load_artifact_state_dict(export.state)
    runtime.adapter.requires_grad_(True)
    # Borrowed Qwen projections are registered under the Adapter; restore their
    # frozen ownership after enabling the Adapter-owned parameters.
    for name, parameter in runtime.adapter.named_parameters():
        if name.startswith(("main_projection.", "d_deepstack_projections.")):
            parameter.requires_grad_(False)
    family_adapter = Qwen3VLAdapter()
    base_builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family_adapter,
        prompt=training.prompt,
        image_loader=_load_rgb_image,
        image_max_pixels=training.model.image_max_pixels,
    )
    group_builder = Qwen3AnswerUtilityGroupBuilder(
        base_builder=base_builder,
        runtime=runtime,
        prompt=training.prompt,
        profile=experiment.profile,
    )
    sampler = AnswerSafeSameImageBatchSampler(
        train_data.samples,
        batch_size=training.data.train.batch_size,
        seed=training.data.train.sampler_seed,
        data_manifest_sha256=train_data.manifest.manifest_sha256,
        rank=0,
        world_size=1,
    )
    owned_parameters = tuple(
        parameter
        for name, parameter in runtime.adapter.named_parameters()
        if not name.startswith(("main_projection.", "d_deepstack_projections."))
        and parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        owned_parameters,
        lr=run.learning_rate,
        betas=training.optimizer.betas,
        eps=training.optimizer.eps,
        weight_decay=training.optimizer.weight_decay,
        **training.optimizer.torch_options,
    )
    legacy_objective = _legacy_objective(training, experiment)
    initial_global_step = 0
    if run.resume_checkpoint_path is not None:
        checkpoint = inputs.resume_checkpoint
        if checkpoint is None:
            raise RuntimeError("resume checkpoint disappeared after CPU preflight")
        runtime.adapter.load_artifact_state_dict(checkpoint.adapter_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        sampler.load_state_dict(checkpoint.sampler_state)
        restore_representation_rng_state(checkpoint.rng_state)
        initial_global_step = checkpoint.global_step
    if initial_global_step > invocation_target:
        raise ValueError("invocation target is behind the restored global step")
    resume_metrics_high_water = _prepare_output(
        run,
        run_identity_sha256=run_identity_sha256,
    )
    if resume_metrics_high_water != inputs.resume_metrics_high_water:
        raise RuntimeError("resume metrics changed after CPU preflight")
    if run.resume_checkpoint_path is not None:
        _append_metric(
            run.metrics_path,
            {
                "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
                "event": "resume",
                "run_identity_sha256": run_identity_sha256,
                "from_global_step": initial_global_step,
                "superseded_metrics_through_global_step": (
                    resume_metrics_high_water
                    if resume_metrics_high_water is not None
                    and resume_metrics_high_water > initial_global_step
                    else None
                ),
                "checkpoint_path": str(run.resume_checkpoint_path),
            },
        )
    trainer = AnswerUtilityTrainer(
        adapter=runtime.adapter,
        qwen_model=model,
        family_adapter=family_adapter,
        samples=train_data.samples,
        sampler=sampler,
        group_builder=group_builder,
        profile=experiment.profile,
        objective=experiment.objective,
        legacy_objective=legacy_objective,
        supervision_view=experiment.profile.answer_supervision_view,
        accumulation=training.training.accumulation_identity(world_size=1),
        optimizer=optimizer,
        scheduler=None,
        config=training.execution.trainer_config,
        accumulation_controller=None,
        initial_global_step=initial_global_step,
    )
    while trainer.global_step < invocation_target:
        metrics = trainer.train_step()
        if (
            metrics.global_step % run.log_every_optimizer_steps == 0
            or metrics.global_step == invocation_target
        ):
            _append_metric(
                run.metrics_path,
                {
                    "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
                    "event": "step",
                    "run_identity_sha256": run_identity_sha256,
                    **asdict(metrics),
                },
            )
        if metrics.global_step % run.checkpoint_every_optimizer_steps == 0:
            _save_checkpoint(
                run.checkpoint_directory
                / f"answer-utility-step-{metrics.global_step:08d}.pt",
                run_identity_sha256=run_identity_sha256,
                global_step=metrics.global_step,
                adapter=runtime.adapter,
                optimizer=optimizer,
                sampler=sampler,
            )
    final_checkpoint = _save_checkpoint(
        run.checkpoint_directory / f"answer-utility-step-{trainer.global_step:08d}.pt",
        run_identity_sha256=run_identity_sha256,
        global_step=trainer.global_step,
        adapter=runtime.adapter,
        optimizer=optimizer,
        sampler=sampler,
        allow_existing=True,
    )
    complete = trainer.global_step == run.target_optimizer_steps
    artifact_path: Path | None = None
    if complete:
        artifact_path = _save_final_artifact(
            run.final_artifact_path,
            run_identity_sha256=run_identity_sha256,
            global_step=trainer.global_step,
            adapter=runtime.adapter,
            source_artifact_sha256=run.source_artifact.file_sha256,
            experiment_config_sha256=run.experiment_config_sha256,
        )
    result: dict[str, object] = {
        "schema_version": "answer-utility-run-result-v1",
        "status": "complete" if complete else "stopped_at_optimizer_boundary",
        "run_id": run.run_id,
        "variant": experiment.variant.value,
        "run_identity_sha256": run_identity_sha256,
        "global_step": trainer.global_step,
        "planned_target_optimizer_steps": run.target_optimizer_steps,
        "checkpoint_path": str(final_checkpoint),
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "metrics_path": str(run.metrics_path),
    }
    _append_metric(
        run.metrics_path,
        {
            "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
            "event": "complete" if complete else "stop",
            "result": result,
        },
    )
    return result


def _load_validated_inputs(
    run_config_path: str | Path,
    *,
    resume_checkpoint_path: str | Path | None = None,
) -> _ValidatedAnswerUtilityInputs:
    run = load_answer_utility_run_config(run_config_path)
    if resume_checkpoint_path is not None:
        override = Path(resume_checkpoint_path).expanduser().resolve()
        if (
            run.resume_checkpoint_path is not None
            and run.resume_checkpoint_path != override
        ):
            raise ValueError("CLI and run-sidecar resume checkpoints differ")
        run = replace(run, resume_checkpoint_path=override)
    experiment = load_answer_utility_experiment_config(run.experiment_config_path)
    if run.run_id != experiment.run_id:
        raise ValueError("run and scientific sidecar run_id values differ")
    if run.experiment_config_sha256 != experiment.source_toml_sha256:
        raise ValueError("run sidecar points at another experiment-config revision")
    training = load_representation_training_config(experiment.base_training_config_path)
    if run.source_artifact.path != training.output.final_artifact_path:
        raise ValueError("source artifact must be the base config final artifact")
    _assert_output_isolated(run, training)
    export = load_rank_zero_adapter_owned_state_export(run.source_artifact.path)
    _validate_source_export(run, export)
    _validate_training_artifact_binding(training, export.manifest.run_identity)
    train_data = load_retained_representation_jsonl(
        training.data.train.jsonl_path,
        expected_source_sha256=training.data.train.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    if (
        export.manifest.run_identity.data_manifest_sha256
        != train_data.manifest.manifest_sha256
    ):
        raise ValueError("source Adapter was trained on another data manifest")
    planned_group_count, skipped_planned_group_count = _validate_planned_sampler(
        run,
        experiment,
        training,
        train_data,
    )
    run_identity_sha256 = _run_identity_sha256(
        run,
        experiment_canonical_sha256=experiment.canonical_config_sha256,
        base_training_config_sha256=experiment.base_training_config_sha256,
        train_manifest_sha256=train_data.manifest.manifest_sha256,
    )
    resume_checkpoint, resume_metrics_high_water = _validate_output_preflight(
        run,
        run_identity_sha256=run_identity_sha256,
    )
    return _ValidatedAnswerUtilityInputs(
        run=run,
        experiment=experiment,
        training=training,
        export=export,
        train_data=train_data,
        run_identity_sha256=run_identity_sha256,
        planned_group_count=planned_group_count,
        skipped_planned_group_count=skipped_planned_group_count,
        resume_checkpoint=resume_checkpoint,
        resume_metrics_high_water=resume_metrics_high_water,
    )


def _validate_output_preflight(
    run: AnswerUtilityRunConfig,
    *,
    run_identity_sha256: str,
) -> tuple[AnswerUtilityCheckpoint | None, int | None]:
    if run.resume_checkpoint_path is None:
        if run.output_directory.exists():
            raise FileExistsError(
                f"fresh experiment output already exists: {run.output_directory}"
            )
        return None, None
    metrics_high_water = _validate_resume_layout(
        run,
        run_identity_sha256=run_identity_sha256,
    )
    checkpoint = _load_checkpoint(
        run.resume_checkpoint_path,
        expected_run_identity_sha256=run_identity_sha256,
        map_location="cpu",
    )
    if _checkpoint_path_step(run.resume_checkpoint_path) != checkpoint.global_step:
        raise ValueError("resume checkpoint filename/global step mismatch")
    if checkpoint.global_step > run.target_optimizer_steps:
        raise ValueError("resume checkpoint is beyond the planned target")
    return checkpoint, metrics_high_water


def _validate_resume_layout(
    run: AnswerUtilityRunConfig,
    *,
    run_identity_sha256: str,
) -> int:
    checkpoint_path = run.resume_checkpoint_path
    if checkpoint_path is None:
        raise ValueError("resume layout validation requires a checkpoint")
    if not run.output_directory.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("resume output/checkpoint is missing")
    if checkpoint_path.parent != run.checkpoint_directory:
        raise ValueError("resume checkpoint must belong to the configured output")
    checkpoint_steps = {
        _checkpoint_path_step(path): path
        for path in run.checkpoint_directory.glob("answer-utility-step-*.pt")
    }
    selected_step = _checkpoint_path_step(checkpoint_path)
    if not checkpoint_steps or selected_step != max(checkpoint_steps):
        raise ValueError("resume must use the latest durable experiment checkpoint")
    return _audit_metrics_for_resume(
        run.metrics_path,
        expected_run_identity_sha256=run_identity_sha256,
    )


def _validate_planned_sampler(
    run: AnswerUtilityRunConfig,
    experiment: Any,
    training: Any,
    train_data: Any,
) -> tuple[int, int]:
    groups_per_step = getattr(
        training.training,
        "groups_per_rank_per_optimizer_step",
        1,
    )
    if groups_per_step <= 1:
        groups_per_step = training.training.gradient_accumulation_steps
    planned_group_count = run.target_optimizer_steps * groups_per_step
    sampler = AnswerSafeSameImageBatchSampler(
        train_data.samples,
        batch_size=training.data.train.batch_size,
        seed=training.data.train.sampler_seed,
        data_manifest_sha256=train_data.manifest.manifest_sha256,
        rank=0,
        world_size=1,
    )
    for group_index in range(planned_group_count):
        samples = tuple(train_data.samples[index] for index in sampler.next_batch())
        if len({sample.target for sample in samples}) != len(samples):
            raise ValueError(
                "planned E3/E4 group contains duplicate target text: "
                f"group={group_index + 1}"
            )
        if experiment.profile.requires_wrong_control:
            identities = tuple(
                _normalized_answer_identity(sample.short_answer) for sample in samples
            )
            if len(set(identities)) < 2:
                raise RuntimeError(
                    "answer-safe sampler emitted an unsafe wrong-D group: "
                    f"group={group_index + 1}"
                )
    return planned_group_count, sampler.skipped_batch_count


def _legacy_objective(
    training: Any, experiment: Any
) -> RepresentationObjectiveConfigV3:
    source = training.objective.objective
    if not isinstance(source, RepresentationObjectiveConfigV3):
        raise TypeError("answer utility requires a v3 balanced source objective")
    return RepresentationObjectiveConfigV3(
        identity=f"{experiment.run_id}:{experiment.variant.value}:legacy-auxiliary",
        kind=source.kind,
        matrix_ce_weight=experiment.objective.existing_matrix_weight,
        l_gen_weight=experiment.objective.existing_evidence_weight,
        norm_weight=experiment.objective.norm_weight,
        matrix_ce_mode=source.matrix_ce_mode,
        matrix_ce_temperature=source.matrix_ce_temperature,
    )


def _validate_source_export(run: AnswerUtilityRunConfig, export: Any) -> None:
    manifest = export.manifest
    source = run.source_artifact
    if state_digest(manifest) != source.manifest_sha256:
        raise ValueError("source Adapter manifest SHA256 mismatch")
    if (
        manifest.run_identity_sha256 != source.expected_run_identity_sha256
        or manifest.run_identity.identity_sha256 != source.expected_run_identity_sha256
    ):
        raise ValueError("source Adapter run identity mismatch")
    if manifest.global_step != source.expected_global_step:
        raise ValueError("source Adapter global step mismatch")


def _run_identity_sha256(
    run: AnswerUtilityRunConfig,
    *,
    experiment_canonical_sha256: str,
    base_training_config_sha256: str,
    train_manifest_sha256: str,
) -> str:
    payload = {
        **run.identity_payload(),
        # Resume is an invocation cursor, not scientific run identity.
        "resume_checkpoint_path": None,
        "source_toml_sha256": None,
        "canonical_config_sha256": None,
        "experiment_canonical_sha256": experiment_canonical_sha256,
        "base_training_config_sha256": base_training_config_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "implementation_sha256": _implementation_sha256(),
        "environment_identity": _environment_identity(),
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _invocation_target(
    run: AnswerUtilityRunConfig,
    *,
    stop_after_global_step: int | None,
) -> int:
    if stop_after_global_step is None:
        return run.target_optimizer_steps
    if (
        isinstance(stop_after_global_step, bool)
        or not isinstance(stop_after_global_step, int)
        or stop_after_global_step <= 0
    ):
        raise ValueError("stop_after_global_step must be a positive integer")
    if stop_after_global_step > run.target_optimizer_steps:
        raise ValueError("stop step cannot exceed the planned target")
    return stop_after_global_step


def _prepare_output(
    run: AnswerUtilityRunConfig,
    *,
    run_identity_sha256: str,
) -> int | None:
    if run.resume_checkpoint_path is None:
        if run.output_directory.exists():
            raise FileExistsError(
                f"fresh experiment output already exists: {run.output_directory}"
            )
        run.checkpoint_directory.mkdir(parents=True)
        _append_metric(
            run.metrics_path,
            {
                "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
                "event": "start",
                "run_id": run.run_id,
                "run_identity_sha256": run_identity_sha256,
                "run_config": run.identity_payload(),
            },
        )
        return None
    return _validate_resume_layout(
        run,
        run_identity_sha256=run_identity_sha256,
    )


def _save_checkpoint(
    path: Path,
    *,
    run_identity_sha256: str,
    global_step: int,
    adapter: Any,
    optimizer: torch.optim.Optimizer,
    sampler: AnswerSafeSameImageBatchSampler,
    allow_existing: bool = False,
) -> Path:
    if path.exists():
        if allow_existing:
            existing = _load_checkpoint(
                path,
                expected_run_identity_sha256=run_identity_sha256,
                map_location="cpu",
            )
            if existing.global_step != global_step:
                raise ValueError("existing checkpoint has another global step")
            return path
        raise FileExistsError(f"checkpoint already exists: {path}")
    adapter_state = _cpu_adapter_state(adapter)
    optimizer_state = optimizer.state_dict()
    sampler_state = sampler.state_dict()
    rng_state = capture_representation_rng_state()
    checkpoint = AnswerUtilityCheckpoint(
        run_identity_sha256=run_identity_sha256,
        global_step=global_step,
        adapter_state=adapter_state,
        optimizer_state=optimizer_state,
        sampler_state=sampler_state,
        rng_state=rng_state,
        adapter_state_sha256=_answer_utility_state_digest(adapter_state),
        optimizer_state_sha256=_answer_utility_state_digest(optimizer_state),
        sampler_state_sha256=_answer_utility_state_digest(sampler_state),
        rng_state_sha256=_answer_utility_state_digest(rng_state),
    )
    _atomic_torch_save(path, checkpoint)
    return path


def _load_checkpoint(
    path: Path,
    *,
    expected_run_identity_sha256: str,
    map_location: object,
) -> AnswerUtilityCheckpoint:
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, AnswerUtilityCheckpoint):
        raise TypeError("file is not an answer-utility checkpoint")
    value.__post_init__()
    if value.run_identity_sha256 != expected_run_identity_sha256:
        raise ValueError("checkpoint belongs to another answer-utility run")
    return value


def _save_final_artifact(
    path: Path,
    *,
    run_identity_sha256: str,
    global_step: int,
    adapter: Any,
    source_artifact_sha256: str,
    experiment_config_sha256: str,
) -> Path:
    if path.exists():
        _validate_existing_final_artifact(
            path,
            run_identity_sha256=run_identity_sha256,
            global_step=global_step,
            adapter=adapter,
            source_artifact_sha256=source_artifact_sha256,
            experiment_config_sha256=experiment_config_sha256,
        )
        return path
    state = _cpu_adapter_state(adapter)
    payload = {
        "schema_version": ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "global_step": global_step,
        "source_artifact_sha256": source_artifact_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "adapter_state_sha256": _answer_utility_state_digest(state),
        "adapter_state": state,
    }
    _atomic_torch_save(path, payload)
    return path


def _validate_existing_final_artifact(
    path: Path,
    *,
    run_identity_sha256: str,
    global_step: int,
    adapter: Any,
    source_artifact_sha256: str,
    experiment_config_sha256: str,
) -> None:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError("existing answer-utility artifact is not a mapping")
    expected_fields = {
        "schema_version",
        "run_identity_sha256",
        "global_step",
        "source_artifact_sha256",
        "experiment_config_sha256",
        "adapter_state_sha256",
        "adapter_state",
    }
    if set(value) != expected_fields:
        raise ValueError("existing answer-utility artifact fields differ")
    expected_values = {
        "schema_version": ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "global_step": global_step,
        "source_artifact_sha256": source_artifact_sha256,
        "experiment_config_sha256": experiment_config_sha256,
    }
    for name, expected in expected_values.items():
        if value.get(name) != expected:
            raise ValueError(f"existing answer-utility artifact {name} differs")
    state = value.get("adapter_state")
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise TypeError("existing answer-utility Adapter state is invalid")
    digest = _answer_utility_state_digest(state)
    if value.get("adapter_state_sha256") != digest:
        raise ValueError("existing answer-utility artifact state digest mismatch")
    if digest != _answer_utility_state_digest(_cpu_adapter_state(adapter)):
        raise ValueError("existing answer-utility artifact differs from checkpoint")


def _cpu_adapter_state(adapter: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in adapter.artifact_state_dict().items()
    }


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _append_metric(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _audit_metrics_for_resume(
    path: Path,
    *,
    expected_run_identity_sha256: str,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"metrics ledger is missing: {path}")
    rows: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics JSON at line {line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(f"metrics line {line_number} is not an object")
            if value.get("schema_version") != ANSWER_UTILITY_METRICS_SCHEMA_VERSION:
                raise ValueError(f"metrics schema mismatch at line {line_number}")
            rows.append(value)
    if not rows or rows[0].get("event") != "start":
        raise ValueError("answer-utility metrics ledger has no valid start record")
    if rows[0].get("run_identity_sha256") != expected_run_identity_sha256:
        raise ValueError("resume metrics belong to another experiment identity")
    active_step = 0
    for line_number, row in enumerate(rows[1:], 2):
        event = row.get("event")
        if event == "step":
            if row.get("run_identity_sha256") != expected_run_identity_sha256:
                raise ValueError(f"metrics identity mismatch at line {line_number}")
            step = row.get("global_step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step <= active_step
            ):
                raise ValueError(f"metrics step order mismatch at line {line_number}")
            active_step = step
        elif event == "resume":
            if row.get("run_identity_sha256") != expected_run_identity_sha256:
                raise ValueError(f"metrics identity mismatch at line {line_number}")
            step = row.get("from_global_step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"resume step is invalid at line {line_number}")
            active_step = step
        elif event in {"stop", "complete"}:
            result = row.get("result")
            if not isinstance(result, Mapping):
                raise ValueError(
                    f"terminal metrics result missing at line {line_number}"
                )
            if result.get("run_identity_sha256") != expected_run_identity_sha256:
                raise ValueError(
                    f"terminal metrics identity mismatch at line {line_number}"
                )
            step = result.get("global_step")
            if step != active_step:
                raise ValueError(
                    f"terminal metrics step mismatch at line {line_number}"
                )
            if event == "complete":
                raise FileExistsError("completed experiment metrics cannot be resumed")
        else:
            raise ValueError(f"unknown metrics event at line {line_number}: {event!r}")
    return active_step


def _checkpoint_path_step(path: Path) -> int:
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("answer-utility checkpoint filename is invalid")
    return int(match.group("step"))


def _implementation_sha256() -> str:
    package = Path(__file__).resolve().parent
    source_root = Path(__file__).resolve().parents[4]
    digest = sha256()
    dependencies = {
        path
        for pattern in _IMPLEMENTATION_DEPENDENCY_GLOBS
        for path in source_root.glob(pattern)
    }
    files = tuple(
        sorted(
            {*package.glob("*.py"), *dependencies},
            key=lambda path: str(path.relative_to(source_root)),
        )
    )
    if not files:
        raise RuntimeError("answer-utility implementation files are missing")
    for path in files:
        if not path.is_file():
            raise RuntimeError(f"answer-utility dependency is missing: {path}")
        encoded_name = str(path.relative_to(source_root)).encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _environment_identity() -> dict[str, object]:
    try:
        transformers_version = package_version("transformers")
    except PackageNotFoundError:
        transformers_version = "missing"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _answer_utility_state_digest(value: object) -> str:
    """Hash private checkpoint state, including scalar optimizer tensors.

    The project-wide digest predates scalar AdamW ``step`` tensors and cannot
    byte-view a zero-dimensional tensor.  This experiment-private schema uses
    a flattened byte view, so CPU/CUDA placement does not affect identity and
    exact resume remains verifiable from the first optimizer checkpoint.
    """

    digest = sha256()
    _update_answer_utility_digest(digest, value)
    return digest.hexdigest()


def _update_answer_utility_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        ordered = sorted(
            value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0]))
        )
        for key, item in ordered:
            _update_answer_utility_digest(digest, key)
            _update_answer_utility_digest(digest, item)
        return
    if isinstance(value, tuple):
        digest.update(b"tuple:")
        for item in value:
            _update_answer_utility_digest(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"list:")
        for item in value:
            _update_answer_utility_digest(digest, item)
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool:true" if value else b"bool:false")
        return
    if isinstance(value, int):
        digest.update(f"int:{value}".encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint float state must be finite")
        digest.update(f"float:{value.hex()}".encode("ascii"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return
    raise TypeError(
        f"unsupported answer-utility checkpoint state type: {type(value).__qualname__}"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_output_isolated(run: AnswerUtilityRunConfig, training: Any) -> None:
    output = run.output_directory.resolve()
    protected = (
        run.source_artifact.path.parent.resolve(),
        training.checkpoint.directory.resolve(),
        training.output.metrics_jsonl_path.parent.resolve(),
    )
    for directory in protected:
        if (
            output == directory
            or directory in output.parents
            or output in directory.parents
        ):
            raise ValueError(
                "experiment output must not overlap a production artifact directory"
            )


def _require_launch_environment(run: AnswerUtilityRunConfig) -> None:
    required = {
        "CUDA_VISIBLE_DEVICES": str(run.physical_gpu_id),
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
        raise ValueError(f"answer-utility launch environment mismatch: {mismatches}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("answer utility requires exactly one visible CUDA GPU")


def _sha(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION",
    "ANSWER_UTILITY_CHECKPOINT_SCHEMA_VERSION",
    "ANSWER_UTILITY_METRICS_SCHEMA_VERSION",
    "AnswerUtilityCheckpoint",
    "run_answer_utility_experiment",
    "validate_answer_utility_experiment",
]
