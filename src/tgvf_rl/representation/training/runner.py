"""Executable two-rank Qwen3 representation-phase training entry point.

This runner is intentionally narrower than the reusable training primitives:
it accepts only the strict Qwen3/FSDP2 TOML identity, requires a ``torchrun``
world of two processes mapped from physical GPUs 2 and 3, and performs only
optimizer-boundary checkpointing.  It does not choose a prompt, dataset,
objective weight, provider, or optimizer value on the caller's behalf.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter

from .checkpoint import (
    RepresentationAdapterContractIdentity,
    RepresentationInitializationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3,
    RepresentationSamplerContractIdentity,
    RepresentationSchedulerIdentity,
    RepresentationTrainerExecutionIdentity,
)
from .config import (
    RepresentationDataConfigV2,
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from .data import (
    RepresentationDataset,
    load_retained_representation_jsonl,
    train_validation_group_overlap,
)
from .distributed_checkpoint import (
    gather_rank_zero_full_adapter_owned_state,
    load_rank_zero_adapter_owned_state_export,
    restore_distributed_representation_checkpoint,
    save_distributed_representation_checkpoint_atomic,
    save_rank_zero_adapter_owned_state_export_atomic,
)
from .fsdp2 import apply_representation_fsdp2
from .history import (
    RepresentationMetricsHistoryIdentity,
    load_representation_metrics_history,
)
from .native_pipeline import Qwen3NativeRepresentationGroupBuilder
from .performance import (
    RepresentationTrainStepPerformance,
    measure_distributed_train_step,
)
from .runtime import create_qwen3_representation_runtime
from .sampling import SameImageBatchSampler
from .trainer import (
    RepresentationStepMetrics,
    RepresentationTrainer,
    build_representation_scheduler,
)
from .validation_identity import (
    REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION,
    RepresentationValidationDataAudit,
    build_representation_validation_data_audit,
)


REPRESENTATION_RUNNER_SCHEMA_VERSION = "representation-runner-v1"
REPRESENTATION_QWEN_PHYSICAL_EXECUTION_SCHEMA_VERSION = (
    "representation-qwen-physical-execution-v1"
)
_REQUIRED_VISIBLE_DEVICES = "2,3"
_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_CODE_IDENTITY_PATHS = (
    "src/tgvf_rl",
    "pyproject.toml",
    "uv.lock",
)


def run_representation_training(
    config_path: str | Path,
    *,
    stop_after_global_step: int | None = None,
) -> dict[str, object] | None:
    """Run one strict representation configuration under ``torchrun``.

    Every rank calls this function.  Rank zero returns the JSON-safe closeout
    payload; other ranks return ``None`` so the CLI emits only one result.
    """

    config = load_representation_training_config(config_path)
    _validate_invocation_stop(config, stop_after_global_step)
    _require_launch_environment(config)
    _verify_live_code_identity(config)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    rank = torch.distributed.get_rank()
    try:
        return _run_initialized(
            config,
            rank=rank,
            stop_after_global_step=stop_after_global_step,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_initialized(
    config: RepresentationTrainingConfig,
    *,
    rank: int,
    stop_after_global_step: int | None = None,
) -> dict[str, object] | None:
    world_size = torch.distributed.get_world_size()
    local_rank = _environment_integer("LOCAL_RANK")
    if world_size != config.fsdp2.world_size or rank != _environment_integer("RANK"):
        raise ValueError("torchrun rank/world identity differs from the TOML contract")
    if local_rank != config.fsdp2.logical_gpu_ids[rank]:
        raise ValueError("torchrun LOCAL_RANK differs from the configured GPU mapping")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _enable_determinism()

    train_data, validation_data = _load_datasets(config)
    overlap = train_validation_group_overlap(
        train_data.samples,
        validation_data.samples,
    )
    validation_data_audit: RepresentationValidationDataAudit | None = None
    if isinstance(config.data, RepresentationDataConfigV2):
        validation_data_audit = build_representation_validation_data_audit(
            train_dataset=train_data,
            validation_dataset=validation_data,
            validation_batch_k=config.data.validation.batch_size,
            validation_sampler_seed=config.data.validation.sampler_seed,
            validation_every_optimizer_steps=(
                config.training.validation_every_optimizer_steps
            ),
            evaluator_schema_version=(
                REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION
            ),
            overlap_policy=config.data.split_overlap_policy,
            expected_overlap_report_sha256=(config.data.expected_overlap_report_sha256),
        )
        if validation_data_audit.overlap_report != overlap:
            raise RuntimeError("validation-data audit changed the split-overlap report")
    else:
        overlap.require_disjoint()
    train_sampler = SameImageBatchSampler(
        train_data.samples,
        batch_size=config.data.train.batch_size,
        seed=config.data.train.sampler_seed,
        data_manifest_sha256=train_data.manifest.manifest_sha256,
        rank=rank,
        world_size=world_size,
    )
    _prepare_output_paths(config, rank=rank)

    processor, model = _load_qwen3(config, device=device, rank=rank)
    tokenizer_length_before = len(processor.tokenizer)
    _seed_current_process(config.initialization.seed)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=processor,
        model_identity=config.model_identity,
        conditioning_config=config.provider,
        adapter_dtype=_torch_dtype(config.fsdp2.parameter_dtype),
        fixture_mode=False,
    )
    if len(processor.tokenizer) != tokenizer_length_before:
        raise RuntimeError("representation runtime changed the tokenizer length")
    initialization = RepresentationInitializationIdentity.from_adapter(
        runtime.adapter,
        kind=config.initialization.kind,
        seed=config.initialization.seed,
        source_artifact_sha256=config.initialization.source_artifact_sha256,
    )
    _require_same_string_across_ranks(
        initialization.initial_adapter_state_sha256,
        name="initial Adapter state SHA256",
    )

    mesh, mixed_precision, offload = _build_fsdp2_policies(config)
    binding = apply_representation_fsdp2(
        adapter=runtime.adapter,
        qwen_model=model,
        mesh=mesh,
        config=config.fsdp2.runtime_config,
        mixed_precision_policy=mixed_precision,
        offload_policy=offload,
    )
    optimizer = torch.optim.AdamW(
        binding.optimizer_parameters(),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        **config.optimizer.torch_options,
    )
    binding.assert_optimizer_ownership(optimizer)
    scheduler = build_representation_scheduler(optimizer, config.scheduler)
    accumulation = config.accumulation_identity
    trainer_execution = RepresentationTrainerExecutionIdentity.from_config(
        config.execution.trainer_config
    )
    run_identity_fields = dict(
        run_id=config.run_id,
        code=config.code_identity,
        model=config.model_identity,
        provider=config.provider,
        data_manifest_sha256=train_data.manifest.manifest_sha256,
        prompt_sha256=config.prompt.sha256,
        objective=config.objective.objective,
        adapter_contract=RepresentationAdapterContractIdentity.from_adapter(
            runtime.adapter
        ),
        accumulation=accumulation,
        optimizer=RepresentationOptimizerIdentity.from_optimizer(optimizer),
        scheduler=RepresentationSchedulerIdentity.from_config(config.scheduler),
        trainer_execution=trainer_execution,
        initialization=initialization,
        sampler_contract=RepresentationSamplerContractIdentity.from_sampler(
            train_sampler
        ),
    )
    run_identity: RepresentationRunIdentity
    if isinstance(config.data, RepresentationDataConfigV2):
        if validation_data_audit is None:
            raise RuntimeError("config v2 did not materialize validation-data identity")
        run_identity = RepresentationRunIdentityV3(
            **run_identity_fields,
            validation_identity=validation_data_audit.identity,
            planned_target_optimizer_steps=(config.training.target_optimizer_steps),
        )
    else:
        run_identity = RepresentationRunIdentity(**run_identity_fields)
    _require_same_string_across_ranks(
        run_identity.identity_sha256,
        name="representation run identity SHA256",
    )

    family_adapter = Qwen3VLAdapter()
    group_builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family_adapter,
        prompt=config.prompt,
        image_loader=_load_rgb_image,
        image_max_pixels=config.model.image_max_pixels,
    )
    initial_global_step = 0
    next_validation_event_index = 0
    latest_checkpoint: Path | None = None
    latest_checkpoint_global_step: int | None = None
    if config.resume.enabled:
        assert config.resume.checkpoint_path is not None
        expected_metrics_history: RepresentationMetricsHistoryIdentity | None = None
        checkpoint_path_step = _checkpoint_step(
            config.resume.checkpoint_path,
            config.checkpoint.filename_prefix,
        )
        if isinstance(run_identity, RepresentationRunIdentityV3):
            expected_metrics_history = load_representation_metrics_history(
                config.output.metrics_jsonl_path,
                run_id=config.run_id,
                run_identity_sha256=run_identity.identity_sha256,
                checkpoint_global_step=checkpoint_path_step,
                runner_schema_version=REPRESENTATION_RUNNER_SCHEMA_VERSION,
            ).identity
            _require_same_string_across_ranks(
                expected_metrics_history.identity_sha256,
                name="representation metrics-history SHA256",
            )
        resume = restore_distributed_representation_checkpoint(
            config.resume.checkpoint_path,
            binding=binding,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_sampler,
            expected_run_identity=run_identity,
            accumulation=accumulation,
            trainer_execution=trainer_execution,
            expected_metrics_history=expected_metrics_history,
        )
        initial_global_step = resume.global_step
        if initial_global_step != checkpoint_path_step:
            raise RuntimeError("checkpoint path step differs from restored global step")
        if isinstance(run_identity, RepresentationRunIdentityV3):
            if resume.next_validation_event_index is None:
                raise RuntimeError("DCP v2 did not restore the validation cursor")
            next_validation_event_index = resume.next_validation_event_index
        latest_checkpoint = config.resume.checkpoint_path
        latest_checkpoint_global_step = initial_global_step
        if not isinstance(run_identity, RepresentationRunIdentityV3):
            _validate_resume_metrics_history_collective(
                config.output.metrics_jsonl_path,
                run_id=config.run_id,
                run_identity_sha256=run_identity.identity_sha256,
                checkpoint_global_step=initial_global_step,
                rank=rank,
            )
    if initial_global_step > config.training.target_optimizer_steps:
        raise ValueError("resume checkpoint is beyond the configured target step")
    invocation_target_step = _invocation_target_step(
        config,
        initial_global_step=initial_global_step,
        stop_after_global_step=stop_after_global_step,
    )
    trainer = RepresentationTrainer(
        adapter=runtime.adapter,
        qwen_model=model,
        family_adapter=family_adapter,
        samples=train_data.samples,
        sampler=train_sampler,
        group_builder=group_builder,
        objective=config.objective.objective,
        accumulation=accumulation,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.execution.trainer_config,
        initial_global_step=initial_global_step,
    )

    if not config.resume.enabled:
        _append_metric_rank_zero_collective(
            config.output.metrics_jsonl_path,
            {
                "event": "start",
                "schema_version": REPRESENTATION_RUNNER_SCHEMA_VERSION,
                "run_id": config.run_id,
                "run_identity_sha256": run_identity.identity_sha256,
                "canonical_config_sha256": config.canonical_config_sha256,
                "source_toml_sha256": config.source_toml_sha256,
                "initial_global_step": initial_global_step,
                "invocation_target_step": invocation_target_step,
                "planned_target_optimizer_steps": (
                    config.training.target_optimizer_steps
                ),
                "groups_per_rank_per_optimizer_step": (
                    config.training.groups_per_rank_per_optimizer_step
                ),
                "train_manifest_sha256": train_data.manifest.manifest_sha256,
                "validation_manifest_sha256": validation_data.manifest.manifest_sha256,
                "conditioning_provider": config.provider.provider.value,
                "image_max_pixels": config.model.image_max_pixels,
                "validation_data_identity_sha256": (
                    None
                    if validation_data_audit is None
                    else validation_data_audit.identity.identity_sha256
                ),
                "split_overlap_report": overlap.canonical_payload(),
                "split_overlap_report_sha256": overlap.identity_sha256,
                "train_image_raw_byte_manifest_sha256": (
                    None
                    if validation_data_audit is None
                    else validation_data_audit.train_image_manifest.manifest_sha256
                ),
                "validation_image_raw_byte_manifest_sha256": (
                    None
                    if validation_data_audit is None
                    else validation_data_audit.validation_image_manifest.manifest_sha256
                ),
            },
        )
    # A resume attempt is operational rather than durable scientific state.
    # Appending it here would advance the exact history past the checkpoint
    # before the next checkpoint commits, making a retry after process failure
    # impossible without manually truncating the JSONL file.  The invocation
    # config and outcome remain in the command's stdout log.

    created_checkpoint_paths: list[Path] = []
    validation_event_index = next_validation_event_index
    while trainer.global_step < invocation_target_step:
        metrics, performance = measure_distributed_train_step(
            trainer.train_step,
            device=device,
            global_matrix_count=(
                config.training.gradient_accumulation_steps
                * config.training.groups_per_rank_per_optimizer_step
                * world_size
            ),
        )
        all_sample_ids = _gather_string_tuples(metrics.local_sample_ids)
        all_qwen_forward_batch_sizes = _gather_positive_int_tuples(
            metrics.local_qwen_forward_batch_sizes
        )
        if metrics.global_step % config.training.log_every_optimizer_steps == 0:
            _log_training_metric(
                config,
                metrics=metrics,
                all_sample_ids=all_sample_ids,
                all_qwen_forward_batch_sizes=all_qwen_forward_batch_sizes,
                run_identity=run_identity,
                performance=performance,
            )
        if metrics.global_step % config.training.validation_every_optimizer_steps == 0:
            validation = _evaluate_validation(
                config=config,
                runtime=runtime,
                model=model,
                family_adapter=family_adapter,
                samples=validation_data.samples,
                group_builder=group_builder,
                validation_manifest_sha256=(validation_data.manifest.manifest_sha256),
                validation_event_index=validation_event_index,
            )
            validation_event_index += 1
            validation_sample_ids = _gather_string_tuples(validation.local_sample_ids)
            validation_group_keys = _gather_string_tuples(
                (validation.local_image_group_key,)
            )
            payload = asdict(validation)
            payload.pop("local_rank")
            payload.pop("local_image_group_key")
            payload.pop("local_sample_ids")
            payload.update(
                {
                    "event": "validation",
                    "global_step": metrics.global_step,
                    "run_identity_sha256": run_identity.identity_sha256,
                    "image_group_keys_by_rank": [
                        values[0] for values in validation_group_keys
                    ],
                    "sample_ids_by_rank": [
                        list(values) for values in validation_sample_ids
                    ],
                }
            )
            _append_metric_rank_zero_collective(
                config.output.metrics_jsonl_path,
                payload,
            )
        if metrics.global_step % config.checkpoint.save_every_optimizer_steps == 0:
            latest_checkpoint = _save_checkpoint(
                config=config,
                binding=binding,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_sampler,
                run_identity=run_identity,
                accumulation=accumulation,
                trainer_execution=trainer_execution,
                global_step=metrics.global_step,
                created_checkpoint_paths=created_checkpoint_paths,
            )
            latest_checkpoint_global_step = metrics.global_step

    invocation_complete = trainer.global_step == config.training.target_optimizer_steps
    if (config.checkpoint.save_final or not invocation_complete) and (
        latest_checkpoint is None
        or latest_checkpoint_global_step != trainer.global_step
    ):
        latest_checkpoint = _save_checkpoint(
            config=config,
            binding=binding,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_sampler,
            run_identity=run_identity,
            accumulation=accumulation,
            trainer_execution=trainer_execution,
            global_step=trainer.global_step,
            created_checkpoint_paths=created_checkpoint_paths,
        )
        latest_checkpoint_global_step = trainer.global_step

    if not invocation_complete:
        if latest_checkpoint is None:
            raise RuntimeError("bounded invocation did not commit a resume checkpoint")
        result = {
            "schema_version": REPRESENTATION_RUNNER_SCHEMA_VERSION,
            "status": "paused_at_optimizer_boundary",
            "run_id": config.run_id,
            "run_identity_sha256": run_identity.identity_sha256,
            "canonical_config_sha256": config.canonical_config_sha256,
            "source_toml_sha256": config.source_toml_sha256,
            "global_step": trainer.global_step,
            "planned_target_optimizer_steps": (config.training.target_optimizer_steps),
            "conditioning_provider": config.provider.provider.value,
            "image_max_pixels": config.model.image_max_pixels,
            "train_manifest_sha256": train_data.manifest.manifest_sha256,
            "validation_manifest_sha256": validation_data.manifest.manifest_sha256,
            "validation_data_identity_sha256": (
                None
                if validation_data_audit is None
                else validation_data_audit.identity.identity_sha256
            ),
            "split_overlap_report_sha256": overlap.identity_sha256,
            "final_artifact_path": None,
            "latest_checkpoint_path": str(latest_checkpoint),
            "metrics_jsonl_path": str(config.output.metrics_jsonl_path),
            "tokenizer_length_before": tokenizer_length_before,
            "tokenizer_length_after": len(processor.tokenizer),
            "world_size": world_size,
            "physical_gpu_ids": list(config.fsdp2.physical_gpu_ids),
            "logical_gpu_ids": list(config.fsdp2.logical_gpu_ids),
        }
        # Deliberately keep the durable metrics prefix checkpoint-aligned.  The
        # invocation result is written by the caller's stdout log; appending an
        # operational pause record here would advance history beyond the
        # checkpoint that the next process must verify before restore.
        return result if rank == 0 else None

    export = gather_rank_zero_full_adapter_owned_state(
        binding=binding,
        run_identity=run_identity,
        global_step=trainer.global_step,
    )
    artifact_write_mode = _save_rank_zero_export_collective(
        config.output.final_artifact_path,
        export,
        allow_existing_identical=config.resume.enabled,
    )
    if latest_checkpoint is None:
        raise RuntimeError("rank zero did not commit the final representation outputs")
    result: dict[str, object] = {
        "schema_version": REPRESENTATION_RUNNER_SCHEMA_VERSION,
        "status": "complete",
        "run_id": config.run_id,
        "run_identity_sha256": run_identity.identity_sha256,
        "canonical_config_sha256": config.canonical_config_sha256,
        "source_toml_sha256": config.source_toml_sha256,
        "global_step": trainer.global_step,
        "conditioning_provider": config.provider.provider.value,
        "image_max_pixels": config.model.image_max_pixels,
        "train_manifest_sha256": train_data.manifest.manifest_sha256,
        "validation_manifest_sha256": validation_data.manifest.manifest_sha256,
        "validation_data_identity_sha256": (
            None
            if validation_data_audit is None
            else validation_data_audit.identity.identity_sha256
        ),
        "split_overlap_report_sha256": overlap.identity_sha256,
        "final_artifact_path": str(config.output.final_artifact_path),
        "final_artifact_manifest_sha256": state_digest(export.manifest),
        "final_artifact_write_mode": artifact_write_mode,
        "latest_checkpoint_path": str(latest_checkpoint),
        "metrics_jsonl_path": str(config.output.metrics_jsonl_path),
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": len(processor.tokenizer),
        "world_size": world_size,
        "physical_gpu_ids": list(config.fsdp2.physical_gpu_ids),
        "logical_gpu_ids": list(config.fsdp2.logical_gpu_ids),
    }
    _append_metric_rank_zero_collective(
        config.output.metrics_jsonl_path,
        {"event": "complete", **result},
    )
    return result if rank == 0 else None


def _validate_invocation_stop(
    config: RepresentationTrainingConfig,
    stop_after_global_step: int | None,
) -> None:
    """Validate an operational stop without changing scientific identity.

    The configured target remains the scheduler horizon and immutable run
    identity.  This boundary only permits a process invocation to stop after a
    durable optimizer-boundary checkpoint so teardown/resume can be tested
    without constructing a different schedule.
    """

    if stop_after_global_step is None:
        return
    if (
        isinstance(stop_after_global_step, bool)
        or not isinstance(stop_after_global_step, int)
        or stop_after_global_step <= 0
    ):
        raise ValueError("stop_after_global_step must be a positive integer")
    if stop_after_global_step > config.training.target_optimizer_steps:
        raise ValueError(
            "stop_after_global_step cannot exceed the configured target step"
        )
    if stop_after_global_step % config.training.log_every_optimizer_steps:
        raise ValueError(
            "stop_after_global_step must align with a durable train-metric step"
        )


def _invocation_target_step(
    config: RepresentationTrainingConfig,
    *,
    initial_global_step: int,
    stop_after_global_step: int | None,
) -> int:
    _validate_invocation_stop(config, stop_after_global_step)
    target = (
        config.training.target_optimizer_steps
        if stop_after_global_step is None
        else stop_after_global_step
    )
    if target <= initial_global_step:
        raise ValueError(
            "invocation target step must be beyond the restored global step"
        )
    return target


def _load_datasets(
    config: RepresentationTrainingConfig,
) -> tuple[RepresentationDataset, RepresentationDataset]:
    train = load_retained_representation_jsonl(
        config.data.train.jsonl_path,
        expected_source_sha256=config.data.train.source_sha256,
        warn_on_leakage=config.data.warn_on_target_leakage,
    )
    validation = load_retained_representation_jsonl(
        config.data.validation.jsonl_path,
        expected_source_sha256=config.data.validation.source_sha256,
        warn_on_leakage=config.data.warn_on_target_leakage,
    )
    return train, validation


def _load_qwen3(
    config: RepresentationTrainingConfig,
    *,
    device: torch.device,
    rank: int,
) -> tuple[Any, torch.nn.Module]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Transformers Qwen runtime is unavailable") from error
    processor: Any | None = None
    processor_error: str | None = None
    try:
        processor = AutoProcessor.from_pretrained(
            config.model.local_path,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
    except Exception as error:
        processor_error = _exception_text(error)
    processor_errors: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(processor_errors, processor_error)
    if any(error is not None for error in processor_errors):
        raise RuntimeError(
            f"Qwen processor loading failed by rank: {tuple(processor_errors)}"
        )
    if processor is None:
        raise RuntimeError("current rank did not materialize the Qwen processor")

    model: torch.nn.Module | None = None
    for loader_rank in range(config.fsdp2.world_size):
        outcome: list[str | None] = [None]
        if rank == loader_rank:
            try:
                candidate = AutoModelForImageTextToText.from_pretrained(
                    config.model.local_path,
                    local_files_only=config.model.local_files_only,
                    trust_remote_code=config.model.trust_remote_code,
                    dtype=_torch_dtype(config.model.dtype),
                    attn_implementation=config.model.attention_backend,
                    low_cpu_mem_usage=True,
                )
                if not isinstance(candidate, torch.nn.Module):
                    raise TypeError("Qwen loader did not return an nn.Module")
                model = candidate.to(device=device)
            except Exception as error:
                outcome[0] = _exception_text(error)
        torch.distributed.broadcast_object_list(outcome, src=loader_rank)
        if outcome[0] is not None:
            raise RuntimeError(
                f"Qwen model loading failed on rank {loader_rank}: {outcome[0]}"
            )
    if model is None:
        raise RuntimeError("current rank did not materialize Qwen3")
    model.requires_grad_(False)
    return processor, model


def _build_fsdp2_policies(
    config: RepresentationTrainingConfig,
) -> tuple[Any, Any, Any]:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

    mesh = init_device_mesh(
        config.fsdp2.device_type,
        config.fsdp2.mesh_shape,
        mesh_dim_names=(config.fsdp2.mesh_dim_name,),
    )
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=_torch_dtype(config.fsdp2.parameter_dtype),
        reduce_dtype=_torch_dtype(config.fsdp2.reduce_dtype),
        output_dtype=_torch_dtype(config.fsdp2.output_dtype),
        cast_forward_inputs=config.fsdp2.cast_forward_inputs,
    )
    return mesh, mixed_precision, OffloadPolicy()


def _evaluate_validation(**kwargs: Any) -> Any:
    # Kept as a narrow late import so configuration validation does not import
    # the evaluation execution surface or initialize any distributed state.
    from .evaluation import evaluate_representation_validation_event

    config = kwargs.pop("config")
    runtime = kwargs.pop("runtime")
    return evaluate_representation_validation_event(
        adapter=runtime.adapter,
        qwen_model=kwargs.pop("model"),
        family_adapter=kwargs.pop("family_adapter"),
        samples=kwargs.pop("samples"),
        group_builder=kwargs.pop("group_builder"),
        objective=config.objective.objective,
        batch_size=config.data.validation.batch_size,
        sampler_seed=config.data.validation.sampler_seed,
        data_manifest_sha256=kwargs.pop("validation_manifest_sha256"),
        validation_event_index=kwargs.pop("validation_event_index"),
        data_parallel_world_size=config.fsdp2.world_size,
    )


def _save_checkpoint(
    *,
    config: RepresentationTrainingConfig,
    binding: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: SameImageBatchSampler,
    run_identity: RepresentationRunIdentity,
    accumulation: Any,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    global_step: int,
    created_checkpoint_paths: list[Path],
) -> Path:
    path = _checkpoint_path(config, global_step)
    metrics_history: RepresentationMetricsHistoryIdentity | None = None
    if isinstance(run_identity, RepresentationRunIdentityV3):
        metrics_history = load_representation_metrics_history(
            config.output.metrics_jsonl_path,
            run_id=config.run_id,
            run_identity_sha256=run_identity.identity_sha256,
            checkpoint_global_step=global_step,
            runner_schema_version=REPRESENTATION_RUNNER_SCHEMA_VERSION,
        ).identity
        _require_same_string_across_ranks(
            metrics_history.identity_sha256,
            name="representation metrics-history SHA256",
        )
    save_distributed_representation_checkpoint_atomic(
        path,
        binding=binding,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        run_identity=run_identity,
        accumulation=accumulation,
        trainer_execution=trainer_execution,
        global_step=global_step,
        metrics_history=metrics_history,
    )
    if path in created_checkpoint_paths:
        raise RuntimeError("current invocation attempted to recreate a checkpoint")
    created_checkpoint_paths.append(path)
    excess = max(
        0,
        len(created_checkpoint_paths) - config.checkpoint.keep_last,
    )
    retired = tuple(created_checkpoint_paths[:excess])
    outcome: list[str | None] = [None]
    if torch.distributed.get_rank() == 0:
        try:
            _remove_created_checkpoints_rank_zero(
                config,
                paths=retired,
                current=path,
            )
        except Exception as error:
            outcome[0] = _exception_text(error)
    torch.distributed.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(
            f"rank zero failed representation checkpoint retention: {outcome[0]}"
        )
    if excess:
        del created_checkpoint_paths[:excess]
    return path


def _checkpoint_path(config: RepresentationTrainingConfig, step: int) -> Path:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or step > 99_999_999
    ):
        raise ValueError("checkpoint step must be an integer in [1, 99999999]")
    return config.checkpoint.directory / (
        f"{config.checkpoint.filename_prefix}-step-{step:08d}"
    )


def _checkpoint_step(path: Path, prefix: str) -> int:
    marker = f"{prefix}-step-"
    if path.parent == path or not path.name.startswith(marker):
        raise ValueError("checkpoint path does not match the configured prefix")
    suffix = path.name[len(marker) :]
    if (
        len(suffix) != 8
        or not suffix.isascii()
        or not suffix.isdigit()
        or int(suffix) <= 0
    ):
        raise ValueError("checkpoint path has an invalid optimizer step suffix")
    return int(suffix)


def _remove_created_checkpoints_rank_zero(
    config: RepresentationTrainingConfig,
    *,
    paths: Sequence[Path],
    current: Path,
) -> None:
    directory = config.checkpoint.directory.resolve(strict=True)
    current_resolved = current.resolve(strict=True)
    if current_resolved.parent != directory:
        raise RuntimeError("current checkpoint escaped its configured directory")
    if len(set(paths)) != len(paths):
        raise RuntimeError(
            "current invocation checkpoint retention contains duplicates"
        )
    for path in paths:
        if path.parent != config.checkpoint.directory or path.is_symlink():
            raise RuntimeError("checkpoint retention received an unsafe created path")
        resolved = path.resolve(strict=True)
        if resolved == current_resolved or resolved.parent != directory:
            raise RuntimeError(
                "checkpoint retention resolved an unsafe deletion target"
            )
        shutil.rmtree(resolved)


def _prepare_output_paths(
    config: RepresentationTrainingConfig,
    *,
    rank: int,
) -> None:
    error: list[str | None] = [None]
    if rank == 0:
        try:
            if config.resume.enabled:
                if not config.output.metrics_jsonl_path.is_file():
                    raise FileNotFoundError(
                        "resume requires the existing metrics JSONL file"
                    )
            else:
                if config.output.final_artifact_path.exists():
                    raise FileExistsError("final Adapter artifact already exists")
                if config.output.metrics_jsonl_path.exists():
                    raise FileExistsError("fresh-run metrics JSONL already exists")
            config.output.final_artifact_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            config.output.metrics_jsonl_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            config.checkpoint.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exception:
            error[0] = f"{type(exception).__name__}: {exception}"
    torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"rank zero could not prepare outputs: {error[0]}")


def _validate_resume_metrics_history_collective(
    path: Path,
    *,
    run_id: str,
    run_identity_sha256: str,
    checkpoint_global_step: int,
    rank: int,
) -> None:
    error: list[str | None] = [None]
    if rank == 0:
        try:
            _validate_resume_metrics_history(
                path,
                run_id=run_id,
                run_identity_sha256=run_identity_sha256,
                checkpoint_global_step=checkpoint_global_step,
            )
        except Exception as exception:
            error[0] = _exception_text(exception)
    torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"resume metrics history is invalid: {error[0]}")


def _validate_resume_metrics_history(
    path: Path,
    *,
    run_id: str,
    run_identity_sha256: str,
    checkpoint_global_step: int,
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(checkpoint_global_step, bool)
        or not isinstance(checkpoint_global_step, int)
        or checkpoint_global_step < 0
    ):
        raise ValueError("checkpoint_global_step must be a non-negative integer")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("metrics JSONL must be non-empty and end with a newline")
    text = raw.decode("utf-8", errors="strict")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"metrics JSONL line {line_number} is empty")
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"metrics JSONL line {line_number} is not strict JSON"
            ) from error
        if not isinstance(value, dict):
            raise TypeError(f"metrics JSONL line {line_number} is not an object")
        records.append(value)

    first = records[0]
    if first.get("event") != "start":
        raise ValueError("metrics history must begin with a start event")
    if first.get("schema_version") != REPRESENTATION_RUNNER_SCHEMA_VERSION:
        raise ValueError("metrics start event has a different runner schema")
    if first.get("run_id") != run_id:
        raise ValueError("metrics start event has a different run_id")
    if first.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("metrics start event has a different run identity")

    exact_checkpoint_train_events = 0
    previous_step = -1
    for line_number, record in enumerate(records, start=1):
        event = record.get("event")
        if not isinstance(event, str) or not event:
            raise ValueError(f"metrics JSONL line {line_number} has no event")
        if event == "complete":
            raise ValueError("completed metrics history cannot be resumed")
        if "run_id" in record and record["run_id"] != run_id:
            raise ValueError(f"metrics JSONL line {line_number} changes run_id")
        if (
            "run_identity_sha256" in record
            and record["run_identity_sha256"] != run_identity_sha256
        ):
            raise ValueError(f"metrics JSONL line {line_number} changes run identity")
        step = record.get("global_step")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(
                    f"metrics JSONL line {line_number} has an invalid global_step"
                )
            if step < previous_step:
                raise ValueError("metrics global_step values must be non-decreasing")
            if step > checkpoint_global_step:
                raise ValueError("metrics history is advanced beyond the checkpoint")
            previous_step = step
        initial_step = record.get("initial_global_step")
        if initial_step is not None:
            if (
                isinstance(initial_step, bool)
                or not isinstance(initial_step, int)
                or initial_step < 0
                or initial_step > checkpoint_global_step
            ):
                raise ValueError(
                    f"metrics JSONL line {line_number} has an invalid initial step"
                )
        if event == "train":
            if step is None:
                raise ValueError("every train event must carry global_step")
            if step == checkpoint_global_step:
                exact_checkpoint_train_events += 1
    if exact_checkpoint_train_events != 1:
        raise ValueError(
            "metrics history must contain exactly one train event at the checkpoint"
        )
    return tuple(records)


def _log_training_metric(
    config: RepresentationTrainingConfig,
    *,
    metrics: RepresentationStepMetrics,
    all_sample_ids: tuple[tuple[str, ...], ...],
    all_qwen_forward_batch_sizes: tuple[tuple[int, ...], ...],
    run_identity: RepresentationRunIdentity,
    performance: RepresentationTrainStepPerformance,
) -> None:
    payload = asdict(metrics)
    payload.pop("local_sample_ids")
    payload.pop("local_qwen_forward_batch_sizes")
    qwen_forward_call_counts = tuple(
        len(batch_sizes) for batch_sizes in all_qwen_forward_batch_sizes
    )
    qwen_cell_evaluation_counts = tuple(
        sum(batch_sizes) for batch_sizes in all_qwen_forward_batch_sizes
    )
    payload.update(
        {
            "event": "train",
            "run_identity_sha256": run_identity.identity_sha256,
            "sample_ids_by_rank": [list(values) for values in all_sample_ids],
            "performance": {
                **asdict(performance),
                "max_rank_elapsed_seconds": (performance.max_rank_elapsed_seconds),
                "global_rows_per_second": performance.global_rows_per_second,
                "global_matrices_per_second": (performance.global_matrices_per_second),
                "qwen_physical_execution": {
                    "schema_version": (
                        REPRESENTATION_QWEN_PHYSICAL_EXECUTION_SCHEMA_VERSION
                    ),
                    "forward_batch_sizes_by_rank": [
                        list(batch_sizes)
                        for batch_sizes in all_qwen_forward_batch_sizes
                    ],
                    "forward_call_count_by_rank": list(qwen_forward_call_counts),
                    "cell_evaluation_count_by_rank": list(qwen_cell_evaluation_counts),
                    "max_forward_batch_size_by_rank": [
                        max(batch_sizes) for batch_sizes in all_qwen_forward_batch_sizes
                    ],
                    "global_forward_call_count": sum(qwen_forward_call_counts),
                    "global_cell_evaluation_count": sum(qwen_cell_evaluation_counts),
                },
            },
        }
    )
    _append_metric_rank_zero_collective(
        config.output.metrics_jsonl_path,
        payload,
    )


def _append_metric_rank_zero_collective(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    outcome: list[str | None] = [None]
    if torch.distributed.get_rank() == 0:
        try:
            _append_metric(path, payload)
        except Exception as error:
            outcome[0] = _exception_text(error)
    torch.distributed.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(f"rank zero could not append metrics: {outcome[0]}")


def _save_rank_zero_export_collective(
    path: Path,
    export: Any,
    *,
    allow_existing_identical: bool,
) -> str:
    outcome: list[str | None] = [None]
    if torch.distributed.get_rank() == 0:
        try:
            if not getattr(export, "is_writer", False):
                raise RuntimeError(
                    "rank zero did not receive the gathered export state"
                )
            if path.exists():
                if not allow_existing_identical:
                    raise FileExistsError("final Adapter artifact already exists")
                existing = load_rank_zero_adapter_owned_state_export(
                    path,
                    expected_run_identity=export.manifest.run_identity,
                )
                if existing.manifest != export.manifest:
                    raise RuntimeError(
                        "existing final Adapter artifact differs from resumed export"
                    )
                outcome[0] = "reused"
            else:
                wrote = save_rank_zero_adapter_owned_state_export_atomic(path, export)
                if wrote is not True:
                    raise RuntimeError("rank zero export writer returned false")
                outcome[0] = "written"
        except Exception as error:
            outcome[0] = f"ERROR:{_exception_text(error)}"
    torch.distributed.broadcast_object_list(outcome, src=0)
    value = outcome[0]
    if not isinstance(value, str):
        raise RuntimeError("rank zero did not report final Adapter artifact status")
    if value.startswith("ERROR:"):
        raise RuntimeError(
            f"rank zero could not publish final Adapter artifact: {value[6:]}"
        )
    if value not in {"written", "reused"}:
        raise RuntimeError("rank zero reported an invalid Adapter artifact status")
    return value


def _append_metric(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _gather_string_tuples(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    local = tuple(values)
    if not all(isinstance(value, str) and value for value in local):
        raise ValueError("sample IDs must be non-empty strings")
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    if not all(
        isinstance(value, tuple)
        and all(isinstance(item, str) and item for item in value)
        for value in gathered
    ):
        raise TypeError("distributed sample-ID gather returned malformed state")
    return tuple(gathered)  # type: ignore[arg-type]


def _gather_positive_int_tuples(
    values: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    local = tuple(values)
    if not local or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in local
    ):
        raise ValueError("Qwen forward batch sizes must be positive integers")
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    if not all(
        isinstance(value, tuple)
        and value
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in value
        )
        for value in gathered
    ):
        raise TypeError(
            "distributed Qwen forward-batch gather returned malformed state"
        )
    return tuple(gathered)  # type: ignore[arg-type]


def _load_rgb_image(path: str) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for representation images") from error
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _seed_current_process(seed: int) -> None:
    """Seed Python, CPU, and only the CUDA device selected for this rank."""

    random.seed(seed)
    torch.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _enable_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _require_launch_environment(config: RepresentationTrainingConfig) -> None:
    required = {
        "CUDA_VISIBLE_DEVICES": _REQUIRED_VISIBLE_DEVICES,
        "CUBLAS_WORKSPACE_CONFIG": _REQUIRED_CUBLAS_WORKSPACE,
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "WORLD_SIZE": str(config.fsdp2.world_size),
    }
    mismatches = {
        name: (expected, os.environ.get(name))
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"representation launch environment mismatch: {mismatches}")
    for name in ("RANK", "LOCAL_RANK"):
        _environment_integer(name)


def _environment_integer(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{name} must be a non-negative torchrun integer")
    return int(raw)


def _verify_live_code_identity(config: RepresentationTrainingConfig) -> None:
    root = Path(__file__).resolve().parents[4]
    configured = config.code.commit
    _run_git(root, "cat-file", "-e", f"{configured}^{{commit}}")
    changed = _run_git(
        root,
        "diff",
        "--name-only",
        configured,
        "HEAD",
        "--",
        *_CODE_IDENTITY_PATHS,
    ).strip()
    if changed:
        raise ValueError(
            "runtime code paths changed after configured code commit: " + changed
        )
    local_patch = _run_git(
        root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *_CODE_IDENTITY_PATHS,
    ).encode("utf-8")
    untracked = _run_git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *_CODE_IDENTITY_PATHS,
    ).encode("utf-8")
    dirty = bool(local_patch or untracked)
    if dirty != config.code.dirty:
        raise ValueError("live code dirty state differs from the TOML identity")
    if dirty:
        digest = sha256(b"tracked\0" + local_patch + b"untracked\0" + untracked)
        for relative_raw in sorted(value for value in untracked.split(b"\0") if value):
            relative = Path(relative_raw.decode("utf-8", errors="strict"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("git returned an unsafe untracked code path")
            file_path = (root / relative).resolve(strict=True)
            if not file_path.is_file() or root not in file_path.parents:
                raise ValueError(
                    "untracked code identity path is not a regular repo file"
                )
            digest.update(relative_raw)
            digest.update(b"\0")
            digest.update(file_path.read_bytes())
            digest.update(b"\0")
        if digest.hexdigest() != config.code.dirty_state_sha256:
            raise ValueError("live dirty code digest differs from the TOML identity")


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


def _require_same_string_across_ranks(value: str, *, name: str) -> None:
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    if any(item != value for item in gathered):
        raise RuntimeError(f"{name} differs across distributed ranks")


def _torch_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported representation dtype {name!r}") from error


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _exception_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


__all__ = [
    "REPRESENTATION_RUNNER_SCHEMA_VERSION",
    "run_representation_training",
]
