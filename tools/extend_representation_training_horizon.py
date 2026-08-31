#!/usr/bin/env python3
"""Clone a terminal representation checkpoint into an audited longer horizon.

This is an explicit lineage migration, not an ordinary resume.  A v3
representation run identity freezes ``planned_target_optimizer_steps`` and a
terminal metrics ledger cannot be resumed.  This tool creates a new isolated
lineage while preserving the checkpoint's DCP payload, optimizer, scheduler,
sampler, and RNG state byte-for-byte.  It never mutates the source checkpoint
or source metrics ledger.

The target training TOML must describe the longer-horizon resume and point at
paths below one new, nonexistent output root.  The migrated metrics ledger is
the exact checkpoint-bound source prefix with identity fields rebound, plus a
single explicit ``horizon_extension`` event.  The new checkpoint metadata is
then rebound to that new metrics-history identity.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch

from tgvf_rl.representation.training.checkpoint import (
    RepresentationOptimizerIdentity,
    RepresentationRunIdentityV3,
    RepresentationSchedulerIdentity,
    RepresentationTrainerExecutionIdentity,
)
from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    DistributedRepresentationCheckpointManifest,
    DistributedRepresentationMetadata,
    load_distributed_representation_checkpoint_metadata,
)
from tgvf_rl.representation.training.history import (
    RepresentationMetricsHistoryIdentity,
    load_representation_metrics_history,
)
from tgvf_rl.representation.training.runner import (
    REPRESENTATION_RUNNER_SCHEMA_VERSION,
)


LINEAGE_SCHEMA_VERSION = "representation-horizon-extension-lineage-v1"
EVENT_SCHEMA_VERSION = "representation-horizon-extension-event-v1"
_METADATA_NAME = "representation_metadata.pt"
_METADATA_DIGEST_NAME = "representation_metadata.sha256"
_DCP_DIRECTORY_NAME = "dcp"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--target-training-config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the migration plan without writing outputs.",
    )
    return parser


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(record)) for record in records)


def _read_strict_jsonl(raw: bytes, *, name: str) -> list[dict[str, object]]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{name} must be non-empty and end with a newline")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{name} line {line_number} is empty")
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{name} line {line_number} is not strict JSON") from error
        if not isinstance(value, dict):
            raise TypeError(f"{name} line {line_number} is not an object")
        records.append(value)
    return records


def _load_metrics_history_bytes(
    raw: bytes,
    *,
    run_id: str,
    run_identity_sha256: str,
    checkpoint_global_step: int,
) -> RepresentationMetricsHistoryIdentity:
    """Run the production history parser against an immutable byte buffer."""

    descriptor, raw_path = tempfile.mkstemp(
        prefix="representation-horizon-extension-history-",
        suffix=".jsonl",
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
        return load_representation_metrics_history(
            path,
            run_id=run_id,
            run_identity_sha256=run_identity_sha256,
            checkpoint_global_step=checkpoint_global_step,
            runner_schema_version=REPRESENTATION_RUNNER_SCHEMA_VERSION,
        ).identity
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _relative_to(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be below target output root") from error
    if not relative.parts:
        raise ValueError(f"{name} cannot equal target output root")
    return relative


def _checkpoint_step(path: Path, prefix: str) -> int:
    marker = f"{prefix}-step-"
    if not path.name.startswith(marker):
        raise ValueError("target resume checkpoint does not use checkpoint prefix")
    suffix = path.name[len(marker) :]
    if len(suffix) != 8 or not suffix.isdigit():
        raise ValueError("target resume checkpoint step must be eight digits")
    return int(suffix)


def _same_config_contract(
    source: RepresentationTrainingConfig,
    target: RepresentationTrainingConfig,
) -> None:
    """Permit only lineage/output/horizon/evaluation-resume changes."""

    exact_fields = (
        "code",
        "model",
        "provider",
        "data",
        "prompt",
        "objective",
        "optimizer",
        "scheduler",
        "execution",
        "initialization",
        "fsdp2",
        "adapter_variant",
    )
    changed = [
        field
        for field in exact_fields
        if getattr(source, field) != getattr(target, field)
    ]
    if changed:
        raise ValueError(
            "horizon extension changes scientific config fields: " + ", ".join(changed)
        )
    for field in (
        "gradient_accumulation_steps",
        "groups_per_rank_per_optimizer_step",
        "validation_every_optimizer_steps",
        "log_every_optimizer_steps",
    ):
        if getattr(source.training, field) != getattr(target.training, field):
            raise ValueError(f"horizon extension changes training.{field}")
    for field in (
        "save_every_optimizer_steps",
        "save_final",
        "keep_last",
        "strict_identity",
        "optimizer_boundary_only",
        "format",
    ):
        if getattr(source.checkpoint, field) != getattr(target.checkpoint, field):
            raise ValueError(f"horizon extension changes checkpoint.{field}")
    if source.run_id == target.run_id:
        raise ValueError("horizon extension requires a new run_id")
    if target.training.target_optimizer_steps <= source.training.target_optimizer_steps:
        raise ValueError("target horizon must be greater than source horizon")
    if target.training.target_optimizer_steps > target.scheduler.total_steps:
        raise ValueError("target horizon exceeds the unchanged scheduler horizon")
    if not target.resume.enabled or target.resume.checkpoint_path is None:
        raise ValueError("target training config must enable an exact resume path")


def _optimizer_identity_matches(
    config: RepresentationTrainingConfig,
    expected: RepresentationOptimizerIdentity,
) -> None:
    actual = config.optimizer
    fields = (
        "learning_rate",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    )
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(f"training config optimizer differs at {field}")
    if expected.optimizer_type != "torch.optim.adamw.AdamW":
        raise ValueError("checkpoint optimizer is not the configured AdamW")


def _config_matches_identity(
    config: RepresentationTrainingConfig,
    identity: RepresentationRunIdentityV3,
) -> None:
    checks = (
        (config.run_id, identity.run_id, "run_id"),
        (config.code_identity, identity.code, "code"),
        (config.model_identity, identity.model, "model"),
        (config.provider, identity.provider, "conditioning provider"),
        (config.prompt.sha256, identity.prompt_sha256, "prompt"),
        (config.objective.objective, identity.objective, "objective"),
        (config.accumulation_identity, identity.accumulation, "accumulation"),
        (
            RepresentationSchedulerIdentity.from_config(config.scheduler),
            identity.scheduler,
            "scheduler",
        ),
        (
            RepresentationTrainerExecutionIdentity.from_config(
                config.execution.trainer_config
            ),
            identity.trainer_execution,
            "trainer execution",
        ),
        (
            config.training.target_optimizer_steps,
            identity.planned_target_optimizer_steps,
            "planned horizon",
        ),
        (
            config.initialization.kind,
            identity.initialization.kind,
            "initialization kind",
        ),
        (
            config.initialization.seed,
            identity.initialization.seed,
            "initialization seed",
        ),
        (
            config.initialization.source_artifact_sha256,
            identity.initialization.source_artifact_sha256,
            "initialization source",
        ),
        (
            config.data.train.batch_size,
            identity.sampler_contract.batch_size,
            "train batch",
        ),
        (config.data.train.sampler_seed, identity.sampler_contract.seed, "train seed"),
        (config.fsdp2.world_size, identity.sampler_contract.world_size, "world size"),
        (
            config.data.validation.batch_size,
            identity.validation_identity.validation_batch_k,
            "validation batch",
        ),
        (
            config.data.validation.sampler_seed,
            identity.validation_identity.validation_sampler_seed,
            "validation seed",
        ),
        (
            config.training.validation_every_optimizer_steps,
            identity.validation_identity.validation_every_optimizer_steps,
            "validation cadence",
        ),
    )
    mismatches = [name for actual, expected, name in checks if actual != expected]
    if mismatches:
        raise ValueError(
            "training config differs from checkpoint identity: " + ", ".join(mismatches)
        )
    _optimizer_identity_matches(config, identity.optimizer)


def _validated_terminal_suffix(
    records: Sequence[Mapping[str, object]],
    *,
    source_identity: RepresentationRunIdentityV3,
    source_history: RepresentationMetricsHistoryIdentity,
    checkpoint_global_step: int,
    validation_every_optimizer_steps: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return terminal validations and the final completion record.

    Representation checkpoints are intentionally committed before validation.
    A clean terminal ledger may therefore contain validation at the checkpoint
    step between the checkpoint-bound prefix and its one final ``complete``
    event.  Those validation records are scientific history and must migrate;
    otherwise the resumed lineage would reuse an earlier validation cursor.
    """

    if not records:
        raise ValueError("source terminal metrics suffix is empty")
    complete_indices = tuple(
        index
        for index, record in enumerate(records)
        if record.get("event") == "complete"
    )
    if complete_indices != (len(records) - 1,):
        raise ValueError(
            "source terminal suffix requires exactly one final complete event"
        )
    complete = dict(records[-1])
    if (
        complete.get("global_step") != checkpoint_global_step
        or complete.get("run_id") != source_identity.run_id
        or complete.get("run_identity_sha256") != source_identity.identity_sha256
    ):
        raise ValueError("source complete event differs from terminal checkpoint")

    validations: list[dict[str, object]] = []
    next_index = source_history.next_validation_event_index
    for offset, source_record in enumerate(records[:-1]):
        record = dict(source_record)
        if record.get("event") != "validation":
            raise ValueError(
                "source terminal suffix may contain only same-step validation "
                "events before complete"
            )
        if record.get("global_step") != checkpoint_global_step:
            raise ValueError(
                "source terminal validation step differs from terminal checkpoint"
            )
        if record.get("validation_event_index") != next_index + offset:
            raise ValueError(
                "source terminal validation indices do not continue checkpoint history"
            )
        if ("run_id" in record and record["run_id"] != source_identity.run_id) or (
            "run_identity_sha256" in record
            and record["run_identity_sha256"] != source_identity.identity_sha256
        ):
            raise ValueError("source terminal validation changes run identity")
        validations.append(record)

    expected_next_index = checkpoint_global_step // validation_every_optimizer_steps
    if next_index + len(validations) != expected_next_index:
        raise ValueError(
            "source terminal validation count does not close the configured cadence"
        )
    return validations, complete


def _rebound_records(
    records: Sequence[Mapping[str, object]],
    *,
    source_identity: RepresentationRunIdentityV3,
    target_identity: RepresentationRunIdentityV3,
    source_checkpoint: Path,
    source_metadata_sha256: str,
    source_history: RepresentationMetricsHistoryIdentity,
    target_config: RepresentationTrainingConfig,
) -> list[dict[str, object]]:
    rebound: list[dict[str, object]] = []
    for line_number, source_record in enumerate(records, start=1):
        record = dict(source_record)
        if "run_id" in record:
            if record["run_id"] != source_identity.run_id:
                raise ValueError(
                    f"committed source metrics line {line_number} changes run_id"
                )
            record["run_id"] = target_identity.run_id
        if "run_identity_sha256" in record:
            if record["run_identity_sha256"] != source_identity.identity_sha256:
                raise ValueError(
                    f"committed source metrics line {line_number} changes run identity"
                )
            record["run_identity_sha256"] = target_identity.identity_sha256
        if "planned_target_optimizer_steps" in record:
            if (
                record["planned_target_optimizer_steps"]
                != source_identity.planned_target_optimizer_steps
            ):
                raise ValueError(
                    "committed source metrics line "
                    f"{line_number} changes planned horizon"
                )
            record["planned_target_optimizer_steps"] = (
                target_identity.planned_target_optimizer_steps
            )
        rebound.append(record)
    rebound.append(
        {
            "event": "horizon_extension",
            "schema_version": EVENT_SCHEMA_VERSION,
            "global_step": source_history.checkpoint_global_step,
            "run_id": target_identity.run_id,
            "run_identity_sha256": target_identity.identity_sha256,
            "source_run_id": source_identity.run_id,
            "source_run_identity_sha256": source_identity.identity_sha256,
            "source_planned_target_optimizer_steps": (
                source_identity.planned_target_optimizer_steps
            ),
            "target_planned_target_optimizer_steps": (
                target_identity.planned_target_optimizer_steps
            ),
            "source_checkpoint_path": str(source_checkpoint),
            "source_checkpoint_metadata_sha256": source_metadata_sha256,
            "source_metrics_history_identity_sha256": source_history.identity_sha256,
            "target_training_config_path": str(target_config.source_path),
            "target_training_config_sha256": target_config.source_toml_sha256,
            "target_canonical_config_sha256": target_config.canonical_config_sha256,
            "state_transition": "identity_rebind_only_no_optimizer_update",
        }
    )
    return rebound


def _tree_manifest(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"checkpoint payload is not a regular directory: {root}")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint payload contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"checkpoint payload contains no files: {root}")
    identity_payload = {"files": records}
    return {
        "files": records,
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "identity_sha256": _sha256_bytes(
            _canonical_json_bytes(identity_payload).rstrip(b"\n")
        ),
    }


def _metadata_bytes(metadata: DistributedRepresentationMetadata) -> bytes:
    metadata.__post_init__()
    buffer = BytesIO()
    torch.save(metadata, buffer)
    return buffer.getvalue()


def _assert_metadata_equivalent(
    actual: DistributedRepresentationMetadata,
    expected: DistributedRepresentationMetadata,
    *,
    phase: str,
) -> None:
    """Compare validated metadata without invoking Tensor-valued ``dict ==``.

    Rank-state ``__post_init__`` checks each sampler/RNG/scheduler payload
    against its recorded digest, while the manifest binds the ordered rank
    digests.  Comparing those typed, tensor-free bindings is therefore both
    stronger and safer than dataclass equality, which asks PyTorch tensors for
    an ambiguous scalar truth value.
    """

    actual.__post_init__()
    expected.__post_init__()
    if actual.manifest != expected.manifest:
        raise RuntimeError(f"{phase} checkpoint manifest differs from expected")
    if len(actual.rank_states) != len(expected.rank_states):
        raise RuntimeError(f"{phase} rank-state count differs from expected")
    fields = (
        "rank",
        "sampler_identity_sha256",
        "sampler_state_sha256",
        "rng_state_sha256",
        "scheduler_type",
        "scheduler_state_sha256",
        "schema_version",
    )
    for rank, (actual_rank, expected_rank) in enumerate(
        zip(actual.rank_states, expected.rank_states, strict=True)
    ):
        mismatches = [
            field
            for field in fields
            if getattr(actual_rank, field) != getattr(expected_rank, field)
        ]
        if mismatches:
            raise RuntimeError(
                f"{phase} rank {rank} digest bindings differ: " + ", ".join(mismatches)
            )


def _write_file_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_file_fsync(path: Path, payload: bytes) -> None:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scientific_metrics_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    normalized: list[dict[str, object]] = []
    for source in records:
        record = dict(source)
        for key in (
            "run_id",
            "run_identity_sha256",
            "planned_target_optimizer_steps",
        ):
            record.pop(key, None)
        normalized.append(record)
    return _sha256_bytes(_jsonl_bytes(normalized))


def _build_plan(
    *,
    source_training_config: Path,
    source_checkpoint: Path,
    source_metrics: Path,
    target_training_config: Path,
) -> dict[str, Any]:
    source_config = load_representation_training_config(source_training_config)
    target_config = load_representation_training_config(
        target_training_config,
        verify_external_files=False,
    )
    _same_config_contract(source_config, target_config)

    source_checkpoint = source_checkpoint.expanduser().resolve()
    source_metrics = source_metrics.expanduser().resolve()
    if source_checkpoint != source_config.resume.checkpoint_path and (
        source_checkpoint.parent != source_config.checkpoint.directory
    ):
        raise ValueError("source checkpoint does not belong to source config output")
    if source_metrics != source_config.output.metrics_jsonl_path:
        raise ValueError("source metrics path differs from source config")
    metadata = load_distributed_representation_checkpoint_metadata(source_checkpoint)
    manifest = metadata.manifest
    if not isinstance(manifest.run_identity, RepresentationRunIdentityV3):
        raise TypeError("horizon extension requires a v3 representation run identity")
    source_identity = manifest.run_identity
    _config_matches_identity(source_config, source_identity)
    if manifest.global_step != source_identity.planned_target_optimizer_steps:
        raise ValueError("source checkpoint is not the declared terminal probe")
    if manifest.global_step != source_config.training.target_optimizer_steps:
        raise ValueError("source checkpoint step differs from source config horizon")
    history = manifest.metrics_history
    if not isinstance(history, RepresentationMetricsHistoryIdentity):
        raise TypeError("source checkpoint does not bind a v2 metrics history")

    source_raw = source_metrics.read_bytes()
    if len(source_raw) <= history.byte_count:
        raise ValueError("source terminal metrics ledger has no terminal suffix")
    committed_raw = source_raw[: history.byte_count]
    if _sha256_bytes(committed_raw) != history.raw_bytes_sha256:
        raise ValueError("source checkpoint-bound metrics prefix digest mismatch")
    committed_records = _read_strict_jsonl(
        committed_raw,
        name="source checkpoint-bound metrics prefix",
    )
    if len(committed_records) != history.line_count:
        raise ValueError("source checkpoint-bound metrics line count mismatch")
    parsed_source_history = _load_metrics_history_bytes(
        committed_raw,
        run_id=source_identity.run_id,
        run_identity_sha256=source_identity.identity_sha256,
        checkpoint_global_step=manifest.global_step,
    )
    if parsed_source_history != history:
        raise ValueError(
            "production parser identity differs from checkpoint-bound source history"
        )
    terminal_records = _read_strict_jsonl(
        source_raw[history.byte_count :],
        name="source terminal metrics suffix",
    )
    terminal_validations, _complete = _validated_terminal_suffix(
        terminal_records,
        source_identity=source_identity,
        source_history=history,
        checkpoint_global_step=manifest.global_step,
        validation_every_optimizer_steps=(
            source_config.training.validation_every_optimizer_steps
        ),
    )
    migratable_records = [*committed_records, *terminal_validations]
    migrated_source_history = _load_metrics_history_bytes(
        _jsonl_bytes(migratable_records),
        run_id=source_identity.run_id,
        run_identity_sha256=source_identity.identity_sha256,
        checkpoint_global_step=manifest.global_step,
    )
    if migrated_source_history.next_validation_event_index != (
        history.next_validation_event_index + len(terminal_validations)
    ):
        raise RuntimeError("terminal validation migration did not advance the cursor")

    target_root = target_config.output.final_artifact_path.parent.resolve()
    if target_root.exists():
        raise FileExistsError(f"target output root already exists: {target_root}")
    if target_config.output.metrics_jsonl_path.parent.resolve() != target_root:
        raise ValueError("target metrics must be directly below target output root")
    if target_config.checkpoint.directory.parent.resolve() != target_root:
        raise ValueError("target checkpoint directory must be below target output root")
    assert target_config.resume.checkpoint_path is not None
    target_checkpoint = target_config.resume.checkpoint_path.resolve()
    if target_checkpoint.parent != target_config.checkpoint.directory.resolve():
        raise ValueError(
            "target resume checkpoint must be in target checkpoint directory"
        )
    if _checkpoint_step(
        target_checkpoint, target_config.checkpoint.filename_prefix
    ) != (manifest.global_step):
        raise ValueError("target resume checkpoint path step differs from source")
    target_metrics = target_config.output.metrics_jsonl_path.resolve()
    target_checkpoint_relative = _relative_to(
        target_checkpoint,
        target_root,
        name="target checkpoint",
    )
    target_metrics_relative = _relative_to(
        target_metrics,
        target_root,
        name="target metrics",
    )

    target_identity = replace(
        source_identity,
        run_id=target_config.run_id,
        planned_target_optimizer_steps=target_config.training.target_optimizer_steps,
    )
    target_identity.__post_init__()
    # This mirrors every identity field the production runner can derive from
    # TOML without loading Qwen.  The remaining initial Adapter checksum and
    # retained/image manifests are inherited unchanged from the fully validated
    # source identity; production restore recomputes and compares the complete
    # identity before applying any optimizer update.
    _config_matches_identity(target_config, target_identity)
    rebound_records = _rebound_records(
        migratable_records,
        source_identity=source_identity,
        target_identity=target_identity,
        source_checkpoint=source_checkpoint,
        source_metadata_sha256=_sha256_file(source_checkpoint / _METADATA_NAME),
        source_history=history,
        target_config=target_config,
    )
    target_metrics_raw = _jsonl_bytes(rebound_records)
    target_history = _load_metrics_history_bytes(
        target_metrics_raw,
        run_id=target_identity.run_id,
        run_identity_sha256=target_identity.identity_sha256,
        checkpoint_global_step=manifest.global_step,
    )
    if target_history.next_validation_event_index != (
        migrated_source_history.next_validation_event_index
    ):
        raise RuntimeError("identity rebind changed the validation cursor")
    target_manifest = replace(
        manifest,
        run_identity=target_identity,
        run_identity_sha256=target_identity.identity_sha256,
        metrics_history=target_history,
        metrics_history_identity_sha256=target_history.identity_sha256,
    )
    target_metadata = replace(metadata, manifest=target_manifest)
    target_metadata_raw = _metadata_bytes(target_metadata)

    source_dcp_tree = _tree_manifest(source_checkpoint / _DCP_DIRECTORY_NAME)
    source_metadata_sha256 = _sha256_file(source_checkpoint / _METADATA_NAME)
    source_metadata_sidecar = (
        (source_checkpoint / _METADATA_DIGEST_NAME).read_text(encoding="ascii").strip()
    )
    if source_metadata_sidecar != source_metadata_sha256:
        raise ValueError("source metadata sidecar digest mismatch")

    return {
        "source_config": source_config,
        "target_config": target_config,
        "source_checkpoint": source_checkpoint,
        "source_metrics": source_metrics,
        "source_metadata": metadata,
        "source_manifest": manifest,
        "source_identity": source_identity,
        "target_identity": target_identity,
        "source_history": history,
        "committed_records": committed_records,
        "terminal_records": terminal_records,
        "terminal_validation_records": terminal_validations,
        "migrated_source_history": migrated_source_history,
        "target_metrics_raw": target_metrics_raw,
        "target_history": target_history,
        "target_manifest": target_manifest,
        "target_metadata": target_metadata,
        "target_metadata_raw": target_metadata_raw,
        "target_metadata_sha256": _sha256_bytes(target_metadata_raw),
        "target_root": target_root,
        "target_checkpoint": target_checkpoint,
        "target_checkpoint_relative": target_checkpoint_relative,
        "target_metrics": target_metrics,
        "target_metrics_relative": target_metrics_relative,
        "source_dcp_tree": source_dcp_tree,
        "source_metadata_sha256": source_metadata_sha256,
        "source_metrics_full_sha256": _sha256_bytes(source_raw),
        "source_metrics_full_byte_count": len(source_raw),
        "source_terminal_suffix_sha256": _sha256_bytes(
            source_raw[history.byte_count :]
        ),
        "source_terminal_suffix_byte_count": len(source_raw) - history.byte_count,
        "source_scientific_metrics_sha256": _scientific_metrics_sha256(
            migratable_records
        ),
        "target_scientific_metrics_sha256": _scientific_metrics_sha256(
            rebound_records[:-1]
        ),
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, object]:
    source_identity = plan["source_identity"]
    target_identity = plan["target_identity"]
    manifest = plan["source_manifest"]
    history = plan["source_history"]
    migrated_source_history = plan["migrated_source_history"]
    target_history = plan["target_history"]
    target_config = plan["target_config"]
    assert isinstance(source_identity, RepresentationRunIdentityV3)
    assert isinstance(target_identity, RepresentationRunIdentityV3)
    assert isinstance(manifest, DistributedRepresentationCheckpointManifest)
    assert isinstance(history, RepresentationMetricsHistoryIdentity)
    assert isinstance(migrated_source_history, RepresentationMetricsHistoryIdentity)
    assert isinstance(target_history, RepresentationMetricsHistoryIdentity)
    assert isinstance(target_config, RepresentationTrainingConfig)
    scientific_equal = (
        plan["source_scientific_metrics_sha256"]
        == plan["target_scientific_metrics_sha256"]
    )
    if not scientific_equal:
        raise RuntimeError("metrics identity rebind changed scientific records")
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "status": "validated",
        "migration_kind": "terminal_probe_horizon_extension",
        "source": {
            "training_config_path": str(plan["source_config"].source_path),
            "training_config_sha256": plan["source_config"].source_toml_sha256,
            "checkpoint_path": str(plan["source_checkpoint"]),
            "checkpoint_metadata_sha256": plan["source_metadata_sha256"],
            "metrics_path": str(plan["source_metrics"]),
            "metrics_full_sha256": plan["source_metrics_full_sha256"],
            "metrics_full_byte_count": plan["source_metrics_full_byte_count"],
            "terminal_suffix_sha256": plan["source_terminal_suffix_sha256"],
            "terminal_suffix_byte_count": plan["source_terminal_suffix_byte_count"],
            "terminal_validation_count": len(plan["terminal_validation_records"]),
            "terminal_validation_event_indices": [
                record["validation_event_index"]
                for record in plan["terminal_validation_records"]
            ],
            "run_id": source_identity.run_id,
            "run_identity_sha256": source_identity.identity_sha256,
            "planned_target_optimizer_steps": (
                source_identity.planned_target_optimizer_steps
            ),
            "checkpoint_global_step": manifest.global_step,
            "metrics_history_identity_sha256": history.identity_sha256,
            "metrics_history_raw_bytes_sha256": history.raw_bytes_sha256,
            "dcp_payload": plan["source_dcp_tree"],
        },
        "target": {
            "training_config_path": str(target_config.source_path),
            "training_config_sha256": target_config.source_toml_sha256,
            "canonical_config_sha256": target_config.canonical_config_sha256,
            "output_root": str(plan["target_root"]),
            "checkpoint_path": str(plan["target_checkpoint"]),
            "metrics_path": str(plan["target_metrics"]),
            "run_id": target_identity.run_id,
            "run_identity_sha256": target_identity.identity_sha256,
            "planned_target_optimizer_steps": (
                target_identity.planned_target_optimizer_steps
            ),
            "expected_checkpoint_metadata_sha256": plan["target_metadata_sha256"],
            "expected_metrics_sha256": _sha256_bytes(plan["target_metrics_raw"]),
            "expected_metrics_byte_count": len(plan["target_metrics_raw"]),
            "expected_metrics_history_identity_sha256": plan[
                "target_history"
            ].identity_sha256,
            "expected_metrics_history_raw_bytes_sha256": plan[
                "target_history"
            ].raw_bytes_sha256,
        },
        "preserved_state": {
            "dcp_payload_byte_identical": True,
            "global_step": manifest.global_step,
            "world_size": manifest.world_size,
            "model_local_shard_sha256": list(manifest.model_local_shard_sha256),
            "optimizer_local_shard_sha256": list(manifest.optimizer_local_shard_sha256),
            "rank_state_sha256": list(manifest.rank_state_sha256),
            "sampler_state_sha256": [
                record.sampler_state_sha256
                for record in plan["source_metadata"].rank_states
            ],
            "rng_state_sha256": [
                record.rng_state_sha256
                for record in plan["source_metadata"].rank_states
            ],
            "scheduler_state_sha256": [
                record.scheduler_state_sha256
                for record in plan["source_metadata"].rank_states
            ],
            "source_checkpoint_next_validation_event_index": (
                history.next_validation_event_index
            ),
            "next_validation_event_index": (target_history.next_validation_event_index),
            "terminal_validation_events_migrated": len(
                plan["terminal_validation_records"]
            ),
            "scientific_metrics_sha256": plan["source_scientific_metrics_sha256"],
            "scientific_metrics_equal_after_identity_rebind": scientific_equal,
        },
        "rewritten_state": {
            "fields": [
                "run_identity.run_id",
                "run_identity.planned_target_optimizer_steps",
                "manifest.run_identity_sha256",
                "metrics run_id/run_identity/planned-horizon fields",
                "manifest.metrics_history",
                "manifest.metrics_history_identity_sha256",
            ],
            "source_complete_event_copied": False,
            "horizon_extension_event_appended": True,
            "optimizer_updates_performed": 0,
        },
    }


def _materialize(plan: dict[str, Any]) -> dict[str, object]:
    public = _public_plan(plan)
    target_root: Path = plan["target_root"]
    target_parent = target_root.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.horizon-extension-",
            dir=target_parent,
        )
    )
    committed = False
    try:
        staging_checkpoint = staging / plan["target_checkpoint_relative"]
        staging_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            plan["source_checkpoint"],
            staging_checkpoint,
            copy_function=shutil.copy2,
            symlinks=False,
        )
        copied_dcp_tree = _tree_manifest(staging_checkpoint / _DCP_DIRECTORY_NAME)
        if copied_dcp_tree != plan["source_dcp_tree"]:
            raise RuntimeError("copied DCP payload differs from source")

        staging_metrics = staging / plan["target_metrics_relative"]
        _write_file_fsync(staging_metrics, plan["target_metrics_raw"])
        target_history = load_representation_metrics_history(
            staging_metrics,
            run_id=plan["target_identity"].run_id,
            run_identity_sha256=plan["target_identity"].identity_sha256,
            checkpoint_global_step=plan["source_manifest"].global_step,
            runner_schema_version=REPRESENTATION_RUNNER_SCHEMA_VERSION,
        ).identity
        if target_history != plan["target_history"]:
            raise RuntimeError("staged metrics history differs from validated plan")
        target_metadata = plan["target_metadata"]
        metadata_raw = plan["target_metadata_raw"]
        metadata_sha256 = plan["target_metadata_sha256"]
        _replace_file_fsync(staging_checkpoint / _METADATA_NAME, metadata_raw)
        _replace_file_fsync(
            staging_checkpoint / _METADATA_DIGEST_NAME,
            f"{metadata_sha256}\n".encode("ascii"),
        )
        reloaded = load_distributed_representation_checkpoint_metadata(
            staging_checkpoint
        )
        _assert_metadata_equivalent(
            reloaded,
            target_metadata,
            phase="staged migrated",
        )
        if (
            _tree_manifest(staging_checkpoint / _DCP_DIRECTORY_NAME)
            != (plan["source_dcp_tree"])
        ):
            raise RuntimeError("metadata rewrite changed the copied DCP payload")

        public["status"] = "complete"
        public["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        target_public = public["target"]
        assert isinstance(target_public, dict)
        target_public.update(
            {
                "checkpoint_metadata_sha256": metadata_sha256,
                "metrics_sha256": _sha256_file(staging_metrics),
                "metrics_byte_count": staging_metrics.stat().st_size,
                "metrics_history_identity_sha256": target_history.identity_sha256,
                "metrics_history_raw_bytes_sha256": target_history.raw_bytes_sha256,
                "dcp_payload": copied_dcp_tree,
            }
        )
        lineage_name = (
            f"horizon-extension-step{plan['source_manifest'].global_step:08d}"
            f"-to{plan['target_identity'].planned_target_optimizer_steps:08d}.json"
        )
        lineage_path = staging / lineage_name
        public["target"]["lineage_manifest_path"] = str(target_root / lineage_name)
        # The manifest contains its final path.  Its own digest cannot be
        # recursively embedded, so that digest is added only to the returned
        # stdout payload after the immutable file has been written.
        _write_file_fsync(lineage_path, _canonical_json_bytes(public))
        public["target"]["lineage_manifest_sha256"] = _sha256_file(lineage_path)

        _fsync_directory(staging_checkpoint)
        _fsync_directory(staging_metrics.parent)
        _fsync_directory(staging)
        if target_root.exists():
            raise FileExistsError(f"target output root appeared: {target_root}")
        os.replace(staging, target_root)
        committed = True
        _fsync_directory(target_parent)

        final_metadata = load_distributed_representation_checkpoint_metadata(
            plan["target_checkpoint"]
        )
        _assert_metadata_equivalent(
            final_metadata,
            target_metadata,
            phase="committed target",
        )
        final_history = load_representation_metrics_history(
            plan["target_metrics"],
            run_id=plan["target_identity"].run_id,
            run_identity_sha256=plan["target_identity"].identity_sha256,
            checkpoint_global_step=plan["source_manifest"].global_step,
            runner_schema_version=REPRESENTATION_RUNNER_SCHEMA_VERSION,
        ).identity
        if final_history != target_history:
            raise RuntimeError("committed target metrics changed after publication")
        # With the target checkpoint now present, run the production config's
        # complete external-file validation as the final CPU-only gate.
        validated_target = load_representation_training_config(
            plan["target_config"].source_path
        )
        if validated_target.source_toml_sha256 != (
            plan["target_config"].source_toml_sha256
        ):
            raise RuntimeError("target config changed during migration")
        return public
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _build_plan(
        source_training_config=args.source_training_config,
        source_checkpoint=args.source_checkpoint,
        source_metrics=args.source_metrics,
        target_training_config=args.target_training_config,
    )
    result = _public_plan(plan) if args.validate_only else _materialize(plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
