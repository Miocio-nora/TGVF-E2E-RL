"""Post-hoc, immutable true-1M evidence for official-visible Crop runs.

The legacy Crop evaluation identity binds the requested image cap but predates
an immutable processor proof.  This module closes that narrow evidence gap
without regenerating model outputs: it validates every durable result row,
checks every recorded native visual grid against the processor geometry, and
binds the exact inference files into a create-once receipt.

It deliberately does not rewrite an existing evaluation identity.  Future
identity schemas should bind the processor contract before inference; the
receipt here attests the effective visual grids of already-completed runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from .policy_coredev import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    load_bound_policy_benchmark_tasks,
    load_frozen_policy_evaluation_snapshot,
    load_policy_benchmark_results,
    load_policy_coredev_config,
    policy_evaluation_identity,
)
from .policy_official_visible import validate_official_visible_processor


TRUE1M_IMAGE_MAX_PIXELS = 1_003_520
TRUE1M_COREDEV_SINGLE_IMAGE_ROWS = 2_240
TRUE1M_AUDIT_RECEIPT_SCHEMA = "tgvf.official-visible-true1m-audit-receipt.v1"
TRUE1M_PROCESSOR_PROOF_SCHEMA = "tgvf.official-visible-processor-proof.v1"
TRUE1M_AUDIT_RECEIPT_FILENAME = "true1m-audit-receipt.json"
TRUE1M_PROCESSOR_PROOF_FILENAME = "official-visible-processor-proof.json"
_SUPPORTED_PLAN_SCHEMAS = {
    "tgvf.resolution-paired-policy-benchmark-plan.v4",
    "tgvf.resolution-projected-policy-benchmark-extension-plan.v5",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def canonical_sha256(payload: object) -> str:
    """Hash one JSON value with the repository's canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one regular, non-symlink file."""

    resolved = Path(path).resolve()
    if Path(path).is_symlink() or not resolved.is_file():
        raise RuntimeError(f"true1M audit input is not a regular file: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} is not a lowercase SHA256")
    return value


def _load_json_file(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return payload


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(f"immutable true1M audit output differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} path is malformed")
    path = Path(value)
    return (path if path.is_absolute() else _REPOSITORY_ROOT / path).resolve()


def _load_plan(path: Path) -> dict[str, Any]:
    payload = _load_json_file(path, name="true1M evaluation plan")
    if payload.get("schema_version") not in _SUPPORTED_PLAN_SCHEMAS:
        raise RuntimeError("true1M audit requires a v4/v5 resolution plan")
    if payload.get("status") != "ready":
        raise RuntimeError("true1M evaluation plan is not ready")
    return payload


def _plan_receipt_record(path: Path, plan: Mapping[str, object]) -> dict[str, object]:
    """Use the same path/size/sha ABI as every other receipt-bound file."""

    file_record = _file_record(path)
    return {
        **file_record,
        "file_sha256": file_record["sha256"],
        "identity_sha256": canonical_sha256(plan),
        "schema_version": plan["schema_version"],
        "evaluation_id": plan["evaluation_id"],
    }


def _plan_arm(plan: Mapping[str, object], arm_name: str) -> dict[str, Any]:
    arms = plan.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("true1M plan arms are malformed")
    matches = [
        arm for arm in arms if isinstance(arm, dict) and arm.get("name") == arm_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"true1M plan does not own exactly one {arm_name} arm")
    arm = matches[0]
    if arm.get("evaluation_image_max_pixels") != TRUE1M_IMAGE_MAX_PIXELS:
        raise RuntimeError(f"{arm_name} is not the true1M arm")
    return arm


def _validate_plan_sampling_rng(
    plan: Mapping[str, object],
    evaluation_identity: Mapping[str, object],
    *,
    arm_name: str,
) -> None:
    planned = plan.get("paired_rng")
    observed = evaluation_identity.get("sampling_rng")
    if not isinstance(planned, dict) or not isinstance(observed, dict):
        raise RuntimeError("true1M paired RNG contract is absent")
    planned_arms = planned.get("arm_protocol_sha256")
    expected = {
        "mode": planned.get("mode"),
        "seed_namespace": planned.get("seed_namespace"),
        "master_seed": planned.get("master_seed"),
        "task_manifest_sha256": planned.get("task_manifest_sha256"),
        "seed_protocol_sha256": planned.get("seed_protocol_sha256"),
        "protocol_projection": planned.get("protocol_projection"),
        "excluded_arm_components": planned.get("excluded_arm_components"),
        "arm_protocol_sha256": (
            planned_arms.get(arm_name) if isinstance(planned_arms, dict) else None
        ),
    }
    if any(observed.get(field) != value for field, value in expected.items()):
        raise RuntimeError("true1M evaluation RNG differs from plan")


def _preflight_complete_rank_files(inference_root: Path, *, world_size: int) -> None:
    """Reject incomplete outputs before acquiring any worker rank lock.

    The unlocked pass is intentionally only a completeness preflight.  Exact
    JSON, identity, and hash validation happens again while all worker locks
    are held.
    """

    line_count = 0
    for rank in range(world_size):
        path = inference_root / f"rank-{rank}.jsonl"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "true1M audit requires 2,240 complete rows before rank locks"
            )
        try:
            with path.open("rb") as handle:
                line_count += sum(bool(line.strip()) for line in handle)
        except OSError as error:
            raise RuntimeError("true1M rank preflight is unreadable") from error
    if line_count != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS:
        raise RuntimeError(
            "true1M audit requires 2,240 complete rows before rank locks"
        )


@contextmanager
def _completed_rank_locks(output_root: Path, *, world_size: int) -> Iterator[None]:
    """Hold existing worker lock files; never create or modify them."""

    handles: list[Any] = []
    try:
        for rank in range(world_size):
            lock_path = output_root / "runtime/locks" / f"rank-{rank}.lock"
            if lock_path.is_symlink() or not lock_path.is_file():
                raise RuntimeError(f"true1M rank lock is unavailable: rank {rank}")
            # Linux requires a writable descriptor for an exclusive flock on
            # these worker lock files.  ``r+b`` is deliberately non-creating;
            # the lstat-style checks above ensure we neither follow a symlink
            # nor materialize a missing runtime artifact.
            handle = lock_path.open("r+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(
                    f"true1M rank {rank} is still active; audit refused"
                ) from error
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def result_identity_sequence_sha256(
    records: Mapping[int, Mapping[str, object]],
) -> str:
    """Hash the sorted ordinal/result-identity sequence used by scoring."""

    sequence: list[dict[str, object]] = []
    for ordinal, row in sorted(records.items()):
        if type(ordinal) is not int:
            raise RuntimeError("true1M result ordinal is malformed")
        digest = _require_sha256(
            row.get("result_identity_sha256"), name="true1M result identity"
        )
        sequence.append({"ordinal": ordinal, "result_identity_sha256": digest})
    return canonical_sha256(sequence)


def audit_official_visible_true1m_records(
    records: Mapping[int, Mapping[str, object]],
    *,
    processor_proof: Mapping[str, object],
) -> dict[str, object]:
    """Validate per-turn visual grids and return deterministic evidence."""

    if len(records) != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS:
        raise RuntimeError("true1M audit requires exactly 2,240 result rows")
    if (
        processor_proof.get("configured_image_max_pixels") != TRUE1M_IMAGE_MAX_PIXELS
        or processor_proof.get("effective_processor_image_size")
        != {
            "shortest_edge": 65_536,
            "longest_edge": TRUE1M_IMAGE_MAX_PIXELS,
        }
        or processor_proof.get("runtime_mm_processor_kwargs")
        != {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": TRUE1M_IMAGE_MAX_PIXELS,
            }
        }
        or processor_proof.get("runtime_override_path")
        != "mm_processor_kwargs.size.longest_edge"
        or processor_proof.get("nested_images_kwargs_present") is not False
        or processor_proof.get("max_pixels_kwarg_present") is not False
        or processor_proof.get("vllm_012_shallow_hashable") is not True
    ):
        raise RuntimeError("official-visible processor proof is not flat true1M")
    patch_size = processor_proof.get("processor_patch_size")
    merge_size = processor_proof.get("processor_merge_size")
    if type(patch_size) is not int or type(merge_size) is not int:
        raise RuntimeError("true1M processor patch/merge geometry is malformed")
    pixel_quantum = (patch_size * merge_size) ** 2
    if pixel_quantum <= 0 or TRUE1M_IMAGE_MAX_PIXELS % pixel_quantum:
        raise RuntimeError("true1M visual-token pixel quantum differs")
    maximum_visual_tokens = TRUE1M_IMAGE_MAX_PIXELS // pixel_quantum

    turn_count = 0
    encoded_image_instance_count = 0
    maximum_observed_visual_tokens = 0
    count_evidence: list[dict[str, object]] = []
    for ordinal, row in sorted(records.items()):
        if row.get("result_kind", "trajectory") != "trajectory":
            raise RuntimeError("official-visible true1M row is not a trajectory")
        turns = row.get("assistant_turns")
        calls = row.get("tool_calls")
        if not isinstance(turns, list) or not turns:
            raise RuntimeError("official-visible true1M row lacks visual turn evidence")
        if not isinstance(calls, list) or any(
            not isinstance(call, dict) for call in calls
        ):
            raise RuntimeError("official-visible true1M tool calls are malformed")
        if (
            row.get("successful_observation_count") != len(calls)
            or row.get("native_original_image_count") != 1
            or row.get("native_crop_image_count") != len(calls)
            or row.get("native_total_image_count") != 1 + len(calls)
        ):
            raise RuntimeError("official-visible native image counters differ")
        native_hashes = row.get("native_image_sha256s")
        if not isinstance(native_hashes, list) or len(native_hashes) != 1 + len(calls):
            raise RuntimeError("official-visible native image hash count differs")

        row_turns: list[list[int]] = []
        for expected_turn_index, turn in enumerate(turns):
            if (
                not isinstance(turn, dict)
                or turn.get("turn_index") != expected_turn_index
            ):
                raise RuntimeError("official-visible assistant turn order differs")
            counts = turn.get("native_visual_token_counts")
            if (
                not isinstance(counts, list)
                or not counts
                or any(type(count) is not int or count <= 0 for count in counts)
            ):
                raise RuntimeError(
                    "official-visible native visual counts are malformed"
                )
            expected_images = 1 + sum(
                type(call.get("assistant_turn_index")) is int
                and call["assistant_turn_index"] < expected_turn_index
                for call in calls
            )
            if len(counts) != expected_images:
                raise RuntimeError(
                    "official-visible native visual count/image sequence differs"
                )
            if any(count > maximum_visual_tokens for count in counts):
                raise RuntimeError("official-visible native visual grid exceeds true1M")
            turn_count += 1
            encoded_image_instance_count += len(counts)
            maximum_observed_visual_tokens = max(
                maximum_observed_visual_tokens, max(counts)
            )
            row_turns.append(list(counts))
        count_evidence.append({"ordinal": ordinal, "turns": row_turns})

    return {
        "accepted_row_count": len(records),
        "assistant_turn_count": turn_count,
        "encoded_image_instance_count": encoded_image_instance_count,
        "visual_token_pixel_quantum": pixel_quantum,
        "maximum_allowed_visual_token_count": maximum_visual_tokens,
        "maximum_observed_visual_token_count": maximum_observed_visual_tokens,
        "maximum_observed_represented_pixel_area": (
            maximum_observed_visual_tokens * pixel_quantum
        ),
        "all_native_images_within_true1m": True,
        "turn_image_sequence_verified": True,
        "result_identity_sequence_sha256": result_identity_sequence_sha256(records),
        "native_visual_count_evidence_sha256": canonical_sha256(count_evidence),
    }


def _implementation_files() -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        (
            _REPOSITORY_ROOT / "src/tgvf_rl/evaluation/policy_official_visible.py"
        ).resolve(),
        (_REPOSITORY_ROOT / "tools/run_policy_benchmark.py").resolve(),
    )
    return [_file_record(path) for path in paths]


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _processor_proof_envelope(
    *,
    config_path: Path,
    evaluation_identity: Mapping[str, object],
    proof: Mapping[str, object],
) -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": TRUE1M_PROCESSOR_PROOF_SCHEMA,
        "attestation_scope": "posthoc_static_processor_contract",
        "evaluation_id": evaluation_identity["evaluation_id"],
        "evaluation_identity_sha256": evaluation_identity["identity_sha256"],
        "benchmark_config": _file_record(config_path),
        "model_identity": evaluation_identity["model_identity"],
        "implementation_files": _implementation_files(),
        "package_versions": {
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
        },
        "proof": dict(proof),
    }
    return {**content, "proof_identity_sha256": canonical_sha256(content)}


def _snapshot_file_records(output_root: Path) -> list[dict[str, object]]:
    candidates = (
        output_root / "runtime/frozen-full-model-state/materialization-receipt.json",
        output_root / "runtime/frozen-full-model-state/snapshot-manifest.json",
        output_root / "runtime/full-model-snapshot.json",
    )
    records = [_file_record(path) for path in candidates if path.is_file()]
    if not records:
        raise RuntimeError("true1M audit found no frozen snapshot evidence")
    return records


def _referenced_receipt_path(
    plan: Mapping[str, object],
    *,
    explicit_path: str | Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    reference = plan.get("rng_reference")
    if reference is None:
        return None
    if not isinstance(reference, dict):
        raise RuntimeError("true1M RNG reference is malformed")
    reference_plan_path = _resolve_repository_path(
        reference.get("plan_path"), name="true1M RNG reference plan"
    )
    if file_sha256(reference_plan_path) != reference.get("plan_sha256"):
        raise RuntimeError("true1M RNG reference plan SHA256 differs")
    reference_plan = _load_plan(reference_plan_path)
    reference_arm = _plan_arm(reference_plan, str(reference.get("arm_name")))
    owner = reference_plan.get("checkpoint_owner")
    if not isinstance(owner, dict) or not isinstance(owner.get("output_root"), str):
        raise RuntimeError("true1M RNG reference owner root is malformed")
    default_path = (
        Path(owner["output_root"]).resolve()
        / "evaluation"
        / str(reference["evaluation_id"])
        / str(reference["arm_name"])
        / "runtime"
        / TRUE1M_AUDIT_RECEIPT_FILENAME
    )
    path = Path(explicit_path).resolve() if explicit_path is not None else default_path
    if (
        reference_plan.get("evaluation_id") != reference.get("evaluation_id")
        or reference_arm.get("evaluation_image_max_pixels")
        != reference.get("evaluation_image_max_pixels")
        or reference_arm.get("evaluation_id") is None
    ):
        raise RuntimeError("true1M RNG reference arm identity differs")
    return path, reference_plan, reference_arm


def _reference_audit_record(
    plan: Mapping[str, object],
    *,
    explicit_path: str | Path | None,
) -> dict[str, object] | None:
    resolved = _referenced_receipt_path(plan, explicit_path=explicit_path)
    if resolved is None:
        return None
    path, reference_plan, reference_arm = resolved
    receipt = load_official_visible_true1m_audit_receipt(path)
    reference = plan["rng_reference"]
    assert isinstance(reference, dict)
    paired = plan.get("paired_rng")
    reference_paired = reference_plan.get("paired_rng")
    arm_protocols = (
        paired.get("arm_protocol_sha256") if isinstance(paired, dict) else None
    )
    reference_protocols = (
        reference_paired.get("arm_protocol_sha256")
        if isinstance(reference_paired, dict)
        else None
    )
    own_arm_name = (
        str(next(iter(arm_protocols))) if isinstance(arm_protocols, dict) else ""
    )
    if (
        receipt.get("plan", {}).get("file_sha256") != reference.get("plan_sha256")
        or receipt.get("plan", {}).get("evaluation_id")
        != reference.get("evaluation_id")
        or receipt.get("arm", {}).get("name") != reference.get("arm_name")
        or receipt.get("arm", {}).get("evaluation_id")
        != reference_arm.get("evaluation_id")
        or not isinstance(paired, dict)
        or not isinstance(reference_paired, dict)
        or paired.get("seed_namespace") != reference_paired.get("seed_namespace")
        or paired.get("seed_protocol_sha256")
        != reference_paired.get("seed_protocol_sha256")
        or not isinstance(arm_protocols, dict)
        or not isinstance(reference_protocols, dict)
        or arm_protocols.get(own_arm_name)
        != reference_protocols.get(str(reference.get("arm_name")))
    ):
        raise RuntimeError("true1M RNG reference audit identity differs")
    return {
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha256(path),
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "evaluation_id": receipt["arm"]["evaluation_id"],
        "arm_name": receipt["arm"]["name"],
        "result_identity_sequence_sha256": receipt["rows"][
            "result_identity_sequence_sha256"
        ],
    }


def materialize_official_visible_true1m_audit(
    *,
    config_path: str | Path,
    plan_path: str | Path,
    arm_name: str,
    rng_reference_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create one post-hoc receipt after all 2,240 inference rows are durable."""

    resolved_config = Path(config_path).resolve()
    resolved_plan = Path(plan_path).resolve()
    plan = _load_plan(resolved_plan)
    arm = _plan_arm(plan, arm_name)
    config = load_policy_coredev_config(resolved_config)
    if (
        config.evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        or config.evaluation_id != arm.get("evaluation_id")
        or config.evaluation_image_max_pixels != TRUE1M_IMAGE_MAX_PIXELS
        or config.expected_single_image_count != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS
        or len(config.gpu_ids) != 4
    ):
        raise RuntimeError("true1M benchmark config/arm contract differs")
    if (
        plan.get("task_manifest_sha256") != config.task_manifest_sha256
        or plan.get("expected_single_image_count") != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS
    ):
        raise RuntimeError("true1M task binding differs from plan")

    inference_root = config.output_root / "inference"
    _preflight_complete_rank_files(inference_root, world_size=4)
    with _completed_rank_locks(config.output_root, world_size=4):
        snapshot = load_frozen_policy_evaluation_snapshot(config)
        if snapshot.policy_version.optimizer_step != arm.get("optimizer_step"):
            raise RuntimeError("true1M snapshot optimizer step differs from plan")
        expected_identity = policy_evaluation_identity(config, snapshot)
        identity_path = config.output_root / "runtime/evaluation-identity.json"
        observed_identity = _load_json_file(
            identity_path, name="true1M evaluation identity"
        )
        if observed_identity != expected_identity:
            raise RuntimeError("true1M evaluation identity differs")
        preprocessing = observed_identity.get("image_preprocessing")
        protocol = observed_identity.get("protocol")
        if (
            not isinstance(preprocessing, dict)
            or preprocessing.get("max_pixels") != TRUE1M_IMAGE_MAX_PIXELS
            or not isinstance(protocol, dict)
            or protocol.get("image_max_pixels") != TRUE1M_IMAGE_MAX_PIXELS
            or protocol.get("profile") != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        ):
            raise RuntimeError("true1M evaluation identity cap/protocol differs")
        _validate_plan_sampling_rng(
            plan,
            observed_identity,
            arm_name=arm_name,
        )
        tasks = load_bound_policy_benchmark_tasks(config)
        records = load_policy_benchmark_results(
            inference_root,
            tasks=tasks,
            evaluation_identity=observed_identity,
            require_complete=True,
        )

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            snapshot.run.model.revision_or_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        proof = validate_official_visible_processor(
            processor,
            tokenizer_length=snapshot.run.model.tokenizer_length,
            image_max_pixels=TRUE1M_IMAGE_MAX_PIXELS,
        )
        row_evidence = audit_official_visible_true1m_records(
            records, processor_proof=proof
        )
        proof_envelope = _processor_proof_envelope(
            config_path=resolved_config,
            evaluation_identity=observed_identity,
            proof=proof,
        )
        proof_path = config.output_root / "runtime" / TRUE1M_PROCESSOR_PROOF_FILENAME
        reference_record = _reference_audit_record(
            plan, explicit_path=rng_reference_receipt_path
        )
        rank_files = []
        for rank in range(4):
            record = _file_record(inference_root / f"rank-{rank}.jsonl")
            record["rank"] = rank
            with (inference_root / f"rank-{rank}.jsonl").open(
                encoding="utf-8"
            ) as handle:
                record["line_count"] = sum(bool(line.strip()) for line in handle)
            rank_files.append(record)
        inference_identity = canonical_sha256(rank_files)
        _write_immutable_json(proof_path, proof_envelope)

        content: dict[str, object] = {
            "schema_version": TRUE1M_AUDIT_RECEIPT_SCHEMA,
            "status": "accepted",
            "attestation_scope": (
                "posthoc_effective_visual_grid_and_static_processor_contract"
            ),
            "generation_identity_extended": False,
            "plan": _plan_receipt_record(resolved_plan, plan),
            "arm": {
                "name": arm_name,
                "optimizer_step": arm["optimizer_step"],
                "evaluation_id": config.evaluation_id,
                "evaluation_image_max_pixels": TRUE1M_IMAGE_MAX_PIXELS,
                "output_root": str(config.output_root.resolve()),
                "benchmark_config": _file_record(resolved_config),
            },
            "evaluation_identity": {
                **_file_record(identity_path),
                "identity_sha256": observed_identity["identity_sha256"],
            },
            "task_manifest": {
                **_file_record(Path(observed_identity["task_manifest"]["path"])),
                "task_count": observed_identity["task_manifest"]["task_count"],
                "single_image_count": observed_identity["task_manifest"][
                    "single_image_count"
                ],
            },
            "snapshot_files": _snapshot_file_records(config.output_root),
            "processor_proof": {
                **_file_record(proof_path),
                "proof_identity_sha256": proof_envelope["proof_identity_sha256"],
            },
            "inference": {
                "world_size": 4,
                "files": rank_files,
                "tree_identity_sha256": inference_identity,
            },
            "rows": row_evidence,
        }
        if reference_record is not None:
            content["rng_reference_audit"] = reference_record
        receipt = {
            **content,
            "receipt_identity_sha256": canonical_sha256(content),
        }
        receipt_path = config.output_root / "runtime" / TRUE1M_AUDIT_RECEIPT_FILENAME
        _write_immutable_json(receipt_path, receipt)
    return load_official_visible_true1m_audit_receipt(receipt_path)


def _revalidate_file_record(record: object, *, name: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise RuntimeError(f"{name} file record is malformed")
    path = Path(str(record["path"]))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"{name} bytes differ from true1M receipt")
    return path.resolve()


def load_official_visible_true1m_audit_receipt(
    path: str | Path,
) -> dict[str, Any]:
    """Load and revalidate every file bound by one accepted receipt."""

    receipt_path = Path(path).resolve()
    payload = _load_json_file(receipt_path, name="true1M audit receipt")
    declared = _require_sha256(
        payload.get("receipt_identity_sha256"), name="true1M receipt identity"
    )
    content = dict(payload)
    content.pop("receipt_identity_sha256", None)
    if canonical_sha256(content) != declared:
        raise RuntimeError("true1M receipt internal identity differs")
    arm = payload.get("arm")
    rows = payload.get("rows")
    inference = payload.get("inference")
    if (
        payload.get("schema_version") != TRUE1M_AUDIT_RECEIPT_SCHEMA
        or payload.get("status") != "accepted"
        or payload.get("generation_identity_extended") is not False
        or not isinstance(arm, dict)
        or arm.get("evaluation_image_max_pixels") != TRUE1M_IMAGE_MAX_PIXELS
        or not isinstance(rows, dict)
        or rows.get("accepted_row_count") != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS
        or rows.get("all_native_images_within_true1m") is not True
        or rows.get("turn_image_sequence_verified") is not True
        or not isinstance(inference, dict)
        or inference.get("world_size") != 4
    ):
        raise RuntimeError("true1M receipt acceptance fields differ")
    for name, record in (
        ("plan", payload.get("plan")),
        ("benchmark config", arm.get("benchmark_config")),
        ("evaluation identity", payload.get("evaluation_identity")),
        ("task manifest", payload.get("task_manifest")),
        ("processor proof", payload.get("processor_proof")),
    ):
        if not isinstance(record, dict):
            raise RuntimeError(f"{name} record is malformed")
        file_record = {
            field: record.get(field) for field in ("path", "size_bytes", "sha256")
        }
        _revalidate_file_record(file_record, name=name)
    plan_record = payload["plan"]
    assert isinstance(plan_record, dict)
    if plan_record.get("file_sha256") != plan_record.get("sha256"):
        raise RuntimeError("plan SHA aliases differ in true1M receipt")
    snapshot_files = payload.get("snapshot_files")
    if not isinstance(snapshot_files, list) or not snapshot_files:
        raise RuntimeError("true1M receipt snapshot evidence is absent")
    for record in snapshot_files:
        _revalidate_file_record(record, name="snapshot")
    files = inference.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise RuntimeError("true1M receipt inference files differ")
    normalized_files: list[dict[str, object]] = []
    total_lines = 0
    for rank, record in enumerate(files):
        if not isinstance(record, dict) or record.get("rank") != rank:
            raise RuntimeError("true1M receipt rank order differs")
        file_record = {
            field: record.get(field) for field in ("path", "size_bytes", "sha256")
        }
        rank_path = _revalidate_file_record(file_record, name=f"rank {rank}")
        line_count = record.get("line_count")
        if type(line_count) is not int or line_count <= 0:
            raise RuntimeError("true1M receipt rank line count is malformed")
        with rank_path.open("rb") as handle:
            observed_lines = sum(bool(line.strip()) for line in handle)
        if observed_lines != line_count:
            raise RuntimeError("true1M receipt rank line count differs")
        total_lines += line_count
        normalized_files.append(dict(record))
    if total_lines != TRUE1M_COREDEV_SINGLE_IMAGE_ROWS or inference.get(
        "tree_identity_sha256"
    ) != canonical_sha256(normalized_files):
        raise RuntimeError("true1M receipt inference tree differs")
    proof_record = payload["processor_proof"]
    proof_payload = _load_json_file(
        Path(str(proof_record["path"])), name="true1M processor proof"
    )
    proof_identity = _require_sha256(
        proof_payload.get("proof_identity_sha256"),
        name="true1M processor proof identity",
    )
    proof_content = dict(proof_payload)
    proof_content.pop("proof_identity_sha256", None)
    if (
        canonical_sha256(proof_content) != proof_identity
        or proof_record.get("proof_identity_sha256") != proof_identity
    ):
        raise RuntimeError("true1M processor proof identity differs")
    static_proof = proof_payload.get("proof")
    if (
        proof_payload.get("schema_version") != TRUE1M_PROCESSOR_PROOF_SCHEMA
        or proof_payload.get("evaluation_id") != arm.get("evaluation_id")
        or not isinstance(static_proof, dict)
        or static_proof.get("configured_image_max_pixels") != TRUE1M_IMAGE_MAX_PIXELS
        or static_proof.get("effective_processor_image_size")
        != {
            "shortest_edge": 65_536,
            "longest_edge": TRUE1M_IMAGE_MAX_PIXELS,
        }
        or static_proof.get("runtime_mm_processor_kwargs")
        != {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": TRUE1M_IMAGE_MAX_PIXELS,
            }
        }
        or static_proof.get("runtime_override_path")
        != "mm_processor_kwargs.size.longest_edge"
        or static_proof.get("nested_images_kwargs_present") is not False
        or static_proof.get("max_pixels_kwarg_present") is not False
        or static_proof.get("vllm_012_shallow_hashable") is not True
    ):
        raise RuntimeError("true1M processor proof contract differs")
    reference = payload.get("rng_reference_audit")
    if reference is not None:
        expected_reference_fields = {
            "receipt_path",
            "receipt_file_sha256",
            "receipt_identity_sha256",
            "evaluation_id",
            "arm_name",
            "result_identity_sequence_sha256",
        }
        if not isinstance(reference, dict) or set(reference) != (
            expected_reference_fields
        ):
            raise RuntimeError("true1M composite reference record is malformed")
        reference_path = Path(str(reference["receipt_path"])).resolve()
        if reference_path == receipt_path:
            raise RuntimeError("true1M composite receipt references itself")
        true1m_file_sha = _require_sha256(
            reference.get("receipt_file_sha256"),
            name="true1M reference receipt file",
        )
        if (
            not reference_path.is_file()
            or reference_path.is_symlink()
            or file_sha256(reference_path) != true1m_file_sha
        ):
            raise RuntimeError("true1M composite reference receipt bytes differ")
        referenced = load_official_visible_true1m_audit_receipt(reference_path)
        if (
            referenced.get("receipt_identity_sha256")
            != reference.get("receipt_identity_sha256")
            or referenced.get("arm", {}).get("evaluation_id")
            != reference.get("evaluation_id")
            or referenced.get("arm", {}).get("name") != reference.get("arm_name")
            or referenced.get("rows", {}).get("result_identity_sequence_sha256")
            != reference.get("result_identity_sequence_sha256")
        ):
            raise RuntimeError("true1M composite reference receipt identity differs")
    return payload


__all__ = [
    "TRUE1M_AUDIT_RECEIPT_FILENAME",
    "TRUE1M_AUDIT_RECEIPT_SCHEMA",
    "TRUE1M_COREDEV_SINGLE_IMAGE_ROWS",
    "TRUE1M_IMAGE_MAX_PIXELS",
    "audit_official_visible_true1m_records",
    "canonical_sha256",
    "file_sha256",
    "load_official_visible_true1m_audit_receipt",
    "materialize_official_visible_true1m_audit",
    "result_identity_sequence_sha256",
]
