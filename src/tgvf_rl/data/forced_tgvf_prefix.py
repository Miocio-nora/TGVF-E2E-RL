"""Strict prefix finalization for an in-progress forced-TGVF run.

The full utility run may continue writing later schedule rows while a one-step
Policy smoke only needs the first prompt batch.  This module snapshots that
complete prefix without changing the parent run identity or accepting partial,
unscored, mis-sharded, or differently identified attempts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tgvf_rl.data.forced_tgvf_counterfactual import (
    FORCED_TGVF_RUN_SCHEMA,
    ForcedTGVFRunPlan,
    _atomic_write,
    _canonical_json_bytes,
    _canonical_json_line,
    _file_sha256,
    _identity_path,
    _ledger_path,
    _load_json_object,
    _load_ledger,
    _positive_integer,
    _require_sha256,
    _sha256_bytes,
    load_forced_tgvf_schedule,
)


FORCED_TGVF_PREFIX_FINAL_SCHEMA = "tgvf.forced-tgvf-counterfactual.prefix-final.v1"
FORCED_TGVF_PREFIX_SELECTION = "training-index-contiguous-prefix-v1"


def _publish_exact(path: Path, payload: bytes, *, field: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"existing {field} must be a regular non-symlink file")
        if path.read_bytes() != payload:
            raise ValueError(f"existing {field} differs")
        return
    _atomic_write(path, payload)


def _ordered_sample_identity(samples: tuple[Any, ...]) -> list[dict[str, object]]:
    return [
        {
            "training_index": sample.training_index,
            "sample_id": sample.sample_id,
            "candidate_sha256": sample.candidate_sha256,
            "image_sha256": sample.image_sha256,
        }
        for sample in samples
    ]


def finalize_forced_tgvf_prefix(
    schedule_root: str | Path,
    output_root: str | Path,
    *,
    run_id: str,
    prefix_sample_count: int,
    attempts_per_sample: int = 8,
    shard_count: int = 4,
) -> dict[str, Any]:
    """Publish a complete canonical attempt prefix from a larger live run.

    Every source ledger is read as one atomic snapshot.  Rows after the
    requested prefix are validated against the full parent plan but excluded
    from the output.  The function is idempotent once identical bytes have
    been published.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    prefix_count = _positive_integer(prefix_sample_count, field="prefix_sample_count")
    expected_attempts_per_sample = _positive_integer(
        attempts_per_sample, field="attempts_per_sample"
    )
    expected_shard_count = _positive_integer(shard_count, field="shard_count")

    output = Path(output_root).resolve(strict=True)
    identity_path = _identity_path(output)
    identity_record = _load_json_object(identity_path, field="forced-TGVF run identity")
    if (
        identity_record.get("schema_version") != FORCED_TGVF_RUN_SCHEMA
        or identity_record.get("run_id") != run_id
    ):
        raise ValueError("forced-TGVF prefix run identity differs")
    run_identity_sha256 = _require_sha256(
        identity_record.get("run_identity_sha256"), field="run_identity_sha256"
    )
    identity = identity_record.get("identity")
    if (
        not isinstance(identity, Mapping)
        or _sha256_bytes(_canonical_json_bytes(identity)) != run_identity_sha256
        or identity.get("schema_version") != FORCED_TGVF_RUN_SCHEMA
        or identity.get("run_id") != run_id
    ):
        raise ValueError("forced-TGVF prefix parent identity differs")

    schedule_identity = identity.get("schedule")
    sampling_identity = identity.get("sampling")
    execution_identity = identity.get("execution")
    if (
        not isinstance(schedule_identity, Mapping)
        or not isinstance(sampling_identity, Mapping)
        or not isinstance(execution_identity, Mapping)
    ):
        raise ValueError("forced-TGVF prefix parent plan is incomplete")
    full_sample_count = _positive_integer(
        schedule_identity.get("sample_count"), field="full sample_count"
    )
    if prefix_count > full_sample_count:
        raise ValueError("prefix_sample_count exceeds the parent run")
    if (
        sampling_identity.get("attempts_per_sample") != expected_attempts_per_sample
        or execution_identity.get("shard_count") != expected_shard_count
        or execution_identity.get("attempt_count")
        != full_sample_count * expected_attempts_per_sample
        or execution_identity.get("shard_assignment")
        != "training_index_mod_shard_count_v1"
    ):
        raise ValueError("forced-TGVF prefix execution parameters differ from run")

    samples, _schedule_manifest, schedule_manifest_sha256 = load_forced_tgvf_schedule(
        schedule_root,
        sample_count=full_sample_count,
    )
    if schedule_identity.get(
        "manifest_sha256"
    ) != schedule_manifest_sha256 or schedule_identity.get(
        "ordered_samples"
    ) != _ordered_sample_identity(samples):
        raise ValueError("forced-TGVF prefix schedule differs from parent run")

    plan = ForcedTGVFRunPlan(
        run_id=run_id,
        run_identity_sha256=run_identity_sha256,
        identity=identity,
        samples=samples,
    )
    prefix_samples = samples[:prefix_count]
    expected = {
        (sample.sample_id, attempt_index): sample
        for sample in prefix_samples
        for attempt_index in range(expected_attempts_per_sample)
    }
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    shard_projections: list[dict[str, object]] = []
    for shard_index in range(expected_shard_count):
        ledger_path = _ledger_path(output, shard_index)
        rows = _load_ledger(
            ledger_path,
            plan=plan,
            shard_index=shard_index,
            shard_count=expected_shard_count,
        )
        overlap = set(merged).intersection(rows)
        if overlap:
            raise ValueError("forced-TGVF prefix shard ledgers overlap")
        merged.update(rows)
        projected = sorted(
            (row for row in rows.values() if int(row["training_index"]) < prefix_count),
            key=lambda row: (int(row["training_index"]), int(row["attempt_index"])),
        )
        projected_payload = b"".join(_canonical_json_line(row) for row in projected)
        shard_projections.append(
            {
                "shard_index": shard_index,
                "source_path": str(ledger_path),
                "prefix_rows": len(projected),
                "prefix_sha256": _sha256_bytes(projected_payload),
            }
        )

    observed_prefix = {
        key for key, row in merged.items() if int(row["training_index"]) < prefix_count
    }
    if observed_prefix != set(expected):
        missing = len(set(expected).difference(observed_prefix))
        extra = len(observed_prefix.difference(expected))
        raise ValueError(
            "forced-TGVF prefix attempts are incomplete: "
            f"missing={missing} extra={extra}"
        )
    ordered = sorted(
        (merged[key] for key in expected),
        key=lambda row: (int(row["training_index"]), int(row["attempt_index"])),
    )
    for row in ordered:
        if row.get("status") != "scored" or type(row.get("correct")) is not bool:
            raise ValueError("forced-TGVF prefix contains an unscored attempt")

    attempts_payload = b"".join(_canonical_json_line(row) for row in ordered)
    attempts_sha256 = _sha256_bytes(attempts_payload)
    prefix_root = output / "prefixes" / f"first-{prefix_count:06d}"
    attempts_path = prefix_root / "attempts.jsonl"
    manifest_path = prefix_root / "manifest.json"
    manifest_identity: dict[str, object] = {
        "schema_version": FORCED_TGVF_PREFIX_FINAL_SCHEMA,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "run_identity_file_sha256": _file_sha256(identity_path),
        "schedule_manifest_sha256": schedule_manifest_sha256,
        "parent_sample_count": full_sample_count,
        "selection": {
            "contract": FORCED_TGVF_PREFIX_SELECTION,
            "training_index_start": 0,
            "training_index_stop_exclusive": prefix_count,
            "sample_count": prefix_count,
        },
        "attempts_per_sample": expected_attempts_per_sample,
        "shard_count": expected_shard_count,
        "attempt_count": len(ordered),
        "correct_count": sum(bool(row["correct"]) for row in ordered),
        "source_shard_prefixes": shard_projections,
        "attempts": {
            "path": "attempts.jsonl",
            "rows": len(ordered),
            "sha256": attempts_sha256,
        },
    }
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest_identity))
    manifest_payload = _canonical_json_line(
        {**manifest_identity, "manifest_sha256": manifest_sha256}
    )
    _publish_exact(attempts_path, attempts_payload, field="forced-TGVF prefix attempts")
    _publish_exact(manifest_path, manifest_payload, field="forced-TGVF prefix manifest")
    return {
        **manifest_identity,
        "manifest_sha256": manifest_sha256,
        "attempts_path": str(attempts_path),
        "manifest_path": str(manifest_path),
    }


__all__ = [
    "FORCED_TGVF_PREFIX_FINAL_SCHEMA",
    "FORCED_TGVF_PREFIX_SELECTION",
    "finalize_forced_tgvf_prefix",
]
