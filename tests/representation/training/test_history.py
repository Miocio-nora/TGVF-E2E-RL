from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.history import (
    RepresentationMetricsHistoryIdentity,
    load_representation_metrics_history,
)


RUNNER_SCHEMA = "representation-runner-v1"
RUN_IDENTITY = "a" * 64


def _records() -> list[dict[str, object]]:
    return [
        {
            "event": "start",
            "schema_version": RUNNER_SCHEMA,
            "run_id": "run",
            "run_identity_sha256": RUN_IDENTITY,
            "initial_global_step": 0,
        },
        {
            "event": "train",
            "run_identity_sha256": RUN_IDENTITY,
            "global_step": 1,
        },
        {
            "event": "validation",
            "run_identity_sha256": RUN_IDENTITY,
            "global_step": 1,
            "validation_event_index": 0,
        },
        {
            "event": "resume",
            "schema_version": RUNNER_SCHEMA,
            "run_id": "run",
            "run_identity_sha256": RUN_IDENTITY,
            "initial_global_step": 1,
        },
        {
            "event": "train",
            "run_identity_sha256": RUN_IDENTITY,
            "global_step": 2,
        },
        {
            "event": "validation",
            "run_identity_sha256": RUN_IDENTITY,
            "global_step": 2,
            "validation_event_index": 1,
        },
    ]


def _write(path: Path, records: list[dict[str, object]]) -> bytes:
    raw = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_history_identity_binds_exact_bytes_and_validation_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    raw = _write(path, _records())

    parsed = load_representation_metrics_history(
        path,
        run_id="run",
        run_identity_sha256=RUN_IDENTITY,
        checkpoint_global_step=2,
        runner_schema_version=RUNNER_SCHEMA,
    )

    assert isinstance(parsed.identity, RepresentationMetricsHistoryIdentity)
    assert parsed.identity.byte_count == len(raw)
    assert parsed.identity.line_count == 6
    assert parsed.identity.next_validation_event_index == 2
    assert len(parsed.identity.identity_sha256) == 64
    assert len(parsed.records) == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("advanced", "advanced beyond"),
        ("complete", "terminal or operational pause"),
        ("paused", "terminal or operational pause"),
        ("identity", "changes run identity"),
        ("validation_gap", "complete, unique, and contiguous"),
        ("missing_checkpoint_train", "exactly one train event"),
    ),
)
def test_history_rejects_non_checkpoint_prefixes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    records = _records()
    if mutation == "advanced":
        records[-1]["global_step"] = 3
    elif mutation == "complete":
        records.append({"event": "complete", "global_step": 2})
    elif mutation == "paused":
        records.append({"event": "paused", "global_step": 2})
    elif mutation == "identity":
        records[-1]["run_identity_sha256"] = "b" * 64
    elif mutation == "validation_gap":
        records[-1]["validation_event_index"] = 2
    elif mutation == "missing_checkpoint_train":
        records.pop(-2)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    path = tmp_path / "metrics.jsonl"
    _write(path, records)

    with pytest.raises((TypeError, ValueError), match=message):
        load_representation_metrics_history(
            path,
            run_id="run",
            run_identity_sha256=RUN_IDENTITY,
            checkpoint_global_step=2,
            runner_schema_version=RUNNER_SCHEMA,
        )


def test_history_rejects_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    raw = _write(path, _records())
    path.write_bytes(raw.removesuffix(b"\n"))

    with pytest.raises(ValueError, match="end with a newline"):
        load_representation_metrics_history(
            path,
            run_id="run",
            run_identity_sha256=RUN_IDENTITY,
            checkpoint_global_step=2,
            runner_schema_version=RUNNER_SCHEMA,
        )
