from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "extend_representation_training_horizon.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "extend_representation_training_horizon", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
extension = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extension
_SPEC.loader.exec_module(extension)


_RUN_ID = "RP-TEST-500"
_RUN_IDENTITY = "a" * 64
_CHECKPOINT_STEP = 500


def _checkpoint_prefix() -> list[dict[str, object]]:
    return [
        {
            "event": "start",
            "schema_version": extension.REPRESENTATION_RUNNER_SCHEMA_VERSION,
            "run_id": _RUN_ID,
            "run_identity_sha256": _RUN_IDENTITY,
            "global_step": 0,
        },
        {
            "event": "train",
            "run_identity_sha256": _RUN_IDENTITY,
            "global_step": _CHECKPOINT_STEP,
        },
    ]


def _source_history():
    return extension._load_metrics_history_bytes(
        extension._jsonl_bytes(_checkpoint_prefix()),
        run_id=_RUN_ID,
        run_identity_sha256=_RUN_IDENTITY,
        checkpoint_global_step=_CHECKPOINT_STEP,
    )


def _identity() -> SimpleNamespace:
    return SimpleNamespace(run_id=_RUN_ID, identity_sha256=_RUN_IDENTITY)


def _validation(*, step: int = _CHECKPOINT_STEP, index: int = 0):
    return {
        "event": "validation",
        "run_identity_sha256": _RUN_IDENTITY,
        "global_step": step,
        "validation_event_index": index,
    }


def _complete():
    return {
        "event": "complete",
        "run_id": _RUN_ID,
        "run_identity_sha256": _RUN_IDENTITY,
        "global_step": _CHECKPOINT_STEP,
    }


def test_terminal_validation_is_promoted_into_resumable_history() -> None:
    source_history = _source_history()

    validations, complete = extension._validated_terminal_suffix(
        [_validation(), _complete()],
        source_identity=_identity(),
        source_history=source_history,
        checkpoint_global_step=_CHECKPOINT_STEP,
        validation_every_optimizer_steps=500,
    )
    migrated_history = extension._load_metrics_history_bytes(
        extension._jsonl_bytes([*_checkpoint_prefix(), *validations]),
        run_id=_RUN_ID,
        run_identity_sha256=_RUN_IDENTITY,
        checkpoint_global_step=_CHECKPOINT_STEP,
    )

    assert validations == [_validation()]
    assert complete == _complete()
    assert source_history.next_validation_event_index == 0
    assert migrated_history.next_validation_event_index == 1


def test_complete_only_suffix_remains_valid_when_no_validation_is_due() -> None:
    validations, complete = extension._validated_terminal_suffix(
        [_complete()],
        source_identity=_identity(),
        source_history=_source_history(),
        checkpoint_global_step=_CHECKPOINT_STEP,
        validation_every_optimizer_steps=2000,
    )

    assert validations == []
    assert complete == _complete()


@pytest.mark.parametrize(
    ("suffix", "message"),
    (
        (
            [_validation(step=499), _complete()],
            "validation step differs",
        ),
        (
            [_validation(index=1), _complete()],
            "indices do not continue",
        ),
        (
            [
                {
                    **_validation(),
                    "run_identity_sha256": "b" * 64,
                },
                _complete(),
            ],
            "validation changes run identity",
        ),
        (
            [{"event": "paused", "global_step": 500}, _complete()],
            "only same-step validation",
        ),
        (
            [_complete(), _complete()],
            "exactly one final complete",
        ),
        (
            [_complete()],
            "count does not close",
        ),
        (
            [
                _validation(),
                {
                    **_complete(),
                    "run_id": "RP-FOREIGN-500",
                },
            ],
            "complete event differs",
        ),
    ),
)
def test_terminal_suffix_rejects_non_replayable_sequences(
    suffix: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        extension._validated_terminal_suffix(
            suffix,
            source_identity=_identity(),
            source_history=_source_history(),
            checkpoint_global_step=_CHECKPOINT_STEP,
            validation_every_optimizer_steps=500,
        )
