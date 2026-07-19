from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.representation.training.distributed_checkpoint import (
    DistributedRepresentationRankState,
)
from tgvf_rl.representation.training.resume_parity import (
    _assert_exact_rank_state_identities,
    _scientific_metric_events,
)


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_scientific_metric_projection_excludes_only_performance(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(
        path,
        [
            {"event": "start", "run_id": "run"},
            {
                "event": "train",
                "global_step": 1,
                "global_total_loss": 2.5,
                "performance": {"elapsed_ns": 99},
            },
            {
                "event": "validation",
                "global_step": 1,
                "validation_event_index": 0,
                "global_total_loss": 2.0,
            },
            {"event": "resume", "initial_global_step": 1},
        ],
    )

    assert _scientific_metric_events(path) == (
        {"event": "train", "global_step": 1, "global_total_loss": 2.5},
        {
            "event": "validation",
            "global_step": 1,
            "validation_event_index": 0,
            "global_total_loss": 2.0,
        },
    )


def test_scientific_metric_projection_rejects_torn_or_nan(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event":"train","loss":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="newline terminated"):
        _scientific_metric_events(path)
    path.write_text('{"event":"train","loss":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        _scientific_metric_events(path)


def _rank_state(*, rng: torch.Tensor) -> DistributedRepresentationRankState:
    sampler_identity = "a" * 64
    sampler_state: dict[str, object] = {
        "identity_sha256": sampler_identity,
        "cursor": 4,
    }
    rng_state: dict[str, object] = {"torch_cpu": rng}
    scheduler_state: dict[str, object] = {"last_epoch": 2}
    return DistributedRepresentationRankState(
        rank=0,
        sampler_identity_sha256=sampler_identity,
        sampler_state=sampler_state,
        sampler_state_sha256=state_digest(sampler_state),
        rng_state=rng_state,
        rng_state_sha256=state_digest(rng_state),
        scheduler_type="torch.optim.lr_scheduler.LambdaLR",
        scheduler_state=scheduler_state,
        scheduler_state_sha256=state_digest(scheduler_state),
    )


def test_rank_state_comparison_uses_validated_tensor_digests() -> None:
    continuous = _rank_state(rng=torch.tensor([1, 2, 3], dtype=torch.uint8))
    independently_loaded = _rank_state(rng=torch.tensor([1, 2, 3], dtype=torch.uint8))

    _assert_exact_rank_state_identities((continuous,), (independently_loaded,))

    changed = _rank_state(rng=torch.tensor([1, 2, 4], dtype=torch.uint8))
    with pytest.raises(ReplayMismatchError, match="rng_state_sha256"):
        _assert_exact_rank_state_identities((continuous,), (changed,))
