from __future__ import annotations

import pytest

from tgvf_rl.representation.training.performance import (
    RepresentationRankTrainStepResources,
    RepresentationTrainStepPerformance,
    rank_resources_from_mapping,
)


def _rank(rank: int, elapsed_ns: int) -> RepresentationRankTrainStepResources:
    return RepresentationRankTrainStepResources(
        rank=rank,
        elapsed_ns=elapsed_ns,
        starting_allocated_bytes=100,
        starting_reserved_bytes=200,
        peak_allocated_bytes=150,
        peak_reserved_bytes=250,
        ending_allocated_bytes=110,
        ending_reserved_bytes=210,
    )


def test_train_step_performance_uses_slowest_rank_for_throughput() -> None:
    summary = RepresentationTrainStepPerformance(
        global_step=2,
        global_row_count=32,
        global_matrix_count=8,
        ranks=(_rank(0, 2_000_000_000), _rank(1, 4_000_000_000)),
    )

    assert summary.max_rank_elapsed_seconds == 4.0
    assert summary.global_rows_per_second == 8.0
    assert summary.global_matrices_per_second == 2.0


def test_rank_resource_json_reconstruction_is_fail_closed() -> None:
    payload = {
        "rank": 0,
        "elapsed_ns": 10,
        "starting_allocated_bytes": 100,
        "starting_reserved_bytes": 200,
        "peak_allocated_bytes": 150,
        "peak_reserved_bytes": 250,
        "ending_allocated_bytes": 110,
        "ending_reserved_bytes": 210,
    }

    assert rank_resources_from_mapping(payload) == _rank(0, 10)
    with pytest.raises(ValueError, match="fields differ"):
        rank_resources_from_mapping({**payload, "unexpected": 1})


def test_rank_resources_reject_impossible_peak() -> None:
    with pytest.raises(ValueError, match="peak allocated"):
        RepresentationRankTrainStepResources(
            rank=0,
            elapsed_ns=1,
            starting_allocated_bytes=100,
            starting_reserved_bytes=200,
            peak_allocated_bytes=99,
            peak_reserved_bytes=250,
            ending_allocated_bytes=100,
            ending_reserved_bytes=200,
        )
