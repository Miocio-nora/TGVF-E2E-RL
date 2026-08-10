from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.trainable_tgvf_supervisor import (
    checkpointed_step,
    is_weight_wake_oom,
    supervisor_decision,
)


def test_checkpointed_step_reads_only_committed_tracker(tmp_path: Path) -> None:
    assert checkpointed_step(tmp_path) == 0
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4\n", encoding="utf-8")
    assert checkpointed_step(tmp_path) == 4


def test_checkpointed_step_rejects_malformed_tracker(tmp_path: Path) -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(
        "step5", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="malformed checkpoint tracker"):
        checkpointed_step(tmp_path)


def test_supervisor_retries_only_exact_weight_wake_oom() -> None:
    exact = (
        'rollout.resume(tags=["weights"])\n'
        "CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:112"
    )
    assert is_weight_wake_oom(exact)
    assert (
        supervisor_decision(
            return_code=1,
            checkpoint_step=4,
            target_step=8,
            attempt_log=exact,
            completed_restarts=0,
            maximum_restarts=8,
        )
        == "retry_weight_wake_oom"
    )
    assert not is_weight_wake_oom("HTTP 402")
    assert (
        supervisor_decision(
            return_code=1,
            checkpoint_step=4,
            target_step=8,
            attempt_log="HTTP 402",
            completed_restarts=0,
            maximum_restarts=8,
        )
        == "fail"
    )


def test_supervisor_completion_and_retry_bound() -> None:
    exact = (
        'resume(tags=["weights"])\n'
        "CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:112"
    )
    assert (
        supervisor_decision(
            return_code=1,
            checkpoint_step=8,
            target_step=8,
            attempt_log=exact,
            completed_restarts=8,
            maximum_restarts=8,
        )
        == "complete"
    )
    assert (
        supervisor_decision(
            return_code=1,
            checkpoint_step=4,
            target_step=8,
            attempt_log=exact,
            completed_restarts=8,
            maximum_restarts=8,
        )
        == "fail"
    )
