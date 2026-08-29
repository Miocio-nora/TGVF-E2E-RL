"""Bounded clean-process recovery for a formal trainable-TGVF run.

The upstream trainer already resumes from the last committed checkpoint.  This
supervisor adds two operational guarantees: the known vLLM CuMem weight-wake
OOM and an exhausted, explicitly classified OpenRouter HTTP-429 judge window
are retried in a fresh process without waiting for a person to notice them. It
never retries an unclassified failure and never treats an in-memory optimizer
update as durable progress.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


SupervisorDecision = Literal[
    "complete",
    "retry_weight_wake_oom",
    "retry_judge_transient_429",
    "fail",
]
_WEIGHT_WAKE_OOM_MARKERS = (
    'resume(tags=["weights"])',
    "CUDA Error: out of memory",
    "cumem_allocator.cpp:112",
)
_JUDGE_TRANSIENT_429_MARKERS = (
    (
        "tgvf_rl.rewards.deepeyes_verl_reward._JudgeTransientHTTPError: "
        "DeepEyes judge transient HTTP failure 429"
    ),
    (
        "tgvf_rl.rewards.deepeyes_batch.JudgeGlobalFailure: DeepEyes transient "
        "judge failures exceed the bounded window; "
        "last_error=_JudgeTransientHTTPError: DeepEyes judge transient HTTP "
        "failure 429"
    ),
)


def checkpointed_step(checkpoint_root: Path) -> int:
    """Return only the trainer's committed checkpoint boundary."""

    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 0
    raw = tracker.read_text(encoding="utf-8").strip()
    try:
        step = int(raw)
    except ValueError as error:
        raise RuntimeError(
            f"malformed checkpoint tracker {tracker}: {raw!r}"
        ) from error
    if step < 0:
        raise RuntimeError(f"checkpoint tracker {tracker} is negative")
    return step


def is_weight_wake_oom(log_text: str) -> bool:
    """Recognize the one failure that a clean-process retry can recover."""

    return all(marker in log_text for marker in _WEIGHT_WAKE_OOM_MARKERS)


def is_judge_transient_429(log_text: str) -> bool:
    """Recognize only the judge's exhausted transient HTTP-429 failure."""

    return all(marker in log_text for marker in _JUDGE_TRANSIENT_429_MARKERS)


def supervisor_decision(
    *,
    return_code: int,
    checkpoint_step: int,
    target_step: int,
    attempt_log: str,
    completed_restarts: int,
    maximum_restarts: int,
) -> SupervisorDecision:
    if checkpoint_step >= target_step:
        return "complete"
    if (
        return_code != 0
        and completed_restarts < maximum_restarts
        and is_weight_wake_oom(attempt_log)
    ):
        return "retry_weight_wake_oom"
    if (
        return_code != 0
        and completed_restarts < maximum_restarts
        and is_judge_transient_429(attempt_log)
    ):
        return "retry_judge_transient_429"
    return "fail"


def _append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_attempt(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
            raise RuntimeError("supervised child has no stdout pipe")
        while chunk := process.stdout.read(64 * 1024):
            log_handle.write(chunk)
            log_handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()


def supervise(
    *,
    run_config: Path,
    target_step: int,
    log_directory: Path,
    maximum_restarts: int,
    cooldown_seconds: float,
) -> int:
    config = load_policy_e2e_smoke_run_config(run_config.resolve())
    checkpoint_root = config.output.root / "checkpoints"
    events_path = log_directory / "supervisor-events.jsonl"
    completed_restarts = 0
    attempt_index = 0

    while True:
        before = checkpointed_step(checkpoint_root)
        if before >= target_step:
            return 0
        attempt_index += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_log = log_directory / (
            f"attempt-{attempt_index:02d}-from-step-{before}-{timestamp}.log"
        )
        command = (
            sys.executable,
            "-m",
            "tgvf_rl.framework.verl.trainable_tgvf_launcher",
            "--run-config",
            str(run_config.resolve()),
            "--mode",
            "formal",
            "--target-step",
            str(target_step),
        )
        _append_event(
            events_path,
            {
                "event": "attempt_started",
                "attempt": attempt_index,
                "checkpoint_step": before,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "log_path": str(attempt_log),
            },
        )
        return_code = _run_attempt(command, attempt_log)
        after = checkpointed_step(checkpoint_root)
        log_text = attempt_log.read_text(encoding="utf-8", errors="replace")
        decision = supervisor_decision(
            return_code=return_code,
            checkpoint_step=after,
            target_step=target_step,
            attempt_log=log_text,
            completed_restarts=completed_restarts,
            maximum_restarts=maximum_restarts,
        )
        _append_event(
            events_path,
            {
                "event": "attempt_finished",
                "attempt": attempt_index,
                "checkpoint_step_before": before,
                "checkpoint_step_after": after,
                "return_code": return_code,
                "decision": decision,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "log_path": str(attempt_log),
            },
        )
        if decision == "complete":
            return 0
        if decision == "fail":
            return return_code if return_code != 0 else 3
        completed_restarts += 1
        time.sleep(cooldown_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--target-step", required=True, type=int)
    parser.add_argument("--log-directory", required=True, type=Path)
    parser.add_argument("--maximum-restarts", type=int, default=8)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.target_step <= 0:
        parser.error("--target-step must be positive")
    if args.maximum_restarts < 0:
        parser.error("--maximum-restarts must be non-negative")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    return supervise(
        run_config=args.run_config,
        target_step=args.target_step,
        log_directory=args.log_directory.resolve(),
        maximum_restarts=args.maximum_restarts,
        cooldown_seconds=args.cooldown_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "checkpointed_step",
    "is_judge_transient_429",
    "is_weight_wake_oom",
    "main",
    "supervise",
    "supervisor_decision",
]
