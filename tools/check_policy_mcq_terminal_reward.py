"""Fail when an unambiguous terminal MCQ decision received zero reward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence


REPORT_SCHEMA = "policy-mcq-terminal-reward-confusion-gate-v1"
AUDIT_SCHEMA = "policy-trajectory-audit-v1"
_TERMINAL_MARKERS = re.compile(r"(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)+$")
_MARKDOWN = re.compile(r"[*_`]")
_TERMINAL_DECISION = re.compile(
    r"^\s*(?:[\(\[]\s*([A-H])\s*[\)\]]|([A-H])\s*[.:]|([A-H])\s*$)",
    re.IGNORECASE,
)
_ANSWER_WRAPPER = re.compile(r"^<answer>\s*(.*?)\s*</answer>$", re.DOTALL)
_BOXED_WRAPPER = re.compile(r"^\\boxed\s*\{(.*)\}$", re.DOTALL)


def terminal_mcq_letter(value: object) -> str | None:
    """Return a strict A-H decision from the final non-empty line only."""

    if not isinstance(value, str):
        return None
    normalized = _TERMINAL_MARKERS.sub("", value.strip()).strip()
    for wrapper in (_ANSWER_WRAPPER, _BOXED_WRAPPER):
        match = wrapper.fullmatch(normalized)
        if match is not None:
            normalized = match.group(1).strip()
    normalized = _MARKDOWN.sub("", normalized)
    final_line = next(
        (line.strip() for line in reversed(normalized.splitlines()) if line.strip()),
        "",
    )
    match = _TERMINAL_DECISION.match(final_line)
    if match is None:
        return None
    return next(group.upper() for group in match.groups() if group is not None)


def _binary_reward(components: dict[str, Any], name: str, path: Path) -> float:
    value = components.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) not in {0.0, 1.0}
    ):
        raise ValueError(f"{path}: {name} must be binary")
    return float(value)


def _load_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: audit payload must be an object")
    if payload.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError(f"{path}: unexpected trajectory audit schema")
    return payload


def audit_run(
    run_output: Path,
    *,
    expected_representative_records: int | None = None,
    expected_optimizer_step: int | None = None,
    minimum_terminal_matches_gold: int = 0,
) -> dict[str, Any]:
    """Audit the retained trajectory subset and return a deterministic report."""

    if expected_representative_records is not None and (
        expected_representative_records <= 0
    ):
        raise ValueError("expected representative records must be positive")
    if expected_optimizer_step is not None and expected_optimizer_step < 0:
        raise ValueError("expected optimizer step must be non-negative")
    if minimum_terminal_matches_gold < 0:
        raise ValueError("minimum terminal matches gold must be non-negative")

    audit_root = run_output / "trajectory_audit"
    if not audit_root.is_dir():
        raise ValueError(f"trajectory audit directory does not exist: {audit_root}")
    paths = sorted(audit_root.glob("step-*/*.json"))
    if not paths:
        raise ValueError(f"trajectory audit contains no records: {audit_root}")

    representative_count = 0
    representative_parse_failed_count = 0
    letter_parse_failed_count = 0
    terminal_parseable_count = 0
    gold_parseable_count = 0
    terminal_matches_gold_count = 0
    answer_false_negative_count = 0
    conditional_false_negative_count = 0
    false_negatives: list[dict[str, Any]] = []
    optimizer_steps: set[int] = set()

    for path in paths:
        payload = _load_record(path)
        reasons = payload.get("selection_reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons
        ):
            raise ValueError(f"{path}: selection_reasons must be a list of strings")
        evidence = payload.get("answer_verifier_evidence")
        if not isinstance(evidence, str):
            raise ValueError(f"{path}: answer_verifier_evidence must be text")
        components = payload.get("reward_components")
        if not isinstance(components, dict):
            raise ValueError(f"{path}: reward_components must be an object")
        answer_reward = _binary_reward(components, "answer_reward", path)
        conditional_reward = _binary_reward(components, "conditional_tool_reward", path)
        observation_count = payload.get("successful_observation_count")
        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
            or observation_count < 0
        ):
            raise ValueError(
                f"{path}: successful_observation_count must be a non-negative integer"
            )
        optimizer_step = payload.get("optimizer_step")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
        ):
            raise ValueError(f"{path}: optimizer_step must be a non-negative integer")
        optimizer_steps.add(optimizer_step)
        expected_step_directory = f"step-{optimizer_step:08d}"
        if path.parent.name != expected_step_directory:
            raise ValueError(
                f"{path}: parent directory must be {expected_step_directory}"
            )

        is_representative = "representative_rollout_zero" in reasons
        parse_failed = "letter_parse_failed" in evidence
        representative_count += int(is_representative)
        letter_parse_failed_count += int(parse_failed)
        representative_parse_failed_count += int(is_representative and parse_failed)

        candidate_letter = terminal_mcq_letter(payload.get("candidate_answer"))
        gold_letter = terminal_mcq_letter(payload.get("expected_answer"))
        terminal_parseable_count += int(candidate_letter is not None)
        gold_parseable_count += int(gold_letter is not None)
        if candidate_letter is None or candidate_letter != gold_letter:
            continue
        terminal_matches_gold_count += 1

        failed_rewards: list[str] = []
        if answer_reward != 1.0:
            failed_rewards.append("answer_reward")
            answer_false_negative_count += 1
        if observation_count > 0 and conditional_reward != 1.0:
            failed_rewards.append("conditional_tool_reward")
            conditional_false_negative_count += 1
        if not failed_rewards:
            continue

        candidate = payload.get("candidate_answer")
        assert isinstance(candidate, str)
        normalized = _TERMINAL_MARKERS.sub("", candidate.strip()).strip()
        final_line = next(
            (
                line.strip()
                for line in reversed(normalized.splitlines())
                if line.strip()
            ),
            "",
        )
        false_negatives.append(
            {
                "path": path.relative_to(run_output).as_posix(),
                "optimizer_step": payload.get("optimizer_step"),
                "trajectory_id": payload.get("trajectory_id"),
                "candidate_letter": candidate_letter,
                "gold_letter": gold_letter,
                "terminal_line": final_line,
                "answer_reward": answer_reward,
                "conditional_tool_reward": conditional_reward,
                "successful_observation_count": observation_count,
                "answer_verifier_evidence": evidence,
                "representative_rollout_zero": is_representative,
                "failed_rewards": failed_rewards,
            }
        )

    clear_false_negative_count = len(false_negatives)
    acceptance_failures: list[str] = []
    if (
        expected_representative_records is not None
        and representative_count != expected_representative_records
    ):
        acceptance_failures.append(
            "representative_records="
            f"{representative_count}, expected={expected_representative_records}"
        )
    if expected_optimizer_step is not None and optimizer_steps != {
        expected_optimizer_step
    }:
        acceptance_failures.append(
            f"optimizer_steps={sorted(optimizer_steps)}, "
            f"expected=[{expected_optimizer_step}]"
        )
    if terminal_matches_gold_count < minimum_terminal_matches_gold:
        acceptance_failures.append(
            "terminal_matches_gold_records="
            f"{terminal_matches_gold_count}, minimum={minimum_terminal_matches_gold}"
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "failed" if clear_false_negative_count or acceptance_failures else "passed"
        ),
        "run_output": str(run_output.resolve()),
        "trajectory_audit_root": str(audit_root.resolve()),
        "coverage": (
            "retained audit subset: rollout-index zero plus trajectories selected "
            "for correctness, format error, or max-token diagnostics"
        ),
        "counts": {
            "audit_records": len(paths),
            "representative_records": representative_count,
            "letter_parse_failed_records": letter_parse_failed_count,
            "representative_letter_parse_failed_records": (
                representative_parse_failed_count
            ),
            "terminal_letter_parseable_records": terminal_parseable_count,
            "gold_letter_parseable_records": gold_parseable_count,
            "terminal_matches_gold_records": terminal_matches_gold_count,
            "answer_reward_false_negative_records": answer_false_negative_count,
            "conditional_tool_reward_false_negative_records": (
                conditional_false_negative_count
            ),
            "clear_false_negative_records": clear_false_negative_count,
        },
        "representative": {
            "record_count": representative_count,
            "letter_parse_failed_count": representative_parse_failed_count,
            "letter_parse_failed_fraction": (
                representative_parse_failed_count / representative_count
                if representative_count
                else None
            ),
        },
        "observed_optimizer_steps": sorted(optimizer_steps),
        "expectations": {
            "representative_records": expected_representative_records,
            "optimizer_step": expected_optimizer_step,
            "minimum_terminal_matches_gold": minimum_terminal_matches_gold,
        },
        "acceptance_failures": acceptance_failures,
        "false_negatives": false_negatives,
    }
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-representative-records", type=int)
    parser.add_argument("--expected-optimizer-step", type=int)
    parser.add_argument("--minimum-terminal-matches-gold", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_run(
            args.run_output,
            expected_representative_records=args.expected_representative_records,
            expected_optimizer_step=args.expected_optimizer_step,
            minimum_terminal_matches_gold=args.minimum_terminal_matches_gold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "error",
            "run_output": str(args.run_output.resolve()),
            "error": str(error),
        }
        _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2

    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
