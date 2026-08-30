from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "check_policy_mcq_terminal_reward.py"
)
_SPEC = importlib.util.spec_from_file_location("mcq_terminal_reward_gate", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def _write_audit(
    run_output: Path,
    name: str,
    *,
    candidate: str,
    expected: str = "B",
    answer_reward: float = 0.0,
    conditional_tool_reward: float = 0.0,
    observations: int = 0,
    representative: bool = True,
    optimizer_step: int = 0,
    evidence: str = "route=multiple_choice_rule; letter_parse_failed",
) -> None:
    path = (
        run_output / "trajectory_audit" / f"step-{optimizer_step:08d}" / f"{name}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "policy-trajectory-audit-v1",
        "selection_reasons": ["representative_rollout_zero"]
        if representative
        else ["format_error"],
        "optimizer_step": optimizer_step,
        "trajectory_id": f"run/sample/{name}",
        "candidate_answer": candidate,
        "expected_answer": expected,
        "reward_components": {
            "answer_reward": answer_reward,
            "conditional_tool_reward": conditional_tool_reward,
        },
        "answer_verifier_evidence": evidence,
        "successful_observation_count": observations,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_passes_and_reports_representative_parse_failure_rate(
    tmp_path: Path,
) -> None:
    run_output = tmp_path / "run"
    _write_audit(
        run_output,
        "correct",
        candidate="Long reasoning.\n\n**B. option text**<|im_end|>",
        answer_reward=1.0,
        conditional_tool_reward=1.0,
        observations=2,
        evidence="route=multiple_choice_rule; candidate=B; expected=B",
    )
    _write_audit(
        run_output,
        "undecidable",
        candidate="B. intermediate branch\nBased on the evidence, no decision follows.",
    )
    _write_audit(
        run_output,
        "wrong",
        candidate="Reasoning.\nA.<|endoftext|>",
        representative=False,
    )
    output = tmp_path / "report.json"

    exit_code = gate.main([str(run_output), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["counts"] == {
        "answer_reward_false_negative_records": 0,
        "audit_records": 3,
        "clear_false_negative_records": 0,
        "conditional_tool_reward_false_negative_records": 0,
        "gold_letter_parseable_records": 3,
        "letter_parse_failed_records": 2,
        "representative_letter_parse_failed_records": 1,
        "representative_records": 2,
        "terminal_letter_parseable_records": 2,
        "terminal_matches_gold_records": 1,
    }
    assert report["representative"] == {
        "letter_parse_failed_count": 1,
        "letter_parse_failed_fraction": 0.5,
        "record_count": 2,
    }
    assert report["observed_optimizer_steps"] == [0]
    assert report["expectations"] == {
        "minimum_terminal_matches_gold": 0,
        "optimizer_step": None,
        "representative_records": None,
    }
    assert report["acceptance_failures"] == []
    assert report["false_negatives"] == []


@pytest.mark.parametrize(
    ("answer_reward", "observations", "conditional_reward", "failed_rewards"),
    (
        (0.0, 0, 0.0, ["answer_reward"]),
        (1.0, 2, 0.0, ["conditional_tool_reward"]),
        (0.0, 2, 0.0, ["answer_reward", "conditional_tool_reward"]),
    ),
)
def test_gate_fails_on_unambiguous_terminal_reward_false_negative(
    tmp_path: Path,
    answer_reward: float,
    observations: int,
    conditional_reward: float,
    failed_rewards: list[str],
) -> None:
    run_output = tmp_path / "run"
    _write_audit(
        run_output,
        "false-negative",
        candidate="The visual evidence supports the second choice.\n\n`B.` choice text\n<|im_end|>",
        answer_reward=answer_reward,
        conditional_tool_reward=conditional_reward,
        observations=observations,
    )
    output = tmp_path / "report.json"

    exit_code = gate.main([str(run_output), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["counts"]["clear_false_negative_records"] == 1
    assert report["false_negatives"][0]["candidate_letter"] == "B"
    assert report["false_negatives"][0]["gold_letter"] == "B"
    assert report["false_negatives"][0]["failed_rewards"] == failed_rewards


def test_gate_returns_error_and_writes_report_for_missing_audit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"

    exit_code = gate.main([str(tmp_path / "missing"), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "error"
    assert "does not exist" in report["error"]


@pytest.mark.parametrize(
    ("extra_args", "failure"),
    (
        (
            ["--expected-representative-records", "2"],
            "representative_records=1, expected=2",
        ),
        (
            ["--expected-optimizer-step", "0"],
            "optimizer_steps=[1], expected=[0]",
        ),
        (
            ["--minimum-terminal-matches-gold", "1"],
            "terminal_matches_gold_records=0, minimum=1",
        ),
    ),
)
def test_gate_fails_closed_when_smoke_expectations_are_not_met(
    tmp_path: Path,
    extra_args: list[str],
    failure: str,
) -> None:
    run_output = tmp_path / "run"
    candidate = "Reasoning remains inconclusive."
    optimizer_step = 1 if "--expected-optimizer-step" in extra_args else 0
    _write_audit(
        run_output,
        "record",
        candidate=candidate,
        optimizer_step=optimizer_step,
    )
    output = tmp_path / "report.json"

    exit_code = gate.main([str(run_output), "--output", str(output), *extra_args])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["acceptance_failures"] == [failure]


def test_gate_accepts_complete_one_step_smoke_expectations(tmp_path: Path) -> None:
    run_output = tmp_path / "run"
    _write_audit(
        run_output,
        "correct",
        candidate="Reasoning.\nB.",
        answer_reward=1.0,
        conditional_tool_reward=1.0,
        observations=1,
    )
    output = tmp_path / "report.json"

    exit_code = gate.main(
        [
            str(run_output),
            "--output",
            str(output),
            "--expected-representative-records",
            "1",
            "--expected-optimizer-step",
            "0",
            "--minimum-terminal-matches-gold",
            "1",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["acceptance_failures"] == []


@pytest.mark.parametrize("candidate", ("<answer>B</answer>", r"\boxed{B}"))
def test_independent_gate_parser_accepts_bound_wrappers(candidate: str) -> None:
    assert gate.terminal_mcq_letter(candidate) == "B"


def test_immutable_report_rejects_different_existing_content(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable reward-gate report differs"):
        gate._write_report(output, {"status": "passed"})

    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


def test_immutable_report_accepts_only_byte_identical_retry(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = {"schema_version": gate.REPORT_SCHEMA, "status": "passed"}

    gate._write_report(output, report)
    first = output.read_bytes()
    gate._write_report(output, report)

    assert output.read_bytes() == first
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


def test_immutable_report_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("target\n", encoding="utf-8")
    output = tmp_path / "report.json"
    output.symlink_to(target)

    with pytest.raises(RuntimeError, match="readable regular file"):
        gate._write_report(output, {"status": "passed"})

    assert target.read_text(encoding="utf-8") == "target\n"


def test_report_publication_failure_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(gate.os, "link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        gate._write_report(output, {"status": "passed"})

    assert list(tmp_path.iterdir()) == []


def test_gate_rejects_optimizer_step_directory_mismatch(tmp_path: Path) -> None:
    run_output = tmp_path / "run"
    _write_audit(run_output, "record", candidate="B", optimizer_step=0)
    source = next((run_output / "trajectory_audit").glob("step-*/*.json"))
    wrong_parent = run_output / "trajectory_audit" / "step-00000001"
    wrong_parent.mkdir()
    os.replace(source, wrong_parent / source.name)
    output = tmp_path / "report.json"

    exit_code = gate.main([str(run_output), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "error"
    assert "parent directory must be step-00000000" in report["error"]
