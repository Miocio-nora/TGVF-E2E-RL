from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tgvf_rl.environment.native_appender import (
    render_qwen_native_matched_tgvf_success_environment_text,
)
from tgvf_rl.evaluation.policy_coredev import _success_environment_text_renderer
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_PLAN = (
    _ROOT
    / "configs/evaluation/"
    "prl16_f1_frozen_rp66_step0_step8_coredev2511_plan.json"
)
_RUN = (
    _ROOT
    / "configs/policy/runs/"
    "prl_16_f1_qwen3_instruct_full_frozen_rp66_bs16_n16_t1_crop16_exact_matched_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SUPERVISOR = _ROOT / "tools/supervise_prl16_f1_evaluation.sh"
_SPEC = importlib.util.spec_from_file_location("prl16_f1_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_prl16_f1_plan_binds_frozen_run_and_official_scoring() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )

    _MODULE._validate_plan_run(plan, run)

    required = plan["required_pairing"]
    assert run.representation.adapter_update_mode.value == "frozen_adapter"
    assert required["adapter_update_mode"] == "frozen_adapter"
    assert required["rp66_state_must_remain_constant"] is True
    assert required["expected_runtime_rp66_weights_sha256"] == (
        "05778a43844f397e0ad898ffbb060cf37a71ce174768437fbe8e782adf820318"
    )
    assert plan["scoring"]["datasets"] == list(_MODULE.COREDEV_DATASETS)
    assert _success_environment_text_renderer(run) is (
        render_qwen_native_matched_tgvf_success_environment_text
    )


def test_frozen_pairing_accepts_equal_state_and_rejects_drift(monkeypatch) -> None:
    plan = _MODULE._load_plan(_PLAN)
    step0 = Path("step0.json")
    step8 = Path("step8.json")
    storage = plan["required_pairing"]["expected_runtime_rp66_weights_sha256"]
    receipts = {
        step0: SimpleNamespace(
            rp66_kind="stage1_artifact",
            rp66_state_sha256="a" * 64,
            rp66_storage_sha256="unused",
        ),
        step8: SimpleNamespace(
            rp66_kind="runtime_snapshot",
            rp66_state_sha256="a" * 64,
            rp66_storage_sha256=storage,
        ),
    }
    monkeypatch.setattr(_MODULE, "load_policy_coredev_config", lambda path: path)
    monkeypatch.setattr(
        _MODULE,
        "load_policy_evaluation_snapshot",
        lambda path: SimpleNamespace(receipt=receipts[path]),
    )

    _MODULE._validate_materialized_frozen_pairing(plan, step0, step8)
    receipts[step8].rp66_state_sha256 = "b" * 64
    with pytest.raises(RuntimeError, match="state changed"):
        _MODULE._validate_materialized_frozen_pairing(plan, step0, step8)


def test_completion_marker_atomically_binds_paired_summary(tmp_path: Path) -> None:
    report = tmp_path / "paired-summary.json"
    report.write_text('{"result":"pass"}\n', encoding="utf-8")

    record = _MODULE._write_evaluation_complete(
        tmp_path, report, evaluation_id="PRL16-F1-EVAL"
    )

    assert record["paired_summary_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    assert json.loads((tmp_path / "evaluation-complete").read_text()) == record


def test_supervisor_waits_retries_and_fails_fast_on_contract_errors() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert "--wait-for-step8" in source
    assert "--wait-for-gpus" in source
    assert "PRL16_F1_EVAL_MAX_RESTARTS" in source
    assert "deterministic evaluation contract failure; refusing retry" in source
    assert "identity differs" in source
    assert "SHA256 differs" in source
