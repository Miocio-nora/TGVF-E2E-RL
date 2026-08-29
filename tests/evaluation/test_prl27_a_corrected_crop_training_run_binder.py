from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def binder() -> ModuleType:
    return _load_module(
        "prl27_a_corrected_crop_binder_under_test",
        TOOLS / "bind_prl27_a_corrected_crop_training_run_evaluation.py",
    )


@pytest.fixture(scope="module")
def paired_runner() -> ModuleType:
    return _load_module(
        "prl27_a_paired_runner_under_test",
        TOOLS / "run_prl15_paired_evaluation.py",
    )


def _source_run(binder: ModuleType) -> object:
    return load_policy_e2e_smoke_run_config(
        binder.CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )


def _plan_run(source: object, owner_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        schema_version=source.schema_version,
        source_sha256=source.source_sha256,
        run_id=source.run_id,
        identity_sha256=source.identity_sha256,
        code=source.code,
        model=source.model,
        policy=source.policy,
        protocol=source.protocol,
        rollout_rng=source.rollout_rng,
        distributed=source.distributed,
        training=source.training,
        output=SimpleNamespace(
            root=owner_root,
            metrics_path=owner_root / "metrics.jsonl",
        ),
    )


def _bound_plan(
    binder: ModuleType, tmp_path: Path
) -> tuple[dict[str, object], object, object, Path]:
    source = _source_run(binder)
    owner_root = tmp_path / "owner"
    completion = owner_root / (
        "permanent-checkpoints/global_step_32/"
        "tgvf_permanent_checkpoint_receipt.json"
    )
    completion.parent.mkdir(parents=True)
    completion.write_text('{"receipt":"prl27-a"}\n', encoding="utf-8")
    run = _plan_run(source, owner_root)
    materialization = load_deepeyes_native_run_contract(
        binder.MATERIALIZATION_CONFIG.resolve()
    )
    plan = binder._crop_plan(
        run=run,
        completion_path=completion,
        materialization_contract=materialization,
    )
    return plan, run, materialization, completion


def test_prl27_a_plan_binds_exact_corrected_crop_training_run_protocol(
    binder: ModuleType,
    paired_runner: ModuleType,
    tmp_path: Path,
) -> None:
    plan, run, _materialization, _completion = _bound_plan(binder, tmp_path)
    expected = paired_runner._training_run_crop_plan_protocol(run)

    assert plan["evaluation_id"] == binder.CROP_EVALUATION_ID
    assert plan["protocol"] == expected
    assert plan["protocol"]["evaluation_protocol"] == "training_run"
    identity = plan["protocol"]["training_run_identity"]
    assert identity == {
        "profile": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "tool_profile": "crop_only",
        "enabled_tool_names": ["image_zoom_in_tool"],
        "maximum_tool_calls": 6,
        "native_pixels": False,
        "assistant_dialect": "qwen3-vl-instruct-v1",
        "precomputed_image_embeds": True,
        "tool_parser": "strict_native_tool_call_parser_v1",
        "success_environment_renderer": (
            "render_qwen_native_matched_crop_success_environment_text"
        ),
        "success_environment_text_sha256": (
            "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
        ),
        "error_environment_renderer": (
            "qwen_native_standard_tool_error_canonical_json_v1"
        ),
        "cap_error_behavior": "one_final_answer_turn",
        "response_budget_scope": "total_response_tokens",
        "single_response_max_tokens": 10_240,
    }
    assert plan["protocol"]["action_boundary"] == {
        "stop_strings": ["</tool_call>"],
        "stop_token_ids": [151645],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
    }
    assert plan["paired_rng"]["seed_namespace"] == binder.PAIRED_SEED_NAMESPACE
    assert "prl27-a" in binder.PAIRED_SEED_NAMESPACE
    assert binder.CROP_EVALUATION_ID in str(binder.EVALUATION_ROOT)
    assert "PRL-27-A-train512" in str(binder.EVALUATION_ROOT)
    assert "no_tool" not in json.dumps(plan, sort_keys=True)


def test_prl27_a_protocol_hash_binds_complete_training_run_identity(
    binder: ModuleType,
    paired_runner: ModuleType,
    tmp_path: Path,
) -> None:
    plan, _run, _materialization, _completion = _bound_plan(binder, tmp_path)
    identity = plan["protocol"]["training_run_identity"]
    expected_sha256 = binder._canonical_sha256(identity)

    assert plan["paired_rng"]["protocol_sha256"] == expected_sha256
    assert expected_sha256 == paired_runner._canonical_json_sha256(identity)
    plan_path = tmp_path / "bound-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert paired_runner._load_plan(plan_path) == plan


@pytest.mark.parametrize(
    "mutation",
    ("renderer", "action_boundary", "protocol_sha256"),
)
def test_prl27_a_protocol_tampering_fails_closed(
    binder: ModuleType,
    paired_runner: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, run, materialization, _completion = _bound_plan(binder, tmp_path)
    tampered = deepcopy(plan)
    if mutation == "renderer":
        tampered["protocol"]["training_run_identity"][
            "success_environment_text_sha256"
        ] = "0" * 64
        # Rehash the forged identity to prove the owner contract, not merely a
        # stale digest, is the fail-closed authority.
        tampered["paired_rng"]["protocol_sha256"] = binder._canonical_sha256(
            tampered["protocol"]["training_run_identity"]
        )
        match = "owner differs"
    elif mutation == "action_boundary":
        tampered["protocol"]["action_boundary"][
            "include_stop_str_in_output"
        ] = False
        match = "owner differs"
    else:
        tampered["paired_rng"]["protocol_sha256"] = "0" * 64
        match = "paired RNG identity differs"

    with pytest.raises(RuntimeError, match=match):
        paired_runner._validate_v3_policy_run_runtime(
            tampered, run, materialization
        )


def test_prl27_a_completion_bytes_tampering_fails_closed(
    binder: ModuleType,
    paired_runner: ModuleType,
    tmp_path: Path,
) -> None:
    plan, _run, _materialization, completion = _bound_plan(binder, tmp_path)
    plan_path = tmp_path / "bound-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    paired_runner._load_plan(plan_path)

    completion.write_text('{"receipt":"tampered"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion identity differs"):
        paired_runner._load_plan(plan_path)


def test_prl27_a_owner_code_commit_tampering_fails_closed(
    binder: ModuleType,
) -> None:
    source = _source_run(binder)
    binder._validate_crop_owner_contract(source)
    tampered = SimpleNamespace(
        schema_version=source.schema_version,
        code=SimpleNamespace(commit="0" * 40),
        policy=source.policy,
        rollout_rng=source.rollout_rng,
        protocol=source.protocol,
        training=source.training,
        distributed=source.distributed,
        output=source.output,
    )

    with pytest.raises(RuntimeError, match="owner contract differs"):
        binder._validate_crop_owner_contract(tampered)
