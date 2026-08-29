from __future__ import annotations

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
        "prl27_b_corrected_crop_binder_under_test",
        TOOLS / "bind_prl27_b_corrected_crop_training_run_evaluation.py",
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


def test_prl27_b_binder_owns_a_new_run_root_evaluation_and_rng(
    binder: ModuleType,
) -> None:
    assert binder.CROP_CONFIG == ROOT / (
        "configs/policy/runs/"
        "prl_27_b_qwen3_instruct_full_crop_train512_replay_byte_parity_"
        "s32_bs16_n16_teacher25_ws8.toml"
    )
    assert binder.CROP_EVALUATION_ID.startswith("PRL27-B-")
    assert "REPLAY-BYTE-PARITY" in binder.CROP_EVALUATION_ID
    assert "PRL-27-B-train512" in str(binder.CROP_OWNER_ROOT)
    assert binder.EVALUATION_ROOT.parent == binder.CROP_OWNER_ROOT / "evaluation"
    assert "prl27-b" in binder.PAIRED_SEED_NAMESPACE
    assert "prl27-a" not in binder.PAIRED_SEED_NAMESPACE
    assert binder.EXPECTED_CODE_COMMIT == ("c448e583887e4e49b79fe52fefb4b42934cd787e")


def test_prl27_b_plan_is_single_arm_exact_training_run_with_independent_rng(
    binder: ModuleType,
    tmp_path: Path,
) -> None:
    source = _source_run(binder)
    binder._validate_crop_owner_contract(source)
    owner_root = tmp_path / "owner"
    completion = owner_root / (
        "permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
    )
    completion.parent.mkdir(parents=True)
    completion.write_text('{"receipt":"prl27-b"}\n', encoding="utf-8")
    run = _plan_run(source, owner_root)
    materialization = load_deepeyes_native_run_contract(
        binder.MATERIALIZATION_CONFIG.resolve()
    )

    plan = binder._crop_plan(
        run=run,
        completion_path=completion,
        materialization_contract=materialization,
    )

    assert plan["evaluation_id"] == binder.CROP_EVALUATION_ID
    assert plan["checkpoint_owner"]["run_id"] == source.run_id
    assert plan["arms"] == [
        {
            "name": "step32",
            "optimizer_step": 32,
            "evaluation_id": binder.CROP_EVALUATION_ID,
            "source": {
                "kind": "owner_checkpoint",
                "relative_path": "permanent-checkpoints/global_step_32",
            },
        }
    ]
    identity = plan["protocol"]["training_run_identity"]
    assert plan["protocol"]["evaluation_protocol"] == "training_run"
    assert identity["tool_profile"] == "crop_only"
    assert identity["enabled_tool_names"] == ["image_zoom_in_tool"]
    assert identity["success_environment_renderer"] == (
        "render_qwen_native_matched_crop_success_environment_text"
    )
    assert plan["protocol"]["action_boundary"] == {
        "stop_strings": ["</tool_call>"],
        "stop_token_ids": [151645],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
    }
    assert plan["paired_rng"]["seed_namespace"] == binder.PAIRED_SEED_NAMESPACE
    assert plan["paired_rng"]["protocol_sha256"] == binder._canonical_sha256(identity)
    assert plan["scoring"]["run_id_prefix"].startswith("T20260830-PRL27-B-")
    assert "prl27-a" not in json.dumps(plan, sort_keys=True).lower()


def test_prl27_b_owner_rejects_an_a_or_unpinned_commit(
    binder: ModuleType,
) -> None:
    source = _source_run(binder)
    tampered = SimpleNamespace(
        schema_version=source.schema_version,
        code=SimpleNamespace(commit="ecddc379d392d154c91783d7651528b20d40afba"),
        policy=source.policy,
        rollout_rng=source.rollout_rng,
        protocol=source.protocol,
        training=source.training,
        distributed=source.distributed,
        output=source.output,
    )
    with pytest.raises(RuntimeError, match="owner contract differs"):
        binder._validate_crop_owner_contract(tampered)
