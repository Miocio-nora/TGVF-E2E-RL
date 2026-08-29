from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from tgvf_rl.environment.native_appender import (
    QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT,
    QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256,
)
from tgvf_rl.evaluation.policy_coredev import (
    PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT,
    PRL26_B_GENERIC_CROP_TRAINING_LAUNCH_COMMIT,
    PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT,
    training_run_evaluation_protocol_identity,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUPERVISOR = TOOLS / "supervise_prl26_b_generic86_s32_evaluation.sh"
LAUNCHER = TOOLS / "launch_prl26_b_generic86_s32_evaluation_tmux.sh"
EXPECTED_ENVIRONMENT_SHA256 = (
    "72a2caecb47a2b775a4497e5846c244061d9455fbb4b9690d3501cbc2521e187"
)


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
        "prl26_b_generic86_binder_under_test",
        TOOLS / "bind_prl26_b_generic86_training_run_evaluation.py",
    )


@pytest.fixture(scope="module")
def paired_runner() -> ModuleType:
    return _load_module(
        "prl26_b_generic86_paired_runner_under_test",
        TOOLS / "run_prl15_paired_evaluation.py",
    )


def test_prl26_b_generic86_plan_is_accepted_by_v3_static_and_runtime_routes(
    binder: ModuleType,
    paired_runner: ModuleType,
) -> None:
    run = load_policy_e2e_smoke_run_config(
        binder.CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    completion = run.output.root / (
        "permanent-checkpoints/global_step_32/"
        "tgvf_permanent_checkpoint_receipt.json"
    )
    protocol = load_deepeyes_native_run_contract(
        binder.MATERIALIZATION_CONFIG.resolve()
    )

    plan = binder._plan(
        run=run,
        completion_path=completion,
        materialization_contract=protocol,
    )
    paired_runner._validate_v3_static_plan(plan)
    paired_runner._validate_v3_policy_run_runtime(plan, run, protocol)

    identity = plan["protocol"]["training_run_identity"]
    assert plan["protocol"]["training_run_variant"] == (
        PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT
    )
    assert identity["success_environment_renderer"] == (
        "render_qwen_native_success_environment_text"
    )
    assert identity["success_environment_text_sha256"] == (
        EXPECTED_ENVIRONMENT_SHA256
    )
    assert identity["success_environment_token_count"] == (
        PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT
    )
    assert identity["training_launch_project_commit"] == (
        PRL26_B_GENERIC_CROP_TRAINING_LAUNCH_COMMIT
    )
    assert plan["protocol"]["action_boundary"] == {
        "stop_strings": ["</tool_call>"],
        "stop_token_ids": [151645],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
    }


def test_prl26_b_generic86_variant_is_owner_locked_and_byte_locked(
    binder: ModuleType,
) -> None:
    run = load_policy_e2e_smoke_run_config(
        binder.CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    identity = training_run_evaluation_protocol_identity(
        run,
        full_model=True,
        precomputed_image_embeds=True,
        training_run_variant=PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT,
    )
    assert QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT_SHA256 == (
        EXPECTED_ENVIRONMENT_SHA256
    )
    assert binder.EXPECTED_GENERIC86_ENVIRONMENT_TEXT_SHA256 == (
        EXPECTED_ENVIRONMENT_SHA256
    )
    assert identity["success_environment_text_sha256"] == (
        EXPECTED_ENVIRONMENT_SHA256
    )

    wrong_run = replace(run, run_id="not-prl26-b")
    with pytest.raises(ValueError, match="exact PRL-26-B"):
        training_run_evaluation_protocol_identity(
            wrong_run,
            full_model=True,
            precomputed_image_embeds=True,
            training_run_variant=PRL26_B_GENERIC_CROP_TRAINING_RUN_VARIANT,
        )


def test_real_qwen_tokenizer_keeps_generic_continuation_at_86_tokens(
    binder: ModuleType,
) -> None:
    run = load_policy_e2e_smoke_run_config(
        binder.CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    tokenizer_path = Path(run.model.revision_or_path)
    if not tokenizer_path.is_dir():
        pytest.skip("pinned local Qwen tokenizer is unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    token_ids = tokenizer.encode(
        QWEN_NATIVE_GENERIC_CROP_SUCCESS_TEXT,
        add_special_tokens=False,
    )
    assert len(token_ids) == PRL26_B_GENERIC_CROP_ENVIRONMENT_TOKEN_COUNT == 86
    assert token_ids.count(tokenizer.convert_tokens_to_ids("<|image_pad|>")) == 1


def test_generic86_supervisor_is_executable_and_orders_fail_closed_route() -> None:
    for script in (SUPERVISOR, LAUNCHER):
        assert script.is_file()
        assert script.stat().st_mode & 0o111
        subprocess.run(["bash", "-n", str(script)], check=True)

    source = SUPERVISOR.read_text(encoding="utf-8")
    release = source.index("while (( quiet < release_stable_polls ))")
    bind = source.index('"$python_bin" "$binder"', release)
    reuse = source.index('"$python_bin" "$reuser"', bind)
    prepare = source.index("--mode prepare", reuse)
    validate = source.index("--mode validate --world-size 4", prepare)
    proof = source.index("--arm crop", validate)
    infer = source.index("--mode infer", proof)
    score = source.index("--mode score", infer)
    summarize = source.index('"$python_bin" "$summarizer"', score)
    complete = source.index('touch "$supervisor_complete"', summarize)
    assert release < bind < reuse < prepare < validate < proof < infer < score
    assert score < summarize < complete
    assert "release_stable_polls < 3" in source
    assert "export VLLM_PLUGINS=tgvf_qwen3_precomputed" in source
    assert "export VLLM_PLUGINS=" not in source.replace(
        "export VLLM_PLUGINS=tgvf_qwen3_precomputed", ""
    )
    assert "read_only_existing_hf_tree_no_merge" in (
        TOOLS / "reuse_prl26_b_s32_full_model_for_generic86_eval.py"
    ).read_text(encoding="utf-8")
