from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPOSITORY_ROOT / "spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py"
REWARD_PATH = REPOSITORY_ROOT / "spikes/verl_compat/verl_sync_fixed_reward.py"
RUN_ID = "SC-21-T211-VERL-VLLM-SYNC-TEST"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_record(step: int, *, diff_scale: float = 1.0) -> dict[str, Any]:
    return {
        "step": step,
        "data": {
            "training/rollout_probs_diff_valid": 1,
            "training/rollout_probs_diff_max": 0.01 * diff_scale,
            "training/rollout_probs_diff_mean": 0.001 * diff_scale,
            "training/rollout_probs_diff_std": 0.0005 * diff_scale,
            "training/rollout_actor_probs_pearson_corr": 0.999,
            "training/off_policy/trajectory_spans/mean": 1,
            "training/off_policy/trajectory_staleness/mean": 0,
            "training/off_policy/trajectory_staleness_worst/mean": 0,
            "response_length/min": 16,
            "response_length/max": 16,
            "actor/grad_norm": 0.75,
            "actor/lr": 1.0e-3,
            "actor/sync_gate_zero_advantage_valid": 1,
            "timing_s/update_weights": 0.25,
        },
    }


def test_plan_is_one_upstream_v1_no_sleep_integration_path() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_plan_test")
    result = smoke.bounded_result_path(
        Path("artifacts/compatibility/proposed-verl-sync-gate.json"),
        require_new=False,
    )
    paths = smoke.derive_paths(result)
    plan = smoke.plan_payload(
        run_id=RUN_ID,
        python=REPOSITORY_ROOT / ".venv-torch211-cu129/bin/python",
        model_path=smoke.ACCEPTED_MODEL_PATH,
        paths=paths,
        timeout_seconds=1800,
    )
    overrides = plan["command"][3:]

    assert plan["run_id"] == RUN_ID
    assert plan["scope"]["manager"] == ("verl.trainer.ppo.v1.AgentLoopManagerTQ")
    assert "generate_sequences(TensorDict)->None" in plan["scope"]["manager_contract"]
    assert "trainer.use_v1=true" in overrides
    assert "trainer.v1.trainer_mode=sync" in overrides
    assert "actor_rollout_ref.actor.strategy=fsdp2" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2" in overrides
    assert "actor_rollout_ref.rollout.name=vllm" in overrides
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in overrides
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=naive" in overrides
    assert "actor_rollout_ref.rollout.free_cache_engine=false" in overrides
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=false" in overrides
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in overrides
    assert "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs" in overrides
    assert (
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class="
        "tgvf_rl.framework.verl.sync_gate_manager.SyncGateAgentLoopManagerTQ"
        in overrides
    )
    assert "trainer.total_training_steps=2" in overrides
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in plan["environment"]
    assert plan["environment"]["VLLM_PLUGINS"] == (
        "__tgvf_native_sync_gate_no_plugins__"
    )
    assert plan["expected_runtime"]["distributions"]["nvidia-nccl-cu12"] == ("2.28.9")


def test_candidate_python_symlink_is_not_resolved_out_of_virtualenv() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_python_path_test")
    candidate_python = REPOSITORY_ROOT / ".venv-torch211-cu129/bin/python"

    assert smoke.absolute_executable(candidate_python) == candidate_python
    assert smoke.absolute_executable(candidate_python) != candidate_python.resolve()


def test_child_environment_allows_ray_to_assign_logical_cuda_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_ray_mapping_test")
    paths = smoke.derive_paths(
        smoke.bounded_result_path(
            Path("artifacts/compatibility/proposed-ray-mapping-gate.json"),
            require_new=False,
        )
    )
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")

    environment = smoke.child_environment(paths)

    assert environment["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in environment


def test_fixture_forces_greedy_rows_and_non_rl_zero_reward() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_fixture_test")
    reward = _load(REWARD_PATH, "tgvf_verl_sync_reward_test")
    rows = smoke.fixture_rows(run_id=RUN_ID)

    assert len(rows) == 2
    assert all(row["run_id"] == RUN_ID for row in rows)
    assert [row["extra_info"]["index"] for row in rows] == [0, 1]
    assert all(row["extra_info"]["run_id"] == RUN_ID for row in rows)
    assert all(row["__do_sample__"] is False for row in rows)
    assert all(row["reward_model"]["ground_truth"] == 0.0 for row in rows)
    assert len(smoke.fixture_logical_sha256(run_id=RUN_ID)) == 64
    assert reward.compute_score("tgvf_verl_vllm_sync_gate", "ignored", 0.0) == 0.0
    with pytest.raises(ValueError, match="data source"):
        reward.compute_score("wrong", "ignored", 0.0)


def test_metric_gate_proves_post_update_version_one_generation() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_metric_test")
    result = smoke.evaluate_metrics(
        [_metric_record(1), _metric_record(2, diff_scale=2.0)]
    )

    assert result["passed"] is True
    assert result["inferred_rollout_weight_versions"] == [0, 1]
    assert all(result["checks"].values())


def test_metric_gate_fails_closed_on_stale_or_numerically_unsynced_rollout() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_fail_test")
    stale = [_metric_record(1), _metric_record(2)]
    stale[1]["data"]["training/off_policy/trajectory_staleness/mean"] = 1
    assert smoke.evaluate_metrics(stale)["passed"] is False

    unsynced = [_metric_record(1), _metric_record(2, diff_scale=15.0)]
    result = smoke.evaluate_metrics(unsynced)
    assert result["passed"] is False
    assert result["checks"]["post_update_no_regression"] is False

    zero_grad = [_metric_record(1), _metric_record(2)]
    zero_grad[0]["data"]["actor/grad_norm"] = 0.0
    result = smoke.evaluate_metrics(zero_grad)
    assert result["passed"] is False
    assert result["checks"]["real_actor_updates"] is False


def test_default_cli_only_prints_plan_and_writes_nothing(capsys) -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_cli_test")
    proposed = Path("artifacts/compatibility/proposed-no-launch-sync-gate.json")
    resolved = REPOSITORY_ROOT / proposed

    assert smoke.main(["--run-id", RUN_ID, "--output", str(proposed)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == smoke.PLAN_SCHEMA_VERSION
    assert payload["run_id"] == RUN_ID
    assert payload["physical_gpus"] == [2, 3]
    assert not resolved.exists()


def test_cli_requires_a_safe_explicit_run_id() -> None:
    smoke = _load(SMOKE_PATH, "tgvf_verl_sync_smoke_run_id_test")
    output = "artifacts/compatibility/proposed-run-id-gate.json"
    with pytest.raises(SystemExit):
        smoke._parse_args(["--output", output])
    with pytest.raises(SystemExit):
        smoke._parse_args(["--run-id", "../unsafe", "--output", output])


def test_infrastructure_objective_has_zero_advantage_and_real_nll_gradient() -> None:
    pytest.importorskip("verl")
    from tgvf_rl.framework.verl.sync_gate_objective import (
        generated_token_nll,
        zero_advantage,
    )

    rewards = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    advantages, returns = zero_advantage(rewards, mask)
    assert torch.equal(advantages, torch.zeros_like(rewards))
    assert torch.equal(returns, torch.zeros_like(rewards))

    log_prob = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True)
    loss, metrics = generated_token_nll(
        old_log_prob=log_prob.detach().clone(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=SimpleNamespace(global_batch_info={}),
    )
    loss.backward()

    assert loss.item() == pytest.approx(2.0)
    assert torch.count_nonzero(log_prob.grad).item() == 3
    assert metrics["sync_gate_zero_advantage_valid"].item() == 1

    with pytest.raises(ValueError, match="sentinel was overwritten"):
        generated_token_nll(
            old_log_prob=log_prob.detach(),
            log_prob=log_prob.detach(),
            advantages=torch.ones_like(advantages),
            response_mask=mask,
            loss_agg_mode="token-mean",
            config=SimpleNamespace(global_batch_info={}),
        )


def test_manager_import_hook_exports_the_exact_upstream_tq_class() -> None:
    pytest.importorskip("verl")
    pytest.importorskip("transfer_queue")
    from verl.trainer.ppo.v1 import AgentLoopManagerTQ

    from tgvf_rl.framework.verl.sync_gate_manager import (
        SyncGateAgentLoopManagerTQ,
    )

    assert SyncGateAgentLoopManagerTQ is AgentLoopManagerTQ
