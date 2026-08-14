from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tomllib

import pytest

from tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset import (
    CROP_TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    TrainableTGVFVerlLaunchPlan,
    build_trainable_tgvf_verl_launch_plan,
    preflight_trainable_tgvf_verl_runtime,
)
from tgvf_rl.policy.config import (
    PolicyCropTGVFMatchedExperimentConfig,
    PolicyTrainableRP66ExperimentConfig,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
)
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    RP66AdapterUpdateMode,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.protocol.schema import NativeToolCapabilityProfile


_ROOT = Path(__file__).parents[2]
_PRL20_FORMAL = (
    _ROOT
    / "configs/policy/runs/"
    "prl_20_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
    "crop_tgvf_8step_ws8.toml"
)
_PRL20_CANARY = (
    _ROOT
    / "configs/policy/runs/"
    "prl_20_r0_c0_qwen3_instruct_full_frozen_rp67_bs4_n2_tfree_"
    "crop_tgvf_canary_ws4.toml"
)
_PRL17_TFREE_FORMAL = (
    _ROOT
    / "configs/policy/runs/"
    "prl_17_r2_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
    "novisual_8step_ws8.toml"
)


def _portable_config(source: Path, destination: Path) -> PolicyE2ESmokeRunConfig:
    """Rebind historical worktree-local dependencies to this checkout."""

    text = source.read_text(encoding="utf-8")
    payload = tomllib.loads(text)
    dependencies = (
        Path(payload["reward"]["judge_config_path"]),
        Path(payload["framework"]["agent_loop_config_path"]),
    )
    for dependency_root in {path.parents[3] for path in dependencies}:
        text = text.replace(str(dependency_root), str(_ROOT))
    destination.write_text(text, encoding="utf-8")
    return load_policy_e2e_smoke_run_config(destination)


def test_repository_root_token_resolves_against_active_checkout() -> None:
    config = load_policy_e2e_smoke_run_config(_PRL20_FORMAL)

    assert config.reward.judge_config_path is not None
    assert config.reward.judge_config_path.is_relative_to(_ROOT)
    assert config.framework.agent_loop_config_path.is_relative_to(_ROOT)


def test_gpu_preflight_checks_git_identity_before_loading_verl(
    prl20_canary: PolicyE2ESmokeRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_trainable_tgvf_verl_launch_plan(prl20_canary, mode="canary")
    calls: list[str] = []
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.trainable_tgvf_launcher.assert_policy_execution_identity",
        lambda *_args, **_kwargs: calls.append("identity"),
    )
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.trainable_tgvf_launcher.load_verl_public_api",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after identity")),
    )

    with pytest.raises(RuntimeError, match="stop after identity"):
        preflight_trainable_tgvf_verl_runtime(plan, object())
    assert calls == ["identity"]




@pytest.fixture(scope="module")
def prl20_formal(tmp_path_factory: pytest.TempPathFactory) -> PolicyE2ESmokeRunConfig:
    return _portable_config(
        _PRL20_FORMAL,
        tmp_path_factory.mktemp("prl20-formal") / _PRL20_FORMAL.name,
    )


@pytest.fixture(scope="module")
def prl20_canary(tmp_path_factory: pytest.TempPathFactory) -> PolicyE2ESmokeRunConfig:
    return _portable_config(
        _PRL20_CANARY,
        tmp_path_factory.mktemp("prl20-canary") / _PRL20_CANARY.name,
    )


@pytest.fixture(scope="module")
def prl17_tfree_formal(
    tmp_path_factory: pytest.TempPathFactory,
) -> PolicyE2ESmokeRunConfig:
    return _portable_config(
        _PRL17_TFREE_FORMAL,
        tmp_path_factory.mktemp("prl17-tfree-formal") / _PRL17_TFREE_FORMAL.name,
    )


@pytest.fixture(scope="module")
def prl20_formal_plan(
    prl20_formal: PolicyE2ESmokeRunConfig,
) -> TrainableTGVFVerlLaunchPlan:
    return build_trainable_tgvf_verl_launch_plan(prl20_formal, mode="formal")


@pytest.fixture(scope="module")
def prl17_tfree_formal_plan(
    prl17_tfree_formal: PolicyE2ESmokeRunConfig,
) -> TrainableTGVFVerlLaunchPlan:
    return build_trainable_tgvf_verl_launch_plan(
        prl17_tfree_formal,
        mode="formal",
    )


def test_prl20_config_is_atomic_frozen_rp67_tfree(
    prl20_formal: PolicyE2ESmokeRunConfig,
) -> None:
    config = prl20_formal

    assert config.schema_version == (
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA
    )
    assert isinstance(config.policy, PolicyCropTGVFMatchedExperimentConfig)
    assert config.representation.adapter_update_mode is (
        RP66AdapterUpdateMode.FROZEN_ADAPTER
    )
    assert config.representation.adapter_trainable is False
    assert config.representation.artifact.version == "rp67-step2000"

    assert config.protocol.tool_profile is NativeToolCapabilityProfile.CROP_TGVF
    assert config.protocol.enabled_tool_names == (
        CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
    )
    assert config.protocol.maximum_tool_calls == 6
    assert config.protocol.prompt_sha256 == (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert config.protocol.tool_schema_sha256 == (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.tool_schema_sha256
    )

    assert config.reward.tool_utility_reward_enabled is False
    assert config.reward.focus_reward_enabled is False
    assert config.reward.grounding_reward_enabled is False
    assert config.reward.visual_quality_judge_config_path is None
    assert config.reward.visual_quality_judge_identity is None


def test_prl20_formal_budget_exactly_matches_prl17_tfree_control(
    prl20_formal: PolicyE2ESmokeRunConfig,
    prl17_tfree_formal: PolicyE2ESmokeRunConfig,
    prl20_formal_plan: TrainableTGVFVerlLaunchPlan,
    prl17_tfree_formal_plan: TrainableTGVFVerlLaunchPlan,
) -> None:
    def config_budget(config: PolicyE2ESmokeRunConfig) -> dict[str, int | float]:
        return {
            "world_size": config.distributed.world_size,
            "global_prompt_batch_size": (
                config.accumulation.global_prompt_batch_size
            ),
            "trajectories_per_prompt": (
                config.policy.sampling.trajectories_per_prompt
            ),
            "prompt_micro_batch_size_per_rank": (
                config.accumulation.prompt_micro_batch_size_per_rank
            ),
            "rollout_prompt_micro_batch_size_per_engine": (
                config.accumulation.rollout_prompt_micro_batch_size_per_engine
            ),
            "gradient_accumulation_steps": (
                config.accumulation.gradient_accumulation_steps
            ),
            "learning_rate": config.optimizer.learning_rate,
            "temperature": config.policy.sampling.temperature,
            "max_response_length": config.policy.sampling.max_response_length,
        }

    expected = {
        "world_size": 8,
        "global_prompt_batch_size": 16,
        "trajectories_per_prompt": 16,
        "prompt_micro_batch_size_per_rank": 2,
        "rollout_prompt_micro_batch_size_per_engine": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-6,
        "temperature": 1.0,
        "max_response_length": 20_480,
    }
    assert config_budget(prl20_formal) == config_budget(prl17_tfree_formal) == expected

    new_values = prl20_formal_plan.overrides
    control_values = prl17_tfree_formal_plan.overrides
    matched_override_paths = (
        "data.train_batch_size",
        "data.gen_batch_size",
        "data.max_response_length",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.response_length",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.optim.total_training_steps",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size",
        "trainer.n_gpus_per_node",
        "trainer.total_training_steps",
    )
    assert {path: new_values[path] for path in matched_override_paths} == {
        path: control_values[path] for path in matched_override_paths
    }
    assert new_values["actor_rollout_ref.rollout.custom"][
        "actor_batch_contract"
    ] == control_values["actor_rollout_ref.rollout.custom"][
        "actor_batch_contract"
    ]


def test_launcher_selects_atomic_dataset_without_mutating_pure_tgvf(
    prl20_formal_plan: TrainableTGVFVerlLaunchPlan,
    prl17_tfree_formal_plan: TrainableTGVFVerlLaunchPlan,
) -> None:
    atomic = prl20_formal_plan.overrides
    pure = prl17_tfree_formal_plan.overrides

    assert atomic["data.custom_cls.name"] == "CropTGVFDeepEyesMatchedDataset"
    assert atomic["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
    )
    assert prl20_formal_plan.external_components["dataset"] == (
        CROP_TGVF_DEEPEYES_MATCHED_DATASET_CLASS
    )
    assert "data.deepeyes_tgvf_matched" not in atomic
    atomic_binding = atomic["data.deepeyes_crop_tgvf_matched"]
    assert isinstance(atomic_binding, Mapping)
    assert atomic_binding["visual_prompt_bundle_sha256"] == (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    # ThinkLite remains on the common direct-only/no-tool prompt route.
    assert atomic_binding["thinklite_prompt_bundle_sha256"] == (
        THINKLITE_PROMPT_IDENTITY.bundle_sha256
    )

    assert isinstance(
        prl17_tfree_formal_plan.config.policy,
        PolicyTrainableRP66ExperimentConfig,
    )
    assert pure["data.custom_cls.name"] == "TGVFDeepEyesMatchedDataset"
    assert pure["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
    )
    assert prl17_tfree_formal_plan.external_components["dataset"] == (
        TGVF_DEEPEYES_MATCHED_DATASET_CLASS
    )
    assert "data.deepeyes_crop_tgvf_matched" not in pure
    pure_binding = pure["data.deepeyes_tgvf_matched"]
    assert isinstance(pure_binding, Mapping)
    assert pure_binding["visual_prompt_bundle_sha256"] == (
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert pure_binding["thinklite_prompt_bundle_sha256"] == (
        THINKLITE_PROMPT_IDENTITY.bundle_sha256
    )


def test_prl20_canary_is_low_cost_but_keeps_atomic_protocol(
    prl20_canary: PolicyE2ESmokeRunConfig,
) -> None:
    plan = build_trainable_tgvf_verl_launch_plan(prl20_canary, mode="canary")
    values = plan.overrides

    assert isinstance(prl20_canary.policy, PolicyCropTGVFMatchedExperimentConfig)
    assert values["trainer.n_gpus_per_node"] == 4
    assert values["data.train_batch_size"] == 4
    assert values["actor_rollout_ref.rollout.n"] == 2
    assert prl20_canary.policy.sampling.max_response_length == 512
    assert values["trainer.total_training_steps"] == 1
    assert values["trainer.logger"] == ["console"]
    assert values["data.custom_cls.name"] == "CropTGVFDeepEyesMatchedDataset"
    assert values["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
    )
    assert values["actor_rollout_ref.rollout.custom"]["protocol"] == {
        "prompt_sha256": (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        ),
        "tool_schema_sha256": (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.tool_schema_sha256
        ),
        "cap_error_sha256": prl20_canary.protocol.cap_error_sha256,
        "maximum_tool_calls": 6,
    }
