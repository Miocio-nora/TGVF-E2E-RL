from __future__ import annotations

from pathlib import Path

from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.representation import TGVFAdapterVariant
from tgvf_rl.representation.training.config import (
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
    load_representation_training_config,
)
from tgvf_rl.representation.training.losses import MatrixCEScoreMode


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FRESH_CONFIG = _REPOSITORY_ROOT / (
    "configs/smoke/"
    "representation_qwen3_instruct_rp65_periodic_boundary_"
    "formaldata_contextual_gpu01.toml"
)
_RESUME_CONFIG = _REPOSITORY_ROOT / (
    "configs/smoke/"
    "representation_qwen3_instruct_rp65_periodic_boundary_"
    "formaldata_contextual_gpu01_resume.toml"
)
_FORMAL_CONFIG = _REPOSITORY_ROOT / (
    "configs/representation/qwen3_instruct_balanced_t1_contextual_2000step_gpu01.toml"
)


def test_instruct_periodic_boundary_smoke_configs_bind_selected_stage1_contract() -> (
    None
):
    fresh = load_representation_training_config(
        _FRESH_CONFIG,
        verify_external_files=False,
    )
    resume = load_representation_training_config(
        _RESUME_CONFIG,
        verify_external_files=False,
    )

    assert fresh.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4
    assert fresh.run_id == resume.run_id
    assert fresh.model.model_name == "Qwen3-VL-8B-Instruct"
    assert str(fresh.model.local_path).endswith("/Qwen3-VL-8B-Instruct")
    assert fresh.model.chat_template_sha256 == (
        "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
    )
    assert fresh.model.image_max_pixels == 262_144
    assert fresh.adapter_variant is TGVFAdapterVariant.FULL_D_DEEPSTACK
    assert (
        fresh.provider.provider
        is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    )
    assert fresh.provider.hidden_layer == -1
    assert fresh.objective.objective.matrix_ce_mode is MatrixCEScoreMode.BALANCED
    assert fresh.objective.objective.matrix_ce_temperature == 1.0
    assert fresh.data.train.batch_size == fresh.data.validation.batch_size == 4
    assert fresh.training.gradient_accumulation_steps == 4
    assert fresh.training.target_optimizer_steps == 2_000
    assert fresh.training.validation_every_optimizer_steps == 1
    assert fresh.training.log_every_optimizer_steps == 1
    assert fresh.checkpoint.save_every_optimizer_steps == 1
    assert fresh.fsdp2.world_size == 2
    assert fresh.fsdp2.physical_gpu_ids == (0, 1)
    assert fresh.fsdp2.logical_gpu_ids == (0, 1)
    assert fresh.fsdp2.reshard_after_forward is False
    assert fresh.resume.enabled is False
    assert resume.resume.enabled is True
    assert resume.resume.checkpoint_path == (
        fresh.checkpoint.directory
        / "representation-qwen3-instruct-rp65-periodic-boundary-step-00000001"
    )
    assert resume.output == fresh.output
    assert resume.checkpoint == fresh.checkpoint


def test_instruct_formal_config_reuses_selected_two_rank_2000_step_cadence() -> None:
    config = load_representation_training_config(
        _FORMAL_CONFIG,
        verify_external_files=False,
    )

    assert config.run_id == (
        "RP-66-QWEN3-INSTRUCT-REP-BALANCED-T1-CONTEXTUAL-2000-GPU01"
    )
    assert config.model.model_name == "Qwen3-VL-8B-Instruct"
    assert config.model.image_max_pixels == 262_144
    assert config.adapter_variant is TGVFAdapterVariant.FULL_D_DEEPSTACK
    assert (
        config.provider.provider
        is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    )
    assert config.provider.hidden_layer == -1
    assert config.objective.objective.matrix_ce_mode is MatrixCEScoreMode.BALANCED
    assert config.objective.objective.matrix_ce_temperature == 1.0
    assert config.data.train.batch_size == config.data.validation.batch_size == 4
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.target_optimizer_steps == 2_000
    assert config.training.validation_every_optimizer_steps == 500
    assert config.training.log_every_optimizer_steps == 10
    assert config.checkpoint.save_every_optimizer_steps == 500
    assert config.checkpoint.keep_last == 4
    assert config.fsdp2.world_size == 2
    assert config.fsdp2.physical_gpu_ids == (0, 1)
    assert config.fsdp2.logical_gpu_ids == (0, 1)
    assert config.fsdp2.reshard_after_forward is False
    assert config.resume.enabled is False
