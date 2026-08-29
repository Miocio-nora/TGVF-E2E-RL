from __future__ import annotations

from pathlib import Path
import tomllib

from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.config import PolicyCropExactPixel512ParityExperimentConfig
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_REFERENCE = _RUNS / (
    "prl_26_b_qwen3_instruct_full_crop_train512_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
_CORRECTED = _RUNS / (
    "prl_27_a_qwen3_instruct_full_crop_train512_exact_continuation_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
_CORE_FIX_COMMIT = "ecddc379d392d154c91783d7651528b20d40afba"
_RUN_ID = (
    "PRL-27-A-TRAIN512-S32-CROP-EXACT-CONTINUATION-"
    "QWEN3-INSTRUCT-BS16-N16-TEACHER25-WS8"
)


def _toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_prl27_crop_is_fresh_s0_core_fixed_and_otherwise_prl26_parity() -> None:
    config = load_policy_e2e_smoke_run_config(_CORRECTED.resolve())

    assert config.schema_version == (
        POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
    )
    assert config.run_id == _RUN_ID
    assert config.code.commit == _CORE_FIX_COMMIT
    assert isinstance(config.policy, PolicyCropExactPixel512ParityExperimentConfig)
    assert config.policy.image_max_pixels == 512 * 512
    assert config.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY
    assert config.protocol.enabled_tool_names == ("image_zoom_in_tool",)
    assert config.policy.sampling.stop_strings == ("</tool_call>",)
    assert config.policy.sampling.include_stop_str_in_output is True
    assert config.training.maximum_optimizer_steps == 32
    assert config.training.checkpoint_steps == (0, 8, 16, 24, 32)
    assert config.training.resume_mode == "auto"
    assert config.training.resume_from_path is None
    assert "PRL-27-A" in str(config.output.root)
    assert "PRL-26-B" not in str(config.output.root)

    plan = build_trainable_tgvf_verl_launch_plan(config, mode="formal")
    assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 512 * 512

    reference = _toml(_REFERENCE)
    corrected = _toml(_CORRECTED)
    assert corrected["code"]["dirty"] is False
    corrected["run_id"] = reference["run_id"]
    corrected["code"]["commit"] = reference["code"]["commit"]
    corrected["reward"]["judge_reason"] = reference["reward"]["judge_reason"]
    corrected["output"] = reference["output"]
    assert corrected == reference
