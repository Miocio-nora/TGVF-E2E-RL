from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
    compose_trainable_tgvf_verl_config,
)
from tgvf_rl.policy.config import (
    PolicyCropExactPixel512ParityExperimentConfig,
    PolicyNoToolPixel512ParityExperimentConfig,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_IMPLEMENTATION_COMMIT = "e756546b273be70992c72471ed549b3e3a2834ae"
_CASES = (
    (
        "prl_26_a_c0_qwen3_instruct_full_no_tool_train512_parity_"
        "bs4_n2_teacher25_1step_ws4.toml",
        PolicyNoToolPixel512ParityExperimentConfig,
        "canary",
        "disable",
        (0, 1, 2, 3),
        (0, 1),
        (),
    ),
    (
        "prl_26_b_c0_qwen3_instruct_full_crop_train512_parity_"
        "bs4_n2_teacher25_1step_ws4.toml",
        PolicyCropExactPixel512ParityExperimentConfig,
        "canary",
        "disable",
        (4, 5, 6, 7),
        (0, 1),
        ("</tool_call>",),
    ),
    (
        "prl_26_a_qwen3_instruct_full_no_tool_train512_parity_"
        "s32_bs16_n16_teacher25_ws8.toml",
        PolicyNoToolPixel512ParityExperimentConfig,
        "formal",
        "auto",
        tuple(range(8)),
        (0, 8, 16, 24, 32),
        (),
    ),
    (
        "prl_26_b_qwen3_instruct_full_crop_train512_parity_"
        "s32_bs16_n16_teacher25_ws8.toml",
        PolicyCropExactPixel512ParityExperimentConfig,
        "formal",
        "auto",
        tuple(range(8)),
        (0, 8, 16, 24, 32),
        ("</tool_call>",),
    ),
)


@pytest.mark.parametrize(
    (
        "filename",
        "policy_type",
        "mode",
        "resume_mode",
        "physical_gpu_ids",
        "checkpoint_steps",
        "stop_strings",
    ),
    _CASES,
)
def test_train512_parity_configs_load_and_compose(
    filename: str,
    policy_type: type,
    mode: str,
    resume_mode: str,
    physical_gpu_ids: tuple[int, ...],
    checkpoint_steps: tuple[int, ...],
    stop_strings: tuple[str, ...],
) -> None:
    config = load_policy_e2e_smoke_run_config((_RUNS / filename).resolve())

    assert config.code.commit == _IMPLEMENTATION_COMMIT
    assert isinstance(config.policy, policy_type)
    assert config.policy.image_max_pixels == 512 * 512
    assert config.training.resume_mode == resume_mode
    assert config.training.resume_from_path is None
    assert config.distributed.physical_gpu_ids == physical_gpu_ids
    assert config.training.checkpoint_steps == checkpoint_steps
    assert config.policy.sampling.stop_strings == stop_strings
    assert config.policy.sampling.include_stop_str_in_output is True

    plan = build_trainable_tgvf_verl_launch_plan(config, mode=mode)
    assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 512 * 512
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == ",".join(
        str(device) for device in physical_gpu_ids
    )
    compose_trainable_tgvf_verl_config(plan)


@pytest.mark.parametrize(
    ("filename", "old_pixels"),
    (
        (
            "prl_25_f_qwen3_instruct_full_no_tool_rl_bs16_n16_tfree_"
            "teacher25_32step_ws8.toml",
            1_003_520,
        ),
        (
            "prl_25_b_qwen3_instruct_full_crop_exact_bs16_n16_tfree_"
            "teacher25_80step_ws8.toml",
            1_003_520,
        ),
    ),
)
def test_legacy_matched_schemas_remain_strict_true1m(
    tmp_path: Path, filename: str, old_pixels: int
) -> None:
    source = _RUNS / filename
    config = load_policy_e2e_smoke_run_config(source.resolve())
    assert config.policy.image_max_pixels == old_pixels

    mutated = tmp_path / filename
    mutated.write_text(
        source.read_text(encoding="utf-8").replace(
            "image_max_pixels = 1003520", "image_max_pixels = 262144"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.image_max_pixels differs"):
        load_policy_e2e_smoke_run_config(mutated.resolve())


@pytest.mark.parametrize(
    ("filename", "old", "new", "message"),
    (
        (
            _CASES[2][0],
            "image_max_pixels = 262144",
            "image_max_pixels = 1003520",
            "model.image_max_pixels differs",
        ),
        (
            _CASES[2][0],
            'resume_mode = "auto"',
            'resume_mode = "disable"',
            "recoverable formal resume mode",
        ),
        (
            _CASES[2][0],
            "checkpoint_steps = [0, 8, 16, 24, 32]",
            "checkpoint_steps = [0, 8, 16, 32]",
            "S32 checkpoint plan",
        ),
        (
            _CASES[1][0],
            'resume_mode = "disable"',
            'resume_mode = "auto"',
            "C0 fresh-S0 resume mode",
        ),
        (
            _CASES[3][0],
            'stop_strings = ["</tool_call>"]',
            "stop_strings = []",
            "Crop action boundary",
        ),
    ),
)
def test_train512_parity_schema_rejects_contract_drift(
    tmp_path: Path,
    filename: str,
    old: str,
    new: str,
    message: str,
) -> None:
    source = _RUNS / filename
    text = source.read_text(encoding="utf-8")
    assert old in text
    mutated = tmp_path / filename
    mutated.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_policy_e2e_smoke_run_config(mutated.resolve())
