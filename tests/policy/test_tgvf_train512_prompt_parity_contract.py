from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
import tomllib

import pytest

from tgvf_rl.framework.verl.policy_teacher_quarter_mix_dataset import (
    POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME,
)
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
    compose_trainable_tgvf_verl_config,
)
from tgvf_rl.policy.config import PolicyTGVFPixel512ParityExperimentConfig
from tgvf_rl.policy.run_config import (
    RP66AdapterUpdateMode,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_FORMAL_SHORT = (
    "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
)
_FORMAL_FULL = (
    "prl_26_d_qwen3_instruct_target_guide_v2_tgvf_train512_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
_CANARY_SHORT = (
    "prl_26_c_c0_qwen3_instruct_short_tgvf_train512_parity_"
    "bs4_n2_teacher25_1step_ws4.toml"
)
_CANARY_FULL = (
    "prl_26_d_c0_qwen3_instruct_target_guide_v2_tgvf_train512_parity_"
    "bs4_n2_teacher25_1step_ws4.toml"
)
_PRL25_TGVF_REFERENCE = (
    "prl_25_c_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
    "teacher25_80step_ws8.toml"
)
_RP67_FILE_SHA256 = "13332865eb30a2b04ce2ee90a9228e490c718e87fa57bc758078cdd28b6f0f68"
_RP67_RUN_ID = "RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01"
_IMPLEMENTATION_COMMIT = "396a25819871f753f40242b5137e4c6f9fd49348"


@pytest.mark.parametrize(
    ("filename", "mode", "prompt_sha256", "steps", "checkpoints", "resume_mode"),
    (
        (
            _CANARY_SHORT,
            "canary",
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            1,
            (0, 1),
            "disable",
        ),
        (
            _CANARY_FULL,
            "canary",
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
            1,
            (0, 1),
            "disable",
        ),
        (
            _FORMAL_SHORT,
            "formal",
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            32,
            (0, 8, 16, 24, 32),
            "auto",
        ),
        (
            _FORMAL_FULL,
            "formal",
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
            32,
            (0, 8, 16, 24, 32),
            "auto",
        ),
    ),
)
def test_tgvf_train512_configs_load_compose_and_bind_exact_prompt_route(
    filename: str,
    mode: str,
    prompt_sha256: str,
    steps: int,
    checkpoints: tuple[int, ...],
    resume_mode: str,
) -> None:
    source = (_RUNS / filename).resolve()
    config = load_policy_e2e_smoke_run_config(source)

    assert config.code.commit == _IMPLEMENTATION_COMMIT
    assert isinstance(config.policy, PolicyTGVFPixel512ParityExperimentConfig)
    assert config.policy.image_max_pixels == 512 * 512
    assert config.policy.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY
    assert config.policy.max_tgvf_call_attempts == 6
    assert config.protocol.prompt_sha256 == prompt_sha256
    assert config.policy.sampling.stop_token_ids == (151645,)
    assert config.policy.sampling.stop_strings == ("</tool_call>",)
    assert config.policy.sampling.include_stop_str_in_output is True
    assert config.representation.adapter_update_mode is (
        RP66AdapterUpdateMode.FROZEN_ADAPTER
    )
    assert config.representation.artifact_file_sha256 == _RP67_FILE_SHA256
    assert config.representation.expected_run_id == _RP67_RUN_ID
    assert config.optimizer.learning_rate == 1.0e-6
    assert config.scheduler.name == "constant"
    assert config.scheduler.total_steps == steps
    assert config.training.maximum_optimizer_steps == steps
    assert config.training.checkpoint_steps == checkpoints
    assert config.training.resume_mode == resume_mode
    assert config.training.resume_from_path is None

    # Preserve the established PRL25/PRL26 matched reward contract.  These
    # assertions do not redefine supervision; they reject accidental sidecars
    # or dense visual-quality components in this prompt ablation.
    assert config.reward.tool_utility_reward_enabled is False
    assert config.reward.tool_utility is None
    assert config.reward.focus_reward_enabled is False
    assert config.reward.grounding_reward_enabled is False
    assert config.reward.visual_quality_judge_config_path is None
    assert config.reward.protocol_error_penalty == 2.0
    source_text = source.read_text(encoding="utf-8").lower()
    assert "tool_utility_sidecar" not in source_text
    assert "visual_quality_judge_config" not in source_text

    plan = build_trainable_tgvf_verl_launch_plan(config, mode=mode)
    assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 512 * 512
    dataset_binding = plan.overrides[f"data.{POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME}"]
    assert isinstance(dataset_binding, Mapping)
    assert dataset_binding["visual_prompt_bundle_sha256"] == prompt_sha256
    custom = plan.overrides["actor_rollout_ref.rollout.custom"]
    assert isinstance(custom, Mapping)
    trainable_tgvf = custom["trainable_tgvf"]
    assert isinstance(trainable_tgvf, Mapping)
    assert trainable_tgvf["adapter_update_mode"] == "frozen_adapter"
    assert trainable_tgvf["adapter_trainable"] is False
    try:
        verl_trainer_available = find_spec("verl.trainer") is not None
    except ModuleNotFoundError:
        verl_trainer_available = False
    if verl_trainer_available:
        compose_trainable_tgvf_verl_config(plan)


def _load_toml(filename: str) -> dict[str, object]:
    with (_RUNS / filename).open("rb") as handle:
        return tomllib.load(handle)


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        return set().union(
            *(
                _leaf_differences(left.get(key), right.get(key), path + (str(key),))
                for key in keys
            )
        )
    return set() if left == right else {path}


@pytest.mark.parametrize(
    ("short_filename", "full_filename"),
    ((_CANARY_SHORT, _CANARY_FULL), (_FORMAL_SHORT, _FORMAL_FULL)),
)
def test_short_and_full_configs_differ_only_in_prompt_identity_and_output(
    short_filename: str, full_filename: str
) -> None:
    short = _load_toml(short_filename)
    full = _load_toml(full_filename)

    expected_differences = {
        ("schema_version",),
        ("run_id",),
        ("protocol", "prompt_sha256"),
        ("output", "root"),
        ("output", "checkpoint_directory"),
        ("output", "metrics_path"),
    }
    if short_filename == _CANARY_SHORT:
        expected_differences.add(("distributed", "physical_gpu_ids"))
        assert short["distributed"]["physical_gpu_ids"] == [0, 1, 2, 3]
        assert full["distributed"]["physical_gpu_ids"] == [4, 5, 6, 7]
    assert _leaf_differences(short, full) == expected_differences
    assert short["reward"] == full["reward"]
    assert short["dataset"] == full["dataset"]
    assert short["representation"] == full["representation"]
    assert short["sampling"] == full["sampling"]


def test_both_prompt_arms_reuse_the_prl25_matched_reward_contract() -> None:
    reference_payload = _load_toml(_PRL25_TGVF_REFERENCE)
    reference_reward = dict(reference_payload["reward"])
    reference_reward.pop("judge_reason")

    for filename in (_FORMAL_SHORT, _FORMAL_FULL):
        observed_payload = _load_toml(filename)
        observed_reward = dict(observed_payload["reward"])
        observed_reward.pop("judge_reason")
        assert observed_reward == reference_reward
        assert observed_payload["dataset"] == reference_payload["dataset"]
        assert observed_payload["representation"] == reference_payload["representation"]


@pytest.mark.parametrize(
    ("filename", "old", "new", "message"),
    (
        (
            _FORMAL_FULL,
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            "protocol.prompt_sha256 differs",
        ),
        (
            _FORMAL_SHORT,
            "image_max_pixels = 262144",
            "image_max_pixels = 1003520",
            "model.image_max_pixels differs",
        ),
        (
            _FORMAL_SHORT,
            'adapter_update_mode = "frozen_adapter"',
            'adapter_update_mode = "joint"',
            "frozen-RP67 ownership",
        ),
        (
            _FORMAL_SHORT,
            'stop_strings = ["</tool_call>"]',
            "stop_strings = []",
            "tool action boundary",
        ),
        (
            _CANARY_FULL,
            'resume_mode = "disable"',
            'resume_mode = "auto"',
            "C0 fresh-S0 resume mode",
        ),
    ),
)
def test_tgvf_train512_schemas_reject_prompt_or_parity_drift(
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
