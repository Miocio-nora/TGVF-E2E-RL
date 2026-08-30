from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.policy.deepeyes_native_contract import (
    DEEPEYES_NATIVE_CHECKPOINT_GATES,
    DEEPEYES_NATIVE_EVALUATION_GATES,
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_PROMPT_IDENTITY,
    VISUAL_PROMPT_IDENTITY,
)


_ROOT = Path(__file__).parents[2]
_TEMPLATES = tuple(
    sorted((_ROOT / "configs/policy/runs").glob("prl_13_*.template.toml"))
)
_LAUNCHABLE = (
    _ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.toml"
)


def test_both_templates_bind_executable_dataset_and_async_reward_manager() -> None:
    assert len(_TEMPLATES) == 2
    for path in _TEMPLATES:
        contract = load_deepeyes_native_run_contract(path)
        assert contract.launch_enabled is False
        assert contract.code_commit == "CORE_COMMIT_REQUIRED"
        dataset = contract.payload["dataset"]
        reward = contract.payload["reward"]
        protocol = contract.payload["protocol"]
        assert dataset["verl_dataset_class_name"] == "TGVFDeepEyesOfficialDataset"
        assert dataset["train_files"][0].endswith("prl13-train.schedule")
        assert dataset["probe_files"][0].endswith("prl13-probe.schedule")
        assert reward["reward_manager_source"] == "importlib"
        assert reward["reward_manager_class_name"] == ("DeepEyesOfficialRewardManager")
        assert reward["reward_num_workers"] == 1
        assert reward["judge_batch_max_concurrency"] == 64
        assert contract.payload["rollout"]["free_cache_engine"] is True
        assert protocol["coordinate_mapper"] == "qwen_0_1000_to_source_v1"
        assert protocol["unified_train_eval_coordinate_mapper"] is True
        assert protocol["max_active_perception"] == 6
        assert contract.payload["rollout"]["max_user_turns"] == 6
        assert contract.payload["rollout"]["max_assistant_turns"] == 7
        assert contract.payload["optimization"]["actor_loss_reduction"] == (
            "deepeyes_official_micro_token_mean"
        )
        assert contract.payload["training"]["checkpoint_steps"] == list(
            DEEPEYES_NATIVE_CHECKPOINT_GATES
        )
        assert contract.payload["training"]["evaluation_steps"] == list(
            DEEPEYES_NATIVE_EVALUATION_GATES
        )


def test_launchable_contract_binds_enabled_judge_bytes() -> None:
    contract = load_deepeyes_native_run_contract(_LAUNCHABLE, allow_template=False)
    assert contract.launch_enabled is True
    assert contract.payload["reward"]["judge_service_config_sha256"] == (
        "f9dac7a2baa727647b2310eae94cd1e0990bfc4e35f30760f454b0d46b1bfff5"
    )


def test_historical_snapshot_contract_loads_while_new_templates_are_clean(
    tmp_path: Path,
) -> None:
    historical_path = tmp_path / "historical-prl13-snapshot.toml"
    historical_path.write_text(
        _LAUNCHABLE.read_text(encoding="utf-8")
        .replace(
            VISUAL_PROMPT_IDENTITY.bundle_sha256,
            "b91cfa2e228f496f745a6e0b368cff836e6786ad104e312cd776a12e0784b2ef",
            1,
        )
        .replace(
            THINKLITE_PROMPT_IDENTITY.bundle_sha256,
            "bd3226c36dea66cc57a9b30fad5c846ce3d3a23a008781425c70214580aba545",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="visual_prompt_bundle_sha256"):
        load_deepeyes_native_run_contract(
            historical_path,
            allow_template=False,
        )
    historical = load_deepeyes_native_run_contract(
        historical_path,
        allow_template=False,
        allow_historical_prompt_contract=True,
    )
    assert historical.payload["protocol"]["visual_prompt_bundle_sha256"] == (
        "b91cfa2e228f496f745a6e0b368cff836e6786ad104e312cd776a12e0784b2ef"
    )
    assert historical.payload["protocol"]["thinklite_prompt_bundle_sha256"] == (
        "bd3226c36dea66cc57a9b30fad5c846ce3d3a23a008781425c70214580aba545"
    )
    for path in _TEMPLATES:
        current = load_deepeyes_native_run_contract(path)
        assert current.payload["protocol"]["visual_prompt_bundle_sha256"] == (
            VISUAL_PROMPT_IDENTITY.bundle_sha256
        )
        assert current.payload["protocol"]["thinklite_prompt_bundle_sha256"] == (
            THINKLITE_PROMPT_IDENTITY.bundle_sha256
        )


def test_template_rejects_noncanonical_judge_config_sha(tmp_path: Path) -> None:
    text = _TEMPLATES[0].read_text(encoding="utf-8")
    path = tmp_path / "drift.toml"
    path.write_text(
        text.replace(
            "bcd45d0e3ef996defd6c50fa227db341859acc86dac2f8d6dd36b8c425ba4a8c",
            "f" * 64,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="template judge service"):
        load_deepeyes_native_run_contract(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'reward_manager = "deepeyes_official_async"',
            'reward_manager = "naive"',
            "reward.reward_manager",
        ),
        (
            "reward_num_workers = 1",
            "reward_num_workers = 64",
            "reward.reward_num_workers",
        ),
        ("lora_rank = 0", "lora_rank = 64", "model.lora_rank"),
        (
            'coordinate_mapper = "qwen_0_1000_to_source_v1"',
            'coordinate_mapper = "source_pixels"',
            "protocol.coordinate_mapper",
        ),
        (
            'actor_loss_reduction = "deepeyes_official_micro_token_mean"',
            'actor_loss_reduction = "token-mean"',
            "optimization.actor_loss_reduction",
        ),
    ),
)
def test_contract_rejects_semantic_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    text = _TEMPLATES[0].read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "drift.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_deepeyes_native_run_contract(path)
