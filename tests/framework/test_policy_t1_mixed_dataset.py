from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.policy_t1_mixed_dataset import (
    VerlPolicyT1MixedDatasetBinding,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
)
from tgvf_rl.protocol import (
    NativeToolCapabilityProfile,
    native_assistant_dialect_for_model,
    visual_tool_prompt_identity,
)


def _binding(
    root: Path, *, decision_stage: str = "final"
) -> VerlPolicyT1MixedDatasetBinding:
    profile = NativeToolCapabilityProfile.CROP_ONLY
    return VerlPolicyT1MixedDatasetBinding(
        root=root.resolve(),
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        samples_sha256="3" * 64,
        iteration_identity_sha256="4" * 64,
        shuffle_seed=42,
        decision_stage=decision_stage,
        expected_sample_count=79_069,
        prompt_bundle_sha256=visual_tool_prompt_identity(
            profile,
            assistant_dialect=native_assistant_dialect_for_model(
                POLICY_PILOT_V1_MODEL_NAME
            ),
        ).bundle_sha256,
        tool_profile=profile,
        tokenizer_length=POLICY_PILOT_V1_TOKENIZER_LENGTH,
        model_name=POLICY_PILOT_V1_MODEL_NAME,
    )


def test_mixed_verl_binding_round_trips_without_weakening_final_identity(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)

    assert VerlPolicyT1MixedDatasetBinding.from_config(binding.as_config()) == binding
    assert binding.runtime_binding.expected_sample_count == 79_069
    assert binding.as_config()["decision_stage"] == "final"


def test_mixed_verl_binding_rejects_nonfinal_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires decision_stage='final'"):
        _binding(tmp_path, decision_stage="provisional")
