from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.data.policy_teacher_ratio_mix import (
    POLICY_TEACHER_RATIO_MIX_DATASET_KIND,
    PolicyTeacherRatioMixRuntimeBinding,
    policy_teacher_ratio_mix_iteration_identity_sha256,
)
from tgvf_rl.framework.verl.policy_teacher_ratio_mix_dataset import (
    POLICY_TEACHER_RATIO_MIX_CONFIG_NAME,
    POLICY_TEACHER_RATIO_MIX_DATASET_CLASS,
    POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH,
    PolicyTeacherRatioMixDatasetBinding,
)
from tgvf_rl.framework.verl.policy_live_runtime import _validate_sample_fields
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
    compose_trainable_tgvf_verl_config,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)


_ROOT = Path(__file__).resolve().parents[2]
_TEACHER25_CONFIG = (
    _ROOT
    / "configs/policy/runs/"
    "prl_22_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher25_8step_ws8.toml"
)


def _ratio_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    teacher_percentage: int,
) -> Path:
    dataset_root = (tmp_path / f"teacher{teacher_percentage}").resolve()
    dataset_root.mkdir()
    binding = PolicyTeacherRatioMixRuntimeBinding(
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        schedule_seed=42,
        expected_sample_count=20_480,
        teacher_percentage=teacher_percentage,
    )
    samples_sha256 = "3" * 64
    iteration_identity = policy_teacher_ratio_mix_iteration_identity_sha256(
        binding, samples_sha256=samples_sha256
    )
    text = _TEACHER25_CONFIG.read_text(encoding="utf-8")
    dataset_begin = text.index("[dataset]")
    representation_begin = text.index("[representation]")
    dataset_text = f'''[dataset]
kind = "{POLICY_TEACHER_RATIO_MIX_DATASET_KIND}"
root = "{dataset_root}"
decision_stage = "final"
sample_count = 20480
manifest_file_sha256 = "{binding.manifest_file_sha256}"
content_sha256 = "{binding.content_sha256}"
samples_sha256 = "{samples_sha256}"
iteration_identity_sha256 = "{iteration_identity}"
shuffle_seed = 42
teacher_percentage = {teacher_percentage}

'''
    text = text[:dataset_begin] + dataset_text + text[representation_begin:]
    old_output = (
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-"
        "8step-ws8"
    )
    output_root = (tmp_path / f"output-teacher{teacher_percentage}").resolve()
    text = text.replace(old_output, str(output_root))
    text = text.replace(
        "PRL-22-A-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-TEACHER25-"
        "8STEP-WS8",
        f"PRL-23-TEACHER{teacher_percentage}-INTEGRATION",
    )
    path = tmp_path / f"teacher{teacher_percentage}.toml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("TGVF_REPOSITORY_ROOT", str(_ROOT))
    monkeypatch.setattr(
        "tgvf_rl.policy.run_config.verify_policy_teacher_ratio_mix_artifact_binding",
        lambda *_args, **_kwargs: None,
    )
    return path


@pytest.mark.parametrize("teacher_percentage", (50, 100))
def test_ratio_config_closes_load_plan_and_pinned_compose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_percentage: int,
) -> None:
    path = _ratio_config(
        tmp_path, monkeypatch, teacher_percentage=teacher_percentage
    )

    config = load_policy_e2e_smoke_run_config(
        path, allow_external_agent_loop_config=True
    )
    assert isinstance(config.dataset.runtime_binding, PolicyTeacherRatioMixRuntimeBinding)
    assert config.dataset.runtime_binding.teacher_percentage == teacher_percentage

    plan = build_trainable_tgvf_verl_launch_plan(
        config, mode="formal", target_step=8
    )
    assert plan.overrides["data.custom_cls.path"] == (
        POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH
    )
    assert plan.overrides["data.custom_cls.name"] == (
        POLICY_TEACHER_RATIO_MIX_DATASET_CLASS.rsplit(".", 1)[-1]
    )
    binding_key = f"data.{POLICY_TEACHER_RATIO_MIX_CONFIG_NAME}"
    emitted_binding = plan.overrides[binding_key]
    assert emitted_binding["teacher_percentage"] == teacher_percentage
    assert plan.overrides["data.train_files"] == [
        str(config.dataset.root / "samples.jsonl")
    ]

    composed = compose_trainable_tgvf_verl_config(plan)
    restored = PolicyTeacherRatioMixDatasetBinding.from_config(
        getattr(composed.data, POLICY_TEACHER_RATIO_MIX_CONFIG_NAME)
    )
    assert restored.teacher_percentage == teacher_percentage
    assert restored.runtime_binding == config.dataset.runtime_binding


def test_ratio_config_rejects_percentage_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _ratio_config(tmp_path, monkeypatch, teacher_percentage=50)
    text = path.read_text(encoding="utf-8").replace(
        "teacher_percentage = 50", "teacher_percentage = 100", 1
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="iteration identity differs"):
        load_policy_e2e_smoke_run_config(
            path, allow_external_agent_loop_config=True
        )


def test_ratio_live_identity_uses_normalized_top_level_schedule_fields(
    tmp_path: Path,
) -> None:
    image_path = (tmp_path / "teacher.png").resolve()
    sample_id = "teacher-ratio:fixture"
    record = {
        "sample_id": sample_id,
        "candidate_sha256": "1" * 64,
        "data_source": "teacher",
        "source_dataset": "chartqa",
        "task_kind": "mcq",
        "question": "What is shown?",
        "ground_truth": "A",
        "image": {
            "path": str(image_path),
            "sha256": "2" * 64,
            "width": 10,
            "height": 10,
        },
        "gt_regions": None,
        "mixture_role": "teacher",
        "parent": {
            "dataset_kind": "fixture",
            "row_index": 0,
            "row_sha256": "3" * 64,
        },
        "schedule_index": 0,
        "schema_version": "fixture",
    }
    config = SimpleNamespace(
        schema_version=POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        protocol=SimpleNamespace(prompt_sha256="8" * 64),
        dataset=SimpleNamespace(
            selected_sample=None,
            root=tmp_path,
            iteration_identity_sha256="9" * 64,
            runtime_binding=PolicyTeacherRatioMixRuntimeBinding(
                manifest_file_sha256="4" * 64,
                content_sha256="5" * 64,
                schedule_seed=42,
                expected_sample_count=20_480,
                teacher_percentage=50,
            ),
        ),
    )
    fields = {
        "sample_id": sample_id,
        "prompt_bundle_sha256": "8" * 64,
        "source_image_path": str(image_path),
        "source_image_sha256": "2" * 64,
        "question": "What is shown?",
        "data_source": "teacher",
        "task_kind": "mcq",
        "reward_model": {"ground_truth": "A"},
    }

    _validate_sample_fields(
        config, sample_id, fields, sample_index={sample_id: record}
    )
