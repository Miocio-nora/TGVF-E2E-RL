import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.evaluation.vlmevalkit import (
    COREDEV_2511,
    COREDEV_2511_MANIFEST_SHA256,
    SHARED_BENCHMARK_ROOT,
    VLMEVALKIT_REVIEW_COMMIT,
    CoreDev2511Spec,
    CoreDevSliceSpec,
    TGVFPolicyEvaluationResult,
    VLMEvalKitLaunchPlan,
    inject_vllm_engine_options_from_factory_kwargs,
    isolate_torchrun_environment_for_spawned_factory,
    materialize_coredev_subset_config,
)


def test_coredev_2511_identity_and_official_scorer_slices_are_fixed() -> None:
    assert COREDEV_2511.manifest_sha256 == COREDEV_2511_MANIFEST_SHA256
    assert COREDEV_2511.seed == 20260625
    assert COREDEV_2511.sample_count == 2511
    assert tuple(item.vlmeval_dataset for item in COREDEV_2511.slices) == (
        "VStarBench",
        "HRBench4K",
        "BLINK",
        "OCRBench_v2",
        "MMMU_Pro_10c",
        "MathVista_MINI",
        "MathVerse_MINI",
    )


def test_coredev_2511_rejects_a_mixed_or_incomplete_suite() -> None:
    with pytest.raises(ValueError, match="exactly seven"):
        CoreDev2511Spec(
            manifest_sha256=COREDEV_2511_MANIFEST_SHA256,
            seed=20260625,
            slices=COREDEV_2511.slices[:1],
        )
    changed = list(COREDEV_2511.slices)
    changed[-1] = CoreDevSliceSpec(
        "mathverse", "mathverse_testmini_3940", "MathVerse_MINI", 499
    )
    with pytest.raises(ValueError, match="sample count drifted"):
        CoreDev2511Spec(
            manifest_sha256=COREDEV_2511_MANIFEST_SHA256,
            seed=20260625,
            slices=tuple(changed),
        )


def test_launch_plan_uses_pinned_external_checkout_and_shared_data() -> None:
    plan = VLMEvalKitLaunchPlan(
        checkout=Path("/opt/VLMEvalKit"),
        config_path=Path("/workspace/configs/coredev2511.json"),
        work_dir=Path("/workspace/results/run-1"),
    )
    assert plan.expected_commit == VLMEVALKIT_REVIEW_COMMIT
    assert plan.argv == (
        "python",
        "/opt/VLMEvalKit/run.py",
        "--config",
        "/workspace/configs/coredev2511.json",
        "--work-dir",
        "/workspace/results/run-1",
        "--mode",
        "all",
    )
    assert plan.environment == {
        "LMUData": str(SHARED_BENCHMARK_ROOT),
        "PRED_FORMAT": "tsv",
        "EVAL_FORMAT": "json",
    }


def test_policy_result_separates_prediction_from_extra_records() -> None:
    result = TGVFPolicyEvaluationResult(
        final_answer="42",
        extra_records={
            "trajectory_id": "trajectory-1",
            "ordered_tool_names": ["image_zoom_in_tool", "tgvf_focus_tool"],
        },
    )
    assert result.as_generate_inner_result() == (
        0,
        "42",
        {
            "trajectory_id": "trajectory-1",
            "ordered_tool_names": ["image_zoom_in_tool", "tgvf_focus_tool"],
        },
    )


def test_policy_result_rejects_non_json_extra_records() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        TGVFPolicyEvaluationResult(
            final_answer="answer",
            extra_records={"bad": float("nan")},
        )


def test_coredev_qwen3_direct_baseline_freezes_decoding_identity() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (repository_root / "configs/evaluation/coredev_2511_qwen3_direct_v1.json").read_text(
            encoding="utf-8"
        )
    )
    model = config["model"]["Qwen3-VL-8B-Thinking"]
    assert model == {
        "model_path": "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking",
        "use_custom_prompt": False,
        "use_vllm": True,
        "min_pixels": None,
        "max_pixels": 512 * 512,
        "total_pixels": None,
        "max_new_tokens": 40960,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "do_sample": True,
        "post_process": False,
        "system_prompt": None,
        "gpu_utils": 0.9,
        "limit_mm_per_prompt": {"image": 24, "video": 0},
        "max_model_len": 65536,
        "mm_encoder_attn_backend": "TORCH_SDPA",
    }


def test_qwen_factory_forwards_only_accepted_vllm_engine_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def original_llm(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    fake_vllm = SimpleNamespace(LLM=original_llm)
    monkeypatch.setitem(__import__("sys").modules, "vllm", fake_vllm)

    def factory(**kwargs: object) -> tuple[dict[str, object], object]:
        from vllm import LLM

        return kwargs, LLM(
            model="model",
            max_num_seqs=8,
            limit_mm_per_prompt={"image": 24},
        )

    wrapped = inject_vllm_engine_options_from_factory_kwargs(factory)
    remaining, _ = wrapped(
        limit_mm_per_prompt={"image": 24, "video": 0},
        max_model_len=65536,
        mm_encoder_attn_backend="TORCH_SDPA",
        max_new_tokens=40960,
    )

    assert remaining == {"max_new_tokens": 40960}
    assert calls == [
        {
            "model": "model",
            "max_num_seqs": 8,
            "limit_mm_per_prompt": {"image": 24, "video": 0},
            "max_model_len": 65536,
            "mm_encoder_attn_backend": "TORCH_SDPA",
        }
    ]
    assert fake_vllm.LLM is original_llm


def test_nested_vllm_spawn_does_not_inherit_torchrun_rank_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "2",
        "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    def factory(value: int) -> tuple[int, dict[str, str | None]]:
        return value, {key: os.environ.get(key) for key in expected}

    wrapped = isolate_torchrun_environment_for_spawned_factory(factory)
    value, inside = wrapped(7)

    assert value == 7
    assert inside == {key: None for key in expected}
    assert {key: os.environ.get(key) for key in expected} == expected


def test_coredev_subset_config_preserves_model_and_complete_slice_identity(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    base_path = repository_root / "configs/evaluation/coredev_2511_qwen3_direct_v1.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))

    resolved_path = materialize_coredev_subset_config(
        base_config_path=base_path,
        output_dir=tmp_path,
        datasets=("VStarBench", "HRBench4K"),
    )
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    assert resolved["model"] == base["model"]
    assert resolved["data"] == {
        name: base["data"][name] for name in ("VStarBench", "HRBench4K")
    }
    assert materialize_coredev_subset_config(
        base_config_path=base_path,
        output_dir=tmp_path,
        datasets=("VStarBench", "HRBench4K"),
    ) == resolved_path


def test_coredev_subset_config_rejects_unknown_or_reordered_slices(
    tmp_path: Path,
) -> None:
    base_path = (
        Path(__file__).resolve().parents[2]
        / "configs/evaluation/coredev_2511_qwen3_direct_v1.json"
    )
    with pytest.raises(ValueError, match="unknown CoreDev"):
        materialize_coredev_subset_config(
            base_config_path=base_path,
            output_dir=tmp_path,
            datasets=("NotABenchmark",),
        )
    with pytest.raises(ValueError, match="canonical suite order"):
        materialize_coredev_subset_config(
            base_config_path=base_path,
            output_dir=tmp_path,
            datasets=("HRBench4K", "VStarBench"),
        )
