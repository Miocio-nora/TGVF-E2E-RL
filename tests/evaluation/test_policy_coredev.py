from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tgvf_rl.evaluation.policy_coredev as policy_coredev
from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    PolicyCoreDevEvaluator,
    StandaloneTGVFVLLMManager,
    _TurnRoute,
    _termination_contract,
    load_benchmark_tasks,
    load_coredev_tasks,
    load_policy_coredev_config,
    policy_version_from_pointer,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFCropTGVFMaterializationResult,
    _crop_tgvf_to_utility_wire,
)
from tgvf_rl.framework.vllm import VLLMTerminationOutcome
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
)
from tests.framework.test_verl_smoke_dataset_prompt import _SourceImageProcessor
from tests.framework.test_vllm_tool_runtime import _crop_tgvf_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_formal_policy_configs_bind_exact_step80_snapshots() -> None:
    expected = {
        "coredev_2511_tgvf_step80_v1.json": (
            "PRL-02-R5-QWEN3-GRPO-BS16-TGVF-T1-FORMAL-PILOT-80STEP-GPU0123",
            "561132e49848fd43f8e7f352ef54782249aff59b2a5d331027a0e5e0f78be321",
        ),
        "coredev_2511_crop_step80_v1.json": (
            "PRL-03-R2-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123",
            "eed4ffeaf5b77277a41dafeba428a20d5f3c8bce73049c02e63f63292d78b0b0",
        ),
    }
    for name, (run_id, weights_sha256) in expected.items():
        config = load_policy_coredev_config(
            REPOSITORY_ROOT / "configs/evaluation" / name
        )
        assert config.inference_concurrency_per_gpu == 8
        version = policy_version_from_pointer(config)
        assert version.run_id == run_id
        assert version.optimizer_step == 80
        assert version.weights_sha256 == weights_sha256


def test_policy_evaluation_accepts_native_vllm_eos_identity() -> None:
    run = load_policy_e2e_smoke_run_config(
        REPOSITORY_ROOT
        / "configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml"
    )

    outcomes = _termination_contract(run).final_turn_outcomes
    assert VLLMTerminationOutcome("stop", None) in outcomes
    assert VLLMTerminationOutcome("stop", 151_643) in outcomes
    assert VLLMTerminationOutcome("stop", 151_644) not in outcomes


def test_coredev_task_loader_keeps_order_and_single_image_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "ordinal": index,
            "dataset": "fixture",
            "row_number": index,
            "index": str(index),
            "question": "question",
            "image_paths": ["a.jpg"] if index != 3 else ["a.jpg", "b.jpg"],
        }
        for index in range(2511)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tasks = load_coredev_tasks(path)
    assert isinstance(tasks[0], CoreDevTask)
    assert tasks[0].single_image is True
    assert tasks[3].single_image is False
    assert tasks[-1].ordinal == 2510


def test_benchmark_loader_verifies_each_unique_bound_canvas_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "shared.png"
    image_path.write_bytes(b"verification is patched")
    identities = (
        ("a" * 64, [32, 24]),
        ("a" * 64, [32, 24]),
        ("b" * 64, [32, 24]),
    )
    rows = [
        {
            "ordinal": ordinal,
            "dataset": "fixture",
            "row_number": ordinal,
            "index": f"sample-{ordinal}",
            "sample_id": f"sample-{ordinal}",
            "question": "question",
            "image_paths": [str(image_path)],
            "image_sha256s": [digest],
            "image_dimensions": [dimensions],
        }
        for ordinal, (digest, dimensions) in enumerate(identities)
    ]
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    verified: list[tuple[Path, str, tuple[int, int]]] = []

    def verify(task: CoreDevTask, image_index: int = 0) -> torch.Tensor:
        verified.append(
            (
                Path(task.image_paths[image_index]),
                task.image_sha256s[image_index],
                task.image_dimensions[image_index],
            )
        )
        return torch.zeros((24, 32, 3), dtype=torch.uint8)

    monkeypatch.setattr(policy_coredev, "load_verified_task_image", verify)

    tasks = load_benchmark_tasks(
        manifest,
        expected_task_count=3,
        expected_single_image_count=3,
    )

    assert len(tasks) == 3
    assert verified == [
        (image_path, "a" * 64, (32, 24)),
        (image_path, "b" * 64, (32, 24)),
    ]


def test_standalone_manager_forwards_atomic_crop_tgvf_as_one_collective() -> None:
    captured: dict[str, object] = {}

    class Engine:
        async def collective_rpc(self, method, *, kwargs):
            captured["method"] = method
            captured["kwargs"] = kwargs
            return [_crop_tgvf_to_utility_wire(_crop_tgvf_result())]

    manager = StandaloneTGVFVLLMManager(
        Engine(),
        None,
        capture_hidden=True,
    )
    manager.turns["trajectory-0"] = _TurnRoute(
        backend_request_id="backend-0",
        output_ids=(11, 12, 13),
        optimizer_step=5,
    )

    result = asyncio.run(
        manager.materialize_crop_tgvf(
            request_id="trajectory-0",
            expected_step=5,
            sampled_output_ids=(11, 12, 13),
            call_index=2,
            pixel_values=torch.ones((4, 6)),
            image_grid_thw=torch.tensor(((1, 2, 2),)),
            source_image_sha256="a" * 64,
            crop_rgb_sha256="c" * 64,
            source_width=100,
            source_height=80,
            crop_bbox=(10, 20, 42, 68),
            crop_width=32,
            crop_height=48,
            target_start=0,
            target_end=2,
            expected_target_token_ids=(11, 12),
            provider="contextual_hidden_state",
        )
    )

    assert isinstance(result, TGVFCropTGVFMaterializationResult)
    assert captured["method"] == "tgvf_materialize_crop_tgvf"
    kwargs = captured["kwargs"]
    assert kwargs["backend_request_id"] == "backend-0"
    assert kwargs["crop_bbox"] == (10, 20, 42, 68)
    assert kwargs["source_image_sha256"] == "a" * 64
    assert kwargs["crop_rgb_sha256"] == "c" * 64


def test_generic_evaluator_renders_from_the_verified_rgb_snapshot() -> None:
    processor = _SourceImageProcessor()
    profile = NativeToolCapabilityProfile.TGVF_ONLY
    dialect = NativeAssistantDialect.QWEN3_VL_THINKING
    evaluator = object.__new__(PolicyCoreDevEvaluator)
    evaluator.processor = processor
    evaluator.run = SimpleNamespace(protocol=SimpleNamespace(tool_profile=profile))
    evaluator.config = SimpleNamespace(
        effective_image_max_pixels=lambda _run: 512 * 512
    )
    evaluator.assistant_dialect = dialect
    evaluator.renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=len(processor.tokenizer),
        tool_names=profile.tool_names,
        tool_schemas=build_native_tool_schemas(profile.tool_names),
        assistant_dialect=dialect,
    )
    task = CoreDevTask(
        ordinal=0,
        dataset="fixture",
        row_number=0,
        index="sample-0",
        sample_id="sample-0",
        question="What texture is shown?",
        image_paths=("/not-read-by-render.png",),
    )

    prompt_ids = evaluator.render_initial_prompt(
        task,
        source_rgb=torch.zeros((8, 16, 3), dtype=torch.uint8),
    )

    assert prompt_ids.count(processor.tokenizer.image_token_id) == 4
