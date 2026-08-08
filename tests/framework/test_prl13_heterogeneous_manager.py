from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tgvf_rl.framework.verl.native_crop_tool import (
    ensure_native_crop_audit_fields,
)
from tgvf_rl.framework.verl.native_deepeyes_manager import (
    normalize_prl13_worker_output_columns,
)


def _internal_output(*, visual: bool, source: str) -> SimpleNamespace:
    prompt_ids = torch.tensor([[11, 12]], dtype=torch.long)
    response_ids = torch.tensor([[21, 22, 0]], dtype=torch.long)
    input_ids = torch.tensor([[11, 12, 21, 22, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    reward_extra_info: dict[str, object] = {
        "reward": 1.0,
        "source": source,
        "crop_boxes": "[]",
        "crop_best_call_iou": math.nan,
    }
    if visual:
        # The real PRL13 visual reward audit includes this native-loop field;
        # ThinkLite has no native visual response counter.  This is the exact
        # source-specific reward metadata split exposed by attempt10.
        reward_extra_info["response_token_count"] = 2
    extra_fields: dict[str, object] = {
        "global_steps": 1,
        # Pinned sync vLLM produces only global_steps.  AgentLoopWorker adds
        # these compatibility columns with null values.
        "min_global_steps": None,
        "max_global_steps": None,
        "turn_scores": [],
        "tool_rewards": [],
        "raw_prompt": [{"role": "user", "content": source}],
        "reward_extra_info": reward_extra_info,
    }
    # Pinned AgentLoopWorker._compute_multi_modal_inputs returns an empty dict
    # (not None) for a text-only sample when a Qwen processor is installed.
    multi_modal_inputs: dict[str, object] = {}
    if visual:
        ensure_native_crop_audit_fields(extra_fields)
        extra_fields.update(
            {
                "native_original_image_count": 1,
                "native_crop_image_count": 0,
                "native_total_image_count": 1,
                "response_token_count": 2,
                "native_pixels_proven": True,
                "legacy_adapter_loaded": False,
                "observation_role": "user",
                "observation_envelope": (
                    "<tool_response><image>...</tool_response>"
                ),
            }
        )
        multi_modal_inputs = {
            "pixel_values": torch.ones((1, 4), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        input_ids=input_ids,
        position_ids=torch.arange(5, dtype=torch.long).unsqueeze(0),
        response_logprobs=None,
        routed_experts=None,
        teacher_logprobs=None,
        teacher_ids=None,
        reward_score=1.0,
        num_turns=2,
        metrics=SimpleNamespace(
            model_dump=lambda: {
                "generate_sequences": 1.0,
                "tool_calls": 0.0,
                "compute_score": 1.0,
                "num_preempted": 0,
            }
        ),
        multi_modal_inputs=multi_modal_inputs,
        extra_fields=extra_fields,
    )


def _exact_smoke_worker_chunks() -> list[object]:
    """Run pinned AgentLoopWorker postprocess on the exact 4x2 source shape."""

    pytest.importorskip("verl")
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    # A live PRL13 worker computes reward inside the agent loop, so the input
    # row columns are intentionally not copied into its generated DataProto.
    worker.reward_loop_worker_handles = [object()]
    source_chunks = (
        ("vstar", "vstar"),
        ("arxivqa", "arxivqa"),
        ("thinklite", "thinklite"),
        ("vstar", "vstar"),
    )
    return [
        worker._postprocess(
            [
                _internal_output(
                    visual=source != "thinklite",
                    source=source,
                )
                for source in sources
            ],
            input_non_tensor_batch={},
            validate=False,
        )
        for sources in source_chunks
    ]


def test_exact_four_worker_smoke_schema_is_rectangularized_losslessly() -> None:
    from verl.protocol import DataProto

    chunks = _exact_smoke_worker_chunks()
    visual_keys = set(chunks[0].non_tensor_batch)
    thinklite_keys = set(chunks[2].non_tensor_batch)
    assert thinklite_keys - visual_keys == set()
    assert visual_keys - thinklite_keys == {
        "crop_call_count",
        "crop_action_count",
        "crop_model_boxes",
        "crop_source_boxes",
        "crop_area_fractions",
        "crop_first_call_iou",
        "crop_best_gt_coverage",
        "crop_error_count",
        "crop_observation_token_spans",
        "crop_coordinate_space",
        "crop_coordinate_conversion_version",
        "crop_coordinate_reference_size",
        "native_original_image_count",
        "native_crop_image_count",
        "native_total_image_count",
        "response_token_count",
        "native_pixels_proven",
        "legacy_adapter_loaded",
        "observation_role",
        "observation_envelope",
    }

    # attempt10 first fails because source-homogeneous workers declare
    # different reward-extra schemas.
    with pytest.raises(AssertionError, match="reward_extra_keys"):
        DataProto.concat(chunks)

    # Even if metadata were manually made identical, the old implementation
    # still formed six-row visual-only columns beside an eight-row tensor
    # batch.  Keep this independent regression so neither half of the fix can
    # mask the other.
    schema_only_chunks = _exact_smoke_worker_chunks()
    for chunk in schema_only_chunks:
        chunk.meta_info["reward_extra_keys"] = ["crop_boxes", "reward", "source"]
    with pytest.raises(AssertionError, match="length 6 is not equal to batch size 8"):
        DataProto.concat(schema_only_chunks)

    normalize_prl13_worker_output_columns(chunks)
    union = set.union(*(set(chunk.non_tensor_batch) for chunk in chunks))
    assert all(set(chunk.non_tensor_batch) == union for chunk in chunks)
    assert all(
        values.shape[0] == len(chunk)
        for chunk in chunks
        for values in chunk.non_tensor_batch.values()
    )

    combined = DataProto.concat(chunks)
    assert len(combined) == 8
    assert all(values.shape[0] == 8 for values in combined.non_tensor_batch.values())
    assert combined.non_tensor_batch["global_steps"].tolist() == [1] * 8
    assert combined.meta_info["reward_extra_keys"] == sorted(
        {
            "crop_best_call_iou",
            "crop_boxes",
            "response_token_count",
            "reward",
            "source",
        }
    )
    # Pinned AgentLoopWorker writes the raw tool audit after the reward
    # columns.  PRL13 must restore the reward manager's batch-safe canonical
    # values: JSON text instead of ragged lists, and NaN instead of null.
    assert combined.non_tensor_batch["crop_boxes"].tolist() == ["[]"] * 8
    assert all(
        math.isnan(value)
        for value in combined.non_tensor_batch["crop_best_call_iou"].tolist()
    )

    from verl.trainer.ppo.reward import extract_reward

    reward_tensor, reward_extra = extract_reward(combined)
    assert reward_tensor.shape[0] == 8
    assert set(reward_extra) == set(combined.meta_info["reward_extra_keys"])
    assert all(values.shape[0] == 8 for values in reward_extra.values())
    assert reward_extra["response_token_count"].tolist()[4:6] == [None, None]

    from verl.trainer.ppo.metric_utils import process_validation_metrics

    collision_metrics = process_validation_metrics(
        ["visual"] * 8,
        ["same-prompt"] * 8,
        {
            "crop_best_call_iou": reward_extra["crop_best_call_iou"].tolist(),
            "crop_boxes": reward_extra["crop_boxes"].tolist(),
            "reward": reward_extra["reward"].tolist(),
        },
    )
    assert math.isnan(
        collision_metrics["visual"]["crop_best_call_iou"]["mean@8"]
    )

    # Real text-only Qwen preprocessing is retained as an empty mapping.  No
    # dummy pixel/grid tensors are introduced by schema normalization.
    multimodal = combined.non_tensor_batch["multi_modal_inputs"].tolist()
    assert multimodal[4:6] == [{}, {}]
    assert all(
        isinstance(multimodal[index], dict)
        for index in (0, 1, 2, 3, 6, 7)
    )
    assert combined.non_tensor_batch["native_pixels_proven"].tolist()[4:6] == [
        None,
        None,
    ]


def test_policy_version_columns_remain_fail_closed() -> None:
    chunks = _exact_smoke_worker_chunks()
    del chunks[0].non_tensor_batch["global_steps"]
    with pytest.raises(RuntimeError, match="omitted required policy column"):
        normalize_prl13_worker_output_columns(chunks)


def test_policy_version_disagreement_is_not_normalized() -> None:
    chunks = _exact_smoke_worker_chunks()
    chunks[1].non_tensor_batch["max_global_steps"] = np.array(
        [2, 2], dtype=object
    )
    with pytest.raises(RuntimeError, match="policy-version columns disagree"):
        normalize_prl13_worker_output_columns(chunks)


def test_reward_metadata_missing_named_column_remains_fail_closed() -> None:
    chunks = _exact_smoke_worker_chunks()
    chunks[0].meta_info["reward_extra_keys"].append("not_a_real_column")
    with pytest.raises(RuntimeError, match="reward metadata names missing columns"):
        normalize_prl13_worker_output_columns(chunks)


def test_reward_manager_null_metric_is_rejected_before_validation() -> None:
    chunks = _exact_smoke_worker_chunks()
    chunks[0].non_tensor_batch["reward_extra_info"][0][
        "crop_best_call_iou"
    ] = None
    with pytest.raises(RuntimeError, match="optional numeric metrics must use NaN"):
        normalize_prl13_worker_output_columns(chunks)


def test_reward_metadata_requires_rm_scores_on_every_worker() -> None:
    chunks = _exact_smoke_worker_chunks()
    chunks[2].batch.pop("rm_scores")
    chunks[2].meta_info.pop("reward_extra_keys")
    with pytest.raises(RuntimeError, match="presence of rm_scores"):
        normalize_prl13_worker_output_columns(chunks)
