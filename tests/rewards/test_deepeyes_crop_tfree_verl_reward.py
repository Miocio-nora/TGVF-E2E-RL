from __future__ import annotations

import asyncio
import json

from omegaconf import OmegaConf
import pytest
from verl.trainer.ppo.reward import resolve_reward_manager_cls

from tests.rewards.test_deepeyes_verl_reward import (
    _Tokenizer,
    _data,
    _response,
    _service,
)
from tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward import (
    DEEPEYES_CROP_TFREE_VERL_REWARD_SCHEMA,
    DeepEyesCropTFreeRewardManager,
)
from tgvf_rl.rewards.deepeyes_verl_reward import AsyncDeepEyesOpenRouterJudge
from tgvf_rl.rewards.stage3_shaped import STAGE3_SHAPED_REWARD_VERSION


def _manager(
    response: str,
    *,
    verdict: str = "1",
    math_verify=lambda _reference, _answer: False,
) -> DeepEyesCropTFreeRewardManager:
    async def judge(_request: object, _payload: object) -> object:
        return _response(verdict)

    return DeepEyesCropTFreeRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer(response),
        judge_transport=AsyncDeepEyesOpenRouterJudge(_service(), request_json=judge),
        math_verify=math_verify,
        trajectory_id_factory=lambda: "crop-tfree",
    )


def _components(extra: dict[str, object]) -> dict[str, float]:
    return {
        item["name"]: float(item["score"])
        for item in json.loads(str(extra["stage3_reward_components"]))
    }


def test_correct_direct_crop_row_gets_answer_only_without_positive_tool_bonus() -> None:
    manager = _manager("<think>done</think>blue")

    result = asyncio.run(manager.run_single(_data(source="vstar", crop_count=0)))
    extra = result["reward_extra_info"]

    assert result["reward_score"] == pytest.approx(2.0)
    assert extra["score"] == pytest.approx(2.0)
    assert extra["conditional_tool"] == 0
    assert extra["reward_profile"] == STAGE3_SHAPED_REWARD_VERSION
    assert extra["stage3_reward_schema"] == DEEPEYES_CROP_TFREE_VERL_REWARD_SCHEMA
    assert _components(extra) == {
        "answer": 2.0,
        "tool": 0.0,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": 0.0,
    }
    assert extra["judge_route"] == "qwen2.5_72b_every_visual_trajectory"
    assert extra["visual_judge_requested"] == 1


def test_repeated_calls_and_crop_error_use_exact_tfree_equation() -> None:
    manager = _manager("<think>done</think>blue")
    data = _data(
        source="vstar",
        crop_count=1,
        call_count=3,
        error_count=1,
        observation_spans=[[0, 1], [1, 2], [2, 3]],
    )

    result = asyncio.run(manager.run_single(data))
    extra = result["reward_extra_info"]

    assert result["reward_score"] == pytest.approx(0.9)
    assert _components(extra) == {
        "answer": 2.0,
        "tool": -0.1,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": -1.0,
    }
    assert json.loads(str(extra["stage3_protocol_errors"])) == ["crop_tool_error"]
    assert extra["stage3_tool_reward"] == pytest.approx(-0.1)
    assert extra["stage3_protocol_reward"] == pytest.approx(-1.0)


def test_bad_final_protocol_penalizes_but_does_not_erase_correct_answer() -> None:
    manager = _manager("<think>unterminated blue")

    result = asyncio.run(manager.run_single(_data(source="arxivqa", crop_count=0)))
    extra = result["reward_extra_info"]

    assert extra["acc"] == 1
    assert extra["format_penalty"] == -1
    assert result["reward_score"] == pytest.approx(1.0)
    assert _components(extra)["answer"] == pytest.approx(2.0)
    assert _components(extra)["protocol"] == pytest.approx(-1.0)
    assert json.loads(str(extra["stage3_protocol_errors"])) == ["protocol_invalid"]


def test_parser_or_dispatch_error_is_not_lost_from_native_crop_audit() -> None:
    manager = _manager("<think>done</think>blue")
    data = _data(
        source="vstar",
        crop_count=1,
        call_count=1,
        error_count=0,
        observation_spans=[[0, 1], [1, 2]],
    )

    result = asyncio.run(manager.run_single(data))
    extra = result["reward_extra_info"]

    assert result["reward_score"] == pytest.approx(0.95)
    assert _components(extra)["tool"] == pytest.approx(-0.05)
    assert _components(extra)["protocol"] == pytest.approx(-1.0)
    assert extra["stage3_tool_attempt_count"] == 2
    assert json.loads(str(extra["stage3_protocol_errors"])) == [
        "tool_parse_or_dispatch_error"
    ]


def test_thinklite_keeps_math_verifier_route_and_uses_same_tfree_kernel() -> None:
    manager = _manager(
        "<think>done</think>\\boxed{blue}",
        math_verify=lambda reference, answer: reference == answer == "blue",
    )
    data = _data(source="thinklite", crop_count=0)
    data.non_tensor_batch["extra_info"][0]["task_kind"] = "math"

    result = asyncio.run(manager.run_single(data))
    extra = result["reward_extra_info"]

    assert result["reward_score"] == pytest.approx(2.0)
    assert extra["judge_route"] == "math_verify"
    assert extra["judge_requested"] == 0
    assert _components(extra) == {
        "answer": 2.0,
        "tool": 0.0,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": 0.0,
    }


def test_wrong_answer_with_successful_crop_gets_no_crop_bonus() -> None:
    manager = _manager("<think>done</think>red", verdict="0")

    result = asyncio.run(manager.run_single(_data(source="vstar", crop_count=1)))
    extra = result["reward_extra_info"]

    assert result["reward_score"] == pytest.approx(0.0)
    assert extra["acc"] == 0
    assert extra["successful_crop_count"] == 1
    assert _components(extra) == {
        "answer": 0.0,
        "tool": 0.0,
        "focus": 0.0,
        "grounding": 0.0,
        "protocol": 0.0,
    }


def test_importlib_resolves_crop_tfree_reward_manager() -> None:
    config = OmegaConf.create(
        {
            "reward": {
                "reward_manager": {
                    "source": "importlib",
                    "name": "DeepEyesCropTFreeRewardManager",
                    "module": {
                        "path": (
                            "pkg://tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward"
                        )
                    },
                }
            }
        }
    )

    assert resolve_reward_manager_cls(config) is DeepEyesCropTFreeRewardManager
