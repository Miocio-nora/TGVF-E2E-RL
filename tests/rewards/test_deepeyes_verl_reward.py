from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from tensordict import TensorDict

from tgvf_rl.protocol.action_boundary import NativeActionBoundaryProtocolId
from tgvf_rl.rewards.deepeyes_batch import JudgeGlobalFailure
from tgvf_rl.rewards.deepeyes_official import (
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    DeepEyesBinaryJudgeRequest,
)
from tgvf_rl.rewards.deepeyes_verl_reward import (
    AsyncDeepEyesOpenRouterJudge,
    DEEPEYES_VERL_AUDIT_SEQUENCE_ENCODING,
    DeepEyesJudgeServiceConfig,
    DeepEyesOfficialRewardManager,
    load_deepeyes_judge_service_config,
)


class _Tokenizer:
    def __init__(self, response: str) -> None:
        self.response = response

    def decode(self, _ids: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.response


def _service(
    *,
    concurrency: int = 2,
    attempts: int = 2,
    failure_window: int = 100,
    failure_fraction: float = 0.01,
) -> DeepEyesJudgeServiceConfig:
    return DeepEyesJudgeServiceConfig(
        file_sha256="a" * 64,
        model="qwen/qwen-2.5-72b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        provider_only=("deepinfra",),
        allow_fallbacks=False,
        timeout_seconds=120.0,
        maximum_concurrency=concurrency,
        maximum_attempts=attempts,
        retry_backoff_seconds=0.0,
        retry_maximum_seconds=0.0,
        cache_max_entries=100,
        transient_failure_window_size=failure_window,
        maximum_transient_failure_fraction=failure_fraction,
    )


def _request(trajectory_id: str) -> DeepEyesBinaryJudgeRequest:
    return DeepEyesBinaryJudgeRequest.build(
        trajectory_id=trajectory_id,
        sample_id="sample",
        question="What color?",
        reference_answer="blue",
        candidate_answer="blue",
        task_kind="open",
        prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    )


def _response(content: str = "1") -> dict[str, object]:
    return {
        "model": "qwen/qwen-2.5-72b-instruct",
        "choices": [{"message": {"content": content}}],
    }


def test_async_transport_bounds_concurrency_retries_and_cache() -> None:
    active = 0
    maximum = 0
    attempts: dict[str, int] = {}

    async def request_json(request: object, payload: object) -> object:
        nonlocal active, maximum
        assert payload["provider"] == {
            "only": ["deepinfra"],
            "allow_fallbacks": False,
        }
        attempts[request.request_id] = attempts.get(request.request_id, 0) + 1
        if request.trajectory_id == "retry" and attempts[request.request_id] == 1:
            raise TimeoutError("temporary")
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _response()

    async def exercise() -> None:
        judge = AsyncDeepEyesOpenRouterJudge(_service(), request_json=request_json)
        outcomes = await asyncio.gather(
            *(judge.judge(_request(name)) for name in ("a", "b", "c", "retry"))
        )
        assert maximum == 2
        assert outcomes[-1].calls == 2
        assert outcomes[-1].retries == 1
        cached = await judge.judge(_request("a"))
        assert cached.cache_hit == 1
        assert cached.calls == 0

    asyncio.run(exercise())


def test_async_transport_isolates_completed_output_but_aborts_model_mismatch() -> None:
    async def malformed(_request: object, _payload: object) -> object:
        return _response("maybe")

    outcome = asyncio.run(
        AsyncDeepEyesOpenRouterJudge(_service(), request_json=malformed).judge(
            _request("bad")
        )
    )
    assert outcome.verdict is False
    assert outcome.failure_kind == "completed_invalid_output"

    async def wrong_model(_request: object, _payload: object) -> object:
        value = _response()
        value["model"] = "wrong/model"
        return value

    with pytest.raises(JudgeGlobalFailure, match="model differs"):
        asyncio.run(
            AsyncDeepEyesOpenRouterJudge(_service(), request_json=wrong_model).judge(
                _request("global")
            )
        )


def test_async_transport_aborts_when_transient_failure_window_is_exceeded() -> None:
    async def unavailable(_request: object, _payload: object) -> object:
        raise TimeoutError("temporary")

    async def exercise() -> None:
        judge = AsyncDeepEyesOpenRouterJudge(
            _service(
                attempts=1,
                failure_window=2,
                failure_fraction=0.5,
            ),
            request_json=unavailable,
        )
        first = await judge.judge(_request("first-transient"))
        assert first.failure_kind == "transient_exhausted"
        with pytest.raises(JudgeGlobalFailure, match="bounded window"):
            await judge.judge(_request("second-transient"))

    asyncio.run(exercise())


def _data(
    *,
    source: str,
    crop_count: int,
    call_count: int | None = None,
    error_count: int = 0,
    observation_spans: list[list[int]] | None = None,
    action_boundary_protocol_id: NativeActionBoundaryProtocolId = (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    ),
    action_boundary_violation_codes: list[str] | None = None,
) -> object:
    data_proto = pytest.importorskip(
        "verl",
        reason="DeepEyes reward DataProto integration requires optional pinned veRL",
    ).DataProto
    call_count = crop_count if call_count is None else call_count
    if observation_spans is None:
        observation_spans = [[1, 2]] if crop_count else []
    if action_boundary_violation_codes is None:
        action_boundary_violation_codes = []
    batch = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        },
        batch_size=1,
    )
    return data_proto(
        batch=batch,
        non_tensor_batch={
            "data_source": np.array([source], dtype=object),
            "reward_model": np.array([{"ground_truth": "blue"}], dtype=object),
            "extra_info": np.array(
                [
                    {
                        "sample_id": "sample",
                        "question": "What color?",
                        "task_kind": "open",
                    }
                ],
                dtype=object,
            ),
            "tool_extra_fields": np.array(
                [
                    {
                        "crop_call_count": call_count,
                        "crop_action_count": crop_count,
                        "crop_boxes": [[10, 10, 50, 50]] if crop_count else [],
                        "crop_area_fractions": [0.16] if crop_count else [],
                        "crop_first_call_iou": 0.5 if crop_count else None,
                        "crop_best_call_iou": 0.5 if crop_count else None,
                        "crop_best_gt_coverage": 0.75 if crop_count else None,
                        "crop_error_count": error_count,
                        "crop_observation_token_spans": observation_spans,
                        "native_original_image_count": 1,
                        "native_crop_image_count": crop_count,
                        "native_total_image_count": 1 + crop_count,
                        "response_token_count": 3,
                        "native_pixels_proven": True,
                        "legacy_adapter_loaded": False,
                        "observation_role": "user",
                        "observation_envelope": "native_multimodal",
                        "action_boundary_protocol_id": (
                            action_boundary_protocol_id.value
                        ),
                        "action_boundary_violation_count": len(
                            action_boundary_violation_codes
                        ),
                        "action_boundary_violation_codes": (
                            action_boundary_violation_codes
                        ),
                    }
                ],
                dtype=object,
            ),
        },
    )


def test_real_dataproto_manager_keeps_accuracy_independent_of_format() -> None:
    async def correct(_request: object, _payload: object) -> object:
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("<think>unterminated blue"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(_service(), request_json=correct),
        trajectory_id_factory=lambda: "rollout-0",
    )
    result = asyncio.run(manager.run_single(_data(source="vstar", crop_count=1)))
    assert result["reward_score"] == pytest.approx(1.8)
    extra = result["reward_extra_info"]
    assert extra["acc"] == 1
    assert extra["format_penalty"] == -1
    assert extra["conditional_tool"] == 1
    assert extra["visual_judge_requested"] == 1
    assert extra["trajectory_id"] == "sample/rollout-0"
    assert extra["source"] == "vstar"
    assert extra["response_length"] == 3
    assert extra["action_length"] == 3
    assert extra["first_iou"] == pytest.approx(0.5)
    assert extra["best_iou"] == pytest.approx(0.5)
    assert extra["gt_coverage"] == pytest.approx(0.75)
    assert json.loads(extra["crop_area"]) == [0.16]
    assert extra["audit_sequence_encoding"] == DEEPEYES_VERL_AUDIT_SEQUENCE_ENCODING
    assert extra["action_count"] == 1


def test_strict_invalid_action_hard_gate_is_dependency_free() -> None:
    requests: list[DeepEyesBinaryJudgeRequest] = []

    async def incorrectly_correct(request: object, _payload: object) -> object:
        assert isinstance(request, DeepEyesBinaryJudgeRequest)
        requests.append(request)
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("unused"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(
            _service(), request_json=incorrectly_correct
        ),
        trajectory_id_factory=lambda: "invalid-boundary",
    )

    result = asyncio.run(
        manager._score_visual(
            trajectory_id="invalid-boundary",
            sample_id="sample",
            question="What color?",
            reference="blue",
            response=(
                '<tool_call>{"name":"image_zoom_in_tool",'
                '"arguments":BROKEN}</tool_call>blue'
            ),
            task_kind="open",
            crop_count=0,
            audit={
                "action_boundary_protocol_id": (
                    NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value
                ),
                "action_boundary_violation_count": 1,
                "action_boundary_violation_codes": [
                    "parsed_tool_call_count_mismatch"
                ],
            },
        )
    )

    assert len(requests) == 1
    assert requests[0].candidate_answer == "[NO VALID FINAL ANSWER]"
    assert "blue" not in requests[0].candidate_answer
    assert result["score"] == pytest.approx(-0.2)
    assert result["acc"] == 0
    assert result["conditional_tool"] == 0
    assert result["format_penalty"] == -1
    assert result["answer_length"] == 0
    assert result["judge_requested"] == 1
    assert result["visual_judge_requested"] == 1
    assert result["judge_calls"] == 1
    assert result["judge_route"].endswith(
        "strict_action_boundary_invalid_hard_gate"
    )


def test_strict_invalid_action_judge_failure_is_not_a_policy_penalty() -> None:
    async def invalid_output(_request: object, _payload: object) -> object:
        return _response("maybe")

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("unused"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(
            _service(), request_json=invalid_output
        ),
    )

    result = asyncio.run(
        manager._score_visual(
            trajectory_id="invalid-boundary-failed-judge",
            sample_id="sample",
            question="What color?",
            reference="blue",
            response="malformed action with a correct-looking blue suffix",
            task_kind="open",
            crop_count=0,
            audit={
                "action_boundary_protocol_id": (
                    NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value
                ),
                "action_boundary_violation_count": 1,
                "action_boundary_violation_codes": [
                    "malformed_tool_call_tags"
                ],
            },
        )
    )

    assert result["score"] == 0.0
    assert result["acc"] == 0
    assert result["conditional_tool"] == 0
    assert result["judge_requested"] == 1
    assert result["judge_calls"] == 1
    assert result["judge_failures"] == 1


def test_strict_invalid_action_dataproto_cannot_receive_semantic_reward() -> None:
    calls = 0

    async def incorrectly_correct(_request: object, _payload: object) -> object:
        nonlocal calls
        calls += 1
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer(
            '<think>done</think><tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":BROKEN}</tool_call>blue'
        ),
        judge_transport=AsyncDeepEyesOpenRouterJudge(
            _service(), request_json=incorrectly_correct
        ),
        trajectory_id_factory=lambda: "invalid-boundary",
    )

    result = asyncio.run(
        manager.run_single(
            _data(
                source="vstar",
                crop_count=0,
                action_boundary_violation_codes=[
                    "parsed_tool_call_count_mismatch"
                ],
            )
        )
    )

    assert calls == 1
    assert result["reward_score"] == pytest.approx(-0.2)
    extra = result["reward_extra_info"]
    assert extra["acc"] == 0
    assert extra["conditional_tool"] == 0
    assert extra["format_penalty"] == -1
    assert extra["answer_length"] == 0
    assert extra["judge_requested"] == 1
    assert extra["visual_judge_requested"] == 1
    assert extra["judge_route"].endswith(
        "strict_action_boundary_invalid_hard_gate"
    )
    assert extra["action_boundary_valid"] == 0
    assert json.loads(extra["action_boundary_violation_codes"]) == [
        "parsed_tool_call_count_mismatch"
    ]


@pytest.mark.parametrize(
    ("case", "data_kwargs", "expected"),
    (
        (
            "direct-no-call",
            {"source": "arxivqa", "crop_count": 0},
            (0, 0, 0, []),
        ),
        (
            "invalid-call",
            {
                "source": "vstar",
                "crop_count": 0,
                "call_count": 1,
                "error_count": 1,
                "observation_spans": [[1, 2]],
            },
            (1, 0, 1, [[1, 2]]),
        ),
        (
            "valid-call",
            {"source": "vstar", "crop_count": 1},
            (1, 1, 0, [[1, 2]]),
        ),
    ),
)
def test_visual_reward_accepts_direct_invalid_and_valid_audit_semantics(
    case: str, data_kwargs: dict[str, object], expected: tuple[object, ...]
) -> None:
    async def correct(_request: object, _payload: object) -> object:
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("<think>done</think>\\boxed{blue}"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(_service(), request_json=correct),
        trajectory_id_factory=lambda: case,
    )
    result = asyncio.run(manager.run_single(_data(**data_kwargs)))
    extra = result["reward_extra_info"]
    assert (
        extra["crop_call_count"],
        extra["crop_action_count"],
        extra["crop_error_count"],
        json.loads(extra["crop_observation_token_spans"]),
    ) == expected


def test_reward_extra_info_batches_direct_and_crop_trajectories() -> None:
    async def correct(_request: object, _payload: object) -> object:
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("<think>done</think>\\boxed{blue}"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(_service(), request_json=correct),
        trajectory_id_factory=lambda: "batch-safe",
    )
    direct = asyncio.run(manager.run_single(_data(source="vstar", crop_count=0)))[
        "reward_extra_info"
    ]
    cropped = asyncio.run(manager.run_single(_data(source="vstar", crop_count=1)))[
        "reward_extra_info"
    ]

    assert direct.keys() == cropped.keys()
    for name in direct:
        column = np.array([direct[name], cropped[name]])
        assert column.shape[0] == 2, name
    assert json.loads(direct["crop_boxes"]) == []
    assert json.loads(cropped["crop_boxes"]) == [[10, 10, 50, 50]]
    assert math.isnan(direct["crop_first_call_iou"])


def test_reward_extra_schema_is_source_stable_and_validation_numeric() -> None:
    process_validation_metrics = pytest.importorskip(
        "verl.trainer.ppo.metric_utils",
        reason="reward metric aggregation requires optional pinned veRL",
    ).process_validation_metrics

    async def correct(_request: object, _payload: object) -> object:
        return _response()

    manager = DeepEyesOfficialRewardManager(
        OmegaConf.create({"reward": {}}),
        _Tokenizer("<think>done</think>\\boxed{blue}"),
        judge_transport=AsyncDeepEyesOpenRouterJudge(_service(), request_json=correct),
        trajectory_id_factory=lambda: "source-stable",
    )
    visual = asyncio.run(manager.run_single(_data(source="vstar", crop_count=0)))[
        "reward_extra_info"
    ]
    thinklite = asyncio.run(
        manager.run_single(_data(source="thinklite", crop_count=0))
    )["reward_extra_info"]

    assert visual.keys() == thinklite.keys()
    assert visual["response_token_count"] == 3
    assert thinklite["response_token_count"] == 3
    assert all(value is not None for value in visual.values())
    assert all(value is not None for value in thinklite.values())
    for name in (
        "crop_first_call_iou",
        "crop_best_call_iou",
        "crop_best_gt_coverage",
        "first_iou",
        "best_iou",
        "gt_coverage",
    ):
        assert math.isnan(visual[name])
        assert math.isnan(thinklite[name])

    infos = {
        name: [visual[name], visual[name], thinklite[name], thinklite[name]]
        for name in visual
    }
    metrics = process_validation_metrics(
        ["vstar", "vstar", "thinklite", "thinklite"],
        ["visual", "visual", "text", "text"],
        infos,
    )
    assert metrics["vstar"]["acc"]["mean@2"] == pytest.approx(float(visual["acc"]))
    assert metrics["thinklite"]["acc"]["mean@2"] == pytest.approx(
        float(thinklite["acc"])
    )


def test_importlib_reward_manager_resolves_on_pinned_verl_path() -> None:
    resolve_reward_manager_cls = pytest.importorskip(
        "verl.trainer.ppo.reward",
        reason="reward-manager resolution requires optional pinned veRL",
    ).resolve_reward_manager_cls
    config = OmegaConf.create(
        {
            "reward": {
                "reward_manager": {
                    "source": "importlib",
                    "name": "DeepEyesOfficialRewardManager",
                    "module": {"path": "pkg://tgvf_rl.rewards.deepeyes_verl_reward"},
                }
            }
        }
    )
    assert resolve_reward_manager_cls(config) is DeepEyesOfficialRewardManager


def test_template_service_binding_loads_without_enabling_network() -> None:
    path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/prl13_qwen25_72b_binary_text_template.json"
    )
    digest = sha256(path.read_bytes()).hexdigest()
    loaded = load_deepeyes_judge_service_config(
        path,
        expected_file_sha256=digest,
        require_launch_enabled=False,
    )
    assert loaded.maximum_concurrency == 64
    with pytest.raises(ValueError, match="not launch-enabled"):
        load_deepeyes_judge_service_config(path, expected_file_sha256=digest)
