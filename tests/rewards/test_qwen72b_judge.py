from __future__ import annotations

import json

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import (
    JudgeRequest,
    OpenAICompatibleJudgeConfig,
    OpenAICompatibleJudgeProvider,
    QWEN25_72B_RL_JUDGE_PROMPT_VERSION,
)


def _identity(name: str, digit: str, *, version: str = "v1") -> ArtifactIdentity:
    return ArtifactIdentity("rl-judge-test", name, version, digit * 64)


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _provider(content: str, captured: list[dict[str, object]]):
    prompt = _identity(
        "prompt", "1", version=QWEN25_72B_RL_JUDGE_PROMPT_VERSION
    )

    def opener(request, *, timeout):
        assert timeout == 120.0
        captured.append(json.loads(request.data))
        return _Response(
            {"choices": [{"message": {"content": content}}]}
        )

    config = OpenAICompatibleJudgeConfig(
        base_url="http://127.0.0.1:8013/v1",
        model_name="Qwen2.5-72B-Instruct",
        prompt_identity=prompt,
        service_identity=_identity("service", "2"),
        model_identity=_identity("model", "3"),
        sampling_identity=_identity("sampling", "4"),
        calibration_identity=_identity("calibration", "5"),
    )
    return OpenAICompatibleJudgeProvider(config, opener=opener), prompt


def test_rl_judge_sends_task_kind_and_accepts_only_binary_json() -> None:
    captured: list[dict[str, object]] = []
    provider, prompt = _provider(
        '{"verdict":1,"rationale":"equivalent value"}', captured
    )
    result = provider.judge(
        JudgeRequest(
            request_id="request-1",
            task_kind="math",
            question="What is one half?",
            candidate_answer="0.5",
            reference_answer="1/2",
            prompt_identity=prompt,
        )
    )

    assert result.score == 1.0
    user_payload = json.loads(captured[0]["messages"][1]["content"])
    assert user_payload["task_kind"] == "math"
    assert captured[0]["temperature"] == 0.0
    assert captured[0]["response_format"] == {"type": "json_object"}


def test_rl_judge_fails_closed_on_nonbinary_or_malformed_output() -> None:
    for content in ('{"verdict":0.5,"rationale":"uncertain"}', "yes"):
        provider, prompt = _provider(content, [])
        with pytest.raises(RuntimeError):
            provider.judge(
                JudgeRequest(
                    request_id="request-2",
                    task_kind="open_vqa",
                    question="What is shown?",
                    candidate_answer="a ship",
                    reference_answer="ship",
                    prompt_identity=prompt,
                )
            )
