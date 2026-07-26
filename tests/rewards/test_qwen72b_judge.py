from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import (
    JudgeRequest,
    OpenAICompatibleJudgeConfig,
    OpenAICompatibleJudgeProvider,
    QWEN25_72B_RL_JUDGE_PROMPT_VERSION,
    load_openai_compatible_judge,
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


def test_local_judge_usage_without_provider_cost_is_retained_at_zero_cost() -> None:
    captured: list[dict[str, object]] = []
    prompt = _identity(
        "prompt", "1", version=QWEN25_72B_RL_JUDGE_PROMPT_VERSION
    )

    def opener(request, *, timeout):
        captured.append(json.loads(request.data))
        return _Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"verdict":1,"rationale":"same"}'
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    config = OpenAICompatibleJudgeConfig(
        base_url="http://127.0.0.1:8013/v1",
        model_name="Qwen2.5-72B-Instruct",
        prompt_identity=prompt,
        service_identity=_identity("service", "2"),
        model_identity=_identity("model", "3"),
        sampling_identity=_identity("sampling", "4"),
        calibration_identity=_identity("calibration", "5"),
        require_usage=False,
    )
    result = OpenAICompatibleJudgeProvider(config, opener=opener).judge(
        JudgeRequest(
            request_id="local-usage",
            task_kind="math",
            question="1+1?",
            candidate_answer="2",
            reference_answer="2",
            prompt_identity=prompt,
        )
    )
    assert result.usage is not None
    assert result.usage.cost_usd == 0.0


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


def test_rl_judge_retries_transient_rate_limit_with_bound() -> None:
    initial, prompt = _provider('{"verdict":1,"rationale":"ok"}', [])
    calls = 0
    delays: list[float] = []

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                BytesIO(b'{"error":{"message":"temporary"}}'),
            )
        return _Response(
            {"choices": [{"message": {"content": '{"verdict":1,"rationale":"ok"}'}}]}
        )

    provider = OpenAICompatibleJudgeProvider(
        replace(
            initial.config,
            maximum_attempts=2,
            retry_backoff_seconds=1.0,
        ),
        opener=opener,
        sleeper=delays.append,
    )
    result = provider.judge(
        JudgeRequest(
            request_id="request-retry",
            task_kind="math",
            question="1+1?",
            candidate_answer="2",
            reference_answer="2",
            prompt_identity=prompt,
        )
    )

    assert result.score == 1.0
    assert calls == 2
    assert delays == [0.0]


def test_rl_judge_retries_response_model_mismatch_without_scoring_it() -> None:
    initial, prompt = _provider('{"verdict":1,"rationale":"ok"}', [])
    calls = 0
    delays: list[float] = []

    def opener(_request, *, timeout):
        nonlocal calls
        assert timeout == 120.0
        calls += 1
        model = "wrong/model" if calls == 1 else "Qwen2.5-72B-Instruct"
        return _Response(
            {
                "model": model,
                "choices": [
                    {"message": {"content": '{"verdict":1,"rationale":"ok"}'}}
                ],
            }
        )

    provider = OpenAICompatibleJudgeProvider(
        replace(
            initial.config,
            expected_response_model="Qwen2.5-72B-Instruct",
            maximum_attempts=2,
            retry_backoff_seconds=1.0,
        ),
        opener=opener,
        sleeper=delays.append,
    )
    result = provider.judge(
        JudgeRequest(
            request_id="request-model-retry",
            task_kind="open_vqa",
            question="What is shown?",
            candidate_answer="ship",
            reference_answer="ship",
            prompt_identity=prompt,
        )
    )

    assert result.score == 1.0
    assert calls == 2
    assert delays == [1.0]


def test_rl_judge_retries_transient_transport_errors_without_attempt_limit() -> None:
    initial, prompt = _provider('{"verdict":1,"rationale":"ok"}', [])
    calls = 0
    delays: list[float] = []

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        if calls <= 12:
            raise URLError(TimeoutError("temporary timeout"))
        return _Response(
            {"choices": [{"message": {"content": '{"verdict":1,"rationale":"ok"}'}}]}
        )

    provider = OpenAICompatibleJudgeProvider(
        replace(
            initial.config,
            maximum_attempts=None,
            retry_backoff_seconds=1.0,
            retry_maximum_seconds=4.0,
            retry_transport_errors=True,
        ),
        opener=opener,
        sleeper=delays.append,
    )
    result = provider.judge(
        JudgeRequest(
            request_id="request-transport-retry",
            task_kind="math",
            question="1+1?",
            candidate_answer="2",
            reference_answer="2",
            prompt_identity=prompt,
        )
    )

    assert result.score == 1.0
    assert calls == 13
    assert delays == [1.0, 2.0] + [4.0] * 10


def test_rl_judge_does_not_retry_permanent_http_failure() -> None:
    initial, prompt = _provider('{"verdict":1,"rationale":"ok"}', [])
    calls = 0
    delays: list[float] = []

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error":{"message":"invalid credential"}}'),
        )

    provider = OpenAICompatibleJudgeProvider(
        replace(
            initial.config,
            maximum_attempts=None,
            retry_backoff_seconds=1.0,
            retry_all_server_errors=True,
            retry_transport_errors=True,
        ),
        opener=opener,
        sleeper=delays.append,
    )
    with pytest.raises(RuntimeError, match="HTTP 401"):
        provider.judge(
            JudgeRequest(
                request_id="request-permanent-failure",
                task_kind="open_vqa",
                question="What is shown?",
                candidate_answer="ship",
                reference_answer="ship",
                prompt_identity=prompt,
            )
        )

    assert calls == 1
    assert delays == []


def test_formal_v3_binding_declares_indefinite_transient_retry() -> None:
    path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/openrouter_qwen25_72b_formal_pilot_judge_v3.json"
    )
    bound = load_openai_compatible_judge(
        path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    config = bound.provider.config
    assert config.maximum_attempts is None
    assert config.retryable_http_statuses == (408, 425, 429)
    assert config.retry_all_server_errors is True
    assert config.retry_transport_errors is True


def test_openrouter_binding_uses_env_auth_pinned_route_and_retains_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/openrouter_qwen25_72b_rl_answer_judge_v1.json"
    )
    captured = []

    def opener(request, *, timeout):
        assert timeout == 120.0
        captured.append(request)
        return _Response(
            {
                "id": "generation-1",
                "model": "qwen/qwen-2.5-72b-instruct",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"verdict":1,"rationale":"equivalent"}'
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 201,
                    "completion_tokens": 17,
                    "total_tokens": 218,
                    "cost": 0.00007916,
                },
            }
        )

    bound = load_openai_compatible_judge(
        path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        opener=opener,
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        bound.provider.validate_credentials()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")

    result = bound.provider.judge(
        JudgeRequest(
            request_id="openrouter-request-1",
            task_kind="open_vqa",
            question="What color is the engine?",
            candidate_answer="yellow",
            reference_answer="The engine is yellow.",
            prompt_identity=bound.prompt_identity,
        )
    )

    assert result.score == 1.0
    assert result.usage is not None
    assert result.usage.prompt_tokens == 201
    assert result.usage.completion_tokens == 17
    assert result.usage.cost_usd == pytest.approx(0.00007916)
    request = captured[0]
    assert request.get_header("Authorization") == "Bearer test-only-secret"
    payload = json.loads(request.data)
    assert payload["model"] == "qwen/qwen-2.5-72b-instruct"
    assert "response_format" not in payload
    assert payload["provider"] == {
        "only": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
