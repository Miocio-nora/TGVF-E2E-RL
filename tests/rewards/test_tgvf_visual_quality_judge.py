from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import (
    AsyncTGVFVisualQualityJudgeProvider,
    BoundTGVFVisualQualityJudge,
    TGVFVisualQualityFailureKind,
    TGVFVisualQualityAsyncTransportPolicy,
    TGVFVisualQualityGlobalFailure,
    TGVFVisualQualityJudgeConfig,
    TGVFVisualQualityJudgeProvider,
    TGVFVisualQualityJudgeRequest,
    load_tgvf_visual_quality_judge,
    tgvf_visual_quality_prompt_identity,
    TGVF_VISUAL_QUALITY_SEQUENCE_JUDGE_PROMPT_VERSION,
)


_PNG_BYTES = b"\x89PNG\r\n\x1a\nvisual-quality-test"


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("visual-quality-test", name, "v1", digit * 64)


def _config(**changes: object) -> TGVFVisualQualityJudgeConfig:
    config = TGVFVisualQualityJudgeConfig(
        base_url="https://judge.invalid/v1",
        model_name="vision-judge-pinned",
        prompt_identity=tgvf_visual_quality_prompt_identity(),
        service_identity=_identity("service", "1"),
        model_identity=_identity("model", "2"),
        sampling_identity=_identity("sampling", "3"),
        expected_response_model="vision-judge-pinned",
    )
    return replace(config, **changes)


def _async_policy(**changes: object) -> TGVFVisualQualityAsyncTransportPolicy:
    values: dict[str, object] = {
        "maximum_concurrency": 2,
        "maximum_attempts": 4,
        "retry_backoff_seconds": 0.0,
        "retry_maximum_seconds": 0.0,
        "cache_max_entries": 16,
        "transient_failure_window_size": 8,
        "maximum_transient_failure_fraction": 0.25,
        "retryable_http_statuses": (408, 425, 429, 500, 502, 503, 504),
    }
    values.update(changes)
    return TGVFVisualQualityAsyncTransportPolicy(**values)


def _image(tmp_path: Path, *, contents: bytes = _PNG_BYTES) -> tuple[Path, str]:
    path = tmp_path / "original.png"
    path.write_bytes(contents)
    return path, sha256(contents).hexdigest()


def _request(tmp_path: Path, **changes: object) -> TGVFVisualQualityJudgeRequest:
    image_path, image_sha256 = _image(tmp_path)
    values: dict[str, object] = {
        "request_id": "sample-7/rollout-3",
        "image_path": image_path,
        "image_sha256": image_sha256,
        "question": "Which marked vessel is nearest the pier?",
        "tool_target": "Inspect the lower-left vessels and the pier boundary.",
        "post_tool_reasoning": "The blue vessel touches the pier boundary.",
        "final_answer": "the blue vessel",
        "prompt_identity": tgvf_visual_quality_prompt_identity(),
    }
    values.update(changes)
    return TGVFVisualQualityJudgeRequest(**values)


def _config_document() -> dict[str, object]:
    prompt = tgvf_visual_quality_prompt_identity()
    return {
        "schema_version": 1,
        "identity": "visual-quality-test-binding-v1",
        "role": "policy_rl_tgvf_visual_quality_judge_only",
        "model": {
            "repository": "test/vision-judge",
            "revision": "revision-123",
            "served_name": "vision-judge-pinned",
        },
        "prompt": {
            "version": prompt.version,
            "sha256": prompt.sha256,
            "output_schema": {
                "exact_keys": ["focus_score", "grounding_score"],
                "score_type": "integer",
                "allowed_values": [0, 1, 2],
            },
            "input_contract": {
                "original_image": "absolute_path_plus_sha256",
                "gold_or_reference_answer": "forbidden",
            },
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "seed": 42,
            "response_format": "json_object",
        },
        "service": {
            "base_url": "https://judge.invalid/v1",
            "timeout_seconds": 120.0,
            "deployment": "test-deployment",
            "api_key_env": "VISUAL_JUDGE_API_KEY",
            "expected_response_model": "vision-judge-pinned",
            "require_usage": True,
            "send_json_response_format": True,
        },
        "failure_policy": {
            "transport": "zero_current_sample_after_audit",
            "malformed_output": "zero_current_sample_after_audit",
            "input_or_identity_error": "raise_and_abort_reward_batch",
        },
        "scope": {
            "allows_policy_rl_reward": True,
            "allows_answer_correctness_judging": False,
            "accepts_gold_or_reference_answer": False,
            "sample_local_provider_failure": True,
            "formal_pilot_accepted": True,
        },
    }


def _write_config(
    tmp_path: Path,
    document: object,
) -> tuple[Path, str]:
    raw = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path = tmp_path / "visual-quality-judge.json"
    path.write_bytes(raw)
    return path, sha256(raw).hexdigest()


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


class _RawResponse(_Response):
    def __init__(self, body: bytes) -> None:
        self._body = body


def test_combined_judge_sends_one_gold_free_multimodal_request(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def opener(request, *, timeout):
        assert timeout == 120.0
        captured.append(json.loads(request.data))
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":2,"grounding_score":1}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 9,
                    "total_tokens": 110,
                },
            }
        )

    config = _config()
    request = _request(tmp_path)
    result = TGVFVisualQualityJudgeProvider(config, opener=opener).judge(request)

    assert result.ok is True
    assert (result.focus_score, result.grounding_score) == (2, 1)
    assert result.failure_kind is None
    assert result.config_identity == config.config_identity
    assert result.usage is not None and result.usage.total_tokens == 110
    assert len(captured) == 1

    payload = captured[0]
    assert payload["response_format"] == {"type": "json_object"}
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["image_url", "text"]
    wire_inputs = json.loads(content[1]["text"])
    assert set(wire_inputs) == {
        "question",
        "tool_target",
        "post_tool_reasoning",
        "final_answer",
        "image_sha256",
    }
    assert wire_inputs["image_sha256"] == request.image_sha256
    assert str(request.image_path) not in json.dumps(payload)


def test_sequence_prompt_sends_all_ordered_targets_in_one_request(
    tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []
    prompt = tgvf_visual_quality_prompt_identity(
        TGVF_VISUAL_QUALITY_SEQUENCE_JUDGE_PROMPT_VERSION
    )

    def opener(request, *, timeout):
        captured.append(json.loads(request.data))
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":1,"grounding_score":2}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            }
        )

    request = _request(
        tmp_path,
        tool_target=None,
        tool_targets=("first visual region", "second visual region"),
        prompt_identity=prompt,
    )
    result = TGVFVisualQualityJudgeProvider(
        _config(prompt_identity=prompt), opener=opener
    ).judge(request)

    assert result.ok is True
    assert len(captured) == 1
    user_text = captured[0]["messages"][1]["content"][1]["text"]
    inputs = json.loads(user_text)
    assert inputs["tool_targets"] == [
        "first visual region",
        "second visual region",
    ]
    assert "tool_target" not in inputs


def test_async_provider_retries_429_then_caches_success(tmp_path: Path) -> None:
    calls = 0

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(request.full_url, 429, "limited", {}, BytesIO())
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":2,"grounding_score":2}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.001,
                },
            }
        )

    strict = TGVFVisualQualityJudgeProvider(
        _config(async_transport=_async_policy()), opener=opener
    )
    provider = AsyncTGVFVisualQualityJudgeProvider(
        strict, local_maximum_concurrency=2
    )
    request = _request(tmp_path)

    async def exercise():
        first = await provider.judge(request)
        second = await provider.judge(request)
        return first, second

    first, second = asyncio.run(exercise())

    assert first.result.ok is True
    assert (first.attempts, first.retries, first.cache_hit) == (2, 1, False)
    assert second.result.ok is True
    assert (second.attempts, second.retries, second.cache_hit) == (0, 0, True)
    assert second.result.usage is None
    assert calls == 2


def test_async_provider_retries_malformed_completion_then_succeeds(
    tmp_path: Path,
) -> None:
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(
                {
                    "model": "vision-judge-pinned",
                    "choices": [{"message": {"content": "not json"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "cost": 0.001,
                    },
                }
            )
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":2,"grounding_score":1}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "total_tokens": 14,
                    "cost": 0.002,
                },
            }
        )

    provider = AsyncTGVFVisualQualityJudgeProvider(
        TGVFVisualQualityJudgeProvider(
            _config(async_transport=_async_policy()), opener=opener
        ),
        local_maximum_concurrency=2,
    )

    outcome = asyncio.run(provider.judge(_request(tmp_path)))

    assert outcome.result.ok is True
    assert (outcome.result.focus_score, outcome.result.grounding_score) == (2, 1)
    assert (outcome.attempts, outcome.retries) == (2, 1)
    assert outcome.result.usage is not None
    assert outcome.result.usage.total_tokens == 26
    assert outcome.result.usage.cost_usd == pytest.approx(0.003)
    assert calls == 2


def test_async_provider_rejects_request_id_content_collision(tmp_path: Path) -> None:
    strict = TGVFVisualQualityJudgeProvider(
        _config(async_transport=_async_policy()),
        opener=lambda *_args, **_kwargs: _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":2,"grounding_score":2}'
                        }
                    }
                ],
            }
        ),
    )
    provider = AsyncTGVFVisualQualityJudgeProvider(
        strict, local_maximum_concurrency=1
    )
    request = _request(tmp_path)
    async def exercise() -> None:
        await provider.judge(request)
        with pytest.raises(TGVFVisualQualityGlobalFailure, match="reused"):
            await provider.judge(replace(request, question="A changed question"))

    asyncio.run(exercise())


def test_request_type_cannot_accept_any_gold_or_reference_answer(tmp_path: Path) -> None:
    field_names = {field.name for field in fields(TGVFVisualQualityJudgeRequest)}
    assert field_names.isdisjoint(
        {"gold_answer", "reference_answer", "expected_answer", "ground_truth"}
    )
    request = _request(tmp_path)
    values = {field.name: getattr(request, field.name) for field in fields(request)}
    values["reference_answer"] = "SECRET_GOLD_VALUE"

    with pytest.raises(TypeError, match="unexpected keyword"):
        TGVFVisualQualityJudgeRequest(**values)


def test_image_sha_is_rechecked_before_any_network_request(tmp_path: Path) -> None:
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    request = _request(tmp_path)
    request.image_path.write_bytes(_PNG_BYTES + b"mutated")

    with pytest.raises(ValueError, match="SHA256 differs"):
        TGVFVisualQualityJudgeProvider(_config(), opener=opener).judge(request)
    assert calls == 0


def test_request_requires_absolute_existing_image_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        _request(tmp_path, image_path=Path("relative.png"))
    with pytest.raises(ValueError, match="regular file"):
        _request(tmp_path, image_path=tmp_path / "missing.png")


def test_image_type_is_derived_from_verified_bytes_not_suffix(tmp_path: Path) -> None:
    unsupported = b"not-an-image"
    request = _request(tmp_path)
    request.image_path.write_bytes(unsupported)
    request = replace(request, image_sha256=sha256(unsupported).hexdigest())

    with pytest.raises(ValueError, match="media type is unsupported"):
        TGVFVisualQualityJudgeProvider(_config(), opener=lambda *_a, **_k: None).judge(
            request
        )


@pytest.mark.parametrize(
    ("content", "reason"),
    (
        ("not json", "score_content_not_json"),
        ('{"focus_score":2,"grounding_score":1,"reason":"extra"}', "score_schema_mismatch"),
        ('{"focus_score":true,"grounding_score":1}', "focus_score_invalid"),
        ('{"focus_score":2,"grounding_score":3}', "grounding_score_invalid"),
        (
            '{"focus_score":2,"focus_score":0,"grounding_score":1}',
            "score_content_not_json",
        ),
    ),
)
def test_malformed_completed_score_is_explicit_sample_local_zero(
    tmp_path: Path,
    content: str,
    reason: str,
) -> None:
    def opener(_request, *, timeout):
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                    "cost": 0.001,
                },
            }
        )

    result = TGVFVisualQualityJudgeProvider(_config(), opener=opener).judge(
        _request(tmp_path)
    )

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.MALFORMED_OUTPUT
    assert result.failure_reason == reason
    assert (result.focus_score, result.grounding_score) == (0, 0)
    assert result.usage is not None and result.usage.cost_usd == 0.001


def test_non_json_http_body_is_explicit_malformed_zero(tmp_path: Path) -> None:
    provider = TGVFVisualQualityJudgeProvider(
        _config(), opener=lambda *_a, **_k: _RawResponse(b"not-json")
    )

    result = provider.judge(_request(tmp_path))

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.MALFORMED_OUTPUT
    assert result.failure_reason == "response_not_json"
    assert (result.focus_score, result.grounding_score) == (0, 0)


def test_response_model_mismatch_is_not_scored(tmp_path: Path) -> None:
    def opener(_request, *, timeout):
        return _Response(
            {
                "model": "unpinned-model",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":2,"grounding_score":2}'
                        }
                    }
                ],
            }
        )

    result = TGVFVisualQualityJudgeProvider(_config(), opener=opener).judge(
        _request(tmp_path)
    )

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.MALFORMED_OUTPUT
    assert result.failure_reason == "response_model_mismatch"
    assert (result.focus_score, result.grounding_score) == (0, 0)


def test_transport_errors_are_explicit_sample_local_zero(tmp_path: Path) -> None:
    def opener(_request, *, timeout):
        raise URLError("offline")

    result = TGVFVisualQualityJudgeProvider(_config(), opener=opener).judge(
        _request(tmp_path)
    )

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.TRANSPORT
    assert result.failure_reason == "URLError"
    assert (result.focus_score, result.grounding_score) == (0, 0)


def test_http_error_is_transport_without_serializing_provider_body(
    tmp_path: Path,
) -> None:
    def opener(request, *, timeout):
        raise HTTPError(
            request.full_url,
            429,
            "rate limited",
            {},
            BytesIO(b'{"error":{"message":"SECRET_PROVIDER_BODY"}}'),
        )

    result = TGVFVisualQualityJudgeProvider(_config(), opener=opener).judge(
        _request(tmp_path)
    )

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.TRANSPORT
    assert result.failure_reason == "http_status_429"


def test_required_usage_failure_is_sample_local_malformed(tmp_path: Path) -> None:
    def opener(_request, *, timeout):
        return _Response(
            {
                "model": "vision-judge-pinned",
                "choices": [
                    {
                        "message": {
                            "content": '{"focus_score":1,"grounding_score":2}'
                        }
                    }
                ],
            }
        )

    result = TGVFVisualQualityJudgeProvider(
        _config(require_usage=True), opener=opener
    ).judge(_request(tmp_path))

    assert result.ok is False
    assert result.failure_kind is TGVFVisualQualityFailureKind.MALFORMED_OUTPUT
    assert result.failure_reason == "response_usage_missing"
    assert (result.focus_score, result.grounding_score) == (0, 0)


def test_config_identity_is_stable_complete_and_secret_free() -> None:
    config = _config(api_key_env="VISUAL_JUDGE_API_KEY")
    same = _config(api_key_env="VISUAL_JUDGE_API_KEY")
    changed = _config(api_key_env="VISUAL_JUDGE_API_KEY", max_tokens=65)

    assert config.config_identity == same.config_identity
    assert config.config_sha256 != changed.config_sha256
    audit = config.audit_payload()
    assert audit["service"]["api_key_env"] == "VISUAL_JUDGE_API_KEY"
    assert "api_key" not in audit["service"]
    assert audit["sample_failure_policy"] == {
        "transport": "zero_current_sample_after_audit",
        "malformed_output": "zero_current_sample_after_audit",
    }


def test_prompt_identity_is_exact_not_version_only() -> None:
    expected = tgvf_visual_quality_prompt_identity()
    impostor = replace(expected, sha256="f" * 64)

    with pytest.raises(ValueError, match="prompt identity differs"):
        _config(prompt_identity=impostor)


def test_strict_sha_bound_loader_constructs_config_and_provider(
    tmp_path: Path,
) -> None:
    path, digest = _write_config(tmp_path, _config_document())

    bound = load_tgvf_visual_quality_judge(
        path,
        expected_file_sha256=digest,
        opener=lambda *_a, **_k: None,
    )

    assert isinstance(bound, BoundTGVFVisualQualityJudge)
    assert bound.provider.config is bound.config
    assert bound.config_file_sha256 == digest
    assert bound.binding_identity.sha256 == digest
    assert bound.config_identity == bound.config.config_identity
    assert bound.declared_identity == "visual-quality-test-binding-v1"
    assert bound.formal_pilot_accepted is True
    assert bound.config.model_name == "vision-judge-pinned"
    assert bound.config.require_usage is True


def test_loader_binds_sequence_prompt_and_async_transport(tmp_path: Path) -> None:
    document = _config_document()
    prompt = tgvf_visual_quality_prompt_identity(
        TGVF_VISUAL_QUALITY_SEQUENCE_JUDGE_PROMPT_VERSION
    )
    document["prompt"]["version"] = prompt.version
    document["prompt"]["sha256"] = prompt.sha256
    document["prompt"]["input_contract"][
        "ordered_successful_tool_targets"
    ] = "required_1_to_6"
    document["async_transport"] = {
        "maximum_concurrency": 16,
        "maximum_attempts": 4,
        "retry_backoff_seconds": 0.25,
        "retry_maximum_seconds": 2.0,
        "cache_max_entries": 8192,
        "transient_failure_window_size": 64,
        "maximum_transient_failure_fraction": 0.25,
        "retryable_http_statuses": [408, 425, 429, 500, 502, 503, 504],
    }
    path, digest = _write_config(tmp_path, document)

    bound = load_tgvf_visual_quality_judge(path, expected_file_sha256=digest)

    assert bound.config.prompt_identity == prompt
    assert bound.config.async_transport is not None
    assert bound.config.async_transport.maximum_concurrency == 16
    assert bound.config.async_transport.retryable_http_statuses == (
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    )


def test_loader_rejects_file_sha_before_decoding(tmp_path: Path) -> None:
    path, _digest = _write_config(tmp_path, _config_document())

    with pytest.raises(ValueError, match="file SHA256 differs"):
        load_tgvf_visual_quality_judge(
            path,
            expected_file_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda document: document.update({"unexpected": True}),
            "config fields differ",
        ),
        (
            lambda document: document["model"].update({"unbound": "metadata"}),
            "model fields differ",
        ),
        (
            lambda document: document["service"].update({"retry": True}),
            "service fields differ",
        ),
    ),
)
def test_loader_rejects_unknown_fields_at_every_binding_level(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    document = _config_document()
    mutation(document)
    path, digest = _write_config(tmp_path, document)

    with pytest.raises(ValueError, match=message):
        load_tgvf_visual_quality_judge(path, expected_file_sha256=digest)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allows_answer_correctness_judging", True),
        ("accepts_gold_or_reference_answer", True),
        ("sample_local_provider_failure", False),
    ),
)
def test_loader_rejects_scope_that_could_expose_answer_or_abort_samples(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    document = _config_document()
    document["scope"][field] = value
    path, digest = _write_config(tmp_path, document)

    with pytest.raises(ValueError, match="scope differs"):
        load_tgvf_visual_quality_judge(path, expected_file_sha256=digest)


def test_loader_rejects_nonlocal_failure_policy(tmp_path: Path) -> None:
    document = _config_document()
    document["failure_policy"]["transport"] = "raise_and_abort_reward_batch"
    path, digest = _write_config(tmp_path, document)

    with pytest.raises(ValueError, match="failure policy differs"):
        load_tgvf_visual_quality_judge(path, expected_file_sha256=digest)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = (
        b'{"schema_version":1,"schema_version":1,'
        b'"identity":"duplicate-does-not-load"}'
    )
    path = tmp_path / "duplicate.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="duplicate visual-quality config key"):
        load_tgvf_visual_quality_judge(
            path,
            expected_file_sha256=sha256(raw).hexdigest(),
        )
