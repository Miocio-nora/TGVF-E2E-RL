from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli

    sys.modules.setdefault("tomllib", tomli)

from tgvf_rl.representation.experiments.answer_bearing_span import deepseek_annotator
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _sample(
    uid: str,
    *,
    evidence: str,
    answer: str,
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=uid,
        image=f"/{uid}.png",
        image_id=f"image-{uid}",
        question="What is the answer?",
        target="the requested value",
        evidence_description=evidence,
        short_answer=answer,
    )


def test_local_exact_requires_one_boundary_safe_occurrence() -> None:
    resolved = _sample("one", evidence="颜色：红色 🟥", answer="红色")
    assert deepseek_annotator.local_exact_annotation(resolved) == {
        "uid": "one",
        "status": "resolved",
        "reason": None,
        "spans": [{"start": 3, "end": 5, "exact_text": "红色"}],
    }
    assert (
        deepseek_annotator.local_exact_annotation(
            _sample("embedded", evidence="credit", answer="red")
        )
        is None
    )
    assert (
        deepseek_annotator.local_exact_annotation(
            _sample("multiple", evidence="red then red", answer="red")
        )
        is None
    )


def test_v4_prompt_requires_minimal_but_complete_semantic_fragments() -> None:
    sample = _sample(
        "semantic-completeness",
        evidence="Narrow horizontal lines create a striped appearance.",
        answer="thin horizontal stripes",
    )
    payload = deepseek_annotator._request_payload(sample, max_tokens=160)
    prompt = payload["messages"][0]["content"]

    assert (
        deepseek_annotator.DEEPSEEK_SPAN_PROMPT_VERSION
        == "rp70-deepseek-v4-flash-span-v4"
    )
    assert "smallest COMPLETE semantic value" in prompt
    assert "every discriminative semantic component" in prompt
    assert '"Narrow horizontal lines" and "striped appearance"' in prompt
    assert '"button" and "top corner"' in prompt
    repair = deepseek_annotator._request_payload(
        sample,
        max_tokens=160,
        repair_code="quote_overbroad",
    )["messages"][-1]["content"]
    assert "shorter than 60%" in repair
    assert "non-overlapping exact fragments" in repair
    assert '"woven fabric" to only "woven"' in prompt


def test_model_quotes_map_occurrences_to_unicode_codepoints() -> None:
    sample = _sample("derived", evidence="苹果 3 个，梨 5 个；再次写 3。", answer="8")
    content = json.dumps(
        {
            "status": "resolved",
            "quotes": [
                {"exact_text": "3", "occurrence_index": 0},
                {"exact_text": "5", "occurrence_index": 0},
            ],
        },
        ensure_ascii=False,
    )
    assert deepseek_annotator.annotation_from_model_content(sample, content) == {
        "uid": "derived",
        "status": "resolved",
        "reason": None,
        "spans": [
            {"start": 3, "end": 4, "exact_text": "3"},
            {"start": 9, "end": 10, "exact_text": "5"},
        ],
    }
    with pytest.raises(deepseek_annotator.DeepSeekAnnotationError, match="occurrence"):
        deepseek_annotator.annotation_from_model_content(
            sample,
            '{"status":"resolved","quotes":[{"exact_text":"3","occurrence_index":2}]}',
        )


def test_empty_json_mode_response_is_retried_boundedly() -> None:
    sample = _sample("retry", evidence="value: blue", answer="Blue")
    response_payloads = [
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "   "}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        },
        {
            "id": "accepted",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"status":"resolved","quotes":'
                            '[{"exact_text":"blue","occurrence_index":0}]}'
                        )
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    ]

    class Response:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode()

    sleeps: list[float] = []
    request_bodies: list[dict[str, Any]] = []

    def opener(request: Any, **_kwargs: object) -> Response:
        request_bodies.append(json.loads(request.data))
        return Response(response_payloads.pop(0))

    result = deepseek_annotator._annotate_api_sample(
        sample,
        api_key="fake-test-key",
        max_tokens=160,
        timeout_seconds=1,
        max_attempts=2,
        opener=opener,
        sleeper=sleeps.append,
    )
    assert result.audit["attempts"] == 2
    assert result.annotation["spans"] == [{"start": 7, "end": 11, "exact_text": "blue"}]
    assert sleeps == [1]
    assert len(request_bodies[0]["messages"]) == 2
    assert len(request_bodies[1]["messages"]) == 3
    assert request_bodies[0] != request_bodies[1]


def test_rejects_numeric_no_span_and_obviously_overbroad_quote() -> None:
    numeric = _sample(
        "chart",
        evidence="In 2011 the chart shows 98 757 units.",
        answer="94,153",
    )
    with pytest.raises(deepseek_annotator.DeepSeekAnnotationError, match="contributor"):
        deepseek_annotator.annotation_from_model_content(
            numeric, '{"status":"no_span","quotes":[]}'
        )

    short = _sample(
        "count",
        evidence="There are two dogs standing near the open wooden door.",
        answer="two",
    )
    with pytest.raises(deepseek_annotator.DeepSeekAnnotationError, match="overbroad"):
        deepseek_annotator.annotation_from_model_content(
            short,
            json.dumps(
                {
                    "status": "resolved",
                    "quotes": [
                        {
                            "exact_text": short.evidence_description,
                            "occurrence_index": 0,
                        }
                    ],
                }
            ),
        )


def test_end_to_end_fake_api_checkpoints_and_resumes_without_key_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = (
        _sample("local", evidence="color: red", answer="red"),
        _sample("remote", evidence="inputs 3 and 5", answer="8"),
    )
    source = tmp_path / "source.jsonl"
    source.write_text("fixture\n", encoding="utf-8")
    dataset = SimpleNamespace(
        samples=samples,
        manifest=SimpleNamespace(source_sha256=sha256(source.read_bytes()).hexdigest()),
    )
    config = tmp_path / "training.toml"
    config.write_text("fixture\n", encoding="utf-8")
    split = SimpleNamespace(
        jsonl_path=source,
        source_sha256=dataset.manifest.source_sha256,
    )
    training = SimpleNamespace(
        source_path=config.resolve(),
        source_toml_sha256=sha256(config.read_bytes()).hexdigest(),
        data=SimpleNamespace(
            train=split, validation=split, warn_on_target_leakage=False
        ),
    )
    monkeypatch.setattr(
        deepseek_annotator,
        "load_representation_training_config",
        lambda _path: training,
    )
    monkeypatch.setattr(
        deepseek_annotator,
        "load_retained_representation_jsonl",
        lambda *_args, **_kwargs: dataset,
    )
    secret = "test-secret-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    calls: list[dict[str, Any]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "response-1",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "status": "resolved",
                                        "quotes": [
                                            {
                                                "exact_text": "3",
                                                "occurrence_index": 0,
                                            },
                                            {
                                                "exact_text": "5",
                                                "occurrence_index": 0,
                                            },
                                        ],
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                }
            ).encode()

    def opener(request: Any, *, timeout: float) -> Response:
        body = json.loads(request.data)
        calls.append({"url": request.full_url, "body": body, "timeout": timeout})
        assert request.get_header("Authorization") == f"Bearer {secret}"
        return Response()

    output = tmp_path / "annotations.jsonl"
    result = deepseek_annotator.annotate_answer_bearing_spans(
        training_config_path=config,
        split="train",
        output_path=output,
        concurrency=2,
        maximum_estimated_usd=1.0,
        checkpoint_every=1,
        opener=opener,
        sleeper=lambda _delay: None,
    )
    assert result["complete"] is True
    assert result["method_counts"] == {
        "deepseek_v4_flash": 1,
        "local_unique_boundary_exact": 1,
    }
    assert len(calls) == 1
    assert calls[0]["url"] == deepseek_annotator.DEEPSEEK_ENDPOINT
    assert calls[0]["body"]["model"] == "deepseek-v4-flash"
    assert calls[0]["body"]["thinking"] == {"type": "disabled"}
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["local", "remote"]
    assert rows[1]["spans"] == [
        {"start": 7, "end": 8, "exact_text": "3"},
        {"start": 13, "end": 14, "exact_text": "5"},
    ]
    for path in tmp_path.iterdir():
        assert secret.encode() not in path.read_bytes()

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    resumed = deepseek_annotator.annotate_answer_bearing_spans(
        training_config_path=config,
        split="train",
        output_path=output,
        concurrency=1,
        maximum_estimated_usd=1.0,
        opener=lambda *_args, **_kwargs: pytest.fail("resume called API"),
    )
    assert resumed["complete"] is True
    assert len(calls) == 1


def test_one_exhausted_sample_does_not_abort_batch_and_resume_retries_only_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _sample(
        "bad",
        evidence=(
            "The bottle has a broad rectangular outline with softened rounded corners."
        ),
        answer="rectangular with rounded corners",
    )
    good = _sample("good", evidence="The color is blue.", answer="azure")
    dataset = SimpleNamespace(
        samples=(bad, good), manifest=SimpleNamespace(source_sha256="a" * 64)
    )
    config = tmp_path / "training.toml"
    config.write_text("fixture\n", encoding="utf-8")
    split = SimpleNamespace(
        jsonl_path=tmp_path / "source.jsonl", source_sha256="a" * 64
    )
    training = SimpleNamespace(
        source_path=config.resolve(),
        source_toml_sha256=sha256(config.read_bytes()).hexdigest(),
        data=SimpleNamespace(
            train=split, validation=split, warn_on_target_leakage=False
        ),
    )
    monkeypatch.setattr(
        deepseek_annotator,
        "load_representation_training_config",
        lambda _path: training,
    )
    monkeypatch.setattr(
        deepseek_annotator,
        "load_retained_representation_jsonl",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-isolation-key")

    class Response:
        def __init__(self, content: dict[str, object]) -> None:
            self.content = content

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(self.content)},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                    },
                }
            ).encode()

    calls: Counter[str] = Counter()

    def failing_opener(request: Any, **_kwargs: object) -> Response:
        body = json.loads(request.data)
        uid = json.loads(body["messages"][1]["content"])["uid"]
        calls[uid] += 1
        if uid == "bad":
            return Response(
                {
                    "status": "resolved",
                    "quotes": [
                        {
                            "exact_text": "rectangular with rounded corners",
                            "occurrence_index": 0,
                        }
                    ],
                }
            )
        return Response(
            {
                "status": "resolved",
                "quotes": [{"exact_text": "blue", "occurrence_index": 0}],
            }
        )

    output = tmp_path / "isolated.jsonl"
    first = deepseek_annotator.annotate_answer_bearing_spans(
        training_config_path=config,
        split="train",
        output_path=output,
        concurrency=2,
        maximum_estimated_usd=1,
        max_attempts=2,
        checkpoint_every=1,
        opener=failing_opener,
        sleeper=lambda _delay: None,
    )
    assert first["complete"] is False
    assert first["completed_rows"] == 1
    assert first["failed_rows"] == 1
    assert first["pending_rows"] == 0
    assert first["failed_uids"] == ["bad"]
    assert calls == {"bad": 2, "good": 1}
    assert [json.loads(line)["uid"] for line in output.read_text().splitlines()] == [
        "good"
    ]
    failure_path = output.with_name(output.name + ".failures.jsonl")
    failure = json.loads(failure_path.read_text())
    assert set(failure) == {
        "uid",
        "status",
        "attempts",
        "error_code",
        "request_sha256",
    }
    assert bad.evidence_description not in failure_path.read_text()

    def repaired_opener(request: Any, **_kwargs: object) -> Response:
        body = json.loads(request.data)
        assert json.loads(body["messages"][1]["content"])["uid"] == "bad"
        return Response(
            {
                "status": "resolved",
                "quotes": [
                    {"exact_text": "rectangular", "occurrence_index": 0},
                    {"exact_text": "rounded corners", "occurrence_index": 0},
                ],
            }
        )

    resumed = deepseek_annotator.annotate_answer_bearing_spans(
        training_config_path=config,
        split="train",
        output_path=output,
        concurrency=1,
        maximum_estimated_usd=1,
        max_attempts=2,
        checkpoint_every=1,
        opener=repaired_opener,
        sleeper=lambda _delay: None,
    )
    assert resumed["complete"] is True
    assert resumed["completed_rows"] == 2
    assert resumed["failed_rows"] == 0
    assert failure_path.read_bytes() == b""
