from __future__ import annotations

import json

import pytest

from tgvf_rl.representation.experiments.answer_bearing_span import (
    deepseek_index_repair,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _sample(
    evidence: str,
    answer: str = "top corner button",
    *,
    uid: str = "repair-sample",
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=uid,
        image=f"/{uid}.png",
        image_id=f"image-{uid}",
        question="Which part is red?",
        target="the red part",
        evidence_description=evidence,
        short_answer=answer,
    )


def _resolved(uid: str) -> dict[str, object]:
    return {
        "uid": uid,
        "status": "resolved",
        "reason": None,
        "spans": [{"start": 0, "end": 1, "exact_text": "x"}],
    }


def _no_span(uid: str) -> dict[str, object]:
    return {
        "uid": uid,
        "status": "verified_no_answer_bearing_evidence",
        "reason": (
            "verified_no_answer_bearing_evidence_in_bound_evidence_description_v1"
        ),
        "spans": [],
    }


def _audit(uid: str, method: str) -> dict[str, object]:
    return {"uid": uid, "method": method}


def _failure(uid: str, error_code: str) -> dict[str, object]:
    return {
        "uid": uid,
        "status": "retryable_failure",
        "attempts": 1,
        "error_code": error_code,
        "request_sha256": "a" * 64,
    }


def test_evidence_tokens_preserve_exact_unicode_offsets() -> None:
    tokens = deepseek_index_repair.evidence_tokens("红色 button-like, top corner.")
    assert [token.text for token in tokens] == [
        "红色",
        "button-like",
        ",",
        "top",
        "corner",
        ".",
    ]
    assert [(token.start, token.end) for token in tokens] == [
        (0, 2),
        (3, 14),
        (14, 15),
        (16, 19),
        (20, 26),
        (26, 27),
    ]


def test_components_merge_consecutive_ids_into_discontinuous_exact_spans() -> None:
    sample = _sample("A small button near the top corner is red.")
    content = json.dumps(
        {
            "status": "resolved",
            "components": [
                {"name": "object", "token_indices": [2]},
                {"name": "relation", "token_indices": [5, 6]},
            ],
        }
    )
    assert deepseek_index_repair.annotation_from_index_content(sample, content) == {
        "uid": "repair-sample",
        "status": "resolved",
        "reason": None,
        "spans": [
            {"start": 8, "end": 14, "exact_text": "button"},
            {"start": 24, "end": 34, "exact_text": "top corner"},
        ],
    }

    adjacent_components = json.dumps(
        {
            "status": "resolved",
            "components": [
                {"name": "modifier", "token_indices": [1]},
                {"name": "object", "token_indices": [2]},
            ],
        }
    )
    assert deepseek_index_repair.annotation_from_index_content(
        sample, adjacent_components
    )["spans"] == [{"start": 2, "end": 14, "exact_text": "small button"}]


@pytest.mark.parametrize(
    ("components", "match"),
    [
        ([{"name": "object", "token_indices": [999]}], "out of bounds"),
        ([{"name": "object", "token_indices": [True]}], "integers"),
    ],
)
def test_component_token_ids_are_strictly_validated(
    components: list[dict[str, object]], match: str
) -> None:
    sample = _sample("A small button near the top corner is red.")
    with pytest.raises(Exception, match=match):
        deepseek_index_repair.annotation_from_index_content(
            sample,
            json.dumps({"status": "resolved", "components": components}),
        )


def test_no_span_is_explicitly_gated_per_sample() -> None:
    sample = _sample("No useful local evidence is present.")
    content = '{"status":"no_span","components":[]}'
    with pytest.raises(Exception, match="disabled"):
        deepseek_index_repair.annotation_from_index_content(sample, content)
    assert deepseek_index_repair.annotation_from_index_content(
        sample,
        content,
        allow_no_span=True,
    ) == _no_span(sample.sample_id)

    with pytest.raises(Exception, match="fields"):
        deepseek_index_repair.annotation_from_index_content(
            sample,
            '{"ranges":[{"start_token":0,"end_token":1}]}',
        )


def test_component_names_may_repeat_because_token_ids_are_authoritative() -> None:
    sample = _sample("A small button near the top corner is red.")
    annotation = deepseek_index_repair.annotation_from_index_content(
        sample,
        json.dumps(
            {
                "status": "resolved",
                "components": [
                    {"name": "answer", "token_indices": [2]},
                    {"name": "answer", "token_indices": [5, 6]},
                ],
            }
        ),
    )
    assert [span["exact_text"] for span in annotation["spans"]] == [
        "button",
        "top corner",
    ]


def test_component_token_ids_are_locally_sorted_and_deduplicated() -> None:
    sample = _sample("A small button near the top corner is red.")
    annotation = deepseek_index_repair.annotation_from_index_content(
        sample,
        json.dumps(
            {
                "status": "resolved",
                "components": [
                    {"name": "relation", "token_indices": [6, 5, 2]},
                    {"name": "object", "token_indices": [2]},
                ],
            }
        ),
    )
    assert [span["exact_text"] for span in annotation["spans"]] == [
        "button",
        "top corner",
    ]


def test_v3_request_uses_named_components_and_complete_semantic_policy() -> None:
    sample = _sample("A small button near the top corner is red.")
    payload = deepseek_index_repair._request_payload(
        sample,
        max_tokens=160,
        allow_no_span=True,
    )
    user = json.loads(payload["messages"][1]["content"])
    prompt = payload["messages"][0]["content"]
    assert user["evidence_tokens"][2] == {"index": 2, "text": "button"}
    assert user["allow_no_span"] is True
    assert "evidence_description" not in user
    assert '"status":"resolved","components"' in prompt
    assert "individual integer token IDs" in prompt
    assert "smallest COMPLETE semantic" in prompt
    assert "numeric operand or local contributor" in prompt
    assert "top corner button" in prompt
    assert "thin horizontal stripes" in prompt
    assert payload["thinking"] == {"type": "disabled"}
    assert (
        deepseek_index_repair.DEEPSEEK_INDEX_REPAIR_PROMPT_VERSION
        == "rp70-deepseek-v4-flash-token-index-repair-v4"
    )


def test_reaudit_targets_no_span_legacy_methods_explicit_and_resumed_uids() -> None:
    selected = tuple(
        _sample("x", uid=uid)
        for uid in (
            "no-span",
            "legacy",
            "explicit",
            "keep",
            "confirmed-no-span",
            "confirmed-explicit",
            "resume",
        )
    )
    annotations = {
        "no-span": _no_span("no-span"),
        "legacy": _resolved("legacy"),
        "explicit": _resolved("explicit"),
        "keep": _resolved("keep"),
        "confirmed-no-span": _no_span("confirmed-no-span"),
        "confirmed-explicit": _resolved("confirmed-explicit"),
    }
    audits = {
        "no-span": _audit("no-span", "deepseek_v4_flash"),
        "legacy": _audit("legacy", "deepseek_v4_flash_token_index_repair_v2"),
        "explicit": _audit("explicit", "deepseek_v4_flash"),
        "keep": _audit("keep", "deepseek_v4_flash"),
        "confirmed-no-span": _audit(
            "confirmed-no-span", deepseek_index_repair.DEEPSEEK_INDEX_REAUDIT_METHOD
        ),
        "confirmed-explicit": _audit(
            "confirmed-explicit", deepseek_index_repair.DEEPSEEK_INDEX_REAUDIT_METHOD
        ),
    }
    failures = {"resume": _failure("resume", "token_index_reaudit_v3_pending")}
    targets = deepseek_index_repair._reaudit_target_uids(
        selected=selected,
        annotations=annotations,
        audits=audits,
        failures=failures,
        reaudit_uids=("explicit", "confirmed-explicit"),
    )
    assert targets == ("no-span", "legacy", "explicit", "resume")

    with pytest.raises(Exception, match="not in the split"):
        deepseek_index_repair._reaudit_target_uids(
            selected=selected,
            annotations=annotations,
            audits=audits,
            failures=failures,
            reaudit_uids=("unknown",),
        )


def test_reaudit_clone_removes_annotations_and_audits_fail_closed() -> None:
    annotations = {"one": _resolved("one"), "keep": _resolved("keep")}
    audits = {
        "one": _audit("one", "deepseek_v4_flash"),
        "keep": _audit("keep", "deepseek_v4_flash"),
    }
    failures: dict[str, dict[str, object]] = {}
    cloned_annotations, cloned_audits, cloned_failures = (
        deepseek_index_repair._clone_reaudit_fail_closed(
            target_uids=("one",),
            annotations=annotations,
            audits=audits,
            failures=failures,
        )
    )
    assert set(annotations) == {"one", "keep"}
    assert set(audits) == {"one", "keep"}
    assert set(cloned_annotations) == {"keep"}
    assert set(cloned_audits) == {"keep"}
    assert cloned_failures["one"]["error_code"] == "token_index_reaudit_v4_pending"
    assert len(str(cloned_failures["one"]["request_sha256"])) == 64


def test_cli_accepts_scope_and_repeatable_reaudit_uids() -> None:
    args = deepseek_index_repair._parser().parse_args(
        [
            "--training-config",
            "training.toml",
            "--split",
            "train",
            "--output",
            "annotations.jsonl",
            "--max-estimated-usd",
            "1",
            "--scope",
            "reaudit",
            "--reaudit-uid",
            "one",
            "--reaudit-uid",
            "two",
        ]
    )
    assert args.scope == "reaudit"
    assert args.reaudit_uid == ["one", "two"]
