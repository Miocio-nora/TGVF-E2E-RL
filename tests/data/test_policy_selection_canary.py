from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from tgvf_rl.data.policy_selection import POLICY_SELECTION_CANDIDATE_SCHEMA
from tgvf_rl.data.policy_selection_canary import (
    POLICY_SELECTION_CANARY_ALGORITHM_VERSION,
    T1_CANARY_TOTAL,
    THINKLITE_ANSWER_FORM_QUOTAS,
    VSTAR_CANARY_SOURCE_FILES,
    build_t1_canary_selection,
    canonical_canary_manifest_bytes,
    classify_thinklite_answer_form,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _base_candidate(sample_id: str, source: str, ground_truth: str) -> dict[str, Any]:
    return {
        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
        "sample_id": sample_id,
        "source": source,
        "question": f"Question for {sample_id}?",
        "ground_truth": ground_truth,
        "image": {
            "path": f"/fixture/{sample_id}.png",
            "sha256": _digest(f"image:{sample_id}"),
            "width": 1000,
            "height": 1000,
        },
        "gt_regions": [],
        "provenance": {
            "dataset_id": f"fixture/{source}",
            "revision": "fixture-v1",
            "source_file": f"{source}.jsonl",
            "source_row_index": int(sample_id.rsplit("-", 1)[-1]),
        },
    }


def _vstar_candidates() -> list[dict[str, Any]]:
    answers = (
        "red",
        "the red ball",
        "the small red ball beside the box is visible",
        "the unusually small red ball beside the old box is clearly visible "
        "near the far corner of the room",
    )
    boxes = (
        [0, 0, 10, 10],
        [0, 0, 50, 50],
        [0, 0, 200, 200],
        [0, 0, 500, 500],
    )
    records: list[dict[str, Any]] = []
    row_index = 0
    for source_file in VSTAR_CANARY_SOURCE_FILES:
        for index in range(20):
            record = _base_candidate(f"vstar-{row_index}", "vstar", answers[index % 4])
            record["gt_regions"] = [boxes[(index // 4) % 4]]
            record["provenance"]["source_file"] = source_file
            records.append(record)
            row_index += 1
    return records


def _arxiv_candidate(
    row_index: int,
    *,
    option_count: int,
    clean_answer_index: int,
    label_form: str,
    removal_reason: str | None,
) -> dict[str, Any]:
    record = _base_candidate(
        f"arxivqa-{row_index}",
        "arxivqa",
        chr(ord("A") + clean_answer_index),
    )
    raw_choices = [f"raw choice {index}" for index in range(option_count)]
    source_indices = list(range(option_count))
    removed_options: list[dict[str, Any]] = []
    if removal_reason is not None:
        removed_text = "-" if removal_reason == "separator" else "# Next Figure"
        raw_choices.insert(1, removed_text)
        source_indices = [0, *range(2, option_count + 1)]
        removed_options = [
            {
                "source_index": 1,
                "raw_option": removed_text,
                "reason": removal_reason,
            }
        ]
    answer_letter = chr(ord("A") + clean_answer_index)
    raw_labels = {
        "bare": answer_letter,
        "dot": f"{answer_letter}. answer text",
        "paren": f"{answer_letter}) answer text",
        "bracketed": f"[{answer_letter}]",
    }
    record["question"] = "Which option?\nChoices:\n" + "\n".join(
        f"{chr(ord('A') + index)}. clean choice {index}"
        for index in range(option_count)
    )
    record["selection_metadata"] = {
        "options": [
            f"{chr(ord('A') + index)}. clean choice {index}"
            for index in range(option_count)
        ],
        "raw_options": raw_choices,
        "source_option_indices": source_indices,
        "removed_options": removed_options,
        "option_count": option_count,
        "option_transform_version": "arxivqa-canonical-options-v2",
        "raw_label": raw_labels[label_form],
        "label_source_index": source_indices[clean_answer_index],
        "label_clean_index": clean_answer_index,
        "rationale": None,
    }
    return record


def _arxiv_candidates() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    label_forms = ("bare", "dot", "paren", "bracketed")
    removal_reasons = (None, "separator", "markdown_figure_heading")
    for index in range(84):
        option_count = 2 + index % 8
        answer_index = index % option_count
        records.append(
            _arxiv_candidate(
                index,
                option_count=option_count,
                clean_answer_index=answer_index,
                label_form=label_forms[index % len(label_forms)],
                removal_reason=removal_reasons[index % len(removal_reasons)],
            )
        )
    records.append(
        _arxiv_candidate(
            84,
            option_count=10,
            clean_answer_index=9,
            label_form="paren",
            removal_reason="separator",
        )
    )
    return records


_THINKLITE_ANSWERS = {
    "integer": lambda index: f"{1000 + index}",
    "decimal": lambda index: f"{index + 1}.25",
    "fraction": lambda index: f"{index + 1}/{index + 2}",
    "percent": lambda index: f"{index + 10}%",
    "expression": lambda index: f"x = {index + 1}",
    "yes-no": lambda index: "yes" if index % 2 else "no",
    "short-text": lambda index: f"blue object {index}",
    "other": lambda index: f"this is a deliberately long textual answer number {index}",
}


def _thinklite_candidates() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    row_index = 0
    for family, quota in THINKLITE_ANSWER_FORM_QUOTAS.items():
        for family_index in range(quota + (0 if family == "percent" else 2)):
            records.append(
                _base_candidate(
                    f"thinklite-{row_index}",
                    "thinklite",
                    _THINKLITE_ANSWERS[family](family_index),
                )
            )
            row_index += 1
    return records


def _population() -> list[dict[str, Any]]:
    return [*_vstar_candidates(), *_arxiv_candidates(), *_thinklite_candidates()]


def test_canary_is_exact_stratified_and_invariant_to_input_order() -> None:
    population = _population()
    first = build_t1_canary_selection(population)
    second = build_t1_canary_selection(reversed(population))

    assert len(first.selected_candidates) == T1_CANARY_TOTAL
    assert first.selected_candidates == second.selected_candidates
    assert first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["selection_algorithm_version"] == (
        POLICY_SELECTION_CANARY_ALGORITHM_VERSION
    )
    assert first.manifest["selection_is_outcome_independent"] is True
    assert (
        first.manifest["content_sha256"]
        == hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in first.manifest.items()
                    if key != "content_sha256"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert canonical_canary_manifest_bytes(first.manifest).endswith(b"\n")

    by_source = Counter(record["source"] for record in first.selected_candidates)
    assert by_source == {"vstar": 64, "arxivqa": 64, "thinklite": 64}
    entries = first.manifest["selected"]
    assert len({entry["sample_id"] for entry in entries}) == T1_CANARY_TOTAL
    assert len({entry["candidate_sha256"] for entry in entries}) == T1_CANARY_TOTAL
    assert all(len(entry["candidate_sha256"]) == 64 for entry in entries)

    vstar_files = Counter(
        record["provenance"]["source_file"]
        for record in first.selected_candidates
        if record["source"] == "vstar"
    )
    assert vstar_files == {source_file: 16 for source_file in VSTAR_CANARY_SOURCE_FILES}
    vstar_strata: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"answer": set(), "area": set()}
    )
    for entry in entries:
        if entry["source"] != "vstar":
            continue
        source_file = entry["stratum"]["source_file"]
        vstar_strata[source_file]["answer"].add(entry["stratum"]["answer_length_bin"])
        vstar_strata[source_file]["area"].add(
            entry["stratum"]["bbox_relative_area_bin"]
        )
    assert all(len(value["answer"]) == 4 for value in vstar_strata.values())
    assert all(len(value["area"]) == 4 for value in vstar_strata.values())

    arxiv_selected = [
        record for record in first.selected_candidates if record["source"] == "arxivqa"
    ]
    assert any(
        record["ground_truth"] == "J"
        and record["selection_metadata"]["option_count"] == 10
        for record in arxiv_selected
    )
    arxiv_strata = [
        entry["stratum"] for entry in entries if entry["source"] == "arxivqa"
    ]
    assert {stratum["removed_option_case"] for stratum in arxiv_strata} == {
        "none",
        "separator",
        "markdown-figure-heading",
    }
    assert {stratum["raw_label_form"] for stratum in arxiv_strata} == {
        "bare-letter",
        "dot-suffix",
        "paren-suffix",
        "bracketed-letter",
    }

    thinklite_counts = Counter(
        classify_thinklite_answer_form(record["ground_truth"])
        for record in first.selected_candidates
        if record["source"] == "thinklite"
    )
    assert thinklite_counts == THINKLITE_ANSWER_FORM_QUOTAS


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("7,690", "integer"),
        ("-0.25", "decimal"),
        ("3/7", "fraction"),
        (r"\frac{3}{7}", "fraction"),
        ("62.5%", "percent"),
        ("No.", "yes-no"),
        ("x = 4", "expression"),
        ("Southport-Fort Fisher", "short-text"),
        ("a textual response containing more than four words", "other"),
    ],
)
def test_thinklite_answer_form_classification(answer: str, expected: str) -> None:
    assert classify_thinklite_answer_form(answer) == expected


def test_selector_fails_closed_on_old_arxiv_metadata_and_duplicate_rows() -> None:
    population = _population()
    old_arxiv = next(record for record in population if record["source"] == "arxivqa")
    old_arxiv["selection_metadata"].pop("option_transform_version")
    with pytest.raises(ValueError, match="option_transform_version"):
        build_t1_canary_selection(population)

    population = _population()
    with pytest.raises(ValueError, match="duplicate candidate sample_id"):
        build_t1_canary_selection([*population, population[0]])


def test_selector_fails_when_a_fixed_thinklite_quota_is_unavailable() -> None:
    population = _population()
    removed_one_percent = False
    reduced: list[dict[str, Any]] = []
    for record in population:
        if (
            not removed_one_percent
            and record["source"] == "thinklite"
            and classify_thinklite_answer_form(record["ground_truth"]) == "percent"
        ):
            removed_one_percent = True
            continue
        reduced.append(record)
    with pytest.raises(ValueError, match="'percent'.*needs 5"):
        build_t1_canary_selection(reduced)


def test_module_import_does_not_load_torch() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tgvf_rl.data.policy_selection_canary; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
