from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from tgvf_rl.evaluation.deepeyes_dev import (
    DEEPEYES_DEV_SEED,
    _task_rows,
    hrbench_sample_id,
    official_mcq_prompt,
    select_hrbench_rows,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for category in ("cross", "single"):
        for cycle, answer in enumerate("ABCD"):
            for number in range(100):
                rows.append(
                    {
                        "index": len(rows),
                        "category": category,
                        "cycle_category": cycle,
                        "answer": answer,
                        "number": number,
                    }
                )
    return rows


def test_hrbench_selection_is_exactly_balanced_and_stable() -> None:
    rows = _rows()
    first = select_hrbench_rows(rows)
    second = select_hrbench_rows(rows)
    assert first == second
    assert len(first) == len(set(first)) == 200
    counts = Counter(
        (
            rows[index]["category"],
            rows[index]["cycle_category"],
            rows[index]["answer"],
        )
        for index in first
    )
    assert set(counts.values()) == {25}
    assert len(counts) == 8


def test_hrbench_selection_uses_the_documented_stable_hash() -> None:
    rows = _rows()
    selected = select_hrbench_rows(rows)
    cross_a = [
        index
        for index in selected
        if rows[index]["category"] == "cross" and rows[index]["answer"] == "A"
    ]
    ranked = sorted(
        range(100),
        key=lambda index: sha256(
            (
                f"{DEEPEYES_DEV_SEED}:"
                + hrbench_sample_id(
                    population_id="hr_bench_8k_800",
                    source_file="hr_bench_8k/snapshot/hr_bench_8k.parquet",
                    raw_id=index,
                    row_index=index,
                )
            ).encode()
        ).hexdigest(),
    )[:25]
    assert cross_a == ranked


def test_hrbench_selection_rejects_population_drift() -> None:
    with pytest.raises(ValueError, match="strata differ"):
        select_hrbench_rows(_rows()[:-100])


def test_official_mcq_prompt_matches_vlmevalkit_text() -> None:
    row = {
        "question": "Where?",
        "A": "North",
        "B": "South",
        "C": "East",
        "D": "West",
    }
    assert official_mcq_prompt(row) == (
        "Question: Where?\nOptions:\n"
        "A. North\nB. South\nC. East\nD. West\n"
        "Please select the correct answer from the options above. \n"
    )


def test_task_rows_retain_gold_options_metadata_and_image_identity(
    tmp_path: Path,
) -> None:
    image = tmp_path / "fixture.png"
    Image.new("RGB", (7, 5), (1, 2, 3)).save(image)
    source = {
        "index": "sample-1",
        "sample_id": "sample-1",
        "answer": "C",
        "question": "Where?",
        "A": "North",
        "B": "South",
        "C": "East",
        "D": "West",
        "category": "single",
        "cycle_category": "2",
        "image_path": str(image),
    }

    (task,) = _task_rows((("HRBench8K", (source,)),))

    assert task["answer"] == "C"
    assert task["options"] == [
        ["A", "North"],
        ["B", "South"],
        ["C", "East"],
        ["D", "West"],
    ]
    assert task["metadata"] == [
        ["category", "single"],
        ["cycle_category", "2"],
    ]
    assert task["image_dimensions"] == [[7, 5]]
    assert task["image_sha256s"] == [sha256(image.read_bytes()).hexdigest()]
