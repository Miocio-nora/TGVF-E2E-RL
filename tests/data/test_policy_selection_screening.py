from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.policy_selection import (
    POLICY_SELECTION_CANDIDATE_SCHEMA,
    canonical_json_line,
)
from tgvf_rl.data.policy_selection_screening import (
    screen_policy_selection_candidates,
)


def _candidate(index: int, image_path: Path) -> dict[str, object]:
    return {
        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
        "sample_id": f"sample-{index}",
        "source": "thinklite",
        "question": f"Question {index}?",
        "ground_truth": "answer",
        "image": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "width": 8,
            "height": 8,
        },
        "provenance": {"row": index},
    }


def test_screening_excludes_exact_heldout_image_bytes(tmp_path: Path) -> None:
    leaked_image = tmp_path / "leaked.png"
    safe_image = tmp_path / "safe.png"
    leaked_image.write_bytes(b"held-out-image")
    safe_image.write_bytes(b"training-only-image")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "dataset": "heldout-a",
                "image_paths": [str(leaked_image)],
            }
        )
        + "\n"
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_bytes(
        canonical_json_line(_candidate(0, leaked_image))
        + canonical_json_line(_candidate(1, safe_image))
    )

    result = screen_policy_selection_candidates(
        candidates, tasks, tmp_path / "screened"
    )

    assert result.input_rows == 2
    assert result.eligible_rows == 1
    assert result.leakage_rows == 1
    eligible = json.loads(
        (result.output_root / "candidates.jsonl").read_text().strip()
    )
    excluded = json.loads(
        (result.output_root / "heldout_leakage.jsonl").read_text().strip()
    )
    assert eligible["sample_id"] == "sample-1"
    assert excluded["sample_id"] == "sample-0"
    assert excluded["heldout_datasets"] == ["heldout-a"]


def test_screening_refuses_to_overwrite_output(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    candidates.write_text("")
    tasks.write_text("")
    output = tmp_path / "screened"
    output.mkdir()

    with pytest.raises(FileExistsError):
        screen_policy_selection_candidates(candidates, tasks, output)
