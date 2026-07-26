from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.policy_selection import SelectionCandidate, canonical_json_line
from tgvf_rl.data.policy_selection_full import (
    T1_FULL_SELECTION_ALGORITHM_VERSION,
    T1_FULL_SELECTION_MANIFEST_SCHEMA,
    materialize_t1_full_selection,
)


def _candidate(tmp_path: Path, source: str, index: int) -> dict[str, object]:
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": f"fixture:{source}:{index}",
        "source": source,
        "question": f"Question {source} {index}?",
        "ground_truth": "answer",
        "image": {
            "path": str((tmp_path / f"{source}-{index}.png").resolve()),
            "sha256": f"{index + 1:064x}",
            "width": 32,
            "height": 32,
        },
        "gt_regions": [],
        "provenance": {"fixture": True, "index": index},
    }


def _sources(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for source in ("vstar", "arxivqa", "thinklite"):
        path = tmp_path / f"{source}.jsonl"
        path.write_bytes(
            b"".join(
                canonical_json_line(_candidate(tmp_path, source, index))
                for index in range(2)
            )
        )
        paths.append(path)
    return paths


def test_materialize_full_selection_is_ordered_and_content_bound(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path)
    output = tmp_path / "full" / "candidates.jsonl"
    manifest_path = tmp_path / "full" / "manifest.json"

    result = materialize_t1_full_selection(
        paths, output_path=output, manifest_path=manifest_path
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["source"] for record in records] == [
        "vstar",
        "vstar",
        "arxivqa",
        "arxivqa",
        "thinklite",
        "thinklite",
    ]
    assert len({SelectionCandidate.from_record(row).identity_sha256 for row in records}) == 6
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == T1_FULL_SELECTION_MANIFEST_SCHEMA
    assert (
        manifest["selection_algorithm_version"]
        == T1_FULL_SELECTION_ALGORITHM_VERSION
    )
    assert manifest["selection_is_outcome_independent"] is True
    assert manifest["source_counts"] == {
        "arxivqa": 2,
        "thinklite": 2,
        "vstar": 2,
    }
    assert result["rows"] == 6
    assert result["candidates_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_materialize_full_selection_fails_closed_on_source_order_and_replacement(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path)
    output = tmp_path / "full.jsonl"
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="expected source vstar"):
        materialize_t1_full_selection(
            paths[::-1], output_path=output, manifest_path=manifest
        )
    assert not output.exists()
    assert not manifest.exists()

    materialize_t1_full_selection(
        paths, output_path=output, manifest_path=manifest
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        materialize_t1_full_selection(
            paths, output_path=output, manifest_path=manifest
        )
