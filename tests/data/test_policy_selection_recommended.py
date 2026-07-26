from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.policy_selection import SelectionCandidate, canonical_json_line
from tgvf_rl.data.policy_selection_recommended import (
    T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION,
    T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA,
    materialize_t1_recommended_selection,
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
    paths = []
    for source in ("vstar", "arxivqa", "thinklite"):
        path = tmp_path / f"{source}.jsonl"
        path.write_bytes(
            b"".join(
                canonical_json_line(_candidate(tmp_path, source, index))
                for index in range(5)
            )
        )
        paths.append(path)
    return paths


def _score(namespace: str, source: str, identity: str) -> str:
    payload = (
        b"tgvf-policy-selection-t1-source-quota-v1\0"
        + namespace.encode()
        + b"\0"
        + source.encode()
        + b"\0"
        + identity.encode()
    )
    return hashlib.sha256(payload).hexdigest()


def test_recommended_selection_is_exact_bottom_k_and_source_ordered(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path)
    quotas = {"vstar": 3, "arxivqa": 2, "thinklite": 5}
    namespace = "fixture-v1"
    root = tmp_path / "selection"

    result = materialize_t1_recommended_selection(
        paths,
        output_root=root,
        source_quotas=quotas,
        namespace=namespace,
    )

    rows = [
        json.loads(line)
        for line in (root / "candidates.jsonl").read_text().splitlines()
    ]
    assert result["rows"] == 10
    assert result["logical_attempts"] == 80
    assert result["source_counts"] == quotas
    assert [row["source"] for row in rows] == [
        "vstar",
        "vstar",
        "vstar",
        "arxivqa",
        "arxivqa",
        "thinklite",
        "thinklite",
        "thinklite",
        "thinklite",
        "thinklite",
    ]
    for source, quota in quotas.items():
        population = [
            SelectionCandidate.from_record(_candidate(tmp_path, source, index))
            for index in range(5)
        ]
        expected = {
            identity
            for _, identity in sorted(
                (_score(namespace, source, item.identity_sha256), item.identity_sha256)
                for item in population
            )[:quota]
        }
        actual = {
            SelectionCandidate.from_record(row).identity_sha256
            for row in rows
            if row["source"] == source
        }
        assert actual == expected

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA
    assert (
        manifest["selection_algorithm_version"]
        == T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION
    )
    assert manifest["selection_is_outcome_independent"] is True
    assert (
        manifest["candidates_sha256"]
        == hashlib.sha256((root / "candidates.jsonl").read_bytes()).hexdigest()
    )


def test_recommended_selection_is_immutable_and_failure_is_atomic(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path)
    root = tmp_path / "selection"
    materialize_t1_recommended_selection(
        paths,
        output_root=root,
        source_quotas={"vstar": 1, "arxivqa": 1, "thinklite": 1},
        namespace="fixture-v1",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_t1_recommended_selection(
            paths,
            output_root=root,
            source_quotas={"vstar": 1, "arxivqa": 1, "thinklite": 1},
            namespace="fixture-v1",
        )

    bad_root = tmp_path / "bad"
    with pytest.raises(ValueError, match="expected arxivqa"):
        materialize_t1_recommended_selection(
            [paths[0], paths[0], paths[2]],
            output_root=bad_root,
            source_quotas={"vstar": 1, "arxivqa": 1, "thinklite": 1},
            namespace="fixture-v1",
        )
    assert not bad_root.exists()
