from __future__ import annotations

from pathlib import Path

import pytest

from tools.judge_policy_data_selection_t1_streaming import (
    _completed_index_inventory,
    _has_legacy_judge_index,
)


def test_legacy_index_presence_supports_flat_and_sharded_layouts(
    tmp_path: Path,
) -> None:
    payload_sha256 = "ab" + "1" * 62

    assert not _has_legacy_judge_index(tmp_path, payload_sha256)

    flat = tmp_path / "requests" / f"{payload_sha256}.json"
    flat.parent.mkdir(parents=True)
    flat.write_text("not parsed", encoding="utf-8")
    assert _has_legacy_judge_index(tmp_path, payload_sha256)

    flat.unlink()
    sharded = (
        tmp_path
        / "requests"
        / payload_sha256[:2]
        / f"{payload_sha256}.json"
    )
    sharded.parent.mkdir(parents=True)
    sharded.write_text("still not parsed", encoding="utf-8")
    assert _has_legacy_judge_index(tmp_path, payload_sha256)

    flat.write_text("deliberately different", encoding="utf-8")
    assert _has_legacy_judge_index(tmp_path, payload_sha256)


def test_legacy_index_presence_rejects_non_file_claim(tmp_path: Path) -> None:
    payload_sha256 = "cd" + "2" * 62
    claimed_index = (
        tmp_path
        / "requests"
        / payload_sha256[:2]
        / f"{payload_sha256}.json"
    )
    claimed_index.mkdir(parents=True)

    with pytest.raises(ValueError, match="not a regular file"):
        _has_legacy_judge_index(tmp_path, payload_sha256)


def test_completed_index_inventory_supports_sharded_and_flat_layouts(
    tmp_path: Path,
) -> None:
    sharded_sha256 = "ab" + "1" * 62
    flat_sha256 = "cd" + "2" * 62
    sharded = (
        tmp_path / "requests" / "ab" / f"{sharded_sha256}.json"
    )
    flat = tmp_path / "requests" / f"{flat_sha256}.json"
    sharded.parent.mkdir(parents=True)
    sharded.write_text("not parsed", encoding="utf-8")
    flat.write_text("also not parsed", encoding="utf-8")

    assert _completed_index_inventory(tmp_path) == frozenset(
        {sharded_sha256, flat_sha256}
    )


def test_completed_index_inventory_rejects_wrong_shard(tmp_path: Path) -> None:
    payload_sha256 = "ab" + "1" * 62
    wrong_shard = tmp_path / "requests" / "ff" / f"{payload_sha256}.json"
    wrong_shard.parent.mkdir(parents=True)
    wrong_shard.write_text("not parsed", encoding="utf-8")

    with pytest.raises(ValueError, match="shard differs"):
        _completed_index_inventory(tmp_path)
