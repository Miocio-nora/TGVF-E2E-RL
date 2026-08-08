from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.deepeyes_official_schedule import build_deepeyes_schedule
from tgvf_rl.data.deepeyes_official_schedule_index import (
    load_deepeyes_schedule_index,
    write_deepeyes_schedule_index,
)
from tgvf_rl.framework.verl import deepeyes_official_dataset as dataset_module

from tests.data.test_deepeyes_official_schedule import synthetic_official_pool


def test_index_round_trip_preserves_schedule_and_contains_only_consumed_rows(
    tmp_path: Path,
) -> None:
    samples = synthetic_official_pool()
    legacy = build_deepeyes_schedule(samples, mode="stratified")
    output = tmp_path / "index.json"
    file_sha256, identity_sha256, _ = write_deepeyes_schedule_index(samples, output)
    indexed = load_deepeyes_schedule_index(
        output,
        expected_file_sha256=file_sha256,
        expected_identity_sha256=identity_sha256,
    )
    assert indexed.schedule_identity_sha256 == legacy.identity_sha256
    assert [sample.sample_id for sample in indexed.train] == [
        legacy.samples[index].sample_id
        for batch in legacy.batches
        for index in batch
    ]
    assert [sample.sample_id for sample in indexed.probe] == [
        sample.sample_id for sample in legacy.probe
    ]
    assert len(indexed.train) + len(indexed.probe) + len(indexed.smoke) == 20_740
    assert not {
        sample.sample_id for sample in (*indexed.train, *indexed.probe)
    }.intersection(sample.sample_id for sample in indexed.smoke)


def test_index_fails_closed_on_file_and_parent_identity_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "index.json"
    file_sha256, identity_sha256, _ = write_deepeyes_schedule_index(
        synthetic_official_pool(), output
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["parent_artifacts"]["t1_final"]["samples_sha256"] = "f" * 64
    tampered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output.write_bytes(tampered)
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_deepeyes_schedule_index(
            output,
            expected_file_sha256=file_sha256,
            expected_identity_sha256=identity_sha256,
        )
    with pytest.raises(ValueError, match="index identity differs"):
        load_deepeyes_schedule_index(
            output,
            expected_file_sha256=hashlib.sha256(tampered).hexdigest(),
            expected_identity_sha256=identity_sha256,
        )


def test_consumed_image_sha_is_lazy_and_cached_once_per_path(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"first")
    dataset_module._observed_image_sha256.cache_clear()
    first = dataset_module._observed_image_sha256(str(image))
    assert first == hashlib.sha256(b"first").hexdigest()
    image.write_bytes(b"changed-after-first-consumption")
    assert dataset_module._observed_image_sha256(str(image)) == first
    info = dataset_module._observed_image_sha256.cache_info()
    assert info.misses == 1
    assert info.hits == 1
    dataset_module._observed_image_sha256.cache_clear()
    assert dataset_module._observed_image_sha256(str(image)) == hashlib.sha256(
        b"changed-after-first-consumption"
    ).hexdigest()
