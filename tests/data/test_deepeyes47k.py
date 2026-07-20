from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data import (
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_SOURCE_FILES,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyesDependencyError,
    DeepEyesSourceFileSpec,
    DeepEyesSourceValidationError,
    DeepEyesTaskKind,
    classify_deepeyes_task_kind,
    materialize_deepeyes47k_fixture,
    sanitize_deepeyes47k_row,
    stable_deepeyes47k_sample_id,
    validate_materialized_deepeyes47k,
    verify_deepeyes47k_source_files,
)
from tgvf_rl.data import deepeyes47k as implementation


def _spec(filename: str, payload: bytes, rows: int) -> DeepEyesSourceFileSpec:
    return DeepEyesSourceFileSpec(
        filename=filename,
        rows=rows,
        lfs_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


def _row(
    index: int,
    *,
    prompt: str = "FORBIDDEN HISTORICAL CROP PROMPT",
    question: str | None = None,
    answer: object | None = None,
    data_source: str = "open_visual_qa",
) -> dict[str, object]:
    return {
        "images": [
            {
                "bytes": b"\x89PNG\r\n\x1a\n" + bytes([index]),
                "path": f"source-{index}.png",
            }
        ],
        "prompt": [{"role": "user", "content": prompt}],
        "extra_info": {
            "question": question or f"What is visible in image {index}?",
            "ability": "perception",
            "split": "train",
            "unretained": "discard me",
        },
        "reward_model": {
            "ground_truth": f"answer-{index}" if answer is None else answer,
            "style": "qa",
        },
        "data_source": data_source,
        "unretained_top_level": "discard me too",
    }


def test_official_snapshot_sources_are_exactly_pinned() -> None:
    assert DEEPEYES47K_SNAPSHOT == "5546681e28fa2eda9f60a9ea9dd0cf291216ded3"
    assert [
        (item.filename, item.rows, item.lfs_sha256, item.byte_size)
        for item in DEEPEYES47K_SOURCE_FILES
    ] == [
        (
            "data_0.1.2_visual_toolbox_v2.parquet",
            22_362,
            "42992bf5de25e8d766f820fb9730ece275563ba80dd41e3377bf678c9ba2c2c1",
            990_263_397,
        ),
        (
            "data_thinklite_reasoning_acc.parquet",
            11_031,
            "660cea5ff8f74d19f993b575f30b6f5406b6c330dd8f9aacc6be59e299238967",
            1_656_152_904,
        ),
        (
            "data_v0.8_visual_toolbox_v2.parquet",
            13_659,
            "96fc256e6f73e098c1b586f1c37baad616ecbddf1105bfca71aa07a5dda7da5a",
            2_198_504_506,
        ),
    ]
    assert (
        sum(item.rows for item in DEEPEYES47K_SOURCE_FILES)
        == DEEPEYES47K_TOTAL_ROWS
        == 47_052
    )


def test_sanitizer_never_reads_or_emits_source_prompt_and_ids_are_stable() -> None:
    source = _spec("fixture.parquet", b"source", 2)
    first = sanitize_deepeyes47k_row(_row(0), source_spec=source, source_row_index=0)
    changed_prompt = sanitize_deepeyes47k_row(
        _row(0, prompt="A DIFFERENT FORBIDDEN PROMPT"),
        source_spec=source,
        source_row_index=0,
    )
    record = first.manifest_record(image_path=f"images/{first.image_sha256}.png")

    assert first == changed_prompt
    assert first.sample_id == stable_deepeyes47k_sample_id(
        source_file="fixture.parquet", source_row_index=0
    )
    assert first.image_sha256 == hashlib.sha256(first.image_bytes).hexdigest()
    assert first.sample_id != stable_deepeyes47k_sample_id(
        source_file="fixture.parquet", source_row_index=1
    )
    assert set(record) == {
        "sample_id",
        "image",
        "extra_info",
        "reward_model",
        "data_source",
        "task_kind",
        "provenance",
        "ability",
        "style",
        "split",
    }
    assert record["style"] == "qa"
    serialized = json.dumps(record, sort_keys=True)
    assert "FORBIDDEN" not in serialized
    assert "prompt" not in serialized.casefold()
    assert "unretained" not in serialized

    wrong_shape = _row(0)
    wrong_shape["image"] = wrong_shape.pop("images")
    with pytest.raises(ValueError, match="images must contain exactly one"):
        sanitize_deepeyes47k_row(
            wrong_shape,
            source_spec=source,
            source_row_index=0,
        )


@pytest.mark.parametrize(
    ("question", "ground_truth", "data_source", "expected"),
    [
        (
            "Choose one:\n(A) red\n(B) blue\n(C) green",
            "B",
            "general_vqa",
            DeepEyesTaskKind.MCQ,
        ),
        (
            r"Compute \\frac{3}{4}+1.",
            "7/4",
            "general_reasoning",
            DeepEyesTaskKind.MATH,
        ),
        (
            "Which object is closest?",
            "the cup",
            "open_visual_qa",
            DeepEyesTaskKind.OPEN,
        ),
    ],
)
def test_task_kind_classification_is_deterministic(
    question: str,
    ground_truth: object,
    data_source: str,
    expected: DeepEyesTaskKind,
) -> None:
    first = classify_deepeyes_task_kind(
        question=question,
        ground_truth=ground_truth,
        data_source=data_source,
    )
    second = classify_deepeyes_task_kind(
        question=question,
        ground_truth=ground_truth,
        data_source=data_source,
    )
    assert first is second is expected


def test_source_verifier_checks_size_sha_and_rows(tmp_path: Path) -> None:
    payloads = {
        "a.parquet": b"first-fixture",
        "b.parquet": b"second-fixture",
        "c.parquet": b"third-fixture",
    }
    rows = {"a.parquet": 2, "b.parquet": 3, "c.parquet": 1}
    specs = tuple(
        _spec(name, payload, rows[name]) for name, payload in payloads.items()
    )
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    verified = verify_deepeyes47k_source_files(
        tmp_path,
        source_specs=specs,
        row_count_reader=lambda path: rows[path.name],
        snapshot="fixture",
    )
    assert verified.total_rows == 6
    assert tuple(item.filename for item in verified.files) == tuple(payloads)

    (tmp_path / "b.parquet").write_bytes(b"tampered-fixture")
    with pytest.raises(DeepEyesSourceValidationError, match="byte size mismatch"):
        verify_deepeyes47k_source_files(
            tmp_path,
            source_specs=specs,
            row_count_reader=lambda path: rows[path.name],
            snapshot="fixture",
        )


def test_real_parquet_path_fails_closed_without_pyarrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_pyarrow(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow.parquet":
            raise ImportError("fixture blocks pyarrow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pyarrow)
    with pytest.raises(DeepEyesDependencyError, match="pyarrow is required"):
        implementation._load_pyarrow_parquet()


def test_fixture_materialization_is_deterministic_and_seed_is_mandatory(
    tmp_path: Path,
) -> None:
    specs = (
        _spec("a.parquet", b"a", 2),
        _spec("b.parquet", b"b", 2),
        _spec("c.parquet", b"c", 2),
    )
    rows = {
        spec.filename: [_row(index * 2), _row(index * 2 + 1)]
        for index, spec in enumerate(specs)
    }
    first = materialize_deepeyes47k_fixture(
        rows, specs, tmp_path / "first", shuffle_seed=42
    )
    second = materialize_deepeyes47k_fixture(
        rows, specs, tmp_path / "second", shuffle_seed=42
    )

    assert first.sample_count == 6
    assert first.content_sha256 == second.content_sha256
    assert first.samples_sha256 == second.samples_sha256
    assert (tmp_path / "first" / "samples.jsonl").read_bytes() == (
        tmp_path / "second" / "samples.jsonl"
    ).read_bytes()
    all_output_bytes = b"".join(
        path.read_bytes() for path in (tmp_path / "first").rglob("*") if path.is_file()
    )
    assert b"FORBIDDEN HISTORICAL CROP PROMPT" not in all_output_bytes
    assert validate_materialized_deepeyes47k(tmp_path / "first") == {
        "sample_count": 6,
        "samples_sha256": first.samples_sha256,
        "content_sha256": first.content_sha256,
        "fixture": True,
    }

    with pytest.raises(TypeError, match="shuffle_seed"):
        materialize_deepeyes47k_fixture(rows, specs, tmp_path / "missing-seed")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="shuffle_seed"):
        materialize_deepeyes47k_fixture(
            rows, specs, tmp_path / "bool-seed", shuffle_seed=True
        )


def test_materialized_image_tampering_is_detected(tmp_path: Path) -> None:
    spec = _spec("fixture.parquet", b"fixture", 1)
    output = tmp_path / "artifact"
    materialize_deepeyes47k_fixture(
        {spec.filename: [_row(0)]}, (spec,), output, shuffle_seed=7
    )
    record = json.loads((output / "samples.jsonl").read_text().splitlines()[0])
    (output / record["image"]["path"]).write_bytes(b"tampered")

    with pytest.raises(DeepEyesSourceValidationError, match="image SHA-256"):
        validate_materialized_deepeyes47k(output)
