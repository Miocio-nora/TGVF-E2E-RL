from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyes47KRuntimeBinding,
    DeepEyesSourceFileSpec,
    DeepEyesSourceValidationError,
    DeepEyesTaskKind,
    load_deepeyes47k_runtime,
    materialize_deepeyes47k_fixture,
)
from tgvf_rl.data import deepeyes47k_runtime as runtime_implementation


FORBIDDEN_SOURCE_PROMPT = "FORBIDDEN HISTORICAL DEEPEYES PROMPT"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _row(index: int) -> dict[str, object]:
    return {
        "images": [{"bytes": b"\x89PNG\r\n\x1a\n" + bytes([index])}],
        "prompt": FORBIDDEN_SOURCE_PROMPT,
        "extra_info": {
            "question": f"What is object {index}?",
            "ability": "perception",
            "split": "train",
        },
        "reward_model": {"ground_truth": f"answer-{index}", "style": "qa"},
        "data_source": "open_visual_qa",
    }


def _materialize(
    root: Path, *, seed: int = 42
) -> tuple[object, DeepEyes47KRuntimeBinding]:
    source_payload = b"fixture-source"
    source = DeepEyesSourceFileSpec(
        filename="fixture.parquet",
        rows=3,
        lfs_sha256=hashlib.sha256(source_payload).hexdigest(),
        byte_size=len(source_payload),
    )
    result = materialize_deepeyes47k_fixture(
        {source.filename: [_row(0), _row(1), _row(2)]},
        (source,),
        root,
        shuffle_seed=seed,
    )
    binding = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=result.manifest_file_sha256,
        content_sha256=result.content_sha256,
        shuffle_seed=seed,
        expected_sample_count=3,
    )
    return result, binding


def _rewrite_samples(
    root: Path, records: list[dict[str, object]], *, seed: int
) -> DeepEyes47KRuntimeBinding:
    samples_bytes = b"".join(_canonical(record) + b"\n" for record in records)
    (root / "samples.jsonl").write_bytes(samples_bytes)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["samples"]["sha256"] = hashlib.sha256(samples_bytes).hexdigest()
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    content_sha256 = hashlib.sha256(_canonical(descriptor)).hexdigest()
    manifest["content_sha256"] = content_sha256
    manifest_bytes = _canonical(manifest) + b"\n"
    (root / "manifest.json").write_bytes(manifest_bytes)
    return DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        content_sha256=content_sha256,
        shuffle_seed=seed,
        expected_sample_count=len(records),
    )


def test_runtime_loader_is_prompt_free_seed_bound_and_deterministic(
    tmp_path: Path,
) -> None:
    _, binding = _materialize(tmp_path / "first")
    _, second_binding = _materialize(tmp_path / "second")
    dataset = load_deepeyes47k_runtime(tmp_path / "first", binding=binding)
    repeated = load_deepeyes47k_runtime(tmp_path / "second", binding=second_binding)

    ordered_ids = tuple(
        json.loads(line)["sample_id"]
        for line in (tmp_path / "first" / "samples.jsonl").read_text().splitlines()
    )
    assert dataset.dataset_id == DEEPEYES47K_DATASET_ID
    assert dataset.snapshot == "fixture"
    assert len(dataset) == 3
    assert tuple(sample.sample_id for sample in dataset) == ordered_ids
    assert tuple(sample.sample_id for sample in dataset) == tuple(
        sample.sample_id for sample in iter(dataset)
    )
    assert dataset.iteration_identity_sha256 == repeated.iteration_identity_sha256
    assert tuple(sample.prompt_group_uid for sample in dataset) == tuple(
        sample.prompt_group_uid for sample in repeated
    )

    sample = dataset[0]
    assert sample.image_path.is_absolute() and sample.image_path.is_file()
    assert sample.question.startswith("What is object")
    assert sample.ground_truth.startswith("answer-")
    assert sample.data_source == "open_visual_qa"
    assert sample.task_kind is DeepEyesTaskKind.OPEN
    assert sample.metadata["ability"] == "perception"
    assert sample.metadata["style"] == "qa"
    assert sample.metadata["split"] == "train"
    assert sample.prompt_group_uid.startswith("tgvf-pilot-group:")
    assert not hasattr(sample, "prompt")
    assert FORBIDDEN_SOURCE_PROMPT not in (tmp_path / "first" / "samples.jsonl").read_text()


def test_runtime_binding_rejects_wrong_hash_seed_count_and_fixture_mode(
    tmp_path: Path,
) -> None:
    result, binding = _materialize(tmp_path / "artifact")

    wrong_manifest = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256="0" * 64,
        content_sha256=binding.content_sha256,
        shuffle_seed=42,
        expected_sample_count=3,
    )
    with pytest.raises(DeepEyesSourceValidationError, match="manifest file hash"):
        load_deepeyes47k_runtime(tmp_path / "artifact", binding=wrong_manifest)

    wrong_seed = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=binding.manifest_file_sha256,
        content_sha256=binding.content_sha256,
        shuffle_seed=43,
        expected_sample_count=3,
    )
    with pytest.raises(DeepEyesSourceValidationError, match="shuffle-seed"):
        load_deepeyes47k_runtime(tmp_path / "artifact", binding=wrong_seed)

    wrong_count = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=binding.manifest_file_sha256,
        content_sha256=binding.content_sha256,
        shuffle_seed=42,
        expected_sample_count=2,
    )
    with pytest.raises(DeepEyesSourceValidationError, match="row-count"):
        load_deepeyes47k_runtime(tmp_path / "artifact", binding=wrong_count)

    formal = DeepEyes47KRuntimeBinding.formal(
        manifest_file_sha256=result.manifest_file_sha256,
        content_sha256=result.content_sha256,
        shuffle_seed=42,
    )
    assert formal.expected_sample_count == DEEPEYES47K_TOTAL_ROWS
    with pytest.raises(DeepEyesSourceValidationError, match="fixture/formal"):
        load_deepeyes47k_runtime(tmp_path / "artifact", binding=formal)


def test_runtime_rejects_noncanonical_manifest_content_hash(tmp_path: Path) -> None:
    _, binding = _materialize(tmp_path / "artifact")
    manifest_path = tmp_path / "artifact" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["snapshot"] = "tampered"
    manifest_bytes = _canonical(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    tampered_binding = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        content_sha256=binding.content_sha256,
        shuffle_seed=42,
        expected_sample_count=3,
    )

    with pytest.raises(DeepEyesSourceValidationError, match="content hash"):
        load_deepeyes47k_runtime(
            tmp_path / "artifact", binding=tampered_binding
        )


@pytest.mark.parametrize("forbidden_field", ["prompt", "messages"])
def test_runtime_rejects_forbidden_sample_schema(
    tmp_path: Path, forbidden_field: str
) -> None:
    root = tmp_path / "artifact"
    _materialize(root)
    records = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    records[0][forbidden_field] = FORBIDDEN_SOURCE_PROMPT
    binding = _rewrite_samples(root, records, seed=42)

    with pytest.raises(DeepEyesSourceValidationError, match="forbidden schema"):
        load_deepeyes47k_runtime(root, binding=binding)


def test_runtime_rejects_image_escape_symlink_and_sha_tamper(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _, binding = _materialize(root)
    dataset = load_deepeyes47k_runtime(root, binding=binding)
    image_path = dataset[0].image_path
    original = image_path.read_bytes()
    outside = tmp_path / "outside.png"
    outside.write_bytes(original)
    image_path.unlink()
    image_path.symlink_to(outside)

    with pytest.raises(DeepEyesSourceValidationError, match="symlink"):
        load_deepeyes47k_runtime(root, binding=binding)

    image_path.unlink()
    image_path.write_bytes(b"tampered")
    with pytest.raises(DeepEyesSourceValidationError, match="image SHA-256"):
        load_deepeyes47k_runtime(root, binding=binding)


def test_runtime_rejects_lexical_image_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _materialize(root)
    records = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    records[0]["image"]["path"] = "../outside.png"
    binding = _rewrite_samples(root, records, seed=42)

    with pytest.raises(DeepEyesSourceValidationError, match="escapes images"):
        load_deepeyes47k_runtime(root, binding=binding)


def test_runtime_rejects_images_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _, binding = _materialize(root)
    outside_images = tmp_path / "outside-images"
    (root / "images").rename(outside_images)
    (root / "images").symlink_to(outside_images, target_is_directory=True)

    with pytest.raises(DeepEyesSourceValidationError, match="images root"):
        load_deepeyes47k_runtime(root, binding=binding)


def test_runtime_rechecks_image_bytes_after_streaming_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    _, binding = _materialize(root)
    original_validator = runtime_implementation.validate_materialized_deepeyes47k

    def validate_then_mutate(output_root: Path) -> dict[str, object]:
        result = original_validator(output_root)
        first = json.loads((output_root / "samples.jsonl").read_text().splitlines()[0])
        (output_root / first["image"]["path"]).write_bytes(b"post-validation tamper")
        return result

    monkeypatch.setattr(
        runtime_implementation,
        "validate_materialized_deepeyes47k",
        validate_then_mutate,
    )
    with pytest.raises(DeepEyesSourceValidationError, match="changed after"):
        load_deepeyes47k_runtime(root, binding=binding)
