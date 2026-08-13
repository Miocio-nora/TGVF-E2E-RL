from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tgvf_rl.evaluation.texture_bench.last import (
    LAST_DATASET_NAME,
    LAST_DEFAULT_SEED,
    LAST_DIRECTORY_MAP,
    LAST_DIRECTORY_SPECS,
    LAST_PROMPT_PROFILES,
    LAST_TRAINING_DIRECTORY,
    LastIdentity,
    materialize_last_rows,
    resolve_last_prompt,
    sample_last_quizzes,
    validate_last_source,
    write_last_manifest,
)
from tgvf_rl.evaluation.texture_bench.task import TextureTask


FIXTURE_IDENTITIES = 4
FIXTURE_VIEWS = 5
FIXTURE_IMAGE_SIZE = (16, 16)


def _fixture_color(
    directory_index: int, identity_index: int, view: int
) -> tuple[int, int, int]:
    return (
        20 + directory_index * 17,
        20 + identity_index * 37,
        20 + view * 29,
    )


def _make_source_fixture(root: Path) -> Path:
    root.mkdir()
    # The excluded training directory is intentionally malformed.  Its presence
    # must not affect validation or task sampling.
    (root / LAST_TRAINING_DIRECTORY).mkdir()
    for directory_index, spec in reversed(tuple(enumerate(LAST_DIRECTORY_SPECS))):
        physical_root = root / spec.physical_directory
        physical_root.mkdir()
        for identity_index in reversed(range(FIXTURE_IDENTITIES)):
            identity_root = physical_root / f"identity-{identity_index:02d}"
            identity_root.mkdir()
            for view in reversed(range(FIXTURE_VIEWS)):
                Image.new(
                    "RGB",
                    FIXTURE_IMAGE_SIZE,
                    color=_fixture_color(directory_index, identity_index, view),
                ).save(identity_root / f"{view}.jpg", format="JPEG", quality=95)
    return root


def _validate_fixture(root: Path) -> dict[str, tuple[LastIdentity, ...]]:
    return validate_last_source(
        root,
        expected_identity_count=FIXTURE_IDENTITIES,
        expected_images_per_identity=FIXTURE_VIEWS,
        expected_image_size=FIXTURE_IMAGE_SIZE,
    )


def test_canonical_directory_mapping_is_explicit_and_excludes_training() -> None:
    expected = {
        "All_Different_Big_Set": (
            "different_shape_textured_background",
            "different",
            "textured",
            "big_set",
        ),
        "Same_Shape_different_texture_background": (
            "same_shape_textured_background",
            "same",
            "textured",
            "1",
        ),
        "Same_Shape_different_texture_background_2": (
            "same_shape_textured_background",
            "same",
            "textured",
            "2",
        ),
        "Same_Shape_different_texture_background_3": (
            "same_shape_textured_background",
            "same",
            "textured",
            "3",
        ),
        "Same_shape_black_background_1": (
            "same_shape_black_background",
            "same",
            "black",
            "1",
        ),
        "Same_shape_black_background_2": (
            "same_shape_black_background",
            "same",
            "black",
            "2",
        ),
        "Textured_background_diffent_shape": (
            "different_shape_textured_background",
            "different",
            "textured",
            "1",
        ),
        "different_shape_black_background": (
            "different_shape_black_background",
            "different",
            "black",
            "1",
        ),
    }

    assert len(LAST_DIRECTORY_SPECS) == 8
    assert len({spec.canonical_condition for spec in LAST_DIRECTORY_SPECS}) == 4
    assert LAST_TRAINING_DIRECTORY not in LAST_DIRECTORY_MAP
    assert {
        name: (
            spec.canonical_condition,
            spec.shape_relation,
            spec.background_style,
            spec.replicate,
        )
        for name, spec in LAST_DIRECTORY_MAP.items()
    } == expected


def test_source_validation_sorts_every_identity_and_view(tmp_path: Path) -> None:
    source_root = _make_source_fixture(tmp_path / "source")

    sources = _validate_fixture(source_root)

    assert tuple(sources) == tuple(
        spec.physical_directory for spec in LAST_DIRECTORY_SPECS
    )
    for identities in sources.values():
        assert [item.identity for item in identities] == [
            f"identity-{index:02d}" for index in range(FIXTURE_IDENTITIES)
        ]
        for identity in identities:
            assert [path.name for path in identity.image_paths] == [
                f"{view}.jpg" for view in range(FIXTURE_VIEWS)
            ]
            assert all(path.is_absolute() for path in identity.image_paths)


def test_sampling_is_deterministic_balanced_and_prefix_stable(tmp_path: Path) -> None:
    sources = _validate_fixture(_make_source_fixture(tmp_path / "source"))
    # Prove the pure sampler normalizes both identity and view ordering rather
    # than inheriting a caller's mapping order.
    reversed_sources = {
        name: tuple(
            LastIdentity(
                physical_directory=item.physical_directory,
                identity=item.identity,
                image_paths=tuple(reversed(item.image_paths)),
            )
            for item in reversed(identities)
        )
        for name, identities in reversed(tuple(sources.items()))
    }

    full = sample_last_quizzes(sources, quizzes_per_directory=3)
    repeated = sample_last_quizzes(reversed_sources, quizzes_per_directory=3)
    prefix = sample_last_quizzes(sources, quizzes_per_directory=3, max_samples=11)

    assert full == repeated
    assert prefix == full[:11]
    assert len(full) == 8 * 3
    assert Counter(quiz.directory_spec.physical_directory for quiz in full) == Counter(
        {spec.physical_directory: 3 for spec in LAST_DIRECTORY_SPECS}
    )
    assert tuple(quiz.directory_spec.physical_directory for quiz in full[:8]) == tuple(
        spec.physical_directory for spec in LAST_DIRECTORY_SPECS
    )
    assert full != sample_last_quizzes(
        sources, quizzes_per_directory=3, seed=LAST_DEFAULT_SEED + 1
    )

    for quiz in full:
        anchor = quiz.panels[0]
        positive = next(panel for panel in quiz.panels if panel.role == "positive")
        negatives = tuple(panel for panel in quiz.panels if panel.role == "negative")
        assert positive.panel == quiz.answer
        assert positive.identity == anchor.identity
        assert positive.image_path != anchor.image_path
        assert len({panel.identity for panel in negatives}) == 2
        assert anchor.identity not in {panel.identity for panel in negatives}


def test_materialized_rows_are_coredev_compatible_and_bind_sources(
    tmp_path: Path,
) -> None:
    source_root = _make_source_fixture(tmp_path / "source")
    output_root = tmp_path / "prepared"

    rows = materialize_last_rows(
        source_root,
        output_root,
        quizzes_per_directory=2,
        max_samples=8,
        expected_identity_count=FIXTURE_IDENTITIES,
        expected_images_per_identity=FIXTURE_VIEWS,
        expected_image_size=FIXTURE_IMAGE_SIZE,
    )

    assert len(rows) == 8
    assert tuple(row["ordinal"] for row in rows) == tuple(range(8))
    assert {row["dataset"] for row in rows} == {LAST_DATASET_NAME}
    assert {row["question"] for row in rows} == {LAST_PROMPT_PROFILES["neutral_v1"]}
    assert tuple(dict(row["metadata"])["physical_directory"] for row in rows) == tuple(
        spec.physical_directory for spec in LAST_DIRECTORY_SPECS
    )

    for row in rows:
        # TextureTask mirrors the exact JSON fields accepted by CoreDevTask but
        # stays independent of the CUDA/vLLM evaluator import tree.
        task = TextureTask(**row)
        assert task.sample_id == task.index
        assert task.answer in {"B", "C", "D"}
        assert task.options == (
            ("B", "Panel B"),
            ("C", "Panel C"),
            ("D", "Panel D"),
        )
        assert task.image_dimensions == ((1024, 1024),)
        image_path = Path(task.image_paths[0])
        assert image_path.is_absolute()
        assert image_path.suffix == ".png"
        image_bytes = image_path.read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == task.image_sha256s[0]
        with Image.open(image_path) as opened:
            assert opened.format == "PNG"
            assert opened.mode == "RGB"
            assert opened.size == (1024, 1024)
            # Each label is white and located near the top-left of its panel.
            for x, y in ((0, 0), (512, 0), (0, 512), (512, 512)):
                colors = opened.crop((x, y, x + 120, y + 100)).getcolors(
                    maxcolors=120 * 100
                )
                assert colors is not None
                assert any(color == (255, 255, 255) for _count, color in colors)

        metadata = dict(task.metadata)
        assert metadata["condition_id"] == metadata["category"]
        assert metadata["source_dir"] == metadata["physical_directory"]
        source_identity = json.loads(metadata["source_identity_v1"])
        assert source_identity["schema_version"] == "last-source-identity-v1"
        panels = source_identity["panels"]
        assert panels["A"]["role"] == "anchor"
        assert panels[task.answer]["role"] == "positive"
        assert panels["A"]["identity"] == panels[task.answer]["identity"]
        assert all(len(panel["sha256"]) == 64 for panel in panels.values())

    manifest_path = output_root / "tasks.jsonl"
    manifest_sha256 = write_last_manifest(manifest_path, rows)
    assert manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    decoded = tuple(json.loads(line) for line in manifest_path.read_text().splitlines())
    assert decoded == rows


def test_fixed_prompt_profiles_and_validation_fail_closed(tmp_path: Path) -> None:
    source_root = _make_source_fixture(tmp_path / "source")

    assert resolve_last_prompt("official_first_v1").startswith("Which panel B,C or D")
    with pytest.raises(ValueError, match="unknown LAS&T prompt profile"):
        resolve_last_prompt("latest")

    missing_view = (
        source_root
        / LAST_DIRECTORY_SPECS[0].physical_directory
        / "identity-00"
        / "4.jpg"
    )
    missing_view.unlink()
    with pytest.raises(ValueError, match="has 4 JPEGs; expected 5"):
        _validate_fixture(source_root)


def test_materialized_quiz_images_are_immutable(tmp_path: Path) -> None:
    source_root = _make_source_fixture(tmp_path / "source")
    output_root = tmp_path / "prepared"
    rows = materialize_last_rows(
        source_root,
        output_root,
        quizzes_per_directory=1,
        max_samples=1,
        expected_identity_count=FIXTURE_IDENTITIES,
        expected_images_per_identity=FIXTURE_VIEWS,
        expected_image_size=FIXTURE_IMAGE_SIZE,
    )
    image_path = Path(rows[0]["image_paths"][0])
    image_path.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        materialize_last_rows(
            source_root,
            output_root,
            quizzes_per_directory=1,
            max_samples=1,
            expected_identity_count=FIXTURE_IDENTITIES,
            expected_images_per_identity=FIXTURE_VIEWS,
            expected_image_size=FIXTURE_IMAGE_SIZE,
        )


def test_default_release_contract_is_400_quizzes_per_physical_directory() -> None:
    # This guards the public benchmark protocol without constructing the full
    # 20,000-image release fixture inside a unit test.
    from tgvf_rl.evaluation.texture_bench.last import (
        LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
        LAST_EXPECTED_IDENTITY_COUNT,
        LAST_EXPECTED_IMAGES_PER_IDENTITY,
    )

    assert LAST_DEFAULT_QUIZZES_PER_DIRECTORY == 400
    assert LAST_EXPECTED_IDENTITY_COUNT == 500
    assert LAST_EXPECTED_IMAGES_PER_IDENTITY == 5
