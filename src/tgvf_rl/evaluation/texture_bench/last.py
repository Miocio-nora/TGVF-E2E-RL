"""Deterministic LAS&T 2D Texture Retrieval task preparation.

The upstream benchmark stores five views of one texture instance in each leaf
directory.  A quiz places an anchor in panel A, one other view of the same
instance in exactly one of panels B/C/D, and two views of distinct negative
instances in the remaining panels.

This module deliberately keeps source discovery/sampling separate from PNG
materialization.  The sampled :class:`LastQuiz` values are immutable and can
therefore be audited before any derived benchmark assets are written.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random
from types import MappingProxyType
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .io import write_bytes_idempotent


LAST_DATASET_NAME = "LAST_2D_Texture_Retrieval"
LAST_DEFAULT_SEED = 20_260_813
LAST_DEFAULT_QUIZZES_PER_DIRECTORY = 400
LAST_DEFAULT_PROMPT_PROFILE = "neutral_v1"
LAST_EXPECTED_IDENTITY_COUNT = 500
LAST_EXPECTED_IMAGES_PER_IDENTITY = 5
LAST_EXPECTED_SOURCE_IMAGE_SIZE = (512, 512)
LAST_PANEL_SIZE = (512, 512)
LAST_QUIZ_SIZE = (1024, 1024)
LAST_PNG_COMPRESSION_LEVEL = 1
LAST_CHOICES = ("B", "C", "D")
LAST_TRAINING_DIRECTORY = "AllDifferent_Traininig"


@dataclass(frozen=True, slots=True)
class LastDirectorySpec:
    """Explicit semantics for one physical LAS&T test directory."""

    physical_directory: str
    canonical_condition: str
    shape_relation: str
    background_style: str
    replicate: str


# Keep this mapping explicit: the upstream spelling and capitalization are part
# of the dataset identity.  ``AllDifferent_Traininig`` is intentionally absent.
LAST_DIRECTORY_SPECS = (
    LastDirectorySpec(
        physical_directory="All_Different_Big_Set",
        canonical_condition="different_shape_textured_background",
        shape_relation="different",
        background_style="textured",
        replicate="big_set",
    ),
    LastDirectorySpec(
        physical_directory="Same_Shape_different_texture_background",
        canonical_condition="same_shape_textured_background",
        shape_relation="same",
        background_style="textured",
        replicate="1",
    ),
    LastDirectorySpec(
        physical_directory="Same_Shape_different_texture_background_2",
        canonical_condition="same_shape_textured_background",
        shape_relation="same",
        background_style="textured",
        replicate="2",
    ),
    LastDirectorySpec(
        physical_directory="Same_Shape_different_texture_background_3",
        canonical_condition="same_shape_textured_background",
        shape_relation="same",
        background_style="textured",
        replicate="3",
    ),
    LastDirectorySpec(
        physical_directory="Same_shape_black_background_1",
        canonical_condition="same_shape_black_background",
        shape_relation="same",
        background_style="black",
        replicate="1",
    ),
    LastDirectorySpec(
        physical_directory="Same_shape_black_background_2",
        canonical_condition="same_shape_black_background",
        shape_relation="same",
        background_style="black",
        replicate="2",
    ),
    LastDirectorySpec(
        physical_directory="Textured_background_diffent_shape",
        canonical_condition="different_shape_textured_background",
        shape_relation="different",
        background_style="textured",
        replicate="1",
    ),
    LastDirectorySpec(
        physical_directory="different_shape_black_background",
        canonical_condition="different_shape_black_background",
        shape_relation="different",
        background_style="black",
        replicate="1",
    ),
)

LAST_DIRECTORY_MAP: Mapping[str, LastDirectorySpec] = MappingProxyType(
    {spec.physical_directory: spec for spec in LAST_DIRECTORY_SPECS}
)


# ``official_first_v1`` is the first Textures2D prompt shipped in upstream
# ``queries.py``.  It is retained verbatim (including its grammar) so reports
# can state that they used the official-first prompt rather than a paraphrase.
LAST_PROMPT_PROFILES: Mapping[str, str] = MappingProxyType(
    {
        "neutral_v1": (
            "Panel A is the reference. Which panel, B, C, or D, contains the "
            "same texture as panel A? Answer with exactly one letter: B, C, or D."
        ),
        "official_first_v1": (
            "Which panel B,C or D contain shape that have identical texture to "
            "the texture in panel A. Note that the shape on which the texture "
            "appear and the background  will be different for all panels. Your "
            "answer must be a single letter."
        ),
    }
)


@dataclass(frozen=True, slots=True)
class LastIdentity:
    """One source texture identity and its sorted image views."""

    physical_directory: str
    identity: str
    image_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class LastPanelSource:
    """The auditable source assigned to one rendered panel."""

    panel: str
    role: str
    identity: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class LastQuiz:
    """One fully sampled, but not yet rendered, four-panel quiz."""

    seed: int
    directory_spec: LastDirectorySpec
    local_ordinal: int
    answer: str
    panels: tuple[LastPanelSource, ...]

    def __post_init__(self) -> None:
        if self.answer not in LAST_CHOICES:
            raise ValueError("LAS&T quiz answer must be one of B, C, or D")
        if tuple(panel.panel for panel in self.panels) != ("A", *LAST_CHOICES):
            raise ValueError("LAS&T quiz panels must be ordered A, B, C, D")
        if self.panels[0].role != "anchor":
            raise ValueError("LAS&T panel A must be the anchor")
        positive = tuple(panel for panel in self.panels if panel.role == "positive")
        negatives = tuple(panel for panel in self.panels if panel.role == "negative")
        if len(positive) != 1 or len(negatives) != 2:
            raise ValueError("LAS&T quiz must have one positive and two negatives")
        if positive[0].panel != self.answer:
            raise ValueError("LAS&T answer must name the positive panel")
        if positive[0].identity != self.panels[0].identity:
            raise ValueError("LAS&T anchor and positive identities must match")
        if len({panel.identity for panel in negatives}) != 2:
            raise ValueError("LAS&T negative identities must be distinct")
        if any(panel.identity == self.panels[0].identity for panel in negatives):
            raise ValueError("LAS&T negatives must differ from the anchor")

    @property
    def sample_id(self) -> str:
        """Return an output-location-independent identity for this exact quiz."""

        payload = {
            "answer": self.answer,
            "local_ordinal": self.local_ordinal,
            "panels": [
                {
                    "identity": panel.identity,
                    "panel": panel.panel,
                    "path": panel.image_path.name,
                    "role": panel.role,
                }
                for panel in self.panels
            ],
            "physical_directory": self.directory_spec.physical_directory,
            "seed": self.seed,
        }
        digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()[:16]
        return (
            f"last2d:{self.seed}:{self.directory_spec.physical_directory}:"
            f"{self.local_ordinal:04d}:{digest}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_plain_int(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _directory_seed(seed: int, physical_directory: str) -> int:
    payload = f"last-2d-texture-retrieval-v1\0{seed}\0{physical_directory}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_last_prompt(prompt_profile: str) -> str:
    """Resolve a fixed prompt profile, rejecting silent prompt drift."""

    try:
        return LAST_PROMPT_PROFILES[prompt_profile]
    except KeyError as error:
        choices = ", ".join(sorted(LAST_PROMPT_PROFILES))
        raise ValueError(
            f"unknown LAS&T prompt profile {prompt_profile!r}; expected one of {choices}"
        ) from error


def validate_last_source(
    source_root: str | Path,
    *,
    expected_identity_count: int = LAST_EXPECTED_IDENTITY_COUNT,
    expected_images_per_identity: int = LAST_EXPECTED_IMAGES_PER_IDENTITY,
    expected_image_size: tuple[int, int] | None = LAST_EXPECTED_SOURCE_IMAGE_SIZE,
) -> dict[str, tuple[LastIdentity, ...]]:
    """Validate and return the eight canonical LAS&T test directories.

    Discovery is sorted at both the identity and image level.  By default this
    enforces the released snapshot contract of 500 identities with five JPEG
    views each in every physical test directory.
    """

    expected_identity_count = _require_plain_int(
        expected_identity_count, name="expected_identity_count", minimum=3
    )
    expected_images_per_identity = _require_plain_int(
        expected_images_per_identity,
        name="expected_images_per_identity",
        minimum=2,
    )
    if expected_image_size is not None:
        if len(expected_image_size) != 2 or any(
            type(value) is not int or value <= 0 for value in expected_image_size
        ):
            raise ValueError("expected_image_size must be a positive (width, height)")

    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"LAS&T source root is not a directory: {root}")

    discovered: dict[str, tuple[LastIdentity, ...]] = {}
    for spec in LAST_DIRECTORY_SPECS:
        physical_root = root / spec.physical_directory
        if not physical_root.is_dir():
            raise ValueError(
                f"LAS&T source is missing canonical directory: "
                f"{spec.physical_directory}"
            )
        identity_dirs = tuple(
            sorted(
                (entry for entry in physical_root.iterdir() if entry.is_dir()),
                key=lambda path: path.name,
            )
        )
        if len(identity_dirs) != expected_identity_count:
            raise ValueError(
                f"LAS&T directory {spec.physical_directory} has "
                f"{len(identity_dirs)} identities; expected {expected_identity_count}"
            )

        identities: list[LastIdentity] = []
        for identity_dir in identity_dirs:
            image_paths = tuple(
                sorted(
                    (
                        entry.resolve(strict=True)
                        for entry in identity_dir.iterdir()
                        if entry.is_file() and entry.suffix.lower() == ".jpg"
                    ),
                    key=lambda path: path.name,
                )
            )
            if len(image_paths) != expected_images_per_identity:
                raise ValueError(
                    f"LAS&T identity {spec.physical_directory}/{identity_dir.name} "
                    f"has {len(image_paths)} JPEGs; expected "
                    f"{expected_images_per_identity}"
                )
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as opened:
                        dimensions = tuple(int(value) for value in opened.size)
                        opened.verify()
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"LAS&T source image is not decodable: {image_path}"
                    ) from error
                if (
                    expected_image_size is not None
                    and dimensions != expected_image_size
                ):
                    raise ValueError(
                        f"LAS&T source image {image_path} has dimensions "
                        f"{dimensions}; expected {expected_image_size}"
                    )
            identities.append(
                LastIdentity(
                    physical_directory=spec.physical_directory,
                    identity=identity_dir.name,
                    image_paths=image_paths,
                )
            )
        discovered[spec.physical_directory] = tuple(identities)
    return discovered


def sample_last_quizzes(
    sources: Mapping[str, Sequence[LastIdentity]],
    *,
    quizzes_per_directory: int = LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
    seed: int = LAST_DEFAULT_SEED,
    max_samples: int | None = None,
) -> tuple[LastQuiz, ...]:
    """Purely sample a deterministic, balanced sequence of LAS&T quizzes.

    The returned order is round-robin across physical directories.  As a
    result, short smoke-test prefixes remain approximately balanced, and every
    ``max_samples=n`` result is exactly the first ``n`` rows of the full result.
    """

    quizzes_per_directory = _require_plain_int(
        quizzes_per_directory, name="quizzes_per_directory", minimum=1
    )
    seed = _require_plain_int(seed, name="seed", minimum=0)
    if max_samples is not None:
        max_samples = _require_plain_int(max_samples, name="max_samples", minimum=0)

    quizzes_by_directory: list[tuple[LastQuiz, ...]] = []
    for spec in LAST_DIRECTORY_SPECS:
        try:
            canonical_identities = tuple(sources[spec.physical_directory])
        except KeyError as error:
            raise ValueError(
                f"validated LAS&T sources omit {spec.physical_directory}"
            ) from error
        if len(canonical_identities) < quizzes_per_directory:
            raise ValueError(
                f"LAS&T directory {spec.physical_directory} cannot supply "
                f"{quizzes_per_directory} distinct anchor identities"
            )
        if len(canonical_identities) < 3:
            raise ValueError("LAS&T quiz sampling requires at least three identities")

        # Normalize caller-supplied mappings before touching the random stream.
        identities = sorted(
            (
                LastIdentity(
                    physical_directory=identity.physical_directory,
                    identity=identity.identity,
                    image_paths=tuple(
                        sorted(identity.image_paths, key=lambda path: path.name)
                    ),
                )
                for identity in canonical_identities
            ),
            key=lambda item: item.identity,
        )
        if len({identity.identity for identity in identities}) != len(identities):
            raise ValueError("LAS&T source identities must be unique per directory")
        for identity in identities:
            if identity.physical_directory != spec.physical_directory:
                raise ValueError("LAS&T identity is filed under the wrong directory")
            if len(identity.image_paths) < 2:
                raise ValueError("LAS&T identity requires at least two source views")

        rng = random.Random(_directory_seed(seed, spec.physical_directory))
        anchor_order = list(identities)
        rng.shuffle(anchor_order)
        directory_quizzes: list[LastQuiz] = []
        for local_ordinal, anchor_identity in enumerate(
            anchor_order[:quizzes_per_directory]
        ):
            anchor_path = anchor_identity.image_paths[0]
            positive_path = rng.choice(anchor_identity.image_paths[1:])
            negative_identities = rng.sample(
                [item for item in identities if item != anchor_identity], k=2
            )
            negative_sources = tuple(
                (identity, rng.choice(identity.image_paths))
                for identity in negative_identities
            )
            answer = rng.choice(LAST_CHOICES)
            negative_panels = iter(negative_sources)
            panel_sources = [
                LastPanelSource(
                    panel="A",
                    role="anchor",
                    identity=anchor_identity.identity,
                    image_path=anchor_path,
                )
            ]
            for panel in LAST_CHOICES:
                if panel == answer:
                    panel_sources.append(
                        LastPanelSource(
                            panel=panel,
                            role="positive",
                            identity=anchor_identity.identity,
                            image_path=positive_path,
                        )
                    )
                else:
                    negative_identity, negative_path = next(negative_panels)
                    panel_sources.append(
                        LastPanelSource(
                            panel=panel,
                            role="negative",
                            identity=negative_identity.identity,
                            image_path=negative_path,
                        )
                    )
            directory_quizzes.append(
                LastQuiz(
                    seed=seed,
                    directory_spec=spec,
                    local_ordinal=local_ordinal,
                    answer=answer,
                    panels=tuple(panel_sources),
                )
            )
        quizzes_by_directory.append(tuple(directory_quizzes))

    # Interleaving is deterministic because directory order is frozen above.
    all_quizzes = tuple(
        quizzes_by_directory[directory_index][local_ordinal]
        for local_ordinal in range(quizzes_per_directory)
        for directory_index in range(len(LAST_DIRECTORY_SPECS))
    )
    return all_quizzes if max_samples is None else all_quizzes[:max_samples]


def _load_panel_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            panel = opened.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"LAS&T panel source cannot be decoded: {path}") from error
    if panel.size != LAST_PANEL_SIZE:
        panel = panel.resize(LAST_PANEL_SIZE, Image.Resampling.LANCZOS)
    return panel


def _label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Newer Pillow accepts a size for its bundled font.  Keep a compatibility
    # fallback so task preparation does not depend on a host font file.
    try:
        return ImageFont.load_default(size=80)
    except TypeError:  # pragma: no cover - exercised only by older Pillow.
        return ImageFont.load_default()


def render_last_quiz_png(quiz: LastQuiz) -> bytes:
    """Render one quiz as a deterministic, lossless 1024x1024 RGB PNG."""

    canvas = Image.new("RGB", LAST_QUIZ_SIZE, color=(0, 0, 0))
    positions = {
        "A": (0, 0),
        "B": (512, 0),
        "C": (0, 512),
        "D": (512, 512),
    }
    font = _label_font()
    for source in quiz.panels:
        panel = _load_panel_image(source.image_path)
        draw = ImageDraw.Draw(panel)
        # The white glyph follows the official script; the narrow black stroke
        # keeps it readable on pale textures without changing the label color.
        draw.text(
            (10, 2),
            source.panel,
            fill=(255, 255, 255),
            font=font,
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        canvas.paste(panel, positions[source.panel])

    buffer = io.BytesIO()
    # Texture photographs are close to incompressible.  Level 1 remains
    # byte-deterministic and lossless while avoiding a large preparation-only
    # CPU cost for the 3,200-image paper-count-derived profile.
    canvas.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=LAST_PNG_COMPRESSION_LEVEL,
    )
    return buffer.getvalue()


def _source_identity_payload(quiz: LastQuiz, source_root: Path) -> dict[str, object]:
    panel_payload: dict[str, object] = {}
    for panel in quiz.panels:
        try:
            relative_path = panel.image_path.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"LAS&T panel source lies outside source_root: {panel.image_path}"
            ) from error
        panel_payload[panel.panel] = {
            "identity": panel.identity,
            "relative_path": relative_path,
            "role": panel.role,
            "sha256": _sha256_file(panel.image_path),
        }
    return {
        "schema_version": "last-source-identity-v1",
        "panels": panel_payload,
    }


def materialize_last_rows(
    source_root: str | Path,
    output_root: str | Path,
    *,
    prompt_profile: str = LAST_DEFAULT_PROMPT_PROFILE,
    quizzes_per_directory: int = LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
    seed: int = LAST_DEFAULT_SEED,
    max_samples: int | None = None,
    expected_identity_count: int = LAST_EXPECTED_IDENTITY_COUNT,
    expected_images_per_identity: int = LAST_EXPECTED_IMAGES_PER_IDENTITY,
    expected_image_size: tuple[int, int] | None = LAST_EXPECTED_SOURCE_IMAGE_SIZE,
) -> tuple[dict[str, object], ...]:
    """Validate, sample, render, and return CoreDevTask-compatible row dicts."""

    question = resolve_last_prompt(prompt_profile)
    resolved_source_root = Path(source_root).expanduser().resolve(strict=True)
    resolved_output_root = Path(output_root).expanduser().resolve()
    sources = validate_last_source(
        resolved_source_root,
        expected_identity_count=expected_identity_count,
        expected_images_per_identity=expected_images_per_identity,
        expected_image_size=expected_image_size,
    )
    quizzes = sample_last_quizzes(
        sources,
        quizzes_per_directory=quizzes_per_directory,
        seed=seed,
        max_samples=max_samples,
    )

    rows: list[dict[str, object]] = []
    for ordinal, quiz in enumerate(quizzes):
        image_path = (
            resolved_output_root
            / "images"
            / f"seed-{seed}"
            / quiz.directory_spec.physical_directory
            / f"{quiz.local_ordinal:04d}-{quiz.sample_id.rsplit(':', 1)[-1]}.png"
        )
        image_bytes = render_last_quiz_png(quiz)
        write_bytes_idempotent(image_path, image_bytes)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        source_identity = _source_identity_payload(quiz, resolved_source_root)
        spec = quiz.directory_spec
        metadata = [
            ["benchmark", "las_t_2d_texture_retrieval"],
            ["category", spec.canonical_condition],
            ["cycle_category", spec.physical_directory],
            ["condition_id", spec.canonical_condition],
            ["source_dir", spec.physical_directory],
            ["physical_directory", spec.physical_directory],
            ["canonical_condition", spec.canonical_condition],
            ["shape_relation", spec.shape_relation],
            ["background_style", spec.background_style],
            ["replicate", spec.replicate],
            ["prompt_profile", prompt_profile],
            ["quiz_seed", str(seed)],
            ["local_ordinal", str(quiz.local_ordinal)],
            [
                "source_identity_v1",
                _canonical_json_bytes(source_identity).decode("utf-8"),
            ],
        ]
        rows.append(
            {
                "ordinal": ordinal,
                "dataset": LAST_DATASET_NAME,
                "row_number": ordinal,
                "index": quiz.sample_id,
                "sample_id": quiz.sample_id,
                "question": question,
                "image_paths": [str(image_path)],
                "answer": quiz.answer,
                "options": [
                    ["B", "Panel B"],
                    ["C", "Panel C"],
                    ["D", "Panel D"],
                ],
                "metadata": metadata,
                "image_sha256s": [image_sha256],
                "image_dimensions": [[LAST_QUIZ_SIZE[0], LAST_QUIZ_SIZE[1]]],
            }
        )
    return tuple(rows)


def write_last_manifest(path: str | Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Write canonical JSONL rows and return the manifest SHA256."""

    manifest_path = Path(path).expanduser().resolve()
    payload = b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    write_bytes_idempotent(manifest_path, payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "LAST_CHOICES",
    "LAST_DATASET_NAME",
    "LAST_DEFAULT_PROMPT_PROFILE",
    "LAST_DEFAULT_QUIZZES_PER_DIRECTORY",
    "LAST_DEFAULT_SEED",
    "LAST_DIRECTORY_MAP",
    "LAST_DIRECTORY_SPECS",
    "LAST_EXPECTED_IDENTITY_COUNT",
    "LAST_EXPECTED_IMAGES_PER_IDENTITY",
    "LAST_EXPECTED_SOURCE_IMAGE_SIZE",
    "LAST_PANEL_SIZE",
    "LAST_PROMPT_PROFILES",
    "LAST_PNG_COMPRESSION_LEVEL",
    "LAST_QUIZ_SIZE",
    "LAST_TRAINING_DIRECTORY",
    "LastDirectorySpec",
    "LastIdentity",
    "LastPanelSource",
    "LastQuiz",
    "materialize_last_rows",
    "render_last_quiz_png",
    "resolve_last_prompt",
    "sample_last_quizzes",
    "validate_last_source",
    "write_last_manifest",
]
