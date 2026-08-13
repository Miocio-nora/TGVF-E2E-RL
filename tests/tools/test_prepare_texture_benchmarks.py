from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from tgvf_rl.evaluation.texture_bench.io import validate_benchmark_identity
from tgvf_rl.evaluation.texture_bench.last import (
    LAST_DATASET_NAME,
    LAST_DEFAULT_QUIZZES_PER_DIRECTORY,
    LAST_DEFAULT_SEED,
    LAST_DIRECTORY_SPECS,
)
from tgvf_rl.evaluation.texture_bench.mmad import (
    MMAD_OFFICIAL_QUESTION_COUNT,
    MMAD_PINNED_JSON_SHA256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "prepare_texture_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("prepare_texture_benchmarks", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _save_image(
    path: Path,
    *,
    size: tuple[int, int],
    color: tuple[int, int, int],
    image_format: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=image_format)


def _last_source_fixture(root: Path) -> Path:
    for directory_index, spec in enumerate(LAST_DIRECTORY_SPECS):
        for identity_index in range(3):
            for view_index in range(2):
                _save_image(
                    root
                    / spec.physical_directory
                    / f"identity-{identity_index}"
                    / f"{view_index}.jpg",
                    size=(16, 12),
                    color=(
                        20 + directory_index,
                        30 + identity_index,
                        40 + view_index,
                    ),
                    image_format="JPEG",
                )
    return root


def _mmad_source_fixture(root: Path) -> Path:
    query = "DS-MVTec/widget/test/broken/000.png"
    template = "MVTec-AD/widget/train/good/000.png"
    _save_image(
        root / query,
        size=(80, 40),
        color=(220, 20, 20),
        image_format="PNG",
    )
    _save_image(
        root / template,
        size=(40, 80),
        color=(20, 220, 20),
        image_format="PNG",
    )
    annotation = {
        query: {
            "conversation": [
                {
                    "Question": "Which material is shown?",
                    "Answer": "Y",
                    "Options": {"X": "metal", "Y": "fabric"},
                    "type": "Object Details",
                    "annotation": True,
                },
                {
                    "Question": "Is there a defect?",
                    "Answer": "A",
                    "Options": {"A": "Yes.", "B": "No."},
                    "type": "Anomaly Detection",
                    "annotation": True,
                },
            ],
            "mask_path": "missing-mask-must-not-be-read.png",
            "random_templates": [template],
            "similar_templates": [template],
        }
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "mmad.json").write_text(json.dumps(annotation), encoding="utf-8")
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_defaults_bind_the_public_full_protocol_and_source_revision() -> None:
    last = TOOL._build_parser().parse_args(["last"])
    anomaly = TOOL._build_parser().parse_args(["mmad"])

    assert last.seed == LAST_DEFAULT_SEED == 20_260_813
    assert last.quizzes_per_directory == LAST_DEFAULT_QUIZZES_PER_DIRECTORY == 400
    assert last.prefix is None
    assert anomaly.shot == 1
    assert anomaly.template_kind == "random"
    assert anomaly.prefix is None
    assert MMAD_OFFICIAL_QUESTION_COUNT == 39_670
    assert MMAD_PINNED_JSON_SHA256 == (
        "639343b491bc67b2abb3c5d719f221ce27f83b2ed97948f4e88055aaa31f1c1e"
    )


def test_direct_script_help_bootstraps_src_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--help"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{last,mmad,suite}" in result.stdout
    assert "Traceback" not in result.stderr
    assert "\u200b" not in TOOL_PATH.read_text(encoding="utf-8")


def test_last_mmad_and_suite_smoke_are_reproducible_and_identity_bound(
    tmp_path: Path, capsys: object
) -> None:
    last_source = _last_source_fixture(tmp_path / "last-source")
    last_output = tmp_path / "prepared-last"
    last_argv = [
        "last",
        "--source",
        str(last_source),
        "--output",
        str(last_output),
        "--prefix",
        "3",
        "--quizzes-per-directory",
        "2",
        "--expected-identities",
        "3",
        "--expected-images-per-identity",
        "2",
        "--expected-width",
        "16",
        "--expected-height",
        "12",
    ]
    assert TOOL.main(last_argv) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert first_summary["command"] == "last"
    assert first_summary["task_count"] == 3
    assert TOOL.main(last_argv) == 0
    second_summary = json.loads(capsys.readouterr().out)
    assert first_summary == second_summary
    last_rows = _read_jsonl(last_output / "tasks.jsonl")
    assert [row["ordinal"] for row in last_rows] == [0, 1, 2]
    assert {row["dataset"] for row in last_rows} == {LAST_DATASET_NAME}
    assert [dict(row["metadata"])["physical_directory"] for row in last_rows] == [
        spec.physical_directory for spec in LAST_DIRECTORY_SPECS[:3]
    ]
    validate_benchmark_identity(last_output / "identity.json", verify_tasks=True)

    anomaly_source = _mmad_source_fixture(tmp_path / "mmad-source")
    one_shot_output = tmp_path / "prepared-mmad-1shot"
    one_shot_argv = [
        "mmad",
        "--source",
        str(anomaly_source),
        "--output",
        str(one_shot_output),
        "--shot",
        "1",
        "--prefix",
        "2",
        "--allow-unpinned-source",
    ]
    assert TOOL.main(one_shot_argv) == 0
    one_shot_summary = json.loads(capsys.readouterr().out)
    assert one_shot_summary["task_count"] == 2
    assert TOOL.main(one_shot_argv) == 0
    assert json.loads(capsys.readouterr().out) == one_shot_summary
    one_shot_rows = _read_jsonl(one_shot_output / "tasks.jsonl")
    assert {row["dataset"] for row in one_shot_rows} == {"MMAD"}
    assert {tuple(row["image_dimensions"][0]) for row in one_shot_rows} == {(1048, 560)}
    assert {
        dict(row["metadata"])["effective_image_layout"] for row in one_shot_rows
    } == {"normal_template_left__query_right"}
    assert "missing-mask" not in json.dumps(one_shot_rows)
    validate_benchmark_identity(one_shot_output / "identity.json", verify_tasks=True)

    zero_shot_output = tmp_path / "prepared-mmad-0shot"
    assert (
        TOOL.main(
            [
                "mmad",
                "--source",
                str(anomaly_source),
                "--output",
                str(zero_shot_output),
                "--shot",
                "0",
                "--prefix",
                "1",
                "--allow-unpinned-source",
            ]
        )
        == 0
    )
    zero_summary = json.loads(capsys.readouterr().out)
    assert zero_summary["benchmark_id"] == "mmad-0shot"
    zero_row = _read_jsonl(zero_shot_output / "tasks.jsonl")[0]
    assert zero_row["image_paths"] == [
        str((anomaly_source / "DS-MVTec/widget/test/broken/000.png").resolve())
    ]
    assert dict(zero_row["metadata"])["effective_image_layout"] == "query_only"

    suite_output = tmp_path / "prepared-suite"
    assert (
        TOOL.main(
            [
                "suite",
                "--last-manifest",
                str(last_output / "tasks.jsonl"),
                "--mmad-manifest",
                str(one_shot_output / "tasks.jsonl"),
                "--output",
                str(suite_output),
                "--prefix",
                "1",
            ]
        )
        == 0
    )
    suite_summary = json.loads(capsys.readouterr().out)
    assert suite_summary["task_count"] == 2
    assert suite_summary["benchmark_id"] == (
        f"{first_summary['benchmark_id']}__{one_shot_summary['benchmark_id']}"
    )
    suite_rows = _read_jsonl(suite_output / "tasks.jsonl")
    assert [row["ordinal"] for row in suite_rows] == [0, 1]
    assert [row["row_number"] for row in suite_rows] == [0, 1]
    assert [row["dataset"] for row in suite_rows] == [LAST_DATASET_NAME, "MMAD"]
    assert [row["sample_id"] for row in suite_rows] == [
        last_rows[0]["sample_id"],
        one_shot_rows[0]["sample_id"],
    ]
    validate_benchmark_identity(suite_output / "identity.json", verify_tasks=True)


def test_suite_identity_falls_back_to_component_manifest_hashes(
    tmp_path: Path, capsys: object
) -> None:
    image = (tmp_path / "image.png").resolve()
    _save_image(image, size=(4, 4), color=(1, 2, 3), image_format="PNG")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()

    def row(dataset: str, sample_id: str) -> dict[str, object]:
        return {
            "ordinal": 0,
            "dataset": dataset,
            "row_number": 0,
            "index": sample_id,
            "sample_id": sample_id,
            "question": "Choose.",
            "image_paths": [str(image)],
            "answer": "A",
            "options": [["A", "yes"], ["B", "no"]],
            "image_sha256s": [digest],
            "image_dimensions": [[4, 4]],
        }

    last_manifest = tmp_path / "last" / "tasks.jsonl"
    anomaly_manifest = tmp_path / "mmad" / "tasks.jsonl"
    last_manifest.parent.mkdir()
    anomaly_manifest.parent.mkdir()
    last_manifest.write_text(
        json.dumps(row(LAST_DATASET_NAME, "last")) + "\n", encoding="utf-8"
    )
    anomaly_manifest.write_text(
        json.dumps(row("MMAD", "mmad")) + "\n", encoding="utf-8"
    )

    assert (
        TOOL.main(
            [
                "suite",
                "--last-manifest",
                str(last_manifest),
                "--mmad-manifest",
                str(anomaly_manifest),
                "--output",
                str(tmp_path / "suite"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["benchmark_id"].startswith("last-")
    assert "__mmad-" in summary["benchmark_id"]
