from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.evaluation.final_answer_view import (
    INVALID_SENTINEL_PREFIX,
    materialize_coredev_reference_coverage_view,
    materialize_final_answer_view,
    materialize_mathverse_metadata_view,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/materialize_vlmevalkit_final_answers.py"


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def test_extracts_nonempty_suffix_after_last_think_closer(tmp_path: Path) -> None:
    source = tmp_path / "input.tsv"
    derived = tmp_path / "answers.tsv"
    fields = ["index", "question", "prediction", "metadata"]
    rows = [
        {
            "index": "alpha",
            "question": "Unicode 图像?",
            "prediction": "reasoning</think>\n\t  final answer  ",
            "metadata": '{"note":"kept exactly"}',
        },
        {
            "index": "beta",
            "question": "multiple closers",
            "prediction": "first</think>discarded</think>\nB",
            "metadata": "with\ttab\nand newline",
        },
    ]
    _write_tsv(source, fields, rows)

    manifest = materialize_final_answer_view(source_tsv=source, derived_tsv=derived)
    derived_fields, actual = _read_tsv(derived)

    assert derived_fields == fields
    assert [row["prediction"] for row in actual] == [
        "\n\t  final answer  ",
        "\nB",
    ]
    for before, after in zip(rows, actual, strict=True):
        assert {key: value for key, value in after.items() if key != "prediction"} == {
            key: value for key, value in before.items() if key != "prediction"
        }
    assert manifest["counts"] == {
        "row_count": 2,
        "closed_count": 2,
        "invalid_count": 0,
        "missing_think_closer_count": 0,
        "empty_final_answer_count": 0,
        "mcq_row_count": 0,
        "invalid_mcq_count": 0,
        "invalid_non_mcq_count": 0,
    }


def test_manifest_can_record_atomic_publish_paths(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    source = staging / "raw.tsv"
    derived = staging / "answers.tsv"
    published = tmp_path / "published"
    staging.mkdir()
    _write_tsv(
        source,
        ["index", "prediction"],
        [{"index": "one", "prediction": "reasoning</think>answer"}],
    )

    manifest = materialize_final_answer_view(
        source_tsv=source,
        derived_tsv=derived,
        recorded_source_tsv=published / "raw.tsv",
        recorded_derived_tsv=published / "answers.tsv",
    )

    assert derived.is_file()
    assert manifest["source"]["path"] == str((published / "raw.tsv").resolve())
    assert manifest["derived"]["path"] == str((published / "answers.tsv").resolve())


def test_missing_or_empty_final_answer_is_deterministically_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.tsv"
    derived = tmp_path / "answers.tsv"
    fields = ["index", "question", "answer", "A", "B", "prediction"]
    rows = [
        {
            "index": "mcq-invalid",
            "question": "choose",
            "answer": "A",
            "A": "right",
            "B": "wrong",
            "prediction": "reasoning never closed",
        },
        {
            "index": "mcq-valid",
            "question": "choose",
            "answer": "B",
            "A": "wrong",
            "B": "right",
            "prediction": "done</think>B",
        },
        {
            "index": "non-mcq-invalid",
            "question": "answer",
            "answer": "42",
            "A": "",
            "B": "",
            "prediction": "done</think> \n ",
        },
    ]
    _write_tsv(source, fields, rows)

    first = materialize_final_answer_view(source_tsv=source, derived_tsv=derived)
    derived_fields, actual = _read_tsv(derived)

    assert derived_fields == [*fields, "Z"]
    assert actual[0]["prediction"] == "Z"
    assert actual[0]["Z"].startswith(INVALID_SENTINEL_PREFIX)
    assert actual[0]["answer"] != actual[0]["prediction"]
    assert actual[1]["prediction"] == "B"
    assert actual[1]["Z"] == ""
    assert actual[2]["prediction"].startswith(INVALID_SENTINEL_PREFIX)
    assert actual[2]["prediction"] != actual[0]["Z"]
    assert actual[2]["Z"] == ""
    assert first["counts"]["invalid_count"] == 2
    assert first["counts"]["missing_think_closer_count"] == 1
    assert first["counts"]["empty_final_answer_count"] == 1
    assert first["invalid_policy"]["llm_or_random_fallback_allowed"] is False

    second_dir = tmp_path / "repeat"
    second_dir.mkdir()
    second = materialize_final_answer_view(
        source_tsv=source,
        derived_tsv=second_dir / "answers.tsv",
    )
    _, repeated = _read_tsv(second_dir / "answers.tsv")
    assert [row["prediction"] for row in repeated] == [
        row["prediction"] for row in actual
    ]
    assert repeated[0]["Z"] == actual[0]["Z"]
    assert second["derived"]["sha256"] == first["derived"]["sha256"]


def test_existing_option_z_column_forces_use_of_another_unused_label(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.tsv"
    derived = tmp_path / "answers.tsv"
    fields = ["index", "answer", "A", "B", "Z", "prediction"]
    rows = [
        {
            "index": "1",
            "answer": "A",
            "A": "correct",
            "B": "wrong",
            "Z": "legitimate existing option",
            "prediction": "unclosed",
        },
        {
            "index": "2",
            "answer": "B",
            "A": "wrong",
            "B": "correct",
            "Z": "another legitimate option",
            "prediction": "reason</think>B",
        },
    ]
    _write_tsv(source, fields, rows)

    materialize_final_answer_view(source_tsv=source, derived_tsv=derived)
    derived_fields, actual = _read_tsv(derived)

    assert derived_fields == [*fields, "Y"]
    assert actual[0]["prediction"] == "Y"
    assert actual[0]["Y"].startswith(INVALID_SENTINEL_PREFIX)
    assert actual[1]["Y"] == ""
    assert [row["Z"] for row in actual] == [row["Z"] for row in rows]


def test_mathverse_problem_version_is_exactly_joined_into_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "MathVerse_MINI.tsv"
    derived = tmp_path / "answers.tsv"
    mathverse = tmp_path / "testmini.json"
    fields = [
        "index",
        "source_row_index",
        "question",
        "metadata",
        "prediction",
    ]
    rows = [
        {
            "index": "mv/1",
            "source_row_index": "1",
            "question": "q1",
            "metadata": '{"subject":"geometry"}',
            "prediction": "work</think>17",
        },
        {
            "index": "mv/0",
            "source_row_index": "0",
            "question": "q0",
            "metadata": '{"subject":"algebra","problem_version":"Vision Dominant"}',
            "prediction": "work</think>x=2",
        },
    ]
    source_rows = [
        {"problem_version": "Vision Dominant", "question": "source q0"},
        {"problem_version": "Text Dominant", "question": "source q1"},
    ]
    _write_tsv(source, fields, rows)
    mathverse.write_text(json.dumps(source_rows), encoding="utf-8")

    manifest = materialize_final_answer_view(
        source_tsv=source,
        derived_tsv=derived,
        mathverse_source_json=mathverse,
    )
    _, actual = _read_tsv(derived)

    assert json.loads(actual[0]["metadata"]) == {
        "problem_version": "Text Dominant",
        "subject": "geometry",
    }
    assert json.loads(actual[1]["metadata"]) == {
        "problem_version": "Vision Dominant",
        "subject": "algebra",
    }
    for before, after in zip(rows, actual, strict=True):
        for field in fields:
            if field not in {"prediction", "metadata"}:
                assert after[field] == before[field]
    enrichment = manifest["mathverse_metadata_enrichment"]
    assert enrichment["joined_row_count"] == 2
    assert (
        enrichment["source_json_sha256"] == sha256(mathverse.read_bytes()).hexdigest()
    )


def test_mathverse_metadata_only_view_preserves_predictions(tmp_path: Path) -> None:
    source = tmp_path / "MathVerse_MINI.tsv"
    derived = tmp_path / "metadata-view.tsv"
    mathverse = tmp_path / "testmini.json"
    fields = ["index", "source_row_index", "metadata", "prediction"]
    rows = [
        {
            "index": "mv/0",
            "source_row_index": "0",
            "metadata": '{"subject":"geometry"}',
            "prediction": "raw answer without a think closer\n17",
        }
    ]
    _write_tsv(source, fields, rows)
    mathverse.write_text(
        json.dumps([{"problem_version": "Vision Only"}]), encoding="utf-8"
    )

    manifest = materialize_mathverse_metadata_view(
        source_tsv=source,
        derived_tsv=derived,
        mathverse_source_json=mathverse,
    )
    _, actual = _read_tsv(derived)

    assert actual[0]["prediction"] == rows[0]["prediction"]
    assert json.loads(actual[0]["metadata"])["problem_version"] == "Vision Only"
    assert manifest["prediction_values_identical"] is True


def test_reference_coverage_view_preserves_raw_direct_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "BLINK-result.tsv"
    derived = tmp_path / "BLINK-reference-view.tsv"
    task_manifest = tmp_path / "tasks.jsonl"
    rows = [
        {
            "index": f"blink-{index}",
            "prediction": "A",
            "hit": "1" if index in {0, 180} else "0",
        }
        for index in range(181)
    ]
    _write_tsv(source, ["index", "prediction", "hit"], rows)
    task_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": "BLINK",
                    "index": f"blink-{index}",
                    "image_paths": ["one.jpg"] if index < 180 else ["a.jpg", "b.jpg"],
                }
            )
            + "\n"
            for index in range(181)
        ),
        encoding="utf-8",
    )

    manifest = materialize_coredev_reference_coverage_view(
        source_tsv=source,
        derived_tsv=derived,
        task_manifest_path=task_manifest,
        dataset="BLINK",
    )
    _, actual = _read_tsv(derived)

    assert [row["prediction"] for row in actual] == [row["prediction"] for row in rows]
    assert [row["hit"] for row in actual] == [row["hit"] for row in rows]
    assert manifest["counts"] == {
        "single_image_evaluated": 180,
        "excluded_multi_image_reference": 1,
    }
    assert (
        json.loads(actual[-1]["extra_records"])["coverage"]
        == "excluded_multi_image_reference"
    )


def test_materializer_rejects_overwrite_duplicate_indices_and_bad_mathverse_join(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.tsv"
    derived = tmp_path / "answers.tsv"
    fields = ["index", "source_row_index", "metadata", "prediction"]
    rows = [
        {
            "index": "same",
            "source_row_index": "3",
            "metadata": "{}",
            "prediction": "work</think>answer",
        },
        {
            "index": "same",
            "source_row_index": "4",
            "metadata": "{}",
            "prediction": "work</think>answer",
        },
    ]
    _write_tsv(source, fields, rows)
    with pytest.raises(ValueError, match="duplicate indices"):
        materialize_final_answer_view(source_tsv=source, derived_tsv=derived)

    rows.pop()
    _write_tsv(tmp_path / "unique.tsv", fields, rows)
    mathverse = tmp_path / "mathverse.json"
    mathverse.write_text(json.dumps([{"problem_version": "v"}]), encoding="utf-8")
    with pytest.raises(IndexError, match="outside MathVerse JSON"):
        materialize_final_answer_view(
            source_tsv=tmp_path / "unique.tsv",
            derived_tsv=derived,
            mathverse_source_json=mathverse,
        )

    _write_tsv(
        tmp_path / "ok.tsv",
        ["index", "prediction"],
        [
            {
                "index": "1",
                "prediction": "work</think>ok",
            }
        ],
    )
    materialize_final_answer_view(source_tsv=tmp_path / "ok.tsv", derived_tsv=derived)
    with pytest.raises(FileExistsError, match="immutable"):
        materialize_final_answer_view(
            source_tsv=tmp_path / "ok.tsv", derived_tsv=derived
        )


def test_cli_writes_requested_manifest_and_reports_hashes(tmp_path: Path) -> None:
    source = tmp_path / "input.tsv"
    derived = tmp_path / "answers.tsv"
    manifest_path = tmp_path / "identity.json"
    _write_tsv(
        source,
        ["index", "prediction"],
        [{"index": "1", "prediction": "reason</think>yes"}],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(source),
            str(derived),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(completed.stdout)
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stdout == recorded
    assert recorded["source"]["sha256"] == sha256(source.read_bytes()).hexdigest()
    assert recorded["derived"]["sha256"] == sha256(derived.read_bytes()).hexdigest()
