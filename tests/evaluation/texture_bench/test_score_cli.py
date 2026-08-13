from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPOSITORY_ROOT / "tools/score_texture_benchmark.py"


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location(
        "score_texture_benchmark", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_paths_binds_inputs_and_scores_last(tmp_path: Path) -> None:
    image_path = (tmp_path / "quiz.png").resolve()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(image_path)
    image_payload = image_path.read_bytes()
    tasks_path = (tmp_path / "tasks.jsonl").resolve()
    row = {
        "ordinal": 0,
        "dataset": "LAST_2D_Texture_Retrieval",
        "row_number": 0,
        "index": "last:test:0",
        "sample_id": "last:test:0",
        "question": "Which panel matches A?",
        "image_paths": [str(image_path)],
        "answer": "B",
        "options": [["B", "Panel B"], ["C", "Panel C"], ["D", "Panel D"]],
        "metadata": [
            ["condition_id", "different_shape_black_background"],
            ["source_dir", "different_shape_black_background"],
        ],
        "image_sha256s": [hashlib.sha256(image_payload).hexdigest()],
        "image_dimensions": [[8, 8]],
    }
    tasks_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    results_path = (tmp_path / "results.jsonl").resolve()
    task_manifest_sha256 = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    results_path.write_text(
        json.dumps(
            {
                "ordinal": 0,
                "sample_id": "last:test:0",
                "task_manifest_sha256": task_manifest_sha256,
                "final_answer": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    module = _load_script()
    report = module.score_paths(
        tasks_path=tasks_path,
        result_paths=[results_path],
        verify_images=True,
    )

    assert report["task_manifest"]["task_count"] == 1
    assert report["score"]["micro"]["accuracy"] == 1.0
    assert report["score"]["last"]["four_condition_macro_accuracy"] == 1.0
