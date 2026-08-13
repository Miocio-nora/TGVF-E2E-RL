from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tgvf_rl.evaluation.texture_bench import task as task_module


def test_task_loader_verifies_a_shared_image_once(tmp_path: Path, monkeypatch) -> None:
    image_path = (tmp_path / "shared.png").resolve()
    Image.new("RGB", (8, 6), "navy").save(image_path)
    digest, dimensions = task_module.image_file_identity(image_path)
    rows = []
    for ordinal in range(2):
        sample_id = f"sample-{ordinal}"
        rows.append(
            {
                "ordinal": ordinal,
                "dataset": "MMAD",
                "row_number": ordinal,
                "index": sample_id,
                "question": "Choose one.",
                "image_paths": [str(image_path)],
                "sample_id": sample_id,
                "answer": "A",
                "options": [["A", "yes"], ["B", "no"]],
                "image_sha256s": [digest],
                "image_dimensions": [list(dimensions)],
            }
        )
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    original = task_module.image_file_identity
    calls = 0

    def counted(path: str | Path) -> tuple[str, tuple[int, int]]:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(task_module, "image_file_identity", counted)
    tasks = task_module.load_texture_tasks(manifest, verify_images=True)

    assert len(tasks) == 2
    assert calls == 1
