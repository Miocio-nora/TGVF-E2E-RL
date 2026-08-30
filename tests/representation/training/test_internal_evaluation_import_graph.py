from __future__ import annotations

import ast
from pathlib import Path


_TRAINING_PACKAGE = (
    Path(__file__).parents[3] / "src" / "tgvf_rl" / "representation" / "training"
)


def _relative_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_internal_evaluation_artifact_boundary_has_no_import_back_edge() -> None:
    evaluator_imports = _relative_imports(_TRAINING_PACKAGE / "internal_evaluation.py")
    artifact_imports = _relative_imports(
        _TRAINING_PACKAGE / "internal_evaluation_artifact.py"
    )

    assert "internal_evaluation_artifact" in evaluator_imports
    assert "internal_evaluation" not in artifact_imports
