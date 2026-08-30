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


def _top_level_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }


def test_internal_evaluation_artifact_boundary_has_no_import_back_edge() -> None:
    evaluator_imports = _relative_imports(_TRAINING_PACKAGE / "internal_evaluation.py")
    artifact_imports = _relative_imports(
        _TRAINING_PACKAGE / "internal_evaluation_artifact.py"
    )

    assert "internal_evaluation_artifact" in evaluator_imports
    assert "internal_evaluation" not in artifact_imports


def test_internal_evaluation_contract_is_a_one_way_dependency() -> None:
    evaluator = _TRAINING_PACKAGE / "internal_evaluation.py"
    contract = _TRAINING_PACKAGE / "internal_evaluation_contract.py"
    evaluator_imports = _relative_imports(evaluator)
    contract_imports = _relative_imports(contract)

    assert "internal_evaluation_contract" in evaluator_imports
    assert not contract_imports & {
        "internal_evaluation",
        "internal_evaluation_artifact",
        "native_pipeline",
        "qwen3_counterfactual",
        "qwen3_grounding",
        "streaming",
        "trainer",
    }
    assert _top_level_classes(evaluator) == {
        "InjectedNativeCounterfactualEvaluator"
    }
    assert len(_top_level_classes(contract)) == 33


def test_native_materializers_import_contracts_without_the_evaluator_facade() -> None:
    for filename in ("qwen3_counterfactual.py", "qwen3_grounding.py"):
        imports = _relative_imports(_TRAINING_PACKAGE / filename)
        assert "internal_evaluation_contract" in imports
        assert "internal_evaluation" not in imports
