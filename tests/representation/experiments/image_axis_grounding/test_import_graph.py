from __future__ import annotations

import ast
from pathlib import Path
import pickle
from types import FunctionType
from typing import get_type_hints

import tgvf_rl.representation.experiments.image_axis_grounding as image_axis_grounding
from tgvf_rl.representation.experiments.image_axis_grounding import (
    config,
    objective,
    streaming,
    trainer,
)


_PACKAGE = (
    Path(__file__).parents[4]
    / "src"
    / "tgvf_rl"
    / "representation"
    / "experiments"
    / "image_axis_grounding"
)
_MODULES = (
    "config",
    "handoff",
    "matching",
    "native_pipeline",
    "objective",
    "runner",
    "streaming",
    "trainer",
)


def _relative_imports(module: str) -> set[str]:
    path = _PACKAGE / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # ast.walk deliberately includes imports under TYPE_CHECKING and inside
    # functions: either form recreates the architectural dependency edge.
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and (node.module or "") in _MODULES
    }


def _reachable(graph: dict[str, set[str]], source: str) -> set[str]:
    pending = list(graph[source])
    reached: set[str] = set()
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        reached.add(module)
        pending.extend(graph[module] - reached)
    return reached


def test_image_axis_import_graph_has_no_nontrivial_scc() -> None:
    graph = {module: _relative_imports(module) for module in _MODULES}

    assert graph == {
        "config": {"objective"},
        "handoff": set(),
        "matching": set(),
        "native_pipeline": {"matching"},
        "objective": set(),
        "runner": {"config", "matching", "native_pipeline", "trainer"},
        "streaming": {"native_pipeline", "objective"},
        "trainer": {"native_pipeline", "objective", "streaming"},
    }
    for left in _MODULES:
        for right in _MODULES:
            if left == right:
                continue
            assert not (
                right in _reachable(graph, left) and left in _reachable(graph, right)
            ), f"non-trivial import SCC contains {left!r} and {right!r}"


def test_image_axis_objective_preserves_legacy_facades_and_pickle_paths() -> None:
    contract = objective.ImageAxisGroundingObjectiveConfig

    assert trainer.ImageAxisGroundingObjectiveConfig is contract
    assert config.ImageAxisGroundingObjectiveConfig is contract
    assert image_axis_grounding.ImageAxisGroundingObjectiveConfig is contract
    assert contract.__module__ == trainer.__name__
    assert pickle.loads(pickle.dumps(contract)) is contract
    value = contract()
    assert pickle.loads(pickle.dumps(value)) == value
    assert (
        get_type_hints(streaming.backward_image_axis_grounding_group)[
            "image_axis_objective"
        ]
        is contract
    )
    for member in contract.__dict__.values():
        if isinstance(member, FunctionType) and member.__module__ != "dataclasses":
            assert member.__module__ == trainer.__name__
        elif isinstance(member, property):
            for accessor in (member.fget, member.fset, member.fdel):
                if isinstance(accessor, FunctionType):
                    assert accessor.__module__ == trainer.__name__
