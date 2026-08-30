from __future__ import annotations

import ast
from pathlib import Path


_SOURCE_ROOT = Path(__file__).parents[1] / "src"
_PACKAGE_ROOT = _SOURCE_ROOT / "tgvf_rl"


def _module_paths() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in _PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(_SOURCE_ROOT)
        if path.name == "__init__.py":
            module = ".".join(relative.parent.parts)
        else:
            module = ".".join(relative.with_suffix("").parts)
        modules[module] = path
    return modules


def _resolved_import(
    *, current: str, path: Path, module: str, level: int
) -> str:
    if level == 0:
        return module
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - level + 1])
    return ".".join(component for component in (base, module) if component)


def _import_graph(modules: dict[str, Path]) -> dict[str, set[str]]:
    graph = {module: set() for module in modules}
    for current, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [
                    _resolved_import(
                        current=current,
                        path=path,
                        module=node.module or "",
                        level=node.level,
                    )
                ]
            else:
                continue
            for candidate in imported:
                while candidate and candidate not in modules:
                    candidate = candidate.rpartition(".")[0]
                if candidate in modules and candidate != current:
                    graph[current].add(candidate)
    return graph


def _nontrivial_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []
    next_index = 0

    def visit(module: str) -> None:
        nonlocal next_index
        index[module] = next_index
        lowlink[module] = next_index
        next_index += 1
        stack.append(module)
        on_stack.add(module)
        for dependency in graph[module]:
            if dependency not in index:
                visit(dependency)
                lowlink[module] = min(lowlink[module], lowlink[dependency])
            elif dependency in on_stack:
                lowlink[module] = min(lowlink[module], index[dependency])
        if lowlink[module] != index[module]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in graph:
        if module not in index:
            visit(module)
    return sorted(components)


def test_source_import_graph_has_no_nontrivial_strongly_connected_component() -> None:
    modules = _module_paths()
    components = _nontrivial_components(_import_graph(modules))

    assert not components, "non-trivial import SCCs remain: " + repr(components)
