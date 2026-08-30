from __future__ import annotations

import ast
from pathlib import Path
import pickle
from types import FunctionType

import tgvf_rl.framework.verl as verl
from tgvf_rl.framework.verl import (
    policy_live_runtime,
    policy_runtime,
    policy_runtime_contract,
)


_VERL_PACKAGE = Path(__file__).parents[2] / "src" / "tgvf_rl" / "framework" / "verl"
_RUNTIME_MODULES = (
    "policy_runtime",
    "policy_live_runtime",
    "policy_runtime_contract",
)
_LEGACY_CONTRACT_EXPORTS = (
    "PolicyAgentLoopWorkerPlacement",
    "PolicyE2ERuntimeBuildContext",
    "PolicyE2ERuntimeProduct",
    "PolicyLoRASnapshotConsumer",
)


def _relative_imports(module: str) -> set[str]:
    path = _VERL_PACKAGE / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and (node.module or "") in _RUNTIME_MODULES
    }


def _reachable(graph: dict[str, set[str]], source: str) -> set[str]:
    pending = list(graph[source])
    reached: set[str] = set()
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(graph[node] - reached)
    return reached


def test_policy_runtime_composition_import_graph_has_no_nontrivial_scc() -> None:
    graph = {module: _relative_imports(module) for module in _RUNTIME_MODULES}

    assert graph == {
        "policy_runtime": {"policy_live_runtime", "policy_runtime_contract"},
        "policy_live_runtime": {"policy_runtime_contract"},
        "policy_runtime_contract": set(),
    }
    for left in _RUNTIME_MODULES:
        for right in _RUNTIME_MODULES:
            if left == right:
                continue
            assert not (
                right in _reachable(graph, left) and left in _reachable(graph, right)
            ), f"non-trivial import SCC contains {left!r} and {right!r}"


def test_policy_runtime_preserves_contract_identity_and_pickle_paths() -> None:
    for name in _LEGACY_CONTRACT_EXPORTS:
        public_value = getattr(policy_runtime, name)
        contract_value = getattr(policy_runtime_contract, name)

        assert public_value is contract_value
        assert getattr(verl, name) is public_value
        assert public_value.__module__ == policy_runtime.__name__
        assert pickle.loads(pickle.dumps(public_value)) is public_value
        for member in public_value.__dict__.values():
            if isinstance(member, FunctionType):
                assert member.__module__ != policy_runtime_contract.__name__
            elif isinstance(member, property):
                for accessor in (member.fget, member.fset, member.fdel):
                    if isinstance(accessor, FunctionType):
                        assert accessor.__module__ != policy_runtime_contract.__name__
    for name in ("PolicyE2ERuntimeBuildContext", "PolicyE2ERuntimeProduct"):
        assert getattr(policy_live_runtime, name) is getattr(policy_runtime, name)

    placement = policy_runtime.PolicyAgentLoopWorkerPlacement(0, 0, 0, 4)
    assert pickle.loads(pickle.dumps(placement)) == placement
