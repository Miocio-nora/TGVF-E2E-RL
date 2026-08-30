from __future__ import annotations

import ast
from pathlib import Path
import pickle
from types import FunctionType

from tgvf_rl.evaluation import (
    policy_coredev,
    policy_evaluation_identity,
    policy_full_model_snapshot,
    policy_vllm_manager,
)


_EVALUATION_PACKAGE = Path(__file__).parents[2] / "src" / "tgvf_rl" / "evaluation"
_MODULES = (
    "policy_coredev",
    "policy_evaluation_identity",
    "policy_full_model_snapshot",
    "policy_vllm_manager",
)


def _relative_imports(module: str) -> set[str]:
    path = _EVALUATION_PACKAGE / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Include imports nested in functions so a late/local import cannot hide a
    # dependency edge from this architectural regression test.
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


def test_policy_snapshot_import_graph_has_no_nontrivial_scc() -> None:
    graph = {module: _relative_imports(module) for module in _MODULES}

    assert graph == {
        "policy_coredev": {
            "policy_evaluation_identity",
            "policy_full_model_snapshot",
            "policy_vllm_manager",
        },
        "policy_evaluation_identity": set(),
        "policy_full_model_snapshot": {
            "policy_evaluation_identity",
            "policy_vllm_manager",
        },
        "policy_vllm_manager": set(),
    }
    for left in _MODULES:
        for right in _MODULES:
            if left == right:
                continue
            assert not (
                right in _reachable(graph, left) and left in _reachable(graph, right)
            ), f"non-trivial import SCC contains {left!r} and {right!r}"


def test_policy_coredev_facade_does_not_redefine_extracted_contracts() -> None:
    path = _EVALUATION_PACKAGE / "policy_coredev.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {
        "PolicyEvalContract",
        "StandaloneTGVFVLLMManager",
        "_TurnRoute",
        "_single_collective",
        "_canonical_json_sha256",
        "_policy_eval_parser_identity",
        "_policy_eval_action_boundary_identity",
        "_policy_eval_observation_identity",
        "_base_equivalent_step_zero_full_model",
        "_decoding_contract",
        "_termination_contract",
        "effective_evaluation_image_max_pixels",
        "build_policy_eval_contract",
        "policy_benchmark_task_path",
    }.isdisjoint(definitions)


def test_policy_coredev_preserves_standalone_manager_import_and_pickle_path() -> None:
    manager = policy_vllm_manager.StandaloneTGVFVLLMManager

    assert policy_coredev.StandaloneTGVFVLLMManager is manager
    assert policy_full_model_snapshot.StandaloneTGVFVLLMManager is manager
    assert manager.__module__ == policy_coredev.__name__
    assert pickle.loads(pickle.dumps(manager)) is manager
    assert policy_coredev._single_collective is policy_vllm_manager._single_collective
    assert policy_coredev._TurnRoute is policy_vllm_manager._TurnRoute
    for value in (
        policy_coredev._single_collective,
        policy_coredev._TurnRoute,
        policy_coredev.StandaloneTGVFVLLMManager,
    ):
        assert value.__module__ == policy_coredev.__name__
        assert pickle.loads(pickle.dumps(value)) is value
    for member in manager.__dict__.values():
        if isinstance(member, FunctionType):
            assert member.__module__ == policy_coredev.__name__


def test_policy_coredev_preserves_identity_contract_imports_and_pickle_path() -> None:
    contract = policy_evaluation_identity.PolicyEvalContract

    assert policy_coredev.PolicyEvalContract is contract
    assert policy_coredev.build_policy_eval_contract is (
        policy_evaluation_identity.build_policy_eval_contract
    )
    assert policy_coredev.effective_evaluation_image_max_pixels is (
        policy_evaluation_identity.effective_evaluation_image_max_pixels
    )
    assert policy_coredev.policy_benchmark_task_path is (
        policy_evaluation_identity.policy_benchmark_task_path
    )
    assert policy_coredev._canonical_json_sha256 is (
        policy_evaluation_identity.canonical_json_sha256
    )
    assert policy_coredev._policy_eval_parser_identity is (
        policy_evaluation_identity.policy_eval_parser_identity
    )
    assert policy_coredev._policy_eval_action_boundary_identity is (
        policy_evaluation_identity.policy_eval_action_boundary_identity
    )
    assert policy_coredev._policy_eval_observation_identity is (
        policy_evaluation_identity.policy_eval_observation_identity
    )
    assert policy_coredev._decoding_contract is (
        policy_evaluation_identity._decoding_contract
    )
    assert policy_coredev._termination_contract is (
        policy_evaluation_identity._termination_contract
    )
    assert policy_coredev._base_equivalent_step_zero_full_model is (
        policy_full_model_snapshot._base_equivalent_step_zero_full_model
    )
    assert contract.__module__ == policy_coredev.__name__
    assert pickle.loads(pickle.dumps(contract)) is contract
    for value in (
        policy_coredev.build_policy_eval_contract,
        policy_coredev.effective_evaluation_image_max_pixels,
        policy_coredev.policy_benchmark_task_path,
        policy_coredev._canonical_json_sha256,
        policy_coredev._policy_eval_parser_identity,
        policy_coredev._policy_eval_action_boundary_identity,
        policy_coredev._policy_eval_observation_identity,
        policy_coredev._decoding_contract,
        policy_coredev._termination_contract,
        policy_coredev._base_equivalent_step_zero_full_model,
    ):
        assert value.__module__ == policy_coredev.__name__
        assert pickle.loads(pickle.dumps(value)) is value
    for member in contract.__dict__.values():
        if isinstance(member, FunctionType):
            assert member.__module__ == policy_coredev.__name__
        elif isinstance(member, property) and member.fget is not None:
            assert member.fget.__module__ == policy_coredev.__name__


def test_shared_identity_leaf_does_not_depend_on_either_snapshot_backend() -> None:
    identity_imports = _relative_imports("policy_evaluation_identity")
    manager_imports = _relative_imports("policy_vllm_manager")

    assert "policy_coredev" not in identity_imports | manager_imports
    assert "policy_full_model_snapshot" not in identity_imports | manager_imports
