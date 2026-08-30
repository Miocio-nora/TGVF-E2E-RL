from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import FunctionType
from typing import get_type_hints

from tgvf_rl.evaluation import (
    policy_coredev,
    policy_evaluation_identity,
    policy_full_model_snapshot,
    policy_lora_snapshot,
    policy_vllm_manager,
)


_EVALUATION_PACKAGE = Path(__file__).parents[2] / "src" / "tgvf_rl" / "evaluation"
_MODULES = (
    "policy_coredev",
    "policy_evaluation_identity",
    "policy_full_model_snapshot",
    "policy_lora_snapshot",
    "policy_vllm_manager",
)
_SHARED_DATACLASS_PICKLE_HELPERS = (
    dataclasses._dataclass_getstate,
    dataclasses._dataclass_setstate,
)


def _assert_legacy_class_member_modules(value: type[object]) -> None:
    for member in value.__dict__.values():
        if isinstance(member, FunctionType):
            if member in _SHARED_DATACLASS_PICKLE_HELPERS:
                assert member.__module__ == "dataclasses"
            else:
                assert member.__module__ == policy_coredev.__name__
        elif isinstance(member, property):
            for accessor in (member.fget, member.fset, member.fdel):
                if isinstance(accessor, FunctionType):
                    assert accessor.__module__ == policy_coredev.__name__


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
            "policy_lora_snapshot",
            "policy_vllm_manager",
        },
        "policy_evaluation_identity": set(),
        "policy_full_model_snapshot": {
            "policy_evaluation_identity",
            "policy_vllm_manager",
        },
        "policy_lora_snapshot": {
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
        "PolicyEvaluationSnapshot",
        "VLLMLoRAAdapterIntegrityVerifier",
        "_vllm_lora_adapter_payloads",
        "_open_absolute_directory_nofollow",
        "_open_vllm_lora_adapter_root",
        "_open_or_create_private_directory_at",
        "_read_private_vllm_lora_file_at",
        "_write_private_vllm_lora_file_at",
        "_assert_private_vllm_lora_file_equals_at",
        "_publish_private_vllm_lora_file_at",
        "build_vllm_lora_adapter_integrity_verifier",
        "materialize_vllm_lora_adapter",
        "policy_lora_request_name",
        "_base_equivalent_step_zero_lora",
        "_standalone_engine_kwargs",
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
    _assert_legacy_class_member_modules(policy_coredev._TurnRoute)
    _assert_legacy_class_member_modules(manager)


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
    _assert_legacy_class_member_modules(contract)


def test_policy_coredev_preserves_lora_snapshot_imports_and_pickle_path() -> None:
    for name in (
        "VLLM_LORA_ADAPTER_CONFIG_FILENAME",
        "VLLM_LORA_ADAPTER_IDENTITY_FILENAME",
        "VLLM_LORA_ADAPTER_MODEL_FILENAME",
        "VLLM_LORA_ADAPTER_SCHEMA",
        "VLLM_LORA_ENGINE_ATTESTATION",
        "VLLM_LORA_RESIDUAL_RACE",
    ):
        assert getattr(policy_coredev, name) is getattr(policy_lora_snapshot, name)

    names = (
        "PolicyEvaluationSnapshot",
        "VLLMLoRAAdapterIntegrityVerifier",
        "_vllm_lora_adapter_payloads",
        "_open_absolute_directory_nofollow",
        "_open_vllm_lora_adapter_root",
        "_open_or_create_private_directory_at",
        "_read_private_vllm_lora_file_at",
        "_write_private_vllm_lora_file_at",
        "_assert_private_vllm_lora_file_equals_at",
        "_publish_private_vllm_lora_file_at",
        "build_vllm_lora_adapter_integrity_verifier",
        "materialize_vllm_lora_adapter",
        "policy_lora_request_name",
        "_base_equivalent_step_zero_lora",
        "_standalone_engine_kwargs",
    )
    for name in names:
        facade_value = getattr(policy_coredev, name)
        leaf_value = getattr(policy_lora_snapshot, name)
        assert facade_value is leaf_value
        assert facade_value.__module__ == policy_coredev.__name__
        assert facade_value.__name__ == name
        assert facade_value.__qualname__ == name
        assert pickle.loads(pickle.dumps(facade_value)) is facade_value

    for contract in (
        policy_lora_snapshot.PolicyEvaluationSnapshot,
        policy_lora_snapshot.VLLMLoRAAdapterIntegrityVerifier,
    ):
        _assert_legacy_class_member_modules(contract)

    snapshot = policy_lora_snapshot.PolicyEvaluationSnapshot
    assert get_type_hints(snapshot)["run"].__name__ == "PolicyE2ESmokeRunConfig"
    assert get_type_hints(snapshot)["lora"].__name__ == "PolicyLoRASnapshot"
    assert get_type_hints(
        policy_lora_snapshot.build_vllm_lora_adapter_integrity_verifier
    )["snapshot"] is snapshot
    assert get_type_hints(policy_lora_snapshot.materialize_vllm_lora_adapter)[
        "snapshot"
    ] is snapshot


def test_extracted_dataclasses_do_not_mutate_stdlib_pickle_helpers() -> None:
    for helper in _SHARED_DATACLASS_PICKLE_HELPERS:
        assert helper.__module__ == "dataclasses"


def test_lora_snapshot_leaf_imports_without_loading_either_backend_facade() -> None:
    script = """
import sys
import tgvf_rl.evaluation.policy_lora_snapshot
assert 'tgvf_rl.evaluation.policy_coredev' not in sys.modules
assert 'tgvf_rl.evaluation.policy_full_model_snapshot' not in sys.modules
"""
    repository_root = _EVALUATION_PACKAGE.parents[2]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            (str(repository_root / "src"), str(repository_root))
        ),
    }
    environment.pop("OPENROUTER_API_KEY", None)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_shared_identity_leaf_does_not_depend_on_either_snapshot_backend() -> None:
    identity_imports = _relative_imports("policy_evaluation_identity")
    lora_imports = _relative_imports("policy_lora_snapshot")
    manager_imports = _relative_imports("policy_vllm_manager")

    assert "policy_coredev" not in identity_imports | manager_imports
    assert "policy_full_model_snapshot" not in identity_imports | manager_imports
    assert "policy_coredev" not in lora_imports
    assert "policy_full_model_snapshot" not in lora_imports
