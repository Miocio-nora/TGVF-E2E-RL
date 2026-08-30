from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import FunctionType
from typing import get_type_hints

import torch

from tgvf_rl.representation.experiments.answer_utility.evaluation import runner
from tgvf_rl.representation.experiments.answer_utility.evaluation import (
    input_artifact,
    input_identity,
    input_matching,
    inputs,
)


_PACKAGE = (
    Path(__file__).parents[4]
    / "src"
    / "tgvf_rl"
    / "representation"
    / "experiments"
    / "answer_utility"
    / "evaluation"
)
_SPLIT_MODULES = (
    "input_artifact",
    "input_audit",
    "input_identity",
    "input_matching",
    "inputs",
    "runner",
)
_LEGACY_ALL = [
    "ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION",
    "DEFAULT_INSTRUCT_EOS_TOKEN_IDS",
    "DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS",
    "AnswerUtilityAdapterArtifact",
    "AnswerUtilityEvaluationArm",
    "AnswerUtilityEvaluationCandidate",
    "AnswerUtilityWrongImageDonor",
    "build_answer_safe_wrong_mapping",
    "build_same_target_wrong_image_mapping",
    "load_answer_utility_adapter_artifact",
    "run_answer_utility_evaluation",
    "run_production_source_answer_utility_evaluation",
    "validate_answer_utility_evaluation",
    "validate_production_source_answer_utility_evaluation",
]


def _relative_split_imports(module: str) -> set[str]:
    source = _PACKAGE / f"{module}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and (node.module or "") in _SPLIT_MODULES
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


def test_answer_utility_input_split_is_a_one_way_import_dag() -> None:
    graph = {module: _relative_split_imports(module) for module in _SPLIT_MODULES}

    assert graph == {
        "input_artifact": {"input_matching"},
        "input_audit": {"input_matching"},
        "input_identity": {"input_matching", "inputs"},
        "input_matching": set(),
        "inputs": {"input_artifact", "input_audit", "input_matching"},
        "runner": {"input_identity", "input_matching", "inputs"},
    }
    for left in _SPLIT_MODULES:
        for right in _SPLIT_MODULES:
            if left == right:
                continue
            assert not (
                right in _reachable(graph, left) and left in _reachable(graph, right)
            ), f"non-trivial import SCC contains {left!r} and {right!r}"


def test_answer_utility_runner_keeps_frozen_public_namespace() -> None:
    assert runner.__all__ == _LEGACY_ALL


def test_moved_contracts_and_functions_keep_exact_facade_identity() -> None:
    contract_pairs = (
        (
            runner.AnswerUtilityAdapterArtifact,
            input_artifact.AnswerUtilityAdapterArtifact,
        ),
        (
            runner.AnswerUtilityEvaluationCandidate,
            inputs.AnswerUtilityEvaluationCandidate,
        ),
        (runner.AnswerUtilityEvaluationInputs, inputs.AnswerUtilityEvaluationInputs),
        (runner.AnswerUtilityEvaluationArm, inputs.AnswerUtilityEvaluationArm),
        (
            runner.AnswerUtilityWrongImageDonor,
            input_matching.AnswerUtilityWrongImageDonor,
        ),
        (runner._QwenImageGridContract, input_matching._QwenImageGridContract),
    )
    for facade_type, leaf_type in contract_pairs:
        assert facade_type is leaf_type
        assert facade_type.__module__ == runner.__name__
        assert facade_type.__qualname__ == facade_type.__name__
        assert pickle.loads(pickle.dumps(facade_type)) is facade_type
        assert get_type_hints(facade_type) == get_type_hints(leaf_type)
        for member in vars(facade_type).values():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (classmethod, staticmethod)):
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if isinstance(function, FunctionType):
                    if function.__module__ in {dataclasses.__name__, "enum"}:
                        continue
                    assert function.__module__ == runner.__name__

    function_pairs = (
        (
            runner.load_answer_utility_adapter_artifact,
            input_artifact.load_answer_utility_adapter_artifact,
        ),
        (
            runner.build_answer_safe_wrong_mapping,
            input_matching.build_answer_safe_wrong_mapping,
        ),
        (
            runner.build_same_target_wrong_image_mapping,
            input_matching.build_same_target_wrong_image_mapping,
        ),
        (runner._qwen_image_grid_thw, input_matching._qwen_image_grid_thw),
        (
            runner._same_target_wrong_image_model_inputs,
            input_matching._same_target_wrong_image_model_inputs,
        ),
        (
            runner._evaluation_arm_contract,
            input_identity._evaluation_arm_contract,
        ),
    )
    for facade_function, leaf_function in function_pairs:
        assert facade_function is leaf_function
        assert facade_function.__module__ == runner.__name__
        assert facade_function.__qualname__ == facade_function.__name__
        assert pickle.loads(pickle.dumps(facade_function)) is facade_function
        assert get_type_hints(facade_function) == get_type_hints(leaf_function)


def test_moved_contract_instances_pickle_through_runner_facade(
    tmp_path: Path,
) -> None:
    arm = runner.AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT
    assert pickle.loads(pickle.dumps(arm)) is arm

    adapter_path = (tmp_path / "source.pt").resolve()
    candidate = runner.AnswerUtilityEvaluationCandidate(
        kind="production_source",
        candidate_id="source",
        adapter_path=adapter_path,
        adapter_file_sha256="a" * 64,
        adapter_state_sha256="b" * 64,
        adapter_state={"weight": torch.arange(3)},
        global_step=32,
        training_run_identity_sha256="c" * 64,
        production_source_artifact_path=adapter_path,
        production_source_artifact_sha256="a" * 64,
        production_source_manifest_sha256="d" * 64,
        production_source_run_identity_sha256="c" * 64,
        production_source_global_step=32,
        protected_paths=(tmp_path.resolve(),),
    )
    restored_candidate = pickle.loads(pickle.dumps(candidate))
    assert type(restored_candidate) is runner.AnswerUtilityEvaluationCandidate
    assert restored_candidate.candidate_id == candidate.candidate_id
    assert torch.equal(
        restored_candidate.adapter_state["weight"],
        candidate.adapter_state["weight"],
    )

    donor = runner.AnswerUtilityWrongImageDonor(
        anchor_image_group_key="anchor",
        anchor_image_sha256="1" * 64,
        donor_sample_id="donor-sample",
        donor_sample_content_sha256="2" * 64,
        donor_image_group_key="donor",
        donor_image="/fixture/donor.png",
        donor_image_sha256="3" * 64,
        image_grid_thw=(1, 16, 16),
        match_tier="exact_grid_cross_domain",
    )
    restored_donor = pickle.loads(pickle.dumps(donor))
    assert restored_donor == donor
    assert type(restored_donor) is runner.AnswerUtilityWrongImageDonor

    materialized = runner.AnswerUtilityEvaluationInputs(
        training="training",  # type: ignore[arg-type]
        source_evaluation="source",
        candidate=candidate,
        selected_groups=(),
        wrong_source_by_sample_id={},
        same_target_wrong_image_by_group_key={},
        wrong_image_pool_manifest_sha256=None,
        data_manifest_sha256="e" * 64,
        ordered_group_manifest_identity="identity",
        arms=(arm,),
        max_new_tokens=64,
        eos_token_ids=(151645, 151643),
        decode_mode="cached",
        arm_batch_size=1,
        group_start=0,
        group_limit=None,
        shard_index=0,
        shard_count=1,
    )
    restored_inputs = pickle.loads(pickle.dumps(materialized))
    assert type(restored_inputs) is runner.AnswerUtilityEvaluationInputs
    assert restored_inputs.arms == (arm,)
    assert torch.equal(
        restored_inputs.candidate.adapter_state["weight"],
        candidate.adapter_state["weight"],
    )


def test_answer_utility_split_modules_keep_size_headroom() -> None:
    ceilings = {
        "runner": 850,
        "inputs": 700,
        "input_artifact": 850,
        "input_matching": 850,
        "input_identity": 850,
        "input_audit": 850,
    }
    for module, ceiling in ceilings.items():
        line_count = len((_PACKAGE / f"{module}.py").read_text().splitlines())
        assert line_count <= ceiling, f"{module}.py grew to {line_count} lines"


def test_input_leaves_pickle_before_explicit_facade_import() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    script = textwrap.dedent(
        """
        import importlib
        import pickle
        import sys
        import types

        import tgvf_rl.representation.experiments.answer_utility

        package_name = (
            "tgvf_rl.representation.experiments.answer_utility.evaluation"
        )
        facade_name = f"{package_name}.runner"
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [sys.argv[1]]
        sys.modules[package_name] = package

        matching = importlib.import_module(f"{package_name}.input_matching")
        artifact = importlib.import_module(f"{package_name}.input_artifact")
        inputs = importlib.import_module(f"{package_name}.inputs")
        identity = importlib.import_module(f"{package_name}.input_identity")

        assert facade_name not in sys.modules
        assert inputs.AnswerUtilityEvaluationCandidate.__module__ == facade_name
        assert matching.build_answer_safe_wrong_mapping.__module__ == facade_name
        assert identity._evaluation_arm_contract.__module__ == facade_name

        encoded_candidate_type = pickle.dumps(
            inputs.AnswerUtilityEvaluationCandidate
        )
        encoded_artifact_loader = pickle.dumps(
            artifact.load_answer_utility_adapter_artifact
        )
        encoded_mapping_function = pickle.dumps(
            matching.build_answer_safe_wrong_mapping
        )

        facade = importlib.import_module(facade_name)
        assert (
            facade.AnswerUtilityEvaluationCandidate
            is inputs.AnswerUtilityEvaluationCandidate
        )
        assert (
            facade.load_answer_utility_adapter_artifact
            is artifact.load_answer_utility_adapter_artifact
        )
        assert (
            facade.build_answer_safe_wrong_mapping
            is matching.build_answer_safe_wrong_mapping
        )
        assert (
            pickle.loads(encoded_candidate_type)
            is inputs.AnswerUtilityEvaluationCandidate
        )
        assert (
            pickle.loads(encoded_artifact_loader)
            is artifact.load_answer_utility_adapter_artifact
        )
        assert (
            pickle.loads(encoded_mapping_function)
            is matching.build_answer_safe_wrong_mapping
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(repository_root / "src"),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    environment.pop("OPENROUTER_API_KEY", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""

    subprocess.run(
        [sys.executable, "-c", script, str(_PACKAGE)],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )
