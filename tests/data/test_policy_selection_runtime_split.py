from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import FunctionType
from typing import get_type_hints

import pytest

from tgvf_rl.data import (
    policy_selection_config,
    policy_selection_config_schema,
    policy_selection_config_values,
    policy_selection_evidence,
    policy_selection_runtime,
    policy_selection_verification,
)
from tgvf_rl.data.policy_selection import SelectionSource


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
_DATA_PACKAGE = Path(policy_selection_runtime.__file__).parent
_CONFIG_SCHEMA_OBJECTS = (
    policy_selection_config_schema.T1ResponseBudget,
    policy_selection_config_schema.T1DataSource,
    policy_selection_config_schema.T1RunConfig,
)
_CONFIG_VALUE_OBJECTS = (
    policy_selection_config_values.derive_t1_attempt_seed,
    policy_selection_config_values.candidate_rank,
)
_CONFIG_LOADER_OBJECTS = (policy_selection_config.load_t1_run_config,)
_CONFIG_OBJECTS = (
    _CONFIG_SCHEMA_OBJECTS + _CONFIG_VALUE_OBJECTS + _CONFIG_LOADER_OBJECTS
)
_EVIDENCE_OBJECTS = (
    policy_selection_evidence.GenerationDisposition,
    policy_selection_evidence.T1RawGenerationEvidence,
    policy_selection_evidence.classify_generation_finish,
    policy_selection_evidence.sampled_token_ids_sha256,
)
_VERIFICATION_OBJECTS = (
    policy_selection_verification.VerificationOutcome,
    policy_selection_verification.DeterministicVerification,
    policy_selection_verification.extract_final_answer,
    policy_selection_verification.extract_direct_completion,
    policy_selection_verification.parse_t1_answer,
    policy_selection_verification.verify_arxivqa_answer,
    policy_selection_verification.verify_thinklite_answer,
    policy_selection_verification.verify_vstar_answer,
    policy_selection_verification.verify_t1_answer,
)
_MOVED_OBJECTS = _CONFIG_OBJECTS + _EVIDENCE_OBJECTS + _VERIFICATION_OBJECTS
_DATACLASS_PICKLE_HELPERS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
)


def _definitions(filename: str) -> set[str]:
    path = _DATA_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _relative_imports(filename: str) -> set[str]:
    path = _DATA_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_facade_reexports_exact_config_and_verification_objects() -> None:
    for implementation_object in _MOVED_OBJECTS:
        public_object = getattr(
            policy_selection_runtime, implementation_object.__name__
        )
        assert public_object is implementation_object
        assert public_object.__module__ == _PUBLIC_RUNTIME_MODULE
        assert public_object.__qualname__ == implementation_object.__name__


def test_facade_keeps_exact_public_constant_and_export_contract() -> None:
    for name in (
        "T1_ATTEMPTS",
        "T1_ATTEMPT_SEED_SCHEMA",
        "T1_SEED_MODULUS",
        "T1_SHARD_COUNT",
    ):
        assert getattr(policy_selection_runtime, name) is getattr(
            policy_selection_config_values, name
        )
        assert getattr(policy_selection_config, name) is getattr(
            policy_selection_config_values, name
        )
    for name in (
        "T1_INSTRUCT_ANSWER_PARSER",
        "T1_MAX_PIXELS",
        "T1_MODEL_PATH_BY_REPOSITORY",
        "T1_PROMPT_SCHEMA",
        "T1_RUN_CONFIG_SCHEMA",
        "T1_SOURCE_RGB_SCHEMA",
        "T1_THINKING_ANSWER_PARSER",
    ):
        assert getattr(policy_selection_runtime, name) is getattr(
            policy_selection_config_schema, name
        )
        assert getattr(policy_selection_config, name) is getattr(
            policy_selection_config_schema, name
        )
    for name in ("T1_RAW_GENERATION_SCHEMA", "T1_TOKEN_IDS_SCHEMA"):
        assert getattr(policy_selection_runtime, name) is getattr(
            policy_selection_evidence, name
        )

    assert policy_selection_runtime.__all__ == [
        "DeterministicVerification",
        "GenerationDisposition",
        "T1_ATTEMPTS",
        "T1_ATTEMPT_SEED_SCHEMA",
        "T1_CHUNK_MANIFEST_SCHEMA",
        "T1_MAX_PIXELS",
        "T1_INSTRUCT_ANSWER_PARSER",
        "T1_MODEL_PATH_BY_REPOSITORY",
        "T1_PROMPT_SCHEMA",
        "T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA",
        "T1_RAW_GENERATION_SCHEMA",
        "T1_RUN_CONFIG_SCHEMA",
        "T1_SHARD_COUNT",
        "T1_SOURCE_RGB_SCHEMA",
        "T1_THINKING_ANSWER_PARSER",
        "T1ChunkManifest",
        "T1DataSource",
        "T1RawGenerationEvidence",
        "T1ResponseBudget",
        "T1RunConfig",
        "VerificationOutcome",
        "atomic_write_chunk_manifest",
        "candidate_rank",
        "classify_generation_finish",
        "derive_t1_attempt_seed",
        "evidence_to_attempt_record",
        "extract_final_answer",
        "extract_direct_completion",
        "parse_t1_answer",
        "load_resumable_chunk",
        "load_t1_run_config",
        "native_prompt_identity_sha256",
        "native_user_message_descriptor",
        "rendered_prompt_token_ids_sha256",
        "sampled_token_ids_sha256",
        "source_rgb_sha256",
        "validate_chunk_manifest",
        "verify_arxivqa_answer",
        "verify_t1_answer",
        "verify_thinklite_answer",
        "verify_vstar_answer",
        "write_content_addressed_chunk",
    ]
    legacy_dependency_exports = {
        "Any",
        "AttemptStatus",
        "Decimal",
        "Enum",
        "Fraction",
        "InvalidOperation",
        "Mapping",
        "MappingProxyType",
        "POLICY_SELECTION_ATTEMPT_SCHEMA",
        "Path",
        "SelectionBranch",
        "SelectionCandidate",
        "SelectionSource",
        "Sequence",
        "T1_MODEL_IDENTITY_SCHEMA",
        "T1_PROCESSOR_IDENTITY_SCHEMA",
        "T1_PROMPT_IDENTITY_SCHEMA",
        "T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION",
        "T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA",
        "T1_RECOMMENDED_SELECTION_NAMESPACE",
        "T1_RECOMMENDED_SELECTION_ROWS",
        "T1_RECOMMENDED_SOURCE_QUOTAS",
        "T1_RUNTIME_IDENTITY_SCHEMA",
        "T1_SEED_MODULUS",
        "T1_TOKEN_IDS_SCHEMA",
        "annotations",
        "dataclass",
        "hashlib",
        "json",
        "math",
        "os",
        "re",
        "stable_selection_request_id",
        "tempfile",
    }
    assert {
        name for name in vars(policy_selection_runtime) if not name.startswith("_")
    } == set(policy_selection_runtime.__all__) | legacy_dependency_exports


def test_moved_classes_functions_and_instances_keep_pickle_coordinates() -> None:
    for public_object in _MOVED_OBJECTS:
        assert pickle.loads(pickle.dumps(public_object)) is public_object

    run = policy_selection_runtime.T1RunConfig(
        run_id="pickle-characterization",
        manifest_sha256="a" * 64,
        model={},
        prompt={},
        image={},
        sampling={},
        response_budgets=(),
        runtime={},
        data_sources=(),
        selection={},
        verifier={},
        output_root=Path("/tmp/policy-selection-pickle-characterization"),
        _record_bytes=b"{}",
    )
    instances = (
        policy_selection_runtime.T1ResponseBudget(0, 65_536, 40_960),
        policy_selection_runtime.T1DataSource(
            SelectionSource.VSTAR,
            Path("/tmp/vstar.jsonl"),
            "b" * 64,
            1,
        ),
        run,
        policy_selection_runtime.T1RawGenerationEvidence(
            run_id="pickle-characterization",
            run_manifest_sha256="a" * 64,
            request_id="request",
            sample_id="sample",
            candidate_sha256="b" * 64,
            source=SelectionSource.VSTAR,
            attempt_index=0,
            attempt_seed=1,
            budget_revision=0,
            max_model_len=2,
            max_new_tokens=1,
            prompt_sha256="c" * 64,
            rendered_prompt_token_ids_sha256="d" * 64,
            prompt_token_count=1,
            image_sha256="e" * 64,
            source_width=1,
            source_height=1,
            source_mode="RGB",
            source_rgb_sha256="f" * 64,
            processed_width=1,
            processed_height=1,
            sampled_token_ids_sha256="0" * 64,
            sampled_token_count=0,
            sampled_token_ids=(),
            raw_text="",
            finish_reason="stop",
            stop_reason=None,
            backend={},
            generation_error=None,
            _record_bytes=b"{}",
        ),
        policy_selection_runtime.VerificationOutcome.CORRECT,
        policy_selection_runtime.DeterministicVerification(
            policy_selection_runtime.VerificationOutcome.CORRECT,
            "characterization",
            "pickle",
        ),
    )
    for instance in instances:
        restored = pickle.loads(pickle.dumps(instance))
        assert restored == instance
        assert type(restored) is type(instance)


def test_moved_type_hints_resolve_through_facade_and_leaf() -> None:
    owner_groups = (
        (policy_selection_config_schema, _CONFIG_SCHEMA_OBJECTS),
        (policy_selection_config_values, _CONFIG_VALUE_OBJECTS),
        (policy_selection_config, _CONFIG_LOADER_OBJECTS),
        (policy_selection_evidence, _EVIDENCE_OBJECTS),
        (policy_selection_verification, _VERIFICATION_OBJECTS),
    )
    for owner_module, implementation_objects in owner_groups:
        for implementation_object in implementation_objects:
            public_object = getattr(
                policy_selection_runtime, implementation_object.__name__
            )
            assert get_type_hints(public_object) == get_type_hints(
                implementation_object,
                globalns=vars(owner_module),
            )


def test_schema_rebinding_does_not_mutate_shared_dataclass_helpers() -> None:
    for helper in _DATACLASS_PICKLE_HELPERS:
        assert helper.__module__ == dataclasses.__name__

    for schema_type in (
        policy_selection_runtime.T1ResponseBudget,
        policy_selection_runtime.T1DataSource,
        policy_selection_runtime.T1RunConfig,
        policy_selection_runtime.DeterministicVerification,
    ):
        for name in ("__getstate__", "__setstate__"):
            helper = vars(schema_type).get(name)
            if isinstance(helper, FunctionType):
                assert helper.__module__ == dataclasses.__name__


def test_moved_class_methods_keep_legacy_public_coordinates() -> None:
    schema_types = (
        policy_selection_runtime.T1ResponseBudget,
        policy_selection_runtime.T1DataSource,
        policy_selection_runtime.T1RunConfig,
        policy_selection_runtime.GenerationDisposition,
        policy_selection_runtime.T1RawGenerationEvidence,
        policy_selection_runtime.VerificationOutcome,
        policy_selection_runtime.DeterministicVerification,
    )
    for schema_type in schema_types:
        for member in vars(schema_type).values():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (classmethod, staticmethod)):
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if not isinstance(function, FunctionType):
                    continue
                if function in _DATACLASS_PICKLE_HELPERS:
                    assert function.__module__ == dataclasses.__name__
                elif function.__module__ != "enum":
                    assert function.__module__ == _PUBLIC_RUNTIME_MODULE


@pytest.mark.parametrize(
    "imports",
    [
        (
            "tgvf_rl.data.policy_selection_runtime",
            "tgvf_rl.data.policy_selection_config",
            "tgvf_rl.data.policy_selection_config_schema",
            "tgvf_rl.data.policy_selection_config_values",
            "tgvf_rl.data.policy_selection_evidence",
            "tgvf_rl.data.policy_selection_verification",
        ),
        (
            "tgvf_rl.data.policy_selection_config_values",
            "tgvf_rl.data.policy_selection_config_schema",
            "tgvf_rl.data.policy_selection_config",
            "tgvf_rl.data.policy_selection_evidence",
            "tgvf_rl.data.policy_selection_verification",
            "tgvf_rl.data.policy_selection_runtime",
        ),
    ],
)
def test_public_identity_is_stable_across_supported_import_orders(
    imports: tuple[str, ...],
) -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import pickle

        for module_name in {imports!r}:
            importlib.import_module(module_name)
        facade = importlib.import_module("tgvf_rl.data.policy_selection_runtime")
        config = importlib.import_module("tgvf_rl.data.policy_selection_config")
        config_schema = importlib.import_module("tgvf_rl.data.policy_selection_config_schema")
        config_values = importlib.import_module("tgvf_rl.data.policy_selection_config_values")
        evidence = importlib.import_module("tgvf_rl.data.policy_selection_evidence")
        verification = importlib.import_module("tgvf_rl.data.policy_selection_verification")
        for leaf, names in (
            (config, {tuple(item.__name__ for item in _CONFIG_LOADER_OBJECTS)!r}),
            (config_schema, {tuple(item.__name__ for item in _CONFIG_SCHEMA_OBJECTS)!r}),
            (config_values, {tuple(item.__name__ for item in _CONFIG_VALUE_OBJECTS)!r}),
            (evidence, {tuple(item.__name__ for item in _EVIDENCE_OBJECTS)!r}),
            (verification, {tuple(item.__name__ for item in _VERIFICATION_OBJECTS)!r}),
        ):
            for name in names:
                value = getattr(leaf, name)
                assert getattr(facade, name) is value
                assert value.__module__ == "tgvf_rl.data.policy_selection_runtime"
                assert pickle.loads(pickle.dumps(value)) is value
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_split_has_single_ast_ownership_and_one_way_imports() -> None:
    facade_definitions = _definitions("policy_selection_runtime.py")
    config_definitions = _definitions("policy_selection_config.py")
    schema_definitions = _definitions("policy_selection_config_schema.py")
    value_definitions = _definitions("policy_selection_config_values.py")
    evidence_definitions = _definitions("policy_selection_evidence.py")
    verification_definitions = _definitions("policy_selection_verification.py")
    moved_names = {item.__name__ for item in _MOVED_OBJECTS}

    assert not facade_definitions & moved_names
    assert {item.__name__ for item in _CONFIG_LOADER_OBJECTS} <= config_definitions
    assert {item.__name__ for item in _CONFIG_SCHEMA_OBJECTS} <= schema_definitions
    assert {item.__name__ for item in _CONFIG_VALUE_OBJECTS} <= value_definitions
    assert {item.__name__ for item in _EVIDENCE_OBJECTS} <= evidence_definitions
    assert {item.__name__ for item in _VERIFICATION_OBJECTS} <= verification_definitions
    ownership_sets = (
        facade_definitions,
        config_definitions,
        schema_definitions,
        value_definitions,
        evidence_definitions,
        verification_definitions,
    )
    for index, definitions in enumerate(ownership_sets):
        for other_definitions in ownership_sets[index + 1 :]:
            assert not definitions & other_definitions

    config_imports = _relative_imports("policy_selection_config.py")
    schema_imports = _relative_imports("policy_selection_config_schema.py")
    value_imports = _relative_imports("policy_selection_config_values.py")
    evidence_imports = _relative_imports("policy_selection_evidence.py")
    verification_imports = _relative_imports("policy_selection_verification.py")
    facade_imports = _relative_imports("policy_selection_runtime.py")
    leaves = (
        config_imports
        | schema_imports
        | value_imports
        | evidence_imports
        | verification_imports
    )
    assert "policy_selection_runtime" not in leaves
    assert not value_imports
    assert "policy_selection_config_values" in schema_imports
    assert {"policy_selection_config_schema", "policy_selection_config_values"} <= (
        config_imports
    )
    assert {
        "policy_selection_config_schema",
        "policy_selection_config_values",
    } <= evidence_imports
    assert {
        "policy_selection_config_schema",
        "policy_selection_config_values",
    } <= verification_imports
    assert {
        "policy_selection_config",
        "policy_selection_config_values",
        "policy_selection_evidence",
        "policy_selection_verification",
    } <= facade_imports


def test_facade_and_leaves_have_module_size_headroom() -> None:
    for filename in (
        "policy_selection_runtime.py",
        "policy_selection_config.py",
        "policy_selection_config_schema.py",
        "policy_selection_config_values.py",
        "policy_selection_evidence.py",
        "policy_selection_verification.py",
    ):
        path = _DATA_PACKAGE / filename
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= 850, f"{filename} grew to {line_count} lines"


def test_leaves_do_not_use_dynamic_facade_backreferences() -> None:
    for filename in (
        "policy_selection_config.py",
        "policy_selection_config_schema.py",
        "policy_selection_config_values.py",
        "policy_selection_evidence.py",
        "policy_selection_verification.py",
    ):
        path = _DATA_PACKAGE / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_roots = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "sys" not in imported_roots
