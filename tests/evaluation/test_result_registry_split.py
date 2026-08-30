from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import FunctionType, SimpleNamespace
from typing import get_type_hints

import pytest

from tgvf_rl.evaluation import (
    result_registry,
    result_registry_schema,
    result_registry_support,
)


_PUBLIC_MODULE = "tgvf_rl.evaluation.result_registry"
_EVALUATION_PACKAGE = Path(result_registry.__file__).parent
_IMPLEMENTATION_FILENAMES = (
    "result_registry.py",
    "result_registry_schema.py",
    "result_registry_support.py",
)
_PUBLIC_FACADE_NAMES = {
    "Any",
    "COREDEV_COMPONENTS",
    "ComparisonContract",
    "ComparisonDefinition",
    "CoreDevScore",
    "Enum",
    "INTERVENTION_AXES",
    "INVARIANT_FIELDS",
    "IncomparableResultsError",
    "Mapping",
    "Path",
    "PreregistrationEvidence",
    "PurePosixPath",
    "RESULT_REGISTRY_SCHEMA",
    "RegistryValidationError",
    "ResultRecord",
    "ResultRegistry",
    "ResultStatus",
    "ResultTable",
    "ScoreArtifact",
    "SecureFileReadError",
    "Sequence",
    "WeightIdentity",
    "WeightState",
    "annotations",
    "dataclass",
    "json",
    "load_result_registry",
    "math",
    "read_regular_file_beneath_absolute_directory_nofollow",
    "sha256",
}
_EXPECTED_MOVED_TYPE_NAMES = (
    "RegistryValidationError",
    "IncomparableResultsError",
    "ResultStatus",
    "WeightState",
    "ScoreArtifact",
    "PreregistrationEvidence",
    "WeightIdentity",
    "ComparisonContract",
    "CoreDevScore",
    "ComparisonDefinition",
    "ResultRecord",
    "ResultTable",
)
_EXPECTED_MOVED_FUNCTION_NAMES = (
    "_object",
    "_exact_keys",
    "_text",
    "_sha256",
    "_positive_int",
    "_optional_positive_int",
    "_finite_score",
    "_canonical_sha256",
    "_read_repository_regular_file",
    "_require_matching_score",
    "_validate_comparison_contract",
)
_EXPECTED_FROZEN_ANNOTATION_TYPE_NAMES = (
    "ScoreArtifact",
    "PreregistrationEvidence",
    "WeightIdentity",
    "ComparisonContract",
    "CoreDevScore",
    "ComparisonDefinition",
    "ResultRecord",
    "ResultTable",
)
# Derive compatibility subjects from hard-coded historical names rather than
# the production manifests being tested below.
_MOVED_TYPES = tuple(
    getattr(result_registry_schema, name) for name in _EXPECTED_MOVED_TYPE_NAMES
)
_MOVED_FUNCTIONS = tuple(
    getattr(result_registry_schema, name) for name in _EXPECTED_MOVED_FUNCTION_NAMES
)
_MOVED_DATACLASSES = tuple(
    getattr(result_registry_schema, name)
    for name in _EXPECTED_FROZEN_ANNOTATION_TYPE_NAMES
)


def _definitions(filename: str) -> set[str]:
    path = _EVALUATION_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _relative_imports(filename: str) -> set[str]:
    path = _EVALUATION_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def _member_functions(contract_type: type[object]) -> tuple[FunctionType, ...]:
    functions: list[FunctionType] = []
    for member in vars(contract_type).values():
        if isinstance(member, property):
            candidates = (member.fget, member.fset, member.fdel)
        elif isinstance(member, (classmethod, staticmethod)):
            candidates = (member.__func__,)
        else:
            candidates = (member,)
        functions.extend(
            candidate for candidate in candidates if isinstance(candidate, FunctionType)
        )
    return tuple(functions)


def _unvalidated_dataclass_instance(contract_type: type[object]) -> object:
    value = object.__new__(contract_type)
    for field in dataclasses.fields(contract_type):
        if field.default is not dataclasses.MISSING:
            field_value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            field_value = field.default_factory()
        else:
            field_value = None
        object.__setattr__(value, field.name, field_value)
    return value


def _run_import_script(script: str) -> None:
    repository_root = _EVALUATION_PACKAGE.parents[2]
    environment = os.environ.copy()
    environment.pop("OPENROUTER_API_KEY", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(repository_root / "src"),
            str(repository_root),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_production_compatibility_manifests_match_hard_coded_seam() -> None:
    assert (
        tuple(
            value.__name__
            for value in result_registry_schema._RESULT_REGISTRY_SCHEMA_TYPES
        )
        == _EXPECTED_MOVED_TYPE_NAMES
    )
    assert (
        tuple(
            value.__name__
            for value in result_registry_schema._RESULT_REGISTRY_SCHEMA_FUNCTIONS
        )
        == _EXPECTED_MOVED_FUNCTION_NAMES
    )
    assert (
        tuple(
            value.__name__ for value in result_registry_schema._ANNOTATION_FROZEN_TYPES
        )
        == _EXPECTED_FROZEN_ANNOTATION_TYPE_NAMES
    )


def test_result_registry_facade_keeps_frozen_non_private_namespace() -> None:
    assert {
        name for name in vars(result_registry) if not name.startswith("_")
    } == _PUBLIC_FACADE_NAMES


def test_facade_reexports_exact_schema_objects_and_legacy_dependencies() -> None:
    for value in (*_MOVED_TYPES, *_MOVED_FUNCTIONS):
        assert getattr(result_registry, value.__name__) is value

    for name in (
        "Enum",
        "Mapping",
        "PurePosixPath",
        "WeightState",
        "math",
        "sha256",
    ):
        assert getattr(result_registry, name) is getattr(result_registry_schema, name)

    for name in (
        "Any",
        "COREDEV_COMPONENTS",
        "INTERVENTION_AXES",
        "INVARIANT_FIELDS",
        "RESULT_REGISTRY_SCHEMA",
        "SecureFileReadError",
        "read_regular_file_beneath_absolute_directory_nofollow",
        "_GOLDEN_PROMOTION_BLOCKED_REASON",
        "_PREREGISTRATION_SCHEMA",
        "_SCORE_MATCH_ABS_TOLERANCE",
        "_SHA256_LENGTH",
    ):
        assert getattr(result_registry, name) is getattr(result_registry_support, name)


def test_secure_read_helper_uses_patchable_canonical_support_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    expected = b"canonical support dependency"

    def fake_secure_read(root: Path, relative_path: str) -> SimpleNamespace:
        calls.append((root, relative_path))
        return SimpleNamespace(payload=expected)

    # Arbitrary facade sibling monkeypatching is not promised by
    # public_api_compat.  The canonical implementation dependency remains
    # explicitly controllable at the one-way support leaf instead.
    monkeypatch.setattr(
        result_registry_support,
        "read_regular_file_beneath_absolute_directory_nofollow",
        fake_secure_read,
    )
    assert result_registry._read_repository_regular_file is (
        result_registry_support._read_repository_regular_file
    )
    assert (
        result_registry._read_repository_regular_file(
            Path("/registry-root"),
            "scores/fixture.json",
            context="secure read compatibility fixture",
        )
        == expected
    )
    assert calls == [(Path("/registry-root"), "scores/fixture.json")]


def test_every_moved_object_keeps_historical_coordinates_and_pickle_path() -> None:
    for value in (*_MOVED_TYPES, *_MOVED_FUNCTIONS):
        assert value.__module__ == _PUBLIC_MODULE
        assert value.__qualname__ == value.__name__
        assert pickle.loads(pickle.dumps(value)) is value

    for contract_type in _MOVED_TYPES:
        for function in _member_functions(contract_type):
            if function.__module__ == _PUBLIC_MODULE:
                assert function.__qualname__.startswith(
                    f"{contract_type.__qualname__}."
                )
            else:
                assert function.__module__ in {dataclasses.__name__, "enum"}


def test_moved_instances_pickle_through_historical_facade() -> None:
    for contract_type in _MOVED_DATACLASSES:
        value = _unvalidated_dataclass_instance(contract_type)
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is contract_type
        assert restored == value

    for status in (*result_registry.ResultStatus, *result_registry.WeightState):
        assert pickle.loads(pickle.dumps(status)) is status

    for error_type in (
        result_registry.RegistryValidationError,
        result_registry.IncomparableResultsError,
    ):
        error = error_type("registry compatibility fixture")
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is error_type
        assert restored.args == error.args


def test_moved_type_hints_resolve_and_class_annotations_are_frozen() -> None:
    for contract_type in _MOVED_DATACLASSES:
        own_annotations = vars(contract_type).get("__annotations__", {})
        resolved = get_type_hints(contract_type)
        assert own_annotations == {name: resolved[name] for name in own_annotations}
        assert all(
            not isinstance(annotation, str) for annotation in own_annotations.values()
        )

    for contract_type in _MOVED_TYPES:
        assert get_type_hints(contract_type) == get_type_hints(
            contract_type,
            globalns=vars(result_registry_schema),
        )
    for function in _MOVED_FUNCTIONS:
        owner = (
            result_registry_schema
            if function.__name__
            in {"_require_matching_score", "_validate_comparison_contract"}
            else result_registry_support
        )
        assert get_type_hints(function) == get_type_hints(
            function,
            globalns=vars(owner),
        )
    for contract_type in _MOVED_TYPES:
        for function in _member_functions(contract_type):
            if function.__annotations__:
                assert get_type_hints(function)


def test_result_registry_split_has_single_ast_ownership() -> None:
    facade_definitions = _definitions("result_registry.py")
    schema_definitions = _definitions("result_registry_schema.py")
    support_definitions = _definitions("result_registry_support.py")

    assert facade_definitions == {
        "ResultRegistry",
        "load_result_registry",
    }
    assert schema_definitions == {
        "ComparisonContract",
        "ComparisonDefinition",
        "CoreDevScore",
        "PreregistrationEvidence",
        "ResultRecord",
        "ResultStatus",
        "ResultTable",
        "ScoreArtifact",
        "WeightIdentity",
        "WeightState",
        "_require_matching_score",
        "_validate_comparison_contract",
    }
    assert support_definitions == {
        "IncomparableResultsError",
        "RegistryValidationError",
        "_canonical_sha256",
        "_exact_keys",
        "_finite_score",
        "_object",
        "_optional_positive_int",
        "_positive_int",
        "_publish_result_registry_schema",
        "_read_repository_regular_file",
        "_sha256",
        "_text",
    }
    assert not facade_definitions & schema_definitions
    assert not facade_definitions & support_definitions
    assert not schema_definitions & support_definitions


def test_result_registry_imports_are_one_way_without_dynamic_backrefs() -> None:
    assert _relative_imports("result_registry.py") == {
        "result_registry_schema",
        "result_registry_support",
    }
    assert _relative_imports("result_registry_schema.py") == {"result_registry_support"}
    assert not _relative_imports("result_registry_support.py")

    for filename in ("result_registry_schema.py", "result_registry_support.py"):
        leaf_source = (_EVALUATION_PACKAGE / filename).read_text(encoding="utf-8")
        assert "sys.modules" not in leaf_source
        assert "importlib" not in leaf_source


def test_result_registry_facade_and_leaf_have_headroom() -> None:
    for filename in _IMPLEMENTATION_FILENAMES:
        line_count = len(
            (_EVALUATION_PACKAGE / filename).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 850, f"{filename} grew to {line_count} lines"


def test_result_registry_leaf_first_import_resolves_hints_and_pickle() -> None:
    _run_import_script(
        f"""
        import importlib
        import pickle
        import sys
        from typing import get_type_hints

        import tgvf_rl.evaluation

        facade_name = "tgvf_rl.evaluation.result_registry"
        leaf_name = "tgvf_rl.evaluation.result_registry_schema"
        support_name = "tgvf_rl.evaluation.result_registry_support"
        assert facade_name not in sys.modules
        importlib.import_module(support_name)
        assert facade_name not in sys.modules
        leaf = importlib.import_module(leaf_name)
        assert facade_name not in sys.modules
        expected_type_names = {_EXPECTED_MOVED_TYPE_NAMES!r}
        expected_function_names = {_EXPECTED_MOVED_FUNCTION_NAMES!r}
        expected_frozen_names = {_EXPECTED_FROZEN_ANNOTATION_TYPE_NAMES!r}
        assert tuple(
            value.__name__ for value in leaf._RESULT_REGISTRY_SCHEMA_TYPES
        ) == expected_type_names
        assert tuple(
            value.__name__ for value in leaf._RESULT_REGISTRY_SCHEMA_FUNCTIONS
        ) == expected_function_names
        assert tuple(
            value.__name__ for value in leaf._ANNOTATION_FROZEN_TYPES
        ) == expected_frozen_names
        moved = tuple(
            getattr(leaf, name)
            for name in (*expected_type_names, *expected_function_names)
        )
        for contract_type in leaf._ANNOTATION_FROZEN_TYPES:
            own_annotations = vars(contract_type).get("__annotations__", {{}})
            assert all(
                not isinstance(annotation, str)
                for annotation in own_annotations.values()
            )
            assert get_type_hints(contract_type)
        facade = importlib.import_module(facade_name)
        for value in moved:
            assert getattr(facade, value.__name__) is value
            assert pickle.loads(pickle.dumps(value)) is value
        """
    )


def test_result_registry_facade_first_import_converges_on_leaf_identity() -> None:
    _run_import_script(
        f"""
        import importlib
        import pickle

        facade = importlib.import_module("tgvf_rl.evaluation.result_registry")
        leaf = importlib.import_module(
            "tgvf_rl.evaluation.result_registry_schema"
        )
        expected_names = (
            *{_EXPECTED_MOVED_TYPE_NAMES!r},
            *{_EXPECTED_MOVED_FUNCTION_NAMES!r},
        )
        moved = tuple(
            getattr(leaf, name) for name in expected_names
        )
        for value in moved:
            assert getattr(facade, value.__name__) is value
            assert pickle.loads(pickle.dumps(value)) is value
        """
    )
