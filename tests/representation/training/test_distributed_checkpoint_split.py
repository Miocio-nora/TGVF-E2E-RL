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

from tgvf_rl.representation.training import (
    distributed_checkpoint,
    distributed_checkpoint_export,
    distributed_checkpoint_integrity,
    distributed_checkpoint_schema,
)


_PUBLIC_MODULE = "tgvf_rl.representation.training.distributed_checkpoint"
_TRAINING_PACKAGE = Path(distributed_checkpoint.__file__).parent
_IMPLEMENTATION_FILENAMES = (
    "distributed_checkpoint.py",
    "distributed_checkpoint_export.py",
    "distributed_checkpoint_integrity.py",
    "distributed_checkpoint_schema.py",
)
_PUBLIC_FACADE_NAMES = frozenset(
    {
        "Any",
        "BytesIO",
        "Callable",
        "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION",
        "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2",
        "DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION",
        "DistributedRepresentationCheckpointManifest",
        "DistributedRepresentationMetadata",
        "DistributedRepresentationRankState",
        "DistributedRepresentationResumeResult",
        "IdentityMismatchError",
        "Mapping",
        "Path",
        "Protocol",
        "RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION",
        "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION",
        "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3",
        "RankZeroAdapterOwnedStateExport",
        "RankZeroAdapterOwnedStateManifest",
        "ReplayMismatchError",
        "RepresentationAccumulationIdentity",
        "RepresentationFSDP2Binding",
        "RepresentationMetricsHistoryIdentity",
        "RepresentationOptimizerIdentity",
        "RepresentationRunIdentity",
        "RepresentationRunIdentityV3",
        "RepresentationSamplerContractIdentity",
        "RepresentationSchedulerIdentity",
        "RepresentationTrainerExecutionIdentity",
        "SameImageBatchSampler",
        "TypeVar",
        "annotations",
        "capture_representation_rng_state",
        "cast",
        "dataclass",
        "deepcopy",
        "gather_rank_zero_full_adapter_owned_state",
        "inspect",
        "load_distributed_representation_checkpoint_metadata",
        "load_rank_zero_adapter_owned_state_export",
        "os",
        "restore_distributed_representation_checkpoint",
        "restore_representation_rng_state",
        "save_distributed_representation_checkpoint_atomic",
        "save_rank_zero_adapter_owned_state_export_atomic",
        "sha256",
        "shutil",
        "state_digest",
        "tempfile",
        "tensor_checksum",
        "torch",
        "uuid4",
    }
)
_MOVED_PUBLIC_TYPES = (
    *distributed_checkpoint_schema._DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES,
    *distributed_checkpoint_export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
)
_MOVED_PUBLIC_FUNCTIONS = (
    distributed_checkpoint_integrity.load_distributed_representation_checkpoint_metadata,
    distributed_checkpoint_export.save_rank_zero_adapter_owned_state_export_atomic,
    distributed_checkpoint_export.load_rank_zero_adapter_owned_state_export,
)
_ALL_REBOUND_TYPES = (
    *distributed_checkpoint_schema._DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES,
    *distributed_checkpoint_integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
    *distributed_checkpoint_export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
)
_ALL_REBOUND_FUNCTIONS = (
    *distributed_checkpoint_schema._DISTRIBUTED_CHECKPOINT_SCHEMA_FUNCTIONS,
    *distributed_checkpoint_integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_FUNCTIONS,
    *distributed_checkpoint_export._DISTRIBUTED_CHECKPOINT_EXPORT_FUNCTIONS,
)
_ANNOTATION_FROZEN_TYPES = (
    distributed_checkpoint_schema.DistributedRepresentationCheckpointManifest,
    distributed_checkpoint_schema.DistributedRepresentationMetadata,
    *distributed_checkpoint_integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
    *distributed_checkpoint_export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
)
_DATACLASS_PICKLE_HELPERS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
)


def _definitions(filename: str) -> set[str]:
    path = _TRAINING_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _relative_imports(filename: str) -> set[str]:
    path = _TRAINING_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


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


def test_distributed_checkpoint_facade_keeps_frozen_non_private_namespace() -> None:
    assert {
        name for name in vars(distributed_checkpoint) if not name.startswith("_")
    } == _PUBLIC_FACADE_NAMES


def test_facade_reexports_exact_moved_contracts_and_io_functions() -> None:
    for contract_type in _MOVED_PUBLIC_TYPES:
        assert getattr(distributed_checkpoint, contract_type.__name__) is contract_type
    for function in _MOVED_PUBLIC_FUNCTIONS:
        assert getattr(distributed_checkpoint, function.__name__) is function

    assert (
        distributed_checkpoint.RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION
        is distributed_checkpoint_export.RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION
    )
    for name in (
        "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION",
        "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2",
        "DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION",
    ):
        assert getattr(distributed_checkpoint, name) is getattr(
            distributed_checkpoint_schema, name
        )


def test_every_rebound_object_is_resolvable_and_picklable_through_facade() -> None:
    for value in (*_ALL_REBOUND_TYPES, *_ALL_REBOUND_FUNCTIONS):
        assert getattr(distributed_checkpoint, value.__name__) is value
        assert value.__module__ == _PUBLIC_MODULE
        assert value.__qualname__ == value.__name__
        assert pickle.loads(pickle.dumps(value)) is value


def test_rebound_class_annotations_are_resolved_before_module_rebinding() -> None:
    for contract_type in _ALL_REBOUND_TYPES:
        resolved = get_type_hints(contract_type)
        own_annotations = vars(contract_type).get("__annotations__", {})
        if contract_type in _ANNOTATION_FROZEN_TYPES:
            assert all(
                not isinstance(annotation, str)
                for annotation in own_annotations.values()
            )
            assert own_annotations == {name: resolved[name] for name in own_annotations}


def test_moved_contracts_keep_public_method_type_hint_and_pickle_coordinates() -> None:
    for contract_type in _MOVED_PUBLIC_TYPES:
        assert contract_type.__module__ == _PUBLIC_MODULE
        assert contract_type.__qualname__ == contract_type.__name__
        assert pickle.loads(pickle.dumps(contract_type)) is contract_type
        assert get_type_hints(
            getattr(distributed_checkpoint, contract_type.__name__)
        ) == (get_type_hints(contract_type))

        restored = pickle.loads(
            pickle.dumps(_unvalidated_dataclass_instance(contract_type))
        )
        assert type(restored) is contract_type

        for member in vars(contract_type).values():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (classmethod, staticmethod)):
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if not isinstance(function, FunctionType):
                    continue
                expected_module = (
                    dataclasses.__name__
                    if function in _DATACLASS_PICKLE_HELPERS
                    else _PUBLIC_MODULE
                )
                assert function.__module__ == expected_module

    for function in _MOVED_PUBLIC_FUNCTIONS:
        assert function.__module__ == _PUBLIC_MODULE
        assert function.__qualname__ == function.__name__
        assert pickle.loads(pickle.dumps(function)) is function
        assert get_type_hints(getattr(distributed_checkpoint, function.__name__)) == (
            get_type_hints(function)
        )


def test_compatibility_rebinding_does_not_mutate_shared_dataclass_helpers() -> None:
    for helper in _DATACLASS_PICKLE_HELPERS:
        assert helper.__module__ == dataclasses.__name__


def test_distributed_checkpoint_implementation_has_single_ast_ownership() -> None:
    owners: dict[str, str] = {}
    for filename in _IMPLEMENTATION_FILENAMES:
        for definition in _definitions(filename):
            assert definition not in owners, (
                f"{definition} is owned by both {owners[definition]} and {filename}"
            )
            owners[definition] = filename

    assert _definitions("distributed_checkpoint.py") == {
        "_gather_rank_states",
        "_validate_runtime_identity",
        "_validate_scheduler_runtime",
        "gather_rank_zero_full_adapter_owned_state",
        "restore_distributed_representation_checkpoint",
        "save_distributed_representation_checkpoint_atomic",
    }
    assert not _definitions("distributed_checkpoint_export.py") & _definitions(
        "distributed_checkpoint_schema.py"
    )


def test_distributed_checkpoint_imports_are_one_way_without_dynamic_backrefs() -> None:
    facade_imports = _relative_imports("distributed_checkpoint.py")
    schema_imports = _relative_imports("distributed_checkpoint_schema.py")
    integrity_imports = _relative_imports("distributed_checkpoint_integrity.py")
    export_imports = _relative_imports("distributed_checkpoint_export.py")

    assert {
        "distributed_checkpoint_export",
        "distributed_checkpoint_integrity",
        "distributed_checkpoint_schema",
    } <= facade_imports
    assert not schema_imports & {
        "distributed_checkpoint",
        "distributed_checkpoint_export",
        "distributed_checkpoint_integrity",
    }
    assert "distributed_checkpoint_schema" in integrity_imports
    assert not integrity_imports & {
        "distributed_checkpoint",
        "distributed_checkpoint_export",
    }
    assert {
        "distributed_checkpoint_integrity",
        "distributed_checkpoint_schema",
    } <= export_imports
    assert "distributed_checkpoint" not in export_imports

    for filename in _IMPLEMENTATION_FILENAMES:
        payload = (_TRAINING_PACKAGE / filename).read_text(encoding="utf-8")
        assert "sys.modules" not in payload


def test_distributed_checkpoint_facade_and_leaves_have_headroom() -> None:
    for filename in _IMPLEMENTATION_FILENAMES:
        line_count = len(
            (_TRAINING_PACKAGE / filename).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 850, f"{filename} grew to {line_count} lines"


def test_leaf_first_and_facade_first_import_orders_converge() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    training_directory = (
        repository_root / "src" / "tgvf_rl" / "representation" / "training"
    )
    script = textwrap.dedent(
        """
        import importlib
        import pickle
        import sys
        import types
        from typing import get_type_hints

        import tgvf_rl.representation

        package = types.ModuleType("tgvf_rl.representation.training")
        package.__package__ = "tgvf_rl.representation.training"
        package.__path__ = [sys.argv[1]]
        sys.modules["tgvf_rl.representation.training"] = package

        facade_name = "tgvf_rl.representation.training.distributed_checkpoint"

        def assert_leaf_hints(contract_types, annotation_frozen_types):
            assert facade_name not in sys.modules
            for contract_type in contract_types:
                assert contract_type.__module__ == facade_name
                assert contract_type.__qualname__ == contract_type.__name__
                resolved = get_type_hints(contract_type)
                own_annotations = vars(contract_type).get("__annotations__", {})
                if contract_type in annotation_frozen_types:
                    assert all(
                        not isinstance(annotation, str)
                        for annotation in own_annotations.values()
                    )
                    assert own_annotations == {
                        name: resolved[name] for name in own_annotations
                    }
                assert facade_name not in sys.modules

        schema = importlib.import_module(
            "tgvf_rl.representation.training.distributed_checkpoint_schema"
        )
        assert_leaf_hints(
            schema._DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES,
            (
                schema.DistributedRepresentationCheckpointManifest,
                schema.DistributedRepresentationMetadata,
            ),
        )

        integrity = importlib.import_module(
            "tgvf_rl.representation.training.distributed_checkpoint_integrity"
        )
        assert_leaf_hints(
            integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
            integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
        )

        export = importlib.import_module(
            "tgvf_rl.representation.training.distributed_checkpoint_export"
        )
        assert_leaf_hints(
            export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
            export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
        )

        encoded = [
            pickle.dumps(contract_type)
            for contract_type in (
                *schema._DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES,
                *integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
                *export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
            )
        ]
        facade = importlib.import_module(facade_name)
        assert (
            facade.DistributedRepresentationResumeResult
            is schema.DistributedRepresentationResumeResult
        )
        assert (
            facade.RankZeroAdapterOwnedStateExport
            is export.RankZeroAdapterOwnedStateExport
        )
        assert (
            facade.load_distributed_representation_checkpoint_metadata
            is integrity.load_distributed_representation_checkpoint_metadata
        )
        assert [
            pickle.loads(payload) for payload in encoded
        ] == [
            *schema._DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES,
            *integrity._DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES,
            *export._DISTRIBUTED_CHECKPOINT_EXPORT_TYPES,
        ]
        """
    )
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    environment.pop("OPENROUTER_API_KEY", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""

    subprocess.run(
        [sys.executable, "-c", script, str(training_directory)],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )
