from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
import pickle
from types import FunctionType
from typing import get_type_hints

import pytest

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.representation.training import (
    checkpoint,
    checkpoint_identity,
    checkpoint_integrity,
    checkpoint_schema,
)


_PUBLIC_CHECKPOINT_MODULE = "tgvf_rl.representation.training.checkpoint"
_FROZEN_PUBLIC_FACADE_NAMES = frozenset(
    {
        "CodeIdentity",
        "Enum",
        "IdentityMismatchError",
        "L_GEN_GLOBAL_REDUCTION",
        "MATRIX_CE_GLOBAL_REDUCTION",
        "Mapping",
        "ModelIdentity",
        "Path",
        "Protocol",
        "REPRESENTATION_ACCUMULATION_SCHEMA_VERSION",
        "REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2",
        "REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION",
        "REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION",
        "REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2",
        "REPRESENTATION_INITIALIZATION_SCHEMA_VERSION",
        "REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION",
        "REPRESENTATION_RNG_STATE_SCHEMA_VERSION",
        "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION",
        "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2",
        "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3",
        "REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION",
        "REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION",
        "REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2",
        "REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION",
        "REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION",
        "ReplayMismatchError",
        "RepresentationAccumulationIdentity",
        "RepresentationAccumulationIdentityV2",
        "RepresentationAdapterArtifact",
        "RepresentationAdapterArtifactManifest",
        "RepresentationAdapterContractIdentity",
        "RepresentationAdapterContractIdentityV2",
        "RepresentationInitializationIdentity",
        "RepresentationObjectiveConfig",
        "RepresentationOptimizerIdentity",
        "RepresentationResumeResult",
        "RepresentationRunIdentity",
        "RepresentationRunIdentityV3",
        "RepresentationSamplerContractIdentity",
        "RepresentationSchedulerIdentity",
        "RepresentationSchedulerIdentityV2",
        "RepresentationTensorManifestEntry",
        "RepresentationTrainerExecutionIdentity",
        "RepresentationTrainingCheckpoint",
        "RepresentationTrainingCheckpointManifest",
        "RepresentationValidationDataIdentity",
        "SAMPLER_IDENTITY_SCHEMA_VERSION",
        "SAMPLER_STATE_SCHEMA_VERSION",
        "SameImageBatchSampler",
        "Sequence",
        "TGVFAdapter",
        "TGVFAdapterVariant",
        "TargetConditioningConfig",
        "annotations",
        "capture_representation_rng_state",
        "dataclass",
        "deepcopy",
        "hashlib",
        "json",
        "load_representation_adapter_artifact",
        "load_representation_training_checkpoint",
        "math",
        "os",
        "random",
        "representation_adapter_contract_identity",
        "restore_representation_adapter_artifact",
        "restore_representation_rng_state",
        "restore_representation_training_checkpoint",
        "save_representation_adapter_artifact_atomic",
        "save_representation_training_checkpoint_atomic",
        "spec_identity_sha256",
        "tempfile",
        "tensor_checksum",
        "torch",
    }
)
_DATACLASS_PICKLE_HELPERS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
)
_CHECKPOINT_TYPES = (
    *checkpoint_identity._CHECKPOINT_IDENTITY_TYPES,
    *checkpoint_schema._CHECKPOINT_SCHEMA_TYPES,
)
_SCHEMA_FUNCTIONS = (
    checkpoint_identity._validate_accumulation_identity,
    checkpoint_identity.representation_adapter_contract_identity,
    checkpoint_identity._validate_run_identity,
    checkpoint_schema._validate_tensor_manifest,
)
_PUBLIC_IDENTITY_CONSTANTS = (
    "L_GEN_GLOBAL_REDUCTION",
    "MATRIX_CE_GLOBAL_REDUCTION",
    "REPRESENTATION_ACCUMULATION_SCHEMA_VERSION",
    "REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2",
    "REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION",
    "REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2",
    "REPRESENTATION_INITIALIZATION_SCHEMA_VERSION",
    "REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION",
    "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION",
    "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2",
    "REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3",
    "REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION",
    "REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION",
    "REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2",
    "REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION",
)
_PUBLIC_PAYLOAD_CONSTANTS = (
    "REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION",
    "REPRESENTATION_RNG_STATE_SCHEMA_VERSION",
    "REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION",
)


def test_checkpoint_facade_reexports_exact_schema_objects() -> None:
    for schema_type in _CHECKPOINT_TYPES:
        assert getattr(checkpoint, schema_type.__name__) is schema_type
    for function in _SCHEMA_FUNCTIONS:
        assert getattr(checkpoint, function.__name__) is function
    for name in _PUBLIC_IDENTITY_CONSTANTS:
        assert getattr(checkpoint, name) is getattr(checkpoint_identity, name)
    for name in _PUBLIC_PAYLOAD_CONSTANTS:
        assert getattr(checkpoint, name) is getattr(checkpoint_schema, name)


def test_checkpoint_facade_keeps_frozen_non_private_namespace() -> None:
    assert {
        name for name in vars(checkpoint) if not name.startswith("_")
    } == _FROZEN_PUBLIC_FACADE_NAMES


def test_checkpoint_facade_reexports_exact_integrity_helpers() -> None:
    for function in checkpoint_integrity._INTEGRITY_FUNCTIONS:
        assert getattr(checkpoint, function.__name__) is function


def test_extracted_checkpoint_types_keep_public_and_pickle_identity() -> None:
    for schema_type in _CHECKPOINT_TYPES:
        assert schema_type.__module__ == _PUBLIC_CHECKPOINT_MODULE
        assert pickle.loads(pickle.dumps(schema_type)) is schema_type
        for member in vars(schema_type).values():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (classmethod, staticmethod)):
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if isinstance(function, FunctionType):
                    expected_module = (
                        dataclasses.__name__
                        if function in _DATACLASS_PICKLE_HELPERS
                        else _PUBLIC_CHECKPOINT_MODULE
                    )
                    assert function.__module__ == expected_module

    for function in _SCHEMA_FUNCTIONS:
        assert function.__module__ == _PUBLIC_CHECKPOINT_MODULE


def test_checkpoint_compatibility_does_not_mutate_dataclass_helpers() -> None:
    for helper in _DATACLASS_PICKLE_HELPERS:
        assert helper.__module__ == dataclasses.__name__

    for schema_type in _CHECKPOINT_TYPES:
        for name in ("__getstate__", "__setstate__"):
            helper = vars(schema_type).get(name)
            if isinstance(helper, FunctionType):
                assert helper.__module__ == dataclasses.__name__


def test_checkpoint_schema_type_hints_resolve_through_facade() -> None:
    for schema_type in _CHECKPOINT_TYPES:
        facade_type = getattr(checkpoint, schema_type.__name__)
        assert get_type_hints(schema_type)
        assert get_type_hints(facade_type) == get_type_hints(schema_type)


def test_checkpoint_schema_instance_pickle_round_trip() -> None:
    result = checkpoint_schema.RepresentationResumeResult(
        global_step=7,
        next_global_step=8,
        run_identity_sha256="a" * 64,
        checkpoint_identity_sha256="b" * 64,
    )

    restored = pickle.loads(pickle.dumps(result))

    assert restored == result
    assert type(restored) is checkpoint.RepresentationResumeResult


def test_runtime_loader_still_observes_facade_monkeypatch(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(checkpoint, "_torch_load", lambda _path: sentinel)

    with pytest.raises(
        ReplayMismatchError,
        match="file is not a representation training checkpoint",
    ):
        checkpoint.load_representation_training_checkpoint("unused.pt")


def test_checkpoint_facade_contains_no_moved_schema_declarations() -> None:
    tree = ast.parse(Path(checkpoint.__file__).read_text(encoding="utf-8"))
    facade_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    moved = {schema_type.__name__ for schema_type in _CHECKPOINT_TYPES}

    assert facade_classes == {"_Stateful"}
    assert not facade_classes & moved


def test_checkpoint_leaves_are_acyclic_and_below_module_ceiling() -> None:
    training_directory = Path(checkpoint.__file__).parent
    for module in (
        checkpoint,
        checkpoint_identity,
        checkpoint_schema,
        checkpoint_integrity,
    ):
        path = Path(module.__file__)
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 850

    identity_tree = ast.parse(
        (training_directory / "checkpoint_identity.py").read_text(encoding="utf-8")
    )
    schema_tree = ast.parse(
        (training_directory / "checkpoint_schema.py").read_text(encoding="utf-8")
    )
    integrity_tree = ast.parse(
        (training_directory / "checkpoint_integrity.py").read_text(encoding="utf-8")
    )
    for tree in (identity_tree, schema_tree, integrity_tree):
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "checkpoint" not in imported_modules
        assert "tgvf_rl.representation.training.checkpoint" not in imported_modules

    schema_functions = {
        node.name for node in schema_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert not schema_functions & {
        "save_representation_adapter_artifact_atomic",
        "load_representation_adapter_artifact",
        "restore_representation_adapter_artifact",
        "save_representation_training_checkpoint_atomic",
        "load_representation_training_checkpoint",
        "restore_representation_training_checkpoint",
    }
