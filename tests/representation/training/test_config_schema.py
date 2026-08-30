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
    config,
    config_binding,
    config_parser,
    config_run_schema,
    config_schema,
    config_values,
)


_PUBLIC_CONFIG_MODULE = "tgvf_rl.representation.training.config"
_TRAINING_PACKAGE = (
    Path(__file__).parents[3] / "src" / "tgvf_rl" / "representation" / "training"
)
_SCHEMA_MODULES = (config_schema, config_run_schema)
_SCHEMA_TYPES = tuple(
    value
    for schema_module in _SCHEMA_MODULES
    for name in schema_module.__all__
    if isinstance((value := getattr(schema_module, name)), type)
)
_SHARED_DATACLASS_HELPERS = (
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


def _training_imports(filename: str) -> set[str]:
    path = _TRAINING_PACKAGE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_config_facade_reexports_exact_schema_and_leaf_objects() -> None:
    for schema_module in _SCHEMA_MODULES:
        for name in schema_module.__all__:
            assert getattr(config, name) is getattr(schema_module, name)

    for name in _definitions("config_parser.py"):
        assert getattr(config, name) is getattr(config_parser, name)
    binding_only_values = {
        "_optional_existing_file_probe",
        "_read_existing_file_bytes",
        "_require_existing_file_probe",
    }
    for name in _definitions("config_values.py") - binding_only_values:
        assert getattr(config, name) is getattr(config_values, name)

    assert config._verify_external_files is config_binding._verify_external_files


def test_extracted_schema_preserves_public_method_and_pickle_coordinates() -> None:
    assert len(_SCHEMA_TYPES) == 18
    for schema_type in _SCHEMA_TYPES:
        assert schema_type.__module__ == _PUBLIC_CONFIG_MODULE
        assert pickle.loads(pickle.dumps(schema_type)) is schema_type
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
                expected_module = (
                    dataclasses.__name__
                    if function in _SHARED_DATACLASS_HELPERS
                    else _PUBLIC_CONFIG_MODULE
                )
                assert function.__module__ == expected_module


def test_schema_rebinding_does_not_mutate_shared_dataclass_helpers() -> None:
    for helper in _SHARED_DATACLASS_HELPERS:
        assert helper.__module__ == dataclasses.__name__

    for schema_type in _SCHEMA_TYPES:
        for name in ("__getstate__", "__setstate__"):
            helper = vars(schema_type).get(name)
            if isinstance(helper, FunctionType):
                assert helper.__module__ == dataclasses.__name__


def test_schema_type_hints_resolve_through_facade_and_leaf() -> None:
    for schema_type in _SCHEMA_TYPES:
        facade_type = getattr(config, schema_type.__name__)
        assert facade_type is schema_type
        assert get_type_hints(facade_type) == get_type_hints(schema_type)


def test_schema_instance_round_trips_through_legacy_pickle_path() -> None:
    code = config_schema.RepresentationCodeConfig(
        repository="Miocio-nora/TGVF-E2E-RL",
        commit="000275ad8803ddb3092e1c341bd496e3847d8029",
        dirty=False,
        dirty_state_sha256=None,
    )

    restored = pickle.loads(pickle.dumps(code))

    assert restored == code
    assert type(restored) is config.RepresentationCodeConfig


def test_public_loader_retains_facade_monkeypatch_boundary(monkeypatch) -> None:
    # This is the pre-existing loader hook: the facade resolves it at call
    # time. Moved private parser globals are intentionally not a public
    # monkeypatch surface.
    repository_root = Path(__file__).resolve().parents[3]
    path = (
        repository_root
        / "configs"
        / "representation"
        / "qwen3_v4_contextual_hidden_state_v4.toml"
    )
    calls: list[tuple[object, bool]] = []

    def record_external_binding(
        loaded: object,
        *,
        allow_existing_post_training_report: bool = False,
    ) -> None:
        calls.append((loaded, allow_existing_post_training_report))

    monkeypatch.setattr(config, "_verify_external_files", record_external_binding)

    loaded = config.load_representation_training_config(path)

    assert calls == [(loaded, False)]
    assert loaded.run_id == "REP-QWEN3-V4-CONTEXTUAL-V4"


def test_config_implementation_has_single_ownership_and_one_way_imports() -> None:
    implementation_files = (
        "config.py",
        "config_binding.py",
        "config_parser.py",
        "config_run_schema.py",
        "config_schema.py",
        "config_values.py",
    )
    owners: dict[str, str] = {}
    for filename in implementation_files:
        for definition in _definitions(filename):
            assert definition not in owners, (
                f"{definition} is owned by both {owners[definition]} and {filename}"
            )
            owners[definition] = filename

    assert len(owners) == 62
    assert _definitions("config.py") == {"load_representation_training_config"}

    assert _training_imports("config_values.py") == set()
    assert "config_values" in _training_imports("config_schema.py")
    assert not _training_imports("config_schema.py") & {
        "config",
        "config_binding",
        "config_parser",
        "config_run_schema",
    }
    assert {"config_schema", "config_values"} <= _training_imports(
        "config_run_schema.py"
    )
    assert not _training_imports("config_run_schema.py") & {
        "config",
        "config_binding",
        "config_parser",
    }
    assert {"config_schema", "config_values"} <= _training_imports("config_parser.py")
    assert not _training_imports("config_parser.py") & {
        "config",
        "config_binding",
        "config_run_schema",
    }
    assert {"config_run_schema", "config_values"} <= _training_imports(
        "config_binding.py"
    )
    assert not _training_imports("config_binding.py") & {
        "config",
        "config_parser",
        "config_schema",
    }


def test_config_facade_and_responsibility_leaves_have_schema_headroom() -> None:
    for filename in (
        "config.py",
        "config_binding.py",
        "config_parser.py",
        "config_run_schema.py",
        "config_schema.py",
        "config_values.py",
    ):
        line_count = len(
            (_TRAINING_PACKAGE / filename).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 850, f"{filename} grew to {line_count} lines"


def test_schema_leaf_can_load_without_parser_binding_or_facade() -> None:
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

        import tgvf_rl.representation

        package = types.ModuleType("tgvf_rl.representation.training")
        package.__package__ = "tgvf_rl.representation.training"
        package.__path__ = [sys.argv[1]]
        sys.modules["tgvf_rl.representation.training"] = package
        component_schema = importlib.import_module(
            "tgvf_rl.representation.training.config_schema"
        )

        assert component_schema.RepresentationCodeConfig
        assert "tgvf_rl.representation.training.config" not in sys.modules
        assert "tgvf_rl.representation.training.config_run_schema" not in sys.modules
        assert "tgvf_rl.representation.training.config_parser" not in sys.modules
        assert "tgvf_rl.representation.training.config_binding" not in sys.modules

        run_schema = importlib.import_module(
            "tgvf_rl.representation.training.config_run_schema"
        )
        assert run_schema.RepresentationTrainingConfig
        assert "tgvf_rl.representation.training.config" not in sys.modules
        assert "tgvf_rl.representation.training.config_parser" not in sys.modules
        assert "tgvf_rl.representation.training.config_binding" not in sys.modules

        encoded_type = pickle.dumps(run_schema.RepresentationTrainingConfig)
        facade = importlib.import_module(
            "tgvf_rl.representation.training.config"
        )
        assert (
            facade.RepresentationTrainingConfig
            is run_schema.RepresentationTrainingConfig
        )
        assert pickle.loads(encoded_type) is run_schema.RepresentationTrainingConfig
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
