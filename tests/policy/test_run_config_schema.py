from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import FunctionType
from typing import get_type_hints

from tgvf_rl.policy import run_config, run_config_schema


PUBLIC_RUN_CONFIG_MODULE = "tgvf_rl.policy.run_config"
_DATACLASS_PICKLE_HELPERS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
)


def _schema_types() -> tuple[type[object], ...]:
    return tuple(
        value
        for name in run_config_schema.__all__
        if isinstance((value := getattr(run_config_schema, name)), type)
    )


def test_run_config_facade_reexports_exact_schema_objects() -> None:
    for name in run_config_schema.__all__:
        assert getattr(run_config, name) is getattr(run_config_schema, name)


def test_extracted_schema_types_keep_public_and_pickle_identity() -> None:
    for schema_type in _schema_types():
        assert schema_type.__module__ == PUBLIC_RUN_CONFIG_MODULE
        assert pickle.loads(pickle.dumps(schema_type)) is schema_type
        for member in vars(schema_type).values():
            functions = (
                (member.fget, member.fset, member.fdel)
                if isinstance(member, property)
                else (member,)
            )
            for function in functions:
                if isinstance(function, FunctionType):
                    expected_module = (
                        dataclasses.__name__
                        if function in _DATACLASS_PICKLE_HELPERS
                        else PUBLIC_RUN_CONFIG_MODULE
                    )
                    assert function.__module__ == expected_module


def test_schema_compatibility_does_not_mutate_dataclass_pickle_helpers() -> None:
    for helper in _DATACLASS_PICKLE_HELPERS:
        assert helper.__module__ == dataclasses.__name__

    for schema_type in _schema_types():
        for name in ("__getstate__", "__setstate__"):
            helper = vars(schema_type).get(name)
            if isinstance(helper, FunctionType):
                assert helper.__module__ == dataclasses.__name__


def test_extracted_schema_type_hints_resolve_through_both_facades() -> None:
    for schema_type in _schema_types():
        facade_type = getattr(run_config, schema_type.__name__)
        assert get_type_hints(schema_type)
        assert get_type_hints(facade_type) == get_type_hints(schema_type)


def test_extracted_schema_instance_pickle_round_trip() -> None:
    binding = run_config_schema.SmokeCapacityBinding(
        max_prompt_length=1_024,
        actor_ppo_max_token_len_per_gpu=2_048,
        rollout_log_prob_max_token_len_per_gpu=2_048,
        reference_log_prob_max_token_len_per_gpu=2_048,
        vllm_gpu_memory_utilization=0.5,
        vllm_max_num_batched_tokens=4_096,
        vllm_max_model_len=8_192,
        vllm_max_num_seqs=8,
        vllm_enable_chunked_prefill=True,
        vllm_enforce_eager=False,
    )

    restored = pickle.loads(pickle.dumps(binding))

    assert restored == binding
    assert type(restored) is run_config.SmokeCapacityBinding
    assert restored.response_transport_length == 7_168


def test_schema_leaf_does_not_import_parser_or_judges() -> None:
    # Bypass the package's existing eager facade so this checks the extracted
    # leaf's dependency boundary rather than unrelated policy-package imports.
    repository_root = Path(__file__).resolve().parents[2]
    policy_directory = repository_root / "src" / "tgvf_rl" / "policy"
    script = textwrap.dedent(
        """
        import importlib
        import sys
        import types

        import tgvf_rl

        package = types.ModuleType("tgvf_rl.policy")
        package.__package__ = "tgvf_rl.policy"
        package.__path__ = [sys.argv[1]]
        sys.modules["tgvf_rl.policy"] = package
        importlib.import_module("tgvf_rl.policy.run_config_schema")

        assert "tgvf_rl.policy.run_config" not in sys.modules
        assert "tomllib" not in sys.modules
        assert not any(name.startswith("tgvf_rl.judges") for name in sys.modules)
        """
    )
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )

    subprocess.run(
        [sys.executable, "-c", script, str(policy_directory)],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )
