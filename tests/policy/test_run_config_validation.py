from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import get_type_hints

from tgvf_rl.artifact_contracts import canonical_json_bytes
from tgvf_rl.policy import run_config, run_config_validation


_EXTRACTED_VALIDATION_HELPERS = (
    "_absolute_path",
    "_boolean",
    "_checkpoint_steps",
    "_conditioning",
    "_distributed",
    "_exact_real",
    "_existing_directory",
    "_existing_file",
    "_fqn",
    "_integer",
    "_logprob_measurement",
    "_nonnegative_int",
    "_nonnegative_int_tuple",
    "_nonnegative_real",
    "_normalize_json",
    "_optional_absolute_path",
    "_positive_int",
    "_positive_real",
    "_real",
    "_require_exact",
    "_require_within",
    "_safe_project_name",
    "_safe_run_id",
    "_sha256",
    "_sha256_file",
    "_table",
    "_text",
    "_text_tuple",
    "_unit_interval",
    "_validate_deepeyes_strict_judge",
)


def test_run_config_facade_reexports_exact_validation_helpers() -> None:
    for name in _EXTRACTED_VALIDATION_HELPERS:
        assert getattr(run_config, name) is getattr(run_config_validation, name)
    assert run_config._canonical_json_bytes is canonical_json_bytes  # noqa: SLF001


def test_extracted_validation_type_hints_resolve_from_the_facade() -> None:
    for name in ("_conditioning", "_distributed", "_logprob_measurement"):
        helper = getattr(run_config_validation, name)
        assert get_type_hints(helper)
        assert get_type_hints(getattr(run_config, name)) == get_type_hints(helper)


def test_absolute_path_expands_only_the_repository_root_token() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert run_config_validation._absolute_path(  # noqa: SLF001
        "${TGVF_REPOSITORY_ROOT}/configs/policy",
        name="config path",
    ) == (repository_root / "configs" / "policy")


def test_absolute_path_does_not_expand_arbitrary_environment_syntax() -> None:
    for value in (
        "${HOME}/config.toml",
        "$TGVF_REPOSITORY_ROOT/config.toml",
        "${TGVF_REPOSITORY_ROOT_SUFFIX}/config.toml",
    ):
        try:
            run_config_validation._absolute_path(value, name="config path")  # noqa: SLF001
        except ValueError as error:
            assert "absolute normalized path" in str(error)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"unexpected path-token expansion for {value!r}")


def test_validation_leaf_does_not_import_facade_judges_or_tomllib() -> None:
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
        importlib.import_module("tgvf_rl.policy.run_config_validation")

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
