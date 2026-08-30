from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
import copy
import json
import os
import pickle

import pytest

from tgvf_rl.ops import child_environment as child_environment_module
from tgvf_rl.ops.child_environment import (
    CHILD_ENVIRONMENT_SCHEMA,
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    OPENROUTER_SECRET_ENVIRONMENT_NAME,
    POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
    POLICY_VERL_DRIVER_PROFILE,
    REPRESENTATION_TORCHRUN_PROFILE,
    RUNTIME_PACKAGE_ROOT,
    SUPPORTED_CHILD_ENVIRONMENT_PROFILES,
    TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
    ChildEnvironmentBinding,
    OpenRouterSecretRequirement,
    bind_openrouter_api_key,
    build_child_environment,
    profile_late_overlay_environment_names,
    profile_owned_environment_names,
    scrub_policy_driver_authorization_environment,
    scrub_representation_worker_authorization_environment,
    verify_policy_driver_child_environment,
    verify_representation_torchrun_child_environment,
)


COMMON_BASELINE = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.defpath,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(RUNTIME_PACKAGE_ROOT),
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "TZ": "UTC",
}


def test_supported_profiles_build_from_empty_with_fixed_baseline() -> None:
    representation = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    policy = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={},
    )

    assert SUPPORTED_CHILD_ENVIRONMENT_PROFILES == (
        POLICY_VERL_DRIVER_PROFILE,
        REPRESENTATION_TORCHRUN_PROFILE,
    )
    assert representation.as_environment() == {
        **COMMON_BASELINE,
        "OMP_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    assert policy.as_environment() == {
        **COMMON_BASELINE,
        "RAY_USAGE_STATS_ENABLED": "0",
        "VLLM_NO_USAGE_STATS": "1",
    }


def test_hostile_host_is_audited_by_name_and_never_inherited() -> None:
    hostile = {
        "CUDA_HOME": "/attacker/cuda",
        "HTTPS_PROXY": "https://credential@example.invalid",
        "LD_PRELOAD": "/attacker/inject.so",
        "NCCL_DEBUG": "TRACE",
        OPENROUTER_SECRET_ENVIRONMENT_NAME: "do-not-copy",
        "PET_LOG_DIR": "/attacker/pet",
        "PYTHON_EXECUTABLE": "/attacker/python",
        "RAY_ADDRESS": "ray://attacker",
        "SAFE_UNRELATED": "also-not-copied",
        "TORCHELASTIC_RUN_ID": "host-controlled",
    }
    binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment=hostile,
    )

    assert set(binding.as_environment()).isdisjoint(hostile)
    assert binding.ignored_host_names == ("SAFE_UNRELATED",)
    assert binding.rejected_host_names == tuple(
        sorted(set(hostile).difference({"SAFE_UNRELATED"}))
    )
    rendered = json.dumps(binding.authorization_parameters(), sort_keys=True)
    assert "do-not-copy" not in rendered
    assert "credential@example.invalid" not in rendered


def test_profile_owned_values_are_exact_and_do_not_consult_host() -> None:
    owned = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "4,5",
        "HOME": "/run/safe/home",
        "PYTHONHASHSEED": "42",
        "TOKENIZERS_PARALLELISM": "false",
    }
    binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment=owned,
        host_environment={"CUDA_VISIBLE_DEVICES": "0,1", "HOME": "/host"},
    )

    environment = binding.as_environment()
    assert all(environment[name] == value for name, value in owned.items())
    assert binding.owned_names == tuple(sorted(owned))
    assert binding.rejected_host_names == ("CUDA_VISIBLE_DEVICES", "HOME")


def test_runtime_package_root_is_fixed_and_host_pythonpath_is_never_used() -> None:
    binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={"PYTHONPATH": "/attacker/src"},
    )

    assert binding.as_environment()["PYTHONPATH"] == str(RUNTIME_PACKAGE_ROOT)
    assert binding.rejected_host_names == ("PYTHONPATH",)


@pytest.mark.parametrize(
    ("profile", "owned"),
    [
        (REPRESENTATION_TORCHRUN_PROFILE, {"UNRELATED": "value"}),
        (REPRESENTATION_TORCHRUN_PROFILE, {"OMP_NUM_THREADS": "8"}),
        (POLICY_VERL_DRIVER_PROFILE, {"RAY_ADDRESS": "ray://host"}),
        (POLICY_VERL_DRIVER_PROFILE, {OPENROUTER_SECRET_ENVIRONMENT_NAME: "x"}),
    ],
)
def test_unknown_or_reserved_owned_names_fail_closed(
    profile: str, owned: dict[str, str]
) -> None:
    with pytest.raises(ValueError):
        build_child_environment(
            profile,
            owned_environment=owned,
            host_environment={},
        )


@pytest.mark.parametrize(
    "owned",
    [
        {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
        {"TOKENIZERS_PARALLELISM": "true"},
        {"PYTHONHASHSEED": "-1"},
        {"HOME": "relative/home"},
    ],
)
def test_invalid_representation_owned_values_fail_closed(
    owned: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        build_child_environment(
            REPRESENTATION_TORCHRUN_PROFILE,
            owned_environment=owned,
            host_environment={},
        )


@pytest.mark.parametrize(
    "values",
    [
        {"BAD=NAME": "value"},
        {1: "value"},  # type: ignore[dict-item]
    ],
)
def test_invalid_host_names_fail_closed(values: dict[str, str]) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_child_environment(
            REPRESENTATION_TORCHRUN_PROFILE,
            host_environment=values,
        )


class _NamesOnlyHostEnvironment(Mapping[str, str]):
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, _name: str) -> str:
        raise AssertionError("host environment value was retrieved")

    def items(self) -> object:
        raise AssertionError("host environment items were retrieved")


def test_host_audit_never_retrieves_or_copies_values() -> None:
    host = _NamesOnlyHostEnvironment(
        (OPENROUTER_SECRET_ENVIRONMENT_NAME, "SAFE_UNRELATED")
    )

    binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment=host,
    )

    assert binding.rejected_host_names == (OPENROUTER_SECRET_ENVIRONMENT_NAME,)
    assert binding.ignored_host_names == ("SAFE_UNRELATED",)


def test_late_overlay_accepts_only_delegated_unused_names() -> None:
    binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={"RANK": "99"},
    )
    overlaid = binding.with_late_overlay(
        {
            "LOCAL_RANK": "2",
            "RANK": "2",
            "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": "worker-v1",
            "WORLD_SIZE": "4",
        }
    )

    assert overlaid.as_environment()["RANK"] == "2"
    assert overlaid.as_environment()["LOCAL_RANK"] == "2"
    assert overlaid.late_overlay_names == (
        "LOCAL_RANK",
        "RANK",
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA",
        "WORLD_SIZE",
    )
    assert overlaid.rejected_host_names == ("RANK",)
    assert binding.late_overlay_names == ()
    assert "RANK" not in binding.as_environment()


def test_late_overlay_rejects_unknown_names_and_overwrites() -> None:
    representation = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    with pytest.raises(ValueError, match="outside the profile"):
        representation.with_late_overlay({"RAY_ADDRESS": "ray://host"})

    once = representation.with_late_overlay({"RANK": "0"})
    with pytest.raises(ValueError, match="cannot overwrite"):
        once.with_late_overlay({"RANK": "1"})
    with pytest.raises(ValueError, match="cannot overwrite"):
        once.with_late_overlay({"PATH": "/attacker"})


def test_policy_receipt_and_worker_fields_are_controlled_late_overlays() -> None:
    allowed = profile_late_overlay_environment_names(POLICY_VERL_DRIVER_PROFILE)
    assert "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH" in allowed
    assert "TGVF_CLI_EXECUTION_IDENTITY_JSON" in allowed
    assert "RAY_ADDRESS" not in allowed
    assert "OPENROUTER_API_KEY" not in allowed

    binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={},
    ).with_late_overlay(
        {
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH": (
                "/run/policy/receipt.json"
            ),
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256": "a" * 64,
        }
    )
    assert (
        binding.as_environment()["TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256"]
        == "a" * 64
    )


def test_profile_name_inventories_are_exact_copies() -> None:
    representation_owned = profile_owned_environment_names(
        REPRESENTATION_TORCHRUN_PROFILE
    )
    policy_owned = profile_owned_environment_names(POLICY_VERL_DRIVER_PROFILE)

    assert "CUDA_VISIBLE_DEVICES" in representation_owned
    assert "OMP_NUM_THREADS" not in representation_owned
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in policy_owned
    assert "RAY_ADDRESS" not in policy_owned
    assert OPENROUTER_SECRET_ENVIRONMENT_NAME not in policy_owned
    assert all(
        isinstance(value, tuple) for value in (representation_owned, policy_owned)
    )


def test_binding_is_frozen_and_environment_copies_do_not_mutate_it() -> None:
    binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={},
    )
    copied = binding.as_environment()
    copied["PATH"] = "/attacker"

    assert binding.as_environment()["PATH"] == os.defpath
    with pytest.raises(FrozenInstanceError):
        binding.profile = "changed"  # type: ignore[misc]


def test_hashes_are_stable_across_mapping_order_and_host_value_changes() -> None:
    first = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment={"CUDA_VISIBLE_DEVICES": "4,5", "PYTHONHASHSEED": "7"},
        host_environment={"SAFE_B": "one", "SAFE_A": "two"},
    )
    second = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment={"PYTHONHASHSEED": "7", "CUDA_VISIBLE_DEVICES": "4,5"},
        host_environment={"SAFE_A": "changed", "SAFE_B": "changed"},
    )

    assert first == second
    assert first.environment_sha256 == second.environment_sha256
    assert first.profile_sha256 == second.profile_sha256
    assert first.authorization_parameters() == second.authorization_parameters()
    parameters = first.authorization_parameters()
    assert parameters["child_environment_schema"] == CHILD_ENVIRONMENT_SCHEMA
    assert all(
        len(value) == 64
        for name, value in parameters.items()
        if name.endswith("sha256")
    )


def test_direct_binding_cannot_forge_profile_fields() -> None:
    valid = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    with pytest.raises(ValueError, match="field set differs"):
        ChildEnvironmentBinding(
            profile=valid.profile,
            entries=tuple(
                (name, value) for name, value in valid.entries if name != "PATH"
            ),
            owned_names=(),
            late_overlay_names=(),
            ignored_host_names=(),
            rejected_host_names=(),
        )


def test_openrouter_secret_capability_has_no_repr_auth_or_pickle_leak() -> None:
    requirement = OpenRouterSecretRequirement(role="semantic-judge")
    secret = "sk-or-v1-super-secret-value"
    binding = bind_openrouter_api_key(requirement, secret)

    assert binding.requirement == requirement
    assert secret not in repr(binding)
    assert secret not in json.dumps(binding.authorization_parameters())
    assert binding.authorization_parameters()["secret_requirement_names"] == (
        '["OPENROUTER_API_KEY"]'
    )
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(binding)
    with pytest.raises(TypeError, match="process-local"):
        copy.copy(binding)


def test_openrouter_secret_capability_is_role_bound_and_immutable() -> None:
    requirement = OpenRouterSecretRequirement(role="semantic-judge")
    binding = bind_openrouter_api_key(requirement, "secret")

    read_descriptor, write_descriptor = os.pipe()
    try:
        with pytest.raises(PermissionError, match="requirement differs"):
            binding.consume_into_broker_fd(
                OpenRouterSecretRequirement(role="other-role"), write_descriptor
            )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)
    with pytest.raises(AttributeError):
        binding._secret_bytes = bytearray(b"changed")  # type: ignore[attr-defined]


def test_openrouter_secret_is_single_use_length_framed_and_wiped() -> None:
    requirement = OpenRouterSecretRequirement(role="semantic-judge")
    binding = bind_openrouter_api_key(requirement, "secret")
    read_descriptor, write_descriptor = os.pipe()
    try:
        binding.consume_into_broker_fd(requirement, write_descriptor)
        frame = os.read(read_descriptor, 10)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    assert frame == b"\x00\x00\x00\x06secret"
    assert binding.spent is True
    assert bytes(binding._secret_bytes) == b"\x00" * 6  # noqa: SLF001
    with pytest.raises(RuntimeError, match="already spent"):
        binding.consume_into_broker_fd(requirement, 0)


def test_openrouter_secret_partial_write_and_failure_paths_wipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = OpenRouterSecretRequirement(role="semantic-judge")
    binding = bind_openrouter_api_key(requirement, "secret")
    captured = bytearray()

    def partial_write(_descriptor: int, payload: object) -> int:
        view = memoryview(payload)
        count = min(2, len(view))
        captured.extend(view[:count])
        return count

    monkeypatch.setattr(child_environment_module.os, "write", partial_write)
    binding.consume_into_broker_fd(requirement, 9)

    assert captured == b"\x00\x00\x00\x06secret"
    assert binding.spent is True
    assert bytes(binding._secret_bytes) == b"\x00" * 6  # noqa: SLF001

    refused = bind_openrouter_api_key(requirement, "second-secret")
    monkeypatch.setattr(
        child_environment_module.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic broker refusal")),
    )
    with pytest.raises(OSError, match="synthetic broker refusal"):
        refused.consume_into_broker_fd(requirement, 9)
    assert refused.spent is True
    assert bytes(refused._secret_bytes) == b"\x00" * len(  # noqa: SLF001
        "second-secret"
    )


@pytest.mark.parametrize("value", ["", "line\nbreak", "nul\x00value", 1])
def test_openrouter_secret_rejects_invalid_values(value: str) -> None:
    requirement = OpenRouterSecretRequirement(role="semantic-judge")
    with pytest.raises((TypeError, ValueError)):
        bind_openrouter_api_key(requirement, value)


def test_outer_binding_cannot_accept_openrouter_secret_in_any_stage() -> None:
    with pytest.raises(ValueError):
        build_child_environment(
            POLICY_VERL_DRIVER_PROFILE,
            owned_environment={OPENROUTER_SECRET_ENVIRONMENT_NAME: "secret"},
            host_environment={},
        )
    outer = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={OPENROUTER_SECRET_ENVIRONMENT_NAME: "secret"},
    )
    with pytest.raises(ValueError):
        outer.with_late_overlay({OPENROUTER_SECRET_ENVIRONMENT_NAME: "secret"})
    assert OPENROUTER_SECRET_ENVIRONMENT_NAME not in outer.as_environment()


def _late_values(names: tuple[str, ...]) -> dict[str, str]:
    return {name: f"value-for-{name}" for name in names}


def test_policy_driver_materialized_environment_verifies_exact_base_and_late_set() -> (
    None
):
    binding = build_child_environment(POLICY_VERL_DRIVER_PROFILE, host_environment={})
    late = _late_values(
        (
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
        )
    )
    environment = binding.with_late_overlay(late).as_environment()

    verify_policy_driver_child_environment(
        environment,
        binding.authorization_parameters(),
    )

    with pytest.raises(ValueError, match="entry count differs"):
        verify_policy_driver_child_environment(
            {**environment, "PYTHON_EXEC": "/attacker/python"},
            binding.authorization_parameters(),
        )
    missing = dict(environment)
    del missing[CLI_WORKER_LATE_ENVIRONMENT_NAMES[0]]
    with pytest.raises(ValueError, match="lacks required late names"):
        verify_policy_driver_child_environment(
            missing,
            binding.authorization_parameters(),
        )


def test_representation_worker_materialized_environment_verifies_torchrun_set() -> None:
    binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    late = _late_values(
        (
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
        )
    )
    environment = binding.with_late_overlay(late).as_environment()

    verify_representation_torchrun_child_environment(
        environment,
        binding.authorization_parameters(),
    )

    environment["PATH"] = "/attacker"
    with pytest.raises(ValueError, match="identity differs"):
        verify_representation_torchrun_child_environment(
            environment,
            binding.authorization_parameters(),
        )


def test_verified_worker_proofs_are_scrubbed_without_touching_runtime_fields() -> None:
    policy_binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={},
    )
    policy_late = _late_values(
        (
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
        )
    )
    policy_environment = policy_binding.with_late_overlay(policy_late).as_environment()
    scrub_policy_driver_authorization_environment(policy_environment)
    assert policy_environment == policy_binding.as_environment()

    representation_binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    representation_late = _late_values(
        (
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
        )
    )
    representation_environment = representation_binding.with_late_overlay(
        representation_late
    ).as_environment()
    scrub_representation_worker_authorization_environment(representation_environment)
    assert set(representation_environment) == {
        *representation_binding.as_environment(),
        *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
    }
    assert all(
        representation_environment[name] == representation_late[name]
        for name in TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES
    )


def test_worker_proof_scrub_fails_before_partial_mutation() -> None:
    environment = _late_values(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
    del environment[CLI_WORKER_LATE_ENVIRONMENT_NAMES[-1]]
    before = dict(environment)

    with pytest.raises(RuntimeError, match="lacks fields to scrub"):
        scrub_representation_worker_authorization_environment(environment)
    assert environment == before
