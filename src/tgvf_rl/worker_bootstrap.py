"""Dependency-light, non-dispatch inspection for a future worker bootstrap.

This module is intentionally executable but can never dispatch a target.  It
checks the interpreter/import firebreak before importing project verification
leaves, re-verifies inherited outer CLI evidence, reconstructs the complete
startup envelope, rebinds the current Python image, checks the role environment
base identity and late-field inventory, and compares one hard-coded
role/target/argv contract.  Late-field value semantics remain an explicit
blocker.  A successful inspection remains ordinary diagnostic data and exits
non-zero.

``python -m tgvf_rl.worker_bootstrap`` is not a trusted root: Python executes
the mutable :mod:`tgvf_rl` package before this module and ``-P`` still admits
an explicit ``PYTHONPATH``.  The runtime locator is non-atomic, worker-local
runtime re-verification is absent, and the existing process-local evidence
types are not hostile-same-process authority.  No code in this module imports
a training target or mints ``VerifiedWorkerStartup``.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import sys
import threading


WORKER_BOOTSTRAP_INSPECTION_SCHEMA = "tgvf-worker-bootstrap-inspection-v2"
WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE = "inspection-only"
WORKER_BOOTSTRAP_MODULE = "tgvf_rl.worker_bootstrap"
WORKER_BOOTSTRAP_MODES = (
    "run-policy",
    "run-representation-member",
)
WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE = 2

_POLICY_PHASE = "policy_training"
_POLICY_COMMAND_ID = "tgvf-rl:run-policy:v4"
_REPRESENTATION_PHASE = "representation_training"
_REPRESENTATION_COMMAND_ID = "tgvf-rl:launch-representation:v2"
_POLICY_DRIVER_ROLE = "policy-driver"
_REPRESENTATION_LAUNCHER_ROLE = "representation-launcher"
_REPRESENTATION_MEMBER_ROLE = "representation-member"
_POLICY_TARGET = "tgvf_rl.framework.verl.policy_main:main"
_REPRESENTATION_MEMBER_TARGET = (
    "tgvf_rl.representation.training.runner:run_representation_training"
)
_REQUIRED_INTERPRETER_ARGUMENTS = ("-B", "-P", "-S", "-m", WORKER_BOOTSTRAP_MODULE)
_CLI_WORKER_ENVIRONMENT_NAMES = frozenset(
    {
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH",
        "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256",
        "TGVF_CLI_EXECUTION_IDENTITY_JSON",
        "TGVF_CLI_GATE_DIRECTORY",
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH",
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA",
    }
)
_CHILD_ENVIRONMENT_PARAMETER_NAMES = frozenset(
    {
        "child_environment_entry_count",
        "child_environment_ignored_host_name_count",
        "child_environment_ignored_host_names_sha256",
        "child_environment_late_overlay_name_count",
        "child_environment_late_overlay_names_sha256",
        "child_environment_owned_name_count",
        "child_environment_owned_names_sha256",
        "child_environment_profile",
        "child_environment_profile_sha256",
        "child_environment_rejected_host_name_count",
        "child_environment_rejected_host_names_sha256",
        "child_environment_schema",
        "child_environment_sha256",
    }
)
_FORBIDDEN_PRECHECK_MODULE_ROOTS = (
    "hydra",
    "numpy",
    "omegaconf",
    "ray",
    "site",
    "sitecustomize",
    "sysconfig",
    "torch",
    "transformers",
    "usercustomize",
    "verl",
    "vllm",
)
_PROTECTED_PROJECT_MODULE_ROOTS = (
    "tgvf_rl.framework",
    "tgvf_rl.ops",
    "tgvf_rl.policy",
    "tgvf_rl.representation",
    "tgvf_rl.rewards",
    "tgvf_rl.secure_file_read",
)
_REQUIRED_DELAYED_PROJECT_MODULES = frozenset(
    {
        "tgvf_rl",
        "tgvf_rl.ops",
        "tgvf_rl.ops.child_environment",
        "tgvf_rl.ops.cli_authorization",
        "tgvf_rl.ops.cli_authorization_identity",
        "tgvf_rl.ops.launch_gate",
        "tgvf_rl.ops.worker_startup",
        "tgvf_rl.secure_file_read",
    }
)
_OPTIONAL_BOOTSTRAP_MODULES = frozenset({WORKER_BOOTSTRAP_MODULE})
_COMMON_BLOCKERS = (
    "bootstrap-package-origin-unverified",
    "canonical-worker-bootstrap-routing-missing",
    "hostile-same-process-import-machinery-unclosed",
    "immutable-runtime-code-package-missing",
    "project-verifier-module-origin-unverified",
    "role-specific-child-environment-late-value-validation-missing",
    "runtime-locator-worker-reverification-missing",
    "verified-worker-evidence-fork-rebinding-unclosed",
)


class WorkerBootstrapInspectionError(RuntimeError):
    """The bounded inspection refused before target import or dispatch."""


@dataclass(frozen=True, slots=True)
class _ModeContract:
    phase: str
    command_id: str
    entry_role: str
    selected_role: str
    target: str
    blockers: tuple[str, ...]


_MODE_CONTRACTS = {
    "run-policy": _ModeContract(
        phase=_POLICY_PHASE,
        command_id=_POLICY_COMMAND_ID,
        entry_role=_POLICY_DRIVER_ROLE,
        selected_role=_POLICY_DRIVER_ROLE,
        target=_POLICY_TARGET,
        blockers=(
            "outer-exec-process-exact-identity-check-missing",
            *_COMMON_BLOCKERS,
        ),
    ),
    "run-representation-member": _ModeContract(
        phase=_REPRESENTATION_PHASE,
        command_id=_REPRESENTATION_COMMAND_ID,
        entry_role=_REPRESENTATION_LAUNCHER_ROLE,
        selected_role=_REPRESENTATION_MEMBER_ROLE,
        target=_REPRESENTATION_MEMBER_TARGET,
        blockers=(
            "representation-member-consumption-not-performed-by-bootstrap",
            "representation-runtime-locator-authority-missing",
            *_COMMON_BLOCKERS,
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class WorkerBootstrapInspection:
    """Non-authoritative record of one dependency-light startup inspection."""

    mode: str
    entry_role: str
    selected_role: str
    command_sha256: str
    target_arguments_sha256: str
    target: str
    startup_envelope_sha256: str
    startup_identity_sha256: str
    declared_runtime_package_sha256: str
    declared_dependency_roots_sha256: str
    blockers: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return an explicit refusal record with no transferable authority."""

        return {
            "schema_version": WORKER_BOOTSTRAP_INSPECTION_SCHEMA,
            "authorization_scope": WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE,
            "record_trust": "ordinary-caller-constructible-diagnostic",
            "mode": self.mode,
            "entry_role": self.entry_role,
            "selected_role": self.selected_role,
            "command_sha256": self.command_sha256,
            "target_arguments_sha256": self.target_arguments_sha256,
            "target": self.target,
            "startup_envelope_sha256": self.startup_envelope_sha256,
            "startup_identity_sha256": self.startup_identity_sha256,
            "declared_runtime_package_sha256": (
                self.declared_runtime_package_sha256
            ),
            "declared_dependency_roots_sha256": (
                self.declared_dependency_roots_sha256
            ),
            "outer_cli_receipt_checked_by_existing_verifier": True,
            "outer_process_relation": "existing-descendant-check-only",
            "cli_environment_namespace_exact": True,
            "current_python_executable_identity_checked": True,
            "current_python_descriptor_retained": False,
            "role_child_environment_base_identity_checked": True,
            "role_child_environment_late_field_inventory_checked": True,
            "role_child_environment_late_values_checked": False,
            "heavy_import_roots_absent": True,
            "interpreter_flags_accepted": True,
            "default_import_machinery_shape_checked": True,
            "single_threaded": True,
            "trace_profile_absent": True,
            "runtime_origin_verified": False,
            "immutable_runtime_verified": False,
            "target_imported": False,
            "verified_worker_startup_minted": False,
            "dispatch_authorized": False,
            "blockers": list(self.blockers),
        }


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _exact_text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise WorkerBootstrapInspectionError(f"{label} must be an exact tuple")
    for index, item in enumerate(value):
        if type(item) is not str or not item:
            raise WorkerBootstrapInspectionError(
                f"{label}[{index}] must be non-empty exact text"
            )
        if any(character in item for character in ("\x00", "\r", "\n")):
            raise WorkerBootstrapInspectionError(
                f"{label}[{index}] contains a forbidden character"
            )
        try:
            item.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise WorkerBootstrapInspectionError(
                f"{label}[{index}] must be valid UTF-8 text"
            ) from error
    return value


def _snapshot_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    if not isinstance(source, Mapping):
        raise TypeError("worker bootstrap environment must be a mapping")
    snapshot = dict(source)
    if any(type(name) is not str for name in snapshot):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap environment names must be exact text"
        )
    if any(type(value) is not str for value in snapshot.values()):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap environment values must be exact text"
        )
    return snapshot


def _require_exact_cli_environment_namespace(environment: dict[str, str]) -> None:
    observed = {
        name for name in environment if name.startswith("TGVF_CLI_")
    }
    if observed != _CLI_WORKER_ENVIRONMENT_NAMES:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap CLI environment namespace differs"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerBootstrapInspectionError(
                    "worker bootstrap execution identity repeats a JSON field"
                )
            result[key] = value
        return result

    raw_identity = environment["TGVF_CLI_EXECUTION_IDENTITY_JSON"]
    try:
        identity_record = json.loads(
            raw_identity,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                WorkerBootstrapInspectionError(
                    "worker bootstrap execution identity contains a non-finite value"
                )
            ),
        )
        canonical_identity = json.dumps(
            identity_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical_identity.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError, json.JSONDecodeError) as error:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap execution identity JSON is not canonical"
        ) from error
    if raw_identity != canonical_identity:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap execution identity JSON spelling differs"
        )


def _require_interpreter_firebreak(
    *,
    allow_project_verifiers: bool = False,
) -> None:
    flags = sys.flags
    if not (
        flags.dont_write_bytecode
        and flags.safe_path
        and flags.no_site
        and sys.dont_write_bytecode
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap requires exact -B, -P, and -S interpreter flags"
        )
    try:
        native_thread_ids = set(os.listdir("/proc/self/task"))
    except OSError as error:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap cannot inspect the native thread inventory"
        ) from error
    expected_native_thread_id = str(os.getpid())
    if (
        threading.active_count() != 1
        or threading.get_native_id() != os.getpid()
        or native_thread_ids != {expected_native_thread_id}
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap requires one exact native process thread"
        )
    if sys.gettrace() is not None or sys.getprofile() is not None:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap refuses active trace or profile hooks"
        )
    _require_default_import_machinery()
    loaded = tuple(sys.modules)
    for root in _FORBIDDEN_PRECHECK_MODULE_ROOTS:
        if any(name == root or name.startswith(root + ".") for name in loaded):
            raise WorkerBootstrapInspectionError(
                "worker bootstrap heavy-import firebreak was crossed"
            )
    if allow_project_verifiers:
        observed_project_modules = {
            name
            for name in loaded
            if name == "tgvf_rl" or name.startswith("tgvf_rl.")
        }
        if (
            not _REQUIRED_DELAYED_PROJECT_MODULES
            <= observed_project_modules
            or observed_project_modules
            - _REQUIRED_DELAYED_PROJECT_MODULES
            - _OPTIONAL_BOOTSTRAP_MODULES
        ):
            raise WorkerBootstrapInspectionError(
                "worker bootstrap delayed project import closure differs"
            )
    else:
        for root in _PROTECTED_PROJECT_MODULE_ROOTS:
            if any(name == root or name.startswith(root + ".") for name in loaded):
                raise WorkerBootstrapInspectionError(
                    "worker bootstrap project verifier was preloaded"
                )


def _require_default_import_machinery() -> None:
    frozen = sys.modules.get("_frozen_importlib")
    external = sys.modules.get("_frozen_importlib_external")
    zip_module = sys.modules.get("zipimport")
    if frozen is None or external is None or zip_module is None:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap frozen import machinery is unavailable"
        )
    expected_meta_path = (
        getattr(frozen, "BuiltinImporter", None),
        getattr(frozen, "FrozenImporter", None),
        getattr(external, "PathFinder", None),
    )
    if tuple(sys.meta_path) != expected_meta_path or any(
        item is None for item in expected_meta_path
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap meta path differs from the frozen default"
        )
    if (
        type(builtins.__import__) is not type(len)
        or builtins.__import__.__module__ != "builtins"
        or builtins.__import__.__name__ != "__import__"
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap import function differs from the builtin"
        )
    hooks = tuple(sys.path_hooks)
    if len(hooks) != 2 or hooks[0] is not getattr(zip_module, "zipimporter", None):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap path hooks differ from the frozen default"
        )
    file_hook = hooks[1]
    closure = getattr(file_hook, "__closure__", None)
    if (
        type(file_hook).__name__ != "function"
        or getattr(file_hook, "__module__", None) != "_frozen_importlib_external"
        or getattr(file_hook, "__qualname__", None)
        != "FileFinder.path_hook.<locals>.path_hook_for_FileFinder"
        or type(closure) is not tuple
        or len(closure) != 2
        or closure[0].cell_contents is not getattr(external, "FileFinder", None)
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap file path hook differs from the frozen default"
        )
    loader_details = closure[1].cell_contents
    expected_loader_types = (
        getattr(external, "ExtensionFileLoader", None),
        getattr(external, "SourceFileLoader", None),
        getattr(external, "SourcelessFileLoader", None),
    )
    if (
        type(loader_details) is not tuple
        or len(loader_details) != len(expected_loader_types)
        or tuple(item[0] for item in loader_details) != expected_loader_types
        or any(item is None for item in expected_loader_types)
    ):
        raise WorkerBootstrapInspectionError(
            "worker bootstrap file loaders differ from the frozen default"
        )
    file_finder_type = getattr(external, "FileFinder", None)
    expected_finder_loaders = tuple(
        (suffix, loader_type)
        for loader_type, suffixes in loader_details
        for suffix in suffixes
    )
    for path, finder in sys.path_importer_cache.items():
        if type(path) is not str:
            raise WorkerBootstrapInspectionError(
                "worker bootstrap importer cache path differs"
            )
        if finder is None:
            continue
        if (
            type(finder) is not file_finder_type
            or getattr(finder, "path", None) != path
            or tuple(getattr(finder, "_loaders", ())) != expected_finder_loaders
        ):
            raise WorkerBootstrapInspectionError(
                "worker bootstrap importer cache contains a changed file finder"
            )


def _mode_contract(mode: object) -> _ModeContract:
    if type(mode) is not str or mode not in _MODE_CONTRACTS:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap mode must be one exact fixed mode"
        )
    return _MODE_CONTRACTS[mode]


def _require_process_command(
    process_command: object,
    *,
    mode: str,
) -> tuple[str, ...]:
    command = _exact_text_tuple(process_command, label="worker bootstrap command")
    if len(command) < 7:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap command is shorter than its fixed prefix"
        )
    if command[0] != sys.executable:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap command executable differs from current Python"
        )
    if command[1:6] != _REQUIRED_INTERPRETER_ARGUMENTS:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap command interpreter prefix differs"
        )
    if command[6] != mode:
        raise WorkerBootstrapInspectionError("worker bootstrap command mode differs")
    return command


def inspect_inherited_worker_startup(
    mode: str,
    *,
    process_command: tuple[str, ...] | None = None,
    environment: Mapping[str, str] | None = None,
) -> WorkerBootstrapInspection:
    """Inspect inherited worker evidence without importing or dispatching a target.

    The current CLI verifier provides a cooperative descendant check.  This
    function records that exact limitation and deliberately does not upgrade
    the result to startup or dispatch authority.
    """

    _require_interpreter_firebreak()
    environment_snapshot = _snapshot_environment(environment)
    _require_exact_cli_environment_namespace(environment_snapshot)
    _require_interpreter_firebreak()
    contract = _mode_contract(mode)
    observed_command = _require_process_command(
        tuple(sys.orig_argv) if process_command is None else process_command,
        mode=mode,
    )

    # These project imports are deliberately delayed until after the firebreak.
    from tgvf_rl.ops.cli_authorization import (  # noqa: PLC0415
        bind_current_python_executable_for_exec,
        verify_cli_worker_authorization_from_environment,
    )
    from tgvf_rl.ops.child_environment import (  # noqa: PLC0415
        verify_policy_driver_child_environment,
        verify_representation_torchrun_child_environment,
    )
    from tgvf_rl.ops.worker_startup import (  # noqa: PLC0415
        WorkerStartupEnvelope,
    )

    _require_interpreter_firebreak(allow_project_verifiers=True)
    launch_identity = verify_cli_worker_authorization_from_environment(
        environment_snapshot,
        expected_phase=contract.phase,
        expected_command_id=contract.command_id,
    )
    parameters = dict(launch_identity.parameters)
    observed_python_names = {
        name for name in parameters if name.startswith("python_executable")
    }
    with bind_current_python_executable_for_exec(sys.executable) as python_binding:
        expected_python_parameters = (
            python_binding.identity.authorization_parameters()
        )
        if observed_python_names != set(expected_python_parameters):
            raise WorkerBootstrapInspectionError(
                "worker bootstrap Python identity namespace differs"
            )
        for name, expected in expected_python_parameters.items():
            if parameters.get(name) != expected:
                raise WorkerBootstrapInspectionError(
                    f"worker bootstrap current Python identity differs: {name}"
                )
    observed_child_environment_names = {
        name for name in parameters if name.startswith("child_environment_")
    }
    if observed_child_environment_names != _CHILD_ENVIRONMENT_PARAMETER_NAMES:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap child environment parameter namespace differs"
        )
    try:
        if mode == "run-policy":
            verify_policy_driver_child_environment(
                environment_snapshot,
                parameters,
            )
        else:
            verify_representation_torchrun_child_environment(
                environment_snapshot,
                parameters,
            )
    except (TypeError, ValueError) as error:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap role child environment differs"
        ) from error
    try:
        envelope = WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=contract.entry_role,
        )
        selected_identity = envelope.identity_for_role(contract.selected_role)
    except (TypeError, ValueError, PermissionError) as error:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap startup envelope differs"
        ) from error
    if selected_identity.target != contract.target:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap selected target differs from fixed role target"
        )
    if selected_identity.command != observed_command:
        raise WorkerBootstrapInspectionError(
            "worker bootstrap process command differs from startup identity"
        )
    _require_interpreter_firebreak(allow_project_verifiers=True)

    return WorkerBootstrapInspection(
        mode=mode,
        entry_role=contract.entry_role,
        selected_role=contract.selected_role,
        command_sha256=_canonical_json_sha256(list(observed_command)),
        target_arguments_sha256=_canonical_json_sha256(list(observed_command[7:])),
        target=contract.target,
        startup_envelope_sha256=envelope.envelope_sha256,
        startup_identity_sha256=selected_identity.identity_sha256,
        declared_runtime_package_sha256=(
            selected_identity.runtime_package_sha256
        ),
        declared_dependency_roots_sha256=(
            selected_identity.dependency_roots_sha256
        ),
        blockers=contract.blockers,
    )


def main() -> int:
    """Inspect one inherited worker request and always refuse target dispatch."""

    try:
        _require_interpreter_firebreak()
        arguments = _exact_text_tuple(
            tuple(sys.argv[1:]),
            label="worker bootstrap arguments",
        )
        if not arguments:
            raise WorkerBootstrapInspectionError(
                "worker bootstrap requires one fixed mode"
            )
        if tuple(sys.orig_argv[6:]) != arguments:
            raise WorkerBootstrapInspectionError(
                "worker bootstrap argv differs from original process command"
            )
        inspection = inspect_inherited_worker_startup(arguments[0])
    except (OSError, TypeError, ValueError, RuntimeError):
        print(
            "worker bootstrap refusal: inspection did not establish dispatch authority",
            file=sys.stderr,
        )
        return WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    print(
        json.dumps(
            inspection.as_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    print(
        "worker bootstrap refusal: inspection-only scaffold cannot dispatch",
        file=sys.stderr,
    )
    return WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE",
    "WORKER_BOOTSTRAP_INSPECTION_SCHEMA",
    "WORKER_BOOTSTRAP_MODES",
    "WORKER_BOOTSTRAP_MODULE",
    "WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE",
    "WorkerBootstrapInspection",
    "WorkerBootstrapInspectionError",
    "inspect_inherited_worker_startup",
    "main",
]
