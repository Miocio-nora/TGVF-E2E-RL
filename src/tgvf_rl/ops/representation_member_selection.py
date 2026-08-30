"""Authorization-only selection of one Representation torchrun member.

This dependency-light leaf reconstructs the *complete*
:class:`~tgvf_rl.ops.representation_startup.RepresentationStartupPlan` from a
caller-supplied outer CLI identity and matches one rank to that plan using an
exact copied worker environment.  This leaf does not establish that the caller
consumed the identity's authorization.  The result is ordinary
selection/authorization data.  It is deliberately replayable, performs no
filesystem consumption, verifies no runtime origin, imports no training
target, and must never be treated as ``VerifiedWorkerStartup`` or other
execution evidence.

An individual member claim is not accepted as input authority.  A future
bootstrap must combine this selection with a one-use receipt and immutable
runtime verification before dispatching any target.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePath
from uuid import UUID

from tgvf_rl.ops.cli_authorization_identity import (
    CLIExecutionAuthorizationIdentity,
)
from tgvf_rl.ops.child_environment import (
    verify_representation_torchrun_child_environment,
)
from tgvf_rl.ops.representation_startup import (
    RepresentationMemberClaim,
    RepresentationStartupPlan,
)


REPRESENTATION_MEMBER_SELECTION_SCHEMA = "tgvf-representation-member-selection-v1"
REPRESENTATION_TRAINING_PHASE = "representation_training"
REPRESENTATION_TRAINING_COMMAND_ID = "tgvf-rl:launch-representation:v2"

_SELECTION_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "ROLE_NAME",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "WORLD_SIZE",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_exact_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"representation member selection {name} must be exactly str")
    if (
        not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(
            f"representation member selection {name} must be non-empty canonical text"
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"representation member selection {name} must be valid UTF-8 text"
        ) from error
    return value


def _require_ascii_decimal(value: object, *, name: str) -> int:
    text = _require_exact_text(value, name=name)
    if (
        not text.isascii()
        or not text.isdecimal()
        or (len(text) > 1 and text.startswith("0"))
    ):
        raise ValueError(
            f"representation member selection {name} must be a canonical "
            "non-negative ASCII decimal"
        )
    return int(text)


def _require_parameter(
    parameters: dict[str, str],
    name: str,
) -> str:
    if name not in parameters:
        raise ValueError(
            f"representation member selection CLI parameter is missing: {name}"
        )
    value = parameters[name]
    if type(value) is not str:
        raise TypeError(
            f"representation member selection CLI parameter {name} must be exactly str"
        )
    return value


def _require_environment(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise TypeError(
            "representation member selection environment must be an exact copied dict"
        )
    if any(type(name) is not str for name in value):
        raise TypeError(
            "representation member selection environment keys must be exactly str"
        )
    if any(type(item) is not str for item in value.values()):
        raise TypeError(
            "representation member selection environment values must be exactly str"
        )
    missing = sorted(_SELECTION_ENVIRONMENT_NAMES.difference(value))
    if missing:
        raise ValueError(
            "representation member selection environment is missing fields: "
            f"{missing!r}"
        )
    return value


def _validate_dynamic_torchrun_fields(environment: dict[str, str]) -> None:
    run_id = _require_exact_text(
        environment["TORCHELASTIC_RUN_ID"],
        name="TORCHELASTIC_RUN_ID",
    )
    try:
        parsed_run_id = UUID(run_id)
    except ValueError as error:
        raise ValueError(
            "representation member selection TORCHELASTIC_RUN_ID must be a "
            "canonical lowercase UUID4"
        ) from error
    if parsed_run_id.version != 4 or str(parsed_run_id) != run_id:
        raise ValueError(
            "representation member selection TORCHELASTIC_RUN_ID must be a "
            "canonical lowercase UUID4"
        )
    _require_exact_text(environment["MASTER_ADDR"], name="MASTER_ADDR")
    master_port = _require_ascii_decimal(
        environment["MASTER_PORT"],
        name="MASTER_PORT",
    )
    if not 0 < master_port <= 65_535:
        raise ValueError(
            "representation member selection MASTER_PORT must be in [1, 65535]"
        )
    if environment["TORCHELASTIC_USE_AGENT_STORE"] != "True":
        raise ValueError(
            "representation member selection TORCHELASTIC_USE_AGENT_STORE "
            "must be exactly True"
        )
    error_file = _require_exact_text(
        environment["TORCHELASTIC_ERROR_FILE"],
        name="TORCHELASTIC_ERROR_FILE",
    )
    error_path = PurePath(error_file)
    if (
        not error_path.is_absolute()
        or error_path == PurePath("/")
        or ".." in error_path.parts
        or str(error_path) != error_file
    ):
        raise ValueError(
            "representation member selection TORCHELASTIC_ERROR_FILE must be "
            "canonical lexical absolute"
        )


def _select_claim_for_environment(
    plan: RepresentationStartupPlan,
    environment: dict[str, str],
) -> RepresentationMemberClaim:
    expected_world_size = plan.world_size
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE"):
        observed = _require_ascii_decimal(environment[name], name=name)
        if observed != expected_world_size:
            raise ValueError(
                f"representation member selection {name} differs from startup plan"
            )

    rank = _require_ascii_decimal(environment["RANK"], name="RANK")
    if not 0 <= rank < expected_world_size:
        raise ValueError(
            "representation member selection RANK is outside the startup plan"
        )
    for name in ("LOCAL_RANK", "ROLE_RANK"):
        observed = _require_ascii_decimal(environment[name], name=name)
        if observed != rank:
            raise ValueError(
                f"representation member selection {name} differs from RANK"
            )

    fixed_numeric = {
        "GROUP_RANK": 0,
        "GROUP_WORLD_SIZE": 1,
        "TORCHELASTIC_MAX_RESTARTS": 0,
        "TORCHELASTIC_RESTART_COUNT": 0,
    }
    for name, expected in fixed_numeric.items():
        observed = _require_ascii_decimal(environment[name], name=name)
        if observed != expected:
            raise ValueError(
                f"representation member selection {name} must be exactly {expected}"
            )
    if environment["ROLE_NAME"] != "default":
        raise ValueError(
            "representation member selection ROLE_NAME must be exactly 'default'"
        )

    expected_visible_devices = ",".join(
        str(member.physical_gpu_id) for member in plan.members
    )
    if environment["CUDA_VISIBLE_DEVICES"] != expected_visible_devices:
        raise ValueError(
            "representation member selection CUDA_VISIBLE_DEVICES differs from "
            "startup plan"
        )
    _validate_dynamic_torchrun_fields(environment)

    claim = plan.members[rank]
    if (
        claim.global_rank != rank
        or claim.local_rank != rank
        or claim.world_size != expected_world_size
        or claim.physical_gpu_id
        != tuple(member.physical_gpu_id for member in plan.members)[rank]
    ):  # pragma: no cover - plan construction already guarantees this invariant
        raise RuntimeError("representation startup plan selected an inconsistent claim")
    return claim


def _validate_cli_identity(
    identity: object,
) -> tuple[CLIExecutionAuthorizationIdentity, RepresentationStartupPlan]:
    if type(identity) is not CLIExecutionAuthorizationIdentity:
        raise TypeError(
            "representation member selection identity must be exactly "
            "CLIExecutionAuthorizationIdentity"
        )
    _require_exact_text(identity.run_id, name="CLI run_id")
    if (
        type(identity.phase) is not str
        or identity.phase != REPRESENTATION_TRAINING_PHASE
    ):
        raise ValueError("representation member selection CLI phase differs")
    if (
        type(identity.command_id) is not str
        or identity.command_id != REPRESENTATION_TRAINING_COMMAND_ID
    ):
        raise ValueError("representation member selection CLI command differs")
    if type(identity.run_identity_sha256) is not str:
        raise TypeError(
            "representation member selection CLI run identity must be exactly str"
        )
    if type(identity.parameters) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in identity.parameters
    ):
        raise TypeError(
            "representation member selection CLI parameters must be an exact "
            "tuple of exact string pairs"
        )

    parameters = dict(identity.parameters)
    plan = RepresentationStartupPlan.from_cli_authorization_parameters(parameters)
    if plan.run_identity_sha256 != identity.run_identity_sha256:
        raise ValueError(
            "representation startup plan run identity differs from CLI identity"
        )

    config_source_sha256 = _require_parameter(parameters, "config_source_sha256")
    canonical_config_sha256 = _require_parameter(
        parameters,
        "canonical_config_sha256",
    )
    if not (
        plan.config_identity_sha256 == config_source_sha256 == canonical_config_sha256
    ):
        raise ValueError(
            "representation startup plan config identity differs from bound TOML source"
        )

    expected_world_size = str(plan.world_size)
    for name in ("nproc_per_node", "world_size"):
        value = _require_parameter(parameters, name)
        if value != expected_world_size:
            raise ValueError(
                f"representation startup plan world size differs from CLI {name}"
            )
    return identity, plan


def _validate_full_environment(
    identity: CLIExecutionAuthorizationIdentity,
    plan: RepresentationStartupPlan,
    environment: dict[str, str],
) -> RepresentationMemberClaim:
    parameters = dict(identity.parameters)
    verify_representation_torchrun_child_environment(environment, parameters)
    expected_identity_json = _canonical_json(identity.as_record())
    if environment["TGVF_CLI_EXECUTION_IDENTITY_JSON"] != expected_identity_json:
        raise ValueError(
            "representation member selection CLI environment identity JSON differs"
        )
    return _select_claim_for_environment(plan, environment)


@dataclass(frozen=True, slots=True)
class RepresentationMemberSelection:
    """Replayable authorization data matching one rank to one complete plan."""

    identity: CLIExecutionAuthorizationIdentity
    plan: RepresentationStartupPlan
    claim: RepresentationMemberClaim
    full_environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        identity, expected_plan = _validate_cli_identity(self.identity)
        if type(self.plan) is not RepresentationStartupPlan:
            raise TypeError("representation member selection plan type differs")
        if self.plan != expected_plan:
            raise ValueError(
                "representation member selection plan differs from complete CLI identity"
            )
        if type(self.claim) is not RepresentationMemberClaim:
            raise TypeError("representation member selection claim type differs")
        if type(self.full_environment) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in self.full_environment
        ):
            raise TypeError(
                "representation member selection full environment must be an "
                "exact tuple of exact string pairs"
            )
        if self.full_environment != tuple(sorted(self.full_environment)):
            raise ValueError(
                "representation member selection full environment must be sorted"
            )
        environment = dict(self.full_environment)
        if len(environment) != len(self.full_environment):
            raise ValueError(
                "representation member selection full environment repeats a field"
            )
        selected = _validate_full_environment(identity, self.plan, environment)
        if selected != self.claim:
            raise ValueError(
                "representation member selection claim differs from environment rank"
            )

    @property
    def replay_protected(self) -> bool:
        """Always false until a future one-use receipt is consumed."""

        return False

    @property
    def full_environment_sha256(self) -> str:
        """Bind every exact raw environment field, including CLI receipts."""

        return _canonical_sha256(dict(self.full_environment))

    def as_record(self) -> dict[str, object]:
        """Return fields suitable for future receipt binding, not authority."""

        return {
            "schema": REPRESENTATION_MEMBER_SELECTION_SCHEMA,
            "authorization_scope": "selection-only",
            "cli_run_id": self.identity.run_id,
            "cli_phase": REPRESENTATION_TRAINING_PHASE,
            "cli_command_id": REPRESENTATION_TRAINING_COMMAND_ID,
            "cli_gate_run_identity_sha256": self.identity.gate_run_identity[
                "identity_sha256"
            ],
            "run_identity_sha256": self.plan.run_identity_sha256,
            "config_identity_sha256": self.plan.config_identity_sha256,
            "plan_sha256": self.plan.plan_sha256,
            "claim": self.claim.as_record(),
            "torchrun_environment": {
                name: dict(self.full_environment)[name]
                for name in sorted(_SELECTION_ENVIRONMENT_NAMES)
            },
            "full_environment_sha256": self.full_environment_sha256,
            "replay_protected": False,
        }


def select_representation_member(
    identity: object,
    environment: object,
) -> RepresentationMemberSelection:
    """Select one rank from complete CLI authority without minting evidence."""

    exact_identity, plan = _validate_cli_identity(identity)
    copied_environment = _require_environment(environment)
    claim = _validate_full_environment(
        exact_identity,
        plan,
        copied_environment,
    )
    return RepresentationMemberSelection(
        identity=exact_identity,
        plan=plan,
        claim=claim,
        full_environment=tuple(sorted(copied_environment.items())),
    )


__all__ = [
    "REPRESENTATION_MEMBER_SELECTION_SCHEMA",
    "REPRESENTATION_TRAINING_COMMAND_ID",
    "REPRESENTATION_TRAINING_PHASE",
    "RepresentationMemberSelection",
    "select_representation_member",
]
