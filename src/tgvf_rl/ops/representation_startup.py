"""Pure contracts for a future verified Representation startup.

This dependency-light module binds one complete :class:`WorkerStartupEnvelope`
to the exact members of the current single-node Representation topology.  It
performs no import dispatch, environment access, process creation, or evidence
minting.  In particular, a plan is authorization data only; it is not proof
that a runtime locator, launcher, or member bootstrap has verified anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from tgvf_rl.ops.worker_startup import (
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    WorkerStartupEnvelope,
)


REPRESENTATION_MEMBER_CLAIM_SCHEMA = "tgvf-representation-member-claim-v1"
REPRESENTATION_STARTUP_PLAN_SCHEMA = "tgvf-representation-startup-plan-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMBER_CLAIM_FIELDS = {
    "schema",
    "envelope_sha256",
    "member_identity_sha256",
    "run_identity_sha256",
    "config_identity_sha256",
    "world_size",
    "global_rank",
    "local_rank",
    "physical_gpu_id",
    "claim_sha256",
}
_STARTUP_PLAN_FIELDS = {
    "schema",
    "envelope",
    "envelope_sha256",
    "member_identity_sha256",
    "run_identity_sha256",
    "config_identity_sha256",
    "world_size",
    "members",
}
_AUTHORIZATION_PARAMETER_FIELDS = {
    "representation_startup_plan_schema",
    "representation_startup_plan_json",
    "representation_startup_plan_sha256",
}


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


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"representation startup {name} must be a lowercase SHA-256")
    return value


def _require_world_size(value: object) -> int:
    if type(value) is not int or value not in {2, 4}:
        raise ValueError("representation startup world_size must be exactly 2 or 4")
    return value


def _require_rank(value: object, *, name: str, world_size: int) -> int:
    if type(value) is not int or not 0 <= value < world_size:
        raise ValueError(f"representation startup {name} must be in [0, world_size)")
    return value


def _require_physical_gpu_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            "representation startup physical_gpu_id must be a non-negative integer"
        )
    return value


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"representation startup JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"representation startup JSON contains non-finite value {value!r}")


def _load_strict_json(value: str) -> object:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(
            "representation startup JSON must be valid UTF-8 text"
        ) from error
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("representation startup JSON is malformed") from error


@dataclass(frozen=True, slots=True)
class RepresentationMemberClaim:
    """One exact single-node torchrun member admitted by a startup plan."""

    envelope_sha256: str
    member_identity_sha256: str
    run_identity_sha256: str
    config_identity_sha256: str
    world_size: int
    global_rank: int
    local_rank: int
    physical_gpu_id: int

    def __post_init__(self) -> None:
        _require_sha256(self.envelope_sha256, name="envelope_sha256")
        _require_sha256(
            self.member_identity_sha256,
            name="member_identity_sha256",
        )
        _require_sha256(self.run_identity_sha256, name="run_identity_sha256")
        _require_sha256(self.config_identity_sha256, name="config_identity_sha256")
        world_size = _require_world_size(self.world_size)
        global_rank = _require_rank(
            self.global_rank,
            name="global_rank",
            world_size=world_size,
        )
        local_rank = _require_rank(
            self.local_rank,
            name="local_rank",
            world_size=world_size,
        )
        if global_rank != local_rank:
            raise ValueError(
                "representation startup single-node global_rank must equal local_rank"
            )
        _require_physical_gpu_id(self.physical_gpu_id)

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema": REPRESENTATION_MEMBER_CLAIM_SCHEMA,
            "envelope_sha256": self.envelope_sha256,
            "member_identity_sha256": self.member_identity_sha256,
            "run_identity_sha256": self.run_identity_sha256,
            "config_identity_sha256": self.config_identity_sha256,
            "world_size": self.world_size,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "physical_gpu_id": self.physical_gpu_id,
        }

    @property
    def claim_sha256(self) -> str:
        return _canonical_sha256(self._identity_record())

    def as_record(self) -> dict[str, object]:
        """Return the exact JSON-safe member record including its digest."""

        return {**self._identity_record(), "claim_sha256": self.claim_sha256}

    @classmethod
    def from_record(cls, value: object) -> RepresentationMemberClaim:
        """Reconstruct one member from an exact digest-bound record."""

        if type(value) is not dict or set(value) != _MEMBER_CLAIM_FIELDS:
            raise ValueError("representation member claim field set differs")
        if value["schema"] != REPRESENTATION_MEMBER_CLAIM_SCHEMA:
            raise ValueError("representation member claim schema differs")
        claim = cls(
            envelope_sha256=value["envelope_sha256"],  # type: ignore[arg-type]
            member_identity_sha256=value[  # type: ignore[arg-type]
                "member_identity_sha256"
            ],
            run_identity_sha256=value["run_identity_sha256"],  # type: ignore[arg-type]
            config_identity_sha256=value[  # type: ignore[arg-type]
                "config_identity_sha256"
            ],
            world_size=value["world_size"],  # type: ignore[arg-type]
            global_rank=value["global_rank"],  # type: ignore[arg-type]
            local_rank=value["local_rank"],  # type: ignore[arg-type]
            physical_gpu_id=value["physical_gpu_id"],  # type: ignore[arg-type]
        )
        observed_digest = _require_sha256(
            value["claim_sha256"],
            name="member claim_sha256",
        )
        if observed_digest != claim.claim_sha256:
            raise ValueError("representation member claim digest differs")
        return claim


@dataclass(frozen=True, slots=True)
class RepresentationStartupPlan:
    """Complete authorization data for one launcher and all of its members."""

    envelope: WorkerStartupEnvelope
    run_identity_sha256: str
    config_identity_sha256: str
    world_size: int
    members: tuple[RepresentationMemberClaim, ...]

    def __post_init__(self) -> None:
        if type(self.envelope) is not WorkerStartupEnvelope:
            raise TypeError("representation startup envelope type differs")
        if self.envelope.entry_role != REPRESENTATION_LAUNCHER_ROLE:
            raise ValueError(
                "representation startup envelope entry role must be "
                "representation-launcher"
            )
        member_identity = self.envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
        _require_sha256(self.run_identity_sha256, name="run_identity_sha256")
        _require_sha256(self.config_identity_sha256, name="config_identity_sha256")
        world_size = _require_world_size(self.world_size)
        if type(self.members) is not tuple:
            raise TypeError("representation startup members must be an exact tuple")
        if any(
            type(member) is not RepresentationMemberClaim for member in self.members
        ):
            raise TypeError(
                "representation startup members must be exactly "
                "RepresentationMemberClaim"
            )
        if len(self.members) != world_size:
            raise ValueError(
                "representation startup member count must equal world_size"
            )
        ordered = tuple(sorted(self.members, key=lambda member: member.global_rank))
        object.__setattr__(self, "members", ordered)

        expected_ranks = tuple(range(world_size))
        global_ranks = tuple(member.global_rank for member in ordered)
        local_ranks = tuple(sorted(member.local_rank for member in ordered))
        if global_ranks != expected_ranks:
            raise ValueError(
                "representation startup global ranks must exactly cover "
                "0 through world_size - 1"
            )
        if local_ranks != expected_ranks:
            raise ValueError(
                "representation startup local ranks must exactly cover "
                "0 through world_size - 1"
            )
        physical_gpu_ids = tuple(member.physical_gpu_id for member in ordered)
        if len(set(physical_gpu_ids)) != world_size:
            raise ValueError(
                "representation startup physical GPU mapping must be one-to-one"
            )

        shared_expected = {
            "envelope_sha256": self.envelope.envelope_sha256,
            "member_identity_sha256": member_identity.identity_sha256,
            "run_identity_sha256": self.run_identity_sha256,
            "config_identity_sha256": self.config_identity_sha256,
            "world_size": world_size,
        }
        for member in ordered:
            for name, expected in shared_expected.items():
                if getattr(member, name) != expected:
                    raise ValueError(
                        f"representation member claim differs from plan: {name}"
                    )

    @property
    def envelope_sha256(self) -> str:
        return self.envelope.envelope_sha256

    @property
    def member_identity_sha256(self) -> str:
        return self.envelope.identity_for_role(
            REPRESENTATION_MEMBER_ROLE
        ).identity_sha256

    def as_record(self) -> dict[str, object]:
        """Return the complete role envelope and rank-ordered member claims."""

        return {
            "schema": REPRESENTATION_STARTUP_PLAN_SCHEMA,
            "envelope": self.envelope.as_record(),
            "envelope_sha256": self.envelope_sha256,
            "member_identity_sha256": self.member_identity_sha256,
            "run_identity_sha256": self.run_identity_sha256,
            "config_identity_sha256": self.config_identity_sha256,
            "world_size": self.world_size,
            "members": [member.as_record() for member in self.members],
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.as_record())

    def to_json(self) -> str:
        """Serialize the plan with its one canonical UTF-8 JSON spelling."""

        return _canonical_json(self.as_record())

    @classmethod
    def from_record(cls, value: object) -> RepresentationStartupPlan:
        """Reconstruct a plan from an exact nested record."""

        if type(value) is not dict or set(value) != _STARTUP_PLAN_FIELDS:
            raise ValueError("representation startup plan field set differs")
        if value["schema"] != REPRESENTATION_STARTUP_PLAN_SCHEMA:
            raise ValueError("representation startup plan schema differs")
        member_records = value["members"]
        if type(member_records) is not list:
            raise ValueError("representation startup plan members must be an array")
        envelope = WorkerStartupEnvelope.from_record(value["envelope"])
        plan = cls(
            envelope=envelope,
            run_identity_sha256=value["run_identity_sha256"],  # type: ignore[arg-type]
            config_identity_sha256=value[  # type: ignore[arg-type]
                "config_identity_sha256"
            ],
            world_size=value["world_size"],  # type: ignore[arg-type]
            members=tuple(
                RepresentationMemberClaim.from_record(item) for item in member_records
            ),
        )
        observed_envelope_sha256 = _require_sha256(
            value["envelope_sha256"],
            name="envelope_sha256",
        )
        if observed_envelope_sha256 != plan.envelope_sha256:
            raise ValueError("representation startup envelope digest differs")
        observed_member_sha256 = _require_sha256(
            value["member_identity_sha256"],
            name="member_identity_sha256",
        )
        if observed_member_sha256 != plan.member_identity_sha256:
            raise ValueError("representation startup member identity digest differs")
        return plan

    @classmethod
    def from_json(cls, value: object) -> RepresentationStartupPlan:
        """Parse strict JSON and require its canonical UTF-8 spelling."""

        if type(value) is not str:
            raise TypeError("representation startup plan JSON must be exactly str")
        plan = cls.from_record(_load_strict_json(value))
        if plan.to_json() != value:
            raise ValueError("representation startup plan JSON is not canonical")
        return plan

    def authorization_parameters(self) -> dict[str, str]:
        """Return one exact collision-free authorization parameter group."""

        return {
            "representation_startup_plan_schema": REPRESENTATION_STARTUP_PLAN_SCHEMA,
            "representation_startup_plan_json": self.to_json(),
            "representation_startup_plan_sha256": self.plan_sha256,
        }

    @classmethod
    def from_authorization_parameters(
        cls,
        value: object,
    ) -> RepresentationStartupPlan:
        """Verify one standalone exact group and reconstruct its plan.

        This method deliberately does not accept a broader CLI parameter map.
        A future integration must validate the complete protected namespace;
        it must not silently project these three keys out of a larger map.
        """

        if type(value) is not dict:
            raise TypeError(
                "representation startup authorization parameters must be an exact dict"
            )
        if any(type(key) is not str for key in value):
            raise TypeError(
                "representation startup authorization parameter keys must be "
                "exactly str"
            )
        if set(value) != _AUTHORIZATION_PARAMETER_FIELDS:
            raise ValueError(
                "representation startup authorization parameter field set differs"
            )
        if any(type(item) is not str for item in value.values()):
            raise TypeError(
                "representation startup authorization parameter values must be "
                "exactly str"
            )
        if value["representation_startup_plan_schema"] != (
            REPRESENTATION_STARTUP_PLAN_SCHEMA
        ):
            raise ValueError("representation startup authorization schema differs")
        plan = cls.from_json(value["representation_startup_plan_json"])
        observed_digest = _require_sha256(
            value["representation_startup_plan_sha256"],
            name="plan_sha256",
        )
        if observed_digest != plan.plan_sha256:
            raise ValueError("representation startup authorization digest differs")
        return plan


def build_representation_startup_plan(
    envelope: WorkerStartupEnvelope,
    *,
    run_identity_sha256: str,
    config_identity_sha256: str,
    physical_gpu_ids: tuple[int, ...],
) -> RepresentationStartupPlan:
    """Build deterministic claims for an explicit single-node GPU mapping."""

    if type(envelope) is not WorkerStartupEnvelope:
        raise TypeError("representation startup envelope type differs")
    if type(physical_gpu_ids) is not tuple:
        raise TypeError("representation physical_gpu_ids must be an exact tuple")
    world_size = _require_world_size(len(physical_gpu_ids))
    for gpu_id in physical_gpu_ids:
        _require_physical_gpu_id(gpu_id)
    if len(set(physical_gpu_ids)) != world_size:
        raise ValueError("representation physical_gpu_ids must be distinct")
    _require_sha256(run_identity_sha256, name="run_identity_sha256")
    _require_sha256(config_identity_sha256, name="config_identity_sha256")
    if envelope.entry_role != REPRESENTATION_LAUNCHER_ROLE:
        raise ValueError(
            "representation startup envelope entry role must be representation-launcher"
        )
    member_identity = envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    members = tuple(
        RepresentationMemberClaim(
            envelope_sha256=envelope.envelope_sha256,
            member_identity_sha256=member_identity.identity_sha256,
            run_identity_sha256=run_identity_sha256,
            config_identity_sha256=config_identity_sha256,
            world_size=world_size,
            global_rank=rank,
            local_rank=rank,
            physical_gpu_id=physical_gpu_id,
        )
        for rank, physical_gpu_id in enumerate(physical_gpu_ids)
    )
    return RepresentationStartupPlan(
        envelope=envelope,
        run_identity_sha256=run_identity_sha256,
        config_identity_sha256=config_identity_sha256,
        world_size=world_size,
        members=members,
    )


__all__ = [
    "REPRESENTATION_MEMBER_CLAIM_SCHEMA",
    "REPRESENTATION_STARTUP_PLAN_SCHEMA",
    "RepresentationMemberClaim",
    "RepresentationStartupPlan",
    "build_representation_startup_plan",
]
