"""Cooperative one-use consumption of a Representation rank selection.

This leaf binds a replayable :class:`RepresentationMemberSelection` to the
already-consumed outer CLI launch token, then burns exactly one rank filename
with ``openat(O_EXCL|O_NOFOLLOW)``.  It is deliberately only a cooperative
same-UID replay guard.  It is neither hostile-peer protection nor
``VerifiedWorkerStartup`` evidence, and it does not import or dispatch a
training target.

The authorized outer launcher must pre-create the exact private
``representation-members`` directory.  Workers never create ancestors.
After the exclusive leaf is created, every failure leaves that path in place;
retry requires a new outer launch token.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat

from tgvf_rl.ops.cli_authorization import verify_cli_worker_authorization
from tgvf_rl.ops.cli_authorization_identity import (
    CLIExecutionAuthorizationIdentity,
)
from tgvf_rl.ops.launch_gate import assert_process_liveness
from tgvf_rl.ops.representation_member_selection import (
    RepresentationMemberSelection,
)
from tgvf_rl.ops.representation_startup import RepresentationStartupPlan
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    create_regular_file_exclusive_beneath_nofollow,
    read_regular_file_absolute_nofollow,
    retain_directory_absolute_nofollow,
)


REPRESENTATION_MEMBER_CONSUMPTION_SCHEMA = "tgvf-representation-member-consumption-v1"
REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY = "representation-members"
REPRESENTATION_MEMBER_CONSUMPTION_SECURITY_MODEL = "cooperative-same-uid-v1"
REPRESENTATION_MEMBER_CONSUMPTION_RETRY_POLICY = "new-launch-token-required"

_TOKEN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONSTRUCTION_KEY = object()


class RepresentationMemberConsumptionError(RuntimeError):
    """The rank could not be durably consumed under the declared contract."""


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


def _exact_absolute_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise RepresentationMemberConsumptionError(
            f"representation member {label} must be exact non-empty path text"
        )
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or pure.root != "/"
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise RepresentationMemberConsumptionError(
            f"representation member {label} must be canonical lexical absolute"
        )
    return Path(value)


def _require_private_owned_directory(
    metadata: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RepresentationMemberConsumptionError(
            f"representation member {label} must be owned by the worker EUID "
            "with exact mode 0700"
        )


def _directory_identity_record(metadata: os.stat_result) -> dict[str, object]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def _assert_direct_child_directory(
    parent_descriptor: int,
    *,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        observed = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RepresentationMemberConsumptionError(
            "representation member receipt directory is no longer beneath its token"
        ) from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
        or observed.st_uid != expected.st_uid
        or observed.st_gid != expected.st_gid
        or stat.S_IMODE(observed.st_mode) != stat.S_IMODE(expected.st_mode)
    ):
        raise RepresentationMemberConsumptionError(
            "representation member receipt directory binding changed"
        )


def _process_start_ticks(pid: int) -> int:
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise RepresentationMemberConsumptionError(
            "representation member cannot read its Linux process identity"
        ) from error
    closing = value.rfind(")")
    if closing < 0:
        raise RepresentationMemberConsumptionError(
            "representation member Linux process identity is malformed"
        )
    fields = value[closing + 2 :].split()
    try:
        ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise RepresentationMemberConsumptionError(
            "representation member Linux process identity has no start time"
        ) from error
    if ticks <= 0:
        raise RepresentationMemberConsumptionError(
            "representation member Linux process start time is invalid"
        )
    return ticks


class ConsumedRepresentationMemberAuthorization:
    """PID-bound proof that one cooperative token/rank slot was burned.

    This object is consumption authorization data only.  It does not establish
    runtime immutability, target provenance, cohort atomicity, or worker-startup
    verification.
    """

    __slots__ = (
        "_creation_device",
        "_creation_inode",
        "_creation_size",
        "_receipt_json",
        "_receipt_path",
        "_receipt_sha256",
        "_initialized",
        "_selection_environment",
        "_selection_identity_json",
        "_selection_plan_json",
        "_selection_rank",
        "_selection_record_json",
        "_selection_sha256",
        "_worker_pid",
        "_worker_process_start_ticks",
    )

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError(
            "ConsumedRepresentationMemberAuthorization cannot be subclassed"
        )

    def __init__(
        self,
        *,
        selection: RepresentationMemberSelection,
        receipt_path: Path,
        receipt_json: str,
        receipt_sha256: str,
        creation_device: int,
        creation_inode: int,
        creation_size: int,
        worker_pid: int,
        worker_process_start_ticks: int,
        _construction_key: object,
    ) -> None:
        if _construction_key is not _CONSTRUCTION_KEY:
            raise TypeError(
                "ConsumedRepresentationMemberAuthorization is created only by "
                "consume_representation_member_selection"
            )
        selection_record_json = _canonical_json(selection.as_record())
        object.__setattr__(
            self,
            "_selection_identity_json",
            _canonical_json(selection.identity.as_record()),
        )
        object.__setattr__(self, "_selection_plan_json", selection.plan.to_json())
        object.__setattr__(
            self,
            "_selection_environment",
            tuple(selection.full_environment),
        )
        object.__setattr__(self, "_selection_rank", selection.claim.global_rank)
        object.__setattr__(self, "_selection_record_json", selection_record_json)
        object.__setattr__(
            self,
            "_selection_sha256",
            sha256(selection_record_json.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(self, "_receipt_path", receipt_path)
        object.__setattr__(self, "_receipt_json", receipt_json)
        object.__setattr__(self, "_receipt_sha256", receipt_sha256)
        object.__setattr__(self, "_creation_device", creation_device)
        object.__setattr__(self, "_creation_inode", creation_inode)
        object.__setattr__(self, "_creation_size", creation_size)
        object.__setattr__(self, "_worker_pid", worker_pid)
        object.__setattr__(
            self,
            "_worker_process_start_ticks",
            worker_process_start_ticks,
        )
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(
                "ConsumedRepresentationMemberAuthorization is immutable"
            )
        raise AttributeError(
            "ConsumedRepresentationMemberAuthorization fields are constructor-owned"
        )

    @property
    def selection(self) -> RepresentationMemberSelection:
        """Return a freshly reconstructed immutable selection snapshot."""

        return self._reconstruct_selection()

    @property
    def receipt_path(self) -> Path:
        return self._receipt_path

    @property
    def receipt_sha256(self) -> str:
        return self._receipt_sha256

    @property
    def replay_protected(self) -> bool:
        """True only while used by the consuming process with an intact receipt."""

        self.assert_current_process_and_receipt()
        return True

    def assert_current_process_and_receipt(self) -> None:
        """Reject fork transfer, PID reuse, replacement, and content tampering."""

        current_pid = os.getpid()
        current_process_start_ticks = _process_start_ticks(current_pid)
        if (
            current_pid != self._worker_pid
            or current_process_start_ticks != self._worker_process_start_ticks
        ):
            raise RepresentationMemberConsumptionError(
                "consumed representation member authorization is process-local"
            )
        try:
            snapshot = read_regular_file_absolute_nofollow(self._receipt_path)
        except (OSError, SecureFileReadError) as error:
            raise RepresentationMemberConsumptionError(
                "representation member receipt cannot be securely re-read"
            ) from error
        expected_payload = (self._receipt_json + "\n").encode("utf-8")
        if (
            snapshot.payload != expected_payload
            or sha256(snapshot.payload).hexdigest() != self._receipt_sha256
            or snapshot.before.st_dev != self._creation_device
            or snapshot.before.st_ino != self._creation_inode
            or snapshot.after.st_dev != self._creation_device
            or snapshot.after.st_ino != self._creation_inode
            or snapshot.after.st_size != self._creation_size
            or stat.S_IMODE(snapshot.after.st_mode) != 0o600
        ):
            raise RepresentationMemberConsumptionError(
                "representation member receipt identity or bytes changed"
            )
        selection = self._reconstruct_selection()
        selection_record = selection.as_record()
        selection_record_json = _canonical_json(selection_record)
        if (
            selection_record_json != self._selection_record_json
            or sha256(selection_record_json.encode("utf-8")).hexdigest()
            != self._selection_sha256
        ):
            raise RepresentationMemberConsumptionError(
                "representation member canonical selection snapshot changed"
            )
        receipt_record = json.loads(self._receipt_json)
        receipt_worker_pid = receipt_record.get("worker_pid")
        receipt_worker_process_start_ticks = receipt_record.get(
            "worker_process_start_ticks"
        )
        if (
            type(receipt_worker_pid) is not int
            or type(receipt_worker_process_start_ticks) is not int
            or receipt_worker_pid != self._worker_pid
            or receipt_worker_pid != current_pid
            or receipt_worker_process_start_ticks != self._worker_process_start_ticks
            or receipt_worker_process_start_ticks != current_process_start_ticks
        ):
            raise RepresentationMemberConsumptionError(
                "representation member receipt worker process identity differs"
            )
        if (
            receipt_record.get("selection_sha256") != self._selection_sha256
            or receipt_record.get("claim") != selection.claim.as_record()
            or receipt_record.get("plan_sha256") != selection.plan.plan_sha256
            or receipt_record.get("run_identity_sha256")
            != selection.plan.run_identity_sha256
            or receipt_record.get("config_identity_sha256")
            != selection.plan.config_identity_sha256
            or receipt_record.get("full_environment_sha256")
            != selection.full_environment_sha256
        ):
            raise RepresentationMemberConsumptionError(
                "representation member receipt differs from canonical selection"
            )

    def _reconstruct_selection(self) -> RepresentationMemberSelection:
        try:
            identity_record = json.loads(self._selection_identity_json)
            identity = CLIExecutionAuthorizationIdentity.from_record(identity_record)
            plan = RepresentationStartupPlan.from_json(self._selection_plan_json)
            claim = plan.members[self._selection_rank]
            selection = RepresentationMemberSelection(
                identity=identity,
                plan=plan,
                claim=claim,
                full_environment=self._selection_environment,
            )
        except Exception as error:
            raise RepresentationMemberConsumptionError(
                "representation member canonical selection cannot be reconstructed"
            ) from error
        return selection

    def receipt_record(self) -> dict[str, object]:
        self.assert_current_process_and_receipt()
        value = json.loads(self._receipt_json)
        if type(value) is not dict:  # pragma: no cover - constructor owns JSON
            raise RuntimeError("representation member receipt record ceased to be dict")
        return value

    def __reduce__(self) -> object:
        raise TypeError(
            "ConsumedRepresentationMemberAuthorization is process-local and "
            "not serializable"
        )

    def __copy__(self) -> object:
        raise TypeError(
            "ConsumedRepresentationMemberAuthorization is process-local and "
            "not copyable"
        )

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError(
            "ConsumedRepresentationMemberAuthorization is process-local and "
            "not copyable"
        )


def consume_representation_member_selection(
    selection: object,
) -> ConsumedRepresentationMemberAuthorization:
    """Burn the current selection's exact rank under its verified outer token."""

    if type(selection) is not RepresentationMemberSelection:
        raise TypeError("selection must be exactly RepresentationMemberSelection")
    # Reconstruct to detect post-construction field mutation via object internals.
    validated = RepresentationMemberSelection(
        identity=selection.identity,
        plan=selection.plan,
        claim=selection.claim,
        full_environment=selection.full_environment,
    )
    environment = dict(validated.full_environment)
    try:
        consumption = verify_cli_worker_authorization(
            validated.identity,
            gate_directory=environment["TGVF_CLI_GATE_DIRECTORY"],
            consumption_receipt_path=environment["TGVF_CLI_CONSUMPTION_RECEIPT_PATH"],
            expected_consumption_receipt_sha256=environment[
                "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256"
            ],
            launcher_liveness_receipt_path=environment[
                "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH"
            ],
        )
    except Exception as error:
        raise RepresentationMemberConsumptionError(
            "representation member outer CLI authorization verification failed"
        ) from error

    token_id = consumption.get("token_id")
    if type(token_id) is not str or not _TOKEN_ID_RE.fullmatch(token_id):
        raise RepresentationMemberConsumptionError(
            "representation member outer token ID is malformed"
        )
    gate = _exact_absolute_path(
        environment["TGVF_CLI_GATE_DIRECTORY"],
        label="gate directory",
    )
    consumption_path = _exact_absolute_path(
        environment["TGVF_CLI_CONSUMPTION_RECEIPT_PATH"],
        label="outer consumption receipt path",
    )
    liveness_path = _exact_absolute_path(
        environment["TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH"],
        label="launcher liveness receipt path",
    )
    if consumption_path != gate / "consumptions" / f"{token_id}.json":
        raise RepresentationMemberConsumptionError(
            "representation member outer consumption path differs from token"
        )
    expected_consumption_sha256 = environment["TGVF_CLI_CONSUMPTION_RECEIPT_SHA256"]
    if not _SHA256_RE.fullmatch(expected_consumption_sha256):
        raise RepresentationMemberConsumptionError(
            "representation member outer consumption digest is malformed"
        )

    token_directory = gate / "cli-launches" / token_id
    if liveness_path != token_directory / "launcher-liveness.json":
        raise RepresentationMemberConsumptionError(
            "representation member launcher liveness path differs from token"
        )
    try:
        verified_liveness = assert_process_liveness(
            liveness_path,
            expected_run_id=validated.identity.run_id,
            expected_phase=validated.identity.phase,
        )
        liveness_snapshot = read_regular_file_absolute_nofollow(liveness_path)
        observed_liveness = json.loads(liveness_snapshot.payload.decode("utf-8"))
    except Exception as error:
        raise RepresentationMemberConsumptionError(
            "representation member launcher liveness binding failed"
        ) from error
    if observed_liveness != verified_liveness:
        raise RepresentationMemberConsumptionError(
            "representation member launcher liveness changed during verification"
        )

    receipt_directory = token_directory / REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY
    try:
        with retain_directory_absolute_nofollow(token_directory) as token_binding:
            token_directory_metadata = token_binding.assert_path_binding()
            _require_private_owned_directory(
                token_directory_metadata,
                label="token launch directory",
            )
            with retain_directory_absolute_nofollow(receipt_directory) as binding:
                receipt_directory_metadata = binding.assert_path_binding()
                _require_private_owned_directory(
                    receipt_directory_metadata,
                    label="receipt directory",
                )
                _assert_direct_child_directory(
                    token_binding.fileno(),
                    name=REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY,
                    expected=receipt_directory_metadata,
                )
                worker_pid = os.getpid()
                worker_ticks = _process_start_ticks(worker_pid)
                rank = validated.claim.global_rank
                relative_name = f"rank-{rank}.json"
                receipt_path = receipt_directory / relative_name
                selection_record = validated.as_record()
                record: dict[str, object] = {
                    "schema": REPRESENTATION_MEMBER_CONSUMPTION_SCHEMA,
                    "status": "consumed",
                    "authorization_scope": "cooperative-token-scoped-rank",
                    "security_model": (
                        REPRESENTATION_MEMBER_CONSUMPTION_SECURITY_MODEL
                    ),
                    "hostile_same_uid_protected": False,
                    "retry_policy": REPRESENTATION_MEMBER_CONSUMPTION_RETRY_POLICY,
                    "token_id": token_id,
                    "cli_run_id": validated.identity.run_id,
                    "cli_phase": validated.identity.phase,
                    "cli_command_id": validated.identity.command_id,
                    "cli_gate_run_identity_sha256": (
                        validated.identity.gate_run_identity["identity_sha256"]
                    ),
                    "outer_consumption_receipt_path": str(consumption_path),
                    "outer_consumption_receipt_sha256": (expected_consumption_sha256),
                    "launcher_liveness_receipt_path": str(liveness_path),
                    "launcher_liveness_receipt_sha256": sha256(
                        liveness_snapshot.payload
                    ).hexdigest(),
                    "launcher_pid": verified_liveness["pid"],
                    "launcher_process_start_ticks": verified_liveness[
                        "process_start_ticks"
                    ],
                    "launcher_host": consumption["consumer_host"],
                    "token_directory_identity": _directory_identity_record(
                        token_directory_metadata
                    ),
                    "receipt_directory_identity": _directory_identity_record(
                        receipt_directory_metadata
                    ),
                    "run_identity_sha256": validated.plan.run_identity_sha256,
                    "config_identity_sha256": validated.plan.config_identity_sha256,
                    "plan_sha256": validated.plan.plan_sha256,
                    "envelope_sha256": validated.plan.envelope_sha256,
                    "member_identity_sha256": (validated.plan.member_identity_sha256),
                    "selection_sha256": _canonical_sha256(selection_record),
                    "claim": validated.claim.as_record(),
                    "full_environment_sha256": validated.full_environment_sha256,
                    "torchelastic_run_id": environment["TORCHELASTIC_RUN_ID"],
                    "master_addr": environment["MASTER_ADDR"],
                    "master_port": environment["MASTER_PORT"],
                    "torchelastic_error_file": environment["TORCHELASTIC_ERROR_FILE"],
                    "worker_pid": worker_pid,
                    "worker_process_start_ticks": worker_ticks,
                    "worker_host": socket.gethostname(),
                    "receipt_relative_path": (
                        f"{REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY}/{relative_name}"
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "replay_protected": True,
                }
                receipt_json = _canonical_json(record)
                creation = create_regular_file_exclusive_beneath_nofollow(
                    binding,
                    relative_name,
                    (receipt_json + "\n").encode("utf-8"),
                    mode=0o600,
                )
                _assert_direct_child_directory(
                    token_binding.fileno(),
                    name=REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY,
                    expected=receipt_directory_metadata,
                )
                token_binding.assert_path_binding()
                if (
                    creation.metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(creation.metadata.st_mode) != 0o600
                ):
                    raise RepresentationMemberConsumptionError(
                        "representation member receipt ownership or mode differs"
                    )
    except FileExistsError as error:
        raise RepresentationMemberConsumptionError(
            "representation member rank is already consumed for this token"
        ) from error
    except RepresentationMemberConsumptionError:
        raise
    except Exception as error:
        raise RepresentationMemberConsumptionError(
            "representation member receipt creation failed; the rank may be "
            "permanently burned for this token"
        ) from error

    authorization = ConsumedRepresentationMemberAuthorization(
        selection=validated,
        receipt_path=receipt_path,
        receipt_json=receipt_json,
        receipt_sha256=creation.payload_sha256,
        creation_device=creation.metadata.st_dev,
        creation_inode=creation.metadata.st_ino,
        creation_size=creation.byte_length,
        worker_pid=worker_pid,
        worker_process_start_ticks=worker_ticks,
        _construction_key=_CONSTRUCTION_KEY,
    )
    authorization.assert_current_process_and_receipt()
    return authorization


__all__ = [
    "REPRESENTATION_MEMBER_CONSUMPTION_DIRECTORY",
    "REPRESENTATION_MEMBER_CONSUMPTION_RETRY_POLICY",
    "REPRESENTATION_MEMBER_CONSUMPTION_SCHEMA",
    "REPRESENTATION_MEMBER_CONSUMPTION_SECURITY_MODEL",
    "ConsumedRepresentationMemberAuthorization",
    "RepresentationMemberConsumptionError",
    "consume_representation_member_selection",
]
