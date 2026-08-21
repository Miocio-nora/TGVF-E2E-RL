"""Project checkpoint contributors used alongside veRL's public FSDP2 I/O."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tgvf_rl.checkpoint import CheckpointCoordinator
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion

from .compatibility import FSDP2BridgeConfig

if TYPE_CHECKING:
    from tgvf_rl.policy.checkpoint import (
        OpaqueProjectState,
        PilotOptimizerDataCursor,
        PilotProjectCheckpointState,
        PilotRunIdentityHashes,
    )
    from tgvf_rl.policy.lifecycle import PolicyBatchLifecycleManager
    from tgvf_rl.policy.metrics import PilotMetricsAccumulator


POLICY_PILOT_VERL_CHECKPOINT_PAIR_SCHEMA = (
    "policy-pilot-v1-verl-checkpoint-pair-v1"
)
POLICY_PILOT_PROJECT_STATE_FILENAME = "tgvf_policy_project_state.json"
POLICY_PILOT_CHECKPOINT_PAIR_FILENAME = "tgvf_policy_checkpoint_pair.json"


@runtime_checkable
class StatefulTeacher(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@runtime_checkable
class PolicyPilotCheckpointStatePort(Protocol):
    """Mutable project owners paired with one framework-owned checkpoint."""

    def run_identity(self) -> "PilotRunIdentityHashes": ...

    def progress(self) -> "PilotOptimizerDataCursor": ...

    def rollout_sampler_state(self) -> "OpaqueProjectState": ...

    def rollout_rng_state(self) -> "OpaqueProjectState": ...

    def current_policy_version(self) -> PolicyVersion: ...

    def reference_policy_version(self) -> PolicyVersion: ...

    def restore_progress(self, value: "PilotOptimizerDataCursor") -> None: ...

    def restore_rollout_sampler_state(self, value: "OpaqueProjectState") -> None: ...

    def restore_rollout_rng_state(self, value: "OpaqueProjectState") -> None: ...


@runtime_checkable
class VerlCheckpointIO(Protocol):
    """Public worker-group/engine checkpoint calls used by the pair bridge."""

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> object: ...

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class PolicyPilotVerlCheckpointPair:
    """Commit marker pairing one upstream checkpoint with its adjunct."""

    run_id: str
    optimizer_step: int
    project_state_sha256: str
    upstream_save_contents: tuple[str, ...]
    upstream_load_contents: tuple[str, ...]
    schema_version: str = POLICY_PILOT_VERL_CHECKPOINT_PAIR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_PILOT_VERL_CHECKPOINT_PAIR_SCHEMA:
            raise ValueError("unsupported Policy Pilot veRL checkpoint pair schema")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("checkpoint pair run_id must be non-empty")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("checkpoint pair optimizer_step must be non-negative")
        _require_sha256(self.project_state_sha256, "project checkpoint digest")
        object.__setattr__(
            self, "upstream_save_contents", tuple(self.upstream_save_contents)
        )
        object.__setattr__(
            self, "upstream_load_contents", tuple(self.upstream_load_contents)
        )
        required = {"model", "optimizer", "extra"}
        if not required.issubset(self.upstream_save_contents):
            raise ValueError("checkpoint pair save contents omit framework state")
        if not required.issubset(self.upstream_load_contents):
            raise ValueError("checkpoint pair load contents omit framework state")

    @property
    def integrity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._content_mapping())).hexdigest()

    def _content_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "optimizer_step": self.optimizer_step,
            "project_state_sha256": self.project_state_sha256,
            "upstream_save_contents": list(self.upstream_save_contents),
            "upstream_load_contents": list(self.upstream_load_contents),
        }

    def to_checkpoint_mapping(self) -> dict[str, object]:
        value = self._content_mapping()
        value["integrity_sha256"] = self.integrity_sha256
        return value

    @classmethod
    def from_checkpoint_mapping(
        cls, value: object
    ) -> "PolicyPilotVerlCheckpointPair":
        expected = {
            "schema_version",
            "run_id",
            "optimizer_step",
            "project_state_sha256",
            "upstream_save_contents",
            "upstream_load_contents",
            "integrity_sha256",
        }
        mapping = _strict_mapping(value, expected, "veRL checkpoint pair")
        try:
            _require_sha256(mapping["integrity_sha256"], "checkpoint pair digest")
            content = {
                name: mapping[name] for name in mapping if name != "integrity_sha256"
            }
            actual = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("checkpoint pair digest is malformed") from error
        if not hmac.compare_digest(actual, mapping["integrity_sha256"]):
            raise ReplayMismatchError("checkpoint pair integrity mismatch")
        try:
            save_contents = mapping["upstream_save_contents"]
            load_contents = mapping["upstream_load_contents"]
            if not isinstance(save_contents, list) or not isinstance(load_contents, list):
                raise TypeError("checkpoint content identities must be JSON lists")
            return cls(
                schema_version=mapping["schema_version"],
                run_id=mapping["run_id"],
                optimizer_step=mapping["optimizer_step"],
                project_state_sha256=mapping["project_state_sha256"],
                upstream_save_contents=tuple(save_contents),
                upstream_load_contents=tuple(load_contents),
            )
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("veRL checkpoint pair is malformed") from error


@dataclass(frozen=True, slots=True)
class PolicyPilotVerlResumeResult:
    optimizer_step: int
    project_state: "PilotProjectCheckpointState"
    pair: PolicyPilotVerlCheckpointPair


class PairedPolicyPilotVerlCheckpoint:
    """Pair upstream FSDP2 state and project state at one quiescent boundary.

    The upstream object remains the sole owner of LoRA/model shards, optimizer,
    scheduler, scaler/RNG and distributed I/O.  This adapter holds the Policy
    lifecycle checkpoint gate across that synchronous call and commits the
    project adjunct last.  Absence of the final pair file means the checkpoint
    generation is incomplete and cannot be resumed.
    """

    def __init__(
        self,
        *,
        upstream: VerlCheckpointIO,
        fsdp2: FSDP2BridgeConfig,
        lifecycle_manager: "PolicyBatchLifecycleManager",
        state_port: PolicyPilotCheckpointStatePort,
        metrics_accumulator: "PilotMetricsAccumulator",
    ) -> None:
        from tgvf_rl.policy.lifecycle import PolicyBatchLifecycleManager
        from tgvf_rl.policy.metrics import PilotMetricsAccumulator

        if any(
            not callable(getattr(upstream, method, None))
            for method in ("save_checkpoint", "load_checkpoint")
        ):
            raise TypeError("upstream must expose public save/load checkpoint calls")
        validate_fsdp2_checkpoint_config(fsdp2)
        if not isinstance(lifecycle_manager, PolicyBatchLifecycleManager):
            raise TypeError("lifecycle_manager must be PolicyBatchLifecycleManager")
        if not isinstance(state_port, PolicyPilotCheckpointStatePort):
            raise TypeError("state_port must implement Policy Pilot checkpoint owners")
        if not isinstance(metrics_accumulator, PilotMetricsAccumulator):
            raise TypeError("metrics_accumulator must be PilotMetricsAccumulator")
        self.upstream = upstream
        self.fsdp2 = fsdp2
        self.lifecycle_manager = lifecycle_manager
        self.state_port = state_port
        self.metrics_accumulator = metrics_accumulator

    def save_checkpoint(
        self,
        local_path: str | Path,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> "PilotProjectCheckpointState":
        """Synchronously save upstream state and commit its project pair last."""

        destination = _local_checkpoint_path(local_path, hdfs_path=hdfs_path)
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        state_path = destination / POLICY_PILOT_PROJECT_STATE_FILENAME
        pair_path = destination / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME
        if state_path.exists() or pair_path.exists():
            raise RuntimeError("Policy Pilot checkpoint generation already exists")

        with self.lifecycle_manager.checkpoint_barrier() as barrier:
            state = self._capture_project_state(barrier)
            if state.progress.optimizer_step != global_step:
                raise IdentityMismatchError(
                    "upstream global_step differs from Policy project state"
                )
            self.upstream.save_checkpoint(
                str(destination),
                None,
                global_step,
                max_ckpt_to_keep,
            )
            if not destination.is_dir():
                raise RuntimeError("upstream checkpoint did not materialize its directory")
            pair = PolicyPilotVerlCheckpointPair(
                run_id=state.run_identity.run_id,
                optimizer_step=state.progress.optimizer_step,
                project_state_sha256=state.integrity_sha256,
                upstream_save_contents=self.fsdp2.checkpoint_save_contents,
                upstream_load_contents=self.fsdp2.checkpoint_load_contents,
            )
            _write_json_exclusive(state_path, state.to_checkpoint_mapping())
            _write_json_exclusive(pair_path, pair.to_checkpoint_mapping())
            return state

    def load_checkpoint(
        self,
        local_path: str | Path,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> PolicyPilotVerlResumeResult:
        """Clean-process resume entry for one complete committed generation."""

        destination = _local_checkpoint_path(local_path, hdfs_path=hdfs_path)
        state, pair = self._read_committed_pair(destination)
        expected_run_identity = self.state_port.run_identity()
        if state.run_identity != expected_run_identity:
            raise IdentityMismatchError(
                "Policy Pilot resume run identity differs from checkpoint"
            )

        with self.lifecycle_manager.checkpoint_barrier() as barrier:
            if state.rollout_barrier != barrier:
                raise ReplayMismatchError(
                    "saved rollout barrier differs from the clean-process barrier"
                )
            self.upstream.load_checkpoint(
                str(destination), None, bool(del_local_after_load)
            )
            record_loaded_policy = getattr(
                self.state_port, "record_loaded_policy_version", None
            )
            if callable(record_loaded_policy):
                # Full-model sync receipts are operational identities rather
                # than tensor snapshots.  Give that state owner a chance to
                # bind the successfully loaded upstream checkpoint before the
                # same strict project identity check used by snapshot paths.
                record_loaded_policy(state.policy_version)
            loaded_policy = self.state_port.current_policy_version()
            loaded_reference = self.state_port.reference_policy_version()
            from tgvf_rl.policy.checkpoint import (
                validate_pilot_project_checkpoint_restore,
            )

            validated = validate_pilot_project_checkpoint_restore(
                state,
                expected_run_identity=expected_run_identity,
                loaded_policy_version=loaded_policy,
                loaded_reference_version=loaded_reference,
            )
            self._restore_project_owners(validated)
            return PolicyPilotVerlResumeResult(
                optimizer_step=validated.progress.optimizer_step,
                project_state=validated,
                pair=pair,
            )

    def _capture_project_state(self, barrier: object) -> "PilotProjectCheckpointState":
        from tgvf_rl.policy.checkpoint import (
            PilotRolloutBarrier,
            capture_pilot_project_checkpoint,
        )

        if not isinstance(barrier, PilotRolloutBarrier):
            raise TypeError("lifecycle checkpoint barrier is malformed")
        return capture_pilot_project_checkpoint(
            run_identity=self.state_port.run_identity(),
            progress=self.state_port.progress(),
            rollout_sampler_state=self.state_port.rollout_sampler_state(),
            rollout_rng_state=self.state_port.rollout_rng_state(),
            metrics_accumulator=self.metrics_accumulator,
            policy_version=self.state_port.current_policy_version(),
            reference_version=self.state_port.reference_policy_version(),
            rollout_barrier=barrier,
        )

    def _read_committed_pair(
        self, destination: Path
    ) -> tuple["PilotProjectCheckpointState", PolicyPilotVerlCheckpointPair]:
        return read_committed_policy_checkpoint_pair(destination, fsdp2=self.fsdp2)

    def _restore_project_owners(self, state: "PilotProjectCheckpointState") -> None:
        before_progress = self.state_port.progress()
        before_sampler = self.state_port.rollout_sampler_state()
        before_rng = self.state_port.rollout_rng_state()
        before_metrics = self.metrics_accumulator.state
        try:
            self.state_port.restore_progress(state.progress)
            self.state_port.restore_rollout_sampler_state(state.rollout_sampler_state)
            self.state_port.restore_rollout_rng_state(state.rollout_rng_state)
            self.metrics_accumulator.restore_checkpoint_state(state.metrics_state)
            restored = (
                self.state_port.progress(),
                self.state_port.rollout_sampler_state(),
                self.state_port.rollout_rng_state(),
                self.metrics_accumulator.state,
            )
            expected = (
                state.progress,
                state.rollout_sampler_state,
                state.rollout_rng_state,
                state.metrics_state,
            )
            if restored != expected:
                raise ReplayMismatchError(
                    "project checkpoint owner did not restore its exact state"
                )
        except BaseException as error:
            rollback_errors: list[str] = []
            for operation in (
                lambda: self.state_port.restore_progress(before_progress),
                lambda: self.state_port.restore_rollout_sampler_state(before_sampler),
                lambda: self.state_port.restore_rollout_rng_state(before_rng),
                lambda: self.metrics_accumulator.restore_checkpoint_state(before_metrics),
            ):
                try:
                    operation()
                except BaseException as rollback_error:
                    rollback_errors.append(
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    )
            if rollback_errors:
                error.add_note(
                    "Policy project-state rollback also failed: "
                    f"{tuple(rollback_errors)!r}"
                )
            raise


class SDPOTeacherCheckpointContributor:
    """Make the independently stateful SDPO teacher a strict project section."""

    checkpoint_name = "sdpo_teacher_state"
    checkpoint_version = "sdpo-teacher-checkpoint-v1"

    def __init__(self, teacher: StatefulTeacher) -> None:
        if not isinstance(teacher, StatefulTeacher):
            raise TypeError("SDPO teacher must expose state_dict/load_state_dict")
        self.teacher = teacher

    def checkpoint_state(self) -> object:
        state = self.teacher.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("SDPO teacher state_dict must return a mapping")
        # A contributor owns a snapshot, never a live alias of teacher state.
        return deepcopy(dict(state))

    def restore_checkpoint_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("SDPO teacher checkpoint state must be a mapping")
        self.teacher.load_state_dict(deepcopy(dict(state)))

    # State-dict aliases make the contributor useful to other maintained
    # orchestration layers without changing CheckpointCoordinator's protocol.
    def state_dict(self) -> Mapping[str, object]:
        value = self.checkpoint_state()
        assert isinstance(value, Mapping)
        return value

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.restore_checkpoint_state(state)


def register_sdpo_teacher_checkpoint(
    coordinator: CheckpointCoordinator,
    teacher: StatefulTeacher,
) -> SDPOTeacherCheckpointContributor:
    """Register teacher state separately from actor/reference model shards."""

    if not isinstance(coordinator, CheckpointCoordinator):
        raise TypeError("coordinator must be CheckpointCoordinator")
    contributor = SDPOTeacherCheckpointContributor(teacher)
    coordinator.register(contributor)
    return contributor


def validate_fsdp2_checkpoint_config(config: FSDP2BridgeConfig) -> None:
    """Type-check hook used before delegating model shards to veRL."""

    if not isinstance(config, FSDP2BridgeConfig):
        raise TypeError("config must be a validated FSDP2BridgeConfig")
    # Construction performs all fail-closed semantic checks.  Access the
    # required fields here so an incompatible look-alike cannot pass by duck type.
    if config.checkpoint_async_save or not config.checkpoint_strict:
        raise ValueError("FSDP2 project checkpoints must be synchronous and strict")


def _local_checkpoint_path(
    value: str | Path, *, hdfs_path: str | None
) -> Path:
    if hdfs_path is not None:
        raise ValueError("Policy Pilot paired checkpoint currently supports local I/O only")
    if not isinstance(value, (str, Path)):
        raise TypeError("checkpoint local_path must be str or Path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Policy Pilot checkpoint local_path must be absolute")
    return path


def read_committed_policy_checkpoint_pair(
    destination: str | Path,
    *,
    fsdp2: FSDP2BridgeConfig,
) -> tuple["PilotProjectCheckpointState", PolicyPilotVerlCheckpointPair]:
    """Read one committed project/upstream pair without restoring its owners.

    Checkpoint retention and permanent-copy code must make deletion decisions
    from the same integrity-checked commit marker used by clean-process resume.
    Keeping this reader beside the save/load bridge prevents a weaker directory
    naming check from being mistaken for checkpoint identity.
    """

    from tgvf_rl.policy.checkpoint import PilotProjectCheckpointState

    validate_fsdp2_checkpoint_config(fsdp2)
    path = Path(destination)
    if not path.is_absolute():
        raise ValueError("Policy Pilot checkpoint destination must be absolute")
    state_path = path / POLICY_PILOT_PROJECT_STATE_FILENAME
    pair_path = path / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME
    if not state_path.is_file() or not pair_path.is_file():
        raise ReplayMismatchError(
            "Policy Pilot checkpoint is incomplete: committed pair files missing"
        )
    state = PilotProjectCheckpointState.from_checkpoint_mapping(
        _read_json(state_path)
    )
    pair = PolicyPilotVerlCheckpointPair.from_checkpoint_mapping(
        _read_json(pair_path)
    )
    mismatches: dict[str, object] = {}
    expected = {
        "run_id": state.run_identity.run_id,
        "optimizer_step": state.progress.optimizer_step,
        "project_state_sha256": state.integrity_sha256,
        "upstream_save_contents": fsdp2.checkpoint_save_contents,
        "upstream_load_contents": fsdp2.checkpoint_load_contents,
    }
    for name, value in expected.items():
        if getattr(pair, name) != value:
            mismatches[name] = (getattr(pair, name), value)
    if mismatches:
        raise ReplayMismatchError(
            f"Policy Pilot checkpoint pair identity mismatch: {mismatches!r}"
        )
    return state, pair


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint file {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            raise RuntimeError(f"refusing to overwrite checkpoint file {path.name!r}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayMismatchError(
            f"Policy Pilot checkpoint file is unreadable: {path.name}"
        ) from error


def _strict_mapping(
    value: object, expected: set[str], owner: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReplayMismatchError(f"{owner} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ReplayMismatchError(
            f"{owner} fields differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    if any(type(name) is not str for name in value):
        raise ReplayMismatchError(f"{owner} keys must be strings")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint pair is not canonical JSON") from error


def _require_sha256(value: object, owner: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{owner} must be lowercase SHA-256")
