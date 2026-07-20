"""Batch-scoped lifetime for transient Policy Pilot exact-replay evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING

from tgvf_rl.observations.store import ObservationReleaseCounts, ObservationStore
from tgvf_rl.trajectories.behavior import BehaviorTraceStore

if TYPE_CHECKING:
    from tgvf_rl.environment.focus_runtime import FocusExecutionLedger
    from tgvf_rl.framework.verl.data_bridge import DataProtoPayload
    from tgvf_rl.policy.checkpoint import PilotRolloutBarrier


class PolicyBatchMilestone(str, Enum):
    BEHAVIOR_REPLAY = "behavior_replay"
    CURRENT_REPLAY = "current_replay"
    REFERENCE_REPLAY = "reference_replay"
    LOSS_BACKWARD = "loss_backward"
    OPTIMIZER_STEP = "optimizer_step"
    ZERO_STALENESS_BARRIER = "zero_staleness_barrier"


class PolicyBatchState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ABORTED = "aborted"
    COMMIT_UNKNOWN = "commit_unknown"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True, slots=True)
class PolicyBatchReleaseReport:
    batch_id: str
    state: PolicyBatchState
    ledger_entries: int
    observation_records: int
    replay_records: int
    tensors: int
    behavior_traces: int
    data_proto_sidecar_fields: int
    transient_owners: int


class PolicyBatchTransientState:
    """Project-owned holder whose release is a non-throwing mapping clear."""

    def __init__(self, values: Mapping[str, object]) -> None:
        if not isinstance(values, Mapping) or not values:
            raise ValueError("transient state requires a non-empty mapping")
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError("transient state names must be non-empty strings")
        self._lock = RLock()
        self._values = dict(values)
        self._released = False

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def get(self, name: str) -> object:
        with self._lock:
            if self._released:
                raise RuntimeError("Policy batch transient state has been released")
            try:
                return self._values[name]
            except KeyError as error:
                raise KeyError(f"unknown Policy batch transient {name!r}") from error

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            self._values.clear()
            self._released = True
            return True


_REQUIRED_MILESTONES = frozenset(PolicyBatchMilestone)
_MILESTONE_DEPENDENCIES = {
    PolicyBatchMilestone.BEHAVIOR_REPLAY: frozenset(),
    PolicyBatchMilestone.CURRENT_REPLAY: frozenset(
        {PolicyBatchMilestone.BEHAVIOR_REPLAY}
    ),
    PolicyBatchMilestone.REFERENCE_REPLAY: frozenset(
        {PolicyBatchMilestone.BEHAVIOR_REPLAY}
    ),
    PolicyBatchMilestone.LOSS_BACKWARD: frozenset(
        {
            PolicyBatchMilestone.CURRENT_REPLAY,
            PolicyBatchMilestone.REFERENCE_REPLAY,
        }
    ),
    PolicyBatchMilestone.OPTIMIZER_STEP: frozenset(
        {PolicyBatchMilestone.LOSS_BACKWARD}
    ),
    PolicyBatchMilestone.ZERO_STALENESS_BARRIER: frozenset(
        {PolicyBatchMilestone.OPTIMIZER_STEP}
    ),
}


class PolicyBatchLifecycle:
    """One exact-observation lease from rollout through optimizer commit."""

    def __init__(
        self,
        *,
        manager: PolicyBatchLifecycleManager,
        batch_id: str,
        trajectory_ids: tuple[str, ...],
    ) -> None:
        self._manager = manager
        self.batch_id = batch_id
        self.trajectory_ids = trajectory_ids
        self._lock = RLock()
        self._state = PolicyBatchState.OPEN
        self._completed: set[PolicyBatchMilestone] = set()
        self._inflight: set[PolicyBatchMilestone] = set()
        self._active_consumers = 0
        self._data_protos: list[DataProtoPayload] = []
        self._transient_states: list[PolicyBatchTransientState] = []
        self._release_report: PolicyBatchReleaseReport | None = None
        self._cleanup_errors: tuple[str, ...] = ()

    @property
    def state(self) -> PolicyBatchState:
        with self._lock:
            return self._state

    @property
    def completed_milestones(self) -> frozenset[PolicyBatchMilestone]:
        with self._lock:
            return frozenset(self._completed)

    @property
    def active_consumer_count(self) -> int:
        with self._lock:
            return self._active_consumers

    @property
    def cleanup_errors(self) -> tuple[str, ...]:
        with self._lock:
            return self._cleanup_errors

    def assert_open(self) -> None:
        with self._lock:
            self._assert_open_locked()

    def attach_data_proto(self, data: DataProtoPayload) -> None:
        """Register one project-owned neutral DataProto sidecar holder."""

        _validate_data_proto_sidecars(data)
        with self._lock:
            self._assert_open_locked()
            if any(existing is data for existing in self._data_protos):
                return
            self._data_protos.append(data)

    def attach_transient_state(self, state: PolicyBatchTransientState) -> None:
        if not isinstance(state, PolicyBatchTransientState):
            raise TypeError("transient state must be PolicyBatchTransientState")
        if state.released:
            raise RuntimeError("cannot attach released transient state")
        with self._lock:
            self._assert_open_locked()
            if any(existing is state for existing in self._transient_states):
                return
            self._transient_states.append(state)

    @contextmanager
    def consume(self, milestone: PolicyBatchMilestone) -> Iterator[None]:
        """Keep evidence leased for one real consumer and mark only on success."""

        if not isinstance(milestone, PolicyBatchMilestone):
            raise TypeError("milestone must be PolicyBatchMilestone")
        self._manager._begin_consumer(self, milestone)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            self._manager._end_consumer(self, milestone, succeeded=succeeded)

    def close(self) -> PolicyBatchReleaseReport:
        """Release a successfully consumed batch; repeated calls are no-ops."""

        return self._manager._finish(self, aborted=False)

    def abort(self) -> PolicyBatchReleaseReport:
        """Release after an exception, but never while a consumer is active."""

        return self._manager._finish(self, aborted=True)

    def _assert_open_locked(self) -> None:
        if self._state is not PolicyBatchState.OPEN:
            raise RuntimeError(
                f"Policy batch {self.batch_id!r} is already {self._state.value}"
            )

    def _validate_dependencies_locked(self, milestone: PolicyBatchMilestone) -> None:
        if milestone in self._completed:
            raise RuntimeError(
                f"Policy batch milestone {milestone.value!r} was already consumed"
            )
        if milestone in self._inflight:
            raise RuntimeError(
                f"Policy batch milestone {milestone.value!r} is already in flight"
            )
        missing = _MILESTONE_DEPENDENCIES[milestone] - self._completed
        if missing:
            names = tuple(sorted(value.value for value in missing))
            raise RuntimeError(
                f"Policy batch milestone {milestone.value!r} is premature; "
                f"missing={names!r}"
            )


class PolicyBatchLifecycleManager:
    """Own active batch identities and enforce update/checkpoint gates."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        focus_execution_ledger: FocusExecutionLedger,
    ) -> None:
        if not isinstance(observation_store, ObservationStore):
            raise TypeError("observation_store must be ObservationStore")
        if not isinstance(behavior_store, BehaviorTraceStore):
            raise TypeError("behavior_store must be BehaviorTraceStore")
        if any(
            not callable(getattr(focus_execution_ledger, method, None))
            for method in (
                "assert_releasable",
                "release_trajectories",
                "entry_count",
            )
        ):
            raise TypeError(
                "focus_execution_ledger must implement the batch release interface"
            )
        self.observation_store = observation_store
        self.behavior_store = behavior_store
        self.focus_execution_ledger = focus_execution_ledger
        self._lock = RLock()
        self._active: dict[str, PolicyBatchLifecycle] = {}
        self._trajectory_owner: dict[str, str] = {}
        self._seen_batch_ids: set[str] = set()
        self._seen_trajectory_ids: set[str] = set()
        self._checkpoint_gate_active = False
        self._update_gate_owner: str | None = None

    def open_batch(
        self, *, batch_id: str, trajectory_ids: tuple[str, ...]
    ) -> PolicyBatchLifecycle:
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        identities = _trajectory_id_set(trajectory_ids)
        with self._lock:
            if self._checkpoint_gate_active:
                raise RuntimeError("cannot open a batch during checkpoint capture")
            if self._update_gate_owner is not None:
                raise RuntimeError("cannot open a batch during a Policy update gate")
            failed = tuple(
                name
                for name, batch in self._active.items()
                if batch.state is PolicyBatchState.CLEANUP_FAILED
            )
            if failed:
                raise RuntimeError(
                    f"cannot continue after terminal batch cleanup failure: {failed!r}"
                )
            if batch_id in self._seen_batch_ids:
                raise RuntimeError("Policy batch identity cannot be reused")
            reused_trajectories = tuple(
                identity
                for identity in identities
                if identity in self._seen_trajectory_ids
            )
            if reused_trajectories:
                raise RuntimeError(
                    "Policy trajectory identities cannot be reused: "
                    f"{reused_trajectories!r}"
                )
            collisions = {
                identity: self._trajectory_owner[identity]
                for identity in identities
                if identity in self._trajectory_owner
            }
            if collisions:
                raise RuntimeError(
                    f"trajectories already belong to an open batch: {collisions!r}"
                )
            batch = PolicyBatchLifecycle(
                manager=self,
                batch_id=batch_id,
                trajectory_ids=identities,
            )
            self._active[batch_id] = batch
            self._seen_batch_ids.add(batch_id)
            self._seen_trajectory_ids.update(identities)
            for identity in identities:
                self._trajectory_owner[identity] = batch_id
            return batch

    @property
    def outstanding_batch_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def outstanding_batch_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))

    def assert_owns(self, batch: PolicyBatchLifecycle) -> None:
        if not isinstance(batch, PolicyBatchLifecycle):
            raise TypeError("batch must be PolicyBatchLifecycle")
        with self._lock:
            if self._active.get(batch.batch_id) is not batch:
                raise RuntimeError("Policy batch is not open in this manager")

    def assert_quiescent(self) -> None:
        with self._lock:
            self._assert_quiescent_locked()

    @contextmanager
    def checkpoint_barrier(self) -> Iterator[PilotRolloutBarrier]:
        """Hold a gate that rejects new batches for the entire checkpoint."""

        from tgvf_rl.policy.checkpoint import PilotRolloutBarrier

        with self._lock:
            self._assert_quiescent_locked()
            if self._checkpoint_gate_active:
                raise RuntimeError("a Policy checkpoint gate is already active")
            if self._update_gate_owner is not None:
                raise RuntimeError("cannot checkpoint during a Policy update gate")
            self._checkpoint_gate_active = True
        try:
            yield PilotRolloutBarrier(
                asynchronous_staleness_steps=0,
                outstanding_rollout_count=0,
            )
        finally:
            with self._lock:
                self._checkpoint_gate_active = False

    def _assert_quiescent_locked(self) -> None:
        if self._active:
            raise RuntimeError(
                "Policy Pilot checkpoint requires no outstanding batches: "
                f"{tuple(sorted(self._active))!r}"
            )
        observation_counts = self.observation_store.resource_counts()
        orphan_counts = {
            "focus_ledger_entries": self.focus_execution_ledger.entry_count(),
            "observation_records": observation_counts.records,
            "replay_records": observation_counts.replays,
            "observation_tensors": observation_counts.tensors,
            "behavior_traces": self.behavior_store.record_count(),
        }
        nonzero = {name: count for name, count in orphan_counts.items() if count}
        if nonzero:
            raise RuntimeError(
                "Policy Pilot checkpoint found manager-owned orphan evidence: "
                f"{nonzero!r}"
            )

    def _begin_consumer(
        self, batch: PolicyBatchLifecycle, milestone: PolicyBatchMilestone
    ) -> None:
        with self._lock, batch._lock:
            if self._active.get(batch.batch_id) is not batch:
                raise RuntimeError("Policy batch is not open in this manager")
            batch._assert_open_locked()
            batch._validate_dependencies_locked(milestone)
            if milestone is PolicyBatchMilestone.OPTIMIZER_STEP:
                others = tuple(name for name in self._active if name != batch.batch_id)
                if others:
                    raise RuntimeError(
                        "conservative Policy optimizer gate requires no other open "
                        f"batch; outstanding={others!r}"
                    )
                if self._update_gate_owner is not None:
                    raise RuntimeError("another Policy update gate is active")
                self._update_gate_owner = batch.batch_id
            elif milestone is PolicyBatchMilestone.ZERO_STALENESS_BARRIER:
                if self._update_gate_owner != batch.batch_id:
                    raise RuntimeError(
                        "zero-staleness barrier must immediately commit the owned "
                        "optimizer gate"
                    )
            batch._inflight.add(milestone)
            batch._active_consumers += 1

    def _end_consumer(
        self,
        batch: PolicyBatchLifecycle,
        milestone: PolicyBatchMilestone,
        *,
        succeeded: bool,
    ) -> None:
        with self._lock, batch._lock:
            if batch._active_consumers <= 0:
                raise RuntimeError("Policy batch consumer count underflow")
            batch._active_consumers -= 1
            if milestone not in batch._inflight:
                raise RuntimeError("Policy batch milestone inflight state was lost")
            batch._inflight.remove(milestone)
            if succeeded:
                batch._completed.add(milestone)
            if milestone is PolicyBatchMilestone.OPTIMIZER_STEP and not succeeded:
                # The optimizer may have mutated an arbitrary subset of parameters
                # before raising.  Treat its commit status as unknowable: retain the
                # update gate and the owned evidence permanently rather than permit
                # abort, checkpoint, or another batch on a potentially torn policy.
                batch._state = PolicyBatchState.COMMIT_UNKNOWN
            if milestone is PolicyBatchMilestone.ZERO_STALENESS_BARRIER:
                if succeeded and self._update_gate_owner == batch.batch_id:
                    self._update_gate_owner = None

    def _finish(
        self, batch: PolicyBatchLifecycle, *, aborted: bool
    ) -> PolicyBatchReleaseReport:
        with self._lock, batch._lock:
            if batch._state in {PolicyBatchState.CLOSED, PolicyBatchState.ABORTED}:
                assert batch._release_report is not None
                return batch._release_report
            if batch._state is PolicyBatchState.CLEANUP_FAILED:
                raise RuntimeError(
                    "Policy batch is terminal cleanup_failed; checkpoint and "
                    "continued execution are blocked"
                )
            if batch._state is PolicyBatchState.COMMIT_UNKNOWN:
                raise RuntimeError(
                    "Policy batch is terminal commit_unknown after an optimizer "
                    "exception; abort, checkpoint, and continued execution are blocked"
                )
            if self._active.get(batch.batch_id) is not batch:
                raise RuntimeError("Policy batch is not owned by this manager")
            if any(
                self._trajectory_owner.get(identity) != batch.batch_id
                for identity in batch.trajectory_ids
            ):
                raise RuntimeError("Policy batch trajectory ownership changed")
            if batch._active_consumers:
                raise RuntimeError(
                    "cannot release a Policy batch while a consumer is active"
                )
            optimizer_committed = (
                PolicyBatchMilestone.OPTIMIZER_STEP in batch._completed
            )
            zero_staleness_committed = (
                PolicyBatchMilestone.ZERO_STALENESS_BARRIER in batch._completed
            )
            if aborted and optimizer_committed and not zero_staleness_committed:
                raise RuntimeError(
                    "cannot abort after a successful optimizer step; the owned "
                    "zero-staleness barrier must complete"
                )
            if not aborted:
                missing = _REQUIRED_MILESTONES - batch._completed
                if missing:
                    names = tuple(sorted(value.value for value in missing))
                    raise RuntimeError(
                        "cannot close Policy batch before the complete update "
                        f"barrier; missing={names!r}"
                    )
                if not batch._data_protos:
                    raise RuntimeError(
                        "cannot close Policy batch without registered DataProto sidecars"
                    )

            # Preflight every potentially rejecting operation before mutation.
            self.focus_execution_ledger.assert_releasable(batch.trajectory_ids)
            self.observation_store.prepare_release_trajectories(batch.trajectory_ids)
            for data in batch._data_protos:
                _validate_data_proto_sidecars(data)
            if any(state.released for state in batch._transient_states):
                raise RuntimeError("Policy batch transient state was released early")

            errors: list[tuple[str, BaseException]] = []

            def attempt(name: str, operation, default):
                try:
                    return operation()
                except BaseException as error:
                    errors.append((name, error))
                    return default

            ledger_entries = attempt(
                "focus_execution_ledger",
                lambda: self.focus_execution_ledger.release_trajectories(
                    batch.trajectory_ids
                ),
                0,
            )
            observation_counts = attempt(
                "observation_store",
                lambda: self.observation_store.release_trajectories(
                    batch.trajectory_ids
                ),
                ObservationReleaseCounts(records=0, replays=0, tensors=0),
            )
            behavior_traces = attempt(
                "behavior_store",
                lambda: self.behavior_store.release_trajectories(batch.trajectory_ids),
                0,
            )
            sidecar_fields = sum(
                attempt(
                    "data_proto_sidecars",
                    lambda data=data: _release_data_proto_sidecars(data),
                    0,
                )
                for data in batch._data_protos
            )
            transient_owner_count = len(batch._transient_states)
            for state in batch._transient_states:
                attempt("transient_state", state.release, False)
            if self._update_gate_owner == batch.batch_id:
                self._update_gate_owner = None

            if errors:
                batch._state = PolicyBatchState.CLEANUP_FAILED
                batch._cleanup_errors = tuple(
                    f"{name}: {type(error).__name__}: {error}" for name, error in errors
                )
                names = tuple(name for name, _error in errors)
                raise RuntimeError(
                    "Policy batch cleanup failed after exhaustive release attempts; "
                    f"components={names!r}; checkpoint is blocked"
                ) from errors[0][1]

            state = PolicyBatchState.ABORTED if aborted else PolicyBatchState.CLOSED
            report = PolicyBatchReleaseReport(
                batch_id=batch.batch_id,
                state=state,
                ledger_entries=ledger_entries,
                observation_records=observation_counts.records,
                replay_records=observation_counts.replays,
                tensors=observation_counts.tensors,
                behavior_traces=behavior_traces,
                data_proto_sidecar_fields=sidecar_fields,
                transient_owners=transient_owner_count,
            )
            batch._state = state
            batch._release_report = report
            batch._data_protos.clear()
            batch._transient_states.clear()
            del self._active[batch.batch_id]
            for identity in batch.trajectory_ids:
                del self._trajectory_owner[identity]
            return report


def _validate_data_proto_sidecars(data: object) -> None:
    from tgvf_rl.framework.verl.data_bridge import DataProtoPayload

    if not isinstance(data, DataProtoPayload):
        raise TypeError("batch sidecar owner must be project DataProtoPayload")
    data.assert_sidecars_available()


def _release_data_proto_sidecars(data: DataProtoPayload) -> int:
    count = len(data.non_tensor_batch)
    return count if data.release_sidecars() else 0


def _trajectory_id_set(trajectory_ids: tuple[str, ...]) -> tuple[str, ...]:
    identities = tuple(trajectory_ids)
    if not identities or any(
        not isinstance(identity, str) or not identity for identity in identities
    ):
        raise ValueError("trajectory_ids must contain non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("trajectory_ids must be unique")
    return identities


__all__ = [
    "PolicyBatchLifecycle",
    "PolicyBatchLifecycleManager",
    "PolicyBatchMilestone",
    "PolicyBatchReleaseReport",
    "PolicyBatchState",
    "PolicyBatchTransientState",
]
