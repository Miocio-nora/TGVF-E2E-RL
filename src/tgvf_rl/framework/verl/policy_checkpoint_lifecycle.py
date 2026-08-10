"""Restart-stable retention for exact paired Policy checkpoints.

veRL's FSDP checkpoint manager tracks retained paths only in process memory.
That is sufficient inside one uninterrupted process, but a clean restart loses
the list and can accumulate one full model/optimizer checkpoint per restart.
This module makes the project-owned paired commit marker the durable source of
truth: only complete checkpoints for the exact run identity are considered for
deletion, and the newest committed generation remains available until its
successor and tracker have both been committed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.policy.checkpoint import (
    PilotProjectCheckpointState,
    PilotRunIdentityHashes,
)

from .checkpoint_bridge import (
    PolicyPilotVerlCheckpointPair,
    read_committed_policy_checkpoint_pair,
)
from .compatibility import FSDP2BridgeConfig


POLICY_CHECKPOINT_LIFECYCLE_SCHEMA = "tgvf.prl15-checkpoint-lifecycle.v1"
POLICY_PERMANENT_CHECKPOINT_RECEIPT_SCHEMA = (
    "tgvf.prl15-permanent-checkpoint-receipt.v1"
)
POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME = "tgvf_permanent_checkpoint_receipt.json"
_CHECKPOINT_NAME = re.compile(r"global_step_([1-9][0-9]*)\Z")


@dataclass(frozen=True, slots=True)
class CommittedPolicyCheckpoint:
    """One physically complete checkpoint with an integrity-checked pair."""

    optimizer_step: int
    path: Path
    state: PilotProjectCheckpointState
    pair: PolicyPilotVerlCheckpointPair


@dataclass(frozen=True, slots=True)
class PolicyCheckpointLifecycle:
    """Cross-process rolling retention plus immutable permanent checkpoints."""

    checkpoint_root: Path
    maximum_checkpoints_to_keep: int
    checkpoint_steps: tuple[int, ...]
    every_completed_step: bool
    permanent_steps: tuple[int, ...]
    permanent_root: Path | None
    world_size: int
    run_identity: PilotRunIdentityHashes
    fsdp2: FSDP2BridgeConfig

    def __post_init__(self) -> None:
        root = Path(self.checkpoint_root)
        object.__setattr__(self, "checkpoint_root", root)
        if not root.is_absolute():
            raise ValueError("Policy checkpoint root must be absolute")
        if (
            type(self.maximum_checkpoints_to_keep) is not int
            or self.maximum_checkpoints_to_keep <= 0
        ):
            raise ValueError("Policy rolling checkpoint retention must be positive")
        steps = tuple(self.checkpoint_steps)
        object.__setattr__(self, "checkpoint_steps", steps)
        if (
            not steps
            or steps[0] != 0
            or any(type(step) is not int for step in steps)
            or tuple(sorted(set(steps))) != steps
        ):
            raise ValueError("Policy checkpoint steps must increase from zero")
        if type(self.every_completed_step) is not bool:
            raise TypeError("every_completed_step must be bool")
        if self.every_completed_step and steps != tuple(range(steps[-1] + 1)):
            raise ValueError("every-step Policy retention has a checkpoint gap")
        permanent = tuple(self.permanent_steps)
        object.__setattr__(self, "permanent_steps", permanent)
        if (
            any(type(step) is not int or step <= 0 for step in permanent)
            or tuple(sorted(set(permanent))) != permanent
            or any(step not in steps for step in permanent)
        ):
            raise ValueError("permanent Policy steps must be saved positive steps")
        if permanent:
            if self.permanent_root is None:
                raise ValueError("permanent Policy steps require a destination")
            permanent_root = Path(self.permanent_root)
            if not permanent_root.is_absolute():
                raise ValueError("permanent Policy checkpoint root must be absolute")
            if permanent_root.parent != root.parent:
                raise ValueError(
                    "permanent Policy checkpoints must share the output filesystem"
                )
            object.__setattr__(self, "permanent_root", permanent_root)
        elif self.permanent_root is not None:
            raise ValueError("unused permanent Policy checkpoint root must be absent")
        if type(self.world_size) is not int or self.world_size < 2:
            raise ValueError("Policy checkpoint world size must be at least two")
        if self.fsdp2.world_size != self.world_size:
            raise ValueError("Policy checkpoint lifecycle and FSDP world sizes differ")
        if not isinstance(self.run_identity, PilotRunIdentityHashes):
            raise TypeError("Policy checkpoint lifecycle requires a run identity")

    def prepare_for_save(self, optimizer_step: int) -> None:
        """Bound disk use before writing while retaining the current tracker."""

        self._require_configured_step(optimizer_step)
        generations = self.scan_committed()
        # The upstream save and tracker commit happen before
        # ``finalize_saved_checkpoint`` publishes a permanent hard-link tree.
        # A process death in that narrow window must not let the next rolling
        # prune delete the only committed copy of a permanent milestone.
        self._reconcile_visible_permanent_generations(generations)
        if any(item.optimizer_step >= optimizer_step for item in generations):
            raise RuntimeError(
                "Policy checkpoint destination already has this or a future generation"
            )
        # Match veRL's safe pre-save rule: with keep=1, retain the current
        # checkpoint until its successor is committed.  With keep>=2, reserve
        # exactly one slot for the generation about to be written.
        if self.maximum_checkpoints_to_keep > 1:
            self._prune_to_limit(self.maximum_checkpoints_to_keep - 1)

    def finalize_saved_checkpoint(
        self, optimizer_step: int
    ) -> CommittedPolicyCheckpoint:
        """Validate, permanently retain if requested, then finish rotation."""

        self._require_configured_step(optimizer_step)
        tracker_step = self._tracker_step()
        if tracker_step != optimizer_step:
            raise RuntimeError(
                "Policy checkpoint tracker does not name the just-saved generation"
            )
        generation = self._load_generation(
            self.checkpoint_root / f"global_step_{optimizer_step}"
        )
        if optimizer_step in self.permanent_steps:
            self._retain_permanently(generation)
        self._prune_to_limit(self.maximum_checkpoints_to_keep)
        return generation

    def scan_committed(self) -> tuple[CommittedPolicyCheckpoint, ...]:
        """Return only complete generations belonging to this exact run."""

        if not self.checkpoint_root.is_dir():
            return ()
        committed: list[CommittedPolicyCheckpoint] = []
        for path in self.checkpoint_root.iterdir():
            if not path.is_dir() or _CHECKPOINT_NAME.fullmatch(path.name) is None:
                continue
            try:
                committed.append(self._load_generation(path))
            except (
                IdentityMismatchError,
                ReplayMismatchError,
                OSError,
                TypeError,
                ValueError,
            ):
                # Unknown, foreign, partial, or corrupt directories are never
                # deletion candidates.  Resume will still fail closed if its
                # tracker explicitly selects one of them.
                continue
        return tuple(sorted(committed, key=lambda item: item.optimizer_step))

    def permanent_checkpoint(self, optimizer_step: int) -> Path | None:
        if optimizer_step not in self.permanent_steps:
            return None
        assert self.permanent_root is not None
        return self.permanent_root / f"global_step_{optimizer_step}"

    def _reconcile_visible_permanent_generations(
        self, generations: Sequence[CommittedPolicyCheckpoint]
    ) -> None:
        """Finish permanent publication interrupted after an upstream commit."""

        if not self.permanent_steps:
            return
        for generation in generations:
            if generation.optimizer_step in self.permanent_steps:
                self._retain_permanently(generation)

    def _require_configured_step(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step <= 0:
            raise ValueError("Policy checkpoint optimizer step must be positive")
        if optimizer_step not in self.checkpoint_steps:
            raise ValueError("Policy checkpoint step is absent from its fixed schedule")

    def _load_generation(self, path: Path) -> CommittedPolicyCheckpoint:
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None or path.parent != self.checkpoint_root:
            raise ValueError("Policy checkpoint is outside its rolling root")
        optimizer_step = int(match.group(1))
        actor = path / "actor"
        state, pair = read_committed_policy_checkpoint_pair(actor, fsdp2=self.fsdp2)
        if state.run_identity != self.run_identity:
            raise IdentityMismatchError(
                "retained Policy checkpoint belongs to another run identity"
            )
        if (
            state.progress.optimizer_step != optimizer_step
            or pair.optimizer_step != optimizer_step
        ):
            raise IdentityMismatchError(
                "retained Policy checkpoint path and paired step differ"
            )
        _require_nonempty_file(path / "data.pt", "Policy dataloader checkpoint")
        _require_nonempty_file(actor / "fsdp_config.json", "Policy FSDP config")
        for rank in range(self.world_size):
            for prefix, owner in (
                ("model", "model shard"),
                ("optim", "optimizer shard"),
                ("extra_state", "extra-state shard"),
            ):
                _require_nonempty_file(
                    actor / f"{prefix}_world_size_{self.world_size}_rank_{rank}.pt",
                    f"Policy {owner}",
                )
        return CommittedPolicyCheckpoint(optimizer_step, path, state, pair)

    def _prune_to_limit(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Policy pre/post-save retention limit must be positive")
        generations = list(self.scan_committed())
        if len(generations) <= limit:
            return
        tracker_step = self._tracker_step(optional=True)
        if tracker_step is None:
            raise RuntimeError(
                "committed Policy checkpoints exist without a tracker; refusing prune"
            )
        if tracker_step is not None and tracker_step not in {
            item.optimizer_step for item in generations
        }:
            raise RuntimeError(
                "Policy tracker is not an exact committed generation; refusing prune"
            )
        removable = [
            item for item in generations if item.optimizer_step != tracker_step
        ]
        while len(generations) > limit and removable:
            selected = removable.pop(0)
            shutil.rmtree(selected.path)
            generations.remove(selected)
        if len(generations) > limit:
            raise RuntimeError("Policy retention could not preserve its tracker safely")
        _fsync_directory(self.checkpoint_root)

    def _tracker_step(self, *, optional: bool = False) -> int | None:
        tracker = self.checkpoint_root / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            if optional:
                return None
            raise RuntimeError("Policy checkpoint tracker is missing")
        try:
            value = tracker.read_text(encoding="utf-8").strip()
            step = int(value)
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError("Policy checkpoint tracker is unreadable") from error
        if step <= 0 or str(step) != value:
            raise RuntimeError("Policy checkpoint tracker step is malformed")
        return step

    def _retain_permanently(self, generation: CommittedPolicyCheckpoint) -> Path:
        assert self.permanent_root is not None
        destination = self.permanent_root / generation.path.name
        if destination.exists():
            self._validate_permanent(destination, generation)
            return destination

        self.permanent_root.mkdir(parents=True, exist_ok=True)
        temporary_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{generation.path.name}.",
                suffix=".partial",
                dir=self.permanent_root,
            )
        )
        temporary = temporary_parent / generation.path.name
        try:
            shutil.copytree(generation.path, temporary, copy_function=os.link)
            _write_json(
                temporary / POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME,
                _permanent_receipt(generation),
            )
            self._validate_permanent(temporary, generation, temporary=True)
            try:
                os.rename(temporary, destination)
            except FileExistsError:
                self._validate_permanent(destination, generation)
            else:
                self._validate_permanent(destination, generation)
            _fsync_directory(self.permanent_root)
        finally:
            shutil.rmtree(temporary_parent, ignore_errors=True)
        return destination

    def _validate_permanent(
        self,
        destination: Path,
        source: CommittedPolicyCheckpoint,
        *,
        temporary: bool = False,
    ) -> None:
        assert self.permanent_root is not None
        expected_parent = destination.parent.parent if temporary else destination.parent
        if expected_parent != self.permanent_root:
            raise ValueError("permanent Policy checkpoint is outside its root")
        # Temporarily validate the copied generation against its actual parent;
        # all pair/shard checks remain identical to rolling checkpoint checks.
        copied = _load_generation_at_root(
            destination,
            checkpoint_root=destination.parent,
            world_size=self.world_size,
            run_identity=self.run_identity,
            fsdp2=self.fsdp2,
        )
        if copied.state != source.state or copied.pair != source.pair:
            raise IdentityMismatchError(
                "permanent Policy checkpoint differs from its paired source"
            )
        receipt_path = destination / POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReplayMismatchError(
                "permanent Policy checkpoint receipt is unreadable"
            ) from error
        if receipt != _permanent_receipt(source):
            raise IdentityMismatchError(
                "permanent Policy checkpoint receipt differs from its source"
            )


def policy_checkpoint_lifecycle_from_runtime(
    config: object,
    *,
    run_identity: PilotRunIdentityHashes,
    world_size: int,
) -> PolicyCheckpointLifecycle | None:
    """Construct the optional PRL15 lifecycle from the composed veRL config."""

    rollout = _field(_field(config, "actor_rollout_ref"), "rollout")
    custom = _field(rollout, "custom")
    record = _optional_field(custom, "checkpoint_lifecycle")
    if record is None:
        return None
    mapping = _strict_record(
        record,
        {
            "schema_version",
            "checkpoint_steps",
            "every_completed_step",
            "rolling_retention_across_restarts",
            "rolling_max_checkpoints",
            "permanent_steps",
            "permanent_directory",
        },
    )
    if mapping["schema_version"] != POLICY_CHECKPOINT_LIFECYCLE_SCHEMA:
        raise ValueError("unsupported Policy checkpoint lifecycle schema")
    if mapping["rolling_retention_across_restarts"] is not True:
        raise ValueError("Policy checkpoint lifecycle must scan across restarts")
    checkpoint_steps = _integer_tuple(mapping["checkpoint_steps"], "checkpoint_steps")
    custom_steps = _integer_tuple(
        _field(custom, "checkpoint_steps"), "custom checkpoint_steps"
    )
    if checkpoint_steps != custom_steps:
        raise ValueError("Policy checkpoint lifecycle and runtime schedules differ")
    every_completed_step = mapping["every_completed_step"]
    if type(every_completed_step) is not bool:
        raise TypeError("every_completed_step must be bool")
    trainer = _field(config, "trainer")
    maximum_to_keep = _field(trainer, "max_actor_ckpt_to_keep")
    if maximum_to_keep != mapping["rolling_max_checkpoints"]:
        raise ValueError("Policy runtime and lifecycle retention counts differ")
    root = Path(_field(trainer, "default_local_dir"))
    permanent_value = mapping["permanent_directory"]
    if type(permanent_value) is not str:
        raise TypeError("permanent_directory must be a string")
    permanent_root = Path(permanent_value) if permanent_value else None
    lifecycle = PolicyCheckpointLifecycle(
        checkpoint_root=root,
        maximum_checkpoints_to_keep=maximum_to_keep,
        checkpoint_steps=checkpoint_steps,
        every_completed_step=every_completed_step,
        permanent_steps=_integer_tuple(mapping["permanent_steps"], "permanent_steps"),
        permanent_root=permanent_root,
        world_size=world_size,
        run_identity=run_identity,
        fsdp2=FSDP2BridgeConfig(world_size=world_size, fsdp_size=world_size),
    )
    total_steps = _field(trainer, "total_training_steps")
    if every_completed_step and lifecycle.checkpoint_steps != tuple(
        range(total_steps + 1)
    ):
        raise ValueError(
            "formal Policy checkpoint lifecycle does not cover its horizon"
        )
    return lifecycle


def _load_generation_at_root(
    path: Path,
    *,
    checkpoint_root: Path,
    world_size: int,
    run_identity: PilotRunIdentityHashes,
    fsdp2: FSDP2BridgeConfig,
) -> CommittedPolicyCheckpoint:
    probe = PolicyCheckpointLifecycle(
        checkpoint_root=checkpoint_root,
        maximum_checkpoints_to_keep=1,
        checkpoint_steps=(0, int(path.name.removeprefix("global_step_"))),
        every_completed_step=False,
        permanent_steps=(),
        permanent_root=None,
        world_size=world_size,
        run_identity=run_identity,
        fsdp2=fsdp2,
    )
    return probe._load_generation(path)


def _permanent_receipt(
    generation: CommittedPolicyCheckpoint,
) -> dict[str, object]:
    identity = generation.state.run_identity.to_checkpoint_mapping()
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return {
        "schema_version": POLICY_PERMANENT_CHECKPOINT_RECEIPT_SCHEMA,
        "optimizer_step": generation.optimizer_step,
        "run_identity_sha256": identity_sha256,
        "project_state_sha256": generation.state.integrity_sha256,
        "pair_integrity_sha256": generation.pair.integrity_sha256,
    }


def _require_nonempty_file(path: Path, owner: str) -> None:
    try:
        valid = path.is_file() and path.stat().st_size > 0
    except OSError as error:
        raise ReplayMismatchError(f"{owner} is unreadable") from error
    if not valid:
        raise ReplayMismatchError(f"{owner} is missing or empty")


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"Policy runtime config is missing {name}")
        return value[name]
    selected = getattr(value, name, None)
    if selected is None:
        raise ValueError(f"Policy runtime config is missing {name}")
    return selected


def _optional_field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _strict_record(value: object, expected: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Policy checkpoint lifecycle record must be a mapping")
    if set(value) != expected:
        raise ValueError("Policy checkpoint lifecycle record fields differ")
    return value


def _integer_tuple(value: object, owner: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(type(item) is not int for item in value)
    ):
        raise TypeError(f"{owner} must be an integer sequence")
    return tuple(value)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CommittedPolicyCheckpoint",
    "POLICY_CHECKPOINT_LIFECYCLE_SCHEMA",
    "POLICY_PERMANENT_CHECKPOINT_RECEIPT_FILENAME",
    "POLICY_PERMANENT_CHECKPOINT_RECEIPT_SCHEMA",
    "PolicyCheckpointLifecycle",
    "policy_checkpoint_lifecycle_from_runtime",
]
