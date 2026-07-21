"""Repo-owned lifecycle integration for pinned veRL's v0 training path.

Pinned e003 hard-codes both ``ActorRolloutRefWorker`` and ``RayPPOTrainer`` in
its v0 TaskRunner.  This module preserves that public workflow while replacing
only the two project-owned seams: exact-sidecar cleanup on the worker and the
paired Policy checkpoint lifecycle on the trainer driver.
"""

from __future__ import annotations

from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.environment.focus_runtime import FocusExecutionLedger
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropTGVFObservationRecord,
    FocusedObservationRecord,
)
from tgvf_rl.observations.store import ObservationStore, TrajectoryReplayBundle
from tgvf_rl.policy.checkpoint import (
    DATA_CURSOR_OWNER,
    ROLLOUT_RNG_OWNER,
    ROLLOUT_SAMPLER_OWNER,
    OpaqueProjectState,
    PilotOptimizerDataCursor,
    PilotRunIdentityHashes,
)
from tgvf_rl.policy.lifecycle import PolicyBatchLifecycleManager
from tgvf_rl.policy.metrics import (
    PilotMetricsAccumulator,
    PilotOptimizerStepMetricsObservation,
    PilotTrajectoryMetricsObservation,
)
from tgvf_rl.policy.run_config import (
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.trajectories.behavior import BehaviorTraceStore
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .checkpoint_bridge import (
    PairedPolicyPilotVerlCheckpoint,
    PolicyPilotVerlResumeResult,
)
from .compatibility import FSDP2BridgeConfig
from .data_bridge import (
    make_sidecar_releasing_actor_rollout_ref_worker_class,
    release_verl_data_proto_sidecars,
    validate_data_proto_integrity,
)
from .exact_replay_engine import _operational_base_identity_sha256
from .policy_weight_sync import (
    PolicyWeightSyncState,
    load_latest_policy_version,
)
from .reward_bridge import validate_policy_pilot_reward_data_proto
from tgvf_rl.rewards.verl_adapter import PILOT_VERL_REWARD_COMPONENTS_FIELD
from .rollout_bridge import (
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
)


POLICY_PILOT_TASK_RUNNER_FQN = (
    "tgvf_rl.framework.verl.policy_task_runner.create_policy_pilot_task_runner_class"
)
POLICY_PILOT_TRAINER_LIFECYCLE_SCHEMA = "tgvf-policy-trainer-lifecycle-v1"
POLICY_REFERENCE_DIAGNOSTIC_ENABLED = True


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_state(owner: str, value: object) -> OpaqueProjectState:
    return OpaqueProjectState(owner, "canonical-json-v1", _canonical_json_bytes(value))


def _torch_state(owner: str, value: object) -> OpaqueProjectState:
    stream = io.BytesIO()
    torch.save(value, stream)
    return OpaqueProjectState(owner, "torch-save-v1", stream.getvalue())


def _load_torch_state(value: OpaqueProjectState) -> object:
    if value.codec != "torch-save-v1":
        raise ValueError("data cursor codec differs from the Policy trainer contract")
    return torch.load(io.BytesIO(value.payload), map_location="cpu", weights_only=False)


def _strict_json_state(value: OpaqueProjectState, *, owner: str) -> Mapping[str, object]:
    if value.owner != owner or value.codec != "canonical-json-v1":
        raise ValueError(f"{owner} state identity differs")
    try:
        decoded = json.loads(value.payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{owner} state is malformed") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{owner} state must decode to a mapping")
    return decoded


def _reference_weights_sha256(config: PolicyE2ESmokeRunConfig) -> str:
    # Reference replay and paired checkpointing must name the frozen base with
    # exactly the same operational identity.  Reuse the replay engine's
    # canonical definition instead of maintaining a nearly-identical digest.
    return _operational_base_identity_sha256(config.model)


def _run_identity(config: PolicyE2ESmokeRunConfig) -> PilotRunIdentityHashes:
    return PilotRunIdentityHashes.from_hashes(
        config.run_id,
        {
            "agent_loop_config": config.framework.agent_loop_config_sha256,
            "cap_error": config.protocol.cap_error_sha256,
            "chat_template": config.model.chat_template_sha256,
            "dataset_content": config.dataset.runtime_binding.content_sha256,
            "dataset_iteration": config.dataset.iteration_identity_sha256,
            "dataset_manifest_file": config.dataset.runtime_binding.manifest_file_sha256,
            "dataset_samples": config.dataset.samples_sha256,
            "prompt": config.protocol.prompt_sha256,
            "representation_artifact_file": config.representation.artifact_file_sha256,
            "representation_manifest": config.representation.artifact.sha256,
            "representation_run": config.representation.expected_run_identity_sha256,
            "reward_verifier": config.reward.answer_verifier_sha256,
            "rollout_rng_derivation": config.rollout_rng.derivation_sha256,
            "run_config": config.identity_sha256,
            "run_config_file": config.source_sha256,
            "tool_schema": config.protocol.tool_schema_sha256,
        },
    )


class PolicyPilotTrainerCheckpointState:
    """Driver-owned project state paired with the framework checkpoint."""

    def __init__(self, trainer: object, config: PolicyE2ESmokeRunConfig) -> None:
        self.trainer = trainer
        self.config = config
        self._run_identity = _run_identity(config)
        self._weight_state = PolicyWeightSyncState.from_environment()
        if self._weight_state.run_id != config.run_id or (
            self._weight_state.run_identity_sha256 != config.identity_sha256
        ):
            raise RuntimeError("Policy checkpoint and weight-sync run identities differ")
        self.metrics_accumulator = PilotMetricsAccumulator()
        self.lifecycle_manager = PolicyBatchLifecycleManager(
            observation_store=ObservationStore(),
            behavior_store=BehaviorTraceStore(),
            focus_execution_ledger=FocusExecutionLedger(),
        )
        self._progress = PilotOptimizerDataCursor(
            0,
            _torch_state(DATA_CURSOR_OWNER, self._dataloader_state()),
        )
        self._sampler = self._sampler_state(0)
        self._rng = self._rng_state(0)
        self._prepared_policy: PolicyVersion | None = None
        self.last_resume: PolicyPilotVerlResumeResult | None = None

    @classmethod
    def from_environment(cls, trainer: object) -> "PolicyPilotTrainerCheckpointState":
        path = os.environ.get("TGVF_POLICY_RUN_CONFIG_PATH")
        if not path:
            raise RuntimeError("TGVF_POLICY_RUN_CONFIG_PATH is required by Policy TaskRunner")
        config = load_policy_e2e_smoke_run_config(Path(path))
        expected = os.environ.get("TGVF_POLICY_RUN_IDENTITY_SHA256")
        if expected != config.identity_sha256:
            raise RuntimeError("Policy TaskRunner config/environment identity differs")
        return cls(trainer, config)

    def prepare_checkpoint(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step <= 0:
            raise ValueError("Policy checkpoint optimizer step must be positive")
        if self.metrics_accumulator.state.optimizer_steps != optimizer_step:
            raise RuntimeError("Policy metrics do not reach the checkpoint optimizer step")
        policy = load_latest_policy_version(
            self._weight_state,
            expected_optimizer_step=optimizer_step,
        )
        self._progress = PilotOptimizerDataCursor(
            optimizer_step,
            _torch_state(DATA_CURSOR_OWNER, self._dataloader_state()),
        )
        self._sampler = self._sampler_state(optimizer_step)
        self._rng = self._rng_state(optimizer_step)
        self._prepared_policy = policy

    def record_optimizer_step(self, data: object, *, elapsed_seconds: float) -> None:
        step = getattr(self.trainer, "global_steps", None)
        if type(step) is not int or step <= 0:
            raise RuntimeError("veRL trainer global step is unavailable after update")
        observation = policy_metrics_observation_from_data_proto(
            data,
            optimizer_step=step,
            elapsed_seconds=elapsed_seconds,
        )
        self.metrics_accumulator.record_optimizer_step(observation)

    def run_identity(self) -> PilotRunIdentityHashes:
        return self._run_identity

    def progress(self) -> PilotOptimizerDataCursor:
        return self._progress

    def rollout_sampler_state(self) -> OpaqueProjectState:
        return self._sampler

    def rollout_rng_state(self) -> OpaqueProjectState:
        return self._rng

    def current_policy_version(self) -> PolicyVersion:
        if self._prepared_policy is not None:
            return self._prepared_policy
        # Clean-process resume is intentionally latest-checkpoint-only in this
        # first executable slice.  The strict pair subsequently verifies the
        # loaded optimizer step against this content-addressed snapshot.
        return load_latest_policy_version(self._weight_state)

    def reference_policy_version(self) -> PolicyVersion:
        return PolicyVersion(
            "qwen3-vl-8b-thinking-frozen-reference",
            0,
            _reference_weights_sha256(self.config),
        )

    def restore_progress(self, value: PilotOptimizerDataCursor) -> None:
        state = _load_torch_state(value.data_cursor)
        loader = getattr(self.trainer, "train_dataloader", None)
        restore = getattr(loader, "load_state_dict", None)
        if not callable(restore):
            raise TypeError("veRL train_dataloader must implement load_state_dict()")
        restore(state)
        self._progress = value
        self._prepared_policy = None

    def restore_rollout_sampler_state(self, value: OpaqueProjectState) -> None:
        self._validate_derived_state(value, owner=ROLLOUT_SAMPLER_OWNER)
        self._sampler = value

    def restore_rollout_rng_state(self, value: OpaqueProjectState) -> None:
        self._validate_derived_state(value, owner=ROLLOUT_RNG_OWNER)
        self._rng = value

    def _dataloader_state(self) -> object:
        loader = getattr(self.trainer, "train_dataloader", None)
        state_dict = getattr(loader, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("veRL train_dataloader must implement state_dict()")
        return state_dict()

    def _derived_content(self, step: int) -> dict[str, object]:
        return {
            "schema_version": POLICY_PILOT_TRAINER_LIFECYCLE_SCHEMA,
            "run_identity_sha256": self.config.identity_sha256,
            "optimizer_step": step,
            "master_seed": self.config.rollout_rng.master_seed,
            "derivation_name": self.config.rollout_rng.derivation_name,
            "derivation_sha256": self.config.rollout_rng.derivation_sha256,
            "dataset_iteration_sha256": self.config.dataset.iteration_identity_sha256,
        }

    def _sampler_state(self, step: int) -> OpaqueProjectState:
        return _json_state(ROLLOUT_SAMPLER_OWNER, self._derived_content(step))

    def _rng_state(self, step: int) -> OpaqueProjectState:
        return _json_state(ROLLOUT_RNG_OWNER, self._derived_content(step))

    def _validate_derived_state(self, value: OpaqueProjectState, *, owner: str) -> None:
        decoded = _strict_json_state(value, owner=owner)
        step = decoded.get("optimizer_step")
        if type(step) is not int or decoded != self._derived_content(step):
            raise RuntimeError(f"restored {owner} differs from the fixed run identity")


class PairedActorWorkerGroup:
    """Delegate all actor calls, pairing only public save/load operations."""

    def __init__(self, upstream: object, state: PolicyPilotTrainerCheckpointState) -> None:
        self.upstream = upstream
        self.state = state
        self.paired = PairedPolicyPilotVerlCheckpoint(
            upstream=upstream,
            fsdp2=FSDP2BridgeConfig(
                world_size=state.config.distributed.world_size,
                fsdp_size=state.config.distributed.world_size,
            ),
            lifecycle_manager=state.lifecycle_manager,
            state_port=state,
            metrics_accumulator=state.metrics_accumulator,
        )

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> object:
        self.state.prepare_checkpoint(global_step)
        return self.paired.save_checkpoint(
            local_path,
            hdfs_path,
            global_step,
            max_ckpt_to_keep,
        )

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> object:
        result = self.paired.load_checkpoint(
            local_path,
            hdfs_path,
            del_local_after_load,
        )
        self.state.last_resume = result
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self.upstream, name)


class CheckpointAfterWeightSyncManager:
    """Commit a pending checkpoint only after current-step LoRA publication."""

    def __init__(self, upstream: object, trainer: object) -> None:
        self.upstream = upstream
        self.trainer = trainer

    def update_weights(self, global_steps: int | None = None) -> object:
        result = self.upstream.update_weights(global_steps)
        if getattr(self.trainer, "_policy_checkpoint_pending", False):
            self.upstream.sleep_replicas()
            try:
                self.trainer._commit_policy_checkpoint_after_weight_sync(global_steps)
            finally:
                self.upstream.wake_up_replicas()
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self.upstream, name)


def make_policy_pilot_ray_trainer_class(upstream_trainer_cls: type[Any]) -> type[Any]:
    """Return the pinned v0 trainer with lifecycle/checkpoint hooks installed."""

    if not isinstance(upstream_trainer_cls, type):
        raise TypeError("upstream RayPPOTrainer must be a class")
    for name in ("init_workers", "_get_gen_batch", "_update_actor", "_save_checkpoint"):
        if not callable(getattr(upstream_trainer_cls, name, None)):
            raise TypeError(f"upstream RayPPOTrainer is missing {name}()")

    class PolicyPilotRayPPOTrainer(upstream_trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The Pilot keeps both mathematical KL coefficients at zero, so
            # upstream's ``need_reference_policy`` is intentionally false.
            # We nevertheless execute a frozen-base reference forward as a
            # diagnostic on the exact rollout-recorded observation bundle.
            # The ActorRolloutRef role installed below owns that explicit ref
            # engine; this flag makes the v0 trainer schedule the forward.
            self.use_reference_policy = POLICY_REFERENCE_DIAGNOSTIC_ENABLED

        def init_workers(self):
            result = super().init_workers()
            state = PolicyPilotTrainerCheckpointState.from_environment(self)
            original_actor_wg = self.actor_rollout_wg
            self._policy_checkpoint_state = state
            self.actor_rollout_wg = PairedActorWorkerGroup(original_actor_wg, state)
            if getattr(self, "ref_policy_wg", None) is original_actor_wg:
                self.ref_policy_wg = self.actor_rollout_wg
            self.checkpoint_manager = CheckpointAfterWeightSyncManager(
                self.checkpoint_manager,
                self,
            )
            self._policy_checkpoint_pending = False
            self._policy_step_started_at = None
            return result

        def _get_gen_batch(self, *args, **kwargs):
            self._policy_step_started_at = perf_counter()
            return super()._get_gen_batch(*args, **kwargs)

        def _compute_ref_log_prob(self, batch):
            if not self.ref_in_actor:
                raise RuntimeError(
                    "Policy reference diagnostic requires the colocated "
                    "ActorRolloutRef worker"
                )
            # Upstream interprets ``ref_in_actor`` as "temporarily disable the
            # actor LoRA" and calls the actor engine.  That is insufficient for
            # our exact replay contract because engine role is identified by
            # its forward_only state.  The same worker group also owns an
            # explicit frozen ref engine, so route this synchronous driver call
            # through upstream's dedicated-ref branch and immediately restore
            # the invariant.
            self.ref_in_actor = False
            try:
                return super()._compute_ref_log_prob(batch)
            finally:
                self.ref_in_actor = True

        def _update_actor(self, batch, *args, **kwargs):
            try:
                # Reject transport/reward corruption before the upstream actor
                # can mutate optimizer state.  Metric extraction repeats the
                # reward check after the synchronous update, but that later
                # check is deliberately not the mutation safety boundary.
                validate_data_proto_integrity(batch)
                validate_policy_pilot_reward_data_proto(batch)
                output = super()._update_actor(batch, *args, **kwargs)
                started = self._policy_step_started_at
                elapsed = perf_counter() - started if started is not None else 0.0
                self._policy_checkpoint_state.record_optimizer_step(
                    batch,
                    elapsed_seconds=max(elapsed, 1.0e-12),
                )
                return output
            finally:
                # All current/reference/update consumers are synchronous in the
                # accepted v0 path.  The driver copy can release only after the
                # actor call has returned; every Ray-local copy has its own
                # worker ``finally`` in the wrapped TrainingWorker.
                release_verl_data_proto_sidecars(batch)
                self._policy_step_started_at = None

        def _save_checkpoint(self):
            if self._policy_checkpoint_pending:
                raise RuntimeError("a Policy checkpoint request is already pending")
            self._policy_checkpoint_pending = True

        def _commit_policy_checkpoint_after_weight_sync(self, global_steps):
            if not self._policy_checkpoint_pending:
                raise RuntimeError("no Policy checkpoint is pending")
            if global_steps != self.global_steps:
                raise RuntimeError("weight-sync and trainer checkpoint steps differ")
            try:
                return super(PolicyPilotRayPPOTrainer, self)._save_checkpoint()
            finally:
                self._policy_checkpoint_pending = False

    PolicyPilotRayPPOTrainer.__name__ = "PolicyPilotRayPPOTrainer"
    PolicyPilotRayPPOTrainer.__qualname__ = "PolicyPilotRayPPOTrainer"
    PolicyPilotRayPPOTrainer.__module__ = __name__
    return PolicyPilotRayPPOTrainer


def policy_metrics_observation_from_data_proto(
    data: object,
    *,
    optimizer_step: int,
    elapsed_seconds: float,
) -> PilotOptimizerStepMetricsObservation:
    """Recover the checkpointed raw Pilot metrics from one exact update batch."""

    validate_policy_pilot_reward_data_proto(data)
    batch = getattr(data, "batch")
    non_tensors = getattr(data, "non_tensor_batch")
    trajectories = tuple(non_tensors[TRAJECTORY_PAYLOAD_FIELD])
    replay_bundles = tuple(non_tensors[TRAJECTORY_REPLAY_BUNDLE_FIELD])
    components = tuple(non_tensors[PILOT_VERL_REWARD_COMPONENTS_FIELD])
    response_mask = batch["response_mask"]
    if not isinstance(response_mask, torch.Tensor):
        raise TypeError("Policy metric response_mask must be a tensor")
    if not (
        len(trajectories)
        == len(replay_bundles)
        == len(components)
        == response_mask.shape[0]
    ):
        raise RuntimeError("Policy metric rows differ from the update batch")

    rows: list[PilotTrajectoryMetricsObservation] = []
    for index, (trajectory, bundle, raw_components) in enumerate(
        zip(trajectories, replay_bundles, components, strict=True)
    ):
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("Policy metric trajectory sidecar is invalid")
        if not isinstance(bundle, TrajectoryReplayBundle):
            raise TypeError("Policy metric replay bundle sidecar is invalid")
        try:
            reward = {str(name): float(value) for name, value in raw_components}
        except (TypeError, ValueError) as error:
            raise TypeError("Policy reward component sidecar is malformed") from error
        if set(reward) != {
            "answer_reward",
            "format_reward",
            "conditional_tool_reward",
        }:
            raise ValueError("Policy metric reward components differ")
        original_visual = len(bundle.replay_record.source_visual.positions)
        tool_visual = sum(_observation_visual_tokens(item) for item in bundle.observation_records)
        reasoning = sum(
            0 if turn.think_span is None else turn.think_span.end - turn.think_span.start
            for turn in trajectory.assistant_turns
        )
        successful = len(trajectory.observations)
        errors = tuple(item.code for item in trajectory.tool_errors)
        rows.append(
            PilotTrajectoryMetricsObservation(
                prompt_id=trajectory.identity.group_id,
                trajectory_id=trajectory.identity.canonical_id,
                generated_policy_tokens=int(response_mask[index].sum().item()),
                successful_tgvf_observations=successful,
                tool_call_attempts=successful + len(errors),
                answer_reward=reward["answer_reward"],
                format_error=reward["format_reward"] == -1.0,
                conditional_tool_reward=reward["conditional_tool_reward"],
                reasoning_tokens=reasoning,
                original_visual_tokens=original_visual,
                total_visual_tokens=original_visual + tool_visual,
                tool_error_codes=errors,
            )
        )
    return PilotOptimizerStepMetricsObservation(
        optimizer_step=optimizer_step,
        step_time_seconds=elapsed_seconds,
        trajectories=tuple(rows),
    )


def _observation_visual_tokens(value: object) -> int:
    if isinstance(value, (FocusedObservationRecord, CropTGVFObservationRecord)):
        return len(value.layout.d_positions)
    if isinstance(value, CropObservationRecord):
        return len(value.crop_visual.positions)
    raise TypeError("unknown exact visual observation record")


def add_policy_actor_rollout_worker(
    runner: object,
    config: object,
    *,
    ray_module: object,
    role_type: type[Any],
    actor_worker_cls: type[Any],
    ray_worker_group_cls: type[Any],
    need_reference_policy_fn: Any,
) -> tuple[type[Any], type[Any]]:
    """Publicly testable worker-map seam used by the repo-owned TaskRunner."""

    wrapped = make_sidecar_releasing_actor_rollout_ref_worker_class(actor_worker_cls)
    if need_reference_policy_fn(config):
        raise ValueError(
            "Policy reference diagnostic requires zero-weight KL configuration"
        )
    # Force the combined worker even though upstream's objective-driven helper
    # returns false.  Its reference engine is frozen/base-only; the trainer
    # override schedules it solely for the recorded KL diagnostic.
    role = role_type.ActorRolloutRef
    runner.role_worker_mapping[role] = ray_module.remote(wrapped)
    runner.mapping[role] = "global_pool"
    return wrapped, ray_worker_group_cls


def create_policy_pilot_task_runner_class() -> object:
    """Create the Ray-remote repo TaskRunner on the pinned public v0 surface."""

    import ray
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    from verl.trainer.ppo.utils import need_reference_policy
    from verl.workers.engine_workers import ActorRolloutRefWorker

    trainer_cls = make_policy_pilot_ray_trainer_class(RayPPOTrainer)

    class PolicyPilotTaskRunner(BaseTaskRunner):
        def add_actor_rollout_worker(self, config):
            return add_policy_actor_rollout_worker(
                self,
                config,
                ray_module=ray,
                role_type=Role,
                actor_worker_cls=ActorRolloutRefWorker,
                ray_worker_group_cls=RayWorkerGroup,
                need_reference_policy_fn=need_reference_policy,
            )

        def run(self, config):
            return run_policy_pilot_v0_task(self, config, trainer_cls=trainer_cls)

    PolicyPilotTaskRunner.__name__ = "PolicyPilotTaskRunner"
    PolicyPilotTaskRunner.__qualname__ = "PolicyPilotTaskRunner"
    PolicyPilotTaskRunner.__module__ = __name__
    return ray.remote(PolicyPilotTaskRunner)


def run_policy_pilot_v0_task(runner: object, config: object, *, trainer_cls: type[Any]) -> None:
    """Pinned TaskRunner.run with only the trainer class made project-owned."""

    import socket
    from pprint import pprint

    from omegaconf import OmegaConf
    from verl.trainer.ppo.utils import (
        create_rl_dataset,
        create_rl_sampler,
        need_critic,
    )
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.config import validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    print(f"PolicyPilotTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(config)
    runner.add_critic_worker(config)
    runner.add_reward_model_resource_pool(config)
    runner.add_teacher_model_resource_pool(config)
    runner.add_ref_policy_worker(config, actor_rollout_cls)
    validate_config(
        config=config,
        use_reference_policy=POLICY_REFERENCE_DIAGNOSTIC_ENABLED,
        use_critic=need_critic(config),
    )
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(
        local_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    resource_pool_manager = runner.init_resource_pool_mgr(config)
    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=config.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)
    trainer = trainer_cls(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
    trainer.init_workers()
    trainer.fit()


__all__ = [
    "CheckpointAfterWeightSyncManager",
    "POLICY_PILOT_TASK_RUNNER_FQN",
    "POLICY_REFERENCE_DIAGNOSTIC_ENABLED",
    "PairedActorWorkerGroup",
    "PolicyPilotTrainerCheckpointState",
    "add_policy_actor_rollout_worker",
    "create_policy_pilot_task_runner_class",
    "make_policy_pilot_ray_trainer_class",
    "policy_metrics_observation_from_data_proto",
    "run_policy_pilot_v0_task",
]
