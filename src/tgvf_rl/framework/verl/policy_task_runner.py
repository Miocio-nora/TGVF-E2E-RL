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
    PilotMetricsSummary,
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
from .padding_compat import install_verl_sdpa_padding_compat
from .policy_weight_sync import (
    PolicyWeightSyncState,
    load_latest_policy_version,
)
from .reward_bridge import validate_policy_pilot_reward_data_proto
from tgvf_rl.rewards.verl_adapter import (
    PILOT_VERL_JUDGE_USAGE_FIELD,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
)
from .rollout_bridge import (
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
)


POLICY_PILOT_TASK_RUNNER_FQN = (
    "tgvf_rl.framework.verl.policy_task_runner.create_policy_pilot_task_runner_class"
)
POLICY_PILOT_TRAINER_LIFECYCLE_SCHEMA = "tgvf-policy-trainer-lifecycle-v1"
POLICY_REFERENCE_DIAGNOSTIC_ENABLED = True
POLICY_PILOT_METRICS_EVENT_SCHEMA = "policy-pilot-v1-metrics-event-v1"
POLICY_TRACKING_METRIC_NAMES = frozenset(
    {
        "training/global_step",
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/lr",
        "actor/pg_clipfrac",
        "actor/behavior_current_log_ratio_abs_mean",
        "actor/perf/max_memory_allocated_gb",
        "critic/rewards/mean",
        "policy_pilot/generated_policy_tokens",
        "policy_pilot/successful_tgvf_observations",
        "policy_pilot/tool_call_attempt_rate",
        "policy_pilot/mean_tool_call_attempts",
        "policy_pilot/mean_answer_reward",
        "policy_pilot/format_error_rate",
        "policy_pilot/mean_conditional_tool_reward",
        "policy_pilot/mean_reasoning_length",
        "policy_pilot/mean_original_visual_tokens",
        "policy_pilot/mean_total_visual_tokens",
        "policy_pilot/judge_calls",
        "policy_pilot/judge_prompt_tokens",
        "policy_pilot/judge_completion_tokens",
        "policy_pilot/judge_cost_usd",
        "policy_timing/end_to_end_step_seconds",
        "perf/throughput",
        "response_length/mean",
        "num_turns/mean",
    }
)


def _finish_tracking_backends(tracker: object) -> None:
    """Finish veRL loggers before Ray tears down the TaskRunner process."""

    backends = getattr(tracker, "logger", None)
    if not isinstance(backends, dict):
        raise TypeError("veRL Tracking.logger must be a dictionary")
    errors: list[Exception] = []
    for name, backend in tuple(backends.items()):
        finish = getattr(backend, "finish", None)
        if not callable(finish):
            continue
        try:
            if name in {"wandb", "vemlp_wandb"}:
                finish(exit_code=0)
            else:
                finish()
        except Exception as error:  # pragma: no cover - external logger failure
            errors.append(error)
    # Tracking.__del__ otherwise retries W&B finish during interpreter/Ray
    # teardown, after W&B's service socket may already have been closed.
    backends.clear()
    if errors:
        raise ExceptionGroup("Policy tracking shutdown failed", errors)


def _zero_safe_mean(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _pilot_metrics_event(
    observation: PilotOptimizerStepMetricsObservation,
    summary: PilotMetricsSummary,
) -> dict[str, object]:
    rows = observation.trajectories
    trajectories = len(rows)
    attempts = sum(row.tool_call_attempts for row in rows)
    successful = sum(row.successful_tgvf_observations for row in rows)
    errors: dict[str, int] = {}
    for row in rows:
        for code in row.tool_error_codes:
            errors[code] = errors.get(code, 0) + 1
    step = {
        "prompts": observation.prompt_count,
        "trajectories": trajectories,
        "generated_policy_tokens": sum(row.generated_policy_tokens for row in rows),
        "successful_tgvf_observations": successful,
        "tool_call_attempts": attempts,
        "tool_call_attempt_rate": _zero_safe_mean(
            sum(row.tool_call_attempts > 0 for row in rows), trajectories
        ),
        "mean_tool_call_attempts": _zero_safe_mean(attempts, trajectories),
        "mean_answer_reward": _zero_safe_mean(
            sum(int(row.answer_reward) for row in rows), trajectories
        ),
        "format_error_rate": _zero_safe_mean(
            sum(row.format_error for row in rows), trajectories
        ),
        "mean_conditional_tool_reward": _zero_safe_mean(
            sum(int(row.conditional_tool_reward) for row in rows), trajectories
        ),
        "mean_reasoning_length": _zero_safe_mean(
            sum(row.reasoning_tokens for row in rows), trajectories
        ),
        "mean_original_visual_tokens": _zero_safe_mean(
            sum(row.original_visual_tokens for row in rows), trajectories
        ),
        "mean_total_visual_tokens": _zero_safe_mean(
            sum(row.total_visual_tokens for row in rows), trajectories
        ),
        "judge_calls": sum(row.judge_calls for row in rows),
        "judge_prompt_tokens": sum(row.judge_prompt_tokens for row in rows),
        "judge_completion_tokens": sum(
            row.judge_completion_tokens for row in rows
        ),
        "judge_cost_usd": sum(row.judge_cost_usd for row in rows),
        "pre_publication_elapsed_seconds": observation.step_time_seconds,
        "tool_error_counts": dict(sorted(errors.items())),
    }
    cumulative = {
        "optimizer_steps": summary.optimizer_steps,
        "prompts": summary.prompts,
        "trajectories": summary.trajectories,
        "generated_policy_tokens": summary.generated_policy_tokens,
        "successful_tgvf_observations": summary.successful_tgvf_observations,
        "tool_call_attempt_rate": summary.tool_call_attempt_rate,
        "mean_tool_call_attempts": summary.mean_tool_call_attempts,
        "mean_answer_reward": summary.mean_answer_reward,
        "format_error_rate": summary.format_error_rate,
        "mean_conditional_tool_reward": summary.mean_conditional_tool_reward,
        "mean_reasoning_length": summary.mean_reasoning_length,
        "mean_original_visual_tokens": summary.mean_original_visual_tokens,
        "mean_total_visual_tokens": summary.mean_total_visual_tokens,
        "judge_calls": summary.judge_calls,
        "judge_prompt_tokens": summary.judge_prompt_tokens,
        "judge_completion_tokens": summary.judge_completion_tokens,
        "judge_cost_usd": summary.judge_cost_usd,
        "tool_error_counts": {
            item.code: item.count for item in summary.tool_error_counts
        },
    }
    return {
        "schema_version": POLICY_PILOT_METRICS_EVENT_SCHEMA,
        "optimizer_step": observation.optimizer_step,
        "step": step,
        "cumulative": cumulative,
    }


def _wandb_metrics_from_event(event: Mapping[str, object]) -> dict[str, float | int]:
    step = event.get("step")
    cumulative = event.get("cumulative")
    if not isinstance(step, Mapping) or not isinstance(cumulative, Mapping):
        raise TypeError("Policy metrics event is missing step/cumulative mappings")
    result: dict[str, float | int] = {}
    for prefix, values in (("policy_pilot", step), ("policy_pilot_total", cumulative)):
        for name, value in values.items():
            if name == "tool_error_counts":
                if not isinstance(value, Mapping):
                    raise TypeError("Policy tool error counts must be a mapping")
                for code, count in value.items():
                    result[f"{prefix}/tool_errors/{code}"] = int(count)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{prefix}/{name}"] = value
    return result


def _policy_tracking_metrics(data: Mapping[str, object]) -> dict[str, object]:
    """Keep the compact operator-facing metric surface for console and W&B."""

    if not isinstance(data, Mapping):
        raise TypeError("Policy tracking data must be a mapping")
    return {
        name: value
        for name, value in data.items()
        if name in POLICY_TRACKING_METRIC_NAMES
    }


def _append_policy_metrics_event(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(event, sort_keys=True))
    step = normalized.get("optimizer_step")
    if type(step) is not int or step <= 0:
        raise ValueError("Policy metrics event optimizer step must be positive")
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            previous = json.loads(lines[-1])
            previous_step = previous.get("optimizer_step")
            if previous_step == step:
                if previous != normalized:
                    raise RuntimeError("Policy metrics replay changed an existing step")
                return
            if type(previous_step) is not int or previous_step + 1 != step:
                raise RuntimeError("Policy metrics steps must be contiguous")
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _completed_resume_checkpoint_step(
    trainer: object,
    *,
    latest_checkpoint_resolver: object | None = None,
) -> int | None:
    """Return an already-complete resume step without mutating trainer state."""

    config = getattr(getattr(trainer, "config", None), "trainer", None)
    if config is None:
        raise TypeError("Policy trainer config is unavailable")
    mode = getattr(config, "resume_mode", None)
    if mode == "disable":
        return None
    if mode not in {"auto", "resume_path"}:
        raise ValueError("unsupported Policy trainer resume_mode")
    if getattr(config, "default_hdfs_dir", None) is not None:
        return None  # Let pinned veRL retain ownership of its explicit error.

    checkpoint_path: str | None
    if mode == "auto":
        resolver = latest_checkpoint_resolver
        if resolver is None:
            from verl.utils.checkpoint.checkpoint_manager import (
                find_latest_ckpt_path,
            )

            resolver = find_latest_ckpt_path
        if not callable(resolver):
            raise TypeError("latest checkpoint resolver must be callable")
        checkpoint_path = resolver(getattr(config, "default_local_dir", None))
    else:
        raw_path = getattr(config, "resume_from_path", None)
        checkpoint_path = raw_path if isinstance(raw_path, str) else None
    if checkpoint_path is None:
        return None

    resolved = Path(checkpoint_path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.is_dir():
        return None  # Pinned veRL will produce the canonical missing-path error.
    prefix = "global_step_"
    if not resolved.name.startswith(prefix):
        raise RuntimeError("Policy resume checkpoint name does not encode a step")
    try:
        checkpoint_step = int(resolved.name.removeprefix(prefix))
    except ValueError as error:
        raise RuntimeError("Policy resume checkpoint step is malformed") from error

    total_steps = getattr(trainer, "total_training_steps", None)
    if type(total_steps) is not int or total_steps <= 0:
        raise TypeError("Policy trainer total_training_steps must be positive")
    if checkpoint_step > total_steps:
        raise RuntimeError("Policy resume checkpoint exceeds configured total steps")
    return checkpoint_step if checkpoint_step == total_steps else None


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

    def record_optimizer_step(
        self, data: object, *, elapsed_seconds: float
    ) -> tuple[PilotOptimizerStepMetricsObservation, PilotMetricsSummary]:
        step = getattr(self.trainer, "global_steps", None)
        if type(step) is not int or step <= 0:
            raise RuntimeError("veRL trainer global step is unavailable after update")
        observation = policy_metrics_observation_from_data_proto(
            data,
            optimizer_step=step,
            elapsed_seconds=elapsed_seconds,
        )
        summary = self.metrics_accumulator.record_optimizer_step(observation)
        return observation, summary

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
        sync_started = perf_counter()
        result = self.upstream.update_weights(global_steps)
        sync_seconds = perf_counter() - sync_started
        checkpoint_seconds = 0.0
        if getattr(self.trainer, "_policy_checkpoint_pending", False):
            checkpoint_started = perf_counter()
            self.upstream.sleep_replicas()
            try:
                self.trainer._commit_policy_checkpoint_after_weight_sync(global_steps)
            finally:
                self.upstream.wake_up_replicas()
            checkpoint_seconds = perf_counter() - checkpoint_started
        complete = getattr(self.trainer, "_complete_policy_metric_publication", None)
        if callable(complete):
            complete(
                weight_sync_seconds=sync_seconds,
                checkpoint_seconds=checkpoint_seconds,
            )
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
            # Pinned veRL constructs its LLM server manager from a module
            # global.  Replace that construction boundary only while upstream
            # creates workers so the normal trainer/lifecycle remains intact,
            # while vLLM receives the TGVF worker extension and sticky client.
            import verl.trainer.ppo.ray_trainer as ray_trainer_module

            from .vllm_tool_runtime import tgvf_llm_server_manager_class

            original_manager_class = ray_trainer_module.LLMServerManager
            ray_trainer_module.LLMServerManager = tgvf_llm_server_manager_class()
            try:
                result = super().init_workers()
            finally:
                ray_trainer_module.LLMServerManager = original_manager_class
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
            self._policy_runtime_shutdown = False
            self._policy_metrics_pending = None
            return result

        def fit(self):
            from verl.utils import tracking as tracking_module

            original_tracking_class = tracking_module.Tracking
            tracker_instances: list[object] = []

            class CapturedPolicyTracking(original_tracking_class):
                def __init__(captured_self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    tracker_instances.append(captured_self)

                def log(captured_self, data, step, backend=None):
                    return super(CapturedPolicyTracking, captured_self).log(
                        data=_policy_tracking_metrics(data),
                        step=step,
                        backend=backend,
                    )

            tracking_module.Tracking = CapturedPolicyTracking
            try:
                completed_step = _completed_resume_checkpoint_step(self)
                if completed_step is None:
                    return super().fit()

                # Pinned veRL increments ``global_steps`` before checking its loop
                # bound, so resuming an already-complete one-step run would perform
                # an unauthorized extra update.  Load and validate the complete
                # paired checkpoint, publish exactly those weights, then exit.
                self.global_steps = 0
                self._load_checkpoint()
                if self.global_steps != completed_step:
                    raise RuntimeError("loaded Policy resume step changed")
                resumed = getattr(self._policy_checkpoint_state, "last_resume", None)
                if getattr(resumed, "optimizer_step", None) != completed_step:
                    raise RuntimeError("paired Policy resume step differs from veRL")
                self.checkpoint_manager.update_weights(self.global_steps)
                self._shutdown_dump_executor()
                return None
            finally:
                tracking_module.Tracking = original_tracking_class
                errors: list[Exception] = []
                for tracker in tracker_instances:
                    try:
                        _finish_tracking_backends(tracker)
                    except Exception as error:
                        errors.append(error)
                try:
                    self._shutdown_policy_runtime()
                except Exception as error:
                    errors.append(error)
                if errors:
                    raise ExceptionGroup("Policy runner shutdown failed", errors)

        def _shutdown_policy_runtime(self) -> None:
            if getattr(self, "_policy_runtime_shutdown", False):
                return
            errors: list[Exception] = []
            for name in ("train_dataloader", "val_dataloader"):
                loader = getattr(self, name, None)
                iterator = getattr(loader, "_iterator", None)
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as error:  # pragma: no cover - worker failure
                        errors.append(error)
                    else:
                        loader._iterator = None
            manager = getattr(self, "llm_server_manager", None)
            shutdown = getattr(manager, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as error:  # pragma: no cover - backend failure
                    errors.append(error)
            if errors:
                raise ExceptionGroup("Policy runtime shutdown failed", errors)
            self._policy_runtime_shutdown = True

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
            completed = False
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
                observation, summary = self._policy_checkpoint_state.record_optimizer_step(
                    batch,
                    elapsed_seconds=max(elapsed, 1.0e-12),
                )
                if getattr(self, "_policy_metrics_pending", None) is not None:
                    raise RuntimeError("a Policy metrics publication is already pending")
                event = _pilot_metrics_event(observation, summary)
                metrics = getattr(output, "meta_info", {}).get("metrics")
                if not isinstance(metrics, dict):
                    raise TypeError("veRL actor output metrics must be a dictionary")
                for name, value in _wandb_metrics_from_event(event).items():
                    if name in metrics:
                        raise RuntimeError(f"Policy metric already exists: {name}")
                    metrics[name] = [value]
                self._policy_metrics_pending = (output, event)
                completed = True
                return output
            finally:
                # All current/reference/update consumers are synchronous in the
                # accepted v0 path.  The driver copy can release only after the
                # actor call has returned; every Ray-local copy has its own
                # worker ``finally`` in the wrapped TrainingWorker.
                release_verl_data_proto_sidecars(batch)
                if not completed:
                    self._policy_step_started_at = None

        def _complete_policy_metric_publication(
            self,
            *,
            weight_sync_seconds: float,
            checkpoint_seconds: float,
        ) -> None:
            pending = getattr(self, "_policy_metrics_pending", None)
            if pending is None:
                return
            output, event = pending
            started = self._policy_step_started_at
            if started is None:
                raise RuntimeError("Policy metric publication lost its step timer")
            end_to_end_seconds = perf_counter() - started
            if min(weight_sync_seconds, checkpoint_seconds, end_to_end_seconds) < 0:
                raise RuntimeError("Policy timing metrics must be non-negative")
            metrics = getattr(output, "meta_info", {}).get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError("veRL actor output metrics must be a dictionary")
            timings = {
                "policy_timing/weight_sync_seconds": weight_sync_seconds,
                "policy_timing/checkpoint_seconds": checkpoint_seconds,
                "policy_timing/end_to_end_step_seconds": end_to_end_seconds,
            }
            for name, value in timings.items():
                if name in metrics:
                    raise RuntimeError(f"Policy timing metric already exists: {name}")
                metrics[name] = [value]
            persisted = dict(event)
            persisted["timing"] = {
                "weight_sync_seconds": weight_sync_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "end_to_end_step_seconds": end_to_end_seconds,
            }
            _append_policy_metrics_event(
                self._policy_checkpoint_state.config.output.metrics_path,
                persisted,
            )
            self._policy_metrics_pending = None
            self._policy_step_started_at = None

        def _save_checkpoint(self):
            configured_steps = (
                self._policy_checkpoint_state.config.training.checkpoint_steps
            )
            if self.global_steps not in configured_steps:
                return None
            if self._policy_checkpoint_pending:
                raise RuntimeError("a Policy checkpoint request is already pending")
            self._policy_checkpoint_pending = True
            return None

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
    raw_judge_usages = non_tensors.get(PILOT_VERL_JUDGE_USAGE_FIELD)
    judge_usages = (
        (None,) * len(trajectories)
        if raw_judge_usages is None
        else tuple(raw_judge_usages)
    )
    response_mask = batch["response_mask"]
    if not isinstance(response_mask, torch.Tensor):
        raise TypeError("Policy metric response_mask must be a tensor")
    if not (
        len(trajectories)
        == len(replay_bundles)
        == len(components)
        == len(judge_usages)
        == response_mask.shape[0]
    ):
        raise RuntimeError("Policy metric rows differ from the update batch")

    rows: list[PilotTrajectoryMetricsObservation] = []
    for index, (trajectory, bundle, raw_components, raw_judge_usage) in enumerate(
        zip(trajectories, replay_bundles, components, judge_usages, strict=True)
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
        if raw_judge_usage is None:
            judge_calls = 0
            judge_prompt_tokens = 0
            judge_completion_tokens = 0
            judge_cost_usd = 0.0
        else:
            try:
                (
                    judge_prompt_tokens,
                    judge_completion_tokens,
                    judge_total_tokens,
                    judge_cost_usd,
                ) = raw_judge_usage
                judge_prompt_tokens = int(judge_prompt_tokens)
                judge_completion_tokens = int(judge_completion_tokens)
                judge_total_tokens = int(judge_total_tokens)
                judge_cost_usd = float(judge_cost_usd)
            except (TypeError, ValueError) as error:
                raise TypeError("Policy judge usage sidecar is malformed") from error
            if judge_total_tokens != judge_prompt_tokens + judge_completion_tokens:
                raise ValueError("Policy judge usage token total differs")
            judge_calls = 1
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
                judge_calls=judge_calls,
                judge_prompt_tokens=judge_prompt_tokens,
                judge_completion_tokens=judge_completion_tokens,
                judge_cost_usd=judge_cost_usd,
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
    install_verl_sdpa_padding_compat()
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
