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
from tgvf_rl.policy.horizon_extension import (
    PolicyHorizonExtension,
    policy_horizon_extension_from_environment,
)
from tgvf_rl.policy.metrics import (
    POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT,
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
    POLICY_PILOT_CHECKPOINT_PAIR_FILENAME,
    POLICY_PILOT_PROJECT_STATE_FILENAME,
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
from tgvf_rl.rewards.stage3_verl_adapter import (
    STAGE3_VERL_QUALITY_APPLICABLE_FIELD,
    STAGE3_VERL_QUALITY_COVERED_FIELD,
    STAGE3_VERL_QUALITY_FAILURE_FIELD,
    STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD,
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
        "policy_pilot/mean_stage3_answer_reward",
        "policy_pilot/mean_stage3_tool_reward",
        "policy_pilot/mean_stage3_focus_reward",
        "policy_pilot/mean_stage3_grounding_reward",
        "policy_pilot/mean_stage3_protocol_reward",
        "policy_pilot/stage3_quality_judge_applicable",
        "policy_pilot/stage3_quality_judge_covered",
        "policy_pilot/stage3_quality_judge_failures",
        "policy_pilot/stage3_quality_judge_coverage",
        "policy_pilot/stage3_visual_judge_calls",
        "policy_pilot/stage3_visual_judge_prompt_tokens",
        "policy_pilot/stage3_visual_judge_completion_tokens",
        "policy_pilot/stage3_visual_judge_cost_usd",
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
        "judge_completion_tokens": sum(row.judge_completion_tokens for row in rows),
        "judge_cost_usd": sum(row.judge_cost_usd for row in rows),
        "pre_publication_elapsed_seconds": observation.step_time_seconds,
        "tool_error_counts": dict(sorted(errors.items())),
    }
    stage3_rows = tuple(row for row in rows if row.reward_profile == "stage3-shaped-v1")
    if stage3_rows:
        if len(stage3_rows) != trajectories:
            raise ValueError("one optimizer step cannot mix reward profiles")
        component_names = ("answer", "tool", "focus", "grounding", "protocol")
        for component_index, component_name in enumerate(component_names):
            step[f"mean_stage3_{component_name}_reward"] = _zero_safe_mean(
                sum(
                    row.stage3_reward_components[component_index]
                    for row in stage3_rows
                    if row.stage3_reward_components is not None
                ),
                trajectories,
            )
        applicable = sum(row.stage3_quality_judge_applicable for row in stage3_rows)
        covered = sum(row.stage3_quality_judge_covered for row in stage3_rows)
        step.update(
            {
                "stage3_quality_judge_applicable": applicable,
                "stage3_quality_judge_covered": covered,
                "stage3_quality_judge_failures": sum(
                    row.stage3_quality_judge_failure is not None for row in stage3_rows
                ),
                "stage3_quality_judge_coverage": _zero_safe_mean(covered, applicable),
                "stage3_visual_judge_calls": sum(
                    row.stage3_visual_judge_calls for row in stage3_rows
                ),
                "stage3_visual_judge_prompt_tokens": sum(
                    row.stage3_visual_judge_prompt_tokens for row in stage3_rows
                ),
                "stage3_visual_judge_completion_tokens": sum(
                    row.stage3_visual_judge_completion_tokens for row in stage3_rows
                ),
                "stage3_visual_judge_cost_usd": sum(
                    row.stage3_visual_judge_cost_usd for row in stage3_rows
                ),
            }
        )
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


def _strict_json_state(
    value: OpaqueProjectState, *, owner: str
) -> Mapping[str, object]:
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
    hashes = {
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
    }
    if config.reward.profile == "stage3-shaped-v1":
        tool_utility = config.reward.tool_utility
        visual_identity = config.reward.visual_quality_judge_identity
        visual_config_sha256 = config.reward.visual_quality_judge_config_sha256
        if (
            tool_utility is None
            or visual_identity is None
            or visual_config_sha256 is None
        ):
            raise ValueError("Stage3 run identity dependencies are missing")
        hashes.update(
            {
                "reward_tool_utility_sidecar": tool_utility.sidecar_sha256,
                "reward_tool_utility_manifest": tool_utility.manifest_sha256,
                "reward_visual_judge_config": visual_config_sha256,
                "reward_visual_judge_identity": visual_identity.sha256,
            }
        )
    return PilotRunIdentityHashes.from_hashes(
        config.run_id,
        hashes,
    )


class PolicyPilotTrainerCheckpointState:
    """Driver-owned project state paired with the framework checkpoint."""

    def __init__(self, trainer: object, config: PolicyE2ESmokeRunConfig) -> None:
        self.trainer = trainer
        self.config = config
        self._run_identity = _run_identity(config)
        self.horizon_extension = policy_horizon_extension_from_environment(config)
        self._weight_state = PolicyWeightSyncState.from_environment()
        if self._weight_state.run_id != config.run_id or (
            self._weight_state.run_identity_sha256 != config.identity_sha256
        ):
            raise RuntimeError(
                "Policy checkpoint and weight-sync run identities differ"
            )
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
        # Keep the next-data cursor at the most recent completed optimizer
        # boundary.  The live StatefulDataLoader may already have yielded the
        # next batch when rollout/reward raises, so recapturing it in an
        # exception handler would silently skip that failed batch on resume.
        self._recovery_progress = self._progress
        self._recovery_sampler = self._sampler
        self._recovery_rng = self._rng
        self._prepared_policy: PolicyVersion | None = None
        self.last_resume: PolicyPilotVerlResumeResult | None = None

    @property
    def effective_checkpoint_steps(self) -> tuple[int, ...]:
        extension = self.horizon_extension
        if isinstance(extension, PolicyHorizonExtension):
            return extension.effective_checkpoint_steps
        return self.config.training.checkpoint_steps

    @classmethod
    def from_environment(cls, trainer: object) -> "PolicyPilotTrainerCheckpointState":
        path = os.environ.get("TGVF_POLICY_RUN_CONFIG_PATH")
        if not path:
            raise RuntimeError(
                "TGVF_POLICY_RUN_CONFIG_PATH is required by Policy TaskRunner"
            )
        config = load_policy_e2e_smoke_run_config(Path(path))
        expected = os.environ.get("TGVF_POLICY_RUN_IDENTITY_SHA256")
        if expected != config.identity_sha256:
            raise RuntimeError("Policy TaskRunner config/environment identity differs")
        return cls(trainer, config)

    def prepare_checkpoint(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step <= 0:
            raise ValueError("Policy checkpoint optimizer step must be positive")
        if self.metrics_accumulator.state.optimizer_steps != optimizer_step:
            raise RuntimeError(
                "Policy metrics do not reach the checkpoint optimizer step"
            )
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
            trajectories_per_prompt=(
                self.config.policy.sampling.trajectories_per_prompt
            ),
        )
        summary = self.metrics_accumulator.record_optimizer_step(observation)
        self._recovery_progress = PilotOptimizerDataCursor(
            step,
            _torch_state(DATA_CURSOR_OWNER, self._dataloader_state()),
        )
        self._recovery_sampler = self._sampler_state(step)
        self._recovery_rng = self._rng_state(step)
        return observation, summary

    def recovery_optimizer_step(self) -> int:
        """Return the last boundary whose optimizer and data both completed."""

        step = self._recovery_progress.optimizer_step
        if self.metrics_accumulator.state.optimizer_steps != step:
            raise RuntimeError("Policy recovery cursor and metrics steps differ")
        return step

    def restore_recovery_cursor_for_checkpoint(self, optimizer_step: int) -> None:
        """Roll the driver data cursor back before an exception checkpoint."""

        if optimizer_step != self.recovery_optimizer_step() or optimizer_step <= 0:
            raise RuntimeError("Policy recovery checkpoint step is unavailable")
        state = _load_torch_state(self._recovery_progress.data_cursor)
        loader = getattr(self.trainer, "train_dataloader", None)
        restore = getattr(loader, "load_state_dict", None)
        if not callable(restore):
            raise TypeError("veRL train_dataloader must implement load_state_dict()")
        restore(state)
        self._progress = self._recovery_progress
        self._sampler = self._recovery_sampler
        self._rng = self._recovery_rng
        self._prepared_policy = None

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
        model_slug = Path(self.config.model.model_name).name.casefold()
        return PolicyVersion(
            f"{model_slug}-frozen-reference",
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
        self._recovery_progress = value
        self._prepared_policy = None

    def restore_rollout_sampler_state(self, value: OpaqueProjectState) -> None:
        self._validate_derived_state(value, owner=ROLLOUT_SAMPLER_OWNER)
        self._sampler = value
        self._recovery_sampler = value

    def restore_rollout_rng_state(self, value: OpaqueProjectState) -> None:
        self._validate_derived_state(value, owner=ROLLOUT_RNG_OWNER)
        self._rng = value
        self._recovery_rng = value

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

    def __init__(
        self, upstream: object, state: PolicyPilotTrainerCheckpointState
    ) -> None:
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

    def __init__(
        self,
        upstream: object,
        trainer: object,
        *,
        replicas_sleeping: bool = False,
    ) -> None:
        self.upstream = upstream
        self.trainer = trainer
        self._replicas_sleeping = replicas_sleeping

    def sleep_replicas(self) -> object:
        result = self.upstream.sleep_replicas()
        self._replicas_sleeping = True
        return result

    def wake_up_replicas(self) -> object:
        result = self.upstream.wake_up_replicas()
        self._replicas_sleeping = False
        return result

    def quiesce_after_training_failure(self) -> None:
        """Stop incomplete rollout work and free memory for FSDP2 checkpointing."""

        if self._replicas_sleeping:
            return
        abort = getattr(self.upstream, "abort_replicas", None)
        if callable(abort):
            abort()
        self.sleep_replicas()

    def update_weights(self, global_steps: int | None = None) -> object:
        sync_started = perf_counter()
        result = self.upstream.update_weights(global_steps)
        self._replicas_sleeping = False
        sync_seconds = perf_counter() - sync_started
        checkpoint_seconds = 0.0
        if getattr(self.trainer, "_policy_checkpoint_pending", False):
            checkpoint_started = perf_counter()
            self.sleep_replicas()
            try:
                self.trainer._commit_policy_checkpoint_after_weight_sync(global_steps)
            finally:
                self.wake_up_replicas()
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


def policy_worker_logical_cuda_ordinal(
    allocated_gpu_id: str,
    visible_devices: str,
) -> int:
    """Map one Ray GPU resource ID to its process-local CUDA ordinal.

    With ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``, veRL normally
    passes Ray's allocated ID directly to ``torch.cuda.set_device``.  That is
    only valid when the physical allocation is ``0..N-1``.  For an isolated
    launch on physical GPUs 4--7, CUDA renumbers those devices to local
    ordinals 0--3, so the resource ID must first be resolved through
    ``CUDA_VISIBLE_DEVICES``.
    """

    if not isinstance(allocated_gpu_id, str) or not allocated_gpu_id:
        raise ValueError("allocated_gpu_id must be a non-empty string")
    if not isinstance(visible_devices, str) or not visible_devices:
        raise ValueError("visible_devices must be a non-empty string")
    devices = tuple(part.strip() for part in visible_devices.split(","))
    if any(not part for part in devices) or len(devices) != len(set(devices)):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique device IDs")
    if allocated_gpu_id in devices:
        return devices.index(allocated_gpu_id)
    try:
        logical_ordinal = int(allocated_gpu_id)
    except ValueError as error:
        raise ValueError(
            "Ray GPU allocation is absent from CUDA_VISIBLE_DEVICES"
        ) from error
    if not 0 <= logical_ordinal < len(devices):
        raise ValueError("Ray GPU allocation exceeds the local CUDA device view")
    return logical_ordinal


def make_policy_colocated_worker_class(upstream_worker_cls: type[Any]) -> type[Any]:
    """Wrap veRL's dynamic WorkerDict with physical-to-logical GPU mapping."""

    if not isinstance(upstream_worker_cls, type):
        raise TypeError("upstream colocated worker must be a class")
    if not callable(
        getattr(upstream_worker_cls, "_setup_env_cuda_visible_devices", None)
    ):
        raise TypeError("upstream colocated worker lacks CUDA environment setup")

    class PolicyPhysicalGPUWorker(upstream_worker_cls):
        def _setup_env_cuda_visible_devices(self):
            from verl.utils.ray_utils import ray_noset_visible_devices

            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if (
                not ray_noset_visible_devices()
                or not visible_devices
                or os.environ.get("HIP_VISIBLE_DEVICES")
                or os.environ.get("ROCR_VISIBLE_DEVICES")
            ):
                return super()._setup_env_cuda_visible_devices()

            import ray
            from verl.utils.device import get_resource_name, get_torch_device

            resource_name = get_resource_name()
            allocated = (
                ray.get_runtime_context().get_accelerator_ids().get(resource_name, [])
            )
            if len(allocated) != 1:
                raise RuntimeError(
                    "Policy colocated worker requires exactly one Ray GPU allocation"
                )
            logical_ordinal = policy_worker_logical_cuda_ordinal(
                str(allocated[0]), visible_devices
            )
            os.environ["LOCAL_RANK"] = str(logical_ordinal)
            get_torch_device().set_device(logical_ordinal)

    PolicyPhysicalGPUWorker.__name__ = "PolicyPhysicalGPUWorker"
    PolicyPhysicalGPUWorker.__qualname__ = "PolicyPhysicalGPUWorker"
    return PolicyPhysicalGPUWorker


def make_policy_pilot_ray_trainer_class(upstream_trainer_cls: type[Any]) -> type[Any]:
    """Return the pinned v0 trainer with lifecycle/checkpoint hooks installed."""

    if not isinstance(upstream_trainer_cls, type):
        raise TypeError("upstream RayPPOTrainer must be a class")
    for name in ("init_workers", "_get_gen_batch", "_update_actor", "_save_checkpoint"):
        if not callable(getattr(upstream_trainer_cls, name, None)):
            raise TypeError(f"upstream RayPPOTrainer is missing {name}()")

    class PolicyPilotRayPPOTrainer(upstream_trainer_cls):
        def __init__(self, *args, **kwargs):
            config = kwargs.get("config", args[0] if args else None)
            actor_scheduler_horizon = _actor_scheduler_horizon(config)
            super().__init__(*args, **kwargs)
            _restore_actor_scheduler_horizon(
                self.config,
                actor_scheduler_horizon,
            )
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
            import verl.single_controller.ray.base as ray_base_module
            import verl.trainer.ppo.ray_trainer as ray_trainer_module

            from .vllm_tool_runtime import tgvf_llm_server_manager_class

            original_manager_class = ray_trainer_module.LLMServerManager
            original_base_class_resolver = (
                ray_base_module._determine_fsdp_megatron_base_class
            )

            def resolve_policy_colocated_worker_base(*args, **kwargs):
                return make_policy_colocated_worker_class(
                    original_base_class_resolver(*args, **kwargs)
                )

            ray_trainer_module.LLMServerManager = tgvf_llm_server_manager_class()
            ray_base_module._determine_fsdp_megatron_base_class = (
                resolve_policy_colocated_worker_base
            )
            try:
                result = super().init_workers()
            finally:
                ray_trainer_module.LLMServerManager = original_manager_class
                ray_base_module._determine_fsdp_megatron_base_class = (
                    original_base_class_resolver
                )
            state = PolicyPilotTrainerCheckpointState.from_environment(self)
            original_actor_wg = self.actor_rollout_wg
            self._policy_checkpoint_state = state
            self.actor_rollout_wg = PairedActorWorkerGroup(original_actor_wg, state)
            if getattr(self, "ref_policy_wg", None) is original_actor_wg:
                self.ref_policy_wg = self.actor_rollout_wg
            self.checkpoint_manager = CheckpointAfterWeightSyncManager(
                self.checkpoint_manager,
                self,
                # Pinned veRL sleeps all rollout replicas at the end of
                # init_workers() before this wrapper is installed.
                replicas_sleeping=True,
            )
            self._policy_checkpoint_pending = False
            self._policy_step_started_at = None
            self._policy_runtime_shutdown = False
            self._policy_metrics_pending = None
            self._policy_actor_update_inflight = False
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
                    resumed = getattr(
                        self._policy_checkpoint_state, "last_resume", None
                    )
                    if getattr(resumed, "optimizer_step", None) != completed_step:
                        raise RuntimeError(
                            "paired Policy resume step differs from veRL"
                        )
                    self.checkpoint_manager.update_weights(self.global_steps)
                    self._shutdown_dump_executor()
                    return None
                except Exception as training_error:
                    # Ray can stop forwarding actor output while the failure
                    # path quiesces rollout servers and writes a distributed
                    # checkpoint.  Persist the original traceback in the
                    # actor's own stderr before either operation so the root
                    # exception is not lost during teardown.
                    import sys
                    import traceback

                    print(
                        "TGVF policy training failed; saving the last completed boundary",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exception(training_error, file=sys.stderr)
                    sys.stderr.flush()
                    try:
                        self._save_last_completed_checkpoint_after_failure()
                    except Exception as checkpoint_error:
                        raise ExceptionGroup(
                            "Policy training and recovery checkpoint both failed",
                            [training_error, checkpoint_error],
                        ) from training_error
                    raise
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

        def _save_last_completed_checkpoint_after_failure(self) -> int | None:
            """Save the last exact boundary without relabeling a failed step."""

            state = getattr(self, "_policy_checkpoint_state", None)
            recovery_step = getattr(state, "recovery_optimizer_step", None)
            if not callable(recovery_step):
                return None
            optimizer_step = recovery_step()
            if optimizer_step == 0:
                return None
            if getattr(self, "_policy_actor_update_inflight", False):
                raise RuntimeError(
                    "cannot checkpoint after an optimizer exception with unknown commit"
                )
            if getattr(self, "_policy_metrics_pending", None) is not None:
                raise RuntimeError(
                    "cannot checkpoint before current-step weight synchronization completes"
                )
            if getattr(self, "_policy_checkpoint_pending", False):
                raise RuntimeError("a Policy checkpoint request is already pending")

            checkpoint_root_value = getattr(
                getattr(self.config, "trainer", None),
                "default_local_dir",
                None,
            )
            if checkpoint_root_value is not None:
                actor_checkpoint = (
                    Path(checkpoint_root_value)
                    / f"global_step_{optimizer_step}"
                    / "actor"
                )
                committed_files = (
                    actor_checkpoint / POLICY_PILOT_PROJECT_STATE_FILENAME,
                    actor_checkpoint / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME,
                )
                if all(path.is_file() for path in committed_files):
                    return optimizer_step

            state.restore_recovery_cursor_for_checkpoint(optimizer_step)
            manager = self.checkpoint_manager
            quiesce = getattr(manager, "quiesce_after_training_failure", None)
            if not callable(quiesce):
                raise TypeError(
                    "Policy checkpoint manager cannot quiesce failed rollout"
                )
            quiesce()

            failed_global_step = getattr(self, "global_steps", None)
            self.global_steps = optimizer_step
            self._policy_checkpoint_pending = True
            try:
                self._commit_policy_checkpoint_after_weight_sync(optimizer_step)
            finally:
                self.global_steps = failed_global_step
            return optimizer_step

        def _compute_ref_log_prob(self, batch):
            if not self.ref_in_actor:
                # Full-model training has no LoRA, so pinned veRL correctly
                # initializes ``ref_in_actor=False`` and routes this call to
                # the explicit frozen reference engine owned by the combined
                # ActorRolloutRef worker group.  No flag override is needed.
                return super()._compute_ref_log_prob(batch)
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
                validate_policy_pilot_reward_data_proto(
                    batch,
                    expected_group_size=(
                        self._policy_checkpoint_state.config.policy.sampling.trajectories_per_prompt
                    ),
                )
                self._policy_actor_update_inflight = True
                output = super()._update_actor(batch, *args, **kwargs)
                started = self._policy_step_started_at
                elapsed = perf_counter() - started if started is not None else 0.0
                observation, summary = (
                    self._policy_checkpoint_state.record_optimizer_step(
                        batch,
                        elapsed_seconds=max(elapsed, 1.0e-12),
                    )
                )
                if getattr(self, "_policy_metrics_pending", None) is not None:
                    raise RuntimeError(
                        "a Policy metrics publication is already pending"
                    )
                event = _pilot_metrics_event(observation, summary)
                metrics = getattr(output, "meta_info", {}).get("metrics")
                if not isinstance(metrics, dict):
                    raise TypeError("veRL actor output metrics must be a dictionary")
                for name, value in _wandb_metrics_from_event(event).items():
                    if name in metrics:
                        raise RuntimeError(f"Policy metric already exists: {name}")
                    metrics[name] = [value]
                self._policy_metrics_pending = (output, event)
                self._policy_actor_update_inflight = False
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
            configured_steps = self._policy_checkpoint_state.effective_checkpoint_steps
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


def _actor_scheduler_horizon(config: object) -> int:
    """Read the run-bound actor horizon before pinned veRL overwrites it."""

    try:
        horizon = config.actor_rollout_ref.actor.optim.total_training_steps  # type: ignore[union-attr]
    except AttributeError as error:
        raise TypeError(
            "Policy actor optimizer scheduler horizon is missing"
        ) from error
    if type(horizon) is not int or horizon <= 0:
        raise TypeError("Policy actor optimizer scheduler horizon must be positive")
    return horizon


def _restore_actor_scheduler_horizon(config: object, horizon: int) -> None:
    """Undo pinned veRL's trainer-step overwrite before workers are created."""

    from omegaconf import DictConfig, open_dict

    if isinstance(config, DictConfig):
        with open_dict(config):
            config.actor_rollout_ref.actor.optim.total_training_steps = horizon
        return
    config.actor_rollout_ref.actor.optim.total_training_steps = horizon  # type: ignore[union-attr]


def _decode_judge_usage(
    raw_usage: object,
    *,
    owner: str,
) -> tuple[int, int, int, float]:
    """Decode one optional judge-usage sidecar without changing Pilot coercion."""

    if raw_usage is None:
        return 0, 0, 0, 0.0
    try:
        prompt_tokens, completion_tokens, total_tokens, cost_usd = raw_usage  # type: ignore[misc]
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        total_tokens = int(total_tokens)
        cost_usd = float(cost_usd)
    except (TypeError, ValueError) as error:
        label = "Policy judge" if owner == "Policy answer judge" else owner
        raise TypeError(f"{label} usage sidecar is malformed") from error
    if total_tokens != prompt_tokens + completion_tokens:
        label = "Policy judge" if owner == "Policy answer judge" else owner
        raise ValueError(f"{label} usage token total differs")
    return 1, prompt_tokens, completion_tokens, cost_usd


def policy_metrics_observation_from_data_proto(
    data: object,
    *,
    optimizer_step: int,
    elapsed_seconds: float,
    trajectories_per_prompt: int = POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT,
) -> PilotOptimizerStepMetricsObservation:
    """Recover the checkpointed raw Pilot metrics from one exact update batch."""

    reward_view = validate_policy_pilot_reward_data_proto(
        data,
        expected_group_size=trajectories_per_prompt,
    )
    stage3_profile = (
        reward_view.reward_bridge_schema_version
        == STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION
    )
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
    if stage3_profile:
        quality_applicable = tuple(non_tensors[STAGE3_VERL_QUALITY_APPLICABLE_FIELD])
        quality_covered = tuple(non_tensors[STAGE3_VERL_QUALITY_COVERED_FIELD])
        quality_failures = tuple(non_tensors[STAGE3_VERL_QUALITY_FAILURE_FIELD])
        visual_judge_usages = tuple(non_tensors[STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD])
    else:
        quality_applicable = (False,) * len(trajectories)
        quality_covered = (False,) * len(trajectories)
        quality_failures = (None,) * len(trajectories)
        visual_judge_usages = (None,) * len(trajectories)
    response_mask = batch["response_mask"]
    if not isinstance(response_mask, torch.Tensor):
        raise TypeError("Policy metric response_mask must be a tensor")
    if not (
        len(trajectories)
        == len(replay_bundles)
        == len(components)
        == len(judge_usages)
        == len(quality_applicable)
        == len(quality_covered)
        == len(quality_failures)
        == len(visual_judge_usages)
        == response_mask.shape[0]
    ):
        raise RuntimeError("Policy metric rows differ from the update batch")

    rows: list[PilotTrajectoryMetricsObservation] = []
    for index, row_values in enumerate(
        zip(
            trajectories,
            replay_bundles,
            components,
            judge_usages,
            quality_applicable,
            quality_covered,
            quality_failures,
            visual_judge_usages,
            strict=True,
        )
    ):
        (
            trajectory,
            bundle,
            raw_components,
            raw_judge_usage,
            raw_quality_applicable,
            raw_quality_covered,
            raw_quality_failure,
            raw_visual_judge_usage,
        ) = row_values
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("Policy metric trajectory sidecar is invalid")
        if not isinstance(bundle, TrajectoryReplayBundle):
            raise TypeError("Policy metric replay bundle sidecar is invalid")
        try:
            reward = {str(name): float(value) for name, value in raw_components}
        except (TypeError, ValueError) as error:
            raise TypeError("Policy reward component sidecar is malformed") from error
        if stage3_profile:
            if tuple(reward) != (
                "answer",
                "tool",
                "focus",
                "grounding",
                "protocol",
            ):
                raise ValueError("Stage3 metric reward components differ")
            stage3_components = tuple(reward.values())
            compatibility_answer_reward = reward["answer"] / 2.0
            compatibility_format_error = reward["protocol"] == -1.0
            compatibility_conditional_tool_reward = 0.0
            if (
                type(raw_quality_applicable) is not bool
                or type(raw_quality_covered) is not bool
                or (
                    raw_quality_failure is not None
                    and not isinstance(raw_quality_failure, str)
                )
            ):
                raise TypeError("Stage3 quality metric sidecars are malformed")
            (
                _visual_usage_present,
                visual_judge_prompt_tokens,
                visual_judge_completion_tokens,
                visual_judge_cost_usd,
            ) = _decode_judge_usage(
                raw_visual_judge_usage,
                owner="Stage3 visual judge",
            )
        else:
            if set(reward) != {
                "answer_reward",
                "format_reward",
                "conditional_tool_reward",
            }:
                raise ValueError("Policy metric reward components differ")
            stage3_components = None
            compatibility_answer_reward = reward["answer_reward"]
            compatibility_format_error = reward["format_reward"] == -1.0
            compatibility_conditional_tool_reward = reward["conditional_tool_reward"]
            visual_judge_prompt_tokens = 0
            visual_judge_completion_tokens = 0
            visual_judge_cost_usd = 0.0
        (
            judge_calls,
            judge_prompt_tokens,
            judge_completion_tokens,
            judge_cost_usd,
        ) = _decode_judge_usage(raw_judge_usage, owner="Policy answer judge")
        original_visual = len(bundle.replay_record.source_visual.positions)
        tool_visual = sum(
            _observation_visual_tokens(item) for item in bundle.observation_records
        )
        reasoning = sum(
            0
            if turn.think_span is None
            else turn.think_span.end - turn.think_span.start
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
                answer_reward=compatibility_answer_reward,
                format_error=compatibility_format_error,
                conditional_tool_reward=compatibility_conditional_tool_reward,
                reasoning_tokens=reasoning,
                original_visual_tokens=original_visual,
                total_visual_tokens=original_visual + tool_visual,
                tool_error_codes=errors,
                judge_calls=judge_calls,
                judge_prompt_tokens=judge_prompt_tokens,
                judge_completion_tokens=judge_completion_tokens,
                judge_cost_usd=judge_cost_usd,
                reward_profile=("stage3-shaped-v1" if stage3_profile else "pilot-v1"),
                stage3_reward_components=stage3_components,
                stage3_quality_judge_applicable=bool(raw_quality_applicable),
                stage3_quality_judge_covered=bool(raw_quality_covered),
                stage3_quality_judge_failure=raw_quality_failure,
                stage3_visual_judge_calls=int(raw_quality_applicable),
                stage3_visual_judge_prompt_tokens=visual_judge_prompt_tokens,
                stage3_visual_judge_completion_tokens=(visual_judge_completion_tokens),
                stage3_visual_judge_cost_usd=visual_judge_cost_usd,
            )
        )
    return PilotOptimizerStepMetricsObservation(
        optimizer_step=optimizer_step,
        step_time_seconds=elapsed_seconds,
        trajectories=tuple(rows),
        trajectories_per_prompt=trajectories_per_prompt,
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

    upstream_training_worker_cls = getattr(actor_worker_cls, "actor_worker_cls", None)
    mapped_training_worker_cls = make_policy_colocated_worker_class(
        upstream_training_worker_cls
    )
    wrapped = make_sidecar_releasing_actor_rollout_ref_worker_class(
        actor_worker_cls,
        upstream_training_worker_cls=mapped_training_worker_cls,
    )
    # ActorRolloutRefWorker calls ``Worker.__init__(self)`` explicitly rather
    # than through ``super()``.  Its CUDA setup still dispatches dynamically,
    # so wrap the role worker itself as well as the outer WorkerDict base.
    # Otherwise alternate physical allocations such as GPUs 4--7 reach
    # ``torch.cuda.set_device(4..7)`` inside a four-device local CUDA view.
    wrapped = make_policy_colocated_worker_class(wrapped)
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


def run_policy_pilot_v0_task(
    runner: object, config: object, *, trainer_cls: type[Any]
) -> None:
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
    "make_policy_colocated_worker_class",
    "policy_worker_logical_cuda_ordinal",
    "policy_metrics_observation_from_data_proto",
    "run_policy_pilot_v0_task",
]
