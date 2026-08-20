from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tgvf_rl.framework.verl.policy_task_runner as policy_task_runner
from tgvf_rl.framework.verl.data_bridge import (
    bind_agent_loop_data_proto_sidecar_lease,
    release_verl_data_proto_sidecars,
)
from tgvf_rl.framework.verl.policy_task_runner import (
    CheckpointAfterWeightSyncManager,
    PairedActorWorkerGroup,
    PolicyPilotTrainerCheckpointState,
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
    POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV,
    _append_policy_metrics_event,
    _completed_resume_checkpoint_step,
    _finish_tracking_backends,
    _pilot_metrics_event,
    _policy_reference_diagnostic_enabled,
    _policy_tracking_metrics,
    _require_successful_tgvf_observation_for_canary,
    _resolved_policy_metrics_path,
    _torch_state,
    _validate_metric_tool_attempt_contract,
    _wandb_metrics_from_event,
    add_policy_actor_rollout_worker,
    make_policy_colocated_worker_class,
    make_policy_pilot_ray_trainer_class,
    policy_worker_logical_cuda_ordinal,
)
from tgvf_rl.policy.checkpoint import DATA_CURSOR_OWNER, PilotOptimizerDataCursor
from tgvf_rl.policy.metrics import (
    PilotMetricsAccumulator,
    PilotOptimizerStepMetricsObservation,
    PilotTrajectoryMetricsObservation,
)
from tgvf_rl.policy.horizon_extension import PolicyHorizonExtension
from tgvf_rl.protocol.schema import ToolErrorCode
from tgvf_rl.framework.verl.rollout_bridge import (
    AGENT_LOOP_EXACT_SIDECAR_FIELDS,
    DATAPROTO_META_SCHEMA_FIELD,
    DATAPROTO_META_SCHEMA_VERSION,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    SIDECAR_RELEASE_FIELDS_FIELD,
    SIDECAR_RELEASE_SCHEMA_FIELD,
    SIDECAR_RELEASE_SCHEMA_VERSION,
)


def test_live_agent_loop_dataproto_gets_driver_and_worker_release_lease() -> None:
    non_tensor_batch = {
        name: np.array([object(), object()], dtype=object)
        for name in AGENT_LOOP_EXACT_SIDECAR_FIELDS
    }
    non_tensor_batch[EXACT_PROMPT_IDS_FIELD] = np.array([(1,), (2,)], dtype=object)
    non_tensor_batch[EXACT_RESPONSE_IDS_FIELD] = np.array([(3,), (4,)], dtype=object)
    data = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[1], [2]], dtype=torch.long),
            "responses": torch.tensor([[3], [4]], dtype=torch.long),
        },
        non_tensor_batch=non_tensor_batch,
        meta_info={"metrics": []},
    )

    assert bind_agent_loop_data_proto_sidecar_lease(data) is data
    assert data.meta_info[DATAPROTO_META_SCHEMA_FIELD] == DATAPROTO_META_SCHEMA_VERSION
    assert (
        data.meta_info[SIDECAR_RELEASE_SCHEMA_FIELD] == SIDECAR_RELEASE_SCHEMA_VERSION
    )
    assert (
        data.meta_info[SIDECAR_RELEASE_FIELDS_FIELD] == AGENT_LOOP_EXACT_SIDECAR_FIELDS
    )
    assert release_verl_data_proto_sidecars(data) == len(
        AGENT_LOOP_EXACT_SIDECAR_FIELDS
    )
    assert release_verl_data_proto_sidecars(data) == 0
    assert data.non_tensor_batch == {}


def test_task_runner_maps_the_real_sidecar_releasing_role_worker() -> None:
    calls: list[object] = []

    class TrainingWorker:
        def _setup_env_cuda_visible_devices(self):
            return None

        def train_mini_batch(self, data):
            calls.append(("train", data))

        def infer_batch(self, data):
            calls.append(("infer", data))

    class ActorRolloutRefWorker:
        actor_worker_cls = TrainingWorker
        ref_worker_cls = TrainingWorker

        def _setup_env_cuda_visible_devices(self):
            return None

    class Role:
        ActorRollout = "actor"
        ActorRolloutRef = "actor_ref"

    class Ray:
        @staticmethod
        def remote(value):
            calls.append(("remote", value))
            return ("remote", value)

    runner = SimpleNamespace(role_worker_mapping={}, mapping={})
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model={"lora": {"rank": 64}, "lora_adapter_path": None}
        )
    )

    wrapped, group_cls = add_policy_actor_rollout_worker(
        runner,
        config,
        ray_module=Ray,
        role_type=Role,
        actor_worker_cls=ActorRolloutRefWorker,
        ray_worker_group_cls=dict,
        need_reference_policy_fn=lambda _config: False,
    )

    assert group_cls is dict
    assert runner.role_worker_mapping[Role.ActorRolloutRef] == ("remote", wrapped)
    assert runner.mapping == {Role.ActorRolloutRef: "global_pool"}
    assert wrapped.actor_worker_cls is wrapped.ref_worker_cls
    assert wrapped.actor_worker_cls is not TrainingWorker
    assert "PolicyPhysicalGPUWorker" in {
        base.__name__ for base in wrapped.actor_worker_cls.__mro__
    }


def test_policy_reference_diagnostic_can_be_disabled_for_zero_kl_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrainingWorker:
        def _setup_env_cuda_visible_devices(self):
            return None

        def train_mini_batch(self, data):
            return data

        def infer_batch(self, data):
            return data

    class ActorRolloutRefWorker:
        actor_worker_cls = TrainingWorker
        ref_worker_cls = TrainingWorker

        def _setup_env_cuda_visible_devices(self):
            return None

    class Role:
        ActorRollout = "actor"
        ActorRolloutRef = "actor_ref"

    class Ray:
        @staticmethod
        def remote(value):
            return value

    monkeypatch.setenv(POLICY_REFERENCE_DIAGNOSTIC_ENV, "0")
    runner = SimpleNamespace(role_worker_mapping={}, mapping={})
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model={"lora": {"rank": 0}, "lora_adapter_path": None}
        )
    )
    add_policy_actor_rollout_worker(
        runner,
        config,
        ray_module=Ray,
        role_type=Role,
        actor_worker_cls=ActorRolloutRefWorker,
        ray_worker_group_cls=dict,
        need_reference_policy_fn=lambda _config: False,
    )

    assert _policy_reference_diagnostic_enabled() is False
    assert set(runner.role_worker_mapping) == {Role.ActorRollout}
    assert runner.mapping == {Role.ActorRollout: "global_pool"}
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _policy_reference_diagnostic_enabled({POLICY_REFERENCE_DIAGNOSTIC_ENV: "false"})


@pytest.mark.parametrize(
    ("allocated_gpu_id", "visible_devices", "expected"),
    (
        ("0", "0,1,2,3", 0),
        ("3", "0,1,2,3", 3),
        ("4", "4,5,6,7", 0),
        ("7", "4,5,6,7", 3),
        ("0", "4,5,6,7", 0),
        ("GPU-b", "GPU-a,GPU-b", 1),
    ),
)
def test_policy_worker_maps_ray_gpu_id_to_process_local_cuda_ordinal(
    allocated_gpu_id: str,
    visible_devices: str,
    expected: int,
) -> None:
    assert (
        policy_worker_logical_cuda_ordinal(allocated_gpu_id, visible_devices)
        == expected
    )


@pytest.mark.parametrize(
    ("allocated_gpu_id", "visible_devices"),
    (
        ("8", "4,5,6,7"),
        ("GPU-z", "GPU-a,GPU-b"),
        ("4", "4,4"),
        ("4", "4,"),
    ),
)
def test_policy_worker_rejects_ambiguous_cuda_mapping(
    allocated_gpu_id: str,
    visible_devices: str,
) -> None:
    with pytest.raises(ValueError):
        policy_worker_logical_cuda_ordinal(allocated_gpu_id, visible_devices)


def test_policy_colocated_worker_wrapper_preserves_upstream_type() -> None:
    import ray.cloudpickle

    class UpstreamWorker:
        def _setup_env_cuda_visible_devices(self):
            return "upstream"

    wrapped = make_policy_colocated_worker_class(UpstreamWorker)
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(wrapped))

    assert issubclass(wrapped, UpstreamWorker)
    assert wrapped is not UpstreamWorker
    assert restored.__name__ == "PolicyPhysicalGPUWorker"


def test_reference_diagnostic_routes_explicit_ref_engine_and_restores_flag() -> None:
    events: list[object] = []

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            return None

        def _compute_ref_log_prob(self, batch):
            events.append(("reference", batch, self.ref_in_actor))
            return "frozen-reference-logprobs"

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.ref_in_actor = True

    assert trainer._compute_ref_log_prob("exact-bundle") == "frozen-reference-logprobs"
    assert events == [("reference", "exact-bundle", False)]
    assert trainer.ref_in_actor is True


def test_reference_diagnostic_preserves_full_model_explicit_ref_route() -> None:
    events: list[object] = []

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            return None

        def _compute_ref_log_prob(self, batch):
            events.append(("reference", batch, self.ref_in_actor))
            return "frozen-reference-logprobs"

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.ref_in_actor = False

    assert trainer._compute_ref_log_prob("exact-bundle") == "frozen-reference-logprobs"
    assert events == [("reference", "exact-bundle", False)]
    assert trainer.ref_in_actor is False


def test_trainer_gate_preserves_the_run_bound_actor_scheduler_horizon() -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "trainer": {"total_training_steps": 20},
            "actor_rollout_ref": {"actor": {"optim": {"total_training_steps": 80}}},
        }
    )
    OmegaConf.set_struct(config, True)

    class UpstreamTrainer:
        def __init__(self, *, config):
            self.config = config
            self.total_training_steps = config.trainer.total_training_steps
            # This is the pinned veRL v0 behavior that previously collapsed
            # the optimizer schedule horizon to the bounded trainer gate.
            config.actor_rollout_ref.actor.optim.total_training_steps = (
                self.total_training_steps
            )

        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            return None

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = trainer_cls(config=config)

    assert trainer.total_training_steps == 20
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == 80


def test_extension_checkpoint_schedule_is_used_after_resume() -> None:
    state = object.__new__(PolicyPilotTrainerCheckpointState)
    state.config = SimpleNamespace(
        training=SimpleNamespace(checkpoint_steps=(0, 1, 5, 10, 20))
    )
    extension = object.__new__(PolicyHorizonExtension)
    object.__setattr__(
        extension,
        "effective_checkpoint_steps",
        (0, 1, 5, 10, 20, 30, 40, 60, 80),
    )
    state.horizon_extension = extension

    assert state.effective_checkpoint_steps == (0, 1, 5, 10, 20, 30, 40, 60, 80)


def test_pending_checkpoint_commits_after_sync_while_rollout_is_asleep() -> None:
    events: list[object] = []

    class UpstreamManager:
        def update_weights(self, step):
            events.append(("sync", step))
            return {"synced": step}

        def sleep_replicas(self):
            events.append("sleep")

        def wake_up_replicas(self):
            events.append("wake")

    class Trainer:
        _policy_checkpoint_pending = True

        def _commit_policy_checkpoint_after_weight_sync(self, step):
            events.append(("checkpoint", step))
            self._policy_checkpoint_pending = False

        def _complete_policy_metric_publication(self, **timings):
            events.append(("metrics", timings))

    trainer = Trainer()
    manager = CheckpointAfterWeightSyncManager(UpstreamManager(), trainer)

    assert manager.update_weights(3) == {"synced": 3}
    assert events[:4] == [("sync", 3), "sleep", ("checkpoint", 3), "wake"]
    assert events[4][0] == "metrics"
    assert events[4][1]["weight_sync_seconds"] >= 0.0
    assert events[4][1]["checkpoint_seconds"] >= 0.0


def test_tgvf_checkpoint_resyncs_discarded_weights_instead_of_bare_wake() -> None:
    events: list[object] = []

    class UpstreamManager:
        requires_post_checkpoint_weight_resync = True

        def update_weights(self, step):
            events.append(("sync", step))
            return {"synced": step}

        def sleep_replicas(self):
            events.append("sleep")

        def wake_up_replicas(self):
            events.append("wake")

    class Trainer:
        _policy_checkpoint_pending = True

        def _commit_policy_checkpoint_after_weight_sync(self, step):
            events.append(("checkpoint", step))
            self._policy_checkpoint_pending = False

        def _complete_policy_metric_publication(self, **timings):
            events.append(("metrics", timings))

    trainer = Trainer()
    manager = CheckpointAfterWeightSyncManager(UpstreamManager(), trainer)

    assert manager.update_weights(3) == {"synced": 3}
    assert events[:4] == [
        ("sync", 3),
        "sleep",
        ("checkpoint", 3),
        ("sync", 3),
    ]
    assert "wake" not in events
    assert manager._replicas_sleeping is False
    assert events[4][0] == "metrics"


def test_full_qwen_crop_checkpoint_resyncs_without_custom_manager_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class UpstreamManager:
        def update_weights(self, step):
            events.append(("sync", step))
            return {"synced": step}

        def sleep_replicas(self):
            events.append("sleep")

        def wake_up_replicas(self):
            events.append("wake")

    class Trainer:
        _policy_checkpoint_pending = True

        def _commit_policy_checkpoint_after_weight_sync(self, step):
            events.append(("checkpoint", step))
            self._policy_checkpoint_pending = False

        def _complete_policy_metric_publication(self, **timings):
            events.append(("metrics", timings))

    monkeypatch.setattr(
        policy_task_runner,
        "_publish_full_qwen_version_if_required",
        lambda _trainer, _step: object(),
    )
    manager = CheckpointAfterWeightSyncManager(UpstreamManager(), Trainer())

    assert manager.update_weights(1) == {"synced": 1}
    assert events[:4] == [
        ("sync", 1),
        "sleep",
        ("checkpoint", 1),
        ("sync", 1),
    ]
    assert "wake" not in events
    assert manager._replicas_sleeping is False
    assert events[4][0] == "metrics"


def test_checkpoint_lifecycle_wraps_upstream_save_in_prepare_finalize_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeLifecycle:
        def prepare_for_save(self, step):
            events.append(("prepare", step))

        def finalize_saved_checkpoint(self, step):
            events.append(("finalize", step))

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            events.append(("upstream_save", self.global_steps))
            return "saved"

    monkeypatch.setattr(policy_task_runner, "PolicyCheckpointLifecycle", FakeLifecycle)
    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.global_steps = 3
    trainer._policy_checkpoint_pending = True
    trainer._policy_checkpoint_lifecycle = FakeLifecycle()

    assert trainer._commit_policy_checkpoint_after_weight_sync(3) == "saved"
    assert events == [
        ("prepare", 3),
        ("upstream_save", 3),
        ("finalize", 3),
    ]
    assert trainer._policy_checkpoint_pending is False


def test_resumed_formal_step2_uses_runtime_lifecycle_checkpoint_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step-1 resume must persist step 2 without changing TOML identity.

    PRL15 deliberately preserves the original scientific milestone schedule
    in its serialized run config.  The formal launcher expands only the
    execution schedule to every completed step.  This models the first update
    after resuming step 1 and ensures the runtime lifecycle, rather than the
    preserved TOML milestones, owns both the save gate and retention calls.
    """

    events: list[object] = []

    class FakeLifecycle:
        checkpoint_steps = tuple(range(9))
        every_completed_step = True

        def prepare_for_save(self, step):
            events.append(("prepare", step))

        def finalize_saved_checkpoint(self, step):
            events.append(("finalize", step))

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            events.append(("upstream_save", self.global_steps))
            return "saved"

    monkeypatch.setattr(policy_task_runner, "PolicyCheckpointLifecycle", FakeLifecycle)
    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.global_steps = 2
    trainer._policy_checkpoint_pending = False
    trainer._policy_checkpoint_state = SimpleNamespace(
        # The source TOML remains byte-identical and names only scientific
        # milestones; it intentionally does not include step 2.
        effective_checkpoint_steps=(0, 1, 4, 8),
        last_resume=SimpleNamespace(optimizer_step=1),
    )
    trainer._policy_checkpoint_lifecycle = FakeLifecycle()

    assert trainer._save_checkpoint() is None
    assert trainer._policy_checkpoint_pending is True
    assert trainer._commit_policy_checkpoint_after_weight_sync(2) == "saved"
    assert events == [
        ("prepare", 2),
        ("upstream_save", 2),
        ("finalize", 2),
    ]
    assert trainer._policy_checkpoint_pending is False


def test_policy_metrics_publish_step_and_cumulative_records_idempotently(
    tmp_path,
) -> None:
    rows = tuple(
        PilotTrajectoryMetricsObservation(
            prompt_id="prompt-a",
            trajectory_id=f"trajectory-{index}",
            generated_policy_tokens=10 + index,
            successful_tgvf_observations=1,
            tool_call_attempts=1,
            answer_reward=float(index % 2 == 0),
            format_error=index == 0,
            conditional_tool_reward=float(index % 2 == 0),
            reasoning_tokens=5 + index,
            original_visual_tokens=4,
            total_visual_tokens=6,
        )
        for index in range(8)
    )
    observation = PilotOptimizerStepMetricsObservation(1, 2.5, rows)
    accumulator = PilotMetricsAccumulator()
    summary = accumulator.record_optimizer_step(observation)
    event = _pilot_metrics_event(observation, summary)
    event["timing"] = {
        "weight_sync_seconds": 0.25,
        "checkpoint_seconds": 0.5,
        "end_to_end_step_seconds": 3.25,
    }

    flat = _wandb_metrics_from_event(event)
    assert flat["policy_pilot/trajectories"] == 8
    assert flat["policy_pilot/mean_tool_call_attempts"] == 1.0
    assert flat["policy_pilot/mean_answer_reward"] == 0.5
    assert flat["policy_pilot_total/generated_policy_tokens"] == 108

    path = tmp_path / "metrics.jsonl"
    _append_policy_metrics_event(path, event)
    _append_policy_metrics_event(path, event)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert '"optimizer_step":1' in lines[0]

    changed = dict(event)
    changed["timing"] = dict(event["timing"], checkpoint_seconds=0.75)
    with pytest.raises(RuntimeError, match="changed an existing step"):
        _append_policy_metrics_event(path, changed)


def test_metric_extraction_validates_six_call_cap_identity_without_clamping() -> None:
    configured_cap_sha256 = "a" * 64
    trajectory = SimpleNamespace(
        tool_calls=tuple(SimpleNamespace(attempt_index=index) for index in range(6)),
        tool_errors=(
            SimpleNamespace(
                attempt_index=6,
                code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
                payload_sha256=configured_cap_sha256,
            ),
        ),
    )

    _validate_metric_tool_attempt_contract(
        trajectory,
        maximum_tool_calls=6,
        cap_error_sha256=configured_cap_sha256,
    )

    with pytest.raises(ValueError, match="payload identity"):
        _validate_metric_tool_attempt_contract(
            trajectory,
            maximum_tool_calls=6,
            cap_error_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="successful tool attempt"):
        _validate_metric_tool_attempt_contract(
            SimpleNamespace(
                tool_calls=(SimpleNamespace(attempt_index=6),),
                tool_errors=(),
            ),
            maximum_tool_calls=6,
            cap_error_sha256=configured_cap_sha256,
        )


def test_policy_metrics_path_separates_formal_and_smoke_roots(tmp_path) -> None:
    config = object.__new__(policy_task_runner.PolicyE2ESmokeRunConfig)
    object.__setattr__(
        config,
        "output",
        SimpleNamespace(
            root=tmp_path,
            metrics_path=tmp_path / "metrics.jsonl",
        ),
    )

    assert _resolved_policy_metrics_path(config, environment={}) == (
        tmp_path / "metrics.jsonl"
    )
    smoke = tmp_path / "smoke/metrics.jsonl"
    assert (
        _resolved_policy_metrics_path(
            config,
            environment={POLICY_METRICS_PATH_ENV: str(smoke)},
        )
        == smoke
    )
    labeled_smoke = tmp_path / "smoke/actor-rollout-only-v1/metrics.jsonl"
    assert (
        _resolved_policy_metrics_path(
            config,
            environment={POLICY_METRICS_PATH_ENV: str(labeled_smoke)},
        )
        == labeled_smoke
    )
    canary = tmp_path / "canary/metrics.jsonl"
    assert (
        _resolved_policy_metrics_path(
            config,
            environment={POLICY_METRICS_PATH_ENV: str(canary)},
        )
        == canary
    )
    with pytest.raises(ValueError, match="inside output.root"):
        _resolved_policy_metrics_path(
            config,
            environment={
                POLICY_METRICS_PATH_ENV: str(tmp_path.parent / "metrics.jsonl")
            },
        )
    with pytest.raises(ValueError, match="formal, smoke, or canary"):
        _resolved_policy_metrics_path(
            config,
            environment={POLICY_METRICS_PATH_ENV: str(tmp_path / "other.jsonl")},
        )


def test_functional_canary_requires_tgvf_replay_before_optimizer_mutation() -> None:
    def observation(successful: int) -> PilotOptimizerStepMetricsObservation:
        rows = tuple(
            PilotTrajectoryMetricsObservation(
                prompt_id="canary-prompt",
                trajectory_id=f"canary-trajectory-{index}",
                generated_policy_tokens=10,
                successful_tgvf_observations=successful if index == 0 else 0,
                tool_call_attempts=successful if index == 0 else 0,
                answer_reward=1.0,
                format_error=False,
                conditional_tool_reward=(float(successful) if index == 0 else 0.0),
                reasoning_tokens=5,
                original_visual_tokens=4,
                total_visual_tokens=6,
            )
            for index in range(2)
        )
        return PilotOptimizerStepMetricsObservation(
            1, 1.0, rows, trajectories_per_prompt=2
        )

    gate = {POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV: "1"}
    _require_successful_tgvf_observation_for_canary(observation(0), environment={})
    with pytest.raises(RuntimeError, match="optimizer was not mutated"):
        _require_successful_tgvf_observation_for_canary(
            observation(0), environment=gate
        )
    _require_successful_tgvf_observation_for_canary(observation(1), environment=gate)
    with pytest.raises(ValueError, match="must be 1 when set"):
        _require_successful_tgvf_observation_for_canary(
            observation(1),
            environment={POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV: "true"},
        )


def test_stage3_metrics_publish_five_components_and_judge_coverage() -> None:
    rows = tuple(
        PilotTrajectoryMetricsObservation(
            prompt_id="prompt-stage3",
            trajectory_id=f"stage3-trajectory-{index}",
            generated_policy_tokens=12,
            successful_tgvf_observations=1,
            tool_call_attempts=1,
            answer_reward=1.0,
            format_error=False,
            conditional_tool_reward=0.0,
            reasoning_tokens=6,
            original_visual_tokens=4,
            total_visual_tokens=6,
            reward_profile="stage3-shaped-v1",
            stage3_reward_components=(
                2.0,
                0.5,
                1.0 if index < 4 else 0.0,
                1.0 if index < 4 else 0.0,
                0.0,
            ),
            stage3_quality_judge_applicable=True,
            stage3_quality_judge_covered=index < 4,
            stage3_quality_judge_failure=(None if index < 4 else "transport"),
            stage3_visual_judge_calls=1,
            stage3_visual_judge_prompt_tokens=10 if index < 4 else 0,
            stage3_visual_judge_completion_tokens=2 if index < 4 else 0,
            stage3_visual_judge_cost_usd=0.001 if index < 4 else 0.0,
        )
        for index in range(8)
    )
    observation = PilotOptimizerStepMetricsObservation(1, 1.0, rows)
    summary = PilotMetricsAccumulator(maximum_tool_calls=1).record_optimizer_step(
        observation
    )

    event = _pilot_metrics_event(observation, summary)
    flat = _wandb_metrics_from_event(event)

    assert flat["policy_pilot/mean_stage3_answer_reward"] == 2.0
    assert flat["policy_pilot/mean_stage3_tool_reward"] == 0.5
    assert flat["policy_pilot/mean_stage3_focus_reward"] == 0.5
    assert flat["policy_pilot/mean_stage3_grounding_reward"] == 0.5
    assert flat["policy_pilot/mean_stage3_protocol_reward"] == 0.0
    assert flat["policy_pilot/group_reward_variance_mean"] == pytest.approx(8 / 7)
    assert flat["policy_pilot/group_reward_std_mean"] == pytest.approx((8 / 7) ** 0.5)
    assert flat["policy_pilot/zero_reward_variance_group_rate"] == 0.0
    assert flat["policy_pilot/format_mixed_group_rate"] == 0.0
    assert flat["policy_pilot/group_format_reward_std_mean"] == 0.0
    assert flat["policy_pilot/stage3_quality_judge_applicable"] == 8
    assert flat["policy_pilot/stage3_quality_judge_covered"] == 4
    assert flat["policy_pilot/stage3_quality_judge_failures"] == 4
    assert flat["policy_pilot/stage3_quality_judge_coverage"] == 0.5
    assert flat["policy_pilot/stage3_visual_judge_calls"] == 8
    assert flat["policy_pilot/stage3_visual_judge_prompt_tokens"] == 40
    assert flat["policy_pilot/stage3_visual_judge_completion_tokens"] == 8
    assert flat["policy_pilot/stage3_visual_judge_cost_usd"] == pytest.approx(0.004)
    assert (
        _policy_tracking_metrics(flat)["policy_pilot/mean_stage3_grounding_reward"]
        == 0.5
    )


def test_policy_tracking_keeps_only_compact_operator_metrics() -> None:
    compact = _policy_tracking_metrics(
        {
            "training/global_step": 3,
            "actor/pg_loss": 0.25,
            "actor/grad_norm": 0.5,
            "policy_pilot/mean_answer_reward": 0.75,
            "policy_pilot/judge_cost_usd": 0.0123,
            "policy_timing/end_to_end_step_seconds": 12.0,
            "global_seqlen/min": 100,
            "policy_pilot_total/generated_policy_tokens": 1_000,
            "timing_per_token_ms/adv": 0.001,
        }
    )

    assert compact == {
        "training/global_step": 3,
        "actor/pg_loss": 0.25,
        "actor/grad_norm": 0.5,
        "policy_pilot/mean_answer_reward": 0.75,
        "policy_pilot/judge_cost_usd": 0.0123,
        "policy_timing/end_to_end_step_seconds": 12.0,
    }


def test_actor_worker_group_routes_save_and_clean_resume_through_pair() -> None:
    events: list[object] = []

    class State:
        last_resume = None

        def prepare_checkpoint(self, step):
            events.append(("prepare", step))

    class Pair:
        def save_checkpoint(self, *args):
            events.append(("save", args))
            return "saved"

        def load_checkpoint(self, *args):
            events.append(("load", args))
            return "resumed"

    proxy = object.__new__(PairedActorWorkerGroup)
    proxy.upstream = SimpleNamespace(marker="upstream")
    proxy.state = State()
    proxy.paired = Pair()

    assert proxy.save_checkpoint("/tmp/checkpoint", None, 4, 2) == "saved"
    assert proxy.load_checkpoint("/tmp/checkpoint", None, False) == "resumed"
    assert proxy.state.last_resume == "resumed"
    assert proxy.marker == "upstream"
    assert events == [
        ("prepare", 4),
        ("save", ("/tmp/checkpoint", None, 4, 2)),
        ("load", ("/tmp/checkpoint", None, False)),
    ]


def test_actor_update_rejects_invalid_batch_before_optimizer_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            events.append("optimizer_mutation")

        def _save_checkpoint(self):
            return None

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer._policy_step_started_at = None
    monkeypatch.setattr(
        policy_task_runner,
        "validate_data_proto_integrity",
        lambda _batch: (_ for _ in ()).throw(ValueError("invalid exact batch")),
    )
    monkeypatch.setattr(
        policy_task_runner,
        "release_verl_data_proto_sidecars",
        lambda _batch: events.append("release"),
    )

    with pytest.raises(ValueError, match="invalid exact batch"):
        trainer._update_actor(object())

    assert events == ["release"]


def test_post_commit_metrics_and_cleanup_failures_do_not_reject_the_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    output = SimpleNamespace(meta_info={"metrics": {}})

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            events.append("optimizer_mutation")
            return output

        def _save_checkpoint(self):
            return None

    class State:
        config = SimpleNamespace(
            policy=SimpleNamespace(sampling=SimpleNamespace(trajectories_per_prompt=16))
        )

        def prepare_optimizer_step(self, _batch):
            events.append("prepare")
            return "prepared"

        def commit_optimizer_step(self, prepared, *, elapsed_seconds):
            assert prepared == "prepared"
            assert elapsed_seconds > 0
            events.append("commit_recovery_boundary")
            return object(), object()

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer._policy_checkpoint_state = State()
    trainer._policy_step_started_at = None
    trainer._policy_metrics_pending = None
    trainer._policy_actor_update_inflight = False
    monkeypatch.setattr(
        policy_task_runner, "validate_data_proto_integrity", lambda _: None
    )
    monkeypatch.setattr(
        policy_task_runner,
        "validate_policy_pilot_reward_data_proto",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_task_runner,
        "_pilot_metrics_event",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("metrics failed")),
    )
    monkeypatch.setattr(
        policy_task_runner,
        "release_verl_data_proto_sidecars",
        lambda _batch: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.warns(RuntimeWarning) as captured:
        observed = trainer._update_actor(object())

    assert observed is output
    assert events == ["prepare", "optimizer_mutation", "commit_recovery_boundary"]
    assert trainer._policy_actor_update_inflight is False
    assert trainer._policy_step_started_at is None
    messages = tuple(str(item.message) for item in captured)
    assert any("metrics failed" in message for message in messages)
    assert any("cleanup failed" in message for message in messages)


def test_recovery_checkpoint_restores_last_completed_data_cursor() -> None:
    class Loader:
        def __init__(self) -> None:
            self.value = {"next_batch": 7}

        def load_state_dict(self, value):
            self.value = value

    loader = Loader()
    state = object.__new__(PolicyPilotTrainerCheckpointState)
    state.trainer = SimpleNamespace(train_dataloader=loader)
    state.metrics_accumulator = SimpleNamespace(
        state=SimpleNamespace(optimizer_steps=6)
    )
    state._recovery_progress = PilotOptimizerDataCursor(
        6,
        _torch_state(DATA_CURSOR_OWNER, {"next_batch": 6}),
    )
    state._recovery_sampler = "sampler-at-6"
    state._recovery_rng = "rng-at-6"
    state._prepared_policy = object()

    state.restore_recovery_cursor_for_checkpoint(6)

    assert loader.value == {"next_batch": 6}
    assert state.progress().optimizer_step == 6
    assert state.rollout_sampler_state() == "sampler-at-6"
    assert state.rollout_rng_state() == "rng-at-6"
    assert state._prepared_policy is None


def test_policy_fit_saves_last_completed_boundary_before_reraising_failure() -> None:
    events: list[object] = []

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            events.append(("save", self.global_steps))

        def fit(self):
            self.global_steps = 7
            raise RuntimeError("judge exhausted retries")

    class State:
        def recovery_optimizer_step(self):
            return 6

        def restore_recovery_cursor_for_checkpoint(self, step):
            events.append(("restore_cursor", step))

    class CheckpointManager:
        def quiesce_after_training_failure(self):
            events.append("quiesce")

    class RuntimeManager:
        def shutdown(self):
            events.append("runtime_shutdown")

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_mode="disable", resume_from_path=None)
    )
    trainer._policy_checkpoint_state = State()
    trainer.checkpoint_manager = CheckpointManager()
    trainer.llm_server_manager = RuntimeManager()
    trainer._policy_checkpoint_pending = False
    trainer._policy_metrics_pending = None
    trainer._policy_actor_update_inflight = False

    with pytest.raises(RuntimeError, match="judge exhausted retries"):
        trainer.fit()

    assert trainer.global_steps == 7
    assert events == [
        ("restore_cursor", 6),
        "quiesce",
        ("save", 6),
        "runtime_shutdown",
    ]


def test_policy_fit_preserves_training_and_recovery_checkpoint_failures() -> None:
    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            raise OSError("checkpoint disk failure")

        def fit(self):
            self.global_steps = 7
            raise RuntimeError("judge exhausted retries")

    class State:
        def recovery_optimizer_step(self):
            return 6

        def restore_recovery_cursor_for_checkpoint(self, _step):
            return None

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_mode="disable", resume_from_path=None)
    )
    trainer._policy_checkpoint_state = State()
    trainer.checkpoint_manager = SimpleNamespace(
        quiesce_after_training_failure=lambda: None
    )
    trainer.llm_server_manager = SimpleNamespace(shutdown=lambda: None)
    trainer._policy_checkpoint_pending = False
    trainer._policy_metrics_pending = None
    trainer._policy_actor_update_inflight = False

    with pytest.raises(ExceptionGroup) as captured:
        trainer.fit()

    messages = tuple(str(error) for error in captured.value.exceptions)
    assert messages == (
        "judge exhausted retries",
        "checkpoint disk failure",
    )


def test_completed_resume_checkpoint_exits_without_an_extra_update(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("1")

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            events.append("optimizer_mutation")

        def _save_checkpoint(self):
            return None

        def fit(self):
            events.append("upstream_fit")

        def _load_checkpoint(self):
            events.append("load")
            self.global_steps = 1
            self._policy_checkpoint_state.last_resume = SimpleNamespace(
                optimizer_step=1
            )

        def _shutdown_dump_executor(self):
            events.append("shutdown")

    class CheckpointManager:
        def update_weights(self, step):
            events.append(("sync", step))

    class RuntimeManager:
        def shutdown(self):
            events.append("runtime_shutdown")

    class FakeLifecycle:
        def finalize_saved_checkpoint(self, step):
            events.append(("finalize", step))

    monkeypatch.setattr(policy_task_runner, "PolicyCheckpointLifecycle", FakeLifecycle)

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            resume_mode="auto",
            resume_from_path=None,
            default_hdfs_dir=None,
            default_local_dir=str(checkpoint_root),
        )
    )
    trainer.total_training_steps = 1
    trainer._policy_checkpoint_state = SimpleNamespace(last_resume=None)
    trainer._policy_checkpoint_lifecycle = FakeLifecycle()
    trainer.checkpoint_manager = CheckpointManager()
    trainer.llm_server_manager = RuntimeManager()

    assert _completed_resume_checkpoint_step(trainer) == 1
    assert trainer.fit() is None
    assert events == [
        "load",
        ("sync", 1),
        ("finalize", 1),
        "shutdown",
        "runtime_shutdown",
    ]


def test_policy_fit_closes_dataloader_workers_and_runtime_after_failure() -> None:
    events: list[str] = []

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            return None

        def fit(self):
            raise RuntimeError("training failed")

    class Iterator:
        def _shutdown_workers(self):
            events.append("dataloader_shutdown")

    class RuntimeManager:
        def shutdown(self):
            events.append("runtime_shutdown")

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_mode="disable", resume_from_path=None)
    )
    trainer.train_dataloader = SimpleNamespace(_iterator=Iterator())
    trainer.val_dataloader = SimpleNamespace(_iterator=None)
    trainer.llm_server_manager = RuntimeManager()

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit()

    assert trainer.train_dataloader._iterator is None
    assert events == ["dataloader_shutdown", "runtime_shutdown"]


def test_policy_tracking_backends_finish_once_before_ray_teardown() -> None:
    events: list[object] = []

    class Wandb:
        def finish(self, *, exit_code):
            events.append(("wandb", exit_code))

    class File:
        def finish(self):
            events.append("file")

    tracker = SimpleNamespace(
        logger={"wandb": Wandb(), "file": File(), "console": object()}
    )

    _finish_tracking_backends(tracker)

    assert events == [("wandb", 0), "file"]
    assert tracker.logger == {}


def test_policy_fit_captures_and_finishes_upstream_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verl.utils import tracking as tracking_module

    events: list[object] = []

    class Wandb:
        def finish(self, *, exit_code):
            events.append(("wandb", exit_code))

    class Tracking:
        def __init__(self):
            self.logger = {"wandb": Wandb()}

    monkeypatch.setattr(tracking_module, "Tracking", Tracking)

    class UpstreamTrainer:
        def init_workers(self):
            return None

        def _get_gen_batch(self):
            return None

        def _update_actor(self, _batch):
            return None

        def _save_checkpoint(self):
            return None

        def fit(self):
            from verl.utils.tracking import Tracking as RuntimeTracking

            RuntimeTracking()
            events.append("upstream_fit")

    class RuntimeManager:
        def shutdown(self):
            events.append("runtime_shutdown")

    trainer_cls = make_policy_pilot_ray_trainer_class(UpstreamTrainer)
    trainer = object.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_mode="disable", resume_from_path=None)
    )
    trainer.llm_server_manager = RuntimeManager()

    assert trainer.fit() is None
    assert events == ["upstream_fit", ("wandb", 0), "runtime_shutdown"]
    assert tracking_module.Tracking is Tracking
