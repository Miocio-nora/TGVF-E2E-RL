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
    add_policy_actor_rollout_worker,
    make_policy_pilot_ray_trainer_class,
)
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
    non_tensor_batch[EXACT_PROMPT_IDS_FIELD] = np.array(
        [(1,), (2,)], dtype=object
    )
    non_tensor_batch[EXACT_RESPONSE_IDS_FIELD] = np.array(
        [(3,), (4,)], dtype=object
    )
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
    assert data.meta_info[SIDECAR_RELEASE_SCHEMA_FIELD] == SIDECAR_RELEASE_SCHEMA_VERSION
    assert data.meta_info[SIDECAR_RELEASE_FIELDS_FIELD] == AGENT_LOOP_EXACT_SIDECAR_FIELDS
    assert release_verl_data_proto_sidecars(data) == len(AGENT_LOOP_EXACT_SIDECAR_FIELDS)
    assert release_verl_data_proto_sidecars(data) == 0
    assert data.non_tensor_batch == {}


def test_task_runner_maps_the_real_sidecar_releasing_role_worker() -> None:
    calls: list[object] = []

    class TrainingWorker:
        def train_mini_batch(self, data):
            calls.append(("train", data))

        def infer_batch(self, data):
            calls.append(("infer", data))

    class ActorRolloutRefWorker:
        actor_worker_cls = TrainingWorker
        ref_worker_cls = TrainingWorker

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

    trainer = Trainer()
    manager = CheckpointAfterWeightSyncManager(UpstreamManager(), trainer)

    assert manager.update_weights(3) == {"synced": 3}
    assert events == [("sync", 3), "sleep", ("checkpoint", 3), "wake"]


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
