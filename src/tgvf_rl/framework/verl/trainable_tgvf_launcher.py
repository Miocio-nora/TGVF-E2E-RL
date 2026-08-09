"""Pinned-veRL launcher for the matched full-Qwen plus trainable-RP66 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_CANDIDATE_SIDECAR,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_ROOT,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_TRAIN_SEED,
)
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import visual_tool_prompt_identity
from tgvf_rl.protocol.native import native_assistant_dialect_for_model

from . import launcher as legacy_launcher
from .launcher import (
    UPSTREAM_VERL_CONFIG_NAME,
    UPSTREAM_VERL_MAIN_MODULE,
    UPSTREAM_VERL_V0_RUNNER_FQN,
)
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from .policy_main import compose_pinned_verl_config
from .policy_task_runner import (
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
)
from .prl14_crop16_reference import (
    PRL14_CROP16_COMMON_OVERRIDES,
    PRL14_CROP16_COMPLETION_SHA256,
    PRL14_CROP16_REMOVED_OVERRIDES,
    PRL14_CROP16_RUN_NAME,
    apply_prl14_crop16_common_controls,
)
from .tgvf_deepeyes_matched_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    DeepEyesTGVFMatchedDatasetBinding,
    TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from .trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from .trainable_tgvf_engine import TRAINABLE_TGVF_MODEL_TYPE


TRAINABLE_TGVF_LAUNCH_SCHEMA = "tgvf.trainable-rp66-verl-launch.v1"
TRAINABLE_TGVF_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_tgvf_external"
TRAINABLE_TGVF_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset"
)
TRAINABLE_TGVF_FORMAL_TARGET = 8
TRAINABLE_TGVF_SMOKE_TARGET = 1
TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES = frozenset({4, 8})
TrainableTGVFLaunchMode = Literal["formal", "smoke"]


# World4 keeps Crop-16's global BS16, n16 and fixed actor trajectory micro32.
# Each rank therefore accumulates two complete micros before the one optimizer
# update instead of world8's one micro per rank.  Only physical execution
# topology changes; the equal-micro DeepEyes loss still averages eight micros.
_WORLD4_TOPOLOGY_OVERRIDES = MappingProxyType(
    {
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.rollout.agent.num_workers": 4,
        "trainer.n_gpus_per_node": 4,
    }
)


def _apply_crop16_mathematical_controls(
    values: dict[str, object], *, world_size: int, optimizer_horizon: int
) -> None:
    if world_size not in TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES:
        raise ValueError("trainable TGVF topology must use world4 or world8")
    if type(optimizer_horizon) is not int or optimizer_horizon <= 0:
        raise ValueError("trainable TGVF optimizer horizon must be positive")
    apply_prl14_crop16_common_controls(values)
    # PRL14's serialized completion contains -1 because its upstream trainer
    # filled the scheduler horizon later.  The project TaskRunner deliberately
    # preserves an explicit run-bound horizon before upstream construction.
    values["actor_rollout_ref.actor.optim.total_training_steps"] = (
        optimizer_horizon
    )
    if world_size == 4:
        values.update(_WORLD4_TOPOLOGY_OVERRIDES)


def _assert_crop16_mathematical_controls(
    values: Mapping[str, object], *, world_size: int, optimizer_horizon: int
) -> None:
    if world_size not in TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES:
        raise ValueError("trainable TGVF topology must use world4 or world8")
    if type(optimizer_horizon) is not int or optimizer_horizon <= 0:
        raise ValueError("trainable TGVF optimizer horizon must be positive")
    expected = dict(PRL14_CROP16_COMMON_OVERRIDES)
    expected["actor_rollout_ref.actor.optim.total_training_steps"] = (
        optimizer_horizon
    )
    if world_size == 4:
        expected.update(_WORLD4_TOPOLOGY_OVERRIDES)
    mismatches = {
        path: (values.get(path), value)
        for path, value in expected.items()
        if values.get(path) != value
    }
    unexpected = {
        path: values[path]
        for path in PRL14_CROP16_REMOVED_OVERRIDES
        if path in values
    }
    if mismatches or unexpected:
        raise ValueError(
            "trainable TGVF differs from the Crop-16 mathematical controls: "
            f"mismatches={mismatches!r}, unexpected={unexpected!r}"
        )


def _plain_binding(binding: DeepEyesTGVFMatchedDatasetBinding) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(binding).items()
    }


def _matched_dataset_binding(
    config: PolicyE2ESmokeRunConfig,
) -> DeepEyesTGVFMatchedDatasetBinding:
    return DeepEyesTGVFMatchedDatasetBinding(
        root=DEEPEYES_T1_ROOT,
        candidate_sidecar_path=DEEPEYES_CANDIDATE_SIDECAR,
        manifest_file_sha256=DEEPEYES_T1_MANIFEST_FILE_SHA256,
        content_sha256=DEEPEYES_T1_CONTENT_SHA256,
        samples_sha256=DEEPEYES_T1_SAMPLES_SHA256,
        candidate_sidecar_sha256=DEEPEYES_CANDIDATE_SHA256,
        expected_sample_count=DEEPEYES_T1_SAMPLE_COUNT,
        schedule_mode="stratified",
        schedule_seed=DEEPEYES_TRAIN_SEED,
        probe_seed=DEEPEYES_PROBE_SEED,
        visual_prompt_bundle_sha256=(
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        ),
        thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
        model_name=config.model.model_name,
        tokenizer_length=config.model.tokenizer_length,
        chat_template_sha256=config.model.chat_template_sha256,
    )


def _legacy_base_plan(config: PolicyE2ESmokeRunConfig):
    """Build e003's complete base map with its historical prompt binding only."""

    legacy_prompt = visual_tool_prompt_identity(
        config.protocol.tool_profile,
        assistant_dialect=native_assistant_dialect_for_model(config.model.model_name),
    ).bundle_sha256
    compatible = replace(
        config,
        protocol=replace(config.protocol, prompt_sha256=legacy_prompt),
    )
    return legacy_launcher.build_policy_e2e_smoke_verl_plan(compatible)


def _replace_custom_record(
    values: dict[str, object],
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode,
    checkpoint_steps: tuple[int, ...],
    output_root: Path,
) -> None:
    custom = dict(values["actor_rollout_ref.rollout.custom"])  # type: ignore[arg-type]
    protocol = dict(custom["protocol"])  # type: ignore[arg-type]
    protocol.update(
        {
            "prompt_sha256": config.protocol.prompt_sha256,
            "maximum_tool_calls": 6,
        }
    )
    reward = dict(custom["reward"])  # type: ignore[arg-type]
    reward.update(
        {
            "schema_version": POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
            "source_aware": True,
        }
    )
    custom.update(
        {
            "schema_version": TRAINABLE_TGVF_LAUNCH_SCHEMA,
            "protocol": protocol,
            "reward": reward,
            "trainable_tgvf": {
                "external_module": TRAINABLE_TGVF_EXTERNAL_MODULE,
                "model_type": TRAINABLE_TGVF_MODEL_TYPE,
                "checkpoint_manager_fqn": (
                    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
                ),
                "policy_lora": False,
                "vision_trainable": True,
                "adapter_trainable": True,
                "sync_every_optimizer_step": True,
            },
            "weight_sync": {
                "mode": config.distributed.weight_sync_mode,
                "interval_optimizer_steps": 1,
                "payload": "full_qwen_plus_trainable_rp66",
            },
            "reference_diagnostic": {
                "enabled": False,
                "coefficient": 0.0,
                "worker_route": "disabled_zero_kl_control",
                "observation_source": "not_computed",
            },
            "checkpoint_steps": list(checkpoint_steps),
            "runtime_state_directory": str(output_root / "runtime-policy-state"),
            "metrics_path": str(output_root / "metrics.jsonl"),
            "launch_mode": mode,
        }
    )
    values["actor_rollout_ref.rollout.custom"] = custom


@dataclass(frozen=True, slots=True)
class TrainableTGVFVerlLaunchPlan:
    config: PolicyE2ESmokeRunConfig
    mode: TrainableTGVFLaunchMode
    target_step: int
    overrides: Mapping[str, object]
    environment: Mapping[str, str]
    external_components: Mapping[str, str]
    schema_version: str = TRAINABLE_TGVF_LAUNCH_SCHEMA
    main_module: str = UPSTREAM_VERL_MAIN_MODULE
    config_name: str = UPSTREAM_VERL_CONFIG_NAME
    runner_fqn: str = UPSTREAM_VERL_V0_RUNNER_FQN

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(
            self,
            "external_components",
            MappingProxyType(dict(self.external_components)),
        )
        expected = (
            TRAINABLE_TGVF_FORMAL_TARGET
            if self.mode == "formal"
            else TRAINABLE_TGVF_SMOKE_TARGET
            if self.mode == "smoke"
            else None
        )
        if self.schema_version != TRAINABLE_TGVF_LAUNCH_SCHEMA or expected is None:
            raise ValueError("trainable TGVF launch identity differs")
        if self.target_step != expected:
            raise ValueError(f"{self.mode} target must be step {expected}")
        self._assert_trainable_path()

    def _assert_trainable_path(self) -> None:
        values = self.overrides
        _assert_crop16_mathematical_controls(
            values,
            world_size=self.config.distributed.world_size,
            optimizer_horizon=self.config.scheduler.total_steps,
        )
        required = {
            "actor_rollout_ref.model.lora_rank": 0,
            "actor_rollout_ref.model.lora.rank": 0,
            "actor_rollout_ref.model.lora.freeze_vision_model": False,
            "actor_rollout_ref.model.lora.freeze_vision_projection": False,
            "actor_rollout_ref.actor.freeze_vision_tower": False,
            "actor_rollout_ref.model.external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "actor_rollout_ref.model.model_type": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            ),
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.actor.loss_agg_mode": NATIVE_DEEPEYES_LOSS_AGG_MODE,
            "data.custom_cls.path": TRAINABLE_TGVF_DATASET_MODULE_PATH,
            "data.custom_cls.name": "TGVFDeepEyesMatchedDataset",
            "data.train_batch_size": 16,
            "actor_rollout_ref.rollout.n": 16,
            "trainer.total_training_steps": self.target_step,
        }
        for path, expected in required.items():
            if values.get(path) != expected:
                raise ValueError(f"trainable TGVF override differs at {path}")
        custom = values.get("actor_rollout_ref.rollout.custom")
        if not isinstance(custom, Mapping):
            raise ValueError("trainable TGVF custom record is missing")
        if custom.get("weight_sync") != {
            "mode": self.config.distributed.weight_sync_mode,
            "interval_optimizer_steps": 1,
            "payload": "full_qwen_plus_trainable_rp66",
        }:
            raise ValueError("trainable TGVF sync identity differs")
        if custom.get("reference_diagnostic") != {
            "enabled": False,
            "coefficient": 0.0,
            "worker_route": "disabled_zero_kl_control",
            "observation_source": "not_computed",
        }:
            raise ValueError("trainable TGVF reference diagnostic must be disabled")
        if self.environment.get(POLICY_REFERENCE_DIAGNOSTIC_ENV) != "0":
            raise ValueError("trainable TGVF reference diagnostic environment differs")

    def hydra_override_args(self) -> tuple[str, ...]:
        return tuple(
            f"++{path}={legacy_launcher._hydra_literal(value)}"
            for path, value in self.overrides.items()
        )


def build_trainable_tgvf_verl_launch_plan(
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode = "formal",
    target_step: int | None = None,
    smoke_id: str | None = None,
) -> TrainableTGVFVerlLaunchPlan:
    if config.schema_version != POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA:
        raise ValueError("trainable TGVF launcher requires the RP66 run schema")
    if mode not in {"formal", "smoke"}:
        raise ValueError(f"unsupported trainable TGVF launch mode: {mode!r}")
    if smoke_id is not None:
        if mode != "smoke":
            raise ValueError("smoke_id is valid only in smoke mode")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", smoke_id) is None:
            raise ValueError("smoke_id must be a safe lowercase run label")
    resolved_target = (
        TRAINABLE_TGVF_FORMAL_TARGET
        if mode == "formal"
        else TRAINABLE_TGVF_SMOKE_TARGET
        if mode == "smoke"
        else -1
    )
    if target_step is not None and target_step != resolved_target:
        raise ValueError(f"{mode} target must be step {resolved_target}")
    base = _legacy_base_plan(config)
    values = dict(base.overrides)
    binding = _matched_dataset_binding(config)
    values.pop("data.tgvf_policy_t1_mixed", None)
    values.update(
        {
            "data.train_files": [str(DEEPEYES_TRAIN_SENTINEL)],
            "data.val_files": [str(DEEPEYES_PROBE_SENTINEL)],
            "data.custom_cls.path": TRAINABLE_TGVF_DATASET_MODULE_PATH,
            "data.custom_cls.name": "TGVFDeepEyesMatchedDataset",
            "data.deepeyes_tgvf_matched": _plain_binding(binding),
            "data.seed": DEEPEYES_TRAIN_SEED,
            "actor_rollout_ref.model.external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "actor_rollout_ref.model.model_type": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_rollout_ref.model.lora_rank": 0,
            "actor_rollout_ref.model.lora.rank": 0,
            "actor_rollout_ref.model.lora.freeze_vision_model": False,
            "actor_rollout_ref.model.lora.freeze_vision_projection": False,
            "actor_rollout_ref.model.lora.freeze_language_model": False,
            "actor_rollout_ref.actor.freeze_vision_tower": False,
            "actor_rollout_ref.actor.loss_agg_mode": NATIVE_DEEPEYES_LOSS_AGG_MODE,
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            ),
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            "trainer.total_training_steps": resolved_target,
            "actor_rollout_ref.actor.optim.total_training_steps": (
                config.scheduler.total_steps
            ),
            "trainer.save_freq": 1,
        }
    )
    _apply_crop16_mathematical_controls(
        values,
        world_size=config.distributed.world_size,
        optimizer_horizon=config.scheduler.total_steps,
    )
    checkpoint_steps = (0, 1, 4, 8) if mode == "formal" else (0, 1)
    output_root = config.output.root
    environment = dict(base.environment)
    if mode == "smoke":
        output_root = output_root / "smoke"
        if smoke_id is not None:
            output_root = output_root / smoke_id
        values.update(
            {
                "trainer.experiment_name": config.run_id + "-SMOKE",
                # One-step engineering canaries belong in the local console
                # log only.  Keep the run config's W&B backend unchanged for
                # the formal eight-step experiment.
                "trainer.logger": ["console"],
                "trainer.default_local_dir": str(output_root / "checkpoints"),
                "trainer.max_actor_ckpt_to_keep": 2,
            }
        )
        environment["TGVF_POLICY_STATE_DIR"] = str(
            output_root / "runtime-policy-state"
        )
    environment[POLICY_METRICS_PATH_ENV] = str(output_root / "metrics.jsonl")
    environment[POLICY_REFERENCE_DIAGNOSTIC_ENV] = "0"
    values["ray_kwargs.ray_init._temp_dir"] = str(output_root / "ray")
    _replace_custom_record(
        values,
        config,
        mode=mode,
        checkpoint_steps=checkpoint_steps,
        output_root=output_root,
    )
    external = dict(base.external_components)
    external.update(
        {
            "control_run": PRL14_CROP16_RUN_NAME,
            "control_completion_sha256": PRL14_CROP16_COMPLETION_SHA256,
            "dataset": TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
            "actor_engine": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "checkpoint_engine_manager": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
        }
    )
    return TrainableTGVFVerlLaunchPlan(
        config=config,
        mode=mode,
        target_step=resolved_target,
        overrides=values,
        environment=environment,
        external_components=external,
    )


def compose_trainable_tgvf_verl_config(plan: TrainableTGVFVerlLaunchPlan) -> object:
    if not isinstance(plan, TrainableTGVFVerlLaunchPlan):
        raise TypeError("plan must be TrainableTGVFVerlLaunchPlan")
    composed = compose_pinned_verl_config(plan.hydra_override_args())
    from omegaconf import OmegaConf

    missing = object()
    for path, expected in plan.overrides.items():
        selected = OmegaConf.select(composed, path, default=missing)
        if selected is missing:
            raise ValueError(f"pinned compose omitted trainable TGVF override {path}")
        actual = (
            OmegaConf.to_container(selected, resolve=True)
            if OmegaConf.is_config(selected)
            else selected
        )
        if actual != legacy_launcher._plain(expected):
            raise ValueError(f"pinned compose changed trainable TGVF override {path}")
    return composed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--smoke-id")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_policy_e2e_smoke_run_config(args.run_config.resolve())
    plan = build_trainable_tgvf_verl_launch_plan(
        config,
        mode=args.mode,
        target_step=args.target_step,
        smoke_id=args.smoke_id,
    )
    os.environ.update(plan.environment)
    if args.compose_only:
        compose_trainable_tgvf_verl_config(plan)
        return
    from .policy_main import main as policy_main

    policy_main(plan.hydra_override_args())


if __name__ == "__main__":
    main()


__all__ = [
    "TRAINABLE_TGVF_EXTERNAL_MODULE",
    "TRAINABLE_TGVF_FORMAL_TARGET",
    "TRAINABLE_TGVF_LAUNCH_SCHEMA",
    "TRAINABLE_TGVF_SMOKE_TARGET",
    "TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES",
    "TrainableTGVFVerlLaunchPlan",
    "build_trainable_tgvf_verl_launch_plan",
    "compose_trainable_tgvf_verl_config",
    "main",
]
