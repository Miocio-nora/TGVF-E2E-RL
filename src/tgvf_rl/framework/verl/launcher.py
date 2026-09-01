"""Read-only launch-plan bridge for the pinned upstream veRL v0 runner.

This module translates one fully validated :class:`PolicyE2ESmokeRunConfig`
into dotted Hydra overrides.  Building or composing a plan has no runtime side
effects: it does not import ``verl.trainer.main_ppo``, initialize Ray/CUDA, or
create output files.  Fields that are not owned by the smoke identity remain
explicit blockers instead of silently inheriting an operational choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType

from tgvf_rl.policy.horizon_extension import PolicyHorizonExtension
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
    PolicyE2ESmokeRunConfig,
)
from tgvf_rl.qwen.deepstack_control import (
    TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD,
)
from tgvf_rl.data import (
    PolicyT1MixedRuntimeBinding,
    PolicyT1RLRuntimeBinding,
    PolicyTeacherQuarterMixRuntimeBinding,
)
from tgvf_rl.framework.vllm.registration import VLLM_012_LORA_PDL_MODE
from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA,
    POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,
    POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA,
    PolicyCompilePrerequisiteBinding,
    PolicyCompilePrerequisiteFileReceipt,
    PolicyCompilePrerequisiteReceipt,
    preflight_policy_compile_prerequisites,
)

from .adapter import LOSSLESS_AGENT_LOOP_MANAGER_FQN, VerlAdapterConfig
from .compatibility import (
    FSDP2BridgeConfig,
    SPIKE_CANDIDATE_VERL_COMMIT,
    VERL_PREFIX_CACHE_WEIGHT_UPDATE_INVALIDATION,
    VLLM_012_PREFIX_CACHE_MM_HASH_IDENTITY,
    VerlRuntimeRequirements,
)
from .dynamic_token_loss_contract import (
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    METHOD_MATRIX_BYPASS_LOSS_MODULE,
    METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME,
)
from .exact_replay_engine import TGVF_EXACT_REPLAY_MODEL_TYPE
from .native_deepeyes_runtime import NATIVE_DEEPEYES_POLICY_LOSS_MODE
from .smoke_dataset import VerlSelectedSampleDatasetBinding
from .deepeyes_dataset import VerlDeepEyes47KDatasetBinding
from .policy_t1_dataset import VerlPolicyT1DatasetBinding
from .policy_t1_mixed_dataset import VerlPolicyT1MixedDatasetBinding
from .policy_teacher_quarter_mix_dataset import (
    POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME,
    POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH,
    PolicyTeacherQuarterMixDatasetBinding,
)


# Pinned e003's upstream main hard-codes its own TaskRunner.  The repo-owned
# module composes the same upstream config and delegates to ``run_ppo`` with the
# lifecycle-wired v0 TaskRunner.
UPSTREAM_VERL_MAIN_MODULE = "tgvf_rl.framework.verl.policy_main"
UPSTREAM_VERL_CONFIG_NAME = "ppo_trainer"
UPSTREAM_VERL_V0_RUNNER_FQN = "verl.trainer.main_ppo_v0.TaskRunner"

SELECTED_SAMPLE_DATASET_CLASS_NAME = "TGVFSelectedSampleDataset"
SELECTED_SAMPLE_DATASET_MODULE_PATH = "pkg://tgvf_rl.framework.verl.smoke_dataset"
DEEPEYES47K_DATASET_CLASS_NAME = "TGVFDeepEyes47KDataset"
DEEPEYES47K_DATASET_MODULE_PATH = "pkg://tgvf_rl.framework.verl.deepeyes_dataset"
POLICY_T1_ARXIVQA_DATASET_CLASS_NAME = "TGVFPolicyT1ArxivQADataset"
POLICY_T1_ARXIVQA_DATASET_MODULE_PATH = "pkg://tgvf_rl.framework.verl.policy_t1_dataset"
POLICY_T1_MIXED_DATASET_CLASS_NAME = "TGVFPolicyT1MixedDataset"
POLICY_T1_MIXED_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.policy_t1_mixed_dataset"
)
POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS_NAME = "PolicyTeacherQuarterMixDataset"
NATIVE_AGENT_LOOP_NAME = "tgvf_native_policy"
NATIVE_AGENT_LOOP_FQN = (
    "tgvf_rl.framework.verl.native_agent_loop.VerlFrameworkNeutralAgentLoop"
)
NATIVE_INVOCATION_FACTORY_FQN = (
    "tgvf_rl.framework.verl.native_agent_loop.BoundVerlNativeAgentLoopInvocationFactory"
)
EXACT_REPLAY_EXTERNAL_MODULE = "tgvf_rl.framework.verl.exact_bypass_loss"
EXACT_REPLAY_ENGINE_REGISTRAR_FQN = (
    "tgvf_rl.framework.verl.exact_replay_engine."
    "register_qwen3_exact_replay_fsdp2_engine"
)
EXACT_REPLAY_FORWARD_PORT_FQN = (
    "tgvf_rl.policy.qwen_replay.Qwen3RecordedPolicyForwardPort"
)
EXACT_CURRENT_REFERENCE_REPLAY_FQN = (
    "tgvf_rl.policy.qwen_replay.replay_qwen3_current_reference"
)
POLICY_REWARD_PIPELINE_FQN = "tgvf_rl.rewards.pipeline.PilotRewardPipeline"
STAGE3_REWARD_PIPELINE_FQN = "tgvf_rl.rewards.stage3_shaped.Stage3ShapedRewardKernel"
POLICY_CHECKPOINT_ENGINE_MANAGER_FQN = (
    "tgvf_rl.framework.verl.policy_weight_sync.TGVFPolicyCheckpointEngineManager"
)

VERL_POLICY_SMOKE_LAUNCH_SCHEMA = "tgvf-verl-policy-smoke-launch-v1"
VLLM_ROLLOUT_DATA_PARALLEL_SIZE = 1
VLLM_ROLLOUT_PIPELINE_PARALLEL_SIZE = 1
VLLM_ROLLOUT_DISAGGREGATION_ENABLED = False


@dataclass(frozen=True, slots=True)
class UpstreamVerlLaunchPlan:
    """Auditable, non-executing description of one upstream invocation."""

    run_identity_sha256: str
    overrides: Mapping[str, object]
    environment: Mapping[str, str]
    compile_prerequisites: PolicyCompilePrerequisiteBinding | None
    external_components: Mapping[str, str]
    launch_blockers: tuple[str, ...]
    inherited_upstream_fields: tuple[str, ...]
    verl_commit: str = SPIKE_CANDIDATE_VERL_COMMIT
    main_module: str = UPSTREAM_VERL_MAIN_MODULE
    config_name: str = UPSTREAM_VERL_CONFIG_NAME
    runner_fqn: str = UPSTREAM_VERL_V0_RUNNER_FQN
    schema_version: str = VERL_POLICY_SMOKE_LAUNCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != VERL_POLICY_SMOKE_LAUNCH_SCHEMA:
            raise ValueError("veRL launch-plan schema differs")
        if self.verl_commit != SPIKE_CANDIDATE_VERL_COMMIT:
            raise ValueError("Policy smoke launch requires the accepted e003 pin")
        if self.main_module != UPSTREAM_VERL_MAIN_MODULE:
            raise ValueError("upstream veRL main module differs")
        if self.config_name != UPSTREAM_VERL_CONFIG_NAME:
            raise ValueError("upstream veRL Hydra config name differs")
        if self.runner_fqn != UPSTREAM_VERL_V0_RUNNER_FQN:
            raise ValueError("accepted e003 launch must use the v0 TaskRunner")
        _require_sha256(self.run_identity_sha256, "run_identity_sha256")
        if self.compile_prerequisites is not None and not isinstance(
            self.compile_prerequisites, PolicyCompilePrerequisiteBinding
        ):
            raise TypeError("compile_prerequisites must be a binding or None")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(
            self,
            "external_components",
            MappingProxyType(dict(self.external_components)),
        )
        object.__setattr__(self, "launch_blockers", tuple(self.launch_blockers))
        object.__setattr__(
            self,
            "inherited_upstream_fields",
            tuple(self.inherited_upstream_fields),
        )
        if self.overrides.get("trainer.use_v1") is not False:
            raise ValueError("accepted e003 launch must retain trainer.use_v1=false")
        if (
            self.overrides.get(
                "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
            )
            != LOSSLESS_AGENT_LOOP_MANAGER_FQN
        ):
            raise ValueError("launch plan lost the accepted lossless v0 manager")
        visible_devices = self.environment.get("CUDA_VISIBLE_DEVICES", "")
        try:
            physical_gpu_ids = tuple(
                int(device) for device in visible_devices.split(",")
            )
        except ValueError as error:
            raise ValueError(
                "Policy launch must bind integer physical GPU IDs"
            ) from error
        if (
            not physical_gpu_ids
            or len(set(physical_gpu_ids)) != len(physical_gpu_ids)
            or any(device < 0 for device in physical_gpu_ids)
        ):
            raise ValueError("Policy launch must bind non-empty unique physical GPUs")
        world_size = len(physical_gpu_ids)
        world_size_overrides = (
            "trainer.n_gpus_per_node",
            "actor_rollout_ref.rollout.n_gpus_per_node",
            "actor_rollout_ref.rollout.agent.num_workers",
        )
        if any(
            type(self.overrides.get(name)) is not int
            or self.overrides.get(name) != world_size
            for name in world_size_overrides
        ):
            raise ValueError(
                "Policy launch GPU and AgentLoop worker counts must match visible "
                "devices"
            )
        tensor_parallel_size = self.overrides.get(
            "actor_rollout_ref.rollout.tensor_model_parallel_size"
        )
        if (
            type(tensor_parallel_size) is not int
            or tensor_parallel_size <= 0
            or world_size % tensor_parallel_size != 0
        ):
            raise ValueError(
                "vLLM tensor parallel size must be positive and divide visible GPUs"
            )
        if (
            self.overrides.get("actor_rollout_ref.actor.fsdp_config.forward_only")
            is not False
        ):
            raise ValueError("Policy actor exact-replay engine must be trainable")
        if (
            self.overrides.get("actor_rollout_ref.ref.fsdp_config.forward_only")
            is not True
        ):
            raise ValueError(
                "Policy reference exact-replay engine must be forward-only"
            )
        if (
            self.overrides.get(
                "actor_rollout_ref.model.override_config.attn_implementation"
            )
            != "sdpa"
        ):
            raise ValueError("Policy actor/reference replay must use explicit SDPA")
        native_deepstack_path = (
            "actor_rollout_ref.model.override_config."
            + TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD
        )
        native_deepstack_enabled = self.overrides.get(native_deepstack_path)
        if type(native_deepstack_enabled) is not bool:
            raise ValueError("Policy native DeepStack control must be explicit bool")
        hf_overrides = self.overrides.get(
            "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"
        )
        if (
            not isinstance(hf_overrides, Mapping)
            or hf_overrides.get(TGVF_NATIVE_DEEPSTACK_ENABLED_CONFIG_FIELD)
            is not native_deepstack_enabled
        ):
            raise ValueError(
                "actor/reference and vLLM native DeepStack controls differ"
            )
        _assert_performance_surface(self.overrides)
        for role in ("actor", "ref"):
            prefix = f"actor_rollout_ref.{role}.fsdp_config"
            if self.overrides.get(f"{prefix}.model_dtype") != "bf16":
                raise ValueError(f"Policy {role} model load dtype must be BF16")
            if self.overrides.get(f"{prefix}.use_torch_compile") is not False:
                raise ValueError(f"Policy {role} smoke must disable torch.compile")
        _assert_checkpoint_and_method_matrix_surface(
            self.overrides,
            self.external_components,
        )
        if self.environment.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError(
                "local AgentLoop workers must retain the physical GPU view"
            )
        if self.environment.get("TGVF_POLICY_RUN_IDENTITY") != self.run_identity_sha256:
            raise ValueError("runtime Policy identity environment differs from plan")
        if (
            self.environment.get("TGVF_POLICY_RUN_IDENTITY_SHA256")
            != self.run_identity_sha256
        ):
            raise ValueError(
                "runtime Policy SHA256 identity environment differs from plan"
            )
        if (
            self.overrides.get("actor_rollout_ref.rollout.full_determinism")
            is not False
        ):
            raise ValueError(
                "TRITON_ATTN multi-turn rollout must disable vLLM batch invariance"
            )
        if self.environment.get("VLLM_BATCH_INVARIANT") != "0":
            raise ValueError(
                "launch plan must explicitly disable vLLM batch invariance"
            )
        if self.environment.get("VERL_FULL_DETERMINISM") != "0":
            raise ValueError(
                "launch plan must isolate rollout from veRL full determinism"
            )
        compile_environment_names = (
            "CC",
            "CXX",
            "CPATH",
            "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256",
            "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_PATH",
            "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256",
        )
        if self.compile_prerequisites is None:
            if any(name in self.environment for name in compile_environment_names):
                raise ValueError("blocked Policy plan leaked compile environment")
            if POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER not in self.launch_blockers:
                raise ValueError(
                    "blocked Policy plan lost its missing-manifest blocker"
                )
        else:
            binding = self.compile_prerequisites
            expected_compile_environment = {
                "CC": str(binding.c_compiler),
                "CXX": str(binding.cxx_compiler),
                "CPATH": binding.cpath,
                "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256": (
                    binding.identity_sha256
                ),
                "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_PATH": str(
                    binding.manifest_source_path
                ),
                "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256": (
                    binding.manifest_source_sha256
                ),
            }
            if any(
                self.environment.get(name) != value
                for name, value in expected_compile_environment.items()
            ):
                raise ValueError("Policy rollout compile binding environment differs")
            if any(
                blocker not in self.launch_blockers
                for blocker in binding.launch_blockers
            ):
                raise ValueError("Policy plan lost a compile residual blocker")
        state_dir = self.environment.get("TGVF_POLICY_STATE_DIR", "")
        if not state_dir.endswith("/runtime-policy-state"):
            raise ValueError("runtime Policy state directory is not explicitly bound")
        if self.inherited_upstream_fields:
            raise ValueError("Policy Pilot plan still inherits launch-critical fields")

    @property
    def launch_ready(self) -> bool:
        return not self.launch_blockers

    def assert_launch_ready(self) -> None:
        if self.launch_blockers:
            raise RuntimeError(
                "upstream veRL launch remains blocked: "
                + "; ".join(self.launch_blockers)
            )

    def preflight_live_prerequisites(self) -> PolicyCompilePrerequisiteReceipt:
        """Validate the minimum declarations without weakening plan purity."""

        if self.compile_prerequisites is None:
            raise RuntimeError(POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER)
        return preflight_policy_compile_prerequisites(self.compile_prerequisites)

    def as_nested_mapping(self) -> dict[str, object]:
        """Return the project-owned overrides as a plain nested mapping."""

        result: dict[str, object] = {}
        for dotted_path, value in self.overrides.items():
            _insert_dotted(result, dotted_path, _plain(value))
        return result

    def to_omegaconf(self) -> object:
        """Create an OmegaConf object without importing veRL or Hydra."""

        from omegaconf import OmegaConf

        return OmegaConf.create(self.as_nested_mapping())

    def hydra_override_args(self) -> tuple[str, ...]:
        """Render stable add-or-replace overrides accepted by Hydra 1.3."""

        return tuple(
            f"++{path}={_hydra_literal(value)}"
            for path, value in self.overrides.items()
        )

    def command(
        self,
        python_executable: str | Path,
        *,
        allow_blocked: bool = False,
    ) -> tuple[str, ...]:
        """Build the exact upstream argv without starting Python, Ray, or CUDA."""

        if not allow_blocked:
            self.assert_launch_ready()
        executable = Path(python_executable)
        if not executable.is_absolute() or not executable.exists():
            raise ValueError("policy launch Python must be an existing absolute path")
        if executable.is_dir():
            raise ValueError("policy launch Python must not be a directory")
        return (
            str(executable),
            "-m",
            self.main_module,
            *self.hydra_override_args(),
        )

    def as_record(
        self,
        *,
        python_executable: str | Path | None = None,
        allow_blocked_command: bool = False,
    ) -> dict[str, object]:
        """Return a JSON-safe audit record for a CLI plan response."""

        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_identity_sha256": self.run_identity_sha256,
            "verl_commit": self.verl_commit,
            "main_module": self.main_module,
            "config_name": self.config_name,
            "runner_fqn": self.runner_fqn,
            "launch_ready": self.launch_ready,
            "launch_blockers": list(self.launch_blockers),
            "inherited_upstream_fields": list(self.inherited_upstream_fields),
            "environment": dict(self.environment),
            "compile_prerequisites": (
                None
                if self.compile_prerequisites is None
                else self.compile_prerequisites.as_record()
            ),
            "compile_prerequisites_sha256": (
                None
                if self.compile_prerequisites is None
                else self.compile_prerequisites.identity_sha256
            ),
            "external_components": dict(self.external_components),
            "overrides": _plain(self.overrides),
        }
        if python_executable is not None:
            record["command"] = list(
                self.command(
                    python_executable,
                    allow_blocked=allow_blocked_command,
                )
            )
        return record


def build_policy_e2e_smoke_verl_plan(
    config: PolicyE2ESmokeRunConfig,
    *,
    horizon_extension: PolicyHorizonExtension | None = None,
    compile_prerequisites: PolicyCompilePrerequisiteBinding | None = None,
) -> UpstreamVerlLaunchPlan:
    """Map one strict smoke identity onto pinned veRL's public config paths."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("config must be PolicyE2ESmokeRunConfig")
    prerequisite_binding = compile_prerequisites
    if prerequisite_binding is not None and not isinstance(
        prerequisite_binding, PolicyCompilePrerequisiteBinding
    ):
        raise TypeError("compile_prerequisites must be a binding or None")
    if horizon_extension is not None:
        if not isinstance(horizon_extension, PolicyHorizonExtension):
            raise TypeError("horizon_extension must be PolicyHorizonExtension")
        horizon_extension.validate_for_config(config)
    if not config.policy.sampling.is_run_bound:
        raise ValueError("Policy smoke sampling identity must be fully run-bound")
    if (
        "</tool_call>" in (config.policy.sampling.stop_strings or ())
        and config.policy.sampling.include_stop_str_in_output is not True
    ):
        raise ValueError("native tool-call closer must remain policy-sampled")

    distributed = config.distributed
    capacity = config.capacity
    performance = config.performance
    if performance is None:
        raise ValueError("Policy launch requires an explicit performance binding")
    performance_capacity = (
        performance.vllm_enable_chunked_prefill,
        performance.vllm_enable_cuda_graph,
        performance.vllm_tensor_parallel_size,
    )
    bound_capacity = (
        capacity.vllm_enable_chunked_prefill,
        not capacity.vllm_enforce_eager,
        distributed.vllm_tensor_parallel_size,
    )
    if performance_capacity != bound_capacity:
        raise ValueError(
            "performance binding differs from distributed/capacity launch values"
        )
    if (
        performance.rollout_logprob_bypass
        is not config.policy.grpo.rollout_correction_bypass_mode
    ):
        raise ValueError(
            "performance rollout-logprob bypass differs from the policy objective"
        )
    cuda_graph_capture_sizes: list[int] | None = (
        list(performance.vllm_cuda_graph_capture_sizes)
        if config.schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA
        else None
    )
    vllm_rollout_replicas = _vllm_rollout_replica_count(
        world_size=distributed.world_size,
        tensor_parallel_size=performance.vllm_tensor_parallel_size,
        data_parallel_size=VLLM_ROLLOUT_DATA_PARALLEL_SIZE,
        pipeline_parallel_size=VLLM_ROLLOUT_PIPELINE_PARALLEL_SIZE,
        disaggregation_enabled=VLLM_ROLLOUT_DISAGGREGATION_ENABLED,
    )
    response_transport_length = capacity.response_transport_length
    if response_transport_length <= config.policy.sampling.max_response_length:
        raise ValueError(
            "vLLM context capacity leaves no response transport reserve for "
            "environment-owned tool tokens"
        )
    fsdp = FSDP2BridgeConfig(
        world_size=distributed.world_size,
        fsdp_size=distributed.world_size,
    )
    runtime = VerlRuntimeRequirements(fsdp2=fsdp)
    adapter = VerlAdapterConfig(
        runtime=runtime,
        max_tool_calls=config.protocol.maximum_tool_calls,
        policy_pilot=config.policy,
        response_transport_length=response_transport_length,
    )
    values = dict(adapter.public_config_overrides())
    if values.get("trainer.use_v1") is not False:
        raise RuntimeError("accepted e003 adapter unexpectedly selected trainer v1")

    sampling = config.policy.sampling
    accumulation = config.accumulation
    framework = config.framework
    optimizer = config.optimizer
    scheduler = config.scheduler
    training = config.training
    selected_binding = None
    full_binding = None
    t1_binding = None
    t1_mixed_binding = None
    teacher_quarter_binding = None
    if config.dataset.selected_sample is None:
        if isinstance(
            config.dataset.runtime_binding, PolicyTeacherQuarterMixRuntimeBinding
        ):
            teacher_typed_binding = config.dataset.runtime_binding
            teacher_quarter_binding = PolicyTeacherQuarterMixDatasetBinding(
                root=config.dataset.root,
                manifest_file_sha256=teacher_typed_binding.manifest_file_sha256,
                content_sha256=teacher_typed_binding.content_sha256,
                samples_sha256=config.dataset.samples_sha256,
                iteration_identity_sha256=config.dataset.iteration_identity_sha256,
                schedule_seed=teacher_typed_binding.schedule_seed,
                expected_sample_count=teacher_typed_binding.expected_sample_count,
                tool_profile=config.protocol.tool_profile,
                visual_prompt_bundle_sha256=config.protocol.prompt_sha256,
                thinklite_prompt_bundle_sha256=(
                    THINKLITE_PROMPT_IDENTITY.bundle_sha256
                ),
                tokenizer_length=config.model.tokenizer_length,
                model_name=config.model.model_name,
                chat_template_sha256=config.model.chat_template_sha256,
            )
        elif isinstance(config.dataset.runtime_binding, PolicyT1MixedRuntimeBinding):
            mixed_typed_binding = config.dataset.runtime_binding
            t1_mixed_binding = VerlPolicyT1MixedDatasetBinding(
                root=config.dataset.root,
                manifest_file_sha256=mixed_typed_binding.manifest_file_sha256,
                content_sha256=mixed_typed_binding.content_sha256,
                samples_sha256=config.dataset.samples_sha256,
                iteration_identity_sha256=config.dataset.iteration_identity_sha256,
                shuffle_seed=mixed_typed_binding.shuffle_seed,
                expected_sample_count=mixed_typed_binding.expected_sample_count,
                prompt_bundle_sha256=config.protocol.prompt_sha256,
                tool_profile=config.protocol.tool_profile,
                tokenizer_length=config.model.tokenizer_length,
                model_name=config.model.model_name,
            )
        elif isinstance(config.dataset.runtime_binding, PolicyT1RLRuntimeBinding):
            t1_typed_binding = config.dataset.runtime_binding
            t1_binding = VerlPolicyT1DatasetBinding(
                root=config.dataset.root,
                manifest_file_sha256=t1_typed_binding.manifest_file_sha256,
                content_sha256=t1_typed_binding.content_sha256,
                samples_sha256=config.dataset.samples_sha256,
                iteration_identity_sha256=config.dataset.iteration_identity_sha256,
                shuffle_seed=t1_typed_binding.shuffle_seed,
                decision_stage=t1_typed_binding.decision_stage,
                expected_sample_count=t1_typed_binding.expected_sample_count,
                prompt_bundle_sha256=config.protocol.prompt_sha256,
                tool_profile=config.protocol.tool_profile,
                tokenizer_length=config.model.tokenizer_length,
                model_name=config.model.model_name,
            )
        else:
            full_binding = VerlDeepEyes47KDatasetBinding(
                root=config.dataset.root,
                manifest_file_sha256=(
                    config.dataset.runtime_binding.manifest_file_sha256
                ),
                content_sha256=config.dataset.runtime_binding.content_sha256,
                samples_sha256=config.dataset.samples_sha256,
                iteration_identity_sha256=(config.dataset.iteration_identity_sha256),
                shuffle_seed=config.dataset.runtime_binding.shuffle_seed,
                fixture=False,
                expected_sample_count=(
                    config.dataset.runtime_binding.expected_sample_count
                ),
                prompt_bundle_sha256=config.protocol.prompt_sha256,
                tool_profile=config.protocol.tool_profile,
                tokenizer_length=config.model.tokenizer_length,
                model_name=config.model.model_name,
            )
    else:
        selected_binding = VerlSelectedSampleDatasetBinding.from_run_config(config)
    dataset_binding = (
        selected_binding
        or teacher_quarter_binding
        or t1_mixed_binding
        or t1_binding
        or full_binding
    )
    if dataset_binding is None:  # pragma: no cover - binding construction invariant
        raise RuntimeError("Policy dataset binding was not constructed")
    if selected_binding is not None:
        dataset_module_path = SELECTED_SAMPLE_DATASET_MODULE_PATH
        dataset_class_name = SELECTED_SAMPLE_DATASET_CLASS_NAME
        dataset_config = {"data.tgvf_selected_sample": selected_binding.as_config()}
        dataset_samples_path = selected_binding.samples_path
    elif teacher_quarter_binding is not None:
        dataset_module_path = POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH
        dataset_class_name = POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS_NAME
        dataset_config = {
            f"data.{POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME}": (
                teacher_quarter_binding.as_config()
            )
        }
        dataset_samples_path = teacher_quarter_binding.samples_path
    elif t1_mixed_binding is not None:
        dataset_module_path = POLICY_T1_MIXED_DATASET_MODULE_PATH
        dataset_class_name = POLICY_T1_MIXED_DATASET_CLASS_NAME
        dataset_config = {"data.tgvf_policy_t1_mixed": t1_mixed_binding.as_config()}
        dataset_samples_path = t1_mixed_binding.root / "samples.jsonl"
    elif t1_binding is not None:
        dataset_module_path = POLICY_T1_ARXIVQA_DATASET_MODULE_PATH
        dataset_class_name = POLICY_T1_ARXIVQA_DATASET_CLASS_NAME
        dataset_config = {"data.tgvf_policy_t1_arxivqa": t1_binding.as_config()}
        dataset_samples_path = t1_binding.root / "samples.jsonl"
    else:
        dataset_module_path = DEEPEYES47K_DATASET_MODULE_PATH
        dataset_class_name = DEEPEYES47K_DATASET_CLASS_NAME
        dataset_config = {"data.tgvf_deepeyes47k": full_binding.as_config()}
        dataset_samples_path = full_binding.root / "samples.jsonl"
    actor_batch = _actor_batch_contract(config)
    effective_checkpoint_steps = (
        horizon_extension.effective_checkpoint_steps
        if horizon_extension is not None
        else training.checkpoint_steps
    )
    effective_maximum_step = (
        horizon_extension.target_optimizer_step
        if horizon_extension is not None
        else training.maximum_optimizer_steps
    )
    save_frequency = _checkpoint_frequency(
        effective_checkpoint_steps,
        maximum_step=effective_maximum_step,
    )
    precision = {
        "param_dtype": _precision_name(config.precision.parameter_dtype),
        "reduce_dtype": _precision_name(config.precision.reduce_dtype),
        "buffer_dtype": _precision_name(config.precision.optimizer_state_dtype),
    }
    conditioning = config.representation.conditioning
    conditioning_record = {
        "schema_version": conditioning.schema_version,
        "provider": conditioning.provider.value,
        "hidden_layer": conditioning.hidden_layer,
        "embedding_identity": conditioning.embedding_identity,
    }
    if config.policy.grpo.verl_external_loss_module != EXACT_REPLAY_EXTERNAL_MODULE:
        raise ValueError(
            "Policy loss and exact replay external module identities differ"
        )

    values.update(
        {
            # The custom Dataset consumes the selected materialized DeepEyes
            # row directly; it performs no upstream parquet conversion.
            "data.train_files": [str(dataset_samples_path)],
            "data.val_files": [str(dataset_samples_path)],
            "data.train_max_samples": -1,
            "data.val_max_samples": -1,
            # veRL's file loader executes modules before registering them in
            # ``sys.modules``; Python 3.12 dataclasses correctly reject that
            # broken import state.  Its public ``pkg://`` route performs a
            # normal package import and preserves the module identity.
            "data.custom_cls.path": dataset_module_path,
            "data.custom_cls.name": dataset_class_name,
            **dataset_config,
            "data.train_batch_size": accumulation.global_prompt_batch_size,
            "data.gen_batch_size": accumulation.global_prompt_batch_size,
            "data.shuffle": False,
            "data.seed": (
                config.dataset.runtime_binding.schedule_seed
                if isinstance(
                    config.dataset.runtime_binding,
                    PolicyTeacherQuarterMixRuntimeBinding,
                )
                else config.dataset.runtime_binding.shuffle_seed
            ),
            "data.validation_shuffle": False,
            "data.return_raw_chat": True,
            "data.return_multi_modal_inputs": True,
            "data.max_response_length": response_transport_length,
            "data.max_prompt_length": capacity.max_prompt_length,
            "data.mm_processor_kwargs.max_pixels": config.policy.image_max_pixels,
            # Model/engine identity.  Both actor and reference use the same
            # registered exact-observation model type; the engine selects the
            # current/reference role from forward_only.
            "actor_rollout_ref.model.external_lib": EXACT_REPLAY_EXTERNAL_MODULE,
            "actor_rollout_ref.model.model_type": TGVF_EXACT_REPLAY_MODEL_TYPE,
            # Keep the actor/reference attention identity explicit while the
            # performance binding owns activation and padding optimizations.
            "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
            "actor_rollout_ref.model.enable_gradient_checkpointing": (
                performance.enable_gradient_checkpointing
            ),
            "actor_rollout_ref.model.use_remove_padding": (
                performance.use_remove_padding
            ),
            "actor_rollout_ref.model.use_liger": False,
            "actor_rollout_ref.model.use_fused_kernels": False,
            "actor_rollout_ref.actor.ppo_mini_batch_size": (
                actor_batch["upstream_ppo_mini_batch_size_prompts"]
            ),
            "actor_rollout_ref.actor.ppo_micro_batch_size": None,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": (
                None
                if performance.dynamic_token_batching
                else actor_batch["upstream_ppo_micro_batch_size_per_gpu_trajectories"]
            ),
            "actor_rollout_ref.actor.use_dynamic_bsz": (
                performance.dynamic_token_batching
            ),
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": (
                capacity.actor_ppo_max_token_len_per_gpu
            ),
            "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": (
                distributed.fsdp_reshard_after_forward
            ),
            "actor_rollout_ref.ref.fsdp_config.reshard_after_forward": (
                distributed.fsdp_reshard_after_forward
            ),
            # These bits select current-versus-frozen-reference behavior in
            # the registered exact-replay engine.  They are semantic run
            # inputs, so never inherit them from veRL's YAML defaults.
            "actor_rollout_ref.actor.fsdp_config.forward_only": False,
            "actor_rollout_ref.ref.fsdp_config.forward_only": True,
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
            "actor_rollout_ref.ref.fsdp_config.model_dtype": "bf16",
            "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
            "actor_rollout_ref.ref.fsdp_config.use_torch_compile": False,
            "actor_rollout_ref.actor.fsdp_config.param_offload": False,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
            "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
            "actor_rollout_ref.ref.fsdp_config.param_offload": False,
            "actor_rollout_ref.ref.fsdp_config.optimizer_offload": False,
            "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
            "actor_rollout_ref.actor.fsdp_config.mixed_precision": precision,
            "actor_rollout_ref.ref.fsdp_config.mixed_precision": precision,
            "actor_rollout_ref.actor.optim.optimizer": "AdamW",
            "actor_rollout_ref.actor.optim.optimizer_impl": "torch.optim",
            "actor_rollout_ref.actor.optim.lr": optimizer.learning_rate,
            "actor_rollout_ref.actor.optim.betas": [
                optimizer.beta1,
                optimizer.beta2,
            ],
            "actor_rollout_ref.actor.optim.weight_decay": optimizer.weight_decay,
            "actor_rollout_ref.actor.optim.override_optimizer_config": {
                "eps": optimizer.epsilon
            },
            "actor_rollout_ref.actor.optim.clip_grad": (
                optimizer.maximum_gradient_norm
            ),
            "actor_rollout_ref.actor.optim.lr_warmup_steps": (scheduler.warmup_steps),
            "actor_rollout_ref.actor.optim.total_training_steps": (
                scheduler.total_steps
            ),
            "actor_rollout_ref.actor.optim.lr_scheduler_type": scheduler.name,
            "actor_rollout_ref.actor.optim.min_lr_ratio": (
                scheduler.minimum_learning_rate_ratio
            ),
            "actor_rollout_ref.rollout.log_prob_micro_batch_size": None,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": (
                None
                if performance.dynamic_token_batching
                else accumulation.rollout_prompt_micro_batch_size_per_engine
                * sampling.trajectories_per_prompt
            ),
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": (
                performance.dynamic_token_batching
            ),
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": (
                capacity.rollout_log_prob_max_token_len_per_gpu
            ),
            "actor_rollout_ref.ref.log_prob_micro_batch_size": None,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (
                None
                if performance.dynamic_token_batching
                else actor_batch[
                    "upstream_inference_micro_batch_size_per_gpu_trajectories"
                ]
            ),
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": (
                performance.dynamic_token_batching
            ),
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": (
                capacity.reference_log_prob_max_token_len_per_gpu
            ),
            "actor_rollout_ref.hybrid_engine": (distributed.placement == "colocated"),
            "actor_rollout_ref.rollout.nnodes": 1,
            "actor_rollout_ref.rollout.n_gpus_per_node": distributed.world_size,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": (
                performance.vllm_tensor_parallel_size
            ),
            "actor_rollout_ref.rollout.data_parallel_size": (
                VLLM_ROLLOUT_DATA_PARALLEL_SIZE
            ),
            "actor_rollout_ref.rollout.pipeline_model_parallel_size": (
                VLLM_ROLLOUT_PIPELINE_PARALLEL_SIZE
            ),
            "actor_rollout_ref.rollout.disaggregation.enabled": (
                VLLM_ROLLOUT_DISAGGREGATION_ENABLED
            ),
            "actor_rollout_ref.rollout.gpu_memory_utilization": (
                capacity.vllm_gpu_memory_utilization
            ),
            "actor_rollout_ref.rollout.max_num_batched_tokens": (
                capacity.vllm_max_num_batched_tokens
            ),
            "actor_rollout_ref.rollout.max_model_len": (capacity.vllm_max_model_len),
            "actor_rollout_ref.rollout.max_num_seqs": capacity.vllm_max_num_seqs,
            "actor_rollout_ref.rollout.enable_chunked_prefill": (
                performance.vllm_enable_chunked_prefill
            ),
            "actor_rollout_ref.rollout.enable_prefix_caching": (
                performance.vllm_enable_prefix_caching
            ),
            "actor_rollout_ref.rollout.enforce_eager": (
                not performance.vllm_enable_cuda_graph
            ),
            "actor_rollout_ref.rollout.cudagraph_capture_sizes": (
                cuda_graph_capture_sizes
            ),
            "actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode": (
                performance.rollout_logprob_bypass
            ),
            "algorithm.rollout_correction.bypass_mode": (
                performance.rollout_logprob_bypass
            ),
            "actor_rollout_ref.rollout.seed": config.rollout_rng.master_seed,
            "actor_rollout_ref.rollout.ignore_eos": sampling.ignore_eos,
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                NATIVE_AGENT_LOOP_NAME
            ),
            "actor_rollout_ref.rollout.agent.num_workers": distributed.world_size,
            "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(
                framework.agent_loop_config_path
            ),
            # Pinned veRL names this field ``checkpoint_manager_class`` even
            # though its value is a fully-qualified class name.
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                POLICY_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            # Pinned upstream has no top-level fields for the full stopping and
            # replay identity.  The custom AgentLoop/factory consumes these
            # project-owned values instead of relying on vLLM defaults.
            "actor_rollout_ref.rollout.custom": {
                "schema_version": VERL_POLICY_SMOKE_LAUNCH_SCHEMA,
                "run_id": config.run_id,
                "run_identity_sha256": config.identity_sha256,
                "sampling": {
                    "min_p": sampling.min_p,
                    "presence_penalty": sampling.presence_penalty,
                    "frequency_penalty": sampling.frequency_penalty,
                    "stop_token_ids": list(sampling.stop_token_ids or ()),
                    "stop_strings": list(sampling.stop_strings or ()),
                    "include_stop_str_in_output": (sampling.include_stop_str_in_output),
                    "ignore_eos": sampling.ignore_eos,
                    "logit_processors": list(sampling.logit_processors),
                    "logprob_measurement": sampling.logprob_measurement.value,
                    "rollout_master_seed": config.rollout_rng.master_seed,
                    "seed_derivation_name": config.rollout_rng.derivation_name,
                    "seed_derivation_sha256": (config.rollout_rng.derivation_sha256),
                    "forward_state": "request_seeded_batch_sensitive_v1",
                    "vllm_batch_invariant": False,
                    "actual_behavior_logprobs_recorded": True,
                },
                "protocol": {
                    "prompt_sha256": config.protocol.prompt_sha256,
                    "tool_schema_sha256": config.protocol.tool_schema_sha256,
                    "cap_error_sha256": config.protocol.cap_error_sha256,
                    "maximum_tool_calls": config.protocol.maximum_tool_calls,
                },
                "representation": {
                    "artifact_path": str(config.representation.artifact_path),
                    "artifact_file_sha256": (
                        config.representation.artifact_file_sha256
                    ),
                    "artifact_manifest_sha256": (config.representation.artifact.sha256),
                    "expected_run_id": config.representation.expected_run_id,
                    "expected_run_identity_sha256": (
                        config.representation.expected_run_identity_sha256
                    ),
                    "conditioning": conditioning_record,
                },
                "agent_loop": {
                    "target_fqn": NATIVE_AGENT_LOOP_FQN,
                    "invocation_factory_fqn": NATIVE_INVOCATION_FACTORY_FQN,
                    "runtime_invocation_factory_fqn": (
                        framework.runtime_invocation_factory_fqn
                    ),
                    "config_path": str(framework.agent_loop_config_path),
                    "config_sha256": framework.agent_loop_config_sha256,
                    "server_timeout_seconds": framework.server_timeout_seconds,
                },
                "capacity": {
                    "max_prompt_length": capacity.max_prompt_length,
                    "actor_ppo_max_token_len_per_gpu": (
                        capacity.actor_ppo_max_token_len_per_gpu
                    ),
                    "rollout_log_prob_max_token_len_per_gpu": (
                        capacity.rollout_log_prob_max_token_len_per_gpu
                    ),
                    "reference_log_prob_max_token_len_per_gpu": (
                        capacity.reference_log_prob_max_token_len_per_gpu
                    ),
                    "vllm_gpu_memory_utilization": (
                        capacity.vllm_gpu_memory_utilization
                    ),
                    "vllm_max_num_batched_tokens": (
                        capacity.vllm_max_num_batched_tokens
                    ),
                    "vllm_max_model_len": capacity.vllm_max_model_len,
                    "vllm_max_num_seqs": capacity.vllm_max_num_seqs,
                },
                "performance": {
                    "source_schema_version": config.schema_version,
                    "dynamic_token_batching": (performance.dynamic_token_batching),
                    "use_remove_padding": performance.use_remove_padding,
                    "enable_gradient_checkpointing": (
                        performance.enable_gradient_checkpointing
                    ),
                    "vllm_enable_prefix_caching": (
                        performance.vllm_enable_prefix_caching
                    ),
                    "vllm_enable_chunked_prefill": (
                        performance.vllm_enable_chunked_prefill
                    ),
                    "vllm_enable_cuda_graph": (performance.vllm_enable_cuda_graph),
                    "vllm_cuda_graph_capture_sizes": cuda_graph_capture_sizes,
                    "vllm_tensor_parallel_size": (
                        performance.vllm_tensor_parallel_size
                    ),
                    "vllm_data_parallel_size": VLLM_ROLLOUT_DATA_PARALLEL_SIZE,
                    "vllm_pipeline_parallel_size": (
                        VLLM_ROLLOUT_PIPELINE_PARALLEL_SIZE
                    ),
                    "vllm_disaggregation_enabled": (
                        VLLM_ROLLOUT_DISAGGREGATION_ENABLED
                    ),
                    "derived_vllm_rollout_replicas": vllm_rollout_replicas,
                    "rollout_logprob_bypass": (performance.rollout_logprob_bypass),
                    "reference_replay_mode": performance.reference_replay_mode,
                    "judge_dispatch_mode": performance.judge_dispatch_mode,
                    "judge_max_concurrency_per_worker": (
                        performance.judge_max_concurrency_per_worker
                    ),
                    "judge_concurrency_scope": "agent_loop_worker_process_local",
                    "prefix_cache_identity_basis": (
                        VLLM_012_PREFIX_CACHE_MM_HASH_IDENTITY
                    ),
                    "prefix_cache_invalidation": (
                        VERL_PREFIX_CACHE_WEIGHT_UPDATE_INVALIDATION
                    ),
                },
                "exact_replay": {
                    "model_type": TGVF_EXACT_REPLAY_MODEL_TYPE,
                    "registration_module": EXACT_REPLAY_EXTERNAL_MODULE,
                    "engine_registrar_fqn": EXACT_REPLAY_ENGINE_REGISTRAR_FQN,
                    "forward_port_fqn": EXACT_REPLAY_FORWARD_PORT_FQN,
                    "current_reference_replay_fqn": (
                        EXACT_CURRENT_REFERENCE_REPLAY_FQN
                    ),
                },
                "reward": _reward_custom_config(config),
                "reference_diagnostic": {
                    "enabled": (performance.reference_replay_mode == "full_diagnostic"),
                    "coefficient": 0.0,
                    "worker_route": "colocated_frozen_base_exact_replay",
                    "observation_source": "rollout_materialized_exact_bundle",
                },
                "weight_sync": {
                    "mode": distributed.weight_sync_mode,
                    "interval_optimizer_steps": (
                        distributed.weight_sync_interval_optimizer_steps
                    ),
                },
                "checkpoint_steps": list(effective_checkpoint_steps),
                "runtime_state_directory": str(
                    config.output.root / "runtime-policy-state"
                ),
                "actor_batch_contract": actor_batch,
                "metrics_path": str(config.output.metrics_path),
            },
            "trainer.total_epochs": training.total_training_epochs,
            "trainer.total_training_steps": effective_maximum_step,
            "trainer.nnodes": 1,
            "trainer.n_gpus_per_node": distributed.world_size,
            "trainer.project_name": training.project_name,
            "trainer.experiment_name": config.run_id,
            "trainer.logger": list(training.logger),
            "trainer.val_before_train": training.validation_before_training,
            "trainer.val_only": False,
            "trainer.test_freq": training.validation_frequency,
            "trainer.resume_mode": training.resume_mode,
            "trainer.resume_from_path": (
                str(training.resume_from_path)
                if training.resume_from_path is not None
                else None
            ),
            "trainer.max_actor_ckpt_to_keep": (
                training.maximum_actor_checkpoints_to_keep
            ),
            "trainer.save_freq": save_frequency,
            "trainer.default_local_dir": str(config.output.checkpoint_directory),
        }
    )

    external_components = {
        "dataset": dataset_module_path.removeprefix("pkg://")
        + "."
        + dataset_class_name,
        "agent_loop_manager": LOSSLESS_AGENT_LOOP_MANAGER_FQN,
        "agent_loop": NATIVE_AGENT_LOOP_FQN,
        "invocation_factory": NATIVE_INVOCATION_FACTORY_FQN,
        "runtime_invocation_factory": framework.runtime_invocation_factory_fqn,
        "checkpoint_engine_manager": POLICY_CHECKPOINT_ENGINE_MANAGER_FQN,
        "task_runner_lifecycle": (
            "tgvf_rl.framework.verl.policy_task_runner."
            "create_policy_pilot_task_runner_class"
        ),
        "exact_replay_registration": EXACT_REPLAY_ENGINE_REGISTRAR_FQN,
        "exact_replay_forward": EXACT_REPLAY_FORWARD_PORT_FQN,
        "exact_current_reference_replay": EXACT_CURRENT_REFERENCE_REPLAY_FQN,
        "reward_pipeline": (
            POLICY_REWARD_PIPELINE_FQN
            if config.reward.profile == "pilot-v1"
            else STAGE3_REWARD_PIPELINE_FQN
        ),
        "reference_diagnostic": (
            "tgvf_rl.framework.verl.policy_task_runner."
            "make_policy_pilot_ray_trainer_class"
        ),
        "vllm_lora_pdl_compatibility": VLLM_012_LORA_PDL_MODE,
    }
    environment = dict(adapter.required_environment())
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(device) for device in distributed.physical_gpu_ids
    )
    environment["TGVF_POLICY_RUN_CONFIG_PATH"] = str(config.source_path)
    environment["TGVF_POLICY_RUN_ID"] = config.run_id
    environment["TGVF_POLICY_RUN_IDENTITY"] = config.identity_sha256
    environment["TGVF_POLICY_RUN_IDENTITY_SHA256"] = config.identity_sha256
    environment["TGVF_POLICY_STATE_DIR"] = str(
        config.output.root / "runtime-policy-state"
    )
    environment["TGVF_POLICY_SERVER_TIMEOUT_SECONDS"] = format(
        framework.server_timeout_seconds, ".17g"
    )
    if horizon_extension is not None:
        environment.update(horizon_extension.environment)
    environment["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    environment["PYTHONHASHSEED"] = str(config.rollout_rng.master_seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    # Triton compiles its CUDA launcher lazily.  Never infer compiler/header
    # paths from this worktree or a sibling checkout: only an explicitly loaded,
    # content-bound manifest may contribute these environment variables.
    if prerequisite_binding is not None:
        environment["CC"] = str(prerequisite_binding.c_compiler)
        environment["CXX"] = str(prerequisite_binding.cxx_compiler)
        environment["CPATH"] = prerequisite_binding.cpath
        environment["TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256"] = (
            prerequisite_binding.identity_sha256
        )
        environment["TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_PATH"] = str(
            prerequisite_binding.manifest_source_path
        )
        environment["TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256"] = (
            prerequisite_binding.manifest_source_sha256
        )

    blockers = (
        (POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,)
        if prerequisite_binding is None
        else prerequisite_binding.launch_blockers
    )
    inherited: tuple[str, ...] = ()
    return UpstreamVerlLaunchPlan(
        run_identity_sha256=config.identity_sha256,
        overrides=values,
        environment=environment,
        compile_prerequisites=prerequisite_binding,
        external_components=external_components,
        launch_blockers=blockers,
        inherited_upstream_fields=inherited,
    )


def compose_upstream_verl_config(
    plan: UpstreamVerlLaunchPlan,
    *,
    config_directory: str | Path,
) -> object:
    """Compose pinned upstream YAML for a CPU-only config-parse check.

    This imports Hydra, not ``verl.trainer.main_ppo``.  It never launches the
    plan and intentionally permits a blocked plan so mapping tests can inspect
    unresolved fields.
    """

    if not isinstance(plan, UpstreamVerlLaunchPlan):
        raise TypeError("plan must be UpstreamVerlLaunchPlan")
    directory = Path(config_directory).resolve()
    if not directory.is_dir():
        raise ValueError("upstream veRL config directory must exist")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        return compose(
            config_name=plan.config_name,
            overrides=list(plan.hydra_override_args()),
        )


def _reward_custom_config(config: PolicyE2ESmokeRunConfig) -> dict[str, object]:
    reward = config.reward
    common: dict[str, object] = {
        "task_kind": reward.task_kind,
        "answer_verifier": reward.answer_verifier,
        "answer_verifier_sha256": reward.answer_verifier_sha256,
        "judge_mode": reward.judge_mode,
        "judge_config_path": (
            str(reward.judge_config_path)
            if reward.judge_config_path is not None
            else None
        ),
        "judge_config_sha256": reward.judge_config_sha256,
    }
    if reward.profile == "pilot-v1":
        if (
            reward.answer_weight is None
            or reward.format_weight is None
            or reward.conditional_tool_weight is None
        ):
            raise ValueError("Pilot-v1 reward weights are incomplete")
        return {
            "pipeline_fqn": POLICY_REWARD_PIPELINE_FQN,
            **common,
            "answer_weight": reward.answer_weight,
            "format_weight": reward.format_weight,
            "conditional_tool_weight": reward.conditional_tool_weight,
        }
    if reward.profile != "stage3-shaped-v1":
        raise ValueError("unsupported reward profile")
    utility_enabled = reward.tool_utility_reward_enabled
    if type(utility_enabled) is not bool:
        raise ValueError("Stage3 tool-utility reward switch is missing")
    if utility_enabled != (reward.tool_utility is not None):
        raise ValueError("Stage3 tool-utility binding differs from its switch")
    focus_enabled = reward.focus_reward_enabled
    grounding_enabled = reward.grounding_reward_enabled
    if type(focus_enabled) is not bool or type(grounding_enabled) is not bool:
        raise ValueError("Stage3 visual reward switches are missing")
    if focus_enabled != grounding_enabled:
        raise ValueError("Stage3 Focus/Grounding reward switches must agree")
    quality_enabled = focus_enabled
    visual_bound = (
        reward.visual_quality_judge_config_path is not None
        and reward.visual_quality_judge_config_sha256 is not None
    )
    if quality_enabled != visual_bound:
        raise ValueError("Stage3 visual-quality binding differs from its switch")
    visual_mode = reward.visual_quality_judge_mode
    if quality_enabled and visual_mode == "disabled":
        raise ValueError("enabled visual quality cannot use disabled judge mode")
    if not quality_enabled and visual_mode != "disabled":
        raise ValueError("disabled visual quality requires judge mode=disabled")
    coefficients = {
        "answer_reward_scale": reward.answer_reward_scale,
        "repeated_call_penalty": reward.repeated_call_penalty,
        "protocol_error_penalty": reward.protocol_error_penalty,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in coefficients.values()
    ):
        raise ValueError("Stage3 reward coefficients must be non-negative numbers")
    if utility_enabled and quality_enabled:
        assert reward.tool_utility is not None
        assert reward.visual_quality_judge_config_path is not None
        assert reward.visual_quality_judge_config_sha256 is not None
        return {
            "pipeline_fqn": STAGE3_REWARD_PIPELINE_FQN,
            "profile": reward.profile,
            **common,
            "tool_utility_sidecar_path": str(reward.tool_utility.sidecar_path),
            "tool_utility_sidecar_sha256": reward.tool_utility.sidecar_sha256,
            "tool_utility_manifest_path": str(reward.tool_utility.manifest_path),
            "tool_utility_manifest_sha256": reward.tool_utility.manifest_sha256,
            "visual_quality_judge_config_path": str(
                reward.visual_quality_judge_config_path
            ),
            "visual_quality_judge_config_sha256": (
                reward.visual_quality_judge_config_sha256
            ),
        }
    shaped = {
        "pipeline_fqn": STAGE3_REWARD_PIPELINE_FQN,
        "profile": reward.profile,
        **common,
        "tool_utility_reward_enabled": utility_enabled,
        "focus_reward_enabled": quality_enabled,
        "grounding_reward_enabled": quality_enabled,
        "visual_quality_judge_mode": visual_mode,
        **coefficients,
    }
    if reward.tool_utility is not None:
        shaped.update(
            {
                "tool_utility_sidecar_path": str(reward.tool_utility.sidecar_path),
                "tool_utility_sidecar_sha256": reward.tool_utility.sidecar_sha256,
                "tool_utility_manifest_path": str(reward.tool_utility.manifest_path),
                "tool_utility_manifest_sha256": (reward.tool_utility.manifest_sha256),
            }
        )
    if quality_enabled:
        shaped.update(
            {
                "visual_quality_judge_config_path": str(
                    reward.visual_quality_judge_config_path
                ),
                "visual_quality_judge_config_sha256": (
                    reward.visual_quality_judge_config_sha256
                ),
            }
        )
    return shaped


def _checkpoint_frequency(steps: Sequence[int], *, maximum_step: int) -> int:
    normalized = tuple(steps)
    if not normalized or normalized[0] != 0 or normalized[-1] != maximum_step:
        raise ValueError("checkpoint steps must include zero and the final step")
    positive = normalized[1:]
    if not positive:
        raise ValueError("checkpoint plan requires at least one positive step")
    # veRL exposes only a periodic ``save_freq``. Use the greatest common
    # divisor of the requested positive steps so every requested checkpoint is
    # visited while avoiding an otherwise pointless hook on every optimizer
    # step. The project trainer hook still applies the exact (possibly
    # non-uniform) schedule before any checkpoint mutation occurs.
    return math.gcd(*positive)


def _actor_batch_contract(
    config: PolicyE2ESmokeRunConfig,
) -> dict[str, object]:
    """Prove prompt-unit config maps to one pinned-veRL optimizer update.

    Pinned e003 v0 treats ``actor.ppo_mini_batch_size`` as prompts and multiplies
    it by ``rollout.n`` in ``RayPPOTrainer._update_actor``. FSDP's
    ``ppo_micro_batch_size_per_gpu`` counts already-expanded trajectories.
    Actor autograd must not retain all ``n`` trajectory graphs at once, so the
    configured per-rank prompt micro-batch is the conservative trajectory
    micro-batch too. The resulting ``n`` internal forward/backward calls still
    form one upstream mini-batch and one optimizer step; they do not change the
    prompt-level gradient-accumulation identity. Reference replay is no-grad
    and retains the expanded inference micro-batch.
    """

    prompts = config.accumulation.global_prompt_batch_size
    prompt_micro = config.accumulation.prompt_micro_batch_size_per_rank
    n = config.policy.sampling.trajectories_per_prompt
    dp_size = config.distributed.world_size
    trajectory_mini = prompts * n
    performance = config.performance
    if performance is None:
        raise ValueError("actor batch contract requires performance binding")
    if performance.dynamic_token_batching:
        return {
            "global_prompt_batch_size": prompts,
            "rollouts_per_prompt": n,
            "fsdp_data_parallel_size": dp_size,
            "prompt_micro_batch_size_per_rank": prompt_micro,
            "configured_gradient_accumulation_steps": (
                config.accumulation.gradient_accumulation_steps
            ),
            "dynamic_token_batching": True,
            "actor_ppo_max_token_len_per_gpu": (
                config.capacity.actor_ppo_max_token_len_per_gpu
            ),
            "rollout_log_prob_max_token_len_per_gpu": (
                config.capacity.rollout_log_prob_max_token_len_per_gpu
            ),
            "reference_log_prob_max_token_len_per_gpu": (
                config.capacity.reference_log_prob_max_token_len_per_gpu
            ),
            "upstream_ppo_mini_batch_size_prompts": prompts,
            "upstream_internal_mini_batch_size_trajectories": trajectory_mini,
            "upstream_ppo_micro_batch_size_per_gpu_trajectories": None,
            "upstream_inference_micro_batch_size_per_gpu_trajectories": None,
            "derived_actor_forward_backward_microbatches": None,
            "derived_gradient_accumulation_steps": None,
            "optimizer_steps_per_trainer_step": 1,
        }
    actor_trajectory_micro_per_gpu = prompt_micro
    inference_trajectory_micro_per_gpu = prompt_micro * n
    denominator = dp_size * actor_trajectory_micro_per_gpu
    if trajectory_mini % denominator:
        raise ValueError(
            "expanded trajectory mini-batch is not divisible by FSDP2 micro-batches"
        )
    actor_forward_backward_microbatches = trajectory_mini // denominator
    configured_accumulation = config.accumulation.gradient_accumulation_steps
    expected_actor_microbatches = configured_accumulation * n
    if actor_forward_backward_microbatches != expected_actor_microbatches:
        raise ValueError(
            "pinned veRL actor microbatches differ from prompt accumulation times n"
        )
    return {
        "global_prompt_batch_size": prompts,
        "rollouts_per_prompt": n,
        "fsdp_data_parallel_size": dp_size,
        "prompt_micro_batch_size_per_rank": prompt_micro,
        "configured_gradient_accumulation_steps": configured_accumulation,
        "dynamic_token_batching": False,
        "upstream_ppo_mini_batch_size_prompts": prompts,
        "upstream_internal_mini_batch_size_trajectories": trajectory_mini,
        "upstream_ppo_micro_batch_size_per_gpu_trajectories": (
            actor_trajectory_micro_per_gpu
        ),
        "upstream_inference_micro_batch_size_per_gpu_trajectories": (
            inference_trajectory_micro_per_gpu
        ),
        "derived_actor_forward_backward_microbatches": (
            actor_forward_backward_microbatches
        ),
        "derived_gradient_accumulation_steps": configured_accumulation,
        "optimizer_steps_per_trainer_step": 1,
    }


def _vllm_rollout_replica_count(
    *,
    world_size: int,
    tensor_parallel_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    disaggregation_enabled: bool,
) -> int:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("vLLM rollout replica derivation requires positive world size")
    parallel_sizes = {
        "tensor": tensor_parallel_size,
        "data": data_parallel_size,
        "pipeline": pipeline_parallel_size,
    }
    for name, value in parallel_sizes.items():
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"vLLM rollout replica derivation requires positive {name} "
                "parallel size"
            )
    if type(disaggregation_enabled) is not bool:
        raise TypeError("vLLM rollout disaggregation flag must be bool")
    if disaggregation_enabled:
        raise ValueError(
            "vLLM rollout replica derivation does not accept disaggregation "
            "without explicit prefill/decode topology"
        )
    rollout_world_size = math.prod(parallel_sizes.values())
    if world_size % rollout_world_size:
        raise ValueError("vLLM rollout parallel footprint must divide world size")
    return world_size // rollout_world_size


def _precision_name(value: str) -> str:
    names = {"bfloat16": "bf16", "float32": "fp32"}
    try:
        return names[value]
    except KeyError as error:
        raise ValueError(f"unsupported veRL precision identity {value!r}") from error


def _insert_dotted(root: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"invalid dotted config path {path!r}")
    current = root
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            child: dict[str, object] = {}
            current[part] = child
            current = child
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(f"dotted config path collides at {part!r}")
    leaf = parts[-1]
    if leaf in current:
        raise ValueError(f"duplicate dotted config path {path!r}")
    current[leaf] = value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _hydra_literal(value: object) -> str:
    plain = _plain(value)
    if isinstance(plain, Mapping):
        fields: list[str] = []
        for key in sorted(plain):
            if not key or any(
                not (character.isalnum() or character in "_-") for character in key
            ):
                raise ValueError(f"Hydra mapping key is not override-safe: {key!r}")
            fields.append(f"{key}:{_hydra_literal(plain[key])}")
        return "{" + ",".join(fields) + "}"
    if isinstance(plain, list):
        return "[" + ",".join(_hydra_literal(item) for item in plain) + "]"
    return json.dumps(
        plain,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _assert_checkpoint_and_method_matrix_surface(
    overrides: Mapping[str, object],
    external_components: Mapping[str, str],
) -> None:
    """Keep the legacy manager invariant or validate one narrow method overlay.

    The method matrix deliberately reuses :class:`UpstreamVerlLaunchPlan`
    instead of adding another launcher hierarchy.  Its profile marker is the
    only condition under which the historical Policy-LoRA checkpoint-manager
    invariant may differ: native full-Qwen NoTool/Crop wrap upstream veRL sync
    only to publish its typed behavior receipt, while TGVF/Atomic additionally
    publish frozen Adapter state after upstream Qwen sync.  Unknown markers
    never relax the legacy path.
    """

    profile = external_components.get("method_matrix_profile")
    manager_path = "actor_rollout_ref.rollout.checkpoint_manager_class"
    engine_kwargs_path = "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs"
    if profile is None:
        if overrides.get(manager_path) != POLICY_CHECKPOINT_ENGINE_MANAGER_FQN:
            raise ValueError("launch plan lost the Policy checkpoint engine manager")
        return

    native_profiles = frozenset({"no_tool", "crop"})
    tgvf_profiles = frozenset({"tgvf_short", "tgvf_target_guide_v2", "atomic"})
    if profile in native_profiles:
        expected_external = "tgvf_rl.framework.verl.trainable_crop_external"
        expected_model_type = "tgvf_trainable_crop_language_model"
        expected_manager = (
            "tgvf_rl.framework.verl.full_qwen_checkpoint_manager."
            "FullQwenBehaviorCheckpointEngineManager"
        )
        if external_components.get("adapter_update_mode") != "unused":
            raise ValueError("method NoTool/Crop unexpectedly owns an Adapter")
        if overrides.get(manager_path) != expected_manager:
            raise ValueError("method NoTool/Crop behavior checkpoint manager differs")
        if engine_kwargs_path in overrides:
            raise ValueError("method NoTool/Crop unexpectedly owns Adapter controls")
    elif profile in tgvf_profiles:
        expected_external = "tgvf_rl.framework.verl.trainable_tgvf_external"
        expected_model_type = "tgvf_trainable_rp66_language_model"
        expected_manager = (
            "tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager."
            "TrainableTGVFCheckpointEngineManager"
        )
        if overrides.get(manager_path) != expected_manager:
            raise ValueError("method TGVF/Atomic checkpoint manager differs")
        adapter_update_mode = external_components.get("adapter_update_mode")
        if adapter_update_mode not in {"joint", "frozen_adapter"}:
            raise ValueError("method TGVF/Atomic Adapter ownership differs")
        if overrides.get(engine_kwargs_path) != {
            "tgvf_control": {"adapter_update_mode": adapter_update_mode}
        }:
            raise ValueError("method TGVF/Atomic Adapter checkpoint controls differ")
    else:
        raise ValueError(f"unknown method-matrix profile {profile!r}")

    if overrides.get("actor_rollout_ref.model.external_lib") != expected_external:
        raise ValueError("method external registration differs")
    if overrides.get("actor_rollout_ref.model.model_type") != expected_model_type:
        raise ValueError("method actor engine differs")


def _assert_performance_surface(overrides: Mapping[str, object]) -> None:
    custom = overrides.get("actor_rollout_ref.rollout.custom")
    if not isinstance(custom, Mapping):
        raise ValueError("Policy launch lost its custom performance receipt")
    performance = custom.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("Policy launch lost its typed performance receipt")
    dynamic = performance.get("dynamic_token_batching")
    cuda_graph = performance.get("vllm_enable_cuda_graph")
    if type(dynamic) is not bool or type(cuda_graph) is not bool:
        raise ValueError("Policy performance switches must be explicit booleans")
    expected = {
        "actor_rollout_ref.actor.use_dynamic_bsz": dynamic,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": dynamic,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": dynamic,
        "actor_rollout_ref.model.use_remove_padding": performance.get(
            "use_remove_padding"
        ),
        "actor_rollout_ref.model.enable_gradient_checkpointing": performance.get(
            "enable_gradient_checkpointing"
        ),
        "actor_rollout_ref.rollout.enable_prefix_caching": performance.get(
            "vllm_enable_prefix_caching"
        ),
        "actor_rollout_ref.rollout.enable_chunked_prefill": performance.get(
            "vllm_enable_chunked_prefill"
        ),
        "actor_rollout_ref.rollout.enforce_eager": not cuda_graph,
        "actor_rollout_ref.rollout.cudagraph_capture_sizes": performance.get(
            "vllm_cuda_graph_capture_sizes"
        ),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": performance.get(
            "vllm_tensor_parallel_size"
        ),
        "actor_rollout_ref.rollout.data_parallel_size": performance.get(
            "vllm_data_parallel_size"
        ),
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": performance.get(
            "vllm_pipeline_parallel_size"
        ),
        "actor_rollout_ref.rollout.disaggregation.enabled": performance.get(
            "vllm_disaggregation_enabled"
        ),
        "actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode": (
            performance.get("rollout_logprob_bypass")
        ),
        "algorithm.rollout_correction.bypass_mode": performance.get(
            "rollout_logprob_bypass"
        ),
    }
    mismatches = {
        path: (overrides.get(path), required)
        for path, required in expected.items()
        if overrides.get(path) != required
    }
    if mismatches:
        raise ValueError(f"Policy performance override drift: {mismatches!r}")
    world_size = overrides.get("trainer.n_gpus_per_node")
    tensor_parallel_size = performance.get("vllm_tensor_parallel_size")
    expected_rollout_replicas = _vllm_rollout_replica_count(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=performance.get("vllm_data_parallel_size"),
        pipeline_parallel_size=performance.get("vllm_pipeline_parallel_size"),
        disaggregation_enabled=performance.get("vllm_disaggregation_enabled"),
    )
    if performance.get("derived_vllm_rollout_replicas") != expected_rollout_replicas:
        raise ValueError("derived vLLM rollout replica receipt differs")
    for path in (
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    ):
        value = overrides.get(path)
        if dynamic and value is not None:
            raise ValueError("dynamic token batching requires fixed micros unset")
        if not dynamic and (type(value) is not int or value <= 0):
            raise ValueError("fixed token batching requires positive micro sizes")
    if overrides.get("actor_rollout_ref.rollout.calculate_log_probs") is not True:
        raise ValueError(
            "rollout-logprob bypass still requires behavior-logprob collection"
        )
    method_matrix = custom.get("method_matrix")
    if isinstance(method_matrix, Mapping):
        expected_loss_mode = (
            DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE
            if dynamic
            else NATIVE_DEEPEYES_POLICY_LOSS_MODE
        )
        if (
            method_matrix.get("actor_loss_mode") != expected_loss_mode
            or method_matrix.get("actor_execution_loss_mode")
            != METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME
            or method_matrix.get("actor_execution_loss_module")
            != METHOD_MATRIX_BYPASS_LOSS_MODULE
            or overrides.get("actor_rollout_ref.actor.policy_loss.loss_mode")
            != expected_loss_mode
        ):
            raise ValueError(
                "method actor loss identity differs from its batching reduction"
            )


__all__ = [
    "EXACT_CURRENT_REFERENCE_REPLAY_FQN",
    "EXACT_REPLAY_ENGINE_REGISTRAR_FQN",
    "EXACT_REPLAY_FORWARD_PORT_FQN",
    "NATIVE_AGENT_LOOP_FQN",
    "NATIVE_AGENT_LOOP_NAME",
    "NATIVE_INVOCATION_FACTORY_FQN",
    "POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA",
    "POLICY_CHECKPOINT_ENGINE_MANAGER_FQN",
    "PolicyCompilePrerequisiteBinding",
    "PolicyCompilePrerequisiteFileReceipt",
    "PolicyCompilePrerequisiteReceipt",
    "UPSTREAM_VERL_CONFIG_NAME",
    "UPSTREAM_VERL_MAIN_MODULE",
    "UPSTREAM_VERL_V0_RUNNER_FQN",
    "UpstreamVerlLaunchPlan",
    "build_policy_e2e_smoke_verl_plan",
    "compose_upstream_verl_config",
    "preflight_policy_compile_prerequisites",
]
