"""Candidate-pinned optional access to veRL's public extension surface.

Importing :mod:`tgvf_rl.framework.verl` must not require a veRL installation.
The real package is resolved only when a runtime-facing factory is called.  In
particular, this module never adds the source snapshot under ``.deps`` to
``sys.path`` and never reaches into a private worker or trainer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata, util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

from tgvf_rl.compatibility_stack import (
    CONTROL_COMPATIBILITY_STACK,
    TORCH211_CU129_COMPATIBILITY_STACK,
    audited_compatibility_stack,
)
from tgvf_rl.framework.vllm import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.policy.config import PolicyPilotV1Config


SPIKE_CANDIDATE_VERL_COMMIT = audited_compatibility_stack(
    CONTROL_COMPATIBILITY_STACK
).verl_commit
TORCH211_CANDIDATE_VERL_COMMIT = audited_compatibility_stack(
    TORCH211_CU129_COMPATIBILITY_STACK
).verl_commit
ACCEPTED_VERL_COMMITS = frozenset(
    {SPIKE_CANDIDATE_VERL_COMMIT, TORCH211_CANDIDATE_VERL_COMMIT}
)
VERL_AGENT_LOOP_RETURN_TRANSPORT = "return_dataproto"
VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT = "transfer_queue"
SUPPORTED_ROLLOUT_BACKEND = "vllm"
SUPPORTED_LOGPROBS_MODE = "processed_logprobs"


class VerlBridgeError(RuntimeError):
    """Base error for a fail-closed veRL bridge."""


class VerlUnavailableError(VerlBridgeError):
    """Raised only when an operation actually needs an unavailable veRL."""


class VerlCompatibilityError(VerlBridgeError):
    """Raised when the installed public veRL surface differs from the pin."""


class VerlConfigurationError(VerlBridgeError):
    """Raised when a runtime configuration weakens a project invariant."""


@dataclass(frozen=True, slots=True)
class VerlPublicAPI:
    """The public symbols used by this repository, resolved without patching."""

    agent_loop_output: type[Any]
    agent_loop_manager: type[Any]
    data_proto: type[Any]
    register_policy_loss: Callable[..., Any]
    get_policy_loss_fn: Callable[[str], Callable[..., Any]]
    compute_grpo_outcome_advantage: Callable[..., Any]
    compute_policy_loss_bypass_mode: Callable[..., Any]
    fsdp_engine_config: type[Any]
    checkpoint_handler: type[Any]
    agent_loop_transport: str


@dataclass(frozen=True, slots=True)
class VerlDistributionIdentity:
    package_version: str
    source_url: str
    commit: str
    source_kind: str
    source_clean: bool | None


def verl_is_available() -> bool:
    """Return whether a normal environment import can locate ``verl``."""

    return util.find_spec("verl") is not None


def _symbol(module: Any, name: str, module_name: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise VerlCompatibilityError(
            f"pinned public veRL symbol {module_name}.{name} is unavailable"
        ) from error


def load_verl_public_api(
    importer: Callable[[str], Any] = import_module,
    *,
    expected_commit: str = SPIKE_CANDIDATE_VERL_COMMIT,
) -> VerlPublicAPI:
    """Resolve only the pinned public APIs when a live integration needs them.

    ``importer`` is injectable so the compatibility contract can be tested in
    an environment where veRL (and its heavyweight Ray/vLLM dependencies) is
    intentionally not installed.
    """

    agent_loop_output_module_name = "verl.experimental.agent_loop"
    if expected_commit == SPIKE_CANDIDATE_VERL_COMMIT:
        agent_loop_manager_module_name = agent_loop_output_module_name
        agent_loop_manager_symbol = "AgentLoopManager"
        agent_loop_transport = VERL_AGENT_LOOP_RETURN_TRANSPORT
    elif expected_commit == TORCH211_CANDIDATE_VERL_COMMIT:
        agent_loop_manager_module_name = "verl.trainer.ppo.v1"
        agent_loop_manager_symbol = "AgentLoopManagerTQ"
        agent_loop_transport = VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT
    else:
        raise VerlCompatibilityError(
            "requested veRL revision is not an accepted compatibility identity"
        )
    module_names = (
        agent_loop_output_module_name,
        agent_loop_manager_module_name,
        "verl.protocol",
        "verl.trainer.ppo.core_algos",
        "verl.workers.config",
        "verl.utils.checkpoint",
    )
    if importer is import_module:
        if sys.version_info < (3, 11):
            raise VerlCompatibilityError(
                "the selected veRL candidate imports enum.StrEnum and requires Python >=3.11"
            )
        if not verl_is_available():
            raise VerlUnavailableError(
                "veRL is optional; install the exact compatibility candidate before constructing the live adapter"
            )
    try:
        (
            agent_loop_output_module,
            agent_loop_manager_module,
            protocol,
            core_algos,
            workers_config,
            checkpoint,
        ) = (importer(name) for name in module_names)
    except (ImportError, ModuleNotFoundError) as error:
        error_type = (
            VerlCompatibilityError
            if importer is import_module
            else VerlUnavailableError
        )
        raise error_type(
            "veRL public modules failed to import for compatibility candidate "
            f"{expected_commit}: {error}"
        ) from error

    if importer is import_module:
        verify_verl_distribution_identity(expected_commit=expected_commit)

    return VerlPublicAPI(
        agent_loop_output=_symbol(
            agent_loop_output_module, "AgentLoopOutput", module_names[0]
        ),
        agent_loop_manager=_symbol(
            agent_loop_manager_module,
            agent_loop_manager_symbol,
            module_names[1],
        ),
        data_proto=_symbol(protocol, "DataProto", module_names[2]),
        register_policy_loss=_symbol(
            core_algos, "register_policy_loss", module_names[3]
        ),
        get_policy_loss_fn=_symbol(core_algos, "get_policy_loss_fn", module_names[3]),
        compute_grpo_outcome_advantage=_symbol(
            core_algos, "compute_grpo_outcome_advantage", module_names[3]
        ),
        compute_policy_loss_bypass_mode=_symbol(
            core_algos, "compute_policy_loss_bypass_mode", module_names[3]
        ),
        fsdp_engine_config=_symbol(workers_config, "FSDPEngineConfig", module_names[4]),
        checkpoint_handler=_symbol(checkpoint, "CheckpointHandler", module_names[5]),
        agent_loop_transport=agent_loop_transport,
    )


def installed_verl_distribution_identity() -> VerlDistributionIdentity:
    """Read the installed wheel/editable provenance and resolve its exact commit."""

    try:
        distribution = metadata.distribution("verl")
    except metadata.PackageNotFoundError as error:
        raise VerlUnavailableError("the veRL distribution is not installed") from error
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise VerlCompatibilityError(
            "veRL direct_url.json is absent; the exact source revision cannot be proven"
        )
    try:
        direct = json.loads(raw)
    except json.JSONDecodeError as error:
        raise VerlCompatibilityError("veRL direct_url.json is malformed") from error
    source_url = direct.get("url")
    if not isinstance(source_url, str) or not source_url:
        raise VerlCompatibilityError("veRL source URL is absent from direct_url.json")
    vcs = direct.get("vcs_info")
    if isinstance(vcs, Mapping):
        commit = vcs.get("commit_id")
        if not isinstance(commit, str):
            raise VerlCompatibilityError("veRL VCS commit is absent")
        return VerlDistributionIdentity(
            distribution.version, source_url, commit, "vcs", None
        )
    parsed = urlparse(source_url)
    if parsed.scheme != "file":
        raise VerlCompatibilityError(
            "veRL must be installed from an exact VCS commit or an auditable local checkout"
        )
    source_path = Path(unquote(parsed.path)).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerlCompatibilityError(
            "local veRL install source is not an auditable git checkout"
        ) from error
    return VerlDistributionIdentity(
        distribution.version, source_url, commit, "local_git", not bool(status)
    )


def verify_verl_distribution_identity(
    *, expected_commit: str = SPIKE_CANDIDATE_VERL_COMMIT
) -> VerlDistributionIdentity:
    if expected_commit not in ACCEPTED_VERL_COMMITS:
        raise VerlCompatibilityError(
            "requested veRL revision is not an accepted compatibility identity"
        )
    identity = installed_verl_distribution_identity()
    if identity.commit != expected_commit:
        raise VerlCompatibilityError(
            "installed veRL revision differs from the requested compatibility identity"
        )
    if identity.source_clean is False:
        raise VerlCompatibilityError("installed local veRL candidate source is dirty")
    return identity


@dataclass(frozen=True, slots=True)
class FSDP2BridgeConfig:
    """Project-required subset of the public veRL FSDP/checkpoint config."""

    world_size: int = 2
    fsdp_size: int = 2
    actor_strategy: str = "fsdp2"
    reference_strategy: str = "fsdp2"
    full_determinism: bool = True
    adapter_dropout: float = 0.0
    checkpoint_async_save: bool = False
    checkpoint_strict: bool = True
    checkpoint_save_contents: tuple[str, ...] = ("model", "optimizer", "extra")
    checkpoint_load_contents: tuple[str, ...] = ("model", "optimizer", "extra")

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_save_contents", tuple(self.checkpoint_save_contents)
        )
        object.__setattr__(
            self, "checkpoint_load_contents", tuple(self.checkpoint_load_contents)
        )
        if type(self.world_size) is not int or self.world_size < 2:
            raise VerlConfigurationError(
                "the accepted FSDP2 probe requires at least two ranks"
            )
        if type(self.fsdp_size) is not int or self.fsdp_size not in {
            -1,
            self.world_size,
        }:
            raise VerlConfigurationError(
                "fsdp_size must be -1 or equal the declared world size"
            )
        if self.actor_strategy != "fsdp2" or self.reference_strategy != "fsdp2":
            raise VerlConfigurationError(
                "actor and frozen reference must both declare FSDP2"
            )
        if self.full_determinism is not True:
            raise VerlConfigurationError(
                "FSDP2 replay requires deterministic forward state"
            )
        if self.adapter_dropout != 0.0:
            raise VerlConfigurationError("policy-adapter dropout must be zero")
        if self.checkpoint_async_save is not False:
            raise VerlConfigurationError(
                "the first proof requires synchronous checkpoints"
            )
        if self.checkpoint_strict is not True:
            raise VerlConfigurationError("checkpoint loading/export must remain strict")
        required = {"model", "optimizer", "extra"}
        if not required.issubset(self.checkpoint_save_contents):
            raise VerlConfigurationError(
                "FSDP2 save must include model, optimizer, and extra state"
            )
        if not required.issubset(self.checkpoint_load_contents):
            raise VerlConfigurationError(
                "FSDP2 load must include model, optimizer, and extra state"
            )


@dataclass(frozen=True, slots=True)
class VerlRuntimeRequirements:
    """Fail-closed values required before a live veRL adapter is constructed."""

    verl_commit: str = SPIKE_CANDIDATE_VERL_COMMIT
    rollout_backend: str = SUPPORTED_ROLLOUT_BACKEND
    calculate_log_probs: bool = True
    logprobs_mode: str = SUPPORTED_LOGPROBS_MODE
    trainer_mode: str = "sync"
    asynchronous_staleness_steps: int = 0
    fsdp2: FSDP2BridgeConfig = field(default_factory=FSDP2BridgeConfig)

    def __post_init__(self) -> None:
        if self.verl_commit not in ACCEPTED_VERL_COMMITS:
            raise VerlConfigurationError(
                "veRL revision must be an exact accepted compatibility identity"
            )
        require_vllm_backend(self.rollout_backend)
        if self.calculate_log_probs is not True:
            raise VerlConfigurationError(
                "actual rollout response_logprobs must be enabled"
            )
        if self.logprobs_mode != SUPPORTED_LOGPROBS_MODE:
            raise VerlConfigurationError(
                "vLLM behavior probabilities must use processed_logprobs"
            )
        if self.trainer_mode != "sync":
            raise VerlConfigurationError(
                "the first replay proof requires the synchronous trainer"
            )
        if (
            type(self.asynchronous_staleness_steps) is not int
            or self.asynchronous_staleness_steps != 0
        ):
            raise VerlConfigurationError(
                "the first replay proof requires zero rollout staleness"
            )


def require_vllm_backend(backend: str) -> None:
    if not isinstance(backend, str) or backend.lower() != SUPPORTED_ROLLOUT_BACKEND:
        raise VerlConfigurationError("vLLM is the only supported rollout backend")


_MISSING = object()


def _path_value(config: object, dotted_path: str) -> object:
    current = config
    for part in dotted_path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            raise VerlConfigurationError(
                f"required veRL config field is missing: {dotted_path}"
            )
    return current


def validate_verl_config_mapping(
    config: object,
    *,
    expected_world_size: int = 2,
    expected_verl_commit: str = SPIKE_CANDIDATE_VERL_COMMIT,
    expected_policy_pilot: PolicyPilotV1Config | None = None,
) -> None:
    """Validate the concrete public veRL config paths used by the pinned commit."""

    if expected_verl_commit not in ACCEPTED_VERL_COMMITS:
        raise VerlConfigurationError(
            "veRL revision must be an exact accepted compatibility identity"
        )

    require_vllm_backend(_path_value(config, "actor_rollout_ref.rollout.name"))
    if _path_value(config, "actor_rollout_ref.rollout.calculate_log_probs") is not True:
        raise VerlConfigurationError(
            "actor_rollout_ref.rollout.calculate_log_probs must be true"
        )
    if (
        _path_value(config, "actor_rollout_ref.rollout.logprobs_mode")
        != SUPPORTED_LOGPROBS_MODE
    ):
        raise VerlConfigurationError(
            "actor_rollout_ref.rollout.logprobs_mode must be processed_logprobs"
        )
    actor_strategy = _path_value(config, "actor_rollout_ref.actor.strategy")
    reference_strategy = _path_value(config, "actor_rollout_ref.ref.strategy")
    actor_fsdp_size = _path_value(
        config, "actor_rollout_ref.actor.fsdp_config.fsdp_size"
    )
    reference_fsdp_size = _path_value(
        config, "actor_rollout_ref.ref.fsdp_config.fsdp_size"
    )
    if reference_fsdp_size != actor_fsdp_size:
        raise VerlConfigurationError(
            "actor and reference FSDP2 shard sizes must be identical"
        )
    actor_determinism = _path_value(
        config, "actor_rollout_ref.actor.fsdp_config.full_determinism"
    )
    reference_determinism = _path_value(
        config, "actor_rollout_ref.ref.fsdp_config.full_determinism"
    )
    if reference_determinism is not actor_determinism:
        raise VerlConfigurationError(
            "actor and reference deterministic-forward settings must be identical"
        )
    checkpoint = _path_value(config, "actor_rollout_ref.actor.checkpoint")
    FSDP2BridgeConfig(
        world_size=expected_world_size,
        fsdp_size=actor_fsdp_size,
        actor_strategy=actor_strategy,
        reference_strategy=reference_strategy,
        full_determinism=actor_determinism,
        # The accepted e003 HF/PEFT builder has no effective dropout override
        # at actor_rollout_ref.model.lora.dropout.  Pilot LoRA dropout is
        # verified against the built PEFT config by policy.model_scope.
        adapter_dropout=0.0,
        checkpoint_async_save=_path_value(checkpoint, "async_save"),
        checkpoint_strict=_path_value(checkpoint, "strict"),
        checkpoint_save_contents=tuple(_path_value(checkpoint, "save_contents")),
        checkpoint_load_contents=tuple(_path_value(checkpoint, "load_contents")),
    )
    expected_use_v1 = expected_verl_commit == TORCH211_CANDIDATE_VERL_COMMIT
    if _path_value(config, "trainer.use_v1") is not expected_use_v1:
        raise VerlConfigurationError(
            "trainer.use_v1 differs from the selected veRL transport"
        )
    if _path_value(config, "trainer.v1.trainer_mode") != "sync":
        raise VerlConfigurationError("trainer.v1.trainer_mode must be sync")
    if expected_use_v1:
        if (
            _path_value(config, "actor_rollout_ref.rollout.free_cache_engine")
            is not False
        ):
            raise VerlConfigurationError(
                "the Torch 2.11 candidate requires no-sleep free_cache_engine=false"
            )
        if (
            _path_value(config, "actor_rollout_ref.rollout.enable_sleep_mode")
            is not False
        ):
            raise VerlConfigurationError(
                "the Torch 2.11 candidate requires enable_sleep_mode=false"
            )
        if (
            _path_value(config, "actor_rollout_ref.rollout.checkpoint_engine.backend")
            != "naive"
        ):
            raise VerlConfigurationError(
                "the Torch 2.11 candidate requires naive colocated weight sync"
            )
    if (
        _path_value(config, "actor_rollout_ref.rollout.enable_prefix_caching")
        is not False
    ):
        raise VerlConfigurationError(
            "prefix caching must be disabled for the first exact-replay proof"
        )
    if (
        _path_value(
            config, "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_mm_embeds"
        )
        is not True
    ):
        raise VerlConfigurationError(
            "vLLM enable_mm_embeds must be true for recorded latent observations"
        )
    if (
        _path_value(
            config, "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb"
        )
        != 0
    ):
        raise VerlConfigurationError(
            "the exact-observation proof requires the vLLM multimodal processor cache off"
        )
    if (
        _path_value(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend",
        )
        != TGVF_VLLM_MM_ENCODER_ATTN_BACKEND
    ):
        raise VerlConfigurationError(
            "vLLM multimodal encoder attention must use TORCH_SDPA for the accepted "
            "driver-portable path"
        )
    hf_overrides = _path_value(
        config, "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"
    )
    if not isinstance(hf_overrides, Mapping) or hf_overrides.get("architectures") != [
        TGVF_QWEN3_VLLM_ARCHITECTURE
    ]:
        raise VerlConfigurationError(
            "vLLM hf_overrides must select the repo-owned Qwen3 latent model"
        )
    limit_images = _path_value(config, "actor_rollout_ref.rollout.limit_images")
    if type(limit_images) is not int or limit_images < 3:
        raise VerlConfigurationError(
            "vLLM image limit must cover source image plus at least two tool calls"
        )
    if expected_policy_pilot is not None:
        _validate_policy_pilot_mapping(
            config,
            expected_policy_pilot,
            expected_verl_commit=expected_verl_commit,
        )


def _validate_policy_pilot_mapping(
    config: object,
    pilot: PolicyPilotV1Config,
    *,
    expected_verl_commit: str,
) -> None:
    """Validate only real public fields in the accepted e003 veRL schema."""

    if not isinstance(pilot, PolicyPilotV1Config):
        raise TypeError("expected_policy_pilot must be PolicyPilotV1Config")
    if expected_verl_commit != SPIKE_CANDIDATE_VERL_COMMIT:
        raise VerlConfigurationError(
            "Policy Pilot v1 overrides are accepted only for the e003 veRL stack"
        )
    sampling = pilot.sampling
    lora = pilot.lora
    grpo = pilot.grpo
    required_values = {
        "actor_rollout_ref.model.path": pilot.model_path,
        "actor_rollout_ref.model.lora_rank": lora.rank,
        "actor_rollout_ref.model.lora_alpha": lora.alpha,
        "actor_rollout_ref.model.target_modules": lora.target_modules,
        "actor_rollout_ref.model.exclude_modules": lora.exclude_modules,
        "actor_rollout_ref.model.external_lib": grpo.verl_external_loss_module,
        "actor_rollout_ref.actor.freeze_vision_tower": True,
        "actor_rollout_ref.actor.optim.lr": lora.initial_learning_rate,
        "actor_rollout_ref.actor.optim.clip_grad": grpo.maximum_gradient_norm,
        "actor_rollout_ref.actor.ppo_epochs": grpo.update_epochs,
        "actor_rollout_ref.actor.clip_ratio": grpo.clip_epsilon_low,
        "actor_rollout_ref.actor.clip_ratio_low": grpo.clip_epsilon_low,
        "actor_rollout_ref.actor.clip_ratio_high": grpo.clip_epsilon_high,
        "actor_rollout_ref.actor.clip_ratio_c": grpo.dual_clip,
        "actor_rollout_ref.actor.loss_agg_mode": grpo.loss_aggregation,
        "actor_rollout_ref.actor.policy_loss.loss_mode": (
            grpo.verl_execution_loss_mode
        ),
        "actor_rollout_ref.actor.entropy_coeff": grpo.entropy_coefficient,
        "actor_rollout_ref.actor.calculate_entropy": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.kl_loss_coef": grpo.kl_loss_coefficient,
        "actor_rollout_ref.rollout.n": sampling.trajectories_per_prompt,
        "actor_rollout_ref.rollout.temperature": sampling.temperature,
        "actor_rollout_ref.rollout.top_p": sampling.top_p,
        "actor_rollout_ref.rollout.top_k": sampling.top_k,
        "actor_rollout_ref.rollout.repetition_penalty": (sampling.repetition_penalty),
        "actor_rollout_ref.rollout.do_sample": sampling.do_sample,
        "actor_rollout_ref.rollout.over_sample_rate": (grpo.rollout_over_sample_rate),
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.limit_images": (1 + pilot.max_tgvf_call_attempts),
        "data.mm_processor_kwargs.max_pixels": pilot.image_max_pixels,
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "algorithm.adv_estimator": grpo.advantage_estimator,
        "algorithm.norm_adv_by_std_in_grpo": grpo.sample_standard_deviation,
        "algorithm.use_kl_in_reward": False,
        "algorithm.kl_ctrl.kl_coef": grpo.kl_reward_coefficient,
        "algorithm.filter_groups.enable": False,
        "algorithm.rollout_correction.rollout_is": (grpo.rollout_importance_sampling),
        "algorithm.rollout_correction.rollout_rs": (grpo.rollout_rejection_sampling),
        "algorithm.rollout_correction.bypass_mode": (
            grpo.rollout_correction_bypass_mode
        ),
        "algorithm.rollout_correction.loss_type": (grpo.rollout_correction_loss_type),
        "algorithm.rollout_correction.rollout_is_batch_normalize": (
            grpo.rollout_is_batch_normalize
        ),
    }
    mismatches = {
        path: (_path_value(config, path), expected)
        for path, expected in required_values.items()
        if _path_value(config, path) != expected
    }
    if mismatches:
        raise VerlConfigurationError(
            f"concrete veRL config differs from Policy Pilot v1: {mismatches!r}"
        )
    rollout_transport_length = _path_value(
        config, "actor_rollout_ref.rollout.response_length"
    )
    data_transport_length = _path_value(config, "data.max_response_length")
    if (
        type(rollout_transport_length) is not int
        or rollout_transport_length <= sampling.max_response_length
        or type(data_transport_length) is not int
        or data_transport_length != rollout_transport_length
    ):
        raise VerlConfigurationError(
            "Policy Pilot requires equal upstream response transport widths that "
            "exceed the cumulative policy-token budget"
        )
