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

from tgvf_rl.framework.vllm import TGVF_QWEN3_VLLM_ARCHITECTURE


SPIKE_CANDIDATE_VERL_COMMIT = "e003163181731412595257a72ec173071efb125f"
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
    fsdp_engine_config: type[Any]
    checkpoint_handler: type[Any]


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
) -> VerlPublicAPI:
    """Resolve only the pinned public APIs when a live integration needs them.

    ``importer`` is injectable so the compatibility contract can be tested in
    an environment where veRL (and its heavyweight Ray/vLLM dependencies) is
    intentionally not installed.
    """

    module_names = (
        "verl.experimental.agent_loop",
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
        agent_loop, protocol, core_algos, workers_config, checkpoint = (
            importer(name) for name in module_names
        )
    except (ImportError, ModuleNotFoundError) as error:
        error_type = (
            VerlCompatibilityError
            if importer is import_module
            else VerlUnavailableError
        )
        raise error_type(
            "veRL public modules failed to import for compatibility candidate "
            f"{SPIKE_CANDIDATE_VERL_COMMIT}: {error}"
        ) from error

    if importer is import_module:
        verify_verl_distribution_identity()

    return VerlPublicAPI(
        agent_loop_output=_symbol(agent_loop, "AgentLoopOutput", module_names[0]),
        agent_loop_manager=_symbol(agent_loop, "AgentLoopManager", module_names[0]),
        data_proto=_symbol(protocol, "DataProto", module_names[1]),
        register_policy_loss=_symbol(
            core_algos, "register_policy_loss", module_names[2]
        ),
        fsdp_engine_config=_symbol(workers_config, "FSDPEngineConfig", module_names[3]),
        checkpoint_handler=_symbol(checkpoint, "CheckpointHandler", module_names[4]),
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


def verify_verl_distribution_identity() -> VerlDistributionIdentity:
    identity = installed_verl_distribution_identity()
    if identity.commit != SPIKE_CANDIDATE_VERL_COMMIT:
        raise VerlCompatibilityError(
            "installed veRL revision differs from the accepted compatibility candidate"
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
        if self.verl_commit != SPIKE_CANDIDATE_VERL_COMMIT:
            raise VerlConfigurationError(
                f"veRL revision must be the spike candidate {SPIKE_CANDIDATE_VERL_COMMIT}"
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
    config: object, *, expected_world_size: int = 2
) -> None:
    """Validate the concrete public veRL config paths used by the pinned commit."""

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
        adapter_dropout=_path_value(config, "actor_rollout_ref.model.lora.dropout"),
        checkpoint_async_save=_path_value(checkpoint, "async_save"),
        checkpoint_strict=_path_value(checkpoint, "strict"),
        checkpoint_save_contents=tuple(_path_value(checkpoint, "save_contents")),
        checkpoint_load_contents=tuple(_path_value(checkpoint, "load_contents")),
    )
    if _path_value(config, "trainer.v1.trainer_mode") != "sync":
        raise VerlConfigurationError("trainer.v1.trainer_mode must be sync")
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
