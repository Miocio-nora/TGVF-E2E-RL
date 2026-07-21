"""Upstream-veRL FSDP2 bridge for exact recorded-observation replay.

The pinned veRL ``TrainingWorker`` expects an engine ``forward_step`` to return
full-sequence jagged ``model_output['log_probs']``.  Its padding helpers later
slice response predictions from ``[prompt_len - 1 : sequence_len - 1]``.  The
Qwen replay boundary, correctly, returns response-aligned values instead.  This
module validates the transported DataProto sidecars, invokes that boundary, and
restores the upstream full-sequence layout without accepting raw images or
recomputing any recorded TGVF observation.

No veRL package is imported until registration is explicitly requested.  The
registered class subclasses the installed ``FSDPEngineWithLMHead`` and changes
only ``_build_module`` (to route a distinct registry model type through the
upstream language-model loader) and ``forward_step``.  FSDP2 construction,
microbatching, backward, optimizer, checkpoint, and worker dispatch remain
upstream-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol

import torch
from torch import nn

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.observations.store import TrajectoryReplayBundle

from .data_bridge import (
    DATAPROTO_META_SCHEMA_FIELD,
    DATAPROTO_META_SCHEMA_VERSION,
    PAD_TOKEN_ID_FIELD,
    PADDING_SCHEMA_FIELD,
    PROMPT_TOKEN_OWNERSHIP_FIELD,
    RESPONSE_TOKEN_OWNERSHIP_FIELD,
    DataProtoIntegrityView,
    validate_data_proto_integrity,
)
from .rollout_bridge import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    BRIDGE_SCHEMA_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_SHA256_FIELD,
)
from .policy_weight_sync import wrap_lora_parameter_stream_for_snapshot


TGVF_EXACT_REPLAY_MODEL_TYPE = "tgvf_exact_replay_language_model"

_INTEGRITY_SIDECAR_FIELDS = (
    BRIDGE_SCHEMA_FIELD,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_SHA256_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
    "__num_turns__",
    PADDING_SCHEMA_FIELD,
    PAD_TOKEN_ID_FIELD,
    PROMPT_TOKEN_OWNERSHIP_FIELD,
    RESPONSE_TOKEN_OWNERSHIP_FIELD,
)


class ExactReplayResponseResult(Protocol):
    """Response-aligned result supplied by the concrete Qwen replay boundary."""

    role: ComponentRole
    bundle_sha256: str
    response_token_ids: tuple[int, ...]
    response_ownership: tuple[TokenOwnership, ...]
    policy_sampled_mask: torch.Tensor
    logprobs: torch.Tensor


class ExactReplayResponsePort(Protocol):
    """Minimum model-bound port consumed by the FSDP2 bridge."""

    binding: Any

    def replay_response_logprobs(
        self,
        *,
        bundle: TrajectoryReplayBundle,
        prompt_token_ids: tuple[int, ...],
        response: OwnedTokenSequence,
        sampling: SamplingIdentity,
    ) -> ExactReplayResponseResult: ...


class ExactReplayPortFactory(Protocol):
    """Create a role/mode/bundle-bound port around the engine's FSDP module."""

    def __call__(
        self,
        *,
        engine: Any,
        model: nn.Module,
        role: ComponentRole,
        bundle: TrajectoryReplayBundle,
        model_training: bool,
    ) -> ExactReplayResponsePort: ...


@dataclass(frozen=True, slots=True)
class Qwen3ConfigBoundReplayPortFactory:
    """Build the existing Qwen3 replay port from worker config and sidecars.

    The factory is deliberately stateless.  Every call binds the live FSDP2
    module and the worker's current/reference role to the first verified replay
    bundle in that microbatch.  There is no process-local Python callback for a
    launcher to inject.
    """

    def __call__(
        self,
        *,
        engine: Any,
        model: nn.Module,
        role: ComponentRole,
        bundle: TrajectoryReplayBundle,
        model_training: bool,
    ) -> ExactReplayResponsePort:
        _validate_qwen3_worker_config(engine, bundle=bundle, role=role)
        binding = _qwen3_forward_binding(
            engine=engine,
            model=model,
            role=role,
            bundle=bundle,
            model_training=model_training,
        )
        # Importing the policy package during HFModelConfig construction would
        # create a framework composition cycle.  external_lib registration only
        # stores this factory; the concrete port is imported after the engine and
        # its model have been built.
        from tgvf_rl.policy.qwen_replay import Qwen3RecordedPolicyForwardPort

        return Qwen3RecordedPolicyForwardPort(model=model, binding=binding)


_CONFIG_BOUND_QWEN3_PORT_FACTORY = Qwen3ConfigBoundReplayPortFactory()
_CONFIG_BOUND_QWEN3_ENGINE_CLASS: type[Any] | None = None


@dataclass(frozen=True, slots=True)
class ExactReplayForwardEvidence:
    """Per-microbatch proof retained by the engine for diagnostics/tests."""

    role: ComponentRole
    bundle_sha256s: tuple[str, ...]
    response_lengths: tuple[int, ...]
    full_sequence_lengths: tuple[int, ...]


def restore_response_logprobs_to_full_sequence(
    response_logprobs: torch.Tensor,
    *,
    prompt_length: int,
) -> torch.Tensor:
    """Restore veRL's full next-token layout from response-aligned log-probs.

    For a sequence of length ``prompt_length + response_length``, veRL stores a
    prediction at each full-sequence index and extracts response predictions at
    ``[prompt_length - 1 : sequence_length - 1]``.  Prefix and terminal values
    are deliberately zero because the Policy Pilot loss owns response positions
    only.
    """

    if not isinstance(response_logprobs, torch.Tensor):
        raise TypeError("response_logprobs must be a tensor")
    if response_logprobs.ndim != 1 or not response_logprobs.dtype.is_floating_point:
        raise ValueError("response_logprobs must be floating [response]")
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive integer")
    if response_logprobs.numel() == 0:
        raise ValueError("response_logprobs must be non-empty")
    return torch.cat(
        (
            response_logprobs.new_zeros(prompt_length - 1),
            response_logprobs,
            response_logprobs.new_zeros(1),
        )
    )


def exact_replay_forward_step(
    *,
    engine: Any,
    micro_batch: Any,
    loss_function: Callable[..., Any] | None,
    forward_only: bool,
    port_factory: ExactReplayPortFactory,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Execute one upstream-shaped FSDP forward step from exact sidecars only."""

    if not isinstance(forward_only, bool):
        raise TypeError("forward_only must be bool")
    if not callable(port_factory):
        raise TypeError("exact replay port_factory must be callable")
    module = getattr(engine, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError("exact replay engine must expose its FSDP torch module")
    role = _engine_role(engine)
    integrity = _validate_worker_microbatch(micro_batch)
    _validate_unsupported_outputs(micro_batch)
    sampling = _single_sampling_identity(integrity)
    _validate_one_worker_bundle_identity(integrity)
    first_bundle = integrity.replay_bundles[0]
    port = port_factory(
        engine=engine,
        model=module,
        role=role,
        bundle=first_bundle,
        model_training=bool(module.training),
    )
    _validate_port_role(port, role)
    _unshard_exact_replay_root(module)

    prompt_rows = _batched_sidecar(micro_batch, EXACT_PROMPT_IDS_FIELD)
    response_rows = _batched_sidecar(micro_batch, EXACT_RESPONSE_IDS_FIELD)
    full_rows: list[torch.Tensor] = []
    response_values: list[torch.Tensor] = []
    for row_index, bundle in enumerate(integrity.replay_bundles):
        _require_single_sequence_replay_bundle(bundle)
        prompt_ids = _token_tuple(prompt_rows[row_index], "exact prompt")
        response_ids = _token_tuple(response_rows[row_index], "exact response")
        ownership = integrity.response_token_ownership[row_index][: len(response_ids)]
        response = OwnedTokenSequence(response_ids, ownership)
        result = port.replay_response_logprobs(
            bundle=bundle,
            prompt_token_ids=prompt_ids,
            response=response,
            sampling=sampling,
        )
        values = _validate_response_result(
            result,
            role=role,
            bundle=bundle,
            response=response,
        )
        response_values.append(values)
        full_rows.append(
            restore_response_logprobs_to_full_sequence(
                values,
                prompt_length=len(prompt_ids),
            )
        )

    _validate_row_devices_and_dtypes(full_rows)
    full_logprobs = torch.nested.as_nested_tensor(full_rows, layout=torch.jagged)
    _validate_upstream_response_slice(
        full_logprobs,
        response_values=response_values,
        prompt_lengths=tuple(
            len(_token_tuple(row, "exact prompt")) for row in prompt_rows
        ),
    )
    model_output = {"log_probs": full_logprobs}

    moved_batch = _move_microbatch(micro_batch, full_logprobs.device)
    if loss_function is None:
        if not forward_only:
            raise ValueError("actor update requires an upstream loss function")
        loss = torch.ones((), device=full_logprobs.device)
        metrics: Mapping[str, object] = {}
    else:
        if role is ComponentRole.REFERENCE:
            raise ValueError("frozen reference replay cannot execute a training loss")
        loss_result = loss_function(
            model_output=model_output,
            data=moved_batch,
            dp_group=engine.get_data_parallel_group(),
        )
        if not isinstance(loss_result, tuple) or len(loss_result) != 2:
            raise TypeError("upstream loss function must return (loss, metrics)")
        loss, metrics = loss_result
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("upstream loss must be a scalar tensor")
        if not isinstance(metrics, Mapping):
            raise TypeError("upstream loss metrics must be a mapping")

    if (
        role is ComponentRole.CURRENT
        and loss_function is not None
        and not loss.requires_grad
    ):
        raise ValueError("current exact-replay loss lost its autograd graph")
    if role is ComponentRole.REFERENCE and loss.requires_grad:
        raise ValueError("reference exact-replay placeholder loss carries gradients")
    detached_output = {
        name: value.detach()
        if isinstance(value, torch.Tensor) and value.grad_fn is not None
        else value
        for name, value in model_output.items()
    }
    engine.exact_replay_evidence = ExactReplayForwardEvidence(
        role=role,
        bundle_sha256s=tuple(
            bundle.bundle_sha256 for bundle in integrity.replay_bundles
        ),
        response_lengths=tuple(len(values) for values in response_rows),
        full_sequence_lengths=tuple(int(row.numel()) for row in full_rows),
    )
    return loss, {
        "model_output": detached_output,
        "loss": loss.detach().item(),
        "metrics": dict(metrics),
    }


def _unshard_exact_replay_root(module: nn.Module) -> None:
    """Run the root FSDP2 pre-forward materialization bypassed by replay.

    Exact replay intentionally invokes the injected inner language-model path
    instead of the raw root forward. Child decoder, embedding, and lm-head
    FSDP2 hooks still run normally, but root-owned parameters (notably the
    Qwen final norm) would otherwise remain DTensors. ``FSDPModule.unshard``
    is non-recursive, so this materializes only that root parameter group and
    preserves the child modules' upstream-managed sharding behavior.

    The root is deliberately not resharded here: pinned veRL/PyTorch FSDP2
    likewise configures the root module with ``reshard_after_forward=False``.
    This is also required for actor autograd to retain the exact forward state
    through backward.
    """

    unshard = getattr(module, "unshard", None)
    if not callable(unshard):
        raise RuntimeError(
            "exact replay requires the live root module to expose FSDP2 unshard()"
        )
    handle = unshard()
    if handle is not None:
        raise RuntimeError("synchronous FSDP2 root unshard returned an async handle")


def make_exact_replay_fsdp2_engine_class(
    upstream_engine_cls: type[Any],
    *,
    port_factory: ExactReplayPortFactory,
    model_type: str = TGVF_EXACT_REPLAY_MODEL_TYPE,
) -> type[Any]:
    """Create the custom-model-type subclass used by two TrainingWorkers."""

    _validate_upstream_engine_surface(upstream_engine_cls)
    if not callable(port_factory):
        raise TypeError("exact replay port_factory must be callable")
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type == "language_model"
    ):
        raise ValueError("exact replay requires a distinct non-empty model_type")

    class ExactReplayFSDPEngineWithLMHead(upstream_engine_cls):
        def _build_module(self):
            engine_config = getattr(self, "engine_config", None)
            if getattr(engine_config, "strategy", None) != "fsdp2":
                raise ValueError("the exact replay engine supports FSDP2 only")
            model_config = getattr(self, "model_config", None)
            if getattr(model_config, "model_type", None) != model_type:
                raise IdentityMismatchError(
                    "TrainingWorker model_type differs from the exact replay registry key"
                )
            model_config.model_type = "language_model"
            try:
                module = super()._build_module()
            finally:
                model_config.model_type = model_type
            if getattr(engine_config, "forward_only", False):
                module.requires_grad_(False)
                module.eval()
            return module

        def _build_lora_module(self, module):
            # Actor/ref configs are copied from the same HFModelConfig.  The
            # reference TrainingWorker must nevertheless remain the unadapted
            # frozen base, while the actor uses upstream's normal LoRA builder.
            if bool(getattr(self.engine_config, "forward_only", False)):
                module.requires_grad_(False)
                module.eval()
                return module
            return super()._build_lora_module(module)

        def get_per_tensor_param(
            self,
            layered_summon=False,
            base_sync_done=False,
            **kwargs,
        ):
            """Preserve veRL's stream while publishing the exact actor LoRA.

            The pinned naive checkpoint path uses ``base_sync_done=True`` for
            the adapter-only stream and ``False`` for the one-time base-model
            stream.  Only the current-policy adapter stream is tee'd.  The
            reference worker and base-model stream are returned byte-for-byte
            through the upstream path without requiring publication state.
            """

            result = super().get_per_tensor_param(
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
                **kwargs,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "pinned FSDP2 get_per_tensor_param() must return "
                    "(parameter_stream, peft_config)"
                )
            parameter_stream, peft_config = result
            if not base_sync_done or _engine_role(self) is ComponentRole.REFERENCE:
                return parameter_stream, peft_config
            if peft_config is None:
                raise ReplayMismatchError(
                    "current Policy Pilot adapter sync did not expose a LoRA-only "
                    "parameter stream"
                )
            return (
                wrap_lora_parameter_stream_for_snapshot(
                    parameter_stream,
                    base_sync_done=True,
                ),
                peft_config,
            )

        def forward_step(self, micro_batch, loss_function, forward_only):
            return exact_replay_forward_step(
                engine=self,
                micro_batch=micro_batch,
                loss_function=loss_function,
                forward_only=forward_only,
                port_factory=port_factory,
            )

    ExactReplayFSDPEngineWithLMHead.__name__ = "ExactReplayFSDPEngineWithLMHead"
    ExactReplayFSDPEngineWithLMHead.__qualname__ = "ExactReplayFSDPEngineWithLMHead"
    ExactReplayFSDPEngineWithLMHead.__module__ = __name__
    ExactReplayFSDPEngineWithLMHead.exact_replay_port_factory = port_factory
    ExactReplayFSDPEngineWithLMHead.exact_replay_model_type = model_type
    return ExactReplayFSDPEngineWithLMHead


def register_exact_replay_fsdp2_engine(
    *,
    port_factory: ExactReplayPortFactory,
    registry: Any | None = None,
    upstream_engine_cls: type[Any] | None = None,
    model_type: str = TGVF_EXACT_REPLAY_MODEL_TYPE,
    devices: tuple[str, ...] = ("cuda", "npu"),
) -> type[Any]:
    """Register one distinct FSDP2 model type without replacing upstream keys."""

    if registry is None or upstream_engine_cls is None:
        try:
            from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "exact replay FSDP2 registration requires the pinned veRL package"
            ) from error
        registry = registry or EngineRegistry
        upstream_engine_cls = upstream_engine_cls or FSDPEngineWithLMHead
    if (
        not isinstance(devices, tuple)
        or not devices
        or any(device not in {"cuda", "npu"} for device in devices)
    ):
        raise ValueError("exact replay FSDP2 devices must be cuda/npu identities")
    register = getattr(registry, "register", None)
    if not callable(register):
        raise TypeError("veRL EngineRegistry must expose register()")
    engine_cls = make_exact_replay_fsdp2_engine_class(
        upstream_engine_cls,
        port_factory=port_factory,
        model_type=model_type,
    )
    decorator = register(
        model_type=model_type,
        backend="fsdp2",
        device=list(devices),
    )
    return decorator(engine_cls)


def register_qwen3_exact_replay_fsdp2_engine() -> type[Any]:
    """Install the config-bound Qwen3 engine used by veRL ``external_lib``.

    Importing the Policy Pilot external module calls this once in every Ray
    worker process, before ``TrainingWorker`` resolves its custom model type.
    Repeated calls in the same process are idempotent.
    """

    global _CONFIG_BOUND_QWEN3_ENGINE_CLASS
    if _CONFIG_BOUND_QWEN3_ENGINE_CLASS is None:
        _CONFIG_BOUND_QWEN3_ENGINE_CLASS = register_exact_replay_fsdp2_engine(
            port_factory=_CONFIG_BOUND_QWEN3_PORT_FACTORY,
        )
    return _CONFIG_BOUND_QWEN3_ENGINE_CLASS


def _validate_qwen3_worker_config(
    engine: Any,
    *,
    bundle: TrajectoryReplayBundle,
    role: ComponentRole,
) -> None:
    model_config = getattr(engine, "model_config", None)
    engine_config = getattr(engine, "engine_config", None)
    if model_config is None or engine_config is None:
        raise TypeError("Qwen3 exact replay requires veRL model/engine config")
    if getattr(model_config, "model_type", None) != TGVF_EXACT_REPLAY_MODEL_TYPE:
        raise IdentityMismatchError(
            "Qwen3 worker did not select the exact replay model type"
        )
    if getattr(engine_config, "strategy", None) != "fsdp2":
        raise ValueError("Qwen3 exact replay requires the FSDP2 strategy")
    if getattr(engine_config, "full_determinism", None) is not True:
        raise ValueError("Qwen3 exact replay requires full_determinism=true")
    expected_forward_only = role is ComponentRole.REFERENCE
    if bool(getattr(engine_config, "forward_only", False)) is not expected_forward_only:
        raise IdentityMismatchError(
            "Qwen3 replay role differs from engine forward_only"
        )

    replay_model = bundle.replay_record.model
    if replay_model.family != "qwen3_vl":
        raise IdentityMismatchError(
            "config-bound replay factory supports Qwen3-VL only"
        )
    configured_path = getattr(model_config, "path", None)
    if configured_path != replay_model.revision_or_path:
        raise IdentityMismatchError("veRL model path differs from the replay bundle")
    tokenizer = getattr(model_config, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer_length = len(tokenizer)
        except TypeError as error:
            raise TypeError("veRL tokenizer must expose its exact length") from error
        if tokenizer_length != replay_model.tokenizer_length:
            raise IdentityMismatchError("veRL tokenizer length differs from replay")
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, str) and template:
            template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
            if template_sha256 != replay_model.chat_template_sha256:
                raise IdentityMismatchError("veRL chat template differs from replay")

    hf_config = getattr(model_config, "hf_config", None)
    hf_family = getattr(hf_config, "model_type", None)
    if hf_family is not None and hf_family != replay_model.family:
        raise IdentityMismatchError("HF config family differs from replay")

    from tgvf_rl.policy.config import DecoderLoRAConfig

    lora = DecoderLoRAConfig()
    required_lora = {
        "lora_rank": lora.rank,
        "lora_alpha": lora.alpha,
        "target_modules": lora.target_modules,
        "exclude_modules": lora.exclude_modules,
        "lora_adapter_path": None,
    }
    mismatches = {
        name: (getattr(model_config, name, None), expected)
        for name, expected in required_lora.items()
        if getattr(model_config, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"veRL Qwen3 LoRA config differs from Pilot: {mismatches!r}")


def _qwen3_forward_binding(
    *,
    engine: Any,
    model: nn.Module,
    role: ComponentRole,
    bundle: TrajectoryReplayBundle,
    model_training: bool,
) -> Any:
    from tgvf_rl.policy.exact_replay import (
        RecordedPolicyForwardBinding,
        ReplayParameterization,
    )
    from tgvf_rl.qwen.base import resolve_language_model

    forward_model = _unwrap_peft_model(model)
    embedding = resolve_language_model(forward_model).get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.dtype.is_floating_point:
        raise TypeError("Qwen3 replay embedding must expose a floating tensor")
    compute_dtype = str(weight.dtype).removeprefix("torch.")

    autocast_dtype = getattr(engine, "_autocast_dtype", torch.float32)
    if (
        not isinstance(autocast_dtype, torch.dtype)
        or not autocast_dtype.is_floating_point
    ):
        raise TypeError("veRL FSDP2 autocast dtype must be floating")
    autocast_enabled = autocast_dtype != torch.float32
    autocast_name = (
        str(autocast_dtype).removeprefix("torch.") if autocast_enabled else None
    )
    hf_config = getattr(getattr(engine, "model_config", None), "hf_config", None)
    attention_backend = getattr(hf_config, "_attn_implementation", None)
    if not isinstance(attention_backend, str) or not attention_backend:
        raise ValueError("Qwen3 exact replay requires an explicit attention backend")

    replay = bundle.replay_record
    base_identity_sha256 = _operational_base_identity_sha256(replay.model)
    if role is ComponentRole.CURRENT:
        policy_version = replay.behavior_policy
        parameterization = ReplayParameterization.BASE_PLUS_LORA
        lora_state_sha256 = replay.behavior_policy.weights_sha256
        parameters_frozen = False
    else:
        policy_version = PolicyVersion(
            "qwen3-frozen-base-operational-v1",
            0,
            base_identity_sha256,
        )
        parameterization = ReplayParameterization.FROZEN_BASE
        lora_state_sha256 = None
        parameters_frozen = True
    return RecordedPolicyForwardBinding(
        role=role,
        model=replay.model,
        policy_version=policy_version,
        parameterization=parameterization,
        base_weights_sha256=base_identity_sha256,
        lora_state_sha256=lora_state_sha256,
        parameters_frozen=parameters_frozen,
        deterministic_forward=replay.deterministic_forward,
        lora_dropout=replay.adapter_dropout,
        model_training=model_training,
        compute_dtype=compute_dtype,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_name,
        attention_backend=attention_backend,
        forward_implementation_sha256=_qwen3_forward_implementation_sha256(),
    )


def _unwrap_peft_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    unwrapped = getter() if callable(getter) else model
    if not isinstance(unwrapped, nn.Module):
        raise TypeError("Qwen3 PEFT model did not expose a torch base model")
    return unwrapped


def _operational_base_identity_sha256(model: ModelIdentity) -> str:
    if not isinstance(model, ModelIdentity):
        raise TypeError("operational base identity requires ModelIdentity")
    canonical = json.dumps(
        {
            "schema": "tgvf-operational-base-model-identity-v1",
            "family": model.family,
            "model_name": model.model_name,
            "revision_or_path": model.revision_or_path,
            "tokenizer_length": model.tokenizer_length,
            "chat_template_sha256": model.chat_template_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _qwen3_forward_implementation_sha256() -> str:
    from tgvf_rl.policy.qwen_replay import Qwen3RecordedPolicyForwardPort
    from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter

    try:
        source = "\n".join(
            (
                inspect.getsource(Qwen3RecordedPolicyForwardPort),
                inspect.getsource(Qwen3VLAdapter.forward_replay_bundle),
            )
        )
    except (OSError, TypeError) as error:
        raise RuntimeError("cannot identify the Qwen3 replay implementation") from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _engine_role(engine: Any) -> ComponentRole:
    engine_config = getattr(engine, "engine_config", None)
    if getattr(engine_config, "strategy", None) != "fsdp2":
        raise ValueError("exact replay forward requires engine strategy fsdp2")
    return (
        ComponentRole.REFERENCE
        if bool(getattr(engine_config, "forward_only", False))
        else ComponentRole.CURRENT
    )


def _validate_worker_microbatch(micro_batch: Any) -> DataProtoIntegrityView:
    if not hasattr(micro_batch, "__getitem__") or not hasattr(
        micro_batch, "__contains__"
    ):
        raise TypeError("veRL exact replay micro_batch must be mapping-like")
    meta_schema = _unwrapped_value(
        micro_batch[DATAPROTO_META_SCHEMA_FIELD]
        if DATAPROTO_META_SCHEMA_FIELD in micro_batch
        else None
    )
    if meta_schema != DATAPROTO_META_SCHEMA_VERSION:
        raise RuntimeError("worker TensorDict lost the DataProto meta schema")
    non_tensors = {
        name: _batched_sidecar(micro_batch, name)
        for name in _INTEGRITY_SIDECAR_FIELDS
        if name in micro_batch
    }
    return validate_data_proto_integrity(
        SimpleNamespace(batch=micro_batch, non_tensor_batch=non_tensors)
    )


def _validate_unsupported_outputs(micro_batch: Any) -> None:
    for name in (
        "calculate_entropy",
        "calculate_sum_pi_squared",
        "distillation_use_topk",
        "distillation_only",
        "use_fused_kernels",
    ):
        if name in micro_batch and _as_bool(_unwrapped_value(micro_batch[name]), name):
            raise ValueError(f"exact replay engine does not materialize {name}")
    if "temperature" in micro_batch:
        temperature = _unwrapped_value(micro_batch["temperature"])
        if isinstance(temperature, torch.Tensor):
            accepted = bool((temperature == 1.0).all().item())
        elif isinstance(temperature, (tuple, list)):
            accepted = bool(temperature) and all(value == 1.0 for value in temperature)
        else:
            accepted = temperature == 1.0
        if not accepted:
            raise ValueError("exact replay engine requires Pilot temperature 1.0")


def _single_sampling_identity(integrity: DataProtoIntegrityView) -> SamplingIdentity:
    samplings: list[SamplingIdentity] = []
    for records, bundle in zip(
        integrity.behavior_trace_records,
        integrity.replay_bundles,
        strict=True,
    ):
        for record in records:
            sampling = getattr(getattr(record, "behavior", None), "sampling", None)
            if not isinstance(sampling, SamplingIdentity):
                raise TypeError("exact replay sidecars lost SamplingIdentity")
            if sampling.policy_version != bundle.replay_record.behavior_policy:
                raise IdentityMismatchError(
                    "behavior sampling policy differs from the replay bundle"
                )
            if sampling.asynchronous_staleness_steps != 0:
                raise ReplayMismatchError(
                    "Policy Pilot exact replay requires staleness zero"
                )
            samplings.append(sampling)
    if not samplings:
        raise ValueError("exact replay requires behavior sampling evidence")
    if len({item.transform_identity_sha256 for item in samplings}) != 1:
        raise ReplayMismatchError("one FSDP2 microbatch mixed sampling measures")
    if any(not item.has_identity_sampling_transforms for item in samplings):
        raise ValueError(
            "current/reference processed-logprob replay remains fail-closed for "
            "non-identity sampling transforms"
        )
    return samplings[0]


def _validate_one_worker_bundle_identity(integrity: DataProtoIntegrityView) -> None:
    models = {bundle.replay_record.model for bundle in integrity.replay_bundles}
    if len(models) != 1:
        raise IdentityMismatchError("one FSDP2 microbatch mixed model identities")
    policies = {
        bundle.replay_record.behavior_policy for bundle in integrity.replay_bundles
    }
    if len(policies) != 1:
        raise IdentityMismatchError("one FSDP2 microbatch mixed policy snapshots")


def _validate_port_role(port: ExactReplayResponsePort, role: ComponentRole) -> None:
    method = getattr(port, "replay_response_logprobs", None)
    if not callable(method):
        raise TypeError("exact replay port must expose replay_response_logprobs()")
    binding_role = getattr(getattr(port, "binding", None), "role", None)
    if binding_role is not role:
        raise IdentityMismatchError(
            "exact replay port role differs from its TrainingWorker"
        )


def _validate_response_result(
    result: ExactReplayResponseResult,
    *,
    role: ComponentRole,
    bundle: TrajectoryReplayBundle,
    response: OwnedTokenSequence,
) -> torch.Tensor:
    if getattr(result, "role", None) is not role:
        raise IdentityMismatchError("exact replay result has another worker role")
    if getattr(result, "bundle_sha256", None) != bundle.bundle_sha256:
        raise ReplayMismatchError("exact replay result used another bundle digest")
    if tuple(getattr(result, "response_token_ids", ())) != response.token_ids:
        raise ReplayMismatchError("exact replay result changed response token IDs")
    if tuple(getattr(result, "response_ownership", ())) != response.ownership:
        raise ReplayMismatchError("exact replay result changed response ownership")
    values = getattr(result, "logprobs", None)
    mask = getattr(result, "policy_sampled_mask", None)
    if not isinstance(values, torch.Tensor) or values.shape != (
        len(response.token_ids),
    ):
        raise ValueError("exact replay result logprobs must be response-aligned")
    if not values.dtype.is_floating_point or not bool(
        torch.isfinite(values.detach()).all().item()
    ):
        raise ValueError(
            "exact replay response logprobs must be finite floating values"
        )
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype is not torch.bool
        or mask.shape != values.shape
    ):
        raise ValueError(
            "exact replay result must carry a bool response ownership mask"
        )
    expected_mask = torch.tensor(
        tuple(owner is TokenOwnership.POLICY_SAMPLED for owner in response.ownership),
        dtype=torch.bool,
        device=mask.device,
    )
    if not torch.equal(mask, expected_mask):
        raise ReplayMismatchError(
            "exact replay result changed the policy ownership mask"
        )
    if values.device != mask.device:
        raise ValueError("exact replay logprobs and ownership mask must share a device")
    if bool(torch.count_nonzero(values[~mask]).item()):
        raise ReplayMismatchError("non-policy response positions carry replay logprobs")
    if role is ComponentRole.CURRENT and not values.requires_grad:
        raise ValueError("current exact replay logprobs lost their autograd graph")
    if role is ComponentRole.REFERENCE and values.requires_grad:
        raise ValueError("reference exact replay logprobs carry gradients")
    return values


def _validate_row_devices_and_dtypes(rows: list[torch.Tensor]) -> None:
    if not rows:
        raise ValueError("exact replay FSDP2 microbatch cannot be empty")
    devices = {row.device for row in rows}
    dtypes = {row.dtype for row in rows}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("exact replay rows must share one device and floating dtype")


def _require_single_sequence_replay_bundle(
    bundle: TrajectoryReplayBundle,
) -> None:
    if not isinstance(bundle, TrajectoryReplayBundle):
        raise TypeError("replay evidence must be a TrajectoryReplayBundle")
    input_shape = bundle.replay_record.tensors.input_ids.descriptor.shape
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ReplayMismatchError(
            "one trajectory replay bundle must contain exactly one sequence (B=1)"
        )


def _validate_upstream_response_slice(
    full_logprobs: torch.Tensor,
    *,
    response_values: list[torch.Tensor],
    prompt_lengths: tuple[int, ...],
) -> None:
    values = full_logprobs.values()
    offsets = full_logprobs.offsets()
    for row_index, (response, prompt_length) in enumerate(
        zip(response_values, prompt_lengths, strict=True)
    ):
        start = int(offsets[row_index].item()) + prompt_length - 1
        actual = values[start : start + response.numel()]
        if not torch.equal(actual, response):
            raise ReplayMismatchError(
                "restored full-sequence logprobs fail veRL's response slice"
            )


def _validate_upstream_engine_surface(upstream_engine_cls: type[Any]) -> None:
    if not isinstance(upstream_engine_cls, type):
        raise TypeError("upstream FSDP engine must be a class")
    for name in ("_build_module", "forward_step", "get_per_tensor_param"):
        if not callable(getattr(upstream_engine_cls, name, None)):
            raise TypeError(f"upstream FSDP engine is missing {name}()")
    parameters = tuple(inspect.signature(upstream_engine_cls.forward_step).parameters)
    if parameters != ("self", "micro_batch", "loss_function", "forward_only"):
        raise RuntimeError("pinned FSDPEngineWithLMHead.forward_step signature changed")
    weight_parameters = tuple(
        inspect.signature(upstream_engine_cls.get_per_tensor_param).parameters
    )
    if weight_parameters != (
        "self",
        "layered_summon",
        "base_sync_done",
        "kwargs",
    ):
        raise RuntimeError(
            "pinned FSDPEngineWithLMHead.get_per_tensor_param signature changed"
        )


def _batched_sidecar(micro_batch: Any, name: str) -> Any:
    if name not in micro_batch:
        raise ValueError(f"worker TensorDict is missing exact sidecar {name!r}")
    value = _unwrapped_value(micro_batch[name])
    if isinstance(value, torch.Tensor):
        raise TypeError(f"exact sidecar {name!r} cannot be a tensor")
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _unwrapped_value(value: Any) -> Any:
    data = getattr(value, "data", None)
    if data is not None and not isinstance(value, torch.Tensor):
        return data
    return value


def _token_tuple(value: Any, owner: str) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise TypeError(f"{owner} IDs must be an iterable") from error
    if not result or any(
        type(token_id) is not int or token_id < 0 for token_id in result
    ):
        raise ValueError(f"{owner} IDs must be non-empty non-negative integers")
    return result


def _as_bool(value: Any, owner: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return bool(value.item())
    if (
        isinstance(value, (tuple, list))
        and value
        and all(isinstance(item, bool) for item in value)
    ):
        if len(set(value)) != 1:
            raise ValueError(f"one microbatch mixed {owner} values")
        return value[0]
    raise TypeError(f"{owner} must be a scalar bool")


def _move_microbatch(micro_batch: Any, device: torch.device) -> Any:
    mover = getattr(micro_batch, "to", None)
    return mover(device) if callable(mover) else micro_batch


__all__ = [
    "ExactReplayForwardEvidence",
    "ExactReplayPortFactory",
    "ExactReplayResponsePort",
    "ExactReplayResponseResult",
    "Qwen3ConfigBoundReplayPortFactory",
    "TGVF_EXACT_REPLAY_MODEL_TYPE",
    "exact_replay_forward_step",
    "make_exact_replay_fsdp2_engine_class",
    "register_exact_replay_fsdp2_engine",
    "register_qwen3_exact_replay_fsdp2_engine",
    "restore_response_logprobs_to_full_sequence",
]
