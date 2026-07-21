"""Two-model TGVF runtime mounted in the existing veRL vLLM worker.

The AgentLoop side of this module owns only CPU protocol state. Source vision,
behavior Hq capture, and frozen TGVF Adapter execution happen inside the same
vLLM worker that samples the native tool call. No Hugging Face policy model is
loaded in an AgentLoop process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
from types import MethodType
from typing import Any
from uuid import uuid4

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.environment.focus_tool import (
    PrecomputedTGVFObservationPayload,
    SourceVisualTensorBundle,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.representation import TGVFAdapter, TGVFAdapterInput
from tgvf_rl.representation.adapter import (
    TGVFAdapterMetadata,
    TGVFAdapterOutput,
    TGVFAdapterVariant,
)
from tgvf_rl.representation.deepstack import (
    DDeepStackPayload,
    DDeepStackProjectionPorts,
    FrozenProjectionPort,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)

try:  # Optional until the live veRL environment imports this extension.
    from verl.workers.rollout.vllm_rollout.utils import (
        vLLMColocateWorkerExtension as _VerlVLLMWorkerExtension,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU base env
    _VerlVLLMWorkerExtension = object


TGVF_VLLM_WORKER_EXTENSION_FQN = (
    "tgvf_rl.framework.verl.vllm_tool_runtime.TGVFVLLMWorkerExtension"
)
TGVF_TWO_MODEL_RUNTIME_SCHEMA = "tgvf-vllm-two-model-runtime-v1"
_SOURCE_WIRE_SCHEMA = "tgvf-source-visual-utility-wire-v1"
_FOCUS_WIRE_SCHEMA = "tgvf-focus-utility-wire-v1"


def _tensor_to_utility_wire(value: torch.Tensor) -> dict[str, object]:
    """Encode a CPU tensor without vLLM's untyped utility tensor extension."""

    tensor = value.detach().cpu().contiguous()
    return {
        "data": tensor.view(torch.uint8).numpy().tobytes(),
        "shape": tuple(int(size) for size in tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _tensor_from_utility_wire(value: Mapping[str, object]) -> torch.Tensor:
    """Decode a tensor after the utility payload crosses both vLLM IPC hops."""

    data = value.get("data")
    shape = value.get("shape")
    dtype_name = value.get("dtype")
    if not isinstance(data, bytes):
        raise TypeError("TGVF utility tensor data must be bytes")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError("TGVF utility tensor shape must be a sequence")
    normalized_shape = tuple(int(size) for size in shape)
    if any(size < 0 for size in normalized_shape):
        raise ValueError("TGVF utility tensor shape must be non-negative")
    if not isinstance(dtype_name, str):
        raise TypeError("TGVF utility tensor dtype must be a string")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("TGVF utility tensor dtype is unsupported")
    tensor = torch.frombuffer(bytearray(data), dtype=dtype).clone()
    expected = 1
    for size in normalized_shape:
        expected *= size
    if tensor.numel() != expected:
        raise ValueError("TGVF utility tensor byte length differs from shape")
    return tensor.reshape(normalized_shape)


def _source_to_utility_wire(value: SourceVisualTensorBundle) -> dict[str, object]:
    source = _source_to_cpu(value)
    return {
        "schema": _SOURCE_WIRE_SCHEMA,
        "image_sha256": source.image_sha256,
        "premerge_main": _tensor_to_utility_wire(source.premerge_main),
        "premerge_deepstack": [
            _tensor_to_utility_wire(item) for item in source.premerge_deepstack
        ],
        "merged_main": _tensor_to_utility_wire(source.merged_main),
        "merged_deepstack": [
            _tensor_to_utility_wire(item) for item in source.merged_deepstack
        ],
        "image_grid_thw": list(source.image_grid_thw),
        "spatial_merge_size": source.spatial_merge_size,
        "decoded_rgb_sha256": source.decoded_rgb_sha256,
    }


def _source_from_utility_wire(value: Mapping[str, object]) -> SourceVisualTensorBundle:
    if value.get("schema") != _SOURCE_WIRE_SCHEMA:
        raise ValueError("vLLM source utility wire schema differs")
    premerge = _wire_tensor_rows(value.get("premerge_deepstack"), owner="premerge")
    merged = _wire_tensor_rows(value.get("merged_deepstack"), owner="merged")
    grid = _wire_integer_tuple(value.get("image_grid_thw"), owner="image grid")
    if len(grid) != 3:
        raise ValueError("vLLM source image grid must contain three values")
    source = SourceVisualTensorBundle(
        image_sha256=_wire_string(value.get("image_sha256"), owner="image SHA256"),
        premerge_main=_tensor_from_utility_wire(
            _wire_mapping(value.get("premerge_main"), owner="premerge main")
        ),
        premerge_deepstack=premerge,
        merged_main=_tensor_from_utility_wire(
            _wire_mapping(value.get("merged_main"), owner="merged main")
        ),
        merged_deepstack=merged,
        image_grid_thw=(grid[0], grid[1], grid[2]),
        spatial_merge_size=_wire_integer(
            value.get("spatial_merge_size"), owner="spatial merge size"
        ),
        decoded_rgb_sha256=_wire_optional_string(
            value.get("decoded_rgb_sha256"), owner="decoded RGB SHA256"
        ),
    )
    _validate_source_geometry(source)
    return source


def _focus_to_utility_wire(
    value: TGVFFocusMaterializationResult,
) -> dict[str, object]:
    observation = value.observation
    metadata = observation.metadata
    if metadata.condition_provenance is not None:
        raise RuntimeError("colocated TGVF utility metadata must omit provenance")
    return {
        "schema": _FOCUS_WIRE_SCHEMA,
        "hq": _tensor_to_utility_wire(value.hq),
        "main_d": _tensor_to_utility_wire(observation.main_d),
        "d_deepstack": {
            "branch_layers": list(observation.d_deepstack.branch_layers),
            "branches": [
                _tensor_to_utility_wire(item)
                for item in observation.d_deepstack.branches
            ],
            "projection_identities": list(
                observation.d_deepstack.projection_identities
            ),
            "schema_version": observation.d_deepstack.schema_version,
        },
        "metadata": {
            "branch_layers": list(metadata.branch_layers),
            "main_projection_identity": metadata.main_projection_identity,
            "deepstack_projection_identities": list(
                metadata.deepstack_projection_identities
            ),
            "batched": metadata.batched,
            "batch_size": metadata.batch_size,
            "target_token_count": metadata.target_token_count,
            "pre_merge_visual_token_count": metadata.pre_merge_visual_token_count,
            "d_token_count": metadata.d_token_count,
            "variant": metadata.variant.value,
            "schema_version": metadata.schema_version,
        },
    }


def _focus_from_utility_wire(
    value: Mapping[str, object],
) -> TGVFFocusMaterializationResult:
    if value.get("schema") != _FOCUS_WIRE_SCHEMA:
        raise ValueError("vLLM focus utility wire schema differs")
    deep = _wire_mapping(value.get("d_deepstack"), owner="D-DeepStack")
    metadata_wire = _wire_mapping(value.get("metadata"), owner="Adapter metadata")
    deepstack = DDeepStackPayload(
        branch_layers=_wire_integer_tuple(
            deep.get("branch_layers"), owner="D-DeepStack layers"
        ),
        branches=_wire_tensor_rows(deep.get("branches"), owner="D-DeepStack"),
        projection_identities=_wire_string_tuple(
            deep.get("projection_identities"), owner="D-DeepStack identities"
        ),
        schema_version=_wire_string(
            deep.get("schema_version"), owner="D-DeepStack schema"
        ),
    )
    metadata = TGVFAdapterMetadata(
        branch_layers=_wire_integer_tuple(
            metadata_wire.get("branch_layers"), owner="Adapter branch layers"
        ),
        main_projection_identity=_wire_string(
            metadata_wire.get("main_projection_identity"),
            owner="main projection identity",
        ),
        deepstack_projection_identities=_wire_string_tuple(
            metadata_wire.get("deepstack_projection_identities"),
            owner="Adapter DeepStack identities",
        ),
        batched=_wire_boolean(metadata_wire.get("batched"), owner="Adapter batched"),
        batch_size=_wire_integer(
            metadata_wire.get("batch_size"), owner="Adapter batch size"
        ),
        target_token_count=_wire_integer(
            metadata_wire.get("target_token_count"), owner="target token count"
        ),
        pre_merge_visual_token_count=_wire_integer(
            metadata_wire.get("pre_merge_visual_token_count"),
            owner="premerge visual token count",
        ),
        d_token_count=_wire_integer(
            metadata_wire.get("d_token_count"), owner="D token count"
        ),
        condition_provenance=None,
        variant=TGVFAdapterVariant(
            _wire_string(metadata_wire.get("variant"), owner="Adapter variant")
        ),
        schema_version=_wire_string(
            metadata_wire.get("schema_version"), owner="Adapter schema"
        ),
    )
    return TGVFFocusMaterializationResult(
        hq=_tensor_from_utility_wire(_wire_mapping(value.get("hq"), owner="Hq")),
        observation=PrecomputedTGVFObservationPayload(
            main_d=_tensor_from_utility_wire(
                _wire_mapping(value.get("main_d"), owner="main D")
            ),
            d_deepstack=deepstack,
            metadata=metadata,
        ),
    )


def _wire_mapping(value: object, *, owner: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} utility wire value must be a mapping")
    return value


def _wire_tensor_rows(value: object, *, owner: str) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} utility tensor rows must be a sequence")
    return tuple(
        _tensor_from_utility_wire(_wire_mapping(item, owner=f"{owner} row"))
        for item in value
    )


def _wire_integer(value: object, *, owner: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} utility wire value must be an integer")
    return value


def _wire_integer_tuple(value: object, *, owner: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} utility wire value must be a sequence")
    return tuple(_wire_integer(item, owner=owner) for item in value)


def _wire_string(value: object, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{owner} utility wire value must be a non-empty string")
    return value


def _wire_optional_string(value: object, *, owner: str) -> str | None:
    return None if value is None else _wire_string(value, owner=owner)


def _wire_string_tuple(value: object, *, owner: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} utility wire value must be a sequence")
    return tuple(_wire_string(item, owner=owner) for item in value)


def _wire_boolean(value: object, *, owner: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{owner} utility wire value must be a boolean")
    return value


@dataclass(slots=True)
class _BehaviorTraceBuffer:
    prompt_length: int
    capacity: int
    hidden: torch.Tensor
    covered: bytearray


@dataclass(frozen=True, slots=True)
class _TurnRoute:
    backend_request_id: str
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    global_step: int


@dataclass(frozen=True, slots=True)
class TGVFFocusMaterializationResult:
    """Typed HTTP-server result reconstructed from the primitive utility wire."""

    hq: torch.Tensor
    observation: PrecomputedTGVFObservationPayload

    def __post_init__(self) -> None:
        if not isinstance(self.hq, torch.Tensor) or self.hq.ndim != 2:
            raise ValueError("TGVF focus Hq must be a rank-two tensor")
        if not isinstance(self.observation, PrecomputedTGVFObservationPayload):
            raise TypeError("TGVF focus result requires a typed observation payload")


class TGVFVLLMWorkerExtension(_VerlVLLMWorkerExtension):
    """veRL worker extension adding small, exact TGVF operations."""

    def tgvf_register_behavior_trace(
        self,
        request_id: str,
        prompt_length: int,
        maximum_output_tokens: int,
    ) -> dict[str, object]:
        if not request_id:
            raise ValueError("behavior trace request_id must be non-empty")
        if prompt_length <= 0 or maximum_output_tokens <= 0:
            raise ValueError("behavior trace lengths must be positive")
        self._tgvf_install_trace_hook()
        traces = self._tgvf_traces()
        if request_id in traces:
            raise RuntimeError("behavior trace request_id was reused")
        model = self._tgvf_model()
        hidden_size = int(model.config.text_config.hidden_size)
        parameter = next(model.language_model.parameters())
        traces[request_id] = _BehaviorTraceBuffer(
            prompt_length=prompt_length,
            capacity=maximum_output_tokens,
            hidden=torch.empty(
                (maximum_output_tokens, hidden_size),
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            covered=bytearray(maximum_output_tokens),
        )
        return {
            "schema_version": TGVF_TWO_MODEL_RUNTIME_SCHEMA,
            "request_id": request_id,
            "prompt_length": prompt_length,
            "maximum_output_tokens": maximum_output_tokens,
        }

    def tgvf_materialize_source(
        self,
        trajectory_id: str,
        pixel_values_wire: Mapping[str, object],
        image_grid_thw: Sequence[int],
        image_sha256: str,
    ) -> dict[str, object]:
        pixel_values = _tensor_from_utility_wire(pixel_values_wire)
        image_grid_thw = torch.tensor((tuple(image_grid_thw),), dtype=torch.long)
        if not trajectory_id:
            raise ValueError("source trajectory_id must be non-empty")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 2:
            raise ValueError("source pixel_values must have shape [N,patch]")
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.shape != (
            1,
            3,
        ):
            raise ValueError("source image_grid_thw must have shape [1,3]")
        model = self._tgvf_model()
        visual = model.visual
        if visual is None:
            raise RuntimeError("vLLM Qwen3 worker has no visual tower")
        cache = self._tgvf_sources()
        cached = cache.get(trajectory_id)
        if cached is not None:
            if cached.image_sha256 != image_sha256:
                raise RuntimeError("trajectory source image identity changed")
            return _source_to_utility_wire(cached)

        mergers = (visual.merger, *tuple(visual.deepstack_merger_list))
        captures: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in mergers
        ]
        handles = tuple(
            merger.register_forward_hook(
                _capture_vllm_merger(captures[index]), with_kwargs=True
            )
            for index, merger in enumerate(mergers)
        )
        try:
            owner = next(visual.parameters())
            with torch.inference_mode():
                visual(
                    pixel_values.detach().to(
                        device=owner.device,
                        dtype=owner.dtype,
                    ),
                    grid_thw=image_grid_thw.detach().to(
                        device="cpu", dtype=torch.long
                    ),
                )
        finally:
            for handle in handles:
                handle.remove()
        if any(len(rows) != 1 for rows in captures):
            raise RuntimeError("vLLM source vision did not execute every merger once")
        premerge = tuple(rows[0][0] for rows in captures)
        merged = tuple(rows[0][1] for rows in captures)
        grid = tuple(int(value) for value in image_grid_thw[0].tolist())
        source = SourceVisualTensorBundle(
            image_sha256=image_sha256,
            premerge_main=premerge[0],
            premerge_deepstack=premerge[1:],
            merged_main=merged[0],
            merged_deepstack=merged[1:],
            image_grid_thw=grid,
            spatial_merge_size=int(visual.spatial_merge_size),
            decoded_rgb_sha256=image_sha256,
        )
        _validate_source_geometry(source)
        cache[trajectory_id] = source
        return _source_to_utility_wire(source)

    def tgvf_materialize_focus(
        self,
        trajectory_id: str,
        backend_request_id: str,
        target_start: int,
        target_end: int,
        expected_target_token_ids: tuple[int, ...],
        provider: str,
    ) -> dict[str, object]:
        source = self._tgvf_sources().get(trajectory_id)
        if source is None:
            raise RuntimeError("TGVF source was not materialized on this vLLM worker")
        if target_start < 0 or target_end <= target_start:
            raise ValueError("target span must be non-empty")
        expected = tuple(expected_target_token_ids)
        if len(expected) != target_end - target_start:
            raise ValueError("target token IDs do not cover target span")
        adapter = self._tgvf_adapter()
        model = self._tgvf_model()
        if provider == "contextual_hidden_state":
            trace = self._tgvf_traces().get(backend_request_id)
            if trace is None:
                raise RuntimeError("behavior trace is unavailable for the tool turn")
            if target_end > trace.capacity or not all(
                trace.covered[target_start:target_end]
            ):
                raise RuntimeError(
                    "behavior forward did not cover every sampled target token"
                )
            hq = trace.hidden[target_start:target_end].detach().clone()
        elif provider == "target_token_embedding":
            token_ids = torch.tensor(
                expected,
                device=next(model.language_model.parameters()).device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                hq = model.language_model.embed_input_ids(token_ids).detach()
        else:
            raise ValueError("unknown target-conditioning provider")
        with torch.inference_mode():
            output = adapter(
                TGVFAdapterInput(
                    target_hidden_states=hq,
                    pre_merge_visual_tokens=source.premerge_main,
                    deepstack_pre_merge_visual_tokens=source.premerge_deepstack,
                )
            )
        # Hq has been consumed into the immutable D payload.  Retaining the
        # per-token behavior buffer until trajectory end would multiply memory
        # across repeated calls without serving replay.
        self._tgvf_traces().pop(backend_request_id, None)
        return _focus_to_utility_wire(
            TGVFFocusMaterializationResult(
                hq=hq.detach().cpu(),
                observation=_adapter_payload_to_cpu(output),
            )
        )

    def tgvf_release_trajectory(
        self,
        trajectory_id: str,
        backend_request_ids: tuple[str, ...] = (),
    ) -> bool:
        source = self._tgvf_sources().pop(trajectory_id, None)
        traces = self._tgvf_traces()
        for request_id in backend_request_ids:
            traces.pop(request_id, None)
        return source is not None

    def _tgvf_model(self) -> Any:
        model = getattr(getattr(self, "model_runner", None), "model", None)
        if model is None or not hasattr(model, "language_model"):
            raise RuntimeError("vLLM Qwen3 model is not initialized")
        return model

    def _tgvf_sources(self) -> dict[str, SourceVisualTensorBundle]:
        value = getattr(self, "_tgvf_source_cache", None)
        if value is None:
            value = {}
            self._tgvf_source_cache = value
        return value

    def _tgvf_traces(self) -> dict[str, _BehaviorTraceBuffer]:
        value = getattr(self, "_tgvf_behavior_traces", None)
        if value is None:
            value = {}
            self._tgvf_behavior_traces = value
        return value

    def _tgvf_adapter(self) -> TGVFAdapter:
        value = getattr(self, "_tgvf_adapter_module", None)
        if value is not None:
            return value
        config_path = __import__("os").environ.get("TGVF_POLICY_RUN_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("TGVF_POLICY_RUN_CONFIG_PATH is required in vLLM")
        config = load_policy_e2e_smoke_run_config(config_path)
        export = load_rank_zero_adapter_owned_state_export(
            config.representation.artifact_path
        )
        run_identity = export.manifest.run_identity
        if state_digest(export.manifest) != config.representation.artifact.sha256:
            raise IdentityMismatchError(
                "vLLM Adapter export manifest identity differs"
            )
        if (
            run_identity.run_id != config.representation.expected_run_id
            or export.manifest.run_identity_sha256
            != config.representation.expected_run_identity_sha256
            or run_identity.identity_sha256
            != config.representation.expected_run_identity_sha256
        ):
            raise IdentityMismatchError("vLLM Adapter export run identity differs")
        if run_identity.model != config.model:
            raise RuntimeError("vLLM Adapter artifact model identity differs")
        if run_identity.provider != config.representation.conditioning:
            raise RuntimeError("vLLM Adapter conditioning identity differs")
        contract = run_identity.adapter_contract
        model = self._tgvf_model()
        visual = model.visual
        mergers = (visual.merger, *tuple(visual.deepstack_merger_list))
        identities = (
            contract.main_projection_identity,
            *contract.deepstack_projection_identities,
        )
        ports = tuple(
            FrozenProjectionPort(
                merger,
                identity=identity,
                input_dim=contract.d_v,
                output_dim=contract.d_lm,
                spatial_merge_size=contract.spatial_merge_size,
            )
            for merger, identity in zip(mergers, identities, strict=True)
        )
        adapter = TGVFAdapter(
            d_lm=contract.d_lm,
            d_v=contract.d_v,
            attn_dim=contract.attention_dim,
            main_projection=ports[0],
            deepstack_projections=DDeepStackProjectionPorts(
                branch_layers=contract.deepstack_branch_layers,
                projections=ports[1:],
            ),
            branch_layers=contract.deepstack_branch_layers,
            variant=TGVFAdapterVariant.FULL_D_DEEPSTACK,
        )
        owner = next(model.language_model.parameters())
        adapter.to(device=owner.device, dtype=owner.dtype)
        if export.state is None:
            raise RuntimeError("representation artifact omitted Adapter state")
        adapter.load_artifact_state_dict(export.state)
        adapter.requires_grad_(False)
        adapter.eval()
        contract.assert_matches(adapter)
        self._tgvf_adapter_module = adapter
        return adapter

    def _tgvf_install_trace_hook(self) -> None:
        runner = self.model_runner
        if getattr(runner, "_tgvf_trace_hook_installed", False):
            return
        original = runner.sample_tokens
        owner = self

        def sample_tokens(_runner: Any, grammar_output: object) -> object:
            owner._tgvf_capture_execute_state()
            return original(grammar_output)

        runner.sample_tokens = MethodType(sample_tokens, runner)
        runner._tgvf_trace_hook_installed = True

    def _tgvf_capture_execute_state(self) -> None:
        runner = self.model_runner
        state = getattr(runner, "execute_model_state", None)
        if state is None:
            return
        if isinstance(state, tuple):
            if len(state) < 5:
                raise RuntimeError("vLLM execute_model_state tuple is incomplete")
            scheduler_output = state[0]
            hidden_states = state[4]
        else:  # compatibility with a typed test/future public state object
            scheduler_output = state.scheduler_output
            hidden_states = state.hidden_states
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("vLLM execute state omitted hidden_states")
        req_ids = tuple(runner.input_batch.req_ids)
        offset = 0
        traces = self._tgvf_traces()
        for index, request_id in enumerate(req_ids):
            count = int(scheduler_output.num_scheduled_tokens[request_id])
            trace = traces.get(request_id)
            if trace is not None and count > 0:
                sequence_start = int(
                    runner.input_batch.num_computed_tokens_cpu[index]
                )
                sequence_end = sequence_start + count
                copy_start = max(sequence_start, trace.prompt_length)
                copy_end = min(
                    sequence_end, trace.prompt_length + trace.capacity
                )
                if copy_start < copy_end:
                    source_start = offset + copy_start - sequence_start
                    source_end = source_start + copy_end - copy_start
                    target_start = copy_start - trace.prompt_length
                    target_end = target_start + copy_end - copy_start
                    trace.hidden[target_start:target_end].copy_(
                        hidden_states[source_start:source_end]
                    )
                    trace.covered[target_start:target_end] = b"\x01" * (
                        target_end - target_start
                    )
            offset += count


def _capture_vllm_merger(destination: list[tuple[torch.Tensor, torch.Tensor]]):
    def hook(
        _module: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> None:
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if source is None:
            source = kwargs.get("hidden_states")
        if not isinstance(source, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError("vLLM merger must expose tensor input/output")
        if source.ndim == 3 and source.shape[1] == 1:
            source = source[:, 0]
        if source.ndim != 2 or output.ndim != 2:
            raise ValueError("vLLM merger boundaries must normalize to rank two")
        destination.append((source.detach().clone(), output.detach().clone()))

    return hook


def _validate_source_geometry(source: SourceVisualTensorBundle) -> None:
    pre = (source.premerge_main, *source.premerge_deepstack)
    merged = (source.merged_main, *source.merged_deepstack)
    if len(pre) != 4 or len(merged) != 4:
        raise ValueError("source requires main plus three DeepStack branches")
    expected_pre = int(torch.tensor(source.image_grid_thw).prod().item())
    expected_merged = expected_pre // (source.spatial_merge_size**2)
    if any(item.shape != pre[0].shape for item in pre):
        raise ValueError("source premerge branches differ")
    if any(item.shape != merged[0].shape for item in merged):
        raise ValueError("source merged branches differ")
    if pre[0].shape[0] != expected_pre or merged[0].shape[0] != expected_merged:
        raise ValueError("source token geometry differs from image grid")


def _source_to_cpu(source: SourceVisualTensorBundle) -> SourceVisualTensorBundle:
    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().contiguous().clone()

    return SourceVisualTensorBundle(
        image_sha256=source.image_sha256,
        premerge_main=cpu(source.premerge_main),
        premerge_deepstack=tuple(cpu(item) for item in source.premerge_deepstack),
        merged_main=cpu(source.merged_main),
        merged_deepstack=tuple(cpu(item) for item in source.merged_deepstack),
        image_grid_thw=source.image_grid_thw,
        spatial_merge_size=source.spatial_merge_size,
        decoded_rgb_sha256=source.decoded_rgb_sha256,
    )


def _adapter_payload_to_cpu(
    value: TGVFAdapterOutput,
) -> PrecomputedTGVFObservationPayload:
    def cpu(item: torch.Tensor) -> torch.Tensor:
        return item.detach().cpu().contiguous().clone()

    return PrecomputedTGVFObservationPayload(
        main_d=cpu(value.main_d),
        d_deepstack=DDeepStackPayload(
            branch_layers=value.d_deepstack.branch_layers,
            branches=tuple(cpu(item) for item in value.d_deepstack.branches),
            projection_identities=value.d_deepstack.projection_identities,
        ),
        metadata=value.metadata,
    )


def _single_collective_result(value: object, *, operation: str) -> object:
    if not isinstance(value, list) or len(value) != 1:
        raise RuntimeError(f"{operation} requires the accepted vLLM TP=1 topology")
    return value[0]


def _runtime_classes() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Create veRL-bound classes lazily so CPU contract imports remain light."""

    import ray
    from verl.utils.ray_utils import auto_await
    from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    class TGVFVLLMHttpServer(vLLMHttpServer):
        def _get_worker_extension_cls(self) -> str:
            return TGVF_VLLM_WORKER_EXTENSION_FQN

        def _require_step(self, expected_step: int) -> None:
            if type(expected_step) is not int or self.global_steps != expected_step:
                raise RuntimeError("TGVF RPC behavior policy step differs from vLLM")

        async def tgvf_shutdown(self) -> None:
            """Stop HTTP and EngineCore before Ray tears down the server actor."""

            errors: list[Exception] = []
            server_task = getattr(self, "_server_task", None)
            if isinstance(server_task, asyncio.Task) and not server_task.done():
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
                except Exception as error:  # pragma: no cover - backend failure
                    errors.append(error)

            engine = getattr(self, "engine", None)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                output_handler = getattr(engine, "output_handler", None)
                try:
                    shutdown()
                except Exception as error:  # pragma: no cover - backend failure
                    errors.append(error)
                finally:
                    self.engine = None
                # vLLM 0.12 schedules cancellation of AsyncLLM's output task,
                # but its synchronous shutdown() does not await that task.  A
                # subsequent immediate ray.kill() can otherwise destroy ZMQ
                # objects while the callback is still running, producing the
                # C++ "pure virtual method called" abort seen after successful
                # training.  Drain the scheduled cancellation before the RPC
                # acknowledges shutdown and sever AsyncLLM's finalizer-owned
                # references so __del__ cannot run the same teardown again.
                if isinstance(output_handler, asyncio.Task):
                    try:
                        await output_handler
                    except asyncio.CancelledError:
                        pass
                    except Exception as error:  # pragma: no cover - backend failure
                        errors.append(error)
                if hasattr(engine, "output_handler"):
                    engine.output_handler = None
                if hasattr(engine, "engine_core"):
                    engine.engine_core = None
                del engine
                gc.collect()
                await asyncio.sleep(0)

            for name in ("_master_sock", "_dp_rpc_sock", "_dp_master_sock"):
                sock = getattr(self, name, None)
                close = getattr(sock, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as error:  # pragma: no cover - OS failure
                        errors.append(error)
                    finally:
                        setattr(self, name, None)

            if errors:
                raise ExceptionGroup("TGVF vLLM server shutdown failed", errors)

        async def tgvf_materialize_source(
            self, *, expected_step: int, **kwargs: object
        ) -> object:
            self._require_step(expected_step)
            pixel_values = kwargs.pop("pixel_values")
            image_grid_thw = kwargs.pop("image_grid_thw")
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError("source pixel_values must be a tensor")
            if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.shape != (
                1,
                3,
            ):
                raise TypeError("source image_grid_thw must be a [1,3] tensor")
            kwargs["pixel_values_wire"] = _tensor_to_utility_wire(pixel_values)
            kwargs["image_grid_thw"] = tuple(
                int(value) for value in image_grid_thw[0].tolist()
            )
            result = await self.engine.collective_rpc(
                method="tgvf_materialize_source", kwargs=dict(kwargs)
            )
            wire = _single_collective_result(result, operation="source materialization")
            return _source_from_utility_wire(
                _wire_mapping(wire, owner="source materialization")
            )

        async def tgvf_register_behavior_trace(
            self, *, expected_step: int, **kwargs: object
        ) -> object:
            self._require_step(expected_step)
            result = await self.engine.collective_rpc(
                method="tgvf_register_behavior_trace", kwargs=dict(kwargs)
            )
            return _single_collective_result(result, operation="trace registration")

        async def tgvf_materialize_focus(
            self, *, expected_step: int, **kwargs: object
        ) -> object:
            self._require_step(expected_step)
            result = await self.engine.collective_rpc(
                method="tgvf_materialize_focus", kwargs=dict(kwargs)
            )
            wire = _single_collective_result(result, operation="focus materialization")
            return _focus_from_utility_wire(
                _wire_mapping(wire, owner="focus materialization")
            )

        async def tgvf_release_trajectory(
            self,
            *,
            trajectory_id: str,
            backend_request_ids: tuple[str, ...],
        ) -> object:
            result = await self.engine.collective_rpc(
                method="tgvf_release_trajectory",
                args=(trajectory_id, backend_request_ids),
            )
            return _single_collective_result(result, operation="trajectory release")

    class TGVFVLLMReplica(vLLMReplica):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(TGVFVLLMHttpServer)

    class TGVFLLMServerClient(LLMServerClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._tgvf_routes: dict[str, tuple[str, object]] = {}
            self._tgvf_turns: dict[str, _TurnRoute] = {}
            self._tgvf_backend_ids: dict[str, list[str]] = {}

        async def _route(self, request_id: str) -> tuple[str, object]:
            route = self._tgvf_routes.get(request_id)
            if route is None:
                route = await self._acquire_server(request_id)
                self._tgvf_routes[request_id] = route
            return route

        async def materialize_source(
            self,
            *,
            request_id: str,
            expected_step: int,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            image_sha256: str,
        ) -> SourceVisualTensorBundle:
            _server_id, server = await self._route(request_id)
            return await server.tgvf_materialize_source.remote(
                expected_step=expected_step,
                trajectory_id=request_id,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_sha256=image_sha256,
            )

        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, Any],
            image_data: list[Any] | None = None,
            video_data: list[Any] | None = None,
            audio_data: list[Any] | None = None,
            mm_processor_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> object:
            del video_data, audio_data
            _server_id, server = await self._route(request_id)
            backend_request_id = f"tgvf-{uuid4().hex}"
            expected_step = int(kwargs.pop("tgvf_expected_step"))
            maximum = sampling_params.get("max_tokens")
            if type(maximum) is not int or maximum <= 0:
                raise ValueError("TGVF vLLM generation requires positive max_tokens")
            await server.tgvf_register_behavior_trace.remote(
                expected_step=expected_step,
                request_id=backend_request_id,
                prompt_length=len(prompt_ids),
                maximum_output_tokens=maximum,
            )
            self._tgvf_backend_ids.setdefault(request_id, []).append(
                backend_request_id
            )
            priority = kwargs.pop("priority", 0)
            output = await server.generate.remote(
                request_id=backend_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                **kwargs,
            )
            step = output.extra_fields.get("global_steps")
            if step != expected_step:
                raise RuntimeError("vLLM generation returned another policy step")
            output.extra_fields.setdefault("min_global_steps", step)
            output.extra_fields.setdefault("max_global_steps", step)
            self._tgvf_turns[request_id] = _TurnRoute(
                backend_request_id=backend_request_id,
                prompt_ids=tuple(prompt_ids),
                output_ids=tuple(output.token_ids),
                global_step=step,
            )
            return output

        async def materialize_focus(
            self,
            *,
            request_id: str,
            expected_step: int,
            sampled_output_ids: tuple[int, ...],
            target_start: int,
            target_end: int,
            expected_target_token_ids: tuple[int, ...],
            provider: str,
        ) -> tuple[torch.Tensor, PrecomputedTGVFObservationPayload]:
            turn = self._tgvf_turns.get(request_id)
            if turn is None or turn.output_ids != tuple(sampled_output_ids):
                raise RuntimeError("TGVF tool call differs from the last vLLM turn")
            if turn.global_step != expected_step:
                raise RuntimeError("TGVF tool call policy step changed")
            if turn.output_ids[target_start:target_end] != tuple(
                expected_target_token_ids
            ):
                raise RuntimeError("TGVF target tokens differ from sampled output")
            _server_id, server = await self._route(request_id)
            result = await server.tgvf_materialize_focus.remote(
                expected_step=expected_step,
                trajectory_id=request_id,
                backend_request_id=turn.backend_request_id,
                target_start=target_start,
                target_end=target_end,
                expected_target_token_ids=tuple(expected_target_token_ids),
                provider=provider,
            )
            if not isinstance(result, TGVFFocusMaterializationResult):
                raise TypeError("vLLM focus RPC returned an untyped result")
            return result.hq, result.observation

        async def release_trajectory(self, request_id: str) -> None:
            route = self._tgvf_routes.pop(request_id, None)
            self._tgvf_turns.pop(request_id, None)
            backend_request_ids = tuple(
                self._tgvf_backend_ids.pop(request_id, ())
            )
            if route is None:
                return
            server_id, server = route
            try:
                await server.tgvf_release_trajectory.remote(
                    trajectory_id=request_id,
                    backend_request_ids=backend_request_ids,
                )
            finally:
                self._release_server(server_id)

    class TGVFLLMServerManager(LLMServerManager):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.rollout_replica_class = TGVFVLLMReplica
            super().__init__(*args, **kwargs)

        def get_client(self, client_cls: type[Any] = TGVFLLMServerClient, **kwargs: Any):
            return super().get_client(client_cls=client_cls, **kwargs)

        @auto_await
        async def shutdown(self) -> None:
            """Gracefully stop EngineCore, then remove the Ray server actors."""

            servers = [
                server
                for replica in getattr(self, "rollout_replicas", ())
                for server in getattr(replica, "servers", ())
            ]
            results = await asyncio.gather(
                *(server.tgvf_shutdown.remote() for server in servers),
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, Exception)]
            for server in servers:
                ray.kill(server, no_restart=True)
            load_balancer = getattr(self, "global_load_balancer", None)
            if load_balancer is not None:
                ray.kill(load_balancer, no_restart=True)
            if errors:
                raise ExceptionGroup("TGVF vLLM manager shutdown failed", errors)

    return (
        TGVFLLMServerManager,
        TGVFLLMServerClient,
        TGVFVLLMReplica,
        TGVFVLLMHttpServer,
    )


def tgvf_llm_server_manager_class() -> type[Any]:
    return _runtime_classes()[0]


__all__ = [
    "TGVF_TWO_MODEL_RUNTIME_SCHEMA",
    "TGVFFocusMaterializationResult",
    "TGVF_VLLM_WORKER_EXTENSION_FQN",
    "TGVFVLLMWorkerExtension",
    "tgvf_llm_server_manager_class",
]
