"""Standalone CoreDev inference for completed visual-tool policy artifacts.

The training runtime already owns the native multi-turn protocol and the
colocated vLLM visual-tool implementation.  This module supplies only the
post-training boundary: an immutable LoRA snapshot, one vLLM replica, and the
official CoreDev prompt rows.  It deliberately performs no reward or update.
"""

from __future__ import annotations

import asyncio
import ast
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from uuid import uuid4
import re

import torch

from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.environment import (
    CropExecutionLedger,
    FocusExecutionLedger,
    FrameworkNeutralAgentLoop,
    ImageZoomInToolRuntime,
    Qwen3NativeToolLayoutBuilder,
    QwenNativeToolObservationAppender,
    record_trajectory_source_visual,
)
from tgvf_rl.environment.qwen3_crop_materializer import preprocess_qwen3_rgb
from tgvf_rl.framework.verl.native_agent_loop import VerlAsyncServerPolicyTurnClient
from tgvf_rl.framework.verl.policy_live_runtime import (
    _BRANCH_LAYERS,
    _RemoteCropVisualMaterializer,
    _RemoteTGVFFocusToolRuntime,
    _VisualTokenCountResolver,
    _artifact_identity,
    _initial_vllm_inputs,
    _load_bound_rgb,
    _source_visual_positions,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_VLLM_WORKER_EXTENSION_FQN,
    TGVFFocusMaterializationResult,
    _focus_from_utility_wire,
    _source_from_utility_wire,
    _tensor_to_utility_wire,
)
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    LiveVLLMTurnContextRegistry,
    Qwen3VLLMObservationPayloadResolver,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMTerminationOutcome,
    VLLMTurnTerminationContract,
)
from tgvf_rl.framework.vllm.registration import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.policy.run_config import (
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.protocol import (
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    StrictToolCallParser,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
)
from tgvf_rl.protocol.state_machine import CapErrorBehavior
from tgvf_rl.trajectories.behavior import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolCallRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    trajectory_checksum,
)


POLICY_COREDEV_SCHEMA = "tgvf-policy-coredev-evaluation-v1"


@dataclass(frozen=True, slots=True)
class PolicyCoreDevConfig:
    evaluation_id: str
    policy_config_path: Path
    lora_pointer_path: Path
    output_root: Path
    gpu_ids: tuple[int, ...]
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    schema_version: str = POLICY_COREDEV_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_config_path", Path(self.policy_config_path))
        object.__setattr__(self, "lora_pointer_path", Path(self.lora_pointer_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))
        if self.schema_version != POLICY_COREDEV_SCHEMA:
            raise ValueError("policy CoreDev config schema differs")
        if not self.evaluation_id:
            raise ValueError("evaluation_id must be non-empty")
        if self.gpu_ids != (0, 1, 2, 3):
            raise ValueError("formal policy CoreDev evaluation requires GPUs 0-3")
        if self.max_model_len != 16384:
            raise ValueError("policy CoreDev max_model_len must match training")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0,1]")


def load_policy_coredev_config(path: str | Path) -> PolicyCoreDevConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "evaluation_id",
        "policy_config_path",
        "lora_pointer_path",
        "output_root",
        "gpu_ids",
        "max_model_len",
        "gpu_memory_utilization",
    }
    if set(payload) != expected:
        raise ValueError("policy CoreDev config fields differ")
    return PolicyCoreDevConfig(**payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_vllm_lora_adapter(config: PolicyCoreDevConfig) -> Path:
    """Expose an exact runtime snapshot through vLLM's PEFT directory ABI."""

    pointer = json.loads(config.lora_pointer_path.read_text(encoding="utf-8"))
    weights_sha256 = pointer["weights_sha256"]
    manifest_path = config.lora_pointer_path.parent / pointer["manifest_file"]
    if _sha256_file(manifest_path) != pointer["manifest_file_sha256"]:
        raise ValueError("LoRA manifest SHA256 differs from latest pointer")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = config.lora_pointer_path.parent / manifest["tensor_file"]
    if snapshot.name != f"{weights_sha256}.safetensors":
        raise ValueError("LoRA tensor mapping identity differs from latest pointer")
    if _sha256_file(snapshot) != manifest["tensor_file_sha256"]:
        raise ValueError("LoRA snapshot file SHA256 differs from manifest")
    adapter_root = config.output_root / "runtime" / "lora-adapter"
    adapter_root.mkdir(parents=True, exist_ok=True)
    model_file = adapter_root / "adapter_model.safetensors"
    if not model_file.exists():
        os.link(snapshot, model_file)
    if _sha256_file(model_file) != manifest["tensor_file_sha256"]:
        raise RuntimeError("materialized vLLM LoRA differs from snapshot")
    adapter_config = {
        "base_model_name_or_path": str(
            load_policy_e2e_smoke_run_config(config.policy_config_path).model.revision_or_path
        ),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 64,
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_text = json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    adapter_config_path = adapter_root / "adapter_config.json"
    if adapter_config_path.exists():
        if adapter_config_path.read_text(encoding="utf-8") != config_text:
            raise RuntimeError("vLLM LoRA adapter config identity collision")
    else:
        adapter_config_path.write_text(config_text, encoding="utf-8")
    identity = {
        "schema_version": "tgvf-policy-vllm-lora-adapter-v1",
        "evaluation_id": config.evaluation_id,
        "optimizer_step": pointer["optimizer_step"],
        "policy_run_id": pointer["run_id"],
        "policy_run_identity_sha256": pointer["run_identity_sha256"],
        "weights_sha256": weights_sha256,
        "tensor_file_sha256": manifest["tensor_file_sha256"],
        "snapshot": str(snapshot),
    }
    identity_path = adapter_root / "identity.json"
    identity_text = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    if identity_path.exists() and identity_path.read_text(encoding="utf-8") != identity_text:
        raise RuntimeError("vLLM LoRA identity collision")
    identity_path.write_text(identity_text, encoding="utf-8")
    return adapter_root


def policy_version_from_pointer(config: PolicyCoreDevConfig) -> PolicyVersion:
    pointer = json.loads(config.lora_pointer_path.read_text(encoding="utf-8"))
    return PolicyVersion(
        run_id=pointer["run_id"],
        optimizer_step=pointer["optimizer_step"],
        weights_sha256=pointer["weights_sha256"],
    )


@dataclass(frozen=True, slots=True)
class CoreDevTask:
    ordinal: int
    dataset: str
    row_number: int
    index: str
    question: str
    image_paths: tuple[str, ...]

    @property
    def single_image(self) -> bool:
        return len(self.image_paths) == 1


def write_official_coredev_tasks(output_path: str | Path) -> dict[str, int]:
    """Materialize pinned TSV contents with their official dataset prompt text."""

    repository_root = Path(__file__).resolve().parents[3]
    pinned = json.loads(
        (repository_root / "configs/evaluation/coredev_2511_vlmevalkit_v1.json").read_text()
    )
    artifact_root = Path(pinned["artifact_root"])
    rows: list[dict[str, object]] = []
    ordinal = 0
    counts = {"total": 0, "single_image": 0, "multi_image": 0}
    for slice_spec in pinned["slices"]:
        dataset_name = slice_spec["dataset"]
        tsv = artifact_root / f"{dataset_name}.tsv"
        if _sha256_file(tsv) != slice_spec["tsv_sha256"]:
            raise ValueError(f"pinned CoreDev TSV changed: {dataset_name}")
        with tsv.open(encoding="utf-8", newline="") as handle:
            source_rows = tuple(csv.DictReader(handle, delimiter="\t"))
        if len(source_rows) != slice_spec["sample_count"]:
            raise ValueError(f"pinned CoreDev row count changed: {dataset_name}")
        for row_number, source in enumerate(source_rows):
            images = _tsv_image_paths(source["image_path"])
            text = _official_prompt_text(dataset_name, source)
            index = source["index"]
            if not text.strip() or not images:
                raise ValueError(f"official prompt is incomplete: {dataset_name}/{index}")
            rows.append(
                {
                    "ordinal": ordinal,
                    "dataset": dataset_name,
                    "row_number": row_number,
                    "index": index,
                    "question": text,
                    "image_paths": list(images),
                }
            )
            ordinal += 1
            counts["single_image" if len(images) == 1 else "multi_image"] += 1
    counts["total"] = ordinal
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return counts


def _tsv_image_paths(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value
    paths = tuple(str(item) for item in parsed) if isinstance(parsed, list) else (str(parsed),)
    if not paths or any(not Path(path).is_file() for path in paths):
        raise ValueError("CoreDev TSV contains a missing image path")
    return paths


def _option_lines(source: Mapping[str, str]) -> str:
    rows = []
    for letter in "ABCDEFGHIJ":
        value = source.get(letter, "")
        if not value:
            break
        rows.append(f"{letter}. {value}")
    if len(rows) < 2:
        raise ValueError("CoreDev MCQ row has fewer than two contiguous choices")
    return "\n".join(rows)


def _official_prompt_text(dataset: str, source: Mapping[str, str]) -> str:
    question = source["question"]
    if dataset in {"VStarBench", "HRBench4K", "BLINK"}:
        return (
            f"Question: {question}\nOptions:\n{_option_lines(source)}\n"
            "Please select the correct answer from the options above. \n"
        )
    if dataset == "MMMU_Pro_10c":
        # VLMEvalKit interleaves image items at these markers.  The accepted
        # visual-tool prompt owns its one image placeholder separately.
        question = re.sub(r"<image\s+\d+>", "", question)
        return (
            f"Question: {question}\nOptions:\n{_option_lines(source)}\n"
            "Answer directly with the option letter from the given choices. "
        )
    if dataset in {"OCRBench_v2", "MathVista_MINI", "MathVerse_MINI"}:
        return question
    raise ValueError(f"unsupported CoreDev dataset: {dataset}")


def load_coredev_tasks(path: str | Path) -> tuple[CoreDevTask, ...]:
    tasks = tuple(
        CoreDevTask(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(tasks) != 2511 or tuple(item.ordinal for item in tasks) != tuple(range(2511)):
        raise ValueError("CoreDev task materialization differs from 2511-row order")
    return tasks


def _single_collective(value: object, *, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 1:
        raise RuntimeError(f"{operation} requires one vLLM worker result")
    result = value[0]
    if not isinstance(result, Mapping):
        raise TypeError(f"{operation} returned a non-mapping utility result")
    return result


@dataclass(frozen=True, slots=True)
class _TurnRoute:
    backend_request_id: str
    output_ids: tuple[int, ...]
    optimizer_step: int


class StandaloneTGVFVLLMManager:
    """Small AsyncLLM adapter matching the already-audited training client ABI."""

    def __init__(self, engine: object, lora_request: object, *, capture_hidden: bool) -> None:
        self.engine = engine
        self.lora_request = lora_request
        self.capture_hidden = capture_hidden
        self.turns: dict[str, _TurnRoute] = {}
        self.backend_ids: dict[str, list[str]] = {}

    async def materialize_source(
        self,
        *,
        request_id: str,
        expected_step: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_sha256: str,
    ) -> object:
        del expected_step
        result = await self.engine.collective_rpc(
            "tgvf_materialize_source",
            kwargs={
                "trajectory_id": request_id,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "image_sha256": image_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="source materialization")
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
        from vllm import SamplingParams

        step = int(kwargs.pop("tgvf_expected_step"))
        if kwargs:
            raise TypeError(f"unsupported standalone vLLM arguments: {sorted(kwargs)}")
        backend_id = f"eval-{uuid4().hex}"
        maximum = sampling_params.get("max_tokens")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("generation requires positive max_tokens")
        if self.capture_hidden:
            await self.engine.collective_rpc(
                "tgvf_register_behavior_trace",
                kwargs={
                    "request_id": backend_id,
                    "prompt_length": len(prompt_ids),
                    "maximum_output_tokens": maximum,
                },
            )
            self.backend_ids.setdefault(request_id, []).append(backend_id)
        prompt = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": image_data},
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        parameters = dict(sampling_params)
        if parameters.get("logprobs") is True:
            parameters["logprobs"] = 0
        final = None
        async for output in self.engine.generate(
            prompt,
            SamplingParams(**parameters),
            backend_id,
            lora_request=self.lora_request,
        ):
            final = output
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("standalone vLLM generation did not finish exactly once")
        completion = final.outputs[0]
        token_ids = tuple(int(value) for value in completion.token_ids)
        logprobs = []
        for token_id, position in zip(token_ids, completion.logprobs, strict=True):
            entry = position.get(token_id)
            if entry is None:
                raise RuntimeError("sampled token is absent from vLLM logprobs")
            logprobs.append(float(entry.logprob))
        self.turns[request_id] = _TurnRoute(backend_id, token_ids, step)
        return SimpleNamespace(
            token_ids=list(token_ids),
            log_probs=logprobs,
            stop_reason="completed",
            extra_fields={
                "global_steps": step,
                "min_global_steps": step,
                "max_global_steps": step,
                "logprobs_mode": "processed_logprobs",
            },
        )

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
    ) -> tuple[torch.Tensor, object]:
        turn = self._validated_turn(request_id, expected_step, sampled_output_ids)
        if turn.output_ids[target_start:target_end] != expected_target_token_ids:
            raise RuntimeError("focus target differs from sampled output")
        result = await self.engine.collective_rpc(
            "tgvf_materialize_focus",
            kwargs={
                "trajectory_id": request_id,
                "backend_request_id": turn.backend_request_id,
                "target_start": target_start,
                "target_end": target_end,
                "expected_target_token_ids": expected_target_token_ids,
                "provider": provider,
            },
        )
        typed = _focus_from_utility_wire(
            _single_collective(result, operation="focus materialization")
        )
        if not isinstance(typed, TGVFFocusMaterializationResult):
            raise TypeError("focus RPC returned an invalid result")
        return typed.hq, typed.observation

    async def materialize_crop(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        call_index: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        crop_sha256: str,
    ) -> object:
        self._validated_turn(request_id, expected_step, sampled_output_ids)
        result = await self.engine.collective_rpc(
            "tgvf_materialize_crop",
            kwargs={
                "trajectory_id": request_id,
                "call_index": call_index,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "crop_sha256": crop_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="crop materialization")
        )

    def _validated_turn(
        self, request_id: str, expected_step: int, output_ids: tuple[int, ...]
    ) -> _TurnRoute:
        turn = self.turns.get(request_id)
        if turn is None or turn.output_ids != tuple(output_ids):
            raise RuntimeError("tool call differs from the last vLLM turn")
        if turn.optimizer_step != expected_step:
            raise RuntimeError("tool call policy step changed")
        return turn

    async def release_trajectory(self, request_id: str) -> None:
        backend_ids = tuple(self.backend_ids.pop(request_id, ()))
        self.turns.pop(request_id, None)
        await self.engine.collective_rpc(
            "tgvf_release_trajectory", args=(request_id, backend_ids)
        )


async def build_standalone_manager(
    config: PolicyCoreDevConfig,
) -> tuple[StandaloneTGVFVLLMManager, object, PolicyE2ESmokeRunConfig]:
    """Construct one single-GPU post-training vLLM replica."""

    from vllm import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    run = load_policy_e2e_smoke_run_config(config.policy_config_path)
    adapter_root = materialize_vllm_lora_adapter(config)
    engine_args = AsyncEngineArgs(
        model=run.model.revision_or_path,
        dtype="bfloat16",
        trust_remote_code=True,
        distributed_executor_backend="mp",
        worker_extension_cls=TGVF_VLLM_WORKER_EXTENSION_FQN,
        max_model_len=config.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=config.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        enforce_eager=False,
        seed=run.rollout_rng.master_seed,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        enable_mm_embeds=True,
        mm_processor_cache_gb=0,
        mm_encoder_attn_backend=TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
        limit_mm_per_prompt={
            "image": 1 + run.protocol.maximum_tool_calls,
            "video": 0,
        },
        hf_overrides={"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    lora = LoRARequest("policy-step80", 1, str(adapter_root))
    manager = StandaloneTGVFVLLMManager(
        engine,
        lora,
        capture_hidden=(
            run.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY
        ),
    )
    return manager, engine, run


def _decoding_contract() -> VLLMOutputDecodingContract:
    return VLLMOutputDecodingContract(
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind="final_only",
    )


def _termination_contract(run: PolicyE2ESmokeRunConfig) -> VLLMTurnTerminationContract:
    sampling = run.policy.sampling
    return VLLMTurnTerminationContract(
        required_request_stop_strings=tuple(sampling.stop_strings or ()),
        required_request_stop_token_ids=tuple(sampling.stop_token_ids or ()),
        include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
        tool_call_terminal_suffixes=("",),
        tool_call_outcomes=(VLLMTerminationOutcome("stop", "</tool_call>"),),
        final_turn_outcomes=tuple(
            VLLMTerminationOutcome("stop", token_id)
            for token_id in tuple(sampling.stop_token_ids or ())
        )
        + (VLLMTerminationOutcome("length", None),),
    )


class PolicyCoreDevEvaluator:
    def __init__(
        self,
        *,
        config: PolicyCoreDevConfig,
        run: PolicyE2ESmokeRunConfig,
        manager: StandaloneTGVFVLLMManager,
        processor: object,
    ) -> None:
        self.config = config
        self.run = run
        self.manager = manager
        self.processor = processor
        self.policy_version = policy_version_from_pointer(config)
        self.store = ObservationStore()
        self.behavior_store = BehaviorTraceStore()
        self.focus_ledger = FocusExecutionLedger()
        self.crop_ledger = CropExecutionLedger()
        self.layout_builder = Qwen3NativeToolLayoutBuilder.from_processor_config(
            processor=processor,
            model_identity=run.model,
            observation_store=self.store,
        )
        schemas = tuple(build_native_tool_schemas(run.protocol.enabled_tool_names))
        self.renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=run.model.tokenizer_length,
            tool_names=run.protocol.enabled_tool_names,
            tool_schemas=schemas,
        )
        conditioning = run.representation.conditioning
        self.contextual_identity = (
            _artifact_identity(
                "policy-evaluation",
                "qwen3-contextual-behavior-forward",
                POLICY_COREDEV_SCHEMA,
                {
                    "evaluation_id": config.evaluation_id,
                    "policy": self.policy_version.weights_sha256,
                    "provider": conditioning.provider.value,
                    "hidden_layer": conditioning.hidden_layer,
                },
            )
            if conditioning.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
            else None
        )
        export = load_rank_zero_adapter_owned_state_export(
            run.representation.artifact_path
        )
        adapter_contract = export.manifest.run_identity.adapter_contract
        self.branch_identities = tuple(
            _artifact_identity(
                "qwen3-vl",
                f"deepstack-merger-{layer}",
                "frozen-base-model-v1",
                {
                    "model": run.model.revision_or_path,
                    "projection_identity": projection_identity,
                    "layer": layer,
                },
            )
            for layer, projection_identity in zip(
                _BRANCH_LAYERS,
                adapter_contract.deepstack_projection_identities,
                strict=True,
            )
        )

    def render_initial_prompt(self, task: CoreDevTask) -> tuple[int, ...]:
        if not task.single_image:
            raise ValueError("current visual-tool protocol has no multi-image selector")
        messages = build_visual_tool_prompt_messages(
            task.question, tool_profile=self.run.protocol.tool_profile
        )
        rendered = self.renderer.render(messages, add_generation_prompt=True)
        self.renderer.assert_generation_prefill(rendered, self.renderer.tokenizer)
        from tgvf_rl.framework.verl.smoke_dataset import (
            _materialize_source_image_prompt_token_ids,
        )

        return _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=rendered.token_ids,
            prompt_text=rendered.text,
            image_path=Path(task.image_paths[0]),
            image_max_pixels=self.run.policy.image_max_pixels,
        )

    async def evaluate(self, task: CoreDevTask) -> TrajectoryRecord:
        prompt_ids = self.render_initial_prompt(task)
        identity = TrajectoryIdentity(
            self.config.evaluation_id,
            f"{task.dataset}:{task.index}",
            0,
            f"coredev:{task.ordinal}",
        )
        trajectory_id = identity.canonical_id
        source_rgb = _load_bound_rgb(Path(task.image_paths[0]))
        pixel_values, image_grid_thw = preprocess_qwen3_rgb(
            processor=self.processor,
            rgb=source_rgb,
            image_max_pixels=self.run.policy.image_max_pixels,
        )
        source = await self.manager.materialize_source(
            request_id=trajectory_id,
            expected_step=self.policy_version.optimizer_step,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sha256=tensor_checksum(source_rgb),
        )
        positions = _source_visual_positions(
            prompt_ids,
            image_token_id=self.layout_builder.image_pad_id,
            expected_count=int(source.merged_main.shape[-2]),
        )
        trajectory_source = record_trajectory_source_visual(
            trajectory_id=trajectory_id,
            source_visual=source,
            source_positions=positions,
            deepstack_branch_layers=_BRANCH_LAYERS,
            deepstack_injection_positions=tuple(positions for _ in _BRANCH_LAYERS),
            observation_store=self.store,
            source_rgb=source_rgb,
        )
        registry = LiveVLLMTurnContextRegistry(
            observation_resolver=Qwen3VLLMObservationPayloadResolver(
                store=self.store, include_multi_modal_uuid=False
            )
        )
        registry.register_initial_prompt(
            prompt_ids,
            _initial_vllm_inputs(
                initial_prompt_token_ids=prompt_ids,
                image_token_id=self.layout_builder.image_pad_id,
                source=source,
                image_max_pixels=self.run.policy.image_max_pixels,
            ),
        )
        appender = QwenNativeToolObservationAppender(
            tokenizer=self.layout_builder.tokenizer,
            registrar=registry,
            visual_token_count_resolver=_VisualTokenCountResolver(self.store),
        )
        if self.run.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY:
            tool_runtime = _RemoteTGVFFocusToolRuntime(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                config=self.run,
                source_visual=source,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                execution_ledger=self.focus_ledger,
                contextual_forward_identity=self.contextual_identity,
                branch_merger_identities=self.branch_identities,
            )
        elif self.run.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            processor_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-shared-vllm-crop-processor",
                POLICY_COREDEV_SCHEMA,
                {"model": self.run.model.revision_or_path, "max_pixels": self.run.policy.image_max_pixels},
            )
            layout_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-native-crop-layout",
                POLICY_COREDEV_SCHEMA,
                {"model": self.run.model.revision_or_path},
            )
            materializer = _RemoteCropVisualMaterializer(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                processor=self.processor,
                model_identity=self.run.model,
                image_max_pixels=self.run.policy.image_max_pixels,
                trajectory_id=trajectory_id,
                behavior_policy=self.policy_version,
            )
            tool_runtime = ImageZoomInToolRuntime(
                model=self.run.model,
                materializer=materializer,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                crop_processor_identity=processor_identity,
                crop_layout_identity=layout_identity,
                execution_ledger=self.crop_ledger,
            )
        else:
            raise RuntimeError("policy CoreDev supports the two trained atomic arms")

        decoding = _decoding_contract()
        client = VerlAsyncServerPolicyTurnClient(
            server_manager=self.manager,
            event_loop=asyncio.get_running_loop(),
            tokenizer=self.renderer.tokenizer,
            prompt_inputs=registry,
            token_byte_span_decoder=FastTokenizerTokenByteSpanDecoder(
                self.renderer.tokenizer
            ),
            sticky_request_id=trajectory_id,
            max_model_len=self.config.max_model_len,
            server_timeout_seconds=2400.0,
            logprobs_mode="processed_logprobs",
        )
        sampler = VLLMPolicySampler(
            client=client,
            behavior_policy=self.policy_version,
            rng=ContentAddressedVLLMTurnRNG(
                master_seed=self.run.rollout_rng.master_seed,
                stream_identity=trajectory_id,
            ),
            request_context=registry,
            decoding=decoding,
            termination=_termination_contract(self.run),
        )
        loop = FrameworkNeutralAgentLoop(
            sampler=sampler,
            tool_runtime=tool_runtime,
            appender=appender,
            parser=StrictToolCallParser(
                enabled_tool_names=self.run.protocol.enabled_tool_names
            ),
            behavior_recorder=VLLMBehaviorRecorder(self.behavior_store),
            max_tool_calls=self.run.protocol.maximum_tool_calls,
            enabled_tool_names=self.run.protocol.enabled_tool_names,
            cap_error_behavior=CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
        )
        from tgvf_rl.environment.agent_loop import RolloutRequest

        request = RolloutRequest(
            "trajectory-v1",
            identity,
            self.run.model,
            self.policy_version,
            trajectory_source,
            prompt_ids,
            {},
            self.run.policy.sampling,
        )
        try:
            return await asyncio.to_thread(loop.run, request)
        finally:
            await self.manager.release_trajectory(trajectory_id)


def trajectory_audit_payload(task: CoreDevTask, trajectory: TrajectoryRecord) -> dict[str, object]:
    def call_payload(call: object) -> dict[str, object]:
        common: dict[str, object]
        if isinstance(call, ToolCallRecord):
            common = {"target": call.target}
        elif isinstance(call, CropToolCallRecord):
            common = {"bbox_2d": list(call.bbox_2d), "label": call.label}
        elif isinstance(call, CropTGVFToolCallRecord):
            common = {"bbox_2d": list(call.bbox_2d), "target": call.target}
        else:
            raise TypeError("unsupported tool call record")
        return {
            "call_index": call.call_index,
            "assistant_turn_index": call.assistant_turn_index,
            "function_name": call.function_name,
            "raw_call_text": call.raw_call_text,
            **common,
        }

    return {
        "schema_version": "tgvf-policy-coredev-trajectory-audit-v1",
        "ordinal": task.ordinal,
        "dataset": task.dataset,
        "row_number": task.row_number,
        "index": task.index,
        "question": task.question,
        "image_paths": list(task.image_paths),
        "trajectory_id": trajectory.identity.canonical_id,
        "trajectory_sha256": trajectory_checksum(trajectory),
        "policy_run_id": trajectory.behavior_policy.run_id,
        "optimizer_step": trajectory.behavior_policy.optimizer_step,
        "policy_weights_sha256": trajectory.behavior_policy.weights_sha256,
        "stop": trajectory.stop.value,
        "final_answer": trajectory.final_answer,
        "assistant_turns": [
            {
                "turn_index": turn.turn_index,
                "raw_text": turn.raw_text,
                "sampled_token_count": len(turn.tokens.token_ids),
                "is_tool_call": turn.is_tool_call,
                "stop_reason": turn.stop_reason,
            }
            for turn in trajectory.assistant_turns
        ],
        "tool_calls": [call_payload(call) for call in trajectory.tool_calls],
        "tool_errors": [
            {
                "attempt_index": error.attempt_index,
                "assistant_turn_index": error.assistant_turn_index,
                "function_name": error.function_name,
                "code": error.code,
                "payload_json": error.payload_json,
                "recoverable": error.recoverable,
            }
            for error in trajectory.tool_errors
        ],
        "successful_observation_count": len(trajectory.observations),
    }


__all__ = [
    "CoreDevTask",
    "POLICY_COREDEV_SCHEMA",
    "PolicyCoreDevConfig",
    "PolicyCoreDevEvaluator",
    "StandaloneTGVFVLLMManager",
    "build_standalone_manager",
    "load_coredev_tasks",
    "load_policy_coredev_config",
    "materialize_vllm_lora_adapter",
    "policy_version_from_pointer",
    "trajectory_audit_payload",
    "write_official_coredev_tasks",
]
