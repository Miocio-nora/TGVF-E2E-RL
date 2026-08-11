"""Isolated forced-TGVF counterfactual generation for Stage3 utility labels.

This module deliberately does not enter the Policy training runtime.  It reads
the immutable TGVF-80 schedule, materializes one RP66 focused observation from
the source image and a declared question-derived target, injects the original
image plus RP66 main-D/DeepStack into the frozen Qwen3-VL-Instruct reader, and
samples eight answer attempts.  The resulting attempt JSONL is the input
contract accepted by :mod:`tgvf_rl.data.tgvf_tool_utility`.

Each GPU owns a deterministic modulo shard and an independently locked,
atomically rewritten ledger.  Attempt seeds depend only on the run identity,
candidate identity, and attempt index, so resuming or changing the physical
GPU assignment cannot change an already declared attempt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterator, Literal

import torch

from tgvf_rl.data.policy_selection import (
    SelectionSource,
    policy_selection_semantic_judge_task_kind,
)
from tgvf_rl.data.policy_selection_runtime import (
    T1_INSTRUCT_ANSWER_PARSER,
    VerificationOutcome,
    parse_t1_answer,
    verify_t1_answer,
)
from tgvf_rl.data.tgvf_tool_utility import (
    SCHEDULE_FILE,
    SCHEDULE_MANIFEST_FILE,
    TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
    TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA,
    TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA,
)
from tgvf_rl.environment.native_appender import (
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX,
    QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX,
)
from tgvf_rl.framework.vllm import (
    FastTokenizerTokenByteSpanDecoder,
    VLLMOutputDecodingContract,
)
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.judges import JudgeRequest, JudgeSampleFailureError
from tgvf_rl.judges.openai_compatible import (
    JUDGE_SAMPLE_FAILURE_ZERO,
    load_openai_compatible_judge,
)
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    SampledAssistantTurn,
    StrictToolCallParser,
    TGVF_FOCUS_TOOL_NAME,
    TokenByteSpan,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
    native_policy_messages_sha256,
    render_successful_visual_tool_response,
    visual_tool_prompt_identity,
)
from tgvf_rl.protocol.schema import validate_native_tool_target_for_echo
from tgvf_rl.protocol.tool_prompts import (
    QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER,
)
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest,
    QwenVLMFamilyAdapter,
    batch_identical_injected_requests,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.experiments.answer_utility.evaluation.runner import (
    _load_validated_production_export,
    _require_instruct_training,
)
from tgvf_rl.representation.training.config import load_representation_training_config
from tgvf_rl.representation.training.evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _seed_current_process,
    _torch_dtype,
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.native_pipeline import (
    NativeActionTarget,
    Qwen3NativeRepresentationGroupBuilder,
    _adapter_output_bundle,
    _expand_native_visual_placeholders,
    _qwen3_position_ids,
    _single_visual_expansion_count,
    _source_bundle,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleArmContext,
    OracleDUtilityArm,
    OracleDUtilityModelInput,
    OracleGeneratedAnswer,
    OracleGroupVisuals,
    _batched_next_logits,
    _detached_bundle,
    _injected_block,
    _integer_sequence_sha256,
    _oracle_target_condition,
    _qwen3_multimodal_token_ids,
)
from tgvf_rl.representation.training.runtime import (
    Qwen3VisionPreMergeRequest,
    create_qwen3_representation_runtime,
)


FORCED_TGVF_RUN_SCHEMA = "tgvf.forced-tgvf-counterfactual.run.v1"
FORCED_TGVF_LEDGER_SCHEMA = "tgvf.forced-tgvf-counterfactual.ledger.v1"
FORCED_TGVF_FINAL_SCHEMA = "tgvf.forced-tgvf-counterfactual.final.v1"
QUESTION_TARGET_PROXY_VERSION = "question_target_proxy_v1"
ATTEMPT_SEED_VERSION = "forced-tgvf-content-addressed-attempt-seed-v1"
SAMPLING_CONTRACT = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "do_sample": True,
}
DEFAULT_EOS_TOKEN_IDS = (151645, 151643)
DEFAULT_MAX_NEW_TOKENS = 40_960
DEFAULT_ATTEMPTS_PER_SAMPLE = 8
DEFAULT_SHARD_COUNT = 4
DEFAULT_MASTER_SEED = 42
_FORCED_ACTION_DECODING = VLLMOutputDecodingContract(
    detokenize=True,
    skip_special_tokens=False,
    spaces_between_special_tokens=False,
    output_kind="final_only",
)
_ASSISTANT_REASONING = (
    "<think>I will inspect the target-conditioned visual evidence before "
    "answering.</think>"
)
_ASSISTANT_CLOSE = "<|im_end|>\n"
_ARXIV_CHOICE_LINE = re.compile(r"(?m)^\s*([A-Z])\.\s+\S")


@dataclass(frozen=True, slots=True)
class ForcedTGVFSample:
    """The model/scorer fields projected by one schedule row."""

    training_index: int
    sample_id: str
    candidate_sha256: str
    data_source: str
    task_kind: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str

    def __post_init__(self) -> None:
        if type(self.training_index) is not int or self.training_index < 0:
            raise ValueError("training_index must be non-negative")
        for field_name in (
            "sample_id",
            "candidate_sha256",
            "data_source",
            "task_kind",
            "image_sha256",
            "question",
            "ground_truth",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        _require_sha256(self.candidate_sha256, field="candidate_sha256")
        _require_sha256(self.image_sha256, field="image_sha256")
        object.__setattr__(self, "image_path", Path(self.image_path).resolve())
        if not self.image_path.is_absolute():  # pragma: no cover - resolve is absolute
            raise ValueError("image_path must be absolute")
        SelectionSource(self.data_source)


@dataclass(frozen=True, slots=True)
class ForcedTGVFRunPlan:
    run_id: str
    run_identity_sha256: str
    identity: Mapping[str, Any]
    samples: tuple[ForcedTGVFSample, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "sample_count": len(self.samples),
            "attempt_count": int(self.identity["execution"]["attempt_count"]),
            "identity": dict(self.identity),
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _load_json_object(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain an object")
    return value


def load_forced_tgvf_schedule(
    schedule_root: str | Path, *, sample_count: int
) -> tuple[tuple[ForcedTGVFSample, ...], dict[str, Any], str]:
    """Load and hash-check the exact first-N TGVF-80 schedule rows."""

    count = _positive_integer(sample_count, field="sample_count")
    root = Path(schedule_root).resolve(strict=True)
    manifest_path = root / SCHEDULE_MANIFEST_FILE
    manifest_with_sha = _load_json_object(
        manifest_path, field="TGVF utility schedule manifest"
    )
    manifest = dict(manifest_with_sha)
    manifest_sha256 = _require_sha256(
        manifest.pop("manifest_sha256", None), field="schedule manifest_sha256"
    )
    if _sha256_bytes(_canonical_json_bytes(manifest)) != manifest_sha256:
        raise ValueError("TGVF utility schedule manifest identity differs")
    if manifest.get("schema_version") != TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA:
        raise ValueError("TGVF utility schedule schema differs")
    files = manifest.get("files")
    descriptor = files.get("schedule") if isinstance(files, Mapping) else None
    if not isinstance(descriptor, Mapping) or descriptor.get("path") != SCHEDULE_FILE:
        raise ValueError("TGVF utility schedule descriptor differs")
    declared_rows = _positive_integer(descriptor.get("rows"), field="schedule rows")
    if count > declared_rows:
        raise ValueError("sample_count exceeds the TGVF-80 schedule")
    schedule_path = root / SCHEDULE_FILE
    payload = schedule_path.read_bytes()
    if _sha256_bytes(payload) != _require_sha256(
        descriptor.get("sha256"), field="schedule SHA-256"
    ):
        raise ValueError("TGVF utility schedule bytes differ")
    raw_lines = payload.splitlines()
    if len(raw_lines) != declared_rows:
        raise ValueError("TGVF utility schedule row count differs")

    samples: list[ForcedTGVFSample] = []
    for training_index, line in enumerate(raw_lines[:count]):
        row = json.loads(line)
        if (
            not isinstance(row, Mapping)
            or row.get("schema_version") != TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA
            or row.get("training_index") != training_index
        ):
            raise ValueError("TGVF utility schedule row identity differs")
        image = row.get("image")
        if not isinstance(image, Mapping):
            raise ValueError("TGVF utility schedule image differs")
        samples.append(
            ForcedTGVFSample(
                training_index=training_index,
                sample_id=str(row.get("sample_id", "")),
                candidate_sha256=str(row.get("candidate_sha256", "")),
                data_source=str(row.get("data_source", "")),
                task_kind=str(row.get("task_kind", "")),
                image_path=Path(str(image.get("path", ""))),
                image_sha256=str(image.get("sha256", "")),
                question=str(row.get("question", "")),
                ground_truth=str(row.get("ground_truth", "")),
            )
        )
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("selected TGVF utility schedule has duplicate sample IDs")
    return tuple(samples), manifest, manifest_sha256


def question_target_proxy(question: str) -> str:
    """Return the fixed newline-free question target used by this experiment."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question target proxy requires non-empty question text")
    target = re.sub(r"\s+", " ", question).strip()
    validate_native_tool_target_for_echo(target)
    return target


def _implementation_sha256() -> str:
    return _file_sha256(Path(__file__).resolve())


def build_forced_tgvf_run_plan(
    schedule_root: str | Path,
    source_evaluation_config: str | Path,
    judge_config: str | Path,
    *,
    run_id: str,
    sample_count: int = 128,
    attempts_per_sample: int = DEFAULT_ATTEMPTS_PER_SAMPLE,
    shard_count: int = DEFAULT_SHARD_COUNT,
    master_seed: int = DEFAULT_MASTER_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    eos_token_ids: Sequence[int] = DEFAULT_EOS_TOKEN_IDS,
) -> ForcedTGVFRunPlan:
    """Build a shard-independent, content-bound run identity without CUDA."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    attempts = _positive_integer(attempts_per_sample, field="attempts_per_sample")
    shards = _positive_integer(shard_count, field="shard_count")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    maximum = _positive_integer(max_new_tokens, field="max_new_tokens")
    eos = tuple(eos_token_ids)
    if (
        not eos
        or len(eos) != len(set(eos))
        or any(type(token_id) is not int or token_id < 0 for token_id in eos)
    ):
        raise ValueError("eos_token_ids must be unique non-negative integers")
    samples, schedule_manifest, schedule_manifest_sha256 = load_forced_tgvf_schedule(
        schedule_root, sample_count=sample_count
    )
    source = load_representation_internal_evaluation_run_config(
        Path(source_evaluation_config).resolve()
    )
    training = load_representation_training_config(
        source.training_config_path,
        allow_existing_post_training_report=True,
    )
    _require_instruct_training(training)
    judge_path = Path(judge_config).resolve(strict=True)
    judge_sha256 = _file_sha256(judge_path)
    # Parsing the judge config now catches a wrong/non-formal route before GPU
    # launch. Credential validation remains a run-time operation.
    load_openai_compatible_judge(judge_path, expected_file_sha256=judge_sha256)
    prompt_identity = visual_tool_prompt_identity(
        NativeToolCapabilityProfile.TGVF_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    identity: dict[str, Any] = {
        "schema_version": FORCED_TGVF_RUN_SCHEMA,
        "claim_scope": "forced_one_tgvf_image_plus_rp66_d_answer_utility",
        "run_id": run_id,
        "schedule": {
            "root": str(Path(schedule_root).resolve()),
            "manifest_sha256": schedule_manifest_sha256,
            "dataset_iteration_identity_sha256": schedule_manifest["dataset"][
                "iteration_identity_sha256"
            ],
            "selection": "exact_sequential_prefix_first_n_v1",
            "sample_count": len(samples),
            "ordered_samples": [
                {
                    "training_index": sample.training_index,
                    "sample_id": sample.sample_id,
                    "candidate_sha256": sample.candidate_sha256,
                    "image_sha256": sample.image_sha256,
                }
                for sample in samples
            ],
        },
        "rp66": {
            "source_evaluation_config": str(source.source_path),
            "source_evaluation_config_sha256": source.source_sha256,
            "training_config": str(source.training_config_path),
            "training_config_sha256": source.training_config_sha256,
            "adapter_path": str(source.artifact_path),
            "adapter_file_sha256": source.artifact_file_sha256,
            "adapter_manifest_sha256": source.artifact_manifest_sha256,
            "run_identity_sha256": source.expected_run_identity_sha256,
            "global_step": source.expected_global_step,
            "model_name": training.model.model_name,
            "model_path": str(training.model.local_path),
            "adapter_visual_contract": "main_D_plus_three_D_DeepStack_v1",
        },
        "prompt": {
            "assistant_dialect": NativeAssistantDialect.QWEN3_VL_INSTRUCT.value,
            "tool_profile": NativeToolCapabilityProfile.TGVF_ONLY.value,
            "prompt_bundle_sha256": prompt_identity.bundle_sha256,
            "target_strategy": QUESTION_TARGET_PROXY_VERSION,
            "target_context_transcript": ("formal_policy_system_user_forced_call_v1"),
            "forced_successful_tool_calls": 1,
            "source_image_present": True,
            "focused_d_present": True,
        },
        "sampling": {
            **SAMPLING_CONTRACT,
            "attempts_per_sample": attempts,
            "max_new_tokens": maximum,
            "eos_token_ids": list(eos),
            "master_seed": master_seed,
            "seed_derivation": ATTEMPT_SEED_VERSION,
            "decode_mode": "batched_native_injected_kv_cache",
        },
        "scoring": {
            "contract": "t1_p_full_source_specific_rule_first_v1",
            "answer_parser": T1_INSTRUCT_ANSWER_PARSER,
            "judge_config": str(judge_path),
            "judge_config_sha256": judge_sha256,
        },
        "execution": {
            "shard_assignment": "training_index_mod_shard_count_v1",
            "shard_count": shards,
            "attempt_count": len(samples) * attempts,
            "attempt_ledger": "atomic_per_shard_jsonl_rewrite_v1",
            "resume_decode": "reconstruct_full_index_stable_attempt_batch_v1",
        },
        "implementation": {
            "module": str(Path(__file__).resolve()),
            "module_sha256": _implementation_sha256(),
        },
    }
    run_identity_sha256 = _sha256_bytes(_canonical_json_bytes(identity))
    return ForcedTGVFRunPlan(
        run_id=run_id,
        run_identity_sha256=run_identity_sha256,
        identity=identity,
        samples=samples,
    )


def attempt_seed(
    plan: ForcedTGVFRunPlan,
    sample: ForcedTGVFSample,
    attempt_index: int,
) -> int:
    attempts = int(plan.identity["sampling"]["attempts_per_sample"])
    if type(attempt_index) is not int or not 0 <= attempt_index < attempts:
        raise ValueError("attempt_index lies outside the run")
    payload = {
        "schema_version": ATTEMPT_SEED_VERSION,
        "run_identity_sha256": plan.run_identity_sha256,
        "candidate_sha256": sample.candidate_sha256,
        "attempt_index": attempt_index,
        "master_seed": int(plan.identity["sampling"]["master_seed"]),
    }
    return int.from_bytes(
        hashlib.sha256(
            b"forced-tgvf-attempt-seed-v1\0" + _canonical_json_bytes(payload)
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)


def _forced_assistant_tool_call(target: str) -> dict[str, Any]:
    validate_native_tool_target_for_echo(target)
    return {
        "role": "assistant",
        "content": _ASSISTANT_REASONING,
        "tool_calls": (
            {
                "type": "function",
                "function": {
                    "name": TGVF_FOCUS_TOOL_NAME,
                    "arguments": {"target": target},
                },
            },
        ),
    }


def _token_positions_overlapping_bytes(
    spans: Sequence[TokenByteSpan], *, byte_start: int, byte_end: int
) -> tuple[int, ...]:
    if not 0 <= byte_start < byte_end:
        raise ValueError("forced Policy target byte span is invalid")
    positions = tuple(
        span.token_index
        for span in spans
        if span.byte_start < byte_end and span.byte_end > byte_start
    )
    if (
        not positions
        or positions != tuple(range(positions[0], positions[-1] + 1))
        or spans[positions[0]].byte_start > byte_start
        or spans[positions[-1]].byte_end < byte_end
    ):
        raise ValueError("forced Policy target token coverage is not exact")
    return positions


def build_forced_policy_action_target(
    *,
    sample: ForcedTGVFSample,
    target: str,
    policy_renderer: NativeProtocolRenderer,
) -> NativeActionTarget:
    """Bind RP66 target conditioning to the exact formal Policy transcript."""

    if policy_renderer.assistant_dialect is not (
        NativeAssistantDialect.QWEN3_VL_INSTRUCT
    ):
        raise ValueError("forced TGVF counterfactual requires Instruct rendering")
    target = question_target_proxy(target)
    base = build_visual_tool_prompt_messages(
        sample.question,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    assistant = _forced_assistant_tool_call(target)
    prefill = policy_renderer.render(base, add_generation_prompt=True)
    policy_renderer.assert_generation_prefill(prefill, policy_renderer.tokenizer)
    transcript = policy_renderer.render(
        base + (assistant,), add_generation_prompt=False
    )
    if not transcript.text.startswith(prefill.text) or not transcript.text.endswith(
        _ASSISTANT_CLOSE
    ):
        raise ValueError("forced Policy action differs from its generation prefill")
    sampled_text = transcript.text[
        len(prefill.text) : len(transcript.text) - len(_ASSISTANT_CLOSE)
    ]
    if not sampled_text:
        raise ValueError("forced Policy action completion is empty")

    sampled_ids = tuple(
        policy_renderer.tokenizer.encode(sampled_text, add_special_tokens=False)
    )
    decoder = FastTokenizerTokenByteSpanDecoder(policy_renderer.tokenizer)
    sampled_turn = SampledAssistantTurn(
        sampled_text=sampled_text,
        token_ids=sampled_ids,
        token_byte_spans=decoder.spans_for_output(
            text=sampled_text,
            token_ids=sampled_ids,
            decoding=_FORCED_ACTION_DECODING,
        ),
    )
    parsed = StrictToolCallParser().parse(sampled_turn)
    if parsed.name != TGVF_FOCUS_TOOL_NAME or parsed.target != target:
        raise ValueError("strict parser changed the forced Policy tool call")

    full_spans = decoder.spans_for_output(
        text=transcript.text,
        token_ids=transcript.token_ids,
        decoding=_FORCED_ACTION_DECODING,
    )
    prefill_byte_count = len(prefill.text.encode("utf-8"))
    positions = _token_positions_overlapping_bytes(
        full_spans,
        byte_start=prefill_byte_count + parsed.target_span.offsets.byte_start,
        byte_end=prefill_byte_count + parsed.target_span.offsets.byte_end,
    )
    canonical_span = TokenSpan(positions[0], positions[-1] + 1)
    canonical_ids = transcript.token_ids[canonical_span.start : canonical_span.end]
    if canonical_ids != parsed.target_span.token_ids:
        raise ValueError("formal Policy target IDs differ from strict tool parser")
    return NativeActionTarget(
        transcript=transcript,
        generation_prefill=prefill,
        sampled_turn=sampled_turn,
        canonical_target_span=canonical_span,
        canonical_target_token_ids=canonical_ids,
        target_text=parsed.target,
    )


def materialize_forced_policy_visuals(
    *,
    sample: ForcedTGVFSample,
    model_input: OracleDUtilityModelInput,
    target: str,
    runtime: Any,
    group_builder: Qwen3NativeRepresentationGroupBuilder,
    policy_renderer: NativeProtocolRenderer,
) -> OracleGroupVisuals:
    """Generate RP66 D from the formal Policy system/user/action context."""

    if model_input.sample_id != sample.sample_id or model_input.target != target:
        raise ValueError("forced Policy visual input identity differs")
    if group_builder.runtime is not runtime:
        raise ValueError("group builder and forced Policy runtime differ")
    if (
        policy_renderer.processor is not runtime.processor
        or policy_renderer.tokenizer is not runtime.tokenizer
    ):
        raise ValueError("Policy renderer and RP66 runtime processor differ")

    with runtime.validated_group_execution():
        action = build_forced_policy_action_target(
            sample=sample,
            target=target,
            policy_renderer=policy_renderer,
        )
        image = group_builder.image_loader(model_input.image)
        if image is None:
            raise ValueError("image_loader returned None")
        model_action, expansion = group_builder._materialize_action_with_expansion(
            action, image
        )
        visual_token_count = _single_visual_expansion_count(expansion)
        vision = runtime.extract_vision_features(
            Qwen3VisionPreMergeRequest(
                pixel_values=model_action.pixel_values,
                image_grid_thw=model_action.image_grid_thw,
            )
        )
        if int(vision.merged_main.shape[-2]) != visual_token_count:
            raise ValueError("source vision tokens differ from Policy action expansion")
        condition = _oracle_target_condition(
            runtime=runtime,
            model_input=model_input,
            action=model_action,
            vision=vision,
        )
        with torch.no_grad():
            adapter_output = runtime.adapter(
                runtime.make_adapter_input(condition, vision)
            )
        return OracleGroupVisuals(
            source=_detached_bundle(_source_bundle(vision)),
            correct_d_by_sample_id={
                sample.sample_id: _detached_bundle(
                    _adapter_output_bundle(adapter_output)
                )
            },
            image_grid_thw=vision.image_grid_thw,
        )


def build_forced_policy_prefix(
    *,
    sample: ForcedTGVFSample,
    target: str,
    policy_renderer: NativeProtocolRenderer,
) -> tuple[tuple[int, ...], str, dict[str, Any]]:
    """Render the live Policy prompt, forced call, and exact success suffix."""

    target = question_target_proxy(target)
    base = build_visual_tool_prompt_messages(
        sample.question,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    action = build_forced_policy_action_target(
        sample=sample,
        target=target,
        policy_renderer=policy_renderer,
    )
    initial = action.generation_prefill
    completed = action.transcript
    close_ids = tuple(
        policy_renderer.tokenizer.encode(_ASSISTANT_CLOSE, add_special_tokens=False)
    )
    if (
        not completed.text.startswith(initial.text)
        or not completed.text.endswith(_ASSISTANT_CLOSE)
        or completed.token_ids[: len(initial.token_ids)] != initial.token_ids
        or not close_ids
        or completed.token_ids[-len(close_ids) :] != close_ids
    ):
        raise RuntimeError("forced assistant call is not an exact policy continuation")
    forced_call_ids = completed.token_ids[
        len(initial.token_ids) : len(completed.token_ids) - len(close_ids)
    ]
    forced_call_text = completed.text[
        len(initial.text) : len(completed.text) - len(_ASSISTANT_CLOSE)
    ]
    if not forced_call_ids or "<tool_call>" not in forced_call_text:
        raise RuntimeError("forced assistant continuation lost its native tool call")

    response_text = render_successful_visual_tool_response(
        TGVF_FOCUS_TOOL_NAME,
        {"target": target},
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    environment_text = (
        QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
        + response_text
        + "\n"
        + QWEN_NATIVE_IMAGE_PLACEHOLDER
        + "\n\n"
        + QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER
        + QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX
    )
    environment_ids = tuple(
        policy_renderer.tokenizer.encode(environment_text, add_special_tokens=False)
    )
    canonical_ids = initial.token_ids + forced_call_ids + environment_ids
    rendered_text = initial.text + forced_call_text + environment_text
    if (
        tuple(policy_renderer.tokenizer.encode(rendered_text, add_special_tokens=False))
        != canonical_ids
    ):
        raise RuntimeError(
            "forced policy prefix changed tokenization across turn joins"
        )
    metadata = {
        "policy_messages_sha256": native_policy_messages_sha256(base),
        "forced_call_text_sha256": _sha256_bytes(forced_call_text.encode("utf-8")),
        "rendered_prefix_text_sha256": _sha256_bytes(rendered_text.encode("utf-8")),
        "canonical_prefix_token_ids_sha256": _integer_sequence_sha256(canonical_ids),
        "target_strategy": QUESTION_TARGET_PROXY_VERSION,
        "target_sha256": _sha256_bytes(target.encode("utf-8")),
    }
    return canonical_ids, rendered_text, metadata


def build_forced_policy_context(
    *,
    sample: ForcedTGVFSample,
    target: str,
    runtime: Any,
    policy_renderer: NativeProtocolRenderer,
    source_visual: Any,
    focused_d: Any,
    image_grid_thw: tuple[int, int, int],
) -> tuple[OracleArmContext, dict[str, Any]]:
    """Build the exact live-policy prefix followed by one successful D turn."""

    canonical_ids, rendered_text, transcript = build_forced_policy_prefix(
        sample=sample,
        target=target,
        policy_renderer=policy_renderer,
    )

    token_count = int(source_visual.main.shape[1])
    if int(focused_d.main.shape[1]) != token_count:
        raise ValueError("source and focused D visual token counts differ")
    model_ids, expansion = _expand_native_visual_placeholders(
        runtime,
        canonical_ids,
        visual_token_counts=(token_count, token_count),
    )
    visual_id = runtime.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    blocks = tuple(
        mapped
        for token_id, mapped in zip(
            expansion.canonical_token_ids,
            expansion.canonical_to_model_positions,
            strict=True,
        )
        if token_id == visual_id
    )
    if len(blocks) != 2:
        raise RuntimeError("forced policy prefix must contain source image and D")
    prefix_ids = torch.tensor((model_ids,), dtype=torch.long)
    attention_mask = torch.ones_like(prefix_ids)
    grid = torch.tensor((image_grid_thw, image_grid_thw), dtype=torch.long)
    position_ids = _qwen3_position_ids(
        runtime.model,
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        image_grid_thw=grid,
    )
    context = OracleArmContext(
        arm=OracleDUtilityArm.IMAGE_CORRECT_D,
        rendered_text_sha256=_sha256_bytes(rendered_text.encode("utf-8")),
        canonical_token_ids_sha256=_integer_sequence_sha256(canonical_ids),
        prefix_input_ids=prefix_ids,
        prefix_position_ids=position_ids,
        image_grid_thw=grid,
        visual_blocks=(
            _injected_block("source_image", blocks[0], source_visual),
            _injected_block("focused_d", blocks[1], focused_d),
        ),
        source_positions=blocks[0],
        d_positions=blocks[1],
        forbidden_multimodal_token_ids=_qwen3_multimodal_token_ids(runtime),
    )
    transcript = {
        **transcript,
        "prefix_token_count": int(context.prefix_input_ids.shape[1]),
        "source_visual_token_count": len(context.source_positions),
        "d_visual_token_count": len(context.d_positions),
    }
    return context, transcript


def sample_injected_answers_batched(
    *,
    context: OracleArmContext,
    attempt_seeds: Sequence[int],
    runtime: Any,
    family_adapter: QwenVLMFamilyAdapter,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[OracleGeneratedAnswer, ...]:
    """Sample independent T=1 categorical lanes from one shared D prefill."""

    seeds = tuple(attempt_seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("attempt_seeds must contain non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("attempt_seeds must be unique within one sample")
    if not eos_token_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in eos_token_ids
    ):
        raise ValueError("eos_token_ids must contain non-negative integers")
    _positive_integer(max_new_tokens, field="max_new_tokens")
    if not family_adapter.capabilities.native_injected_kv_cache:
        raise ValueError("family adapter has no injected KV-cache path")

    prefixes = tuple(context.materialize((), runtime) for _seed in seeds)
    batched_prefix = batch_identical_injected_requests(prefixes)
    with torch.no_grad():
        result = family_adapter.prefill_injected_cache(runtime.model, batched_prefix)
    past_key_values = result.past_key_values
    next_logits = _batched_next_logits(result.logits, lane_count=len(seeds))
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=next_logits.device)
        generator.manual_seed(seed)
        generators.append(generator)
    generated: list[list[int]] = [[] for _seed in seeds]
    cache_suffixes: list[list[int]] = [[] for _seed in seeds]
    finished = [False for _seed in seeds]
    stop_reasons: list[Literal["natural_stop", "length_cap"]] = [
        "length_cap" for _seed in seeds
    ]
    fill_token = eos_token_ids[0]

    for token_index in range(max_new_tokens):
        probabilities = torch.softmax(next_logits, dim=-1)
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("forced TGVF sampling produced invalid probabilities")
        predicted = tuple(
            int(
                torch.multinomial(
                    probabilities[index],
                    1,
                    replacement=True,
                    generator=generators[index],
                ).item()
            )
            for index in range(len(seeds))
        )
        for lane_index, token_id in enumerate(predicted):
            if finished[lane_index]:
                cache_suffixes[lane_index].append(fill_token)
                continue
            generated[lane_index].append(token_id)
            cache_suffixes[lane_index].append(token_id)
            if token_id in eos_token_ids:
                finished[lane_index] = True
                stop_reasons[lane_index] = "natural_stop"
        if all(finished) or token_index + 1 == max_new_tokens:
            break

        full_requests = tuple(
            context.materialize(tuple(cache_suffixes[index]), runtime)
            for index in range(len(seeds))
        )
        if len({request.input_ids.shape[1] for request in full_requests}) != 1:
            raise RuntimeError("forced TGVF batched cache lanes changed length")
        first = full_requests[0]
        position_batch_dimension = 0 if first.position_ids.ndim == 2 else 1
        cache_position = torch.tensor(
            (first.input_ids.shape[1] - 1,),
            dtype=torch.long,
            device=first.input_ids.device,
        )
        with torch.no_grad():
            result = family_adapter.forward_cached_token(
                runtime.model,
                CachedTokenForwardRequest(
                    input_ids=torch.cat(
                        tuple(request.input_ids[:, -1:] for request in full_requests),
                        dim=0,
                    ),
                    attention_mask=torch.cat(
                        tuple(request.attention_mask for request in full_requests),
                        dim=0,
                    ),
                    position_ids=torch.cat(
                        tuple(
                            request.position_ids[..., -1:] for request in full_requests
                        ),
                        dim=position_batch_dimension,
                    ),
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
            )
        past_key_values = result.past_key_values
        next_logits = _batched_next_logits(result.logits, lane_count=len(seeds))

    answers: list[OracleGeneratedAnswer] = []
    for token_values, stop_reason in zip(generated, stop_reasons, strict=True):
        token_ids = tuple(token_values)
        if not token_ids:
            raise RuntimeError("forced TGVF sampling produced no token")
        if any(
            token_id in context.forbidden_multimodal_token_ids for token_id in token_ids
        ):
            raise RuntimeError("forced TGVF answer emitted a multimodal token")
        text = runtime.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or not text:
            raise RuntimeError("forced TGVF sampled IDs decoded to empty text")
        answers.append(
            OracleGeneratedAnswer(
                token_ids=token_ids,
                text=text,
                stop_reason=stop_reason,
            )
        )
    return tuple(answers)


def _arxiv_option_count(question: str) -> int:
    labels = tuple(match.group(1) for match in _ARXIV_CHOICE_LINE.finditer(question))
    if not labels:
        raise ValueError("ArxivQA question has no canonical choice lines")
    expected = tuple(chr(ord("A") + index) for index in range(len(labels)))
    if labels != expected or not 2 <= len(labels) <= 26:
        raise ValueError("ArxivQA question choice labels are not contiguous A..Z")
    return len(labels)


def score_forced_tgvf_answer(
    *,
    plan: ForcedTGVFRunPlan,
    sample: ForcedTGVFSample,
    attempt_index: int,
    generated: OracleGeneratedAnswer,
    bound_judge: Any,
) -> dict[str, Any]:
    """Apply the exact p_full source rule, then its semantic judge fallback."""

    if generated.stop_reason == "length_cap":
        return {
            "status": "scored",
            "correct": False,
            "answer": parse_t1_answer(
                generated.text, answer_parser=T1_INSTRUCT_ANSWER_PARSER
            ),
            "verification": {
                "route": "max_new_tokens_exhausted_scored_incorrect_v1",
                "evidence": (
                    "counterfactual answer did not reach a configured EOS; "
                    "the length-capped attempt is conservatively scored incorrect"
                ),
                "judge_used": False,
                "generation_stop_reason": "length_cap",
            },
        }
    answer = parse_t1_answer(generated.text, answer_parser=T1_INSTRUCT_ANSWER_PARSER)
    source = SelectionSource(sample.data_source)
    option_count = (
        _arxiv_option_count(sample.question)
        if source is SelectionSource.ARXIVQA
        else None
    )
    result = verify_t1_answer(
        source=source,
        candidate_answer=answer,
        expected_answer=sample.ground_truth,
        option_count=option_count,
    )
    if result.outcome is not VerificationOutcome.SEMANTIC_REQUIRED:
        return {
            "status": "scored",
            "correct": bool(result.correct),
            "answer": answer,
            "verification": {
                "route": result.route,
                "evidence": result.evidence,
                "judge_used": False,
            },
        }

    payload = {
        "schema": "forced-tgvf-p-full-compatible-judge-request-v1",
        "run_identity_sha256": plan.run_identity_sha256,
        "sample_id": sample.sample_id,
        "attempt_index": attempt_index,
        "task_kind": policy_selection_semantic_judge_task_kind(
            source=source,
            question=sample.question,
            ground_truth=sample.ground_truth,
        ),
        "question": sample.question,
        "candidate_answer": answer,
        "reference_answer": sample.ground_truth,
    }
    request_id = "forced-tgvf-judge:" + _sha256_bytes(_canonical_json_bytes(payload))
    try:
        judged = bound_judge.provider.judge(
            JudgeRequest(
                request_id=request_id,
                task_kind=str(payload["task_kind"]),
                question=sample.question,
                candidate_answer=answer or "",
                reference_answer=sample.ground_truth,
                prompt_identity=bound_judge.prompt_identity,
            )
        )
        correct = bool(judged.score)
        verification = {
            "route": "qwen2.5_72b_semantic_fallback",
            "evidence": judged.rationale,
            "judge_used": True,
            "judge_request_id": request_id,
            "judge_model_identity_sha256": judged.model_identity.sha256,
            "judge_usage": asdict(judged.usage) if judged.usage is not None else None,
        }
    except JudgeSampleFailureError as error:
        if bound_judge.sample_failure_mode != JUDGE_SAMPLE_FAILURE_ZERO:
            raise
        correct = False
        verification = {
            "route": f"qwen2.5_72b_semantic_fallback_{error.failure_kind}_zero",
            "evidence": f"completed judge response {error.failure_kind}",
            "judge_used": True,
            "judge_request_id": request_id,
            "judge_model_identity_sha256": bound_judge.model_identity.sha256,
            "judge_usage": asdict(error.usage) if error.usage is not None else None,
        }
    return {
        "status": "scored",
        "correct": correct,
        "answer": answer,
        "verification": verification,
    }


def _identity_path(output_root: Path) -> Path:
    return output_root / "run-identity.json"


def _ledger_path(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard-{shard_index:02d}" / "ledger.jsonl"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_identity(output_root: Path, plan: ForcedTGVFRunPlan) -> None:
    path = _identity_path(output_root)
    payload = _canonical_json_line(
        {
            "schema_version": FORCED_TGVF_RUN_SCHEMA,
            "run_id": plan.run_id,
            "run_identity_sha256": plan.run_identity_sha256,
            "identity": dict(plan.identity),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing forced-TGVF run identity differs")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=".run-identity.", suffix=".tmp", dir=output_root
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("racing forced-TGVF run identity differs")
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _shard_lock(output_root: Path, shard_index: int) -> Iterator[None]:
    path = _ledger_path(output_root, shard_index).with_name("worker.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"forced-TGVF shard {shard_index} is already active"
            ) from error
        yield


def _load_ledger(
    path: Path,
    *,
    plan: ForcedTGVFRunPlan,
    shard_index: int,
    shard_count: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    allowed = {
        sample.sample_id: sample
        for sample in plan.samples
        if sample.training_index % shard_count == shard_index
    }
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"forced-TGVF ledger line {line_number} is not an object")
        sample = allowed.get(value.get("sample_id"))
        attempt_index = value.get("attempt_index")
        key = (str(value.get("sample_id")), attempt_index)
        if (
            value.get("schema_version") != TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA
            or value.get("run_id") != plan.run_id
            or value.get("run_identity_sha256") != plan.run_identity_sha256
            or sample is None
            or value.get("training_index") != sample.training_index
            or type(attempt_index) is not int
            or not 0
            <= attempt_index
            < int(plan.identity["sampling"]["attempts_per_sample"])
        ):
            raise ValueError("forced-TGVF ledger record identity differs")
        if key in completed:
            raise ValueError("forced-TGVF ledger contains a duplicate attempt")
        completed[key] = value
    return completed


def _write_ledger(
    path: Path, records: Mapping[tuple[str, int], Mapping[str, Any]]
) -> None:
    ordered = sorted(
        records.values(),
        key=lambda row: (int(row["training_index"]), int(row["attempt_index"])),
    )
    _atomic_write(path, b"".join(_canonical_json_line(row) for row in ordered))


def run_forced_tgvf_shard(
    schedule_root: str | Path,
    source_evaluation_config: str | Path,
    judge_config: str | Path,
    output_root: str | Path,
    *,
    run_id: str,
    sample_count: int = 128,
    attempts_per_sample: int = DEFAULT_ATTEMPTS_PER_SAMPLE,
    shard_index: int,
    shard_count: int = DEFAULT_SHARD_COUNT,
    master_seed: int = DEFAULT_MASTER_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    eos_token_ids: Sequence[int] = DEFAULT_EOS_TOKEN_IDS,
) -> dict[str, Any]:
    """Run or exactly resume one deterministic single-GPU counterfactual shard."""

    if type(shard_index) is not int or shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    if type(shard_count) is not int or shard_count <= 0 or shard_index >= shard_count:
        raise ValueError("shard_index must lie inside shard_count")
    plan = build_forced_tgvf_run_plan(
        schedule_root,
        source_evaluation_config,
        judge_config,
        run_id=run_id,
        sample_count=sample_count,
        attempts_per_sample=attempts_per_sample,
        shard_count=shard_count,
        master_seed=master_seed,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    output = Path(output_root).resolve()
    _publish_identity(output, plan)
    selected = tuple(
        sample
        for sample in plan.samples
        if sample.training_index % shard_count == shard_index
    )
    if not selected:
        raise ValueError("forced-TGVF shard contains no selected samples")

    with _shard_lock(output, shard_index):
        ledger_path = _ledger_path(output, shard_index)
        ledger = _load_ledger(
            ledger_path,
            plan=plan,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        total = len(selected) * attempts_per_sample
        if len(ledger) == total:
            return _shard_summary(plan, shard_index, shard_count, ledger, total)
        _require_single_visible_cuda()
        source_config = load_representation_internal_evaluation_run_config(
            Path(source_evaluation_config).resolve()
        )
        training = load_representation_training_config(
            source_config.training_config_path,
            allow_existing_post_training_report=True,
        )
        export = _load_validated_production_export(training, source_config)
        if export.state is None:
            raise ValueError("RP66 production export has no Adapter state")
        judge_path = Path(judge_config).resolve(strict=True)
        bound_judge = load_openai_compatible_judge(
            judge_path, expected_file_sha256=_file_sha256(judge_path)
        )
        bound_judge.provider.validate_credentials()
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        _enable_determinism()
        _seed_current_process(master_seed)
        processor, model = _load_qwen(training, device=device)
        tokenizer_length_before = len(processor.tokenizer)
        runtime = create_qwen3_representation_runtime(
            model=model,
            processor=processor,
            model_identity=training.model_identity,
            conditioning_config=training.provider,
            adapter_dtype=_torch_dtype(training.model.dtype),
            adapter_variant=training.adapter_variant,
            fixture_mode=False,
        )
        if (
            runtime.renderer.assistant_dialect
            is not NativeAssistantDialect.QWEN3_VL_INSTRUCT
        ):
            raise ValueError("RP66 runtime is not Qwen3-VL-Instruct")
        runtime.adapter.load_artifact_state_dict(export.state)
        runtime.adapter.requires_grad_(False)
        runtime.adapter.eval()
        model.requires_grad_(False)
        model.eval()
        family_adapter = Qwen3VLAdapter()
        group_builder = Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=family_adapter,
            prompt=training.prompt,
            image_loader=_load_rgb_image,
            image_max_pixels=training.model.image_max_pixels,
        )
        policy_renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=training.model.tokenizer_length,
            tool_names=NativeToolCapabilityProfile.TGVF_ONLY.tool_names,
            tool_schemas=build_native_tool_schemas(
                NativeToolCapabilityProfile.TGVF_ONLY.tool_names
            ),
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )
        started = time.monotonic()
        for sample in selected:
            attempt_indices = tuple(range(attempts_per_sample))
            pending = tuple(
                attempt_index
                for attempt_index in attempt_indices
                if (sample.sample_id, attempt_index) not in ledger
            )
            if not pending:
                continue
            if (
                not sample.image_path.is_file()
                or _file_sha256(sample.image_path) != sample.image_sha256
            ):
                raise ValueError(f"source image bytes differ for {sample.sample_id}")
            target = question_target_proxy(sample.question)
            model_input = OracleDUtilityModelInput(
                sample_id=sample.sample_id,
                image_group_key=sample.image_sha256,
                image=str(sample.image_path),
                question=sample.question,
                target=target,
                sample_content_sha256=sample.candidate_sha256,
            )
            visuals = materialize_forced_policy_visuals(
                sample=sample,
                model_input=model_input,
                target=target,
                runtime=runtime,
                group_builder=group_builder,
                policy_renderer=policy_renderer,
            )
            focused_d = visuals.correct_d_by_sample_id[sample.sample_id]
            context, transcript = build_forced_policy_context(
                sample=sample,
                target=target,
                runtime=runtime,
                policy_renderer=policy_renderer,
                source_visual=visuals.source,
                focused_d=focused_d,
                image_grid_thw=visuals.image_grid_thw,
            )
            # Always reconstruct the full, index-stable attempt batch.  A
            # partial resume must not change CUDA batch cardinality or move a
            # missing attempt to a different lane and thereby alter logits.
            seeds = tuple(
                attempt_seed(plan, sample, index) for index in attempt_indices
            )
            generated = sample_injected_answers_batched(
                context=context,
                attempt_seeds=seeds,
                runtime=runtime,
                family_adapter=family_adapter,
                eos_token_ids=tuple(eos_token_ids),
                max_new_tokens=max_new_tokens,
            )
            for attempt_index, seed, answer in zip(
                attempt_indices, seeds, generated, strict=True
            ):
                if attempt_index not in pending:
                    continue
                scored = score_forced_tgvf_answer(
                    plan=plan,
                    sample=sample,
                    attempt_index=attempt_index,
                    generated=answer,
                    bound_judge=bound_judge,
                )
                record = {
                    "schema_version": TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
                    "run_id": plan.run_id,
                    "run_identity_sha256": plan.run_identity_sha256,
                    "sample_id": sample.sample_id,
                    "candidate_sha256": sample.candidate_sha256,
                    "training_index": sample.training_index,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "attempt_index": attempt_index,
                    "attempt_seed": seed,
                    "status": scored["status"],
                    "correct": scored["correct"],
                    "answer": scored["answer"],
                    "data_source": sample.data_source,
                    "task_kind": sample.task_kind,
                    "target_strategy": QUESTION_TARGET_PROXY_VERSION,
                    "target_sha256": transcript["target_sha256"],
                    "transcript": transcript,
                    "sampling": {
                        **SAMPLING_CONTRACT,
                        "max_new_tokens": max_new_tokens,
                        "eos_token_ids": list(eos_token_ids),
                    },
                    "generation": {
                        "token_ids": list(answer.token_ids),
                        "token_ids_sha256": _integer_sequence_sha256(answer.token_ids),
                        "text": answer.text,
                        "text_sha256": _sha256_bytes(answer.text.encode("utf-8")),
                        "stop_reason": answer.stop_reason,
                    },
                    "verification": scored["verification"],
                    "elapsed_seconds_since_process_start": time.monotonic() - started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                ledger[(sample.sample_id, attempt_index)] = record
                _write_ledger(ledger_path, ledger)
            del context, visuals
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("forced TGVF run changed tokenizer length")
        return _shard_summary(plan, shard_index, shard_count, ledger, total)


def _require_single_visible_cuda() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("forced-TGVF shard requires exactly one visible CUDA GPU")


def _shard_summary(
    plan: ForcedTGVFRunPlan,
    shard_index: int,
    shard_count: int,
    ledger: Mapping[tuple[str, int], Mapping[str, Any]],
    total: int,
) -> dict[str, Any]:
    scored = sum(row.get("status") == "scored" for row in ledger.values())
    return {
        "schema_version": FORCED_TGVF_LEDGER_SCHEMA,
        "run_id": plan.run_id,
        "run_identity_sha256": plan.run_identity_sha256,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "completed_attempts": len(ledger),
        "scored_attempts": scored,
        "expected_attempts": total,
        "complete": len(ledger) == total and scored == total,
    }


def finalize_forced_tgvf_attempts(
    schedule_root: str | Path,
    output_root: str | Path,
    *,
    run_id: str,
    sample_count: int = 128,
    attempts_per_sample: int = DEFAULT_ATTEMPTS_PER_SAMPLE,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> dict[str, Any]:
    """Merge complete shard ledgers into the sidecar aggregator's JSONL."""

    output = Path(output_root).resolve(strict=True)
    identity_record = _load_json_object(
        _identity_path(output), field="forced-TGVF run identity"
    )
    if (
        identity_record.get("schema_version") != FORCED_TGVF_RUN_SCHEMA
        or identity_record.get("run_id") != run_id
    ):
        raise ValueError("forced-TGVF finalization run identity differs")
    run_identity_sha256 = _require_sha256(
        identity_record.get("run_identity_sha256"), field="run_identity_sha256"
    )
    identity = identity_record.get("identity")
    if (
        not isinstance(identity, Mapping)
        or _sha256_bytes(_canonical_json_bytes(identity)) != run_identity_sha256
        or identity.get("run_id") != run_id
        or identity.get("schedule", {}).get("sample_count") != sample_count
        or identity.get("sampling", {}).get("attempts_per_sample")
        != attempts_per_sample
        or identity.get("execution", {}).get("shard_count") != shard_count
    ):
        raise ValueError("forced-TGVF finalization parameters differ from run")
    samples, _manifest, schedule_manifest_sha256 = load_forced_tgvf_schedule(
        schedule_root, sample_count=sample_count
    )
    if identity["schedule"]["manifest_sha256"] != schedule_manifest_sha256:
        raise ValueError("forced-TGVF finalization schedule differs")

    expected = {
        (sample.sample_id, attempt_index): sample
        for sample in samples
        for attempt_index in range(attempts_per_sample)
    }
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    plan = ForcedTGVFRunPlan(run_id, run_identity_sha256, identity, samples)
    for shard_index in range(shard_count):
        rows = _load_ledger(
            _ledger_path(output, shard_index),
            plan=plan,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        overlap = set(merged).intersection(rows)
        if overlap:
            raise ValueError("forced-TGVF shard ledgers overlap")
        merged.update(rows)
    if set(merged) != set(expected):
        missing = len(set(expected).difference(merged))
        extra = len(set(merged).difference(expected))
        raise ValueError(
            f"forced-TGVF attempts are incomplete: missing={missing} extra={extra}"
        )
    unscored = sum(row.get("status") != "scored" for row in merged.values())
    if unscored:
        raise ValueError(f"forced-TGVF attempts contain {unscored} unscored records")
    ordered = sorted(
        merged.values(),
        key=lambda row: (int(row["training_index"]), int(row["attempt_index"])),
    )
    attempts_path = output / "attempts.jsonl"
    attempts_payload = b"".join(_canonical_json_line(row) for row in ordered)
    if attempts_path.exists() and attempts_path.read_bytes() != attempts_payload:
        raise ValueError("existing finalized forced-TGVF attempts differ")
    _atomic_write(attempts_path, attempts_payload)
    manifest_identity = {
        "schema_version": FORCED_TGVF_FINAL_SCHEMA,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "schedule_manifest_sha256": schedule_manifest_sha256,
        "sample_count": sample_count,
        "attempts_per_sample": attempts_per_sample,
        "attempt_count": len(ordered),
        "correct_count": sum(bool(row["correct"]) for row in ordered),
        "attempts": {
            "path": "attempts.jsonl",
            "sha256": _sha256_bytes(attempts_payload),
            "rows": len(ordered),
        },
    }
    manifest = {
        **manifest_identity,
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(manifest_identity)),
    }
    _atomic_write(output / "final-manifest.json", _canonical_json_line(manifest))
    return {
        **manifest,
        "attempts_path": str(attempts_path),
    }


__all__ = [
    "ATTEMPT_SEED_VERSION",
    "DEFAULT_ATTEMPTS_PER_SAMPLE",
    "DEFAULT_EOS_TOKEN_IDS",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_SHARD_COUNT",
    "FORCED_TGVF_FINAL_SCHEMA",
    "FORCED_TGVF_LEDGER_SCHEMA",
    "FORCED_TGVF_RUN_SCHEMA",
    "ForcedTGVFRunPlan",
    "ForcedTGVFSample",
    "QUESTION_TARGET_PROXY_VERSION",
    "SAMPLING_CONTRACT",
    "attempt_seed",
    "build_forced_policy_context",
    "build_forced_policy_prefix",
    "build_forced_tgvf_run_plan",
    "finalize_forced_tgvf_attempts",
    "load_forced_tgvf_schedule",
    "question_target_proxy",
    "run_forced_tgvf_shard",
    "sample_injected_answers_batched",
    "score_forced_tgvf_answer",
]
