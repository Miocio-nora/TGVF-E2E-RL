"""Single-GPU gate for the real Qwen3 TGVF live-tool chain.

This intentionally avoids veRL/Ray startup.  It loads the same local model and
representation artifact as the Policy run, executes two contextual TGVF calls
on the selected real sample, and proves the second call can consume the first
recorded D/DeepStack observation before it is packed for the next vLLM turn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoProcessor

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.framework.verl.policy_live_runtime import (
    Qwen3PolicyE2ELiveRuntimeBuilder,
)
from tgvf_rl.framework.verl.policy_runtime import (
    PolicyAgentLoopWorkerPlacement,
    PolicyE2ERuntimeBuildContext,
    _policy_decoding_contract,
)
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
)
from tgvf_rl.framework.verl.smoke_dataset import (
    TGVFSelectedSampleDataset,
    VerlSelectedSampleDatasetBinding,
)
from tgvf_rl.framework.vllm import (
    FastTokenizerTokenByteSpanDecoder,
    VLLMPolicyTurnRequest,
    VLLMTurnRNGIdentity,
    require_preexpanded_prompt_contract,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.trajectories.schema import TrajectoryIdentity


class _NeverSampler:
    def sample(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("the direct tool-chain gate must not sample")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _claim_request(
    registry: object,
    *,
    prompt_ids: tuple[int, ...],
    turn_index: int,
    policy: PolicyVersion,
) -> tuple[VLLMPolicyTurnRequest, object]:
    context_sha = registry.sha256_for_turn(prompt_ids, turn_index=turn_index)
    request = VLLMPolicyTurnRequest(
        request_id=f"direct-tool-chain-turn-{turn_index}",
        prompt_token_ids=prompt_ids,
        sampling_parameters={"temperature": 1.0},
        turn_index=turn_index,
        behavior_policy=policy,
        rng=VLLMTurnRNGIdentity(42 + turn_index, _sha(f"rng-{turn_index}")),
        backend_prompt_payload_sha256=context_sha,
        backend_version="0.12.0",
        logprobs_mode="processed_logprobs",
        decoding=_policy_decoding_contract(),
        termination_contract_sha256=_sha("direct-tool-chain-termination"),
    )
    return request, registry.for_request(request)


def _sampled_tool_call(
    tokenizer: object,
    decoder: FastTokenizerTokenByteSpanDecoder,
    *,
    target: str,
    turn_index: int,
    policy: PolicyVersion,
    backend_request_sha256: str,
) -> SampledPolicyTurn:
    arguments = json.dumps(
        {"name": "tgvf_focus_tool", "arguments": {"target": target}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    text = f"inspect the image</think>\n<tool_call>\n{arguments}\n</tool_call>"
    token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    spans = decoder.spans_for_output(
        text=text,
        token_ids=token_ids,
        decoding=_policy_decoding_contract(),
    )
    think_end = text.index("</think>") + len("</think>")
    think_end_byte = len(text[:think_end].encode("utf-8"))
    think_token_end = next(
        index + 1 for index, span in enumerate(spans) if span.byte_end == think_end_byte
    )
    sampling = SamplingIdentity(
        policy_version=policy,
        backend="vllm",
        backend_version="0.12.0",
        seed=42 + turn_index,
        rng_state_sha256=_sha(f"rng-{turn_index}"),
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    return SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=(-0.1,) * len(token_ids),
        sampling=sampling,
        think_token_span=TokenSpan(0, think_token_end),
        stop_reason="stop",
        backend_request_sha256=backend_request_sha256,
        backend_response_sha256=_sha(f"response-{turn_index}:{text}"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("gate requires exactly one CUDA-visible device")

    started = perf_counter()
    config = load_policy_e2e_smoke_run_config(args.config.resolve())
    processor = AutoProcessor.from_pretrained(
        config.model.revision_or_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    binding = VerlSelectedSampleDatasetBinding.from_run_config(config)
    data_config = {
        "tgvf_selected_sample": binding.as_config(),
        "mm_processor_kwargs": {"max_pixels": config.policy.image_max_pixels},
    }
    dataset = TGVFSelectedSampleDataset(
        str(binding.samples_path),
        tokenizer,
        data_config,
        processor,
    )
    sample_fields = dataset[0]
    prompt_ids = tuple(sample_fields["initial_prompt_token_ids"])

    policy = PolicyVersion(config.run_id, 0, "3" * 64)
    placeholder = (config.output.root / "direct-tool-chain-placeholder").resolve()
    snapshot = PolicyLoRASnapshot(
        policy_version=policy,
        run_identity_sha256=config.identity_sha256,
        request_sha256="1" * 64,
        tensor_file=placeholder,
        tensor_file_sha256="2" * 64,
        manifest_file=placeholder,
        tensors={},
    )
    context = PolicyE2ERuntimeBuildContext(
        config=config,
        placement=PolicyAgentLoopWorkerPlacement(
            worker_index=0,
            logical_gpu_id=0,
            physical_gpu_id=args.physical_gpu,
            world_size=4,
        ),
        initial_snapshot=snapshot,
        weight_sync_state=PolicyWeightSyncState(
            directory=(config.output.root / "direct-tool-chain-state").resolve(),
            run_id=config.run_id,
            run_identity_sha256=config.identity_sha256,
        ),
        trainer_config={},
        server_manager=object(),
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=TGVFSelectedSampleDataset,
        data_config=data_config,
    )
    product = Qwen3PolicyE2ELiveRuntimeBuilder().build(context)
    identity = TrajectoryIdentity(
        config.run_id,
        config.dataset.selected_sample.sample_id,
        0,
        "direct-tool-chain-group",
    )
    components = product.trajectory_components.build_trajectory_components(
        identity=identity,
        model=config.model,
        behavior_policy=policy,
        initial_prompt_token_ids=prompt_ids,
        sample_fields=sample_fields,
    )
    native_loop = components.native_loop_factory(_NeverSampler())
    registry = components.prompt_context
    decoder = FastTokenizerTokenByteSpanDecoder(tokenizer)
    strict_parser = StrictToolCallParser(enabled_tool_names=("tgvf_focus_tool",))

    current_prompt = prompt_ids
    prior_handles = ()
    handles = []
    for turn_index, target in enumerate(
        (
            "the small relevant text or mark needed to answer the question",
            "the most important visual relation needed to verify the answer",
        )
    ):
        backend_request, _ = _claim_request(
            registry,
            prompt_ids=current_prompt,
            turn_index=turn_index,
            policy=policy,
        )
        sampled = _sampled_tool_call(
            tokenizer,
            decoder,
            target=target,
            turn_index=turn_index,
            policy=policy,
            backend_request_sha256=backend_request.backend_request_sha256,
        )
        parsed = strict_parser.parse(sampled.parser_turn())
        handle = native_loop.tool_runtime.execute(
            parsed,
            ToolExecutionContext(
                trajectory_identity=identity,
                model=config.model,
                behavior_policy=policy,
                trajectory_source_visual=components.source_visual,
                prior_observation_handles=tuple(prior_handles),
                prompt_token_ids_before_turn=current_prompt,
                sampled_turn=sampled,
                assistant_turn_index=turn_index,
                attempt_index=turn_index,
                call_index=turn_index,
            ),
        )
        current_prompt, _ = native_loop.appender.append(
            current_prompt,
            sampled,
            handle,
            call_index=turn_index,
            parsed_call=parsed,
        )
        handles.append(handle)
        prior_handles = tuple(handles)

    _, third_turn_inputs = _claim_request(
        registry,
        prompt_ids=current_prompt,
        turn_index=2,
        policy=policy,
    )
    image_items = third_turn_inputs.multi_modal_data["image"]
    require_preexpanded_prompt_contract(
        third_turn_inputs.mm_processor_kwargs,
        prompt_token_ids=current_prompt,
        expected_image_items=3,
    )
    if len(image_items) != 3:
        raise RuntimeError("two successful calls must produce source + two D items")

    store = product.trajectory_components.store
    d_shapes = []
    for handle in handles:
        record = store.resolve_record(handle)
        main = store.resolve_verified_for_trajectory(
            record.payload.main_d,
            trajectory_id=identity.canonical_id,
        )
        branches = tuple(
            store.resolve_verified_for_trajectory(
                branch.d_tensor,
                trajectory_id=identity.canonical_id,
            )
            for branch in record.branches
        )
        if main.ndim != 2 or len(branches) != 3 or any(
            branch.shape != main.shape for branch in branches
        ):
            raise RuntimeError("recorded main D/DeepStack shapes differ")
        d_shapes.append(tuple(main.shape))

    print(
        json.dumps(
            {
                "status": "PASS",
                "physical_gpu": args.physical_gpu,
                "initial_prompt_tokens": len(prompt_ids),
                "third_turn_prompt_tokens": len(current_prompt),
                "observation_count": len(handles),
                "recorded_d_shapes": d_shapes,
                "vllm_item_shapes": [
                    tuple(item["image_embeds"].shape) for item in image_items
                ],
                "elapsed_seconds": perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
