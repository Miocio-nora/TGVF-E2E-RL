#!/usr/bin/python3 -I
"""One-request Qwen3-vLLM smoke over source plus two recorded-D items.

The request is a deterministic synthetic compatibility fixture. It loads the
real Qwen3-VL-8B-Thinking weights but does not evaluate task quality, run an
optimizer, or materialize a production TGVF Adapter checkpoint.
"""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(3):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "spikes/verl_compat/qwen3_vllm_latent_smoke.py",
        ),
    )

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import torch
from transformers import AutoConfig, AutoProcessor
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.inputs import TextPrompt
from vllm.plugins import load_general_plugins


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.framework.vllm import (  # noqa: E402
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_ATTENTION_BACKEND,
)
from tgvf_rl.compatibility_stack import (  # noqa: E402
    TORCH211_CU129_COMPATIBILITY_STACK,
    audited_compatibility_stack,
)
from tgvf_rl.experiment_identity import validate_run_id  # noqa: E402
from tgvf_rl.framework.verl import verify_verl_distribution_identity  # noqa: E402
from tgvf_rl.protocol import NativeProtocolRenderer  # noqa: E402
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


EXPECTED_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
PLUGIN_NAME = "tgvf_qwen3_precomputed"


def _bounded_output(raw: Path) -> Path:
    path = raw if raw.is_absolute() else REPOSITORY_ROOT / raw
    path = path.resolve()
    allowed = (REPOSITORY_ROOT / "artifacts" / "compatibility").resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"output must be a child of {allowed}")
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    return path


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Use the provided native tool when needed."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the relevant visual evidence."},
            ],
        },
        {
            "role": "assistant",
            "content": "<think>Inspect the center.</think>",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "tgvf_focus_tool",
                        "arguments": {"target": "center object"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Target-conditioned visual evidence."},
            ],
        },
        {
            "role": "assistant",
            "content": "<think>Inspect the left detail.</think>",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "tgvf_focus_tool",
                        "arguments": {"target": "left detail"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Target-conditioned visual evidence."},
            ],
        },
    ]


def _synthetic_latents(config: Any) -> dict[str, list[dict[str, torch.Tensor]]]:
    hidden_size = int(config.vision_config.out_hidden_size)
    deepstack = tuple(config.vision_config.deepstack_visual_indexes)
    merge_size = int(config.vision_config.spatial_merge_size)
    if hidden_size != int(config.text_config.hidden_size):
        raise ValueError("Qwen3 visual output and language hidden sizes differ")
    if deepstack != (8, 16, 24):
        raise ValueError("Qwen3 smoke requires DeepStack layers (8,16,24)")
    if merge_size != 2:
        raise ValueError("Qwen3 smoke fixture is pinned to spatial merge size 2")

    generator = torch.Generator(device="cpu").manual_seed(20260719)
    # grid [1,2,2] becomes exactly one merged visual row per item. Columns are
    # main D followed by one block for each of the three DeepStack branches.
    embeddings = torch.randn(
        3,
        hidden_size * (1 + len(deepstack)),
        dtype=torch.bfloat16,
        generator=generator,
    )
    return {
        "image": [
            {
                "image_embeds": embeddings[index : index + 1].clone(),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }
            for index in range(3)
        ]
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=_run_id_argument,
        required=True,
        help="explicit experiment identity; never inferred from --output",
    )
    parser.add_argument(
        "--stack",
        choices=(TORCH211_CU129_COMPATIBILITY_STACK,),
        required=True,
        help="explicit audited candidate stack selector",
    )
    parser.add_argument("--model", type=Path, default=EXPECTED_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validate_candidate_runtime(stack_selector: str) -> dict[str, Any]:
    stack = audited_compatibility_stack(stack_selector)
    versions = {
        name: metadata.version(name)
        for name in ("torch", "transformers", "vllm", "verl")
    }
    checks = {
        "python_3_12": platform.python_version_tuple()[:2] == ("3", "12"),
        "torch_distribution": versions["torch"] == stack.torch_distribution_version,
        "torch_runtime": str(torch.__version__) == stack.torch_runtime_version,
        "transformers": versions["transformers"]
        == stack.transformers_distribution_version,
        "vllm": versions["vllm"] == stack.vllm_distribution_version,
    }
    if not all(checks.values()):
        raise RuntimeError(f"candidate runtime identity failed: {checks}")
    verl = verify_verl_distribution_identity(expected_commit=stack.verl_commit)
    return {
        "compatibility_stack": asdict(stack),
        "python": platform.python_version(),
        "torch_runtime": str(torch.__version__),
        "distributions": versions,
        "verl": {
            "commit": verl.commit,
            "source_url": verl.source_url,
            "source_kind": verl.source_kind,
            "source_clean": verl.source_clean,
        },
        "checks": {**checks, "verl_commit": verl.commit == stack.verl_commit},
    }


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "spikes/verl_compat/qwen3_vllm_latent_smoke.py"
    )
    args = _parse_args(argv)
    runtime_identity = _validate_candidate_runtime(args.stack)
    model_path = args.model.resolve()
    output_path = _bounded_output(args.output)
    if model_path != EXPECTED_MODEL_PATH:
        raise ValueError(f"model path must be exactly {EXPECTED_MODEL_PATH}")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly '2,3'")
    if os.environ.get("VLLM_PLUGINS") != PLUGIN_NAME:
        raise RuntimeError(f"VLLM_PLUGINS must be exactly {PLUGIN_NAME!r}")
    if os.environ.get("VLLM_ATTENTION_BACKEND") != TGVF_VLLM_ATTENTION_BACKEND:
        raise RuntimeError(
            f"VLLM_ATTENTION_BACKEND must be {TGVF_VLLM_ATTENTION_BACKEND!r}"
        )

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer_length = len(processor.tokenizer)
    renderer = NativeProtocolRenderer(
        processor, expected_tokenizer_length=tokenizer_length
    )
    transcript = renderer.render(_messages(), add_generation_prompt=True)
    renderer.assert_generation_prefill(transcript, processor.tokenizer)
    conversation_text = transcript.text.split("<|im_end|>", maxsplit=1)[1]
    if conversation_text.count("<tool_call>") != 2:
        raise AssertionError("synthetic transcript must contain two native tool calls")
    if conversation_text.count("<tool_response>") != 2:
        raise AssertionError("synthetic transcript must contain two tool responses")
    if len(processor.tokenizer) != tokenizer_length:
        raise AssertionError("tokenizer length changed during native rendering")

    load_general_plugins()
    if TGVF_QWEN3_VLLM_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        raise AssertionError("repo-owned vLLM general plugin was not loaded")
    multimodal_data = _synthetic_latents(config)

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=2,
        dtype="bfloat16",
        seed=20260719,
        gpu_memory_utilization=0.72,
        enforce_eager=True,
        hf_overrides={"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
        enable_mm_embeds=True,
        mm_processor_cache_gb=0,
        mm_encoder_attn_backend="TORCH_SDPA",
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 3},
        max_model_len=512,
        max_num_seqs=1,
        logprobs_mode="processed_logprobs",
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.01,
        repetition_penalty=1.05,
        presence_penalty=0.1,
        frequency_penalty=0.05,
        seed=20260719,
        max_tokens=2,
        logprobs=1,
        detokenize=False,
    )
    outputs = llm.generate(
        [TextPrompt(prompt=transcript.text, multi_modal_data=multimodal_data)],
        sampling_params=sampling,
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise AssertionError("vLLM returned an unexpected request/output count")
    completion = outputs[0].outputs[0]
    token_ids = tuple(int(token_id) for token_id in completion.token_ids)
    if len(token_ids) != 2 or completion.logprobs is None:
        raise AssertionError("vLLM did not return two sampled tokens with logprobs")
    actual_logprobs: list[float] = []
    for token_id, token_logprobs in zip(token_ids, completion.logprobs, strict=True):
        selected = token_logprobs.get(token_id)
        if selected is None:
            raise AssertionError("sampled token is absent from vLLM logprobs")
        value = float(selected.logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise AssertionError("vLLM sampled logprob is not finite and non-positive")
        actual_logprobs.append(value)

    latent_tensor = torch.cat(
        [item["image_embeds"] for item in multimodal_data["image"]], dim=0
    )
    payload = {
        "schema_version": "tgvf-qwen3-vllm-latent-smoke-v1",
        "run_id": args.run_id,
        "result": "PASS",
        "stack": args.stack,
        "runtime_identity": runtime_identity,
        "scope": "synthetic source plus two precomputed-D items; no optimizer",
        "model": {
            "path": str(model_path),
            "model_type": config.model_type,
            "architecture_override": TGVF_QWEN3_VLLM_ARCHITECTURE,
            "tokenizer_length_before_after": [
                tokenizer_length,
                len(processor.tokenizer),
            ],
        },
        "versions": {
            name: metadata.version(name)
            for name in ("torch", "transformers", "vllm", "verl")
        },
        "physical_gpu_ids": [2, 3],
        "logical_gpu_ids": [0, 1],
        "tensor_parallel_size": 2,
        "transcript": {
            "token_ids_sha256": transcript.token_ids_sha256,
            "text_sha256": transcript.text_sha256,
            "chat_template_sha256": transcript.chat_template_sha256,
            "tool_schema_sha256": transcript.tool_schema_sha256,
            "prompt_tokens": len(transcript.token_ids),
            "tool_calls": 2,
            "tool_responses": 2,
        },
        "latent": {
            "items": 3,
            "rows_per_item": 1,
            "width": int(latent_tensor.shape[-1]),
            "sha256": hashlib.sha256(
                latent_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        },
        "sampling": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "logprobs_mode": "processed_logprobs",
            "seed": sampling.seed,
        },
        "sampled_token_ids": token_ids,
        "actual_processed_logprobs": actual_logprobs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
