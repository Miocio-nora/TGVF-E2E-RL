# Compatibility Environment Setup

Status: **bounded synthetic framework environment**
Updated: **2026-07-19 JST**

This environment is the accepted veRL/vLLM/FSDP2 compatibility stack. It is
not a production-training lock and does not select data, prompts, reward, GRPO
or SDPO production mathematics, policy tuning scope, or scale topology.

## Requirements

- CPython 3.12 on Linux x86-64;
- Python 3.12 development headers (`Python.h`) and a system GCC/G++ toolchain
  available when Triton compiles runtime extensions;
- an NVIDIA driver compatible with the PyTorch CUDA 12.8 wheel graph for GPU
  smoke work;
- network access for the exact upstream veRL Git revision and package wheels.

Python 3.10 is intentionally unsupported: the accepted veRL revision imports
`enum.StrEnum`. The old local `.venv` cannot execute the public agent-loop API;
use the isolated `.venv312` environment instead.

Do not inherit `CC` or `CXX` from a legacy project environment. Select the
system compiler explicitly before a live vLLM/Triton launch:

```bash
unset CC CXX CPATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

If Python development headers are not installed system-wide, supply an
explicit, version-matched header bundle through `CPATH`; the exact bundle used
by `SC-20-R6` is recorded in the experiment ledger. Do not point `CPATH` at an
unidentified legacy environment.

## Materialize the exact compatibility environment

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install --upgrade pip
.venv312/bin/python -m pip install -r requirements/compatibility.lock
.venv312/bin/python -m pip install -e . --no-deps
```

The lock pins upstream veRL to Git commit
`e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, Torch `2.9.0`,
Transformers `4.57.6`, and the complete resolved wheel graph observed by the
compatibility task. Its SHA256 is recorded in the compatibility report and
experiment ledger.

The editable project install is required so every vLLM process can discover
the repository's public general-plugin entry point. Before a live vLLM launch,
set exactly:

```bash
export VLLM_PLUGINS=tgvf_qwen3_precomputed
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

The live veRL adapter rejects a missing or different value for either variable.
The plugin is loaded by the vLLM core and workers; registering it only in the
caller process is insufficient. The accepted vLLM engine configuration also
sets
`actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA`.
This split uses `TRITON_ATTN` for language-model attention and `TORCH_SDPA` for
the multimodal encoder; substituting FlashInfer is not part of the accepted
compatibility path.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  .venv312/bin/python -m tgvf_rl.cli compat-info --live

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  .venv312/bin/python -m pytest -q

.venv312/bin/ruff format --check src tests spikes
.venv312/bin/ruff check src tests spikes
```

The live compatibility command verifies the installed veRL distribution's
source URL, exact Git commit and clean state before importing the accepted
public API symbols.

## Model identities

The primary operational model path is:

```text
/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct
```

The model is pinned to official revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; its downloaded shard identities
are recorded in `docs/EXTERNAL_REFERENCES.md`. Native transcript, tokenizer
length, processor behavior, and chat template remain fixture identities.

No Qwen2.5-VL local/runtime path or family-specific representation artifact has
been accepted. The current Qwen2.5-VL code is a fail-closed family boundary and
main-D synthetic contract, not an end-to-end compatibility claim. The separate
text-only `Qwen/Qwen2.5-72B-Instruct` benchmark judge is deployed at
`/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct`; its service identity and
launch command are documented in `docs/VLMEVALKIT.md`.

## GPU safety boundary

Only physical GPU indices `2` and `3` are authorized for the current bounded
smoke. Every command must set `CUDA_VISIBLE_DEVICES=2,3`; those devices then
appear to the process as logical CUDA devices `0` and `1`. Do not launch a GPU
command unless its exact identity and command are already `PLANNED` in
[`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md).
