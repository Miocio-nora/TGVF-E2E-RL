# Original/Crop step-16 texture supervisor

For benchmark background, dataset provenance, evaluation protocol, and the
completed result table, use the canonical
[LAS&T and MMAD texture benchmark report](texture_benchmarks.md). This document
only describes supervisor operation and recovery.

`tools/supervise_texture_original_crop_step16.sh` owns the complete two-arm
LAS&T/MMAD run. It launches four durable stock-Qwen ranks on physical GPUs
4--7 and four PRL14 Crop-only step-16 ranks on physical GPUs 0--3. All eight
ranks use ordinal-modulo-four task assignment and can resume their validated
JSONL prefixes after a process restart.

The default invocation is:

```bash
tools/supervise_texture_original_crop_step16.sh
```

This uses `TEXTURE_TWO_ARM_SETUP_MODE=strict`: it repeats full image-byte
validation, Crop materialization, prepare, and static validation before worker
launch. That remains the default fail-closed behavior.

The default matrix is
`configs/evaluation/texture_last_mmad_original_crop_prl14_step16_512_v1.json`.
Use `--matrix /absolute/path/to/matrix.json` only for a matrix with exactly one
`original` arm, one `crop` arm, and Crop `gpu_ids` equal to `[0,1,2,3]`.

## Fixed B200 runtime

Before any worker starts, the supervisor validates every task image and the
Crop snapshot, then binds the accepted cold-JIT environment:

- `/usr/bin/gcc` and `/usr/bin/g++`;
- the extracted Python 3.12 headers, including the architecture-qualified
  `pyconfig.h` root;
- the venv CUDA 12.8 cublas, NVRTC, and runtime headers;
- the local unversioned `libcudart.so` linker alias;
- CUDA 12.8 `nvcc`, vLLM V1, and multiprocessing `spawn`;
- rank-local Triton, TorchInductor, Torch extension, and FlashInfer caches;
- a short private rank-local `/tmp/t2a-*` directory for vLLM ZeroMQ IPC
  sockets, whose Linux path limit is 107 bytes. Logs and durable caches remain
  under the evaluation control root.

The original runner owns and fixes
`mm_encoder_attn_backend=TORCH_SDPA`; the supervisor verifies that constant and
forbids an engine-kwargs override. It also unsets any ambient global
`VLLM_ATTENTION_BACKEND`, so a language-attention setting cannot leak into the
Qwen vision encoder.

The accepted original vLLM capacity defaults are
`gpu_memory_utilization=0.8`, `max_model_len=32768`, `max_num_seqs=8`, and
`max_num_batched_tokens=32768`.

## Explicit setup resume

Use resume mode only after strict setup was already completed for the exact
matrix and Crop config. It avoids repeating full image reads, Crop
materialization, and prepare:

```bash
TEXTURE_TWO_ARM_SETUP_MODE=resume \
  tools/supervise_texture_original_crop_step16.sh
```

Resume still regenerates the Crop command plan, validates the existing config
against the current matrix arm, requires the exact GPU/rank argv mapping, and
runs matrix validation without image-byte reads. It then verifies the bound
task-manifest hash/count, frozen run config, frozen full-model manifest and
receipt, materialized model file-size/small-file closure, and canonical policy
evaluation identity.

The closure also requires evidence from exactly one successful lightweight
policy static validation. When no prior evidence exists in the control root,
resume runs it once and saves
`setup/resume-static-validation.json`; subsequent resume invocations reuse it.
To reuse a manually completed validation without running it again, capture its
JSON stdout and pass it explicitly:

```bash
CONTROL=/absolute/output/root/runtime/original-crop-step16-supervisor
CONFIG=/absolute/output/root/crop-prl14-cleanfinal-step16/policy-benchmark-config.json
mkdir -p "$CONTROL/setup/manual"
PYTHONPATH="$PWD/src:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl" \
  /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python \
  tools/run_policy_benchmark.py --config "$CONFIG" \
  --mode validate --world-size 4 \
  >"$CONTROL/setup/manual/crop-static-validation.json"

TEXTURE_TWO_ARM_SETUP_MODE=resume \
TEXTURE_TWO_ARM_RESUME_VALIDATE_EVIDENCE="$CONTROL/setup/manual/crop-static-validation.json" \
  tools/supervise_texture_original_crop_step16.sh
```

Every resume invocation writes an immutable
`setup/resume-records/attempt-NNN.json` containing the skipped steps and hashes
for all accepted evidence. Mismatched or stale evidence stops before the GPU
idle gate.

## Safety and recovery

The launch gate requires all target devices to be NVIDIA B200s at zero
utilization, no more than 16 MiB baseline memory, and with no listed compute
process. A busy or ambiguous device stops the corresponding launch. The
supervisor never sends signals to a PID discovered via `nvidia-smi`.

Each rank has its own retry loop. The default is four restarts with a 30-second
cooldown; change these only through
`TEXTURE_TWO_ARM_MAX_RESTARTS` and
`TEXTURE_TWO_ARM_RESTART_COOLDOWN`. On an interactive termination, the script
signals only the process groups it created. Unknown or leaked GPU processes
are left untouched, and the next idle gate fails closed.

The launch plan also binds a hash over the complete repo-owned `tgvf_rl` and
pinned veRL source trees. A source change before a retry or finalization stops
the run rather than mixing code revisions across durable rank prefixes.

The output-root supervisor directory is
`runtime/original-crop-step16-supervisor/`. It contains:

- the immutable source/environment `launch-plan.json`;
- CPU-side setup and validation logs;
- per-arm, per-rank, per-attempt command, PID, log, GPU-idle, and exit files;
- rank-local compilation caches;
- final original and Crop status JSON;
- original finalization output and both score reports;
- `final/supervisor-summary.json` after strict completion.

Rerunning the same command is the recovery operation. Durable rank rows are
validated before reuse; completed ranks exit without constructing another
engine. A changed matrix, code identity, execution setting, or immutable output
causes a fail-closed error rather than silent replacement.

## Non-GPU checks

These checks do not construct a model or touch CUDA:

```bash
bash -n tools/supervise_texture_original_crop_step16.sh
.venv312/bin/pytest -q \
  tests/tools/test_supervise_texture_original_crop_step16.py
```
