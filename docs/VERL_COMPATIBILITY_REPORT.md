# veRL Compatibility Closeout Report

Date: 2026-07-19
Status: **bounded compatibility stack accepted; production training not approved**

## Decision

The compatibility spike passed for the framework questions exercised here:

- the exact upstream veRL candidate imports through the maintained/public
  surfaces used by this repository;
- the repo-owned vLLM plugin runs the real local Qwen3-VL-8B-Thinking model on
  two B200 GPUs with a native two-call tool transcript and precomputed main
  `D`/D-DeepStack inputs;
- two-rank composable FSDP2 can save, tear down, reconstruct, and resume model,
  optimizer, and extra state exactly in the accepted environment; and
- the project-owned behavior, exact-observation replay, GRPO, and
  reference-style pure-SDPO contracts pass their CPU test oracles.

This closes the bounded compatibility task. It does **not** approve a policy RL
phase training run, promote synthetic latents to TGVF Adapter evidence, or
freeze production objective mathematics. The detailed experiment identities
and artifacts remain in [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md).

This is not an overall PASS under the spike plan's original full promotion
matrix. I8H-20260719 later re-scoped execution to small smoke cells, so Qwen3
`L0..L4`, Qwen2.5 `L0..L1`, real policy/reference replay parity, and the GPU
SDPO teacher cell remain explicitly unrun or incomplete promotion gates. The
accepted result is limited to the exact framework, dependency, public-extension,
Qwen3 transport, and tiny FSDP2 questions listed above.

## Accepted compatibility stack

| Component | Pinned identity |
| --- | --- |
| Python | CPython `3.12.3` |
| upstream veRL | `0.9.0.dev0`, commit `e003163181731412595257a72ec173071efb125f` |
| vLLM rollout | `0.12.0`, V1 engine, repo-owned `tgvf_qwen3_precomputed` plugin |
| PyTorch | `2.9.0+cu128` |
| Transformers | `4.57.6` |
| dependency lock | `requirements/compatibility.lock`, SHA256 `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058` |
| GPU cells | physical GPUs 2 and 3, NVIDIA B200, TP/FSDP world size 2 |

This is the accepted compatibility environment, not yet a production-training
dependency lock. The vLLM path is deliberately constrained to
`processed_logprobs`, `TORCH_SDPA` for the multimodal encoder, `TRITON_ATTN` for
language attention, disabled prefix caching, zero multimodal processor cache,
and zero policy-adapter dropout.

## Evidence summary

### CPU contracts

The complete CPU suite passed with **111 tests** before the final GPU cell. The
coverage relevant to this decision includes:

- exact native transcript rendering, tokenizer non-growth, repeated
  `tgvf_focus_tool` calls, token ownership, and strict parsing;
- content-addressed, bit-preserving observation and behavior-trace stores;
- lossless trajectory-to-`AgentLoopOutput`-to-`DataProto` transport, including
  replay handles, actual behavior log probabilities, sampled-token masks, and
  integrity sentinels;
- exact vLLM 0.12 sampling-transform parity for penalties, temperature, min-p,
  top-k, top-p, and processed log probabilities, with unsupported transforms
  rejected;
- GRPO value and gradient agreement with an independent CPU oracle for the
  explicitly supplied test specification; and
- executable reference-style pure SDPO value/gradient tests, exact teacher
  observation replay, token alignment, teacher-state evolution, and strict
  checkpoint/resume checks.

These tests establish framework contracts and fail-closed behavior. They are
not a Qwen GPU loss/gradient parity run and do not select production settings.

### Qwen3 vLLM progression

The failed cells were retained because each exposed a distinct integration
requirement. No failure was relabeled as a pass.

| Cell | Result | Finding and bounded correction |
| --- | --- | --- |
| `SC-20` | FAIL | The plugin's variadic constructor was classified as old-style; the repo-owned subclass adopted vLLM's keyword-only `vllm_config`/`prefix` signature. |
| `SC-20-R1` | FAIL | Weights loaded, then Triton inherited a legacy compiler and could not find Python 3.12 headers; the launcher selected the system compiler and isolated headers. |
| `SC-20-R2` | FAIL | `Python.h` resolved but its architecture-specific `pyconfig.h` did not; the complete extracted include roots were supplied. |
| `SC-20-R3` | FAIL | The wheel-bundled visual FlashAttention PTX was unsupported by the host driver; the public multimodal `TORCH_SDPA` path was selected. |
| `SC-20-R4` | FAIL | Visual profiling passed, but implicit FlashInfer JIT required an unavailable CUDA-toolkit header path; the public language `TRITON_ATTN` backend was selected. |
| `SC-20-R5` | FAIL | Engine warmup passed, then one dictionary containing three latent items disagreed with vLLM's UUID cardinality; transport changed to a public list of three per-item dictionaries. |
| `SC-20-R6` | **PASS** | Real Qwen3-VL-8B-Thinking TP=2 executed the exact 295-token native transcript with two tool calls/responses and three precomputed latent items. |

`SC-20-R6` preserved tokenizer length `151669`, transported three one-row
width-16384 main-plus-three-branch latent items with aggregate SHA256
`3f7711522f4bd3c10530b765a186b109003727934459ed388f35e60863d26cda`,
sampled token IDs `(58533, 279)`, and returned finite non-positive vLLM
processed log probabilities `(0.0, 0.0)`. Its result SHA256 is
`473a9dcb438a51336bdfa0ee82d3a46b49262869b36868a1c02064b2e55bc4c7`.

The cell requested `top_k=20`, `top_p=0.9`, `min_p=0.01`, and
`logprobs=1`. The last setting controls only the number of reported top
alternatives; it does not truncate or renormalize the sampling distribution.
vLLM applies the represented sampling transforms before its float32
log-softmax, so `0.0` is a valid post-transform log probability when the
sampled token has probability 1 at float32 precision. The cell did not retain
the full processed distribution and therefore cannot distinguish singleton
support from numerical concentration. This is real backend-output transport
evidence, not rollout/replay or policy/reference logprob parity.

### FSDP2 and checkpoint evidence

`SC-30` **passed** on two ranks. A deterministic tiny FP32 model used public
PyTorch composable `fully_shard` and distributed-checkpoint APIs while also
resolving veRL's public `FSDPEngineConfig` and `CheckpointHandler`. After a
strict model/optimizer/extra-state checkpoint, teardown, reconstruction, and
resume, step-2 outputs, scalar loss, and updated rank-local parameter shards
were bitwise identical to the uninterrupted control (`atol=rtol=0`). Result
SHA256: `b131beb78ce03ba68fb91ba37369959182527c1aa43b237d407520e068fc89b1`.

This proves two-rank FSDP2 and strict checkpoint mechanics. It does not prove
Qwen3 FSDP2 memory fit, production throughput, or an RL optimizer step.

## Public extension surface

The integration uses normal installed-package discovery and these upstream
veRL symbols:

- `verl.experimental.agent_loop.AgentLoopManager` and `AgentLoopOutput`;
- `verl.protocol.DataProto`;
- `verl.trainer.ppo.core_algos.register_policy_loss`;
- `verl.workers.config.FSDPEngineConfig`; and
- `verl.utils.checkpoint.CheckpointHandler`.

The Qwen3 latent path is a repo-owned vLLM general plugin/model/processor
extension selected by `VLLM_PLUGINS=tgvf_qwen3_precomputed`; runtime behavior
is selected with public vLLM configuration and environment controls. The
installed veRL distribution is checked against the exact commit and clean
source identity before live use.

There is no site-package patch, private trainer/worker edit, vendored veRL
tree, SDPO-fork runtime, or DeepEyes-fork runtime. All corrective changes from
`SC-20` through `SC-20-R6` were limited to repo-owned adapters/plugins or
explicit runtime configuration and isolated headers.

## Framework contract scope

### Exact transport and behavior policy

Every assistant turn is required to carry a content-addressed behavior trace
containing all policy-sampled token IDs and the backend-returned processed log
probability for each sampled token. CPU fixtures verify lossless transport of
nontrivial values through the complete project bridge; `SC-20-R6` separately
verifies that real vLLM public output exposes sampled-token processed
logprobs. The trace also binds the behavior-policy version, backend/version,
request and response hashes, sampling parameters, logit processors, seed/RNG
identity, staleness, and whether probability was measured after sampling
transforms. Template-owned and tool-observation tokens are excluded from the
policy mask. Constructing recorded behavior as `new_logprobs.detach()` is
rejected.

The replay handle binds the trajectory to the exact rollout-materialized main
`D`, D-DeepStack tensors, layouts, positions, masks, and model/policy identity
in the content-addressed observation store. Policy, reference, and SDPO teacher
consumers resolve those handles rather than recomputing an observation from an
updated model. The veRL bridge carries these identities and actual behavior
probabilities through its public agent-loop and data boundaries and rejects
mutation, missing fields, duplicates, or mismatched hashes.

### GRPO

The repository contains a framework-neutral pure-tensor GRPO implementation
whose specification explicitly identifies group standard-deviation mode,
zero-variance behavior, behavior-versus-proximal ratio denominator, clipping,
optional dual clipping, reference-KL estimator/coefficient, token mask, and
reduction. CPU tests check value and gradient and verify that template tokens
receive neither loss nor gradient. This is sufficient to preserve the dataflow
and register a project-owned loss through veRL's public hook.

It is **not** acceptance of the production GRPO contract. No production group
normalization, clipping, KL, sequence/token normalization, or accumulation
choice has been frozen, and no veRL GRPO optimizer parity step has run.

### SDPO

The SDPO path is executable repository-owned code, not a frozen interface. It
implements feedback-conditioned teacher contexts, exact student/teacher token
alignment across multiple assistant turns, exact-observation teacher replay,
full-vocabulary and sampled-token reference-style loss modes, explicit teacher
regularization/state identity, and strict teacher checkpoint/resume. CPU tests
cover value/gradient oracles and reject stale, misaligned, or mismatched teacher
artifacts. The implementation is based on the separately pinned
`lasgroup/SDPO@7c457fc1b1f636ae794eb0362ba37d4743b06fbc` reference while keeping
upstream veRL as the runtime base.

This does not approve production pure SDPO or a hybrid. The final feedback
template, divergence, importance weighting, teacher update policy,
coefficients, and any GRPO-plus-SDPO composition remain separate unresolved
objective identities. No Qwen GPU SDPO optimizer step has run.

## Explicit limitations and remaining gates

- `SC-20-R6` used deterministic **synthetic precomputed latents**, not output
  from a trained TGVF Adapter. It proves shape/protocol transport only; it does
  not prove target specificity, causal readability, numerical parity with the
  legacy representation code, or representation-phase quality.
- There is no real-model GPU policy/reference replay parity result on the exact
  recorded observation. CPU storage/packing/replay contracts and the vLLM
  generation smoke do not establish Qwen policy/reference logits or logprobs
  parity, nor a full rollout-to-update cycle.
- Production data, reward, final prompt, and exact GRPO mathematics remain
  unset. Production SDPO mathematics and configuration also remain gated. The
  framework must continue to fail closed rather than inherit library defaults.
- Full Qwen2.5-VL end-to-end support remains blocked at the family boundary.
  It still requires a separately identified representation artifact, a native
  transcript fixture, both target-conditioning providers, a model-supported
  main-`D`/branch path, exact replay, and objective fixtures. Qwen3 checkpoints
  or dummy D-DeepStack branches cannot satisfy this gate.
- The optional Qwen2.5-72B answer judge is only a reserved provider and was not
  deployed or tested by this spike.

Accordingly, the next promotion step requires trained TGVF Adapter evidence
and exact Qwen policy/reference replay parity before any production policy RL
phase optimizer smoke. Data, reward, prompt, and objective decisions remain in
[`DEFERRED_DECISIONS.md`](DEFERRED_DECISIONS.md).
