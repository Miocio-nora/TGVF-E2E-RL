# I8H-20260719 Framework Implementation Report

Status: **bounded framework implementation complete; production gates open**
Branch: `codex/framework-implementation`
Base commit: `168b3ab`
Authorization commit: `2ffa28e`

## Result

The accepted eight-hour framework scope is implemented and smoke-verified. The
selected compatibility stack is CPython `3.12.3`, upstream veRL commit
`e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, Torch
`2.9.0+cu128`, and Transformers `4.57.6`, using the resolved
`requirements/compatibility.lock`. This is an accepted compatibility stack,
not a production-training topology or scale decision.

The runtime uses vLLM only. Live Qwen3 launches fail closed unless
`VLLM_PLUGINS=tgvf_qwen3_precomputed` and
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`; the vLLM multimodal encoder uses
`TORCH_SDPA`. No SGLang runtime, SDPO-bundled veRL tree, private trainer fork,
or site-package patch is used.

## Implemented framework

- framework-neutral identity, token ownership, actual behavior-logprob,
  immutable tensor, focused observation, trajectory, and strict checkpoint
  contracts;
- content-addressed observation store with bit-preserving BF16 round trip;
- native `tgvf_focus_tool` schema, strict parser, repeated-call state machine,
  native Qwen3 transcript golden hashes, and no tokenizer growth;
- batch-aware TGVF Adapter with main `D`, required D-DeepStack branches, frozen
  projection ports, and both target-conditioning providers;
- Qwen3 exact recorded-`D` replay forward and a separately scoped Qwen2.5
  main-`D` family path;
- framework-neutral multi-call rollout loop and decomposed fail-closed reward,
  judge, and run configuration boundaries;
- lossless bridges through veRL's public `AgentLoopOutput`, `AgentLoopManager`,
  `DataProto`, policy-loss registry, `FSDPEngineConfig`, and
  `CheckpointHandler` surfaces;
- a repository-owned public vLLM Qwen3 plugin for precomputed main `D` plus
  three D-DeepStack branches;
- explicit pure GRPO and reference-style pure-SDPO objective identities,
  feedback-conditioned teacher context, exact-observation teacher replay,
  token alignment, full-distribution and sampled-token distillation paths,
  teacher lifecycle, and strict checkpoint/resume state.

## Verification evidence

- `111` CPU tests pass. They cover the public veRL bridge, native transcript and
  parser, tokenizer invariants, two-call environment, exact behavior-logprob
  transport, immutable observations, deterministic replay contracts, both
  target-conditioning providers, sampling-transform oracle, GRPO tensor
  contract, and reference-style SDPO loss/gradient/teacher/checkpoint fixtures.
- `SC-20-R6` passes the bounded real-model transport smoke on physical B200
  GPUs 2 and 3. Qwen3-VL-8B-Thinking loaded with tensor parallelism 2 and
  consumed three precomputed width-16384 latent items (source plus two tool
  calls), each containing main `D` and three D-DeepStack branches, through an
  exact 295-token native two-call transcript. Tokenizer length stayed `151669`.
- `SC-30` passes a two-rank composable-FSDP2 infrastructure smoke. Strict
  distributed checkpoint save, teardown, reconstruction, and resume reproduced
  the next output, scalar loss, and updated local parameter shards bitwise on a
  tiny deterministic model.

The Qwen3 cell proves public precomputed-latent transport and native transcript
execution, not policy/reference replay parity, a trained TGVF Adapter, or
production reward/objective behavior. The FSDP2 cell proves infrastructure and
exact resume for its tiny fixture, not Qwen memory capacity, throughput, or a
production actor/reference/teacher placement.

## Model-family boundary

Qwen3-VL-8B-Thinking is the only real-model latent smoke. The current
`Qwen/Qwen2.5-VL-7B-Instruct` implementation is a family-adapter/main-`D`
synthetic boundary only. Full Qwen2.5-VL end-to-end support remains blocked on
an accepted runtime identity, a family-specific representation artifact,
native transcript fixtures, both-provider fixtures, exact replay/objective
fixtures, and a model-supported branch path. Qwen3 D-DeepStack weights and
dummy branches may not be reused to claim that support.

## Deliberately open production decisions

Real data and manifests, reward values/verifiers, final prompt text, exact
production GRPO and SDPO mathematics, any hybrid objective, policy trainable
scope, production sharding/placement/topology, long training, and activation of
the reserved 72B judge remain deliberately unset and fail closed. See
[`DEFERRED_DECISIONS.md`](DEFERRED_DECISIONS.md).
