# I8H-20260719 Framework Implementation Report

Status: **bounded framework implementation complete; production gates open**
Branch: `main`
Base commit: `168b3ab`
Authorization commit: `2ffa28e`
Representation implementation commit: `ce6a15f`
Updated: **2026-07-19 JST**

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

- `346` CPU tests pass. They cover the public veRL bridge, native transcript and
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
- corrected `RP-10` passes one real Qwen3-VL-8B-Thinking representation
  optimizer step on physical GPUs 2 and 3. The target-token-embedding provider, native main `D`
  and all three D-DeepStack branches, same-image Matrix CE plus `L_gen`, frozen
  Qwen, two-rank FSDP2, validation, content-bound DCP save, and 104-tensor
  Adapter-only export all completed without tokenizer growth.
- `RP-11` passes the real-Qwen3 K=4/GA=4 two-update execution and a separate
  step-1 save, full process teardown, reconstruction, strict DCP restore, and
  matching step-2 update. All 104 Adapter tensors and recorded model/optimizer/
  scheduler/sampler/RNG/shard and scientific metric state compare exactly.

The Qwen3 cell proves public precomputed-latent transport and native transcript
execution, not policy/reference replay parity, a trained TGVF Adapter, or
production reward/objective behavior. `SC-30` proves exact resume for its tiny
fixture. The earlier technical side result remains invalid because its `SC-40`
prefix collides with the reserved SDPO cell; `RP-10` is the accepted bounded
identity. `RP-11` supplies the bounded real distributed restore/next-step
parity result. None of these results implies production data quality, a paired
provider result, or accepted semantic thresholds.

## Representation-parity work after the bounded framework build

The repository now also contains an executable native Qwen3 representation
trainer, without claiming a completed production run:

- the representation transcript is versioned. Historical prompt/config v1/v2
  paths retain their old empty-pre-reasoning/empty-answer behavior for exact
  provenance, while config v3 selects the new trajectory: original image plus
  unmodified question, fixed pre-tool reasoning, one target-bearing native
  `tgvf_focus_tool` call, latent `D`, post-tool `evidence_description`, and
  final dataset `short_answer`;
- canonical Qwen3 evidence supervision labels only `evidence_description` in
  the final post-tool assistant thinking turn. The fixed pre-tool reasoning,
  tool-call target/JSON, latent response, and final answer remain ignored, so
  the accepted `L_gen` and Matrix-CE mathematics do not change;
- the label boundary uses fast-tokenizer offsets. A local Qwen3 smoke observed
  that sentence-final punctuation may share the following template newline;
  the committed smoke-only v1 real-processor golden fixes its historical
  ownership and expanded visual positions. The separately versioned v2 golden
  now freezes the new transcript, final-answer placement, evidence-only labels,
  target span, and expanded visual positions without rewriting that provenance.
  It binds `qwen3-representation-image-question-v1` under
  `native_representation_prompt_v2`;
- a family-owned canonical-to-model map validates processor-expanded original-
  image and tool-observation placeholder runs, leaving every visual position
  ignored by `L_gen`. Qwen2.5-VL explicitly fails closed pending its own
  transcript/artifact/DeepStack fixture;
- a differentiable live-tensor Qwen injection path avoids the detaching
  content-addressed replay store during representation training;
- the streaming K×K executor atomically swaps main `D` plus all branches,
  blocks original-image keys for evidence queries, checks deterministic
  score/recompute equality, and releases each cell graph before traversing each
  Adapter candidate once;
- the trainer performs explicit global numerator/count normalization,
  Adapter-only AdamW, global gradient clipping, collective-safe unequal-K
  padding, validation, strict FSDP2 checkpointing, and Adapter export. Matrix CE,
  `L_gen`, and the one fixed historical Norm objective are separately logged;
  manifold remains zero;
- the executable internal evaluation reproduces correct/target-only/random/
  wrong-same/wrong-different readout controls, full query matrices, main/branch
  distribution and attention health, plus a concrete Qwen3 native D-only
  teacher-forced/free-continuation path. Its audited pair manifest, extraction
  rule, thresholds, and formal counterfactual evaluation under the v2
  trajectory remain deliberately unset.

Production data/manifests, selected provider and hyperparameter identity,
semantic thresholds, paired-provider comparison, exact legacy-state parity,
and formal counterfactual evaluation remain open under Gate A0. The
representation user-message structure is fixed, but it is not yet bound into a
promoted production run. `RP-11` remains an executor/resume proof with its
historical smoke-only data and prompt, not a trained-quality artifact.

## Model-family boundary

Qwen3-VL-8B-Thinking is the only real-model latent and representation-smoke
family. The current
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
