# TGVF End-to-End RL

TGVF End-to-End RL is a clean, independently versioned implementation of
Target-Guided Visual Foveation for a Qwen reasoning policy trained with
end-to-end reinforcement learning. Qwen3-VL-8B-Thinking is the primary target;
`Qwen/Qwen2.5-VL-7B-Instruct` is the required secondary compatibility model.

> **Status:** the bounded framework and representation-phase trainer are
> implemented on `main`. `RP-11` passes a real-Qwen3/two-rank K=4/GA=4
> continuous-versus-process-teardown resume smoke with Matrix CE, `L_gen`, and
> the fixed historical Norm loss. No production
> data/hyperparameter contract, production representation run, promoted TGVF
> Adapter, policy RL prompt/run, or task evaluation result exists yet.

The bounded [veRL compatibility task](docs/VERL_COMPATIBILITY_SPIKE_PLAN.md)
selected upstream veRL commit
`e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, and the resolved
Python 3.12 compatibility environment. vLLM is the only rollout backend. This
is the accepted compatibility stack, not a production parallel-topology or
training lock.

Current evidence is deliberately small:

- `346` CPU tests pass for the framework, representation-phase contracts, and
  synthetic oracles;
- the representation suite covers the audited retained-JSONL transform,
  same-image sampler, native Qwen3 pipeline with both target-conditioning
  providers, streaming Matrix CE plus `L_gen` plus fixed Norm, a frozen-Qwen
  optimizer step, complete internal-evaluation controls and Qwen3 D-only
  materialization, Adapter-only artifacts, exact CPU next-step resume, strict TOML identity,
  composable-FSDP2 parameter ownership, collective-safe four-versus-five-
  candidate padding, and content-bound distributed checkpoints;
- a separate local-Qwen3 representation golden freezes the real processor/chat
  template, strict Unicode target span, action/evidence transcripts, two visual
  expansions, and evidence labels without loading the 8B weights; an independent
  functional oracle checks TGVF Adapter output and input/parameter gradients in
  FP32 and BF16;
- `SC-20-R6` passes real Qwen3-VL-8B-Thinking TP=2 precomputed-latent transport
  with a native two-call transcript, no tokenizer growth, and no site-package
  patch;
- `SC-30` passes two-rank composable FSDP2 save/teardown/resume with a bitwise-
  identical next step on its tiny deterministic infrastructure fixture;
- `RP-10` passes one real Qwen3-VL-8B-Thinking BF16 representation step on
  physical GPUs 2 and 3 using target-token-embedding conditioning, native main
  `D` plus all three DeepStack branches, same-image Matrix CE plus `L_gen`,
  two-rank FSDP2, validation, content-bound DCP save, and Adapter-only export.
  Its identity-valid executable config is
  `configs/smoke/representation_qwen3_embedding_rp10.toml`; the earlier
  colliding config is retained only in git history.
- `RP-11` passes two uninterrupted updates and a separate step-1
  save/teardown/reconstruct/restore/step-2 lane at K=4 with four accumulation
  microsteps. The comparator reports exact equality for all 104 Adapter
  tensors, recorded model/optimizer/scheduler/sampler/RNG/shard state, and two
  train plus two validation scientific records. The measured continuous steps
  sustain about 1.5 global rows/s on physical GPUs 2 and 3.

The live vLLM path requires the repository plugin and accepted attention split:
`VLLM_PLUGINS=tgvf_qwen3_precomputed`,
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`, and multimodal-encoder attention
`TORCH_SDPA`. See [environment setup](docs/SETUP.md) and the
[experiment ledger](docs/EXPERIMENT_LEDGER.md) for exact identities. These
smokes and CPU fixtures are not policy training, production representation
training, a promoted trained artifact, or semantic-quality evidence. `RP-11`
closes the bounded executor/resume question; its then-used prompt and data are
ledger-only historical facts rather than supported current identities.

For new representation configuration schema v3, the native trajectory is
fixed to `Image + Q → pre reasoning → tool call(target) → D → evidence →
answer`. Here `Q` is the unmodified dataset question, evidence is
`evidence_description`, and answer is `short_answer`. The target never appears
as a separately injected user field; any lexical overlap can only come from the
unmodified question. Only evidence-description tokens are training labels.
The fixed Qwen3 identity is `qwen3-representation-image-question-v1` under
schema `native_representation_prompt_v1`. This first accepted contract also
uses `representation_sample_identity_v1`, `retained_focus_rows_v1`, and
`canonical_evidence_supervision_v1`. The earlier target-bearing smoke text and
its executable fixture/branch are removed; immutable experiment-ledger entries
may still record earlier bounded use as historical fact.

The goal is to train one policy that can decide whether to answer directly or
request target-conditioned visual evidence, consume that evidence, continue
reasoning, and answer—without an intermediate policy SFT stage and without
adding project-specific tokenizer tokens.

## Intended trajectory

```text
original image + question + native Qwen tool schema
        ↓
selected Qwen-VL policy reasoning
        ↓
direct answer OR the enabled profile's visual-tool call
        ↓
TGVF Adapter produces latent D, or crop returns exact RGB pixels
        ↓
the exact latent/image observation is appended as a native multimodal response
        ↓
the same Qwen-VL policy continues reasoning
        ↓
zero or more additional tool calls
        ↓
final answer
```

The policy RL visual-tool profiles are:

```text
tgvf_focus_tool(target: str)
image_zoom_in_tool(bbox_2d: [left, top, right, bottom], label?: str)
tgvf_crop_tool(bbox_2d: [left, top, right, bottom], target: str)
```

The third call is atomic: it crops first and then generates target-conditioned
`D` from that exact crop. The three accepted v1 system prompts, shared user
prompt, schemas, and successful response text are fixed in
[TGVF Visual Tool Prompts v1](docs/TGVF_VISUAL_TOOL_PROMPTS_V1.md).

The crop uses half-open integer pixel coordinates over the immutable original
image, clamps to its bounds, and records the exact RGB8 pixels plus rollout-time
processed visual state. Reusing the processed state requires an explicit shared
frozen-vision identity; trainable-vision replay must re-encode the same pixels
and currently fails closed. Repeated mixed calls are a first-class capability:
crop then TGVF and TGVF then crop share one ordered trajectory and one
configurable safety cap. Each assistant action turn contains at most one call.

## System structure

| Component | Responsibility |
|---|---|
| **representation phase** | Train target-specific visual evidence `D` that the base/frozen Qwen language model can causally read. |
| **TGVF Adapter** | Consume target conditioning and original-image visual features; produce main `D` and every D-DeepStack branch. |
| **policy RL phase** | Learn routing, target/bbox generation, native crop and TGVF use, post-observation reasoning, mixed repeated calls, and final answering. |
| **Qwen VLM family adapter** | Isolate Qwen3-VL and Qwen2.5-VL processor, visual, M-RoPE, DeepStack, transcript, and forward differences. |
| **target-condition providers** | Supply contextual-hidden-state or target-token-embedding conditioning through one typed interface; every run selects one explicitly with no default. |
| **veRL integration** | Provide distributed rollout and optimization infrastructure, with required FSDP2 support. |
| **objective layer** | Keep GRPO and reference-style SDPO mathematics/state separately identifiable. |

The representation phase can read only provenance-pinned legacy data through
the new audited transform and native-format pipeline. A production run must
train a new TGVF Adapter checkpoint; the historical checkpoint is a
parity/reference artifact, not a direct initialization. The current data audit
is evidence, not an accepted training manifest: seven resolved image paths
overlap across the candidate splits and no row is silently removed.

The policy initializes from the original Qwen reasoning model. There is no
intermediate policy SFT and no Golden policy adapter.

## Fixed design decisions

- Preserve the TGVF model structure and DeepStack path.
- Use Qwen's native tool, tool-response, thinking, and vision tokens through its
  native chat template.
- Never resize the tokenizer or add protocol-specific embedding/lm-head rows.
- Support ordered mixtures of `tgvf_focus_tool` and `image_zoom_in_tool` calls
  in one trajectory under a shared safety cap.
- Start policy RL from the original Qwen reasoning policy.
- Use Qwen3-VL-8B-Thinking first and keep
  `Qwen/Qwen2.5-VL-7B-Instruct` support behind a tested family adapter;
  representation checkpoints remain family/model-specific.
- Implement both contextual-hidden-state and target-token-embedding providers.
- Use upstream veRL as the RL infrastructure, vLLM as the only rollout backend,
  and require an FSDP2 execution path. The exact compatibility commit and
  environment are selected; production sharding, placement, and parallel
  topology remain evidence-based.
- Freeze the TGVF Adapter for the first policy RL proof.
- Preserve the actual behavior log probability of every policy-sampled token.
- Replay policy, old-policy, and reference likelihoods against the same exact
  rollout-materialized observation for every call: latent main `D` plus every
  D-DeepStack branch for TGVF, or exact crop pixels plus processed visual state.
- Treat exact GRPO mathematics as a project artifact, not a framework default.
- Implement SDPO as a real implementation track in the current framework goal,
  using pinned
  `lasgroup/SDPO@7c457fc1b1f636ae794eb0362ba37d4743b06fbc` as the reference. Port
  its required behavior onto the selected upstream veRL instead of adopting its
  bundled veRL tree. The initial interfaces remain versioned and evolvable; they
  are not declared frozen substitutes for the implementation.
- Reserve, but do not deploy in the first pilot, a
  `Qwen/Qwen2.5-72B-Instruct` judge provider separate from both the RL reference
  policy and the SDPO self-teacher.
- Treat the existing local Qwen3 path as stable; do not spend time hashing all
  weight shards. Continue hashing tokenizer/chat-template fixtures because they
  define the native protocol.

## Why exact rollout observations matter

For call `i`, the tool observation is conceptually:

```text
D_i = TGVF_Adapter(image_features, Hq_i; adapter_version)
```

`Hq_i` comes from the rollout-time target and policy context. Policy and
reference replay may recompute logits, but they must not regenerate `Hq_i` or
`D_i` using different or updated parameters. Otherwise the probability ratio
and reference KL would compare different observations rather than the recorded
trajectory.

Each call therefore retains the exact main `D`, all D-DeepStack tensors,
positions, multimodal types, masks, cache contract, token ownership, sampling
identity, and behavior log probabilities required for deterministic replay.

## Correctness gates

The project prioritizes:

1. zero tokenizer growth and exact native transcript round trips;
2. numerical output and gradient parity of the extracted TGVF Adapter core;
3. exact target-span extraction and `Hq` identity;
4. target specificity and causal readability of main `D` and every D-DeepStack
   branch;
5. exact template-owned, policy-sampled, environment-owned, and padding masks;
6. rollout/replay logit and logprob parity on identical recorded observations;
7. both condition providers and both required Qwen-family adapters pass their
   declared fixtures;
8. GRPO and SDPO loss, gradient, teacher-state, and resume parity with their
   separately approved equations;
9. retention of the original Qwen policy's high-budget reasoning behavior.

## Current open contracts

The framework skeleton may expose versioned interfaces while these research
choices remain unset:

- Qwen2.5 local/runtime path, family-specific representation artifact, and
  native prompt;
- materialized manifests and dataset/image licenses for the selected pinned v4
  clean-imend train and v3 val-2k validation populations, plus the perceptual
  duplicate policy; recorded exact overlaps are accepted without filtering;
- formal execution of the fixed contextual-hidden-state layer `-1` first run
  and target-token-embedding second run. Both use seed `42`, the accepted v1
  transcript, fixed objective/optimizer/batch/cadence, and historical Golden
  evaluation baseline;
- the ordered same-image and real counterfactual manifests required to enable
  the once-after-training representation internal-evaluation switch;
- numerical official-scorer parity fixtures and exact benchmark-arm decoding
  identities for the already materialized, hashed seven-source CoreDev-2511
  VLMEvalKit slices;
- production actor/reference/rollout placement, FSDP2 sharding, and parallel
  topology; the compatibility veRL commit/environment and vLLM backend are
  fixed;
- policy LoRA or full-parameter scope;
- exact GRPO equations, clipping, KL, and normalization;
- RL data sources and audited manifests;
- reward components, verifiers, and coefficients;
- tool-call safety cap and exploration curriculum;
- exact SDPO equations, feedback/teacher policy, target approximation, and any
  separately defined hybrid or later joint TGVF Adapter update;
- the future RL-reward activation conditions and reward-prompt/calibration
  identity for `Qwen/Qwen2.5-72B-Instruct`; its separate VLMEvalKit
  benchmark-judge service is fixed.

These values must remain explicit `[TBD]` fields until accepted. They must not
be inherited silently from a library default.

## Roadmap

1. **Completed:** record provenance and execute the bounded compatibility task.
2. **Completed for the framework fixture:** implement the versioned
   Qwen-family, dual-provider, trajectory, objective, teacher-state, judge, and
   upstream-veRL boundaries; pass the Qwen3 vLLM transport and tiny FSDP2
   resume smokes. Qwen2.5 remains only a fail-closed main-`D`/family boundary,
   not full DeepStack end-to-end support.
3. **Completed for the framework fixture:** implement the native Qwen protocol,
   strict parser, repeated-call runtime, immutable observations, exact behavior
   records, and framework-neutral trajectory conversion.
4. **Completed for bounded fixtures:** establish an independent functional
   output/gradient oracle for the selected TGVF Adapter equations and freeze a
   real local-Qwen3 native transcript/processor golden. Exact 4096/1152 legacy-
   checkpoint parity remains a promotion gate; `RP-10` supplies the
   identity-valid real-Qwen gradient/FSDP2-save evidence and `RP-11` supplies
   the bounded real distributed next-update resume evidence.
5. **Implemented and smoke-tested, not run on production data:** the native-
   format Qwen3 representation data/pipeline, both provider paths, streaming objective,
   trainer, FSDP2 ownership, configuration, and checkpoint/resume scaffolds.
   Bind the fixed v1 transcript schema/hash and resolve the open production
   data/scientific contracts and semantic evaluation thresholds before
   training a new production Qwen3-VL-8B-Thinking TGVF Adapter. Require a
   separate family-specific artifact and full fixture suite before claiming
   Qwen2.5-VL end-to-end support.
6. Bind the GRPO equations and run a minimal frozen-Adapter policy proof.
7. Freeze the production SDPO equations, feedback/teacher policy, approximation,
   and placement, then validate the implemented path on real two-call
   multimodal replay and FSDP2 teacher-state resume.

## Documentation

- [Project task and architectural baseline](docs/PROJECT_TASK.md)
- [Open implementation contracts and promotion gates](docs/OPEN_IMPLEMENTATION_CONTRACTS.md)
- [Accepted bounded veRL compatibility task](docs/VERL_COMPATIBILITY_SPIKE_PLAN.md)
- [veRL compatibility closeout report](docs/VERL_COMPATIBILITY_REPORT.md)
- [Historical framework-skeleton reference draft](docs/TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md)
  (subordinate to the project task and supersession register)
- [Controlled legacy provenance](docs/LEGACY_REFERENCE.md)
- [Representation parity and open scientific gates](docs/REPRESENTATION_PARITY_INVENTORY.md)
- [Controlled external references](docs/EXTERNAL_REFERENCES.md)
- [VLMEvalKit deployment and direct-baseline example](docs/VLMEVALKIT.md)
- [Experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Compatibility environment setup](docs/SETUP.md)
- [Framework implementation report](docs/IMPLEMENTATION_REPORT.md)
- [Decisions intentionally deferred after the framework build](docs/DEFERRED_DECISIONS.md)
- [Contributor and agent rules](AGENTS.md)

## Relationship to the earlier project

This repository is an independent successor to the earlier
[Miocio-nora/TGVF](https://github.com/Miocio-nora/TGVF) project. The earlier
repository is historical context only: it is not a runtime dependency,
submodule, or source of experiment identity for this implementation.

Before contributing implementation code, read [AGENTS.md](AGENTS.md) and the
accepted interfaces in [docs/PROJECT_TASK.md](docs/PROJECT_TASK.md). No GPU work
may start without a complete `PLANNED` entry in the experiment ledger.
