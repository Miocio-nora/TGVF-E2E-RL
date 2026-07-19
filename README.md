# TGVF End-to-End RL

TGVF End-to-End RL is a clean, independently versioned implementation of
Target-Guided Visual Foveation for a Qwen reasoning policy trained with
end-to-end reinforcement learning. Qwen3-VL-8B-Thinking is the primary target;
`Qwen/Qwen2.5-VL-7B-Instruct` is the required secondary compatibility model.

> **Status:** the bounded framework and representation-phase execution scaffold
> are implemented on `main`, and the corrected `RP-10` synthetic-data
> real-Qwen3/two-rank representation optimizer smoke has passed. The earlier
> colliding run ID is retained as invalid. No production
> data/prompt/hyperparameter contract, production representation run, promoted
> TGVF Adapter, policy RL run, or task evaluation result exists yet.

The bounded [veRL compatibility task](docs/VERL_COMPATIBILITY_SPIKE_PLAN.md)
selected upstream veRL commit
`e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, and the resolved
Python 3.12 compatibility environment. vLLM is the only rollout backend. This
is the accepted compatibility stack, not a production parallel-topology or
training lock.

Current evidence is deliberately small:

- `295` CPU tests pass for the framework, representation-phase contracts, and
  synthetic oracles;
- the representation suite covers the audited retained-JSONL transform,
  same-image sampler, native Qwen3 pipeline with both target-conditioning
  providers, streaming Matrix CE plus `L_gen`, a frozen-Qwen optimizer step,
  Adapter-only artifacts, exact CPU next-step resume, strict TOML identity,
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

The live vLLM path requires the repository plugin and accepted attention split:
`VLLM_PLUGINS=tgvf_qwen3_precomputed`,
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`, and multimodal-encoder attention
`TORCH_SDPA`. See [environment setup](docs/SETUP.md) and the
[experiment ledger](docs/EXPERIMENT_LEDGER.md) for exact identities. These
smokes and CPU fixtures are not policy training, production representation
training, representation restore/next-step-resume evidence, a promoted trained
artifact, or production-objective evidence. `RP-10` is deliberately only one
synthetic-data optimizer step and makes no quality or capacity claim.

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
direct answer OR tgvf_focus_tool(target)
        ↓
TGVF Adapter produces target-conditioned main D + D-DeepStack
        ↓
D is appended as a native multimodal tool response
        ↓
the same Qwen-VL policy continues reasoning
        ↓
zero or more additional tool calls
        ↓
final answer
```

The canonical tool is:

```text
tgvf_focus_tool(target: str)
```

Repeated calls are a first-class capability. The exact safety cap is
configurable and remains to be frozen. Each assistant action turn contains at
most one complete tool-call object, while a trajectory may contain multiple
action/response turns.

## System structure

| Component | Responsibility |
|---|---|
| **representation phase** | Train target-specific visual evidence `D` that the base/frozen Qwen language model can causally read. |
| **TGVF Adapter** | Consume target conditioning and original-image visual features; produce main `D` and every D-DeepStack branch. |
| **policy RL phase** | Learn routing, target generation, native tool use, post-`D` reasoning, repeated calls, and final answering. |
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
- Support multiple `tgvf_focus_tool` calls in one trajectory.
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
  rollout-materialized `D` observation for every tool call.
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
- production representation prompt wording, contextual hidden-layer choice,
  and real-Qwen `Hq`/readout validation; the Qwen3 native span/processor smoke
  contract and both provider code paths are implemented;
- accepted train/validation manifests, resolution of the seven exact resolved-
  path overlaps, dataset/image licenses, and perceptual duplicate policy;
- representation initialization seed, Matrix-CE/`L_gen` weights, optimizer,
  scheduler, precision, accumulation, clipping, validation cadence, and
  promotion thresholds;
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
- the future activation conditions and exact service/prompt/calibration identity
  for the reserved `Qwen/Qwen2.5-72B-Instruct` judge.

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
   identity-valid real-Qwen gradient/FSDP2-save evidence.
5. **Implemented and smoke-tested, not run on production data:** the native-
   format Qwen3 representation data/pipeline, both provider paths, streaming objective,
   trainer, FSDP2 ownership, configuration, and checkpoint/resume scaffolds.
   Resolve the open data/scientific contracts, real distributed restore gate,
   and paired-provider experiment before training a new production
   Qwen3-VL-8B-Thinking TGVF Adapter. Require a separate family-specific
   artifact and full fixture suite before claiming Qwen2.5-VL end-to-end
   support.
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
