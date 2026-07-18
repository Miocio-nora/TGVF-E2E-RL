# TGVF End-to-End RL

TGVF End-to-End RL is a clean, independently versioned implementation of
Target-Guided Visual Foveation for a Qwen reasoning policy trained with
end-to-end reinforcement learning. Qwen3-VL-8B-Thinking is the primary target;
`Qwen/Qwen2.5-VL-7B-Instruct` is the required secondary compatibility model.

> **Status:** design and implementation-contract phase. The repository does not
> yet contain an implementation, pinned dependency stack, training run, or
> evaluation result.

The bounded veRL compatibility task is currently a
[proposal](docs/VERL_COMPATIBILITY_SPIKE_PLAN.md). It records candidate commits,
fixtures, hard PASS/FAIL rules, and resource ceilings, but does not authorize an
installation or GPU process.

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
| **target-condition providers** | Supply both contextual-hidden-state and target-token-embedding conditioning behind one interface. |
| **veRL integration** | Provide distributed rollout and optimization infrastructure, with required FSDP2 support. |
| **objective layer** | Keep GRPO and reference-style SDPO mathematics/state separately identifiable. |

The representation phase will reuse only provenance-pinned legacy data and
selected TGVF/DeepStack code. It uses a new native-format pipeline and trains a
new TGVF Adapter checkpoint; the historical checkpoint is a parity/reference
artifact, not a direct initialization.

The policy initializes from the original Qwen reasoning model. There is no
intermediate Stage2-style policy SFT and no Golden policy adapter.

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
- Use upstream veRL as the RL infrastructure and require an FSDP2 execution
  path; select the exact commit, rollout backend, and parallel topology from a
  bounded compatibility spike.
- Freeze the TGVF Adapter for the first policy RL proof.
- Preserve the actual behavior log probability of every policy-sampled token.
- Replay policy, old-policy, and reference likelihoods against the same exact
  rollout-materialized `D` observation for every tool call.
- Treat exact GRPO mathematics as a project artifact, not a framework default.
- Design the first skeleton for SDPO using pinned
  `lasgroup/SDPO@7c457fc1b1f636ae794eb0362ba37d4743b06fbc` as the reference,
  while keeping execution disabled until exact equations, multimodal replay,
  teacher lifecycle, FSDP2, and resume parity pass.
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
- target-span and `Hq` construction details;
- veRL commit, SGLang/vLLM backend, and FSDP2 topology;
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

1. Freeze provenance and approve a bounded compatibility-spike task.
2. Complete the isolated veRL/FSDP2 spike; freeze the Qwen-family,
   dual-provider, trajectory, GRPO/SDPO, teacher-state, and judge-provider seams
   before implementing the framework-binding skeleton.
3. Extract the TGVF Adapter core and establish numerical parity.
4. Implement the native Qwen protocol, strict parser, multi-call runtime, and
   framework-neutral trajectory records.
5. Build the native-format representation pipeline and train a new
   Qwen3-VL-8B-Thinking TGVF Adapter checkpoint. Require a separate
   family-specific artifact and full fixture suite before claiming Qwen2.5-VL
   end-to-end support.
6. Bind the GRPO equations and run a minimal frozen-Adapter policy proof.
7. Validate reference-style SDPO in parallel through text then multimodal
   parity; scale any objective only after reasoning retention, reproducibility,
   and exact resume gates pass.

## Documentation

- [Project task and architectural baseline](docs/PROJECT_TASK.md)
- [Open implementation contracts and promotion gates](docs/OPEN_IMPLEMENTATION_CONTRACTS.md)
- [Proposed veRL compatibility spike](docs/VERL_COMPATIBILITY_SPIKE_PLAN.md)
- [Historical framework-skeleton reference draft](docs/TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md)
  (subordinate to the project task and supersession register)
- [Controlled legacy provenance](docs/LEGACY_REFERENCE.md)
- [Controlled external references](docs/EXTERNAL_REFERENCES.md)
- [Experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Contributor and agent rules](AGENTS.md)

## Relationship to the earlier project

This repository is an independent successor to the earlier
[Miocio-nora/TGVF](https://github.com/Miocio-nora/TGVF) project. The earlier
repository is historical context only: it is not a runtime dependency,
submodule, or source of experiment identity for this implementation.

Before contributing implementation code, read [AGENTS.md](AGENTS.md) and the
accepted interfaces in [docs/PROJECT_TASK.md](docs/PROJECT_TASK.md). No GPU work
may start without a complete `PLANNED` entry in the experiment ledger.
