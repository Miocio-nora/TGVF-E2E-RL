# TGVF End-to-End RL: Project Task

Status: **design baseline updated; framework skeleton direction accepted;
implementation not started**
Recorded: **2026-07-18 JST**
Updated: **2026-07-19 JST**

Unresolved implementation contracts and their promotion gates are tracked in
[`OPEN_IMPLEMENTATION_CONTRACTS.md`](OPEN_IMPLEMENTATION_CONTRACTS.md). An open
field there must not be filled silently from a framework default.

## 1. Objective

Build a new TGVF system in which the original Qwen reasoning policy learns the
complete visual-tool behavior through end-to-end reinforcement learning:

```text
image + question + native tool schema
  -> pre-D reasoning
  -> direct answer OR native TGVF tool call with a generated target
  -> actual TGVF execution and D injection
  -> post-D reasoning
  -> final answer
```

The project must retain the original model's high-budget reasoning ability
while learning when to use TGVF, what target to request, how to read `D`, and
how to use the resulting evidence to answer.

This is a new version and a new repository. It is not a continuation of Golden
Stage2 repair work.

## 2. Core decisions

### 2.1 Preserve the representation phase

The TGVF model structure is unchanged. A representation-learning phase remains
responsible for two independent properties:

1. **Target specificity**: `D(image, target_a)` must be distinguishable from
   `D(image, target_b)` and must encode information relevant to the requested
   target rather than generic full-image content.
2. **Readability**: the Qwen language model must be able to recover the relevant
   evidence from the correct `D`, and correct `D` must causally outperform
   matched wrong/shuffled controls.

The model module trained in this phase and invoked by the tool runtime is named
the **TGVF Adapter**, preserving the established project terminology. The phase,
module, checkpoint, and tool runtime are distinct identities.

The legacy representation dataset and selected representation model/training
logic are eligible for controlled reuse after provenance is frozen. The legacy
training pipeline and serialization are not reusable as the new pipeline: they
must be adapted to the native tool format and the contracts in this repository.

The existing Golden TGVF Adapter checkpoint is a parity and historical
reference only. It must not be used directly as initialization for the new
native-format representation phase. It was trained with Protocol-C target
hidden states, including learned project-specific boundary rows. Native JSON
tool-call serialization changes the context and therefore the `Hq`
distribution. The project must train a new native-format representation
checkpoint and pass the representation gates before policy RL.

### 2.2 Remove Stage2 SFT

There is no supervised Stage2 policy adapter in this version. In particular:

- no short teacher reasoning trajectories;
- no Golden Stage2 LoRA initialization;
- no direct/post-D reasoning replay SFT;
- no logical-microbatch focus/direct SFT composition;
- no protocol-token embedding or lm-head training.

Reasoning preservation is handled by starting from the original policy,
reference-policy regularization, bounded RL updates, and explicit high-budget
reasoning gates.

### 2.3 Use reinforcement learning for the full agent trajectory

The policy RL phase trains the behavior that SFT previously attempted to
impose: route choice, target text, native tool syntax, post-`D` evidence use,
reasoning, and final answer.

GRPO is the first named online objective. SDPO is a candidate consolidation or
self-distillation objective, but its exact paper/repository/equations must be
bound before implementation because `SDPO` is not a unique unambiguous name.
GRPO and SDPO must not be silently mixed into one unidentifiable loss.

GRPO is also not identified by its name alone. Before implementation, the
project must freeze the exact equations and conventions for group standard
deviation, advantage scaling, clipping, KL, per-token versus per-sequence
normalization, and masking. A framework default is never the mathematical
specification.

### 2.4 Use upstream veRL as the RL framework

This project must not build another distributed RL trainer from scratch.
Upstream veRL is the selected base framework and will own the standard
infrastructure:

- distributed policy/reference execution;
- rollout scheduling and batching;
- optimizer, checkpoint, and resume behavior;
- optional asynchronous generation backend;
- metric aggregation.

FSDP2 support is a required implementation capability. The exact parallel
topology, sharding plan, rollout/training placement, and whether additional
parallel dimensions are needed remain evidence-based implementation decisions.

The approved objective mathematics constrains veRL. If its default differs,
the project may use a narrow public pure-tensor extension hook; it must not fork
or rewrite the distributed trainer around private internals.

This repository supplies a narrow custom boundary for latent TGVF execution.
The remaining compatibility spike validates a pinned veRL commit, rollout
backend, Qwen3-VL support, multi-turn latent observations, exact behavior-logprob
retention/replay, FSDP2, and checkpoint/resume. It is not a framework-selection
comparison. veRL is not presumed to be a drop-in solution because TGVF injects
hidden visual embeddings, M-RoPE state, multimodal token types, and D-DeepStack
features rather than a PIL tool image.

## 3. Native Qwen tool protocol

No project-specific tokens may be added. For the pinned local
`Qwen3-VL-8B-Thinking` tokenizer, the existing atomic tokens include:

| Token | Existing ID |
|---|---:|
| `<tool_call>` | 151657 |
| `</tool_call>` | 151658 |
| `<tool_response>` | 151665 |
| `</tool_response>` | 151666 |
| `<think>` | 151667 |
| `</think>` | 151668 |

These are already part of the base tokenizer. The tokenizer length must remain
`151669` for this model snapshot; there is no resize and no new embedding/head
row payload.

The tool is provided through `apply_chat_template(..., tools=[schema])` with a
single function contract equivalent to:

```json
{
  "type": "function",
  "function": {
    "name": "tgvf_focus_tool",
    "description": "Inspect one specific visual region before answering.",
    "parameters": {
      "type": "object",
      "properties": {
        "target": {
          "type": "string",
          "description": "A neutral, visually locatable region description."
        }
      },
      "required": ["target"],
      "additionalProperties": false
    }
  }
}
```

The following is an illustrative rendering of the intended trajectory:

```text
<|im_start|>assistant
<think>
{pre-D reasoning}
</think>

<tool_call>
{"name": "tgvf_focus_tool", "arguments": {"target": "{JSON-escaped target}"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
<|vision_start|>{M native image-pad positions backed by D}<|vision_end|>
</tool_response><|im_end|>
<|im_start|>assistant
<think>
{post-D reasoning}
</think>

{final answer}<|im_end|>
```

This handwritten block is not the canonical byte/token source. The
authoritative transcript must be generated by a pinned processor and pinned
chat template, then stored as a golden token-ID fixture with a template/schema
hash. The test contract is whole-transcript token equality, not visual
similarity to the example above.

For both the initial assistant turn and the assistant turn after the tool
response, Qwen's Thinking template owns the generation prefill
`<|im_start|>assistant\n<think>\n`. Those prefix tokens are environment tokens,
not policy samples and not policy-loss tokens. Sampling starts after that
prefix; the policy must not generate or receive a second opening `<think>`.
Each assistant turn must contain exactly one balanced think span. The action
turn's `</tool_call>` and `<|im_end|>` sampling/stop semantics must be frozen in
the golden fixture rather than inferred by a parser.

The sampled tool call is immutable trajectory data. Preserve the model's exact
token IDs, whitespace, JSON spelling, escapes, closing marker, and termination
tokens. Do not parse a call and then canonicalize or retokenize it for replay.

The message-level input may use `role="tool"`, but Qwen's native template
serializes that observation under a user-framed `<tool_response>` turn. The new
runtime must follow the template exactly; it must not resurrect the historical
`<|im_start|>tool` framing.

Repeated tool calls are a first-class runtime requirement. A trajectory may
alternate between policy reasoning, `tgvf_focus_tool`, and a new `D`
observation more than once before the final answer. A configurable safety cap
greater than one is required; its exact value and stopping policy remain open.

The parser is fail-closed:

- exactly one complete tool-call object per assistant action turn;
- strict `json.loads`;
- exact function name and argument keys;
- non-empty, non-generic target;
- target byte/character offsets mapped back to the original sampled token IDs,
  including JSON escapes; ambiguous boundary-crossing tokens are rejected or
  handled by one predeclared deterministic rule;
- no answer leakage or trailing assistant answer after the tool call;
- explicit configurable maximum tool-call count greater than one;
- malformed calls receive no tool execution.

## 4. End-to-end runtime contract

For each prompt and each sampled group member:

1. Render the original image, question, and neutral native tool schema.
2. Append the template-owned assistant thinking prefix and sample from the
   current policy without a Stage2 adapter or duplicate `<think>` opener.
3. If the assistant answers directly, terminate and score the answer.
4. If it emits a valid `tgvf_focus_tool` call:
   - locate the exact JSON target value span in the raw sampled token stream;
   - retain all sampled token IDs and actual behavior log probabilities;
   - obtain target hidden states `Hq` using a separately frozen contract for
     value-span inclusion, JSON escapes, hidden layer, and token-time alignment;
   - execute the fixed TGVF Adapter on original-image pre-merge and
     DeepStack features;
   - construct main `D`, all D-DeepStack branches, native visual positions,
     multimodal token types, and mask state;
   - materialize and retain the exact main `D`, every D-DeepStack tensor,
     visual layout, positions, multimodal types, mask, and cache contract;
   - append the native tool-response turn and its template-owned next-assistant
     thinking prefix.
5. Continue the same trajectory to post-`D` reasoning. The policy may answer or
   emit another valid `tgvf_focus_tool` call; repeat step 4 until a final answer,
   a malformed action, or the configured safety limit terminates the loop.
6. Record the complete action mask, tool/environment spans, rewards, stop
   causes, and all identity fields needed for mathematically identical replay.
7. Replay policy/reference log probabilities only on policy-generated tokens
   from every assistant turn; template prefill and tool-observation tokens are
   environment output and receive no policy loss.
8. For every tool call, policy, old-policy, and frozen-reference replay all
   consume that call's same rollout-recorded `D` observation. They may recompute
   logits on the recorded trajectory, but they never regenerate `Hq`, main `D`,
   or D-DeepStack from their own or updated parameters.

The minimal framework-facing trajectory record must retain:

- exact prompt, action, observation, continuation, and final token IDs;
- policy-token/loss masks and every tool call's target span;
- actual behavior log probabilities for every sampled pre-reasoning, tool-call,
  post-reasoning, and answer token;
- rollout policy/checkpoint version, asynchronous staleness, sampling
  backend/version, seed/RNG, temperature, top-p, top-k, min-p, penalties, logit
  processors, and raw-versus-transformed logprob convention;
- for every tool call, immutable rollout-time main `D` and every D-DeepStack
  tensor or an immutable artifact handle to those exact tensors;
- for every tool call, visual grid, fake-image span, M-RoPE positions, and
  multimodal token types;
- per-call main and D-DeepStack state plus original-image mask scope;
- exact template-owned/policy-owned/environment-owned token masks;
- reward components and parse/termination metadata;
- model, processor, TGVF-Adapter-checkpoint, code, and prompt identities.

The custom adapter surface should remain narrow, conceptually:

```text
sample_group
execute_tgvf_tool
replay_policy
replay_reference
score_trajectory
```

If SDPO is adopted, an additional feedback/teacher replay surface may be added
only after its token-alignment and context contracts are specified.

## 5. Initial parameter policy

The initial policy is the unmodified original Qwen reasoning model plus a
newly defined policy adapter scope. The following are fixed or unresolved:

- Frozen reference: exact original Qwen model and prompt/tool schema.
- Policy initialization: exact original Qwen model; no Stage2 LoRA.
- TGVF Adapter: a newly trained native-format TGVF Adapter checkpoint, initially
  frozen during the first RL proof. The legacy checkpoint is not a direct
  initialization.
- Policy trainable scope: unresolved; LoRA is expected, but module/rank/dropout
  must be selected and recorded rather than inherited from Golden.
- Original vision tower and frozen Qwen visual mergers: frozen by default.

Keeping the TGVF Adapter parameters frozen for the first RL proof
reduces one source of drift, but it does **not** make the tool environment
policy-independent: the sampled target and `Hq` still come from a changing
policy. Every observation is therefore versioned and materialized at rollout
time. Joint RL updates to TGVF are a later named experiment and must retain the
two representation obligations with auxiliary gates and a newly specified
observation/replay contract.

## 6. Reward contract

The reward is decomposed and logged; no opaque single judge score is allowed.
Candidate components are:

- benchmark-correct final answer score using a final-answer-only official or
  benchmark-compatible scorer;
- valid native tool syntax and successful execution;
- target validity, locality, non-genericity, and answer-leak penalty;
- target/evidence utility or sampled correct-`D` causal advantage;
- malformed protocol, redundant duplicate call, runtime error, loop, cap-hit,
  and non-termination penalties;
- tool/token/latency cost so the policy does not call TGVF indiscriminately;
- KL/reference regularization against the original Qwen policy.

Do not reward longer reasoning. Health metrics and correctness are separate.
Judge-model rewards, if used, require calibration against human-audited
examples and must never replace executable answer/tool checks.

## 7. Representation gates

Before policy RL can claim a usable tool channel, the native-format
TGVF Adapter must pass all three levels:

1. **Readout**: verified evidence has lower NLL under correct `D` than target
   only, matched wrong-same-image, matched wrong-image, shuffled, and random
   controls.
2. **Causal value flip**: on counterfactual image pairs differing only in a
   local value, swapping correct `D` must flip evidence/answer log odds in a
   fresh context that removes original-image tokens, original-image DeepStack,
   and every pre-`D` text KV state that has already attended to the image.
3. **Free continuation**: post-`D` generation must state evidence and answer
   consistently with the swapped value, with healthy termination.

Main `D` and every D-DeepStack branch are always swapped together. Zero `D` is
an out-of-distribution health control, not the primary semantic negative.
Correct and wrong/shuffled controls must match length, visual grid, M-RoPE,
dtype, main/branch shapes, and calibrated norm/statistical ranges.

## 8. Reasoning-preservation gates

The RL version is promoted only if it retains the original model's behavior at
high token budgets, not merely if its average output is longer. Required paired
metrics include:

- original-correct answer retention;
- reasoning-heavy benchmark accuracy;
- output/reasoning token mean, median, maximum, and distribution;
- cap-hit, think-closure, loop/repetition, and natural-stop rates;
- direct versus tool-route accuracy;
- tool trigger and valid-target rates;
- correct-`D` versus counterfactual controls;
- KL and policy-drift statistics on direct reasoning prompts.

The original base, policy-direct, policy-tool, and counterfactual-`D` rows must
share exact sample IDs and scoring identity.

## 9. veRL compatibility gate

veRL is selected. A **compatibility spike** is a bounded integration proof that
selects a usable veRL commit and concrete adapter/backend configuration; it is
not a framework bake-off and it is not a training run. The spike must use an
official maintained version and answer:

- Does the selected veRL commit support Qwen VLM policy/reference models?
- Can it run multi-turn dynamic tools without converting `D` to a PIL image?
- Can a custom model forward receive native visual embedding spans, M-RoPE,
  multimodal token types, and D-DeepStack?
- Does it preserve those latent tensors directly? A framework that only accepts
  token/PIL observations, re-encodes `D` as an image, or cannot carry the exact
  mask/position/DeepStack state fails closed.
- Can sampled behavior log probabilities be retained and replayed exactly?
- Can rollout and update be batched without serial per-trajectory replay?
- Can FSDP2 load, update, checkpoint, and resume the intended policy scope, and
  what parallel topology is justified by measured memory and throughput?
- Are GRPO standard-deviation convention, clipping, KL, and token/sequence
  normalization configurable and identical to the approved equations?
- Is the intended SDPO implementation supported, or can its pure tensor loss
  be added without replacing private trainer internals?
- Are checkpoint/resume and dependency versions reproducible locally?

The veRL family decision is fixed. The exact commit, dependency environment,
rollout backend, FSDP2 topology, and adapter surface remain open until the spike
is approved and recorded. No package installation or pin is authorized merely
by selecting veRL.

## 10. Migration boundary

### Allowed exact extraction

Only the selected TGVF Adapter structure and minimal mathematical helpers may
be copied after the old repository is frozen and numerical parity fixtures are
written. The current candidate is the 8B-specific
`TGVFv2BidirectionalDDeepStack` Adapter.

### Allowed thin reimplementation

- Qwen vision feature taps;
- source visual geometry;
- frozen-merger finalization;
- native visual-span/M-RoPE/multimodal-token construction;
- DeepStack payload construction and explicit mask policy;
- target-span hidden-state capture;
- representation readout diagnostics.

These are rewritten behind new neutral interfaces; whole legacy engines are
not copied.

### Allowed controlled reuse for native representation training

- the legacy representation dataset, after its manifest, provenance, license,
  transforms, and sample identities are audited;
- selected representation losses, diagnostics, and model-training logic after
  their source identities and semantic changes are recorded;
- preprocessing behavior that passes parity and does not reintroduce a legacy
  protocol or tokenizer change.

The new representation training pipeline, native transcript construction, data
schema, configuration, checkpoint manifest, and launcher are implemented in
this repository. The historical pipeline is not ported as the new pipeline.
The historical TGVF Adapter checkpoint is not a new-training initialization.

### Forbidden migration

- all legacy protocol identities and implementations, including Protocol C,
  Protocol D, and Protocol E tokens, renderers, parsers, token-row restoration,
  and tokenizer-resize code;
- all Stage2 SFT trainers, adapters, checkpoints, data mixtures, replay data,
  and loss normalization code;
- all legacy Stage3 trainer, rollout engine, replay, policy-reference, reward,
  and executor modules;
- the historical representation training pipeline, launchers, serialization,
  and checkpoint-resume state as a new runtime pipeline;
- legacy force/softforce modes as training identities;
- runtime imports, symlinks, or submodules pointing to the old repository.

Historical reward ideas and tests may inform a new specification, but no old
Stage3 implementation is assumed correct.

## 11. Planned repository shape

No source directories are created until their interfaces are approved. The
intended shape is:

```text
tgvf-e2e-rl/
  AGENTS.md
  README.md
  pyproject.toml                 # after veRL compatibility/dependency approval
  docs/
    PROJECT_TASK.md
    LEGACY_REFERENCE.md
    EXPERIMENT_LEDGER.md
    OPEN_IMPLEMENTATION_CONTRACTS.md
    TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md
  src/tgvf_rl/
    representation/             # extracted TGVF Adapter and training path
    qwen/                       # native VLM/tool/D adapter
    protocol/                   # schema, strict parser, transcript identity
    environment/                # TGVF tool execution
    trajectories/               # framework-neutral rollout records
    rewards/                    # decomposed executable rewards
    framework/                  # narrow veRL adapter
    evaluation/                 # identity-safe evaluation/scoring
  tests/
    parity/
    protocol/
    representation/
    rollout/
    objectives/
```

## 12. Work phases and promotion gates

### Phase 0: freeze provenance

- Archive the current old-repository dirty worktree under a user-approved
  commit/tag or immutable patch bundle.
- Pin the base model/processor snapshot and the historical representation
  checkpoint only as a parity/reference identity.
- Finalize the exact extraction whitelist and file hashes.

Gate: every imported idea or symbol has immutable provenance.

### Phase 1: representation-core extraction

- Extract only the selected TGVF classes/helpers.
- Export a minimal TGVF Adapter artifact without optimizer, scheduler, or
  protocol-token rows.
- Establish deterministic synthetic and real-row parity with the pinned old
  TGVF Adapter.

Gate: for fixed tensor inputs (`Hq`, pre-merge vision features, every DeepStack
branch, mask/config), main `D`, every D-DeepStack output, and required gradients
match the pinned Adapter. This gate tests Adapter mathematics, not legacy
Protocol-C serialization.

### Phase 2: native protocol and runtime

- Implement native tool schema, strict parser, target-span mapping, and native
  tool-response `D` injection.
- Support repeated `tgvf_focus_tool` calls with a configurable safety cap
  greater than one and per-call immutable observation records.
- Prove no tokenizer resize, exact template-generated transcript/token round
  trip, correct ownership masks, and no duplicate opening `<think>` in either
  assistant turn.
- Measure untrained base prompt-trigger behavior without using it as SFT data.

Gate: native direct/tool trajectories execute deterministically and expose all
required replay state.

### Phase 3: native representation training and compatibility

- Compare old Protocol-C and native-tool `Hq`/`D` behavior.
- Build the new native-format representation data/training/checkpoint pipeline.
- Train a new native-format TGVF Adapter checkpoint; do not initialize it
  directly from the historical TGVF Adapter checkpoint.
- Run target-sensitivity, readout, counterfactual flip, and free-continuation
  gates.

Gate: target-specific, readable native `D`; no claim based only on formatting
or output-length change.

### Phase 4: veRL compatibility spike

- After the spike task is approved, test and pin one upstream veRL commit in an
  isolated optional environment.
- Validate candidate rollout backends and an FSDP2 execution/checkpoint path;
  choose the concrete parallel topology from observed correctness, memory, and
  throughput.
- Adapt the narrow trajectory/tool interfaces.
- Bind the exact GRPO equations before invoking a framework loss.
- Verify same-policy-version behavior/replay logits and logprobs on the exact
  recorded observation, deterministic forward/dropout state, token masks,
  staleness policy, GRPO loss, gradients, checkpoint, and resume parity on a
  tiny CPU/GPU fixture.

Gate: framework infrastructure does not change transcript, D construction, or
approved RL mathematics.

### Phase 5: minimal GRPO proof

- Use a fixed, audited prompt/sample manifest.
- Train only the approved policy scope with the TGVF Adapter frozen.
- Evaluate direct/tool exploration, reward decomposition, D use, and reasoning
  retention at a checkpoint ladder rather than only at the endpoint.

Gate: final reward improves without losing high-budget original reasoning or
learning to ignore/misuse `D`.

### Phase 6: scale and optional SDPO

- Scale rollout/replay only after Phase 5 correctness.
- Bind and test the exact SDPO objective separately.
- Consider constrained joint TGVF updates only as a named later experiment.

Gate: mathematical identity, throughput, memory, and resume behavior are all
recorded and reproducible.

## 13. Open decisions requiring confirmation

1. Compact code/config labels for the representation phase and policy RL phase;
   the formal prose names remain in force until replacements are accepted.
2. Exact legacy representation dataset/code reuse whitelist and the new
   native-format training interface.
3. Whether the TGVF Adapter is frozen for all policy RL or only for
   the first proof; the first proof is fixed to frozen.
4. Policy LoRA/full-parameter scope and reference/KL contract.
5. Pinned veRL commit, rollout backend, dependency environment, and concrete
   FSDP2/parallel topology.
6. Exact meaning and source implementation of SDPO.
7. Tool-call safety cap greater than one, prompt wording, and initial
   exploration curriculum.
8. Original-image visibility and DeepStack mask scope after each `D`.
9. Training population, reward benchmarks, and held-out evaluation manifests.
10. Judge models, if any, and their calibrated role in target/evidence rewards.

No implementation should silently decide any item above.

## 14. Definition of project success

The project succeeds only when one policy, initialized from the original Qwen
reasoning model and trained without Stage2 SFT or new protocol tokens:

- naturally chooses between direct answering and one or more native TGVF tool
  calls when useful;
- produces valid, useful, non-leaking targets;
- receives and causally uses target-specific `D`;
- improves task accuracy net of tool cost;
- retains original high-budget reasoning accuracy and healthy termination;
- trains and resumes reproducibly through a maintained RL framework;
- is supported by exact experiment identities and paired rows, not mechanism
  stories inferred from token length or trigger rate alone.
