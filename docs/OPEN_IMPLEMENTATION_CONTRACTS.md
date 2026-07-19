# Open Implementation Contracts

Status: **bounded framework implementation complete; production gate checklist active**
Recorded: **2026-07-19 JST**

## 1. Purpose

This file records implementation contracts that are still missing or only
partially specified. It separates framework-skeleton work from decisions that
must be frozen before rollout, an optimizer step, or GPU execution.

Rules:

- `[TBD]` is intentional. Code must not replace it silently with a veRL,
  Transformers, vLLM, SGLang, PyTorch, or prior-project default.
- Closing an item requires a value, an acceptance artifact/test, and a dated
  update to the authoritative task or interface in `docs/PROJECT_TASK.md`.
- Except for the explicit I8H-20260719 acceptance below, this checklist does not
  itself authorize implementation, dependency installation, legacy inspection,
  or GPU execution.
- Exact data, reward coefficients, and prompt wording may be late-bound. Token
  ownership, actual behavior log probabilities, observation identity,
  deterministic replay, and objective equations may not.

Status vocabulary:

- `FIXED`: accepted direction; implementation may rely on it.
- `OPEN_BLOCKING`: must close before the named gate.
- `OPEN_CONFIGURABLE`: the skeleton may expose an explicit unset/configurable
  field, but an experiment must bind it before use.
- `DEFERRED`: deliberately outside the first proof.
- `SUPERSEDED`: older specification text that must not drive implementation.

Every closed item should record:

```text
decision/value:
allowed alternatives:
required artifact or test:
accepted by/date:
supersedes:
```

### 1.1 Authoritative I8H-20260719 acceptance

Accepted by the user on **2026-07-19 JST**, this is one bounded eight-hour
framework implementation authorization. The full task, accepted source roots,
and public interface list are authoritative in
[`PROJECT_TASK.md` §0](PROJECT_TASK.md#0-authoritative-8-hour-implementation-acceptance).

This acceptance fixes:

- upstream veRL plus **vLLM only**, with FSDP2 required; no SGLang execution or
  stable-version comparison cell;
- physical GPU indices **2 and 3 only**, after a complete `PLANNED` ledger row;
- isolated dependency installation and necessary public model-weight/artifact
  downloads;
- deterministic synthetic fixtures and small integration smokes only, with no
  real-data or production training run;
- executable reference-style pure SDPO, including teacher construction,
  alignment, exact-observation replay, distillation loss/targets, teacher
  lifecycle, FSDP2 state, checkpoint, and resume. A frozen seam or interface-
  only placeholder is not completion.

It closes `SK-01` and accepts the interface shapes in Project Task §0.2 for
implementation. `SK-02`, `SK-04`, `SK-05`, `SK-06`, `SK-09`, `SK-10`, `SK-11`,
and `SK-12` now track implementation/evidence quality rather than permission to
create the accepted source files. `SK-03` is closed for the exact files used by
this bounded build; the same provenance rule continues to block every new
legacy-derived file until its own entry exists.

The bounded implementation is now complete. The accepted compatibility stack
is upstream veRL
`e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, Torch
`2.9.0+cu128`, Transformers `4.57.6`, and CPython `3.12.3`. `346` CPU tests,
the real-Qwen3 `SC-20-R6` TP=2 latent-transport smoke, and the two-rank `SC-30`
bitwise FSDP2 resume smoke passed. One real-Qwen3 representation execution
completed technically under a colliding `SC-40` prefix and is retained as
invalid because that ID was already reserved for SDPO CPU parity. Corrected
representation run `RP-10` independently passed the same bounded question.
`RP-11` additionally passed real-Qwen3 K=4/GA=4 continuous versus clean
process-teardown/restore next-update parity with the fixed Norm-aware objective.
The CPU baseline includes the bounded representation data/native pipeline,
both conditioning providers, streaming
Matrix CE plus `L_gen`, trainer, FSDP2 ownership, artifact/checkpoint/config,
real local-processor golden, and functional Adapter reference-parity fixtures.
These results close framework construction and bounded implementation evidence
only; later rollout, production representation, objective, model-family, and
scale gates below remain authoritative.

No data or Adapter artifact is accepted for production use. Dataset/image
licenses, the seven resolved-path split overlaps, final representation prompt
and hyperparameters, reward values, production GRPO/SDPO mathematics and
configuration, any hybrid objective, long training, and the 72B judge remain
open and fail closed. A synthetic optimizer smoke may run only after its exact
test-contract equations and pure-tensor oracle are recorded.

Where older text in this file or `VERL_COMPATIBILITY_SPIKE_PLAN.md` conflicts,
this section and Project Task §0 take precedence. The earlier proposal status,
full mandatory cell matrix/overall hard-PASS requirement, SGLang/stable cells,
`SDPO_STATIC_SEAM_READY` stopping point, per-cell request for another user
reply, and separate post-report skeleton approval are superseded. Technical
criteria still apply to any cell that claims the corresponding evidence;
exact-identity records and ledger requirements remain in force. Unrun original
matrix items remain promotion gates rather than implicit passes.

## 2. Fixed directions

- [x] `FIXED` — Upstream veRL is the policy-RL infrastructure.
- [x] `FIXED` — The exact accepted compatibility pin is upstream veRL
  `e003163181731412595257a72ec173071efb125f` with vLLM `0.12.0`; this is not a
  production parallel-topology decision.
- [x] `FIXED` — FSDP2 support is required. Exact topology is evidence-based.
- [x] `FIXED` — Qwen3-VL-8B-Thinking at the accepted stable local path is the
  primary policy/reference target; `Qwen/Qwen2.5-VL-7B-Instruct` is the required
  secondary compatibility model.
- [x] `FIXED` — Contextual-hidden-state and target-token-embedding conditioning
  providers are both required capabilities. A run selects one as part of its
  experiment identity.
- [x] `FIXED` — Provider types are not mixed in one representation run. Their
  optional comparison uses separate paired runs with identical sample/group
  order, initialization, batch plan, and seed; a paired real-GPU comparison is
  not required before one selected provider's run.
- [x] `FIXED` — Same-image Matrix CE is a required target-specificity objective;
  ordinary independent shuffle is not a valid substitute for same-image
  multi-target grouping.
- [x] `FIXED` — The representation baseline includes both Matrix CE and
  `L_gen`, separately logged at weights `1.0` and `1.0` for the accepted initial
  old-configuration comparison.
- [x] `FIXED` — The manifold-loss optimizer contribution is exactly zero.
- [x] `FIXED` — Norm loss uses the one accepted historical formula, detached
  source visual tensors, fixed main/branch reduction, and weight `0.1`; no norm
  mode or target selector is exposed.
- [x] `FIXED` — Pinned historical internal representation tests and metrics must
  be reproduced as exact parity fixtures or documented native-protocol
  adaptations; omissions require an explicit decision. This requirement does
  not mark the still-open AD-13 inventory as complete.
- [x] `FIXED` — The native function name is `tgvf_focus_tool`.
- [x] `FIXED` — A trajectory supports zero or more tool calls; the safety cap is
  configurable and greater than one.
- [x] `FIXED` — Qwen native tool/thinking/vision tokens are used without
  tokenizer growth.
- [x] `FIXED` — There is no intermediate policy SFT or Golden policy adapter.
- [x] `FIXED` — The legacy representation dataset and selected TGVF
  Adapter/DeepStack training code may be reused under provenance and parity
  rules.
- [x] `FIXED` — The representation pipeline and native serialization are new.
- [x] `FIXED` — The historical TGVF Adapter checkpoint is parity/reference only,
  not a direct initialization for the new representation phase.
- [x] `FIXED` — A new native-format TGVF Adapter checkpoint is trained before
  the first policy RL proof.
- [x] `FIXED` — The TGVF Adapter is frozen for the first policy RL proof.
- [x] `FIXED` — For each tool call, policy, old-policy, and reference replay
  consume the same rollout-materialized `D` observation.
- [x] `FIXED` — SDPO compatibility is part of the initial skeleton. Its reference
  repository is `lasgroup/SDPO` at
  `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`; its bundled veRL tree is not the
  selected production framework.
- [x] `FIXED` — Pure GRPO, reference-style pure SDPO, and any future hybrid are
  distinct objective identities. The reference SDPO implementation replaces
  the policy loss and does not define a GRPO-plus-SDPO sum.
- [x] `FIXED` — An external answer judge, frozen RL reference, and SDPO
  self-teacher are separate roles. The `Qwen/Qwen2.5-72B-Instruct` provider is
  reserved but disabled for the first pilot.
- [x] `FIXED` — Full weight-directory hashing is not required for the stable
  local Qwen3 path. Exact tokenizer/chat-template transcript fixtures and hashes
  remain mandatory protocol artifacts.
- [x] `FIXED` — Prose uses `representation phase`, `policy RL phase`, and `TGVF
  Adapter`. Current component names do not use Stage1/Stage2/Stage3.

## 3. Superseded specification register

- [x] `SUPERSEDED C-01` — “The legacy Stage1 pipeline remains unchanged.”
  Replacement: selected data/code may be adapted, but the native-format
  representation pipeline is new.
- [x] `SUPERSEDED C-02` — “The old trained checkpoint initializes policy RL.”
  Replacement: train a new native-format TGVF Adapter checkpoint; the old one
  is reference/parity only.
- [x] `SUPERSEDED C-03` — “No RL framework has been selected.” Replacement:
  veRL is selected, and the bounded compatibility task fixed its exact commit,
  dependency environment, vLLM-only backend, and public adapter surface.
  Production placement and topology remain open.
- [x] `SUPERSEDED C-04` — Function name `tgvf_focus`. Replacement:
  `tgvf_focus_tool`.
- [x] `SUPERSEDED C-05` — A one-call trajectory model. Replacement: at most one
  complete call object per assistant action turn, but repeated action/response
  turns per trajectory.
- [x] `SUPERSEDED C-06` — `differentiable_recompute` as an immediately available
  policy replay mode. Replacement: first proof freezes the TGVF Adapter; any
  joint-update algorithm is separately specified and must preserve immutable
  rollout observations.
- [x] `SUPERSEDED C-07` — Stage1/Stage3 as current component names. Replacement:
  representation phase, policy RL phase, and TGVF Adapter.
- [x] `SUPERSEDED C-08` — “The veRL spike remains an unaccepted proposal.”
  Replacement: I8H-20260719 authorizes the bounded vLLM-only compatibility and
  implementation task.
- [x] `SUPERSEDED C-09` — “Run a bounded SGLang comparison and stable veRL
  qualification.” Replacement: vLLM is the only backend and the exact upstream
  veRL pin is selected from vLLM evidence.
- [x] `SUPERSEDED C-10` — “SDPO work ends at a frozen/static seam.” Replacement:
  implement and test real reference-style pure SDPO during I8H-20260719;
  production configuration remains a later gate.
- [x] `SUPERSEDED C-11` — “Every bounded cell needs another interactive user
  reply.” Replacement: the I8H authorization covers in-scope rows after the
  agent completes their ledger and gates; exceeding scope still requires new
  authority.

## 4. Gate S0 — Before framework-skeleton implementation

The exact reward, RL dataset, and prompt text do not block this gate. Their
interfaces must be versioned and fail closed while unset.

- [x] `FIXED SK-01` — The exact I8H-20260719 task and file/interface surface in
  `docs/PROJECT_TASK.md` §0 are accepted.
  - Value: `I8H-20260719`
  - Minimum modules: TGVF Adapter/representation, Qwen adapter, protocol,
    both target-condition providers, multi-call tool runtime,
    framework-neutral trajectory records, reward/judge interfaces,
    GRPO/SDPO objective and teacher-state boundaries, veRL adapter, evaluation.
- [x] `FIXED SK-02` — Ownership between veRL and this repository is explicit.
  - veRL owns: distributed workers, scheduling, optimizer execution,
    checkpoint/resume, metric aggregation.
  - This repository owns: native transcript identity, strict parser, latent
    tool execution, per-call observation records, rewards/verifiers, exact
    objective extension/parity tests.
  - Accepted public extension points: `AgentLoopOutput`, a custom public
    `AgentLoopManager`, `DataProto`, `register_policy_loss`,
    `FSDPEngineConfig`, and `CheckpointHandler`, plus vLLM's general plugin,
    model, and multimodal processor registries. Private worker/trainer patches
    remain forbidden.
- [x] `FIXED SK-03` — The bounded framework reuse inventory and port records are
  frozen in `docs/LEGACY_REFERENCE.md`: the adapted TGVF core and clean-subtree
  DeepStack semantics are tied to exact source paths and SHA256 identities.
  This closes only the files already used by the framework. Any additional
  representation code or data still requires a new provenance entry, and the
  complete training inventory remains blocked under Gate A0.
- [x] `FIXED SK-04` — The framework TGVF Adapter boundary is typed and
  versioned in `src/tgvf_rl/representation/`: target conditioning,
  original-image pre-merge features, model-supported branches and explicit
  geometry enter; main `D`, ordered D-DeepStack branches, and metadata leave.
  Real `Hq`, merger/artifact identity, and representation-training values
  remain Gate A0 items rather than hidden framework defaults.
- [x] `FIXED SK-05` — A versioned framework-neutral trajectory
  record interface with reserved fields for tokens, ownership masks, behavior
  log probabilities, per-call observations, identities, rewards, and stops.
  It must also preserve typed environment/judge feedback, group identity,
  demonstration provenance, teacher-context identity, per-turn alignment, and
  exact-observation teacher replay handles without making GRPO depend on SDPO.
  - Implemented in `src/tgvf_rl/trajectories/`, `contracts/`, and
    `observations/`; content-addressed records are validated before veRL
    conversion.
- [x] `FIXED SK-06` — Configuration/identity plumbing is fail closed.
  - Every prompt, schema, template, model, Adapter checkpoint, data manifest,
    reward, objective, backend, and code state has a version/hash.
  - Exception: the accepted stable Qwen3 model uses model name + absolute path
    rather than full weight-shard hashes; protocol fixtures are still hashed.
  - Unset research values fail closed rather than receiving defaults.
- [ ] `OPEN_CONFIGURABLE SK-07` — Compact package/config labels.
  - Formal prose names remain fixed.
  - Candidates: `representation` and `policy_rl`.
  - Accepted short labels: `[TBD]`
- [x] `FIXED SK-08` — The first policy RL path exposes only a frozen TGVF
  Adapter. A joint-update extension may be reserved but not implemented as an
  active mode without Gate J0.
- [x] `FIXED SK-09` — The SDPO-ready framework boundary is implemented with:
  - typed feedback, success decision, group/demonstration provenance;
  - versioned teacher-context request/result and truncation report;
  - sampled-token alignment for every policy assistant turn;
  - teacher replay handles for the exact recorded original image, main `D`, all
    D-DeepStack branches, layout, positions, masks, and cache contract;
  - full-distribution and sampled-token distillation paths with valid/sample
    masks;
  - objective plugin/composition identity and metrics;
  - current-policy self-teacher plus EMA/trust-region regularization lifecycle
    and FSDP2/LoRA state ownership;
  - teacher/update counters, configuration, RNG, checkpoint, and resume state.
  Pure `sdpo` and `grpo` are distinct registry entries; no hybrid is implied.
  Exact production feedback, equations, target approximation, placement, and
  parameters remain fail closed under Gate D0.
- [x] `FIXED SK-10` — A versioned Qwen VLM family-adapter
  interface covering processor/template, vision taps, DeepStack, M-RoPE,
  multimodal state, and deterministic forward.
  - Primary implementation fixture: Qwen3-VL-8B-Thinking, including the real
    `SC-20-R6` vLLM latent-transport smoke.
  - Secondary boundary: `Qwen/Qwen2.5-VL-7B-Instruct` main-`D` synthetic family
    contract only. This closes the interface shape, not the full end-to-end
    compatibility gate in `PR-02C`/`SD-11`.
- [x] `FIXED SK-11` — The shared target-conditioning interface has both
  contextual-hidden-state and target-token-embedding provider
  implementations/fixtures. `TCPI-20260719` requires one explicit provider enum
  in every run configuration, a shared typed request/output contract, and a
  configuration-driven factory with no default provider. Contextual hidden-layer
  and token-embedding identities are mutually exclusive and fail closed. No
  provider-specific trajectory schema is allowed.
- [x] `FIXED SK-12` — A separate optional judge-provider interface is
  implemented. Judge output carries model/service/prompt/sampling/calibration
  identity and may never masquerade as reference-policy or SDPO-teacher state.

## 5. Gate R0 — Before native rollout is accepted

### 5.1 Model, template, and protocol identity

- [ ] `OPEN_BLOCKING RO-P01` — Record the accepted Qwen3 model name/path and pin
  processor, tokenizer, chat template, family adapter, golden token fixtures,
  and fixture hashes for the primary rollout. Full weight-shard hashes are not
  required.
- [ ] `OPEN_BLOCKING RO-P02` — Pin exact native tool schema and schema hash for
  `tgvf_focus_tool`. Description/argument wording: `[TBD]`
- [ ] `OPEN_BLOCKING RO-P03` — Pin one initial prompt version and exact hash.
  Prompt text: `[TBD]`
- [ ] `OPEN_BLOCKING RO-P04` — Golden token-ID fixtures for direct answer, one
  call, and at least two calls.
- [ ] `OPEN_BLOCKING RO-P05` — Prove tokenizer length unchanged and no added
  embedding/lm-head rows.
- [ ] `OPEN_BLOCKING RO-P06` — Freeze assistant prefill and stop semantics,
  including ownership of `</tool_call>`, `<|im_end|>`, think closure, and EOS.
- [ ] `OPEN_BLOCKING RO-P07` — Prove exactly one template-owned `<think>` opener
  per assistant turn and no duplicate policy-sampled opener.

### 5.2 Parser, target span, and multi-call state machine

- [ ] `OPEN_BLOCKING RO-S01` — Strict JSON/parser behavior for malformed calls,
  unknown keys, trailing answers, tool errors, timeout, and loops.
- [ ] `OPEN_BLOCKING RO-S02` — Exact target value span mapping across JSON
  escapes, non-ASCII text, repeated strings, and boundary-crossing tokens.
- [ ] `OPEN_BLOCKING RO-S03` — `Hq` identity:
  - included/excluded syntax tokens: `[TBD]`
  - hidden layer: `[TBD]`
  - token-time alignment: `[TBD]`
  - pooling/sequence contract: `[TBD]`
- [ ] `OPEN_CONFIGURABLE RO-S04` — Tool-call cap greater than one.
  - Initial value: `[TBD]`
  - cap-hit/timeout/tool-error termination: `[TBD]`
- [ ] `OPEN_BLOCKING RO-S05` — Multi-call cache/history contract, including
  append/reset/reuse behavior for text KV, original image, main `D`, and all
  D-DeepStack branches.
- [ ] `OPEN_BLOCKING RO-S06` — No decode/rerender/retokenize drift. If a backend
  forces rerendering, whole-token, visual-layout, position, mask, and cache
  parity is required.

### 5.3 Token ownership and loss masks

- [ ] `OPEN_BLOCKING RO-M01` — Mutually exclusive, exhaustive ownership masks:
  template-owned, policy-sampled, environment/tool-observation, and padding.
- [ ] `OPEN_BLOCKING RO-M02` — Policy/loss mask includes every actual sampled
  token in every assistant turn and excludes template prefixes, tool responses,
  image positions, and padding.
- [ ] `OPEN_BLOCKING RO-M03` — Freeze stop/EOS tokens' ownership and whether each
  participates in behavior logprob and loss.
- [ ] `OPEN_BLOCKING RO-M04` — Single, batched, one-call, and two-call mask
  fixtures have exact expected token counts.

### 5.4 Actual behavior log probabilities

- [ ] `OPEN_BLOCKING RO-L01` — Store the actual behavior log probability for
  every policy-sampled token. Length must equal the sampled-token mask count.
- [ ] `OPEN_BLOCKING RO-L02` — Explicitly forbid replayed
  `new_logprobs.detach()` as behavior log probabilities.
- [ ] `OPEN_BLOCKING RO-L03` — Define and separately name:
  - raw model-distribution logprob: `[TBD]`
  - post-temperature/top-k/top-p/min-p/penalty/processor sampling logprob:
    `[TBD]`
  - distribution used in the policy ratio: `[TBD]`
- [ ] `OPEN_BLOCKING RO-L04` — Record processor ordering and any stateful
  repetition/frequency/presence penalty state.
- [ ] `OPEN_BLOCKING RO-L05` — Logprob dtype, storage format, finite-value
  checks, missing-value behavior, and numerical tolerance: `[TBD]`

### 5.5 Sampling identity

- [ ] `OPEN_BLOCKING RO-I01` — Each trajectory records:
  - rollout policy/checkpoint/adapter and weight-sync version;
  - veRL commit and rollout backend/version;
  - temperature, top-p, top-k, min-p, penalties, processors, and stops;
  - max tokens and max turns;
  - global/per-sample seed, RNG derivation/state, rank, and worker;
  - raw-versus-transformed logprob convention;
  - rollout/update staleness.
- [ ] `OPEN_BLOCKING RO-I02` — First-proof staleness and update barrier.
  - Proposed: `staleness = 0`; no update between sampling and consuming its
    behavior log probabilities.
  - Accepted value: `[TBD]`

### 5.6 Exact per-call `D` observation

For every call, the trajectory record must reserve:

```text
observation_id and call_index
target source token IDs/span and Hq identity
rollout policy version and TGVF Adapter checkpoint/version
main D and every D-DeepStack tensor, or immutable artifact handles
shape, dtype, layout, branch order, checksum, and artifact schema
visual grid and fake-image token span
M-RoPE position IDs and rope deltas
multimodal token types
attention, visibility, original-image, main-D, and branch masks
cache append/reset/reuse contract
```

- [ ] `OPEN_BLOCKING RO-D01` — Freeze the exact observation record schema.
- [ ] `OPEN_BLOCKING RO-D02` — Freeze bit-preserving artifact transport/storage;
  lossy serialization is forbidden.
- [ ] `OPEN_BLOCKING RO-D03` — Freeze zero-to-N-call batching, padding, and
  D-DeepStack branch ordering.
- [ ] `OPEN_BLOCKING RO-D04` — Policy, old-policy, and reference workers verify
  identical observation checksums.
- [ ] `OPEN_BLOCKING RO-D05` — Replay recomputes logits only. It never invokes
  the TGVF Adapter to regenerate `Hq`, main `D`, or D-DeepStack.
- [ ] `OPEN_BLOCKING RO-D06` — Freeze replay cache strategy: deterministic
  full-forward or a precisely identified cache artifact. Value: `[TBD]`
- [ ] `OPEN_BLOCKING RO-D07` — SDPO teacher replay, when enabled, consumes the
  same recorded observation handles and checksums. Feedback-conditioned text
  may differ by the accepted teacher-context contract; visual observations may
  not be regenerated or simplified away.

The identity reason is:

```text
D_i = TGVF_Adapter(image_features, Hq_i; adapter_version)
```

`Hq_i` is produced from the rollout-time action/context. Recomputing it with an
updated actor or with the reference model changes the observation, so the policy
ratio and reference KL would no longer compare the same recorded trajectory.

### 5.7 Deterministic forward

- [ ] `OPEN_BLOCKING RO-F01` — Freeze model train/eval state, dtype/autocast,
  attention backend, forward implementation, and deterministic flags.
- [x] `FIXED RO-F02` — Policy-adapter dropout is zero for the first proof unless
  exact RNG/mask replay is separately proven.
- [ ] `OPEN_BLOCKING RO-F03` — Same policy version + tokens + recorded
  observation yields rollout/replay logits and logprobs within `[TBD tolerance]`.
- [ ] `OPEN_BLOCKING RO-F04` — Verify parity for single/batched, direct,
  one-call, and two-call trajectories.
- [ ] `OPEN_BLOCKING RO-F05` — Family-specific deterministic-forward fixtures
  declare Qwen3-VL and `Qwen/Qwen2.5-VL-7B-Instruct` processor, M-RoPE,
  DeepStack, mask, and cache differences behind the common adapter rather than
  branching in objective code.

## 6. Gate V0 — veRL/vLLM compatibility evidence

I8H-20260719 authorizes this gate's vLLM-only work. Checkboxes below that are
not explicitly closed continue to track evidence, not a need for another user
reply. All SGLang and stable-comparison wording below is historical and
superseded by §1.1.

- [x] `FIXED VS-01` — Accept a bounded spike task in
  `docs/PROJECT_TASK.md`; veRL selection itself is not reopened.
  - Accepted task: `I8H-20260719`, Project Task §0.
- [x] `FIXED VS-02` — The isolated compatibility matrix is accepted:
  CPython `3.12.3`, upstream veRL
  `e003163181731412595257a72ec173071efb125f`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`, and Transformers `4.57.6`, with the complete resolved graph in
  `requirements/compatibility.lock`. Official
  `v0.8.0@7aed6b230776f963fa09509c10d9c3a767d1102c` remains historical comparison
  material and was not installed or run.
- [x] `FIXED VS-03` — vLLM is the only rollout backend. SGLang is not installed,
  tested, or maintained by this task. Exact vLLM package/image identity is
  selected and recorded from compatibility evidence. Live Qwen3 execution
  requires `VLLM_PLUGINS=tgvf_qwen3_precomputed`,
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, and multimodal-encoder
  `TORCH_SDPA`.
- [x] `FIXED VS-04` — The minimum bounded FSDP2 evidence is `SC-30`: two B200
  ranks, one-dimensional composable-FSDP2 mesh, FP32 tiny deterministic model,
  strict distributed model/optimizer/extra-state checkpoint, teardown, rebuild,
  and bitwise-identical resumed next step. This closes the compatibility
  infrastructure question only; production Qwen wrap, precision, offload,
  placement, sharding, and topology remain Gate G0/P0 decisions.
- [x] `FIXED VS-05` — The accepted public surface is veRL
  `AgentLoopOutput`/custom `AgentLoopManager`, `DataProto`,
  `register_policy_loss`, `FSDPEngineConfig`, and `CheckpointHandler`, plus
  vLLM's public general-plugin, model, and multimodal processor registries.
  The project owns lossless records, exact replay, objectives, teacher state,
  and plugin code; no private worker/postprocess/trainer or site-package patch
  is used.
- [ ] `OPEN_BLOCKING VS-06` — Numerical tolerances and PASS/FAIL criteria for:
  primary Qwen3 policy/reference forward,
  `Qwen/Qwen2.5-VL-7B-Instruct` family-adapter fixture, both
  target-conditioning providers, two-call latent-observation
  transport, actual behavior logprobs, exact observation replay, FSDP2 one
  step, save/resume.
  - Completed subset: `346` CPU tests, Qwen3 `SC-20-R6` real TP=2 native
    two-call latent transport, `SC-30` exact tiny-model FSDP2 resume, and valid
    `RP-10` real-Qwen3 target-token-embedding backward/FSDP2 update/save/export,
    and `RP-11` exact real-Qwen3 teardown/restore next-update parity.
    The identity-invalid side result remains debugging evidence only.
  - Remaining: real Qwen policy/reference logit/logprob replay parity, the full
    Qwen2.5 family-specific fixture, and production loss/gradient/topology
    criteria. The passed transport smoke does not close those questions.
- [x] `FIXED VS-07` — Failure conditions include private-trainer forks,
  PIL re-encoding of `D`, missing actual sampling logprobs, or inability to
  preserve exact observation state.
  Failed intermediate Qwen cells remain recorded rather than hidden; the
  passing path uses only repository-owned public extensions.
- [x] `FIXED VS-08` — Every completed GPU cell had a complete `PLANNED` ledger
  entry before launch. This remains a mandatory per-cell rule for future GPU
  work rather than a blanket authorization.
- [x] `FIXED VS-09` — The SDPO patch-surface map from pinned
  reference commit `7c457fc1...` to the accepted upstream veRL commit. The
  external-reference record identifies fork-private behavior; the implementation
  ports the objective and teacher contracts onto the public upstream surface
  without adopting the reference repository's veRL tree.
- [x] `FIXED VS-10` — The neutral schema and framework tests carry a complete
  native multi-call teacher context, per-assistant-turn alignment, typed
  feedback, exact multimodal observation handles, and versioned distillation
  targets. Exact-observation teacher replay rejects handle or model-state
  changes; it does not use the pinned reference's text-only final-response
  reconstruction.
- [ ] `OPEN_BLOCKING VS-11` — Establish a feasible FSDP2/LoRA or full-policy
  lifecycle for separate frozen-reference and SDPO teacher state, including
  initialization, update timing, sharding/offload, checkpoint, RNG, and strict
  resume. Repository-owned teacher state and CPU checkpoint/resume tests exist;
  `SC-30` proves generic strict FSDP2 state restoration, not the final
  actor/reference/teacher placement or Qwen LoRA/full-policy lifecycle.
- [x] `FIXED VS-12` — Known reference limitations—text-only reconstruction,
  legacy-worker coupling, KL-reference role ambiguity, full-distribution memory,
  and approximation/alignment choices—are explicit exclusions or configurable
  project-owned contracts. No SDPO-fork runtime behavior is inherited as a
  production default.

Required output: [`VERL_COMPATIBILITY_REPORT.md`](VERL_COMPATIBILITY_REPORT.md)
records PASS/FAIL evidence, the selected commit/backend/dependency matrix,
required public extension surface, bounded FSDP2 evidence, and unresolved
blockers.

Planning artifact: `docs/VERL_COMPATIBILITY_SPIKE_PLAN.md`. Its compatible
technical checks remain useful, while its authorization model is superseded by
I8H-20260719.

## 7. Gate A0 — Before native-format representation training

The code needed for a bounded Qwen3 representation execution is present. This
section distinguishes that implementation evidence from permission to consume
the external data or start a production/GPU run.

- [ ] `OPEN_BLOCKING AD-01` — Candidate retained train/validation JSONL paths
  and SHA256 identities are recorded in `docs/LEGACY_REFERENCE.md`. The strict
  transform audit produced:
  - train: 35,542 accepted rows, 9,186 groups, 6,398 groups with at least four
    targets, 14,480 excluded rows, 3,004 leakage records, manifest
    `4160198e65268e33f1c36d050f74498f4f8fa35f3ac263202ee8bfdf5f5cd820`;
  - validation: 1,382 accepted rows, 376 groups, 226 groups with at least four
    targets, 641 excluded rows, 108 leakage records, manifest
    `e44cbd6f86ff82879b3be312d9a23198b7267bccd710cbe7d1ecc1dc9954ea15`.
  Group key, stable UID, and content hash do not overlap, but seven distinct
  resolved image paths do. The user accepts those seven paths as recorded split
  metadata under `RPI-20260719-NORM-EVAL`; they do not block training and no
  implementation may silently filter them. The loader/config must still record
  the exact overlap report and must not claim image-path disjointness. Dataset/
  image licenses and a perceptual near-duplicate audit remain open production
  metadata items rather than grounds for silently changing this population.
- [x] `FIXED AD-02A` — The protocol-neutral representation row contains the
  retained row `uid` as `sample_id`, exact image reference, already-rendered
  question text, target text, `evidence_description`, optional `image_id`, and
  the recorded evaluation metadata. Its immutable content digest binds every
  field, and the group key is `image_id` with exact image-reference fallback.
  The exact retained data has no separate `choices` field; a future source with
  choices requires a new schema/transform version rather than implicit prompt
  rendering inside this record.
- [x] `FIXED AD-02B` — `retained_focus_rows_v1` implements source-hash
  validation, strict focus metadata, fail-closed fields/image resolution,
  duplicate handling, leakage records/warnings, every-row accepted/excluded
  disposition, immutable manifest identity, and four-key split-overlap reports.
  Evidence: `tests/representation/training/test_data.py` plus the exact audit in
  `docs/LEGACY_REFERENCE.md`. Leakage records are metadata, not implicit row
  removal.
- [ ] `OPEN_BLOCKING AD-02C` — Before any production representation run or
  resume, bind the validation manifest identity and validation sampler seed
  into the run/checkpoint identity, and verify them when appending validation
  history after restore. The bounded synthetic smoke records both values in
  its TOML and metrics, but the current `RepresentationRunIdentity` does not
  make them restore invariants.
- [ ] `OPEN_BLOCKING AD-02D` — Before any production representation run, bind
  the actual bytes (or an accepted immutable content identity) of every image
  consumed by a retained manifest. The current source/transform identities bind
  each image path but do not detect an in-place image-file replacement. This is
  explicitly not closed by the repository-owned synthetic smoke images.
- [ ] `OPEN_BLOCKING AD-03` — New pipeline transcript/prompt construction and
  production `Hq` contract are not fully closed:
  - [x] `FIXED AD-03A` — `evidence_description` is the reasoning content of the
    assistant turn after the latent `tgvf_focus_tool` result. Only the exact
    tokenizer positions owned by this rendered evidence span receive
    teacher-forcing labels. The prompt, first assistant tool-call turn,
    tool-call JSON, tool result, template wrappers, and answer content are
    ignored; the representation evidence turn itself has empty answer content.
    Its preceding teacher-constructed call turn has empty reasoning/answer
    content and only the native target call, so no intermediate reasoning label
    or hidden prompt text is invented.
    Offset ownership must match the rendered token ids exactly. A
    token is evidence-owned iff its start offset lies inside the evidence span;
    this includes Qwen's observed sentence-final token that also carries the
    following template newline. A token that starts before and crosses into the
    evidence is ambiguous and fails closed. Fuzzy decoded-text matching is
    forbidden. Canonical labels are not model labels until the Qwen-family
    adapter verifies the processor's canonical-to-expanded-position map:
    evidence mappings are singleton/contiguous/ID-identical and every visual
    position remains ignored. The executable mapping is Qwen3-specific;
    Qwen2.5-VL fails closed pending its separate family fixture/artifact.
  - [x] `FIXED AD-03B` — The Qwen3 implementation constructs one original-image
    user turn, one native `tgvf_focus_tool` action, one latent-image tool result,
    and the evidence reasoning turn. The strict parser identifies the raw JSON
    target span and exact IDs before/after processor expansion. The separate
    `qwen3_native_representation_smoke_v1.json` golden uses an explicitly
    non-production prompt, a deterministic 56×56 RGB image, a Unicode/slash
    target, and evidence text; it freezes prompt, tokenizer/template/schema,
    action/evidence transcript, target span, expanded-input, two-visual-block,
    and evidence-label identities against the accepted local processor.
  - [ ] `OPEN_CONFIGURABLE AD-03C` — Accept the production prompt wording/hash,
    contextual hidden layer, and any target-visible versus target-omitted
    matched-control policy. The smoke-only prompt is forbidden as a production
    default.
- [ ] `OPEN_BLOCKING AD-04` — TGVF Adapter initialization that does not use the
  historical trained checkpoint directly. The config/runtime implement only
  fresh initialization with an explicit seed and reject legacy checkpoint
  initialization, but the production initialization distribution and seed are
  `[TBD]`.
- [ ] `OPEN_BLOCKING AD-05` — Exact representation objective and execution
  contract is not yet fully closed. Its independently gated parts are:
  - [x] `FIXED AD-05A` — The pinned Matrix-CE equation is implemented: each
    same-image group produces a square score matrix with evidence/query on rows,
    candidate `D` plus all D-DeepStack branches on columns, the diagonal is the
    label, there is no temperature in the historical equation, and cross
    entropy is summed then divided by the total number of valid rows across
    groups. Across data-parallel ranks and gradient-accumulation microbatches,
    the trainer must aggregate the global CE numerator and global valid-row
    denominator; averaging already-normalized local losses is forbidden when
    group sizes differ.
    The pure terms, manual score-gradient, unequal-rank normalization, atomic
    main/branch swap oracle, memory-bounded score/recompute path, post-`D`
    original-image key blocking, and trainer-owned global normalization all have
    CPU fixtures. Loss-excluded zero-gradient padding now aligns composable-
    FSDP forward/backward counts when ranks receive four and five real
    candidates. The streaming executor requires deterministic recomputation and
    releases each cell graph before traversing each Adapter candidate once.
    Corrected `RP-10` supplies one real 8B/two-rank K=2 execution and `RP-11`
    supplies the K=4/GA=4 Norm-aware exact teardown/restore next-update proof
    for the target-token-embedding provider. Production identity, semantic
    thresholds, paired-provider comparison, and numerical controls remain
    gated below.
  - [x] `FIXED AD-05B` — Same-image multi-target grouping implements image key,
    minimum/maximum group batch size, duplicate handling, incomplete-group
    dropping, whole-group distributed ownership, group/member shuffle,
    seed/epoch/cursor state, and deterministic resume. Ordinary independent
    shuffle is forbidden.
    Exact duplicate target strings within one image group now fail closed;
    semantic near-duplicate normalization remains a manifest decision.
  - [x] `FIXED AD-05C` — The baseline contains `L_gen`; setting its weight to
    zero is allowed only for a separately identified ablation. Only the
    `AD-03A` evidence-token labels contribute. The reduction is the historical
    evidence-token mean NLL per sample followed by a global sample mean, and it
    is logged separately from Matrix CE. "Assistant supervision" is not a
    separate module. The exact nonzero baseline coefficient remains
    `OPEN_CONFIGURABLE` under `AD-05F` and must not receive a hidden default.
    The executable objective identity enforces nonzero Matrix-CE and `L_gen`
    weights for the baseline and permits zero only for a named Matrix-only
    ablation; both raw and weighted components are separately returned and
    loggable. Actual trainer logging remains part of `AD-05F`.
  - [x] `FIXED AD-05D` — Manifold-loss optimizer contribution is exactly zero.
    A diagnostic-only computation, if retained, must not contribute gradients.
  - [x] `FIXED AD-05E` — Norm loss uses the one accepted historical formula:
    `mean_t(log((||D_t||_2 + 1e-6) / clamp_min(mean_s ||V_s||_2, 1e-6))^2)`
    against detached corresponding post-merger source visual tensors. Per
    sample, the three D-DeepStack branch losses are averaged and that branch
    mean is averaged with the main loss; distributed/accumulated reduction is a
    global sample mean. The baseline scalar weight is `0.1`. No mode, target, or
    alternative-formula selector is accepted.
  - [ ] `OPEN_CONFIGURABLE AD-05F` — The trainer/config/checkpoint code exposes
    explicit nonzero Matrix-CE and `L_gen` weights, AdamW options, scheduler,
    precision, accumulation, clipping, validation/log cadence, and strict
    optimizer-boundary resume. CPU fixtures prove an accumulated optimizer step
    and bitwise-identical next step after resume. Exact production values and
    the accepted TOML identity remain `[TBD]` and receive no library defaults.
    Historical `L_gen` first divides each sample's summed evidence NLL by that
    sample's evidence-token count, then sums those per-sample means and divides
    by the global sample count. Accumulation and DDP must aggregate that global
    numerator and denominator; a global token-mean or an equal mean of unequal
    local microbatch means is a different objective and is forbidden silently.
  - [x] `FIXED AD-05G` — Run a real local-Qwen3 backward and two-rank
    representation FSDP2 optimizer/checkpoint/resume smoke on physical GPUs 2
    and 3 under a complete experiment-ledger identity. The generic `SC-30`
    infrastructure smoke is not this evidence. Corrected `RP-10` passed the
    real-Qwen3 target-token-embedding backward, optimizer, validation,
    checkpoint-save, and export subset. `RP-11` then passed K=4/GA=4 continuous
    versus clean process teardown/reconstruction, strict distributed restore,
    and exact matching next optimizer step. This is bounded executor evidence,
    not a production-data or semantic-quality result.
- [x] `FIXED AD-06` — Runtime, trainer, and FSDP2 planning enforce every
  and only Adapter-owned trainable parameters while the vision tower, language
  model, and four borrowed Qwen mergers remain frozen. CPU/meta and simulated
  composable-FSDP2 fixtures cover 52 owned leaves and excluded borrowed state.
  Corrected `RP-10` independently audited real 8B/two-rank sharding, optimizer
  ownership, 104 exported tensors, and exclusion of borrowed Qwen state.
  Accepted as bounded ownership evidence under `I8H-20260719` on 2026-07-19;
  production topology remains separately gated.
- [ ] `OPEN_BLOCKING AD-07` — Adapter-only artifact, single-process training
  checkpoint, distributed checkpoint, rank-zero full-owned-state export, and
  strict resume schemas are implemented. They bind model/provider/prompt/data/
  objective/architecture/projection/optimizer/scheduler/sampler/precision/
  accumulation/initialization/RNG identities and exclude optimizer state,
  Qwen mergers, and legacy protocol rows from deployment. CPU artifact/tamper/
  next-step parity passes. The distributed sidecar additionally binds each
  rank's Adapter and optimizer local-shard content digests before restore is
  applied. Corrected `RP-10` passed real two-rank save/export; independent
  post-run loading verified all 104 export tensor digests plus sidecar integrity
  and recorded per-rank shard-content digest fields. `RP-11` restored the
  step-1 payload in a fresh process, revalidated loaded shard contents, and
  matched the uninterrupted step-2 Adapter, optimizer/scheduler/sampler/RNG,
  shard, and scientific metric state exactly. Artifact promotion remains
  blocked on production identity and semantic gates; final step-2 DCP payloads
  were compared through validated sidecar state rather than independently
  restored a second time.
- [ ] `OPEN_BLOCKING AD-08` — Numerical output/gradient parity of the extracted
  TGVF Adapter core against its pinned reference. The independent small-shape
  functional oracle now checks outputs and target/visual/104-owned-parameter
  gradients in FP32 (`2e-6`) and BF16 (`3e-2`). Exact `4096/1152` historical-
  checkpoint state plus accepted Qwen merger parity remains `[TBD tolerance]`.
- [ ] `OPEN_BLOCKING AD-09` — Target specificity, readout, causal flip, and free
  continuation thresholds: `[TBD]`
- [ ] `OPEN_BLOCKING AD-10` — Synthetic sensitivity/gradient tests cover main
  `D` and every D-DeepStack branch. All branches must still pass real-Qwen
  controlled semantic gates and accepted thresholds.
- [ ] `OPEN_BLOCKING AD-11` — Both required target-conditioning providers pass
  through one explicit shared configuration/runtime/native group-builder path,
  and provider identity is checkpoint-bound. Both have synthetic target-span,
  shape, determinism, and training-path fixtures. Corrected `RP-10` proves a
  real-Qwen optimizer path for target-token-embedding conditioning. A paired
  real-GPU provider comparison is not a prerequisite for a selected provider's
  run; the selected provider's own target-specificity, readability, and
  per-branch evidence remains required.
- [ ] `OPEN_BLOCKING AD-12` — Representation artifacts are explicitly bound to
  a Qwen model identity and provider contract in the implemented schema. Qwen3 and
  `Qwen/Qwen2.5-VL-7B-Instruct` compatibility must not be claimed by loading one
  model's Adapter blindly into the other; Qwen2.5 still lacks its separate
  representation artifact/native transcript/DeepStack/objective fixtures.
- [ ] `OPEN_BLOCKING AD-13` — The provenance inventory of historical internal
  representation tests and metrics now maps implemented data, transcript,
  sampler, loss, masking, trainer, checkpoint, config, and functional-parity
  evidence. Explicit acceptance of proposed exclusions plus real-Qwen metric
  thresholds/evaluations is still required. See
  `docs/REPRESENTATION_PARITY_INVENTORY.md`.
- [ ] `OPEN_CONFIGURABLE AD-14` — A paired contextual-hidden-state versus
  target-token-embedding experiment remains an optional named scientific
  comparison. It is not a representation-training prerequisite. If run, it
  uses identical data/group order, Adapter initialization, batch plan, and seed
  and retains separate artifact identities.

## 8. Gate G0 — Before any GRPO optimizer step

This gate applies even to a one-step smoke test.

### 8.1 Exact GRPO mathematics

- [ ] `OPEN_BLOCKING GR-01` — Group construction, group size, invalid trajectory
  handling, and scalar reward aggregation: `[TBD]`
- [ ] `OPEN_BLOCKING GR-02` — Group mean and population-versus-sample standard
  deviation: `[TBD]`
- [ ] `OPEN_BLOCKING GR-03` — Epsilon, zero-variance behavior, and advantage
  scaling/broadcast to tokens: `[TBD]`
- [ ] `OPEN_BLOCKING GR-04` — Behavior/current policy ratio and exact logprob
  distribution used: `[TBD equation]`
- [ ] `OPEN_BLOCKING GR-05` — Symmetric/asymmetric/dual clipping and parameters:
  `[TBD equation]`
- [ ] `OPEN_BLOCKING GR-06` — Policy-token mask and token-sum/token-mean/
  sequence-mean/two-level normalization across variable-length multi-turn
  trajectories: `[TBD equation]`
- [ ] `OPEN_BLOCKING GR-07` — Frozen reference identity and KL estimator:
  `[TBD equation]`
- [ ] `OPEN_BLOCKING GR-08` — Whether KL is in reward or loss, coefficient
  schedule, mask, and normalization; double counting is forbidden: `[TBD]`
- [ ] `OPEN_BLOCKING GR-09` — Entropy or other auxiliary terms: `[TBD or none]`
- [ ] `OPEN_BLOCKING GR-10` — Update epochs, minibatches, policy-version rule,
  gradient clipping, overflow/NaN handling: `[TBD]`
- [ ] `OPEN_BLOCKING GR-11` — Gradient accumulation and FSDP2 numerator/
  denominator reductions preserve the declared global objective: `[TBD]`
- [ ] `OPEN_BLOCKING GR-12` — Pure-tensor oracle, veRL loss parity, one-step
  gradient parity, accumulation parity, and single-rank/FSDP2 parity tolerances:
  `[TBD]`

### 8.2 Policy/reference parameter contract

- [x] `FIXED PR-01` — Policy initializes from the exact original Qwen reasoning
  checkpoint; no policy SFT adapter.
- [x] `FIXED PR-02` — The first policy model is Qwen3-VL-8B-Thinking at
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`; no full weight hash is
  required.
- [ ] `OPEN_BLOCKING PR-02A` — Pin the primary processor/tokenizer/chat-template
  behavior through exact transcript fixtures and hashes: `[TBD artifact]`
- [x] `FIXED PR-02B` — The secondary model ID is
  `Qwen/Qwen2.5-VL-7B-Instruct`.
- [ ] `OPEN_BLOCKING PR-02C` — Before claiming end-to-end secondary-model
  support, configure its local/runtime identity and pass its full family-specific
  Adapter/representation fixture: `[TBD path/artifact]`
- [ ] `OPEN_BLOCKING PR-03` — Reference model identity and prompt/schema:
  `[TBD]`
- [ ] `OPEN_BLOCKING PR-04` — LoRA/full scope, modules, rank, alpha, dropout,
  bias, trainable whitelist: `[TBD]`
- [x] `FIXED PR-05` — TGVF Adapter frozen for the first policy RL proof.
- [ ] `OPEN_BLOCKING PR-06` — Vision tower, visual mergers, DeepStack, and all
  other module freeze/trainable states: `[TBD]`
- [ ] `OPEN_BLOCKING PR-07` — Optimizer, scheduler, precision, gradient scaling,
  rollout weight-sync version/barrier: `[TBD]`

## 9. Gate P0 — Before the first policy RL pilot

These choices may remain open while building the skeleton, but not when the
pilot is identified or launched.

### 9.1 Data

- [ ] `OPEN_CONFIGURABLE DA-01` — RL data source(s), versions, licenses, and
  canonical raw schema: `[TBD]`
- [ ] `OPEN_CONFIGURABLE DA-02` — DeepEyes 47K status: candidate only; accepted
  role and pinned snapshot `[TBD]`
- [ ] `OPEN_BLOCKING DA-03` — Fixed pilot manifest, sample rule, source weights,
  group construction, shuffle seed, and hashes: `[TBD]`
- [ ] `OPEN_BLOCKING DA-04` — Exact/near-duplicate image and normalized-question
  leakage checks against held-out evaluation.
- [ ] `OPEN_BLOCKING DA-05` — Broken/ambiguous sample handling, answer types, and
  verifier routing: `[TBD]`
- [x] `FIXED DA-06` — Canonical samples do not contain one irreversible rendered
  prompt; prompts are versioned at runtime.

### 9.2 Reward

- [ ] `OPEN_CONFIGURABLE RW-01` — Final-answer extraction, normalization, and
  verifier router/version: `[TBD]`
- [ ] `OPEN_CONFIGURABLE RW-02` — Malformed call, tool error, timeout, loop,
  cap-hit, and invalid final output rewards/penalties: `[TBD]`
- [ ] `OPEN_CONFIGURABLE RW-03` — Optional tool bonus and per-call/token/latency
  costs: `[TBD enabled/coefficients]`
- [ ] `OPEN_CONFIGURABLE RW-04` — Target/evidence reward components: `[TBD or
  explicitly disabled]`
- [ ] `OPEN_CONFIGURABLE RW-05` — Judge scope, model, prompt, sampling identity,
  and calibration: `[TBD or none]`
  - Reserved model: `Qwen/Qwen2.5-72B-Instruct` via a separately versioned
    provider/service; disabled for the first pilot.
  - It is not the RL reference or the SDPO teacher.
- [ ] `OPEN_BLOCKING RW-06` — Component ranges, clipping, total reward equation,
  verifier-failure behavior, and separate logging: `[TBD]`

### 9.3 Prompt and tool policy

- [ ] `OPEN_CONFIGURABLE PM-01` — Exact system/tool prompt text and hash: `[TBD]`
- [ ] `OPEN_CONFIGURABLE PM-02` — Exact tool description and target wording:
  `[TBD]`
- [ ] `OPEN_CONFIGURABLE PM-03` — Tool-call safety cap and exploration
  curriculum: `[TBD]`
- [ ] `OPEN_BLOCKING PM-04` — Prompt acceptance fixtures for parse rate,
  non-empty target, continuation, two calls, no duplicate think opener, and no
  example copying.

### 9.4 Evaluation and promotion

- [ ] `OPEN_BLOCKING EV-01` — Held-out manifests, official scorers, and exact
  sample pairing: `[TBD]`
- [ ] `OPEN_BLOCKING EV-02` — Direct/tool/counterfactual rows and reasoning
  retention metrics/thresholds: `[TBD]`
- [ ] `OPEN_BLOCKING EV-03` — Checkpoint ladder, early-stop conditions, and
  promotion rule: `[TBD]`

## 10. GPU execution gates

Only the subsection applicable to a cell is required in addition to
`GPU-ANY`. A rollout-only or teacher-forward cell is not forced to perform an
optimizer step; an optimizer/FSDP cell cannot use that distinction to skip its
own stricter gate.

I8H-20260719 is the user authorization for bounded in-scope GPU rows; no
additional interactive reply is required after the row is complete. Execution
is restricted to physical GPU indices **2 and 3**. The ledger records their
physical-to-logical mapping, and no other physical GPU may be exposed to the
process.

### 10.1 GPU-ANY — Before every GPU command

- [ ] `OPEN_BLOCKING GPU-A01` — Complete a cell-specific `PLANNED` entry in
  `docs/EXPERIMENT_LEDGER.md`, including plan/approval revision and a
  namespace-valid ID. `SC-*` is reserved for the fixed compatibility matrix;
  representation-phase executions use `RP-*` and may not reuse an `SC-*` ID.
- [ ] `OPEN_BLOCKING GPU-A02` — Record code commit/dirty state, model/processor,
  prompt/schema/template, observation fixture/checkpoint or justified `N/A`,
  objective or justified non-RL probe identity, and exact output identities.
- [ ] `OPEN_BLOCKING GPU-A03` — Approve question, fixture, command, hardware,
  PASS/FAIL thresholds, timeout, early stop, cleanup, and recovery plan.
- [ ] `OPEN_BLOCKING GPU-A04` — Pin driver, container/image digest, Python,
  PyTorch, CUDA, NCCL, Transformers, attention implementation/kernel, veRL,
  rollout backend, weight/KV dtype, quantization, tensor parallelism, and the
  complete environment lock.
- [ ] `OPEN_BLOCKING GPU-A05` — Record actor/reference/rollout/TGVF/teacher
  placement applicable to the cell and its maximum GPU-hours/scratch use.

### 10.2 GPU-ROLLOUT — Additional S2/rollout requirements

- [ ] `OPEN_BLOCKING GPU-R01` — Direct/one-call/two-call fixtures preserve
  sampled tokens, ownership masks, exact latent observation, positions, masks,
  branches, cache identity, and source-image separation.
- [ ] `OPEN_BLOCKING GPU-R02` — Record actual behavior-logprob semantics,
  sampling transforms/order, policy/backend/RNG/request identity, and staleness.
- [ ] `OPEN_BLOCKING GPU-R03` — Same recorded observation satisfies single/
  batched, cached/no-cache, actor/proximal/reference replay tolerances without a
  policy update or latent recomputation.
- [ ] `OPEN_BLOCKING GPU-R04` — Estimate per-call observation artifact GPU/CPU/
  disk/network footprint and I/O cost.

### 10.3 GPU-FSDP-OPT — Additional optimizer/FSDP2 requirements

- [ ] `OPEN_BLOCKING GPU-F01` — Record actor/reference device meshes,
  sharding/wrap policy, mixed precision, activation checkpointing, offload,
  state-dict strategy, LoRA handling, and rollout/training placement.
- [ ] `OPEN_BLOCKING GPU-F02` — Pair the distributed path with an exact
  single-device control and freeze initialization, ordered batch, objective,
  optimizer, dtype, seed, and reductions.
- [ ] `OPEN_BLOCKING GPU-F03` — Compare loss, gradient direction/magnitude,
  parameter delta, and post-update result using accepted tolerances.
- [ ] `OPEN_BLOCKING GPU-F04` — Synchronous save/teardown/resume restores model,
  reference, optimizer, scheduler, RNG, dataloader/sampler, policy/version,
  observation/objective custom state, and the matching next step.

### 10.4 GPU-SDPO — Additional teacher-forward requirements

- [ ] `OPEN_BLOCKING GPU-SD01` — Record teacher placement/sharding/offload,
  target type/memory, update timing, policy/reference coexistence, exact
  observation use, target mask, and any teacher/EMA state round trip. A seam-
  only cell cannot claim runtime SDPO compatibility.

### 10.5 GPU-PILOT — Additional long-pilot requirements

- [ ] `OPEN_BLOCKING GPU-P01` — Record tokens/s, tool latency, update latency,
  peak memory, utilization, total-duration estimate, checkpoint cadence, and
  stop/recovery policy before a long pilot.

LoRA + FSDP2 must be executable before the first policy pilot. Full-parameter +
FSDP2 remains required, but its corresponding items close before the first
full-parameter experiment rather than blocking the LoRA proof.

## 11. Gate J0 — Deferred joint TGVF Adapter updates

Status: `DEFERRED` for the first policy RL proof.

- [ ] `DEFERRED JT-01` — Exact objective and reason to update the TGVF Adapter.
- [ ] `DEFERRED JT-02` — Adapter/policy parameter-version identity and update
  ordering.
- [ ] `DEFERRED JT-03` — Stored rollout inputs and a gradient-carrying path that
  does not replace the immutable observation used for likelihood replay.
- [ ] `DEFERRED JT-04` — Auxiliary target-specificity/readability losses and
  gates preventing semantic collapse.
- [ ] `DEFERRED JT-05` — Gradient reachability, RNG, staleness, replay parity,
  checkpoint, and resume tests.

## 12. Gate D0 — Before any SDPO optimizer step

Status: **real reference-style pure-SDPO implementation is required by
I8H-20260719; an optimizer smoke remains fail-closed until every applicable D0
item and its test-contract oracle pass**. D0 work may proceed in parallel with
GRPO correctness work. Production mathematics/configuration remain open after
the synthetic implementation smoke.

- [x] `FIXED SD-01` — Reference identity:
  - paper: arXiv `2601.20802v2`;
  - repository: `https://github.com/lasgroup/SDPO`;
  - commit: `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`.
- [ ] `OPEN_BLOCKING SD-02` — Freeze the exact reference-style SDPO equations:
  forward/reverse/generalized-JSD choice, token importance weighting and clip,
  masks, per-token/sequence normalization, and reduction: `[TBD equations]`
- [ ] `OPEN_BLOCKING SD-03` — Freeze typed success, scalar reward, environment
  feedback, optional judge feedback, group `uid`, demonstration-selection and
  provenance contracts: `[TBD]`
- [ ] `OPEN_BLOCKING SD-04` — Freeze teacher-context template/version,
  complete native multi-call transcript serialization, response alignment for
  every sampled assistant turn, truncation/fail-fast policy, and hashes: `[TBD]`
- [ ] `OPEN_BLOCKING SD-05` — Freeze teacher identity/lifecycle:
  current-policy self-teacher, EMA or trust-region regularization;
  initialization; update timing/rate; interaction with the separate KL
  reference; LoRA/full and FSDP2 state ownership: `[TBD]`
- [ ] `OPEN_BLOCKING SD-06` — Freeze full-logit or top-k-plus-tail
  distillation-target schema, storage/recompute rule, dtype, memory budget,
  token/sample masks, and numerical tolerance: `[TBD]`
- [ ] `OPEN_BLOCKING SD-07` — Freeze objective modes. Reference-style pure SDPO
  is distinct from pure GRPO. Any `grpo_sdpo` hybrid requires a new explicit
  equation, coefficient, normalization, and ablation; it is not inherited from
  the reference repository: `[TBD]`
- [ ] `OPEN_BLOCKING SD-08` — Teacher replay consumes exact rollout-recorded
  original images and each call's main `D`, D-DeepStack, layout, positions,
  masks, and cache contract. Text-only reprompt reconstruction is forbidden.
- [ ] `OPEN_BLOCKING SD-09` — Teacher/EMA/trust-region weights, update counters,
  configuration, optimizer step, RNG, sampler/group state, and feedback/template
  identity checkpoint and resume exactly: `[TBD schema]`
- [ ] `OPEN_BLOCKING SD-10` — Reference text loss and one-step parity, teacher
  update parity, padding/alignment/top-k-tail tests, checkpoint/resume parity,
  then two-call multimodal Qwen3 parity.
- [ ] `OPEN_BLOCKING SD-11` — Validate Qwen2.5-VL SDPO compatibility through the
  `Qwen/Qwen2.5-VL-7B-Instruct` family adapter before advertising cross-family
  SDPO support.

## 13. Next document actions

- [x] Accept I8H-20260719, its vLLM-only compatibility questions, neutral
  interfaces, synthetic fixtures, hard failures, isolated dependencies, and
  bounded execution authority.
- [x] Run the approved isolated compatibility task, record the selected public
  hooks and state ownership, and implement the framework-binding packages.
  `346` CPU tests, `SC-20-R6`, `SC-30`, corrected `RP-10`, and `RP-11` are the accepted
  bounded evidence; the colliding representation side result remains invalid.
- [x] Freeze the bounded framework legacy file inventory and port records.
- [x] Record the bounded representation-training code lineage, strict external-
  data transform audit, manifest hashes, and the seven unresolved exact path
  overlaps. This records evidence but does not accept the data for training.
- [x] Implement the Qwen3 native representation data/group path, both provider
  choices, streaming Matrix CE plus `L_gen`, trainer, FSDP2 ownership,
  checkpoint/config contracts, real processor golden, and functional Adapter
  oracle.
- [x] Run corrected `RP-10`, one bounded real-Qwen3/two-rank FSDP2
  representation backward, validation, checkpoint-save, and export smoke,
  without making a production-training or promoted-artifact claim.
- [x] Run `RP-11`, the bounded real-Qwen3 K=4/GA=4 continuous versus clean
  teardown/restore matching-next-update proof with Matrix CE, `L_gen`, and Norm.
- [ ] Accept the production representation data split, prompt, scientific
  configuration, parity tolerances, semantic evaluation, and promotion
  thresholds before any production training claim. A paired-provider run is
  optional comparison evidence, not a prerequisite.
- [x] Implement the S0 Qwen-family boundary, both condition-provider
  interfaces, SDPO teacher/objective/checkpoint boundary, and optional
  judge-provider interface. This does not close Qwen2.5 end-to-end or D0
  production-mathematics gates.
- [ ] Configure the local/runtime path for the fixed
  `Qwen/Qwen2.5-VL-7B-Instruct` model before its executable fixture.
- [x] Keep the reserved `Qwen/Qwen2.5-72B-Instruct` judge disabled for the first
  pilot; specify service/prompt/calibration only before a later activation.
- [ ] Convert each accepted `[TBD]` into a versioned project artifact rather
  than embedding it only in code or an experiment command.
