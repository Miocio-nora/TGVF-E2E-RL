# Open Implementation Contracts

Status: **active gate checklist**
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
- This checklist does not itself authorize implementation, dependency
  installation, legacy inspection, or GPU execution.
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

## 2. Fixed directions

- [x] `FIXED` — Upstream veRL is the policy-RL infrastructure.
- [x] `FIXED` — FSDP2 support is required. Exact topology is evidence-based.
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
  veRL is selected; its commit/backend/topology still require a spike.
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

## 4. Gate S0 — Before framework-skeleton implementation

The exact reward, RL dataset, and prompt text do not block this gate. Their
interfaces must be versioned and fail closed while unset.

- [ ] `OPEN_BLOCKING SK-01` — Accept the exact skeleton task and file/interface
  list in `docs/PROJECT_TASK.md`.
  - Value: `[TBD]`
  - Minimum modules: TGVF Adapter/representation, Qwen adapter, protocol,
    multi-call tool runtime, framework-neutral trajectory records, reward
    interface, veRL adapter, evaluation.
- [ ] `OPEN_BLOCKING SK-02` — Freeze ownership between veRL and this repository.
  - veRL owns: distributed workers, scheduling, optimizer execution,
    checkpoint/resume, metric aggregation.
  - This repository owns: native transcript identity, strict parser, latent
    tool execution, per-call observation records, rewards/verifiers, exact
    objective extension/parity tests.
  - Public extension points to use: `[TBD after spike]`
- [ ] `OPEN_BLOCKING SK-03` — Freeze the first legacy reuse inventory.
  - Exact files/symbols/data manifests and hashes: `[TBD]`
  - Required artifact: updated `docs/LEGACY_REFERENCE.md` plus reviewed migration
    inventory.
- [ ] `OPEN_BLOCKING SK-04` — Freeze the TGVF Adapter boundary.
  - Inputs: exact `Hq` contract, original-image pre-merge features, all
    DeepStack branches, geometry/mask/config: `[TBD schema]`
  - Outputs: main `D`, all D-DeepStack tensors, metadata: `[TBD schema]`
  - Frozen Qwen merger ownership and artifact identity: `[TBD]`
- [ ] `OPEN_BLOCKING SK-05` — Freeze a versioned framework-neutral trajectory
  record interface with reserved fields for tokens, ownership masks, behavior
  log probabilities, per-call observations, identities, rewards, and stops.
  - Exact dataclass/schema: `[TBD]`
- [ ] `OPEN_BLOCKING SK-06` — Freeze configuration/identity plumbing.
  - Every prompt, schema, template, model, Adapter checkpoint, data manifest,
    reward, objective, backend, and code state has a version/hash.
  - Unset research values fail closed rather than receiving defaults.
- [ ] `OPEN_CONFIGURABLE SK-07` — Compact package/config labels.
  - Formal prose names remain fixed.
  - Candidates: `representation` and `policy_rl`.
  - Accepted short labels: `[TBD]`
- [x] `FIXED SK-08` — The first policy RL path exposes only a frozen TGVF
  Adapter. A joint-update extension may be reserved but not implemented as an
  active mode without Gate J0.
- [x] `FIXED SK-09` — SDPO may have an isolated extension boundary, but it does
  not block the GRPO skeleton and stays disabled until Gate D0.

## 5. Gate R0 — Before native rollout is accepted

### 5.1 Model, template, and protocol identity

- [ ] `OPEN_BLOCKING RO-P01` — Pin exact Qwen model, processor, tokenizer, chat
  template, and their hashes. Value: `[TBD]`
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

## 6. Gate V0 — Before the veRL compatibility spike

- [ ] `OPEN_BLOCKING VS-01` — Accept a bounded spike task in
  `docs/PROJECT_TASK.md`; veRL selection itself is not reopened.
- [ ] `OPEN_BLOCKING VS-02` — Candidate upstream veRL commits and isolated
  dependency matrices: `[TBD]`
- [ ] `OPEN_BLOCKING VS-03` — Candidate rollout backends: `[TBD: SGLang/vLLM]`
- [ ] `OPEN_BLOCKING VS-04` — Minimal FSDP2 configuration(s), model fixture,
  device mesh, and state-dict strategy: `[TBD]`
- [ ] `OPEN_BLOCKING VS-05` — Public extension surface to test: `[TBD]`
- [ ] `OPEN_BLOCKING VS-06` — Numerical tolerances and PASS/FAIL criteria for:
  Qwen policy/reference forward, two-call latent-observation transport, actual
  behavior logprobs, exact observation replay, FSDP2 one step, save/resume.
- [ ] `OPEN_BLOCKING VS-07` — Failure conditions include private-trainer forks,
  PIL re-encoding of `D`, missing actual sampling logprobs, or inability to
  preserve exact observation state.
- [ ] `OPEN_BLOCKING VS-08` — Complete `PLANNED` ledger entry before any GPU use.

Required output: `docs/VERL_COMPATIBILITY_REPORT.md` containing PASS/FAIL
evidence, selected commit/backend, dependency matrix, required public extension
surface, FSDP2 topology evidence, and unresolved blockers.

## 7. Gate A0 — Before native-format representation training

- [ ] `OPEN_BLOCKING AD-01` — Exact retained representation dataset manifest,
  provenance, license, splits, hashes, and exclusions: `[TBD]`
- [ ] `OPEN_BLOCKING AD-02` — New canonical native-format sample schema and
  transform from retained data: `[TBD]`
- [ ] `OPEN_BLOCKING AD-03` — New pipeline transcript/prompt construction and
  `Hq` contract: `[TBD]`
- [ ] `OPEN_BLOCKING AD-04` — TGVF Adapter initialization that does not use the
  historical trained checkpoint directly: `[TBD]`
- [ ] `OPEN_BLOCKING AD-05` — Exact representation losses, weights,
  normalization, sampling, and optimization: `[TBD]`
- [ ] `OPEN_BLOCKING AD-06` — Trainable/frozen parameter whitelist, including
  original vision tower and Qwen mergers: `[TBD]`
- [ ] `OPEN_BLOCKING AD-07` — Checkpoint artifact schema and identity, excluding
  optimizer state and legacy protocol-token rows from the deployable Adapter
  artifact: `[TBD]`
- [ ] `OPEN_BLOCKING AD-08` — Numerical output/gradient parity of the extracted
  TGVF Adapter core against its pinned reference: `[TBD tolerance]`
- [ ] `OPEN_BLOCKING AD-09` — Target specificity, readout, causal flip, and free
  continuation thresholds: `[TBD]`
- [ ] `OPEN_BLOCKING AD-10` — Main `D` and every D-DeepStack branch pass the same
  controlled semantic gates.

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
- [ ] `OPEN_BLOCKING PR-02` — Exact base model/processor snapshot: `[TBD]`
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

## 10. Gate GPU0 — Before any GPU command

- [ ] `OPEN_BLOCKING GPU-01` — Complete `PLANNED` entry in
  `docs/EXPERIMENT_LEDGER.md`.
- [ ] `OPEN_BLOCKING GPU-02` — Record code commit/dirty state, base model,
  processor, TGVF Adapter checkpoint, prompt/schema/template, data, reward,
  objective, and output identities.
- [ ] `OPEN_BLOCKING GPU-03` — Approved question, fixture, command, hardware,
  PASS/FAIL thresholds, timeout, early stop, and recovery plan.
- [ ] `OPEN_BLOCKING GPU-04` — Pin PyTorch, CUDA, NCCL, Transformers, attention
  kernel, veRL, rollout backend, driver/container, and environment lock.
- [ ] `OPEN_BLOCKING GPU-05` — Record FSDP2 device mesh, sharding/wrap policy,
  mixed precision, activation checkpointing, offload, state-dict strategy, and
  LoRA handling.
- [ ] `OPEN_BLOCKING GPU-06` — Record actor/reference/rollout/TGVF placement.
  Rollout tensor parallelism and training FSDP2 are separate identities.
- [ ] `OPEN_BLOCKING GPU-07` — Tiny smoke covers two tool calls, exact `D`
  transport, logprob parity, one-step loss/gradient, save, and resume.
- [ ] `OPEN_BLOCKING GPU-08` — Estimate per-call `D` artifact GPU/CPU/disk/network
  footprint and I/O cost.
- [ ] `OPEN_BLOCKING GPU-09` — Before a long pilot, record tokens/s, tool latency,
  update latency, peak memory, utilization, and total-duration estimate.

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

## 12. Gate D0 — Deferred SDPO

Status: `DEFERRED` until the GRPO path is correct.

- [ ] `DEFERRED SD-01` — Exact paper, repository commit, equations, and intended
  meaning of SDPO: `[TBD]`
- [ ] `DEFERRED SD-02` — Teacher context, live/EMA teacher identity, token
  alignment, and multimodal masks: `[TBD]`
- [ ] `DEFERRED SD-03` — Full-logit/top-k/sampled-token approximation and loss
  composition with GRPO: `[TBD]`
- [ ] `DEFERRED SD-04` — Teacher/EMA checkpoint and resume contract: `[TBD]`
- [ ] `DEFERRED SD-05` — Official text parity, one-step update parity, padding/
  mask tests, then tiny multimodal parity.

## 13. Next document actions

- [ ] Accept or revise Gate S0 skeleton interfaces in `docs/PROJECT_TASK.md`.
- [ ] Approve a bounded veRL compatibility-spike task before dependency changes.
- [ ] Freeze the first legacy representation file/data inventory.
- [ ] Convert each accepted `[TBD]` into a versioned project artifact rather
  than embedding it only in code or an experiment command.
