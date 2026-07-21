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
- Except where Policy Pilot v1 §1.2 now freezes them, exact data, reward
  coefficients, and prompt wording may be late-bound. Token
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

Under the earlier I8H scope, no data or Adapter artifact was accepted for
production use. Policy Pilot v1 §1.2 now supersedes that statement only for its
DeepEyes source snapshot, GRPO/reward mathematics, LoRA envelope, and required
72B-judge activation. Dataset/image-license evidence, Pilot manifest
materialization/hash, the policy-RL prompt, TGVF Adapter artifact/provider,
optimizer/scheduler, exact judge service/prompt/calibration, production SDPO or
hybrid mathematics, and long training remain open and fail closed. The
representation user-message structure itself is fixed separately by
`RPI-20260720-REPRESENTATION-NATIVE-TRAJECTORY`. A synthetic optimizer smoke
may run only after its exact test-contract equations and pure-tensor oracle are
recorded.

Where older text in this file or `VERL_COMPATIBILITY_SPIKE_PLAN.md` conflicts,
this section and Project Task §0 take precedence. The earlier proposal status,
full mandatory cell matrix/overall hard-PASS requirement, SGLang/stable cells,
`SDPO_STATIC_SEAM_READY` stopping point, per-cell request for another user
reply, and separate post-report skeleton approval are superseded. Technical
criteria still apply to any cell that claims the corresponding evidence;
exact-identity records and ledger requirements remain in force. Unrun original
matrix items remain promotion gates rather than implicit passes.

### 1.2 Authoritative Policy Pilot v1 contract

Decision ID: **POLICY-PILOT-V1-20260720**

Accepted by the user on **2026-07-20 JST**. The complete authoritative 13-item
contract and equations are in
[`PROJECT_TASK.md` §0.8](PROJECT_TASK.md#08-authoritative-policy-pilot-v1-runtime-envelope).
For Policy Pilot v1, that decision supersedes older `[TBD]` or conflicting
Pilot text without deleting the historical register. In compact form it fixes:

1. Qwen3-VL-8B-Thinking, native DeepStack enabled, and only
   `tgvf_focus_tool`; crop/TGVF fusion is deferred.
2. `ChenShawn/DeepEyes-Datasets-47k` snapshot
   `5546681e28fa2eda9f60a9ea9dd0cf291216ded3`, three files/47,052 rows, with
   the historical zoom prompt disabled.
3. Original-image processor `max_pixels=262144` plus actual visual-token,
   call-attempt, and step-time telemetry.
4. `n=8`, temperature `1.0`, `top_p=1.0`, top-k disabled, and all repetition,
   frequency, and presence penalties disabled.
5. At most one TGVF call per assistant action turn and four admitted call
   attempts per trajectory; success and standard tool error both consume an
   attempt, while the unexecuted fifth attempt receives the standard cap error.
6. `max_response_length=8192` policy-generated tokens across the complete
   response; template, observation/image, and padding tokens have loss mask
   zero.
7. Actual transformed-distribution behavior logprobs, zero staleness, frozen
   base reference without LoRA, and exact rollout-recorded `D` replay without
   observation recomputation.
8. Group mean plus sample standard deviation,
   `A=(r-mean)/(sample_std+1e-6)`, all-equal rewards mapped to zero advantages,
   no group filtering/dropped trajectories, and trajectory advantage broadcast
   only to its policy tokens.
9. Current/behavior ratio, symmetric clip epsilon `0.2/0.2`, dual clip `c=3`,
   global policy-token mean, one update epoch, entropy coefficient zero,
   maximum gradient norm `1`, and optimization KL coefficient zero.
10. `R=0.8*answer_reward + 0.2*format_reward +
    1.2*conditional_tool_reward`, with the component semantics and once-per-
    trajectory conditional-tool cap in Project Task §0.8.
11. Language-decoder-only LoRA, rank/alpha `64/64`, dropout zero, initial
    learning rate `1e-5`, with vision/merger/native-DeepStack/TGVF-Adapter/
    input-embedding/`lm_head` frozen.
12. Formal-Pilot `Qwen/Qwen2.5-72B-Instruct` judging enabled: MCQ rule/exact,
    required math semantic fallback, and optional open-VQA semantic fallback.
13. A strict run/checkpoint identity binds every selected artifact, state, and
    resume cursor named in Project Task §0.8.

### 1.3 Policy Pilot v1 decisions still open

The following user decisions were **not** selected by the Pilot v1 decision and
remain fail-closed. No library or historical-project default may fill them:

- `PPV1-O01` — physical hardware assignment, actor/reference/rollout worker
  placement, device mesh, FSDP2 sharding, tensor parallelism, and the concrete
  parallel topology;
- `PPV1-O02` — the materialized/filtered manifest and hash for the already
  fixed DeepEyes snapshot, held-out leakage report, and numeric shuffle seed;
- `PPV1-O03` — exact processor-rendered policy-RL prompt bytes/hash,
  stop/EOS ownership, min-p/stop settings, rollout RNG seed/derivation, exact
  cap-error bytes/hash, and post-cap-error recovery/termination semantics;
- `PPV1-O04` — the native TGVF Adapter artifact/checkpoint and the active
  target-conditioning provider for this Pilot;
- `PPV1-O05` — the estimator and normalization used for diagnostic
  current-policy versus frozen-base-reference KL. Its optimization coefficient
  is already fixed to zero;
- `PPV1-O06` — optimizer, scheduler, precision/scaler, implementation of the
  fixed gradient-norm bound, minibatch/accumulation, and weight-sync details
  beyond the fixed `1e-5` learning rate, maximum gradient norm one, one update
  epoch, and zero staleness; and
- `PPV1-O07` — service, prompt, sampling, calibration, and failure-policy
  identities for the required formal-Pilot `Qwen/Qwen2.5-72B-Instruct` judge.
  Judge activation/model/routing are fixed, not open.

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
  accepted comparison uses contextual hidden state layer `-1` first and target
  token embedding second, in separate runs with identical sample/group order,
  fresh initialization, batch plan, and seed `42`.
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
- [x] `FIXED` — Native function names are `tgvf_focus_tool`,
  `image_zoom_in_tool`, and `tgvf_crop_tool`. Crop takes one required
  `bbox_2d` integer array plus optional string `label`; the atomic crop/TGVF
  call takes the array plus one non-empty `target` string. Exact schemas,
  descriptions, canonical JSON, hashes, and strict parser fixtures are owned
  by the protocol package under `ATOMIC-CROP-TGVF-20260721` and
  `TGVF-VISUAL-TOOL-PROMPTS-V1-20260721`.
- [x] `FIXED` — `crop_only`, `tgvf_only`, and `crop_tgvf` are distinct
  configuration profiles exposing only `image_zoom_in_tool`,
  `tgvf_focus_tool`, and `tgvf_crop_tool`, respectively. Policy Pilot v1
  remains explicitly fixed to `tgvf_only`.
- [x] `FIXED` — A trajectory supports zero or more ordered TGVF/crop calls; one
  shared safety cap is configurable and greater than one in a later fusion
  experiment. Policy Pilot v1 enables only `tgvf_focus_tool`, admits four TGVF
  call attempts counting success and standard tool error, and returns an
  environment-owned cap error without executing the fifth attempt.
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
  consume the same rollout-materialized observation: exact `D` state or exact
  crop pixels. Reuse of rollout-time crop visual features requires an explicit
  `shared_frozen_recorded_features` identity/freeze proof.
- [x] `FIXED RO-CROP-01` — Policy Pilot v1 freezes the Qwen vision encoder,
  visual merger and native DeepStack modules, so rollout-recorded crop features
  are the exact shared policy/reference replay state. A future full-fine-tuning
  identity that makes any of these components trainable must instead implement
  exact-pixel per-consumer re-encoding and behavior-forward parity; it may not
  reuse this frozen-feature fast path.
- [x] `FIXED` — SDPO compatibility is part of the initial skeleton. Its reference
  repository is `lasgroup/SDPO` at
  `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`; its bundled veRL tree is not the
  selected production framework.
- [x] `FIXED` — Pure GRPO, reference-style pure SDPO, and any future hybrid are
  distinct objective identities. The reference SDPO implementation replaces
  the policy loss and does not define a GRPO-plus-SDPO sum.
- [x] `FIXED` — An external answer judge, frozen RL reference, and SDPO
  self-teacher are separate roles. The formal Policy Pilot v1 requires
  `Qwen/Qwen2.5-72B-Instruct` judging with the routing in §1.2; its exact
  service/prompt/sampling/calibration/failure-policy artifacts remain open.
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
- [x] `SUPERSEDED C-12` — “The first Pilot tool-call cap and cap-hit behavior
  are entirely `[TBD]`.” Replacement: Policy Pilot v1 enables only TGVF and
  admits four attempts, counting both success and standard tool error; it
  returns an environment-owned cap error without executing the fifth attempt.
  Exact error bytes and recovery/stop fixture remain `PPV1-O03`.
- [x] `SUPERSEDED C-13` — “The first Pilot may inherit sampling count,
  temperature, filtering, penalties, response length, staleness, or update
  epochs from framework defaults.” Replacement: the values in §1.2 are
  authoritative; every remaining sampling identity is explicit under
  `PPV1-O03`.
- [x] `SUPERSEDED C-14` — “DeepEyes 47K is only a first-Pilot candidate.”
  Replacement: Policy Pilot v1 fixes the exact snapshot in §1.2; only its
  materialized manifest/hash, leakage artifact, and shuffle-seed number remain
  open.
- [x] `SUPERSEDED C-15` — “Pilot GRPO mathematics, reward weights, LoRA scope,
  and 72B judge activation are open.” Replacement: §1.2 and Project Task §0.8
  freeze those decisions. Only the separately enumerated implementation/run
  identities in §1.3 remain open.

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

- [x] `FIXED RO-P01` — The accepted local Qwen3 model path, tokenizer length,
  chat-template identity, family adapter, and real-processor prompt/token
  fixture are pinned by `qwen3_visual_tool_prompts_v1.json`. Full weight-shard
  hashes are not required.
- [x] `FIXED RO-P02A` — Exact native schemas, descriptions, versions, canonical
  JSON, and schema hashes are implemented for `tgvf_focus_tool(target)`,
  `image_zoom_in_tool(bbox_2d,label?)`, and the atomic
  `tgvf_crop_tool(bbox_2d,target)` under decisions `CROP-FUSION-20260720`,
  `ATOMIC-CROP-TGVF-20260721`, and
  `TGVF-VISUAL-TOOL-PROMPTS-V1-20260721`.
- [x] `FIXED RO-P02B` — The real Qwen3 processor-rendered prompt/token goldens
  and tool-schema hashes for all three profiles are pinned in
  `qwen3_visual_tool_prompts_v1.json`. The representation-phase TGVF-only
  golden identity remains unchanged.
- [x] `FIXED RO-P03` — The initial policy prompt is
  `tgvf-visual-tool-prompts-v1`; exact source text and profile bundle hashes are
  recorded in `TGVF_VISUAL_TOOL_PROMPTS_V1.md`.
- [ ] `OPEN_BLOCKING RO-P04` — Golden token-ID fixtures for direct answer, one
  call, repeated calls, four admitted attempts, and the unexecuted fifth-attempt
  cap-error response.
- [x] `FIXED RO-P05` — The real-processor fixture proves tokenizer length stays
  `151669`; the implementation adds no tokens and performs no embedding or
  lm-head resize.
- [ ] `OPEN_BLOCKING RO-P06` — Freeze assistant prefill and stop semantics,
  including ownership of `</tool_call>`, `<|im_end|>`, think closure, and EOS.
- [x] `FIXED RO-P07` — Real Qwen direct, one-call, and repeated-call goldens
  prove exactly one template-owned `<think>` opener per assistant turn; the
  runtime records a policy-sampled duplicate opener as an invalid-format
  trajectory without dropping its sampled tokens or behavior log probabilities.

### 5.2 Parser, target span, and multi-call state machine

- [ ] `OPEN_BLOCKING RO-S01` — Strict JSON/parser behavior for malformed calls,
  unknown keys, trailing answers, tool errors, timeout, and loops.
- [x] `FIXED RO-S02` — Exact target value span mapping uses
  `minimal_overlapping_sampled_token_cover_v1`: raw JSON character/UTF-8 byte
  offsets remain exact, while conditioning uses the unique minimal contiguous
  sequence of actual sampled tokens with non-empty byte overlap. This covers
  escapes, non-ASCII text, repeated strings, and tokenizer boundary crossings;
  no retokenization, fuzzy matching, target rewriting, or row filtering is
  permitted. Fixed by `RPI-20260720-TARGET-TOKEN-COVER-V1`.
- [x] `FIXED RO-S02A` — The same exact target-span rule applies to the atomic
  `tgvf_crop_tool` target. Its parser also requires exactly four JSON integers,
  positive requested width/height, no unknown argument, and retains the exact
  sampled call bytes/tokens. Source-bound clamping and empty-effective-box
  rejection remain environment responsibilities.
- [x] `FIXED RO-S02B` — Crop-capable trajectories record dataset/file identity
  separately from the decoded RGB tensor digest and require the source visual
  binding to name that exact decoded digest. Both crop observation kinds retain
  processor/layout identities, and atomic execution is bound to the loaded
  representation manifest's provider, Adapter architecture, and branch
  projections rather than a caller-supplied artifact label.
- [ ] `OPEN_BLOCKING RO-S02C` — Live vLLM next-turn integration must freeze the
  identity mapping from the canonical native-image placeholder sequence to
  vLLM's processor-expanded prompt. The installed plugin consumes recorded
  main/DeepStack embeddings, but the current live client compares the backend's
  expanded echoed token IDs directly with the canonical request and therefore
  fails closed. A concrete initial-trajectory source-payload builder and a real
  `LLM.generate` gate are also still required; resolver/packer smoke is not
  reported as live generation.
- [ ] `OPEN_BLOCKING RO-S03` — `Hq` identity:
  - included/excluded syntax tokens: `[TBD]`
  - hidden layer: `[TBD]`
  - token-time alignment: `[TBD]`
  - pooling/sequence contract: `[TBD]`
- [ ] `OPEN_BLOCKING RO-S04` — Implement and fixture the Policy Pilot v1 cap
  selected in §1.2: four admitted TGVF attempts, with successful execution and
  standard tool error both consuming the bound; the fifth attempted call is not
  executed and receives one environment-owned standardized cap-exceeded error.
  Exact error bytes/hash and recovery/termination after that response remain
  `PPV1-O03`; timeout and malformed-action behavior remain separately open.
  Later experiments may select a different explicit cap or enable crop fusion.
- [ ] `OPEN_BLOCKING RO-S05` — Multi-call cache/history contract, including
  append/reset/reuse behavior for text KV, original image, main `D`, and all
  D-DeepStack branches.
- [ ] `OPEN_BLOCKING RO-S06` — No decode/rerender/retokenize drift. If a backend
  forces rerendering, whole-token, visual-layout, position, mask, and cache
  parity is required.

### 5.3 Token ownership and loss masks

- [ ] `OPEN_BLOCKING RO-M01` — Implement and prove mutually exclusive,
  exhaustive ownership masks under the fixed Pilot v1 semantics:
  template-owned, policy-sampled, environment/tool-observation, and padding.
- [ ] `OPEN_BLOCKING RO-M02` — Under the fixed Pilot v1 semantics, policy/loss
  mask includes every actual sampled token in every assistant turn and excludes
  template prefixes, tool responses (including the cap error), image positions,
  and padding.
- [ ] `OPEN_BLOCKING RO-M03` — Freeze stop/EOS tokens' ownership and whether each
  participates in behavior logprob and loss.
- [ ] `OPEN_BLOCKING RO-M04` — Single, batched, zero/one/repeated/four-attempt,
  and fifth-cap-error mask fixtures have exact expected token counts and prove
  every tool/error observation remains outside policy loss.

### 5.4 Actual behavior log probabilities

- [ ] `OPEN_BLOCKING RO-L01` — Store the actual behavior log probability for
  every policy-sampled token. Length must equal the sampled-token mask count.
- [ ] `OPEN_BLOCKING RO-L02` — Explicitly forbid replayed
  `new_logprobs.detach()` as behavior log probabilities.
- [ ] `OPEN_BLOCKING RO-L03` — Implement and prove the fixed Pilot convention:
  the stored behavior logprob and GRPO ratio denominator are the actual
  post-temperature/filter/penalty/processor sampling-distribution logprobs.
  Any raw-model-distribution logprob is separately named diagnostic data and
  may never replace the behavior denominator; whether to retain that optional
  diagnostic remains a run-storage choice.
- [ ] `OPEN_BLOCKING RO-L04` — Record processor ordering and any stateful
  repetition/frequency/presence penalty state.
- [ ] `OPEN_BLOCKING RO-L05` — Logprob dtype, storage format, finite-value
  checks, missing-value behavior, and numerical tolerance: `[TBD]`

### 5.5 Sampling identity

- [x] `FIXED POLICY-PILOT-V1 RO-I00` — Pilot v1 uses `n = 8`, temperature
  `1.0`, `top_p = 1.0`, top-k disabled, repetition penalty disabled, frequency
  and presence penalties disabled, and a policy-generated response-token budget
  of `8192` across the multi-turn trajectory. This is not a total-context limit.
  Min-p, stop identities, and RNG seed/derivation remain `PPV1-O03`.
- [ ] `OPEN_BLOCKING RO-I01` — Each trajectory records:
  - rollout policy/checkpoint/adapter and weight-sync version;
  - veRL commit and rollout backend/version;
  - temperature, top-p, top-k, min-p, penalties, processors, and stops;
  - max tokens and max turns;
  - global/per-sample seed, RNG derivation/state, rank, and worker;
  - raw-versus-transformed logprob convention;
  - rollout/update staleness.
- [ ] `OPEN_BLOCKING RO-I02` — Implement and prove the accepted Pilot v1
  staleness/update barrier: `staleness = 0`, with no update between sampling and
  consuming the trajectory's behavior log probabilities.

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
  question text, target text, `evidence_description`, `short_answer`, optional
  `image_id`, and the recorded evaluation metadata.
  `representation_sample_identity_v1` requires a non-empty `short_answer` and
  binds it into the immutable content digest. It is the initial accepted sample
  identity; no answer-omitting predecessor or replay compatibility identity is
  supported. The group key is `image_id` with exact image-reference fallback.
  The exact retained data has no separate `choices` field; a future source with
  choices requires a new schema/transform version rather than implicit prompt
  rendering inside this record.
- [x] `FIXED AD-02B` — `retained_focus_rows_v1` implements source-hash
  validation, strict focus metadata, fail-closed fields/image resolution,
  duplicate handling, leakage records/warnings, every-row accepted/excluded
  disposition, immutable manifest identity, four-key split-overlap reports, and
  the non-empty `short_answer` admission rule. It emits
  `representation_sample_identity_v1` for the accepted native trajectory. This
  is the initial accepted transform; there is no executable earlier transform
  path to retain.
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
- [ ] `OPEN_BLOCKING AD-03` — The Qwen3 representation trajectory is fixed;
  the production `Hq` and remaining model/provider evidence are not fully
  closed:
  - [x] `FIXED AD-03A` — `evidence_description` is the reasoning content of the
    assistant turn after the latent `tgvf_focus_tool` result, and the exact
    dataset `short_answer` is the answer content of that same turn. The user
    turn is exactly the original image plus unmodified dataset question. It
    adds no separate target field, tutorial, focus-force instruction, or other
    project-authored text; any question/target lexical overlap can only be
    inherited from the original question. The preceding teacher-constructed
    assistant turn contains the fixed target-independent reasoning `I need
    visual focus before answering.` and one native tool call carrying the
    dataset target.
    Only the exact tokenizer positions owned by the rendered
    `evidence_description` span receive teacher-forcing labels. The fixed
    pre-tool reasoning, tool-call JSON/target, tool result, `short_answer`,
    template wrappers, and visual positions are ignored by `L_gen`.
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
    This supervision contract is `canonical_evidence_supervision_v1`; it is the
    initial accepted canonical schema and has no answer-omitting predecessor.
  - [x] `FIXED AD-03B` — The Qwen3 implementation constructs one original-image
    question-only user turn, one native `tgvf_focus_tool` action, one
    latent-image tool result, and the evidence-reasoning-plus-answer turn. The
    strict parser identifies the raw JSON target span and exact IDs before/after
    processor expansion. The accepted
    `qwen3_native_representation_smoke_v1.json` processor golden uses a
    deterministic 56×56 RGB image and Unicode/slash target and freezes the
    question-only trajectory, final-answer placement, evidence-only labels,
    target span, expanded input, two visual blocks, and expanded visual
    positions against the accepted local processor. Its prompt identity is
    `qwen3-representation-image-question-v1` under schema
    `native_representation_prompt_v1`. The earlier target-bearing golden and
    renderer branch were never accepted and are removed rather than preserved
    as an executable historical version.
  - [x] `FIXED AD-03C` — The first representation run selects contextual hidden
    state at layer `-1`; the next paired run selects target token embedding.
    Both bind `qwen3-representation-image-question-v1` /
    `native_representation_prompt_v1` and differ only by provider identity.
    Separately injecting the teacher target into the user message is forbidden.
    The policy-RL system/tool-use prompt remains a separate open contract.
- [x] `FIXED AD-04` — TGVF Adapter initialization uses the repository's fresh
  constructor under seed `42`; the historical trained checkpoint is never a
  direct initialization. The paired provider runs share this seed and initial
  data/order contract.
- [x] `FIXED AD-05` — The representation objective and execution contract is
  closed for the initial paired runs. Its independently recorded parts are:
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
    separate module. The nonzero baseline coefficient is fixed at `1.0` under
    `AD-05F`.
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
  - [x] `FIXED AD-05F` — The trainer/config/checkpoint code exposes
    explicit nonzero Matrix-CE and `L_gen` weights, AdamW options, scheduler,
    precision, accumulation, clipping, validation/log cadence, and strict
    optimizer-boundary resume. CPU fixtures prove an accumulated optimizer step
    and bitwise-identical next step after resume. The accepted run pair uses
    Matrix CE/L_gen/Norm weights `1.0/1.0/0.1`, balanced temperature `1.0`,
    manifold zero, AdamW LR `1e-4`, cosine/100-step warmup/min-ratio `0.1`,
    K=4, per-rank batch 4, two ranks, GA=4, global batch 32, 2,000 steps,
    log-every 10, validate/save-every 500, BF16, clipping `1.0`, seed `42`, and
    `image_max_pixels=262144`. Training uses the pinned v4 clean-imend focus
    split and validation uses pinned v3 val-2k. Both production TOMLs,
    materialized manifests, overlap report and final artifact paths are bound.
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
- [x] `FIXED AD-08` — The accepted parity scope is deliberately small: retain
  the independent small-shape output/input/owned-parameter gradient oracle and
  run one fixed real-shape finite/shape/main-plus-branch wiring and nonzero-
  gradient check. Exact full-dimension equality to the historical trained
  checkpoint is not required because the native target context is different.
- [x] `FIXED AD-09` — Reuse the historical internal metric definitions and
  Golden report as the comparison baseline rather than inventing new absolute
  thresholds. The reference report records correct-D beat rates of `1.0`
  versus target-only/random, `0.9045` versus wrong-same-image, `0.9592` versus
  wrong-different-image; retrieval top-1/top-2 `0.7778/0.9471`, MRR `0.8794`,
  finite rate `1.0`, and collapse rate `0.0`. Native causal/free-continuation
  additions receive simple directional/finite sanity checks and retained
  outputs, not an invented historical threshold.
- [ ] `OPEN_BLOCKING AD-10` — Synthetic sensitivity/gradient tests cover main
  `D` and every D-DeepStack branch. All branches must still pass real-Qwen
  controlled semantic gates and accepted thresholds.
- [x] `FIXED AD-11` — Both required target-conditioning providers pass
  through one explicit shared configuration/runtime/native group-builder path,
  and provider identity is checkpoint-bound. Both have synthetic target-span,
  shape, determinism, and training-path fixtures. Corrected `RP-10` proves a
  real-Qwen optimizer path for target-token-embedding conditioning. A paired
  first real run uses contextual hidden state layer `-1`; the next paired run
  uses target token embedding with identical data/order, initialization seed
  `42`, objective, batch plan, and cadence. Each artifact retains its provider
  identity and its own specificity/readability report.
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
- [x] `FIXED AD-14` — Run contextual hidden state first and target token
  embedding second as a paired comparison. They use identical data/group
  order, fresh Adapter initialization, batch plan, seed `42`, and separate
  artifact identities.

## 8. Gate G0 — Before any GRPO optimizer step

This gate applies even to a one-step smoke test.

### 8.1 Exact GRPO mathematics

- [ ] `OPEN_BLOCKING GR-01` — Implement and prove the fixed Pilot grouping:
  eight sampled trajectories per prompt remain in the original group; no group
  filter, low-reward drop, or invalid-trajectory removal is permitted. Scalar
  reward is the fixed `RW-06` equation.
- [ ] `OPEN_BLOCKING GR-02` — Implement and prove group mean plus **sample**
  standard deviation (`N-1` denominator), not population standard deviation.
- [ ] `OPEN_BLOCKING GR-03` — Implement and prove
  `A_i=(r_i-mean(r))/(sample_std(r)+1e-6)`; all-identical group rewards produce
  exactly zero advantages. Each trajectory scalar is broadcast only to its
  policy-generated tokens.
- [ ] `OPEN_BLOCKING GR-04` — Implement and prove
  `rho_t=exp(log pi_current,t - log pi_behavior,t)` using the actual
  rollout-recorded, post-sampling-transform behavior logprob.
- [ ] `OPEN_BLOCKING GR-05` — Implement and prove symmetric PPO clip epsilon
  `0.2/0.2` (ratio interval `[0.8,1.2]`) plus dual clip `c=3` on negative
  advantages: `s=min(rho*A,clip(rho,0.8,1.2)*A)`, then use `max(s,3*A)` for
  `A<0` and `s` otherwise; policy loss is the negative masked mean.
- [ ] `OPEN_BLOCKING GR-06` — Implement and prove one global token mean over all
  policy-generated tokens. Template prefixes, tool/error responses, image
  positions, and padding are excluded; there is no sequence-mean or two-level
  reduction in Pilot v1.
- [ ] `OPEN_BLOCKING GR-07` — Reference identity is fixed to the original frozen
  Qwen3-VL-8B-Thinking base without policy LoRA. Freeze the still-open estimator,
  mask, and normalization for diagnostic current/reference KL: `[TBD equation]`.
  Policy Pilot v1 does not select a KL reward/loss contribution; adding one
  requires a separate accepted objective contract.
- [x] `FIXED POLICY-PILOT-V1 GR-08` — KL coefficient is zero in both reward and
  loss. Current-versus-frozen-base KL is logged only as a diagnostic; its
  estimator remains `PPV1-O05`. Double counting is forbidden.
- [x] `FIXED POLICY-PILOT-V1 GR-09` — Entropy coefficient and all other
  auxiliary optimization terms are zero.
- [ ] `OPEN_BLOCKING GR-10` — Policy Pilot v1 fixes update epochs to exactly
  one and maximum gradient norm to `1`. Minibatch/accumulation mechanics,
  policy-version mechanics beyond zero staleness, precision/scaler, and
  overflow/NaN handling remain run/implementation identities.
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
- [ ] `OPEN_BLOCKING PR-03` — Reference model is the original frozen
  Qwen3-VL-8B-Thinking base with no policy LoRA. Bind the still-open exact
  Pilot prompt/schema golden and verify the reference consumes the same
  rollout-recorded observation as policy replay.
- [ ] `OPEN_BLOCKING PR-04` — Implement and record the fixed Pilot whitelist:
  LoRA on Qwen3 language-decoder modules only, rank `64`, alpha `64`, dropout
  `0`. The exact decoder module names are a pinned architecture/implementation
  artifact rather than a research choice; no broad regex may capture frozen
  modules.
- [x] `FIXED PR-05` — TGVF Adapter frozen for the first policy RL proof.
- [ ] `OPEN_BLOCKING PR-06` — Prove Pilot freeze state for the vision encoder,
  visual merger, native DeepStack modules, TGVF Adapter, input embeddings, and
  `lm_head`; only the accepted decoder LoRA is trainable.
- [ ] `OPEN_BLOCKING PR-07` — Initial learning rate is fixed to `1e-5`.
  Optimizer, scheduler, precision/scaler, minibatch/accumulation mechanics, and
  rollout weight-sync implementation remain `PPV1-O06`; zero staleness is
  already fixed.

## 9. Gate P0 — Before the first policy RL pilot

These choices may remain open while building the skeleton, but not when the
pilot is identified or launched.

### 9.1 Data

- [ ] `OPEN_BLOCKING DA-01` — Materialize the fixed Pilot source into the
  repository canonical schema and record its license/provenance audit:
  `ChenShawn/DeepEyes-Datasets-47k` at Hugging Face snapshot
  `5546681e28fa2eda9f60a9ea9dd0cf291216ded3`, exactly three source files and
  47,052 source records.
- [x] `FIXED POLICY-PILOT-V1 DA-02` — DeepEyes 47K is the formal Pilot source,
  not a candidate. The historical rendered zoom/crop prompt is discarded; this
  project's TGVF-only native prompt is rendered at runtime.
- [ ] `OPEN_BLOCKING DA-03` — Materialize and hash the exact Pilot manifest and
  leakage report, then bind the still-open numeric shuffle seed. Each prompt
  produces one `n=8` group; there is no source mixture or group filtering.
- [ ] `OPEN_BLOCKING DA-04` — Exact/near-duplicate image and normalized-question
  leakage checks against held-out evaluation.
- [ ] `OPEN_BLOCKING DA-05` — Bind broken/ambiguous-row disposition and answer
  types while materializing `PPV1-O02`; all retained/excluded IDs enter the
  manifest hash. Verifier routing follows fixed `RW-01` rather than becoming a
  separate data-dependent reward rule.
- [x] `FIXED DA-06` — Canonical samples do not contain one irreversible rendered
  prompt; prompts are versioned at runtime.
- [ ] `OPEN_BLOCKING DA-07` — Implement and fixture original-image processor
  `max_pixels=262144`; record actual original-image visual tokens, total
  trajectory visual tokens, mean call attempts, and step time. This is an
  initial cost/distribution-alignment setting, not a final-method claim.

### 9.2 Reward

- [ ] `OPEN_BLOCKING RW-01` — Implement/version final-answer extraction and the
  fixed `answer_reward`: correct final answer `1`, otherwise `0`. MCQ uses
  deterministic rule/exact scoring; math uses required 72B semantic fallback,
  and open-ended VQA may use 72B semantic fallback.
- [ ] `OPEN_BLOCKING RW-02` — Implement the fixed `format_reward`: valid
  protocol plus valid final answer `0`; invalid protocol or no valid final
  answer `-1`. Tool errors/cap events retain typed logs but receive no separate
  differentiated reward or penalty.
- [x] `FIXED POLICY-PILOT-V1 RW-03` — There is no separate per-call, token,
  latency, malformed-call, cap-hit, or tool-error cost. Only `RW-04` supplies a
  tool-conditional component.
- [ ] `OPEN_BLOCKING RW-04` — Implement `conditional_tool_reward=1` exactly when
  the final answer is correct and at least one successful TGVF observation is
  present; otherwise `0`. Award it at most once per trajectory.
- [ ] `OPEN_BLOCKING RW-05` — Formal-Pilot judging is enabled with
  `Qwen/Qwen2.5-72B-Instruct` and the routing in `RW-01`. Bind its still-open
  provider/service, prompt, sampling, calibration, and failure-policy identity.
  It is not the RL reference or the SDPO teacher.
- [ ] `OPEN_BLOCKING RW-06` — Implement, separately log, and fixture
  `R=0.8*answer_reward + 0.2*format_reward +
  1.2*conditional_tool_reward`; no hidden component or clipping is permitted.

### 9.3 Prompt and tool policy

- [x] `FIXED PM-01` — Exact three-profile system prompts, shared user prompt,
  successful response text, versions, and hashes are fixed by
  `TGVF-VISUAL-TOOL-PROMPTS-V1-20260721` and recorded in
  `TGVF_VISUAL_TOOL_PROMPTS_V1.md`.
- [x] `FIXED PM-02` — Exact function descriptions, argument descriptions, and
  target wording for all three native tool schemas are versioned and hashed in
  `tgvf_rl.protocol.schema`; processor-rendered bytes/token goldens are pinned
  by `RO-P02B`.
- [ ] `OPEN_BLOCKING PM-03` — Policy Pilot v1 enables only
  `tgvf_focus_tool`; crop/TGVF fusion is deferred. It admits four TGVF call
  attempts, counting both successful observation and standard tool error, and
  gives the unexecuted fifth attempt a standard environment-owned cap error.
  Exact error bytes and recovery/termination semantics remain `PPV1-O03`.
- [ ] `OPEN_BLOCKING PM-04` — Prompt acceptance fixtures for parse rate,
  non-empty target, continuation, repeated calls, four-attempt accounting,
  fifth-attempt standard error, no crop call, no duplicate think opener, and no
  example copying.

### 9.4 Evaluation and promotion

- [ ] `OPEN_BLOCKING EV-01` — Evaluation framework is fixed to VLMEvalKit
  `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`; the shared data root and
  historical CoreDev-2511 ordered manifest are pinned. Seven source-specific
  hashed TSV slices are materialized as
  `coredev-2511-vlmevalkit-7055d301-v1`, and CPU validation proves that their
  runtime locator subclasses inherit the corresponding official dataset
  classes and scorers. Still required before a scored quality claim: numerical
  score-parity fixtures and the separately audited representation
  internal-evaluation group/counterfactual manifests.
- [x] `FIXED EV-04` — CoreDev LLM judging uses only
  `Qwen/Qwen2.5-72B-Instruct` through an independently identified local
  OpenAI-compatible service; GPT judge services are forbidden. OCRBench_v2 is
  rule based, the four MCQ sources use Qwen only as extraction fallback, and
  MathVista_MINI/MathVerse_MINI require Qwen judging. This does not close or
  activate reward contract `RW-05`.
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
- [x] Bind the selected pinned v4-train/v3-validation manifests, seven-path
  overlap report, provider-ordered production TOMLs and final artifact paths.
  The contextual configuration also passed a bounded real two-rank step-10
  preflight.
- [ ] Complete the contextual-first / embedding-second pair and
  historical-baseline evaluation before any trained-quality artifact claim.
- [x] Implement the S0 Qwen-family boundary, both condition-provider
  interfaces, SDPO teacher/objective/checkpoint boundary, and optional
  judge-provider interface. This does not close Qwen2.5 end-to-end or D0
  production-mathematics gates.
- [ ] Configure the local/runtime path for the fixed
  `Qwen/Qwen2.5-VL-7B-Instruct` model before its executable fixture.
- [ ] Bind and validate the required formal-Pilot
  `Qwen/Qwen2.5-72B-Instruct` judge service, prompt, sampling, calibration, and
  failure-policy identity under `RW-05`. This RL-judge identity remains
  separate from its VLMEvalKit benchmark-judge role.
- [ ] Convert each accepted `[TBD]` into a versioned project artifact rather
  than embedding it only in code or an experiment command.
