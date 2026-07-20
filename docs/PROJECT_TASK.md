# TGVF End-to-End RL: Project Task

Status: **I8H-20260719 bounded framework implementation complete; production gates open**
Recorded: **2026-07-18 JST**
Updated: **2026-07-20 JST**

Unresolved implementation contracts and their promotion gates are tracked in
[`OPEN_IMPLEMENTATION_CONTRACTS.md`](OPEN_IMPLEMENTATION_CONTRACTS.md). An open
field there must not be filled silently from a framework default.

## 0. Authoritative 8-hour implementation acceptance

Decision ID: **I8H-20260719**

Accepted by: **user**, on **2026-07-19 JST**

This decision authorizes one bounded eight-hour implementation goal. It is the
authoritative execution contract wherever older sections or linked planning
documents still describe the compatibility work as proposed, require a
SGLang/stable comparison, stop at an SDPO seam, or require another user reply
for each already bounded step.

### 0.1 Required result

The goal is a complete framework implementation with deterministic synthetic
fixtures and small integration smokes. It is **not** a production training run
and does not claim a trained policy or representation artifact. The required
stack is:

- upstream veRL as the distributed RL framework;
- vLLM as the only rollout backend; SGLang is out of scope;
- FSDP2 as a required executable path;
- Qwen3-VL-8B-Thinking as the primary executable model and
  `Qwen/Qwen2.5-VL-7B-Instruct` as the required family-adapter compatibility
  target. The bounded build supplies only its fail-closed main-`D`/family
  boundary; full DeepStack end-to-end compatibility remains a separate gate;
- both target-conditioning providers;
- exact multi-call `tgvf_focus_tool` trajectories, actual behavior log
  probabilities, immutable recorded observations, and deterministic replay;
- real reference-style pure SDPO code and tests based on
  `lasgroup/SDPO@7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, including teacher
  context, token alignment, exact-observation teacher replay, distillation
  loss/targets, teacher lifecycle, FSDP2 state, checkpoint, and resume. An
  interface-only or `SDPO_STATIC_SEAM_READY` result is incomplete.

Any GRPO or SDPO optimizer smoke must first record its exact **test-contract**
equations and pass a pure-tensor oracle. That test identity is not a silent
production default. Production GRPO/SDPO mathematics, coefficients, hybrid
composition, and scale configuration remain open.

### 0.2 Accepted implementation and interface surface

The following repository surface is accepted for creation or modification by
the implementation goal:

- `pyproject.toml`, one reproducible dependency lock, and isolated-environment
  setup needed for the accepted stack;
- `src/tgvf_rl/representation/`, `qwen/`, `conditioning/`, `protocol/`,
  `environment/`, `trajectories/`, `objectives/`, `framework/verl/`,
  `evaluation/`, and a minimal CLI surface;
- corresponding `tests/` suites, `configs/smoke/`, and bounded compatibility
  probes under `spikes/verl_compat/`;
- provenance, compatibility, experiment-ledger, and implementation reports
  needed to record exact identities and evidence.

The accepted public interfaces are:

1. a family-neutral TGVF Adapter call from typed target conditioning plus
   original-image pre-merge/main and model-supported DeepStack inputs to main
   `D`, D-DeepStack branches, and versioned metadata;
2. Qwen-family operations for native serialization, target-span identity,
   visual-state capture, recorded-observation forward, and replay;
3. one target-conditioning protocol implemented by contextual-hidden-state and
   target-token-embedding providers;
4. a strict native-tool parser and repeated-call environment whose observation
   result is an immutable, content-identified bundle of `D`, branches, layout,
   positions, masks, cache contract, and provenance;
5. a framework-neutral trajectory record carrying exact tokens, ownership
   masks, sampling identity, actual behavior log probabilities, rewards/stops,
   typed feedback, per-call observations, and SDPO alignment/state;
6. separately named GRPO and reference-style pure-SDPO objective entries;
7. a narrow upstream-veRL adapter using maintained/public extension points for
   rollout, forward/replay, objective execution, FSDP2, checkpoint, and resume.

Implementation helpers may be added within the accepted roots when they do not
broaden these interfaces. Legacy extraction remains limited to provenance-
registered files/symbols and parity requirements.

### 0.3 Execution authority and hard limits

- Isolated dependency installation and the download of necessary public model
  weights/artifacts are authorized. Compatibility evidence accepted upstream
  veRL commit `e003163181731412595257a72ec173071efb125f` and the resolved lock;
  this does not freeze production placement or parallel topology. SDPO's
  bundled veRL tree is reference-only and is not installed as the framework.
- GPU execution is restricted to physical GPU indices **2 and 3**. A process
  may see them as logical devices 0 and 1 only after the physical-to-logical
  mapping is recorded. No other physical GPU is authorized.
- Every GPU command still requires a complete `PLANNED` experiment-ledger row,
  exact command/environment identity, bounded timeout/resources, and applicable
  PASS/FAIL gates. I8H-20260719 supplies the user authorization for an in-scope
  row; it does not waive the ledger or correctness gates.
- Real training data, real reward coefficients, a final production prompt,
  production objective mathematics/configuration, long training, the 72B
  judge, and production checkpoint promotion are outside this goal. Their
  interfaces remain explicit and fail closed while unset.
- The implementation stops and reports evidence if upstream veRL/vLLM cannot
  meet actual-logprob, exact-observation, public-extension, deterministic
  replay, or FSDP2 requirements without a private trainer fork.

### 0.4 Accepted crop-tool extension

Decision ID: **CROP-FUSION-20260720**

Accepted by: **user**, on **2026-07-20 JST**

The policy RL phase has a second native visual tool named
`image_zoom_in_tool`, compatible at the public call boundary with the pinned
DeepEyes `image_zoom_in_tool`. Its exact arguments are
`{"bbox_2d": [left, top, right, bottom]}`. Coordinates are integer pixel
coordinates in the immutable original source image and use the PIL-compatible
half-open convention `[left, top, right, bottom)`. The executor clamps each
edge to source bounds and rejects an empty box after clamping. It does not add
DeepEyes-specific minimum-size, aspect-ratio, prompt, reward, or retry policy.

Every crop is taken from the immutable original image, never recursively from
the previous crop. The runtime records the requested and effective boxes,
source/crop dimensions and hashes, and the exact RGB crop pixels. The model's
rollout-time processed visual state for that crop is materialized alongside
the trajectory as behavior-forward evidence. The current fast replay path may
consume that processed state only in explicit
`shared_frozen_recorded_features` mode, which requires policy, reference, and
teacher to share an identity-proven frozen vision encoder. Otherwise the exact
crop pixels remain the observation and each consumer must re-encode those same
pixels with its own model state; that trainable-vision path remains fail-closed
until implemented. No replay may execute the crop again from an external path
or substitute different pixels.

`image_zoom_in_tool` and `tgvf_focus_tool` share one ordered multi-tool state
machine and one configurable total call cap greater than one. Any ordering is
valid, including crop then TGVF and TGVF then crop. A later policy turn can use
the accumulated native observations, while every TGVF Adapter invocation
continues to use the immutable original-image visual features as its visual
source. This is the initial crop/TGVF fusion contract: fusion occurs through
the shared policy trajectory and exact ordered replay, not by feeding crop
pixels into the TGVF Adapter or turning `D` into pixels.

Crop response content is environment-owned and receives no behavior log
probability or policy loss. Reward coefficients, crop-call cost, training-data
mixture, and any recursive-crop experiment remain open. This extension does
not modify the representation-phase transcript, loss, data, or checkpoint.

### 0.5 Accepted evaluation architecture

Decision ID: **EVAL-ARCH-20260720**

Accepted by: **user**, on **2026-07-20 JST**

This task separates representation diagnostics from policy benchmark scoring.
The implementation objective is an identity-safe post-training switch plus a
VLMEvalKit-owned benchmark path. It may modify the representation configuration,
runner closeout, `src/tgvf_rl/evaluation/`, evaluation configurations, and their
tests. It does not modify representation losses, training data, the TGVF
Adapter, policy objectives, or tool semantics. Verification is CPU contract and
CLI testing only until a separately planned GPU evaluation is recorded.

The representation **internal evaluation** is not periodic validation. Periodic
validation remains loss monitoring at `validation_every_optimizer_steps`.
Internal evaluation is an explicit configuration switch which, when enabled,
runs once only after the configured training target and final Adapter artifact
have been completed. A bounded `--stop-after-global-step` invocation must not
run it. A resumed invocation may run it only when it reaches the final target.
Its report is immutable and content identified; a pre-existing report currently
fails closed rather than causing a second evaluation. Automatic finalizer-only
recovery after a crash between report publication and the training `complete`
event remains an explicit follow-up contract. The switch must fail closed if
the exact evaluation population, checkpoint, prompt, or required native
counterfactual inputs are absent. It must not silently reinterpret the training
validation split as an accepted held-out evaluation population.

All external visual benchmark evaluation in this project uses upstream
**VLMEvalKit** at the exact review commit recorded in
`EXTERNAL_REFERENCES.md`. The project does not port the legacy benchmark parser
or scorer. The project-owned adapter executes the complete direct/crop/TGVF
agent loop, supplies only the final answer as VLMEvalKit's `prediction`, and
stores trajectory/tool/config identities in `extra_records`. Every arm freezes
its own decoding identity; VLMEvalKit's built-in Qwen3 sampling defaults are
not inherited implicitly.

Benchmark data is read from the shared cross-project physical root
`/nvmesv/dredvpn009/datasets/benchmarks` through an explicit `LMUData` binding.
The historical `/home/dredvpn009/Flash_Storage/datasets/benchmarks` spelling is
only an alias to the same directory. Project wrappers, prompts, predictions,
run manifests, and results remain in this repository.

The first recovered fixed subset is **CoreDev-2511**, logical manifest SHA256
`a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`,
seed `20260625`, with this exact allocation:

- VStarBench: 191;
- HRBench4K: 200;
- BLINK: 420;
- OCRBench_v2: 600;
- MMMU_Pro_10c: 300;
- MathVista_MINI: 300;
- MathVerse_MINI: 500.

VLMEvalKit has no generic subset filter and its generic `CustomVQADataset`
does not implement scoring. CoreDev-2511 must therefore be materialized as
seven ordered, hashed source slices. Each slice binds to its corresponding
official dataset class/scorer, reports its own metric, and only then enters a
separately identified 2,511-row aggregate. A mixed TSV scored as one generic
dataset is forbidden. The historical CoreDev-2511 scores remain comparison
provenance only because they were produced by the old custom benchmark path,
not VLMEvalKit.

CoreDev benchmark judging is fixed separately from policy-RL reward judging.
Any VLMEvalKit path that requires an LLM must use the locally served
`Qwen/Qwen2.5-72B-Instruct`; GPT services are forbidden. VStarBench,
HRBench4K, BLINK, and MMMU_Pro_10c use that model only as the official MCQ
fallback when deterministic option extraction is insufficient. OCRBench_v2 is
rule based and uses no judge. MathVista_MINI and MathVerse_MINI require the
Qwen2.5-72B judge. The model snapshot, service, and invocation are independently
identified from the evaluated policy. This benchmark decision does not enable
an LLM reward component, change the frozen RL reference, or select an SDPO
teacher.
The OpenAI-compatible served-model alias is `Qwen2.5-72B-Instruct` (without a
slash) because VLMEvalKit includes the judge string in intermediate filenames;
the repository and revision remain the model identity.

The seven source slices are materialized under identity
`coredev-2511-vlmevalkit-7055d301-v1`; their exact TSV and image-tree hashes are
recorded in `configs/evaluation/coredev_2511_vlmevalkit_v1.json`. The remaining
implementation inputs are the audited post-training internal-evaluation
population/counterfactual manifest, numerical score-parity fixtures, the policy
evaluation service identity, and the exact benchmark-arm decoding
configurations. No benchmark-quality claim is allowed until those inputs and a
real run artifact exist.

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

Decision `RPI-20260719` fixes the initial representation-training parity
boundary:

- Same-image Matrix CE is the required target-specificity objective. Its
  training unit is a semantic group containing multiple distinct targets for
  one image. Ordinary independent sample shuffling is invalid because it can
  remove every off-diagonal same-image comparison.
- The exact historical group construction, group shuffle, incomplete-group
  policy, distributed ownership, Matrix-CE score orientation, reduction, and
  gradients are provenance-pinned and parity-tested before use.
- `L_gen` and Matrix CE have separate intended roles. `L_gen` measures and
  trains whether frozen Qwen can read the evidence from `D`; Matrix CE applies
  relative target-specificity pressure within an image. The accepted baseline
  uses both terms at weights `1.0` and `1.0`, with separate metrics. Balanced
  Matrix CE uses temperature `1.0`. A controlled `L_gen=off` comparison is an
  ablation, not the baseline and not permission to remove the implementation.
- Decision `RPI-20260720-REPRESENTATION-NATIVE-TRAJECTORY` fixes the native
  representation transcript to the same simple role structure used by the
  policy RL phase. The user turn contains exactly the original image and the
  unmodified dataset question. It must not separately inject or append the
  teacher target, a representation tutorial, a focus-force instruction, or any
  other project-authored prompt text. Any lexical overlap between question and
  target may therefore come only from the original question. The neutral native
  tool schema is supplied to Qwen's chat template separately from the user
  message. This is the first user-accepted representation prompt/transcript
  contract and is identified as `qwen3-representation-image-question-v1`
  under prompt schema `native_representation_prompt_v1`. Earlier
  target-bearing smoke text was never accepted and is not a supported prompt
  version, compatibility mode, or production candidate. Its executable
  renderer branch and golden are removed; immutable experiment-ledger entries
  may still mention it only as historical fact.
- The teacher-constructed first assistant turn contains the fixed,
  target-independent pre-tool reasoning text `I need visual focus before
  answering.` followed by exactly one native `tgvf_focus_tool` call whose
  `target` argument is the dataset target. The target is therefore present in
  the assistant tool call and is never serialized as an additional user-prompt
  field.
- The tool turn carries the exact main `D` and all model-supported
  D-DeepStack branches. The final assistant turn places
  `evidence_description` in post-tool reasoning and the dataset
  `short_answer` in answer content. Representation data admitted by this
  contract must provide all of `question`, `target`, `evidence_description`,
  and `short_answer` as non-empty strings.
- The selected v4 source carries structured multiple-choice metadata. The v1
  transform preserves each ordered `(label, text)` choice in the immutable
  sample/data identity, but the accepted user message remains exactly the
  source `question`: choices are not appended, rendered, or trained as labels.
  Open-answer rows preserve an empty choice tuple.
- This is also the initial accepted data/supervision contract. Its identities
  are `representation_sample_identity_v1`, `retained_focus_rows_v1`, and
  `canonical_evidence_supervision_v1`. There is no earlier supported
  answer-omitting sample/data/canonical schema and no compatibility branch for
  one.
- Only tokenizer positions owned exactly by `evidence_description` receive
  representation teacher-forcing labels. The fixed pre-tool reasoning,
  tool-call target/JSON, latent tool response, final `short_answer`, prompt
  text, and chat-template wrappers remain label `-100`; adding the answer to
  the native transcript does not change `L_gen` or Matrix-CE mathematics.
  Token ownership is derived from the rendered native transcript and tokenizer
  offsets. A token is evidence-owned when its start offset lies inside the
  evidence span; this deliberately owns
  Qwen's common sentence-final token that also carries the following template
  newline, while excluding tokens that begin in the closing wrapper. A token
  that begins before and crosses into evidence is rejected as ambiguous.
  Decoded-text or fuzzy substring heuristics are not an accepted training mask.
  These positions first belong to the canonical chat transcript. The selected
  Qwen-family adapter must then map every canonical token to the processor's
  expanded model positions. Evidence tokens must remain singleton, contiguous,
  and ID-identical; all original-image and tool-observation visual positions
  remain label `-100`. Qwen2.5-VL requires its own accepted transcript and
  expansion fixture and cannot reuse the Qwen3 thinking contract.
- The manifold-loss contribution to optimization is exactly zero. The later
  accepted decision `RPI-20260719-NORM-EVAL` fixes the sole Norm-loss formula,
  reduction, and weight; no norm mode or target selector is introduced.
- Alternative geometric or distance-based contrastive losses are a separate
  research comparison, not a silent replacement for Matrix CE. They require
  explicit mathematics, negative construction, branch handling, and an
  ablation against both target specificity and causal readability.
- The historical representation internal tests and evaluation metrics are
  inventoried and must be reproduced as exact mathematical parity fixtures or
  explicit native-protocol semantic adaptations. The authoritative inventory is
  [`REPRESENTATION_PARITY_INVENTORY.md`](REPRESENTATION_PARITY_INVENTORY.md).

Decision `RPI-20260719-NORM-EVAL` extends that accepted boundary:

- This earlier decision left the production representation user prompt
  explicitly `[TBD]`. Decision
  `RPI-20260720-REPRESENTATION-NATIVE-TRAJECTORY` supersedes that clause: the
  representation user turn is now exactly the original image plus unmodified
  dataset question. Earlier target-bearing smoke text is deleted from the
  executable project rather than retained as a prompt version or compatibility
  fixture; immutable experiment-ledger entries may still record that it was
  used in earlier bounded runs. The policy-RL system/tool-use prompt remains a
  separate open contract.
- Norm loss is required and has one fixed historical formula rather than a
  configurable family of modes. For one main or branch tensor `D` and its
  corresponding frozen post-merger source tensor `V`, with `V` detached,
  `L_norm(D,V) = mean_t(log((||D_t||_2 + 1e-6) /
  clamp_min(mean_s ||V_s||_2, 1e-6))^2)`. One sample first averages the three
  D-DeepStack branch losses, then averages that branch mean with the main loss;
  the trainer takes the global sample mean across ranks and accumulation. The
  baseline exposes only the scalar weight `0.1`; there is no norm mode, target,
  or alternative formula selector. Raw and weighted norm values are logged.
- The initial old-configuration comparison uses Matrix CE weight `1.0`,
  `L_gen` weight `1.0`, AdamW learning rate `1e-4`, betas `(0.9,0.999)`, epsilon
  `1e-8`, weight decay `0.01`, gradient clipping at `1.0`, and the exact
  historical cosine schedule with 100 warmup steps and minimum learning-rate
  ratio `0.1` over 2,000 optimizer steps. These are also the accepted values for
  the initial paired provider runs. The selected data pair is v4 clean-imend
  Protocol-C focus train plus v3 val-2k accepted validation; the production
  TOMLs, materialized manifests, overlap report, and final artifact paths are
  bound under `configs/representation/`.
- Same-image group size `K=4` is retained. On physical GPUs 2 and 3 the bounded
  geometry proof uses four accumulation microsteps, giving 32 global rows and
  eight complete `4x4` matrices per optimizer update. A measured throughput
  result is required before launching the full 2,000-step run. RP-30 measured
  4.0565 seconds/step for target-token embedding, and the formal contextual
  step-10 preflight measured 4.7256 seconds/step. The latter implies about
  2.63 train-core hours for 2,000 steps before validation/checkpoint overhead.
- The seven already-audited exact resolved-image-path overlaps are accepted as
  recorded split metadata and do not block training. Configuration must select
  an explicit allow-recorded-overlap policy, preserve and log the exact overlap
  report, and must not describe the resulting validation split as image-path
  disjoint. No rows are silently removed.
- Both target-conditioning providers remain required code capabilities. The
  first run uses contextual hidden state from layer `-1`; the next paired run
  uses target token embedding. Both use identical data/group order, fresh
  Adapter initialization, seed `42`, batch plan, and all other objective and
  execution values. Provider identity remains artifact-bound.
- The accepted run pair uses K=4, per-rank batch 4, two-rank FSDP2, gradient
  accumulation 4, global batch 32, 2,000 optimizer steps, log cadence 10, and
  validation/checkpoint cadence 500. The seven recorded resolved-image-path
  overlaps are logged and accepted without filtering or blocking comparison.
- Internal evaluation is not complete merely because metric reducers have CPU
  fixtures. Before a full representation run, an executable real-Qwen/data
  runner must produce the correct-`D`, target-only, random-`D`, wrong-same-image,
  and wrong-different-image readout controls; full query matrices and retrieval
  reductions; main/branch distribution, norm, collapse, and attention health;
  and native counterfactual-value-flip and free-continuation outputs. The runner
  must retain per-sample records and deterministic checkpoint/data identities.
- A two-rank K=4 teardown/resume proof compares the uninterrupted next optimizer
  step with a separately reconstructed FSDP2 process. It binds Adapter,
  optimizer, scheduler, sampler, RNG, validation history, and distributed shard
  state, and records steady-state time and peak memory. It is a bounded smoke,
  not a promoted representation artifact.
- Execution record `RP-11` passed that bounded proof on physical GPUs 2 and 3:
  continuous and clean-teardown/resumed step 2 produced byte-identical
  104-tensor Adapter exports and exact recorded optimizer/scheduler/sampler/RNG,
  shard, train, and validation state. This closes executor/resume readiness,
  not the production data identity or semantic thresholds. The ledger may
  retain that run's then-used prompt as historical fact, but current code does
  not retain it as an executable prompt mode.
- The post-RP-11 throughput comparison keeps its mathematical global batch at
  32 while removing configured gradient accumulation. With world size two,
  each rank directly materializes four independent same-image K=4 groups in
  one optimizer update. The eight group-local `4x4` Matrix-CE blocks are
  reduced by their 32 valid rows; they must never be replaced by a cross-image
  `16x16` local matrix. This direct-batch geometry is a bounded throughput
  comparison, not permission to change the accepted K=4 objective or to make
  a production batch-size choice from smoke data alone.
- Execution record `RP-12` passed that comparison on physical GPUs 2 and 3.
  Direct GPR4/GA1 reduced steady optimizer-step time from `20.366` to `16.213`
  seconds and raised row throughput from `1.571` to `1.974` rows/s at the same
  32-row/eight-matrix global update; peak allocated memory was about 31.0 GB.
  This supports GA=1 for this B200 geometry while leaving the final production
  batch and data identities open. RP-12's then-used prompt remains ledger-only
  history rather than a supported current identity.
- Decision `RPI-20260719-B200-BATCHED-READOUT` fixes the next direct-batch
  executor correction. The two physical devices are B200 GPUs with about
  180 GB usable memory each, so the representation executor prioritizes
  throughput over the RP-12 low-memory score/recompute schedule. For the
  GPR4/K4 geometry, every Matrix-CE cell is evaluated differentiably once;
  complete four-candidate rows from independent groups may be combined two
  row slots at a time into compatible Qwen batches of at most 32. Each group's
  `4x4` score matrix and CE remain independent, repeated uses of a candidate
  across rows accumulate into that candidate, the diagonal cell is reused for `L_gen`, and the
  globally normalized candidate gradients traverse the TGVF Adapter once.
  Compatibility buckets may split the physical batch without changing the
  objective. This is the direct multi-group execution contract, not a family
  of user-facing memory-mode options. Acceptance requires objective and
  gradient parity, explicit physical-Qwen batch/call metrics, and a bounded
  throughput smoke on physical GPUs 2 and 3 before its timing is extrapolated.
- Execution record `RP-13` passed the CELLB32 geometry and parity gates on
  physical GPUs 2 and 3: both ranks executed exactly two B32 Qwen calls and 64
  cells per update, with one forward/VJP per cell and no readout recompute.
  Peak allocation was about 117.1 GB per rank. Steady step time was `15.937`
  seconds, only `1.70%` faster than RP-12, so B32 is not a material throughput
  solution by itself. At global batch 32, two ranks each own four local K4
  groups; the pinned historical global-batch-32 run used eight ranks with one
  local K4 group each. The later user decision selects the historical two-rank
  per-rank-B4/GA4/global-B32 geometry for the initial provider pair; it is not
  inferred from the RP-13 B32-cell measurement.
- Decision `RPI-20260720-GOLDEN-IMAGE-CAP-AB` accepts one bounded image-
  processing comparison against the pinned Golden representation path. The
  Golden lane sets the Qwen image processor's maximum pixel area to exactly
  `512 * 512 = 262144` while retaining the pinned processor's minimum-pixel
  bound and aspect-ratio-preserving smart resize; it must not force a square
  `512x512` raster. The cap is an explicit configuration/run identity and is
  applied consistently to every original-image and geometry-only processor
  materialization in the representation run. The comparison records source
  dimensions, `image_grid_thw`, pre-/post-merge visual-token counts, sustained
  GPU utilization, peak CUDA memory, and optimizer-step throughput on fixed
  inputs. The user now selects this aspect-ratio-preserving
  `image_max_pixels=262144` cap for the initial contextual/embedding run pair;
  it remains an explicit artifact identity and can be revisited only as a later
  comparison.
- Decision `RPI-20260720-CONTROL-STACK-OPTIMIZATION` accepts continued
  performance work on the already accepted Python 3.12 / Torch `2.9.0+cu128`
  / upstream veRL / FSDP2 / vLLM `0.12.0` control stack. Production dependency
  locks, representation objectives, K=4 same-image grouping, provider
  semantics, native transcripts, deterministic forward state, and exact RL
  observation/replay contracts remain unchanged. FlashAttention 2,
  FlashAttention 4, SGLang, and another framework/runtime upgrade are outside
  this task. The first bounded measurement compares Qwen3's native full-patch
  Conv3D with the algebraically equivalent flattened Linear expression using
  the exact checkpoint weight and bias under FP32 and BF16. It may become a
  repo-owned Qwen3 representation fast path only after numerical parity and a
  measured control-stack speedup; the checkpoint parameter/state identity and
  public family-adapter interface must remain unchanged. Subsequent
  optimizations require measured phase attribution and their own parity gate.
- Execution record `RP-30` removes the dominant host-side utilization gap
  without changing training mathematics. Qwen fast-tokenizer length validation
  now proves the exact base vocabulary plus contiguous added-token suffix
  instead of repeatedly materializing the merged 151k vocabulary; unfamiliar
  tokenizer layouts retain the native fallback. Ordered K=4 action/evidence
  transcript batching preserves scalar bytes, IDs, offsets, labels and hashes.
  Against retained RP-28, exact final 104-tensor Adapter parity passed while
  steady optimizer-step time fell from `9.4355` to `4.0565` seconds and mean
  GPU2/GPU3 utilization rose from `33.8%`/`36.0%` to about `84.5%`/`83.8%`.
  This retained path estimates `2.2536` train-core hours for 2,000 steps;
  periodic validation/checkpoint time remains additional.
- Formal preflight `REP-QWEN3-V4-CONTEXTUAL-V1-PREFLIGHT` binds the selected
  v4/v3 manifests, seven recorded image-path overlaps, initial v1 prompt,
  Balanced Matrix CE/L_gen/Norm objective, contextual layer `-1`, seed 42 and
  K4/B4/two-rank/GA4 geometry. Its bounded first 10 steps passed with finite
  nonzero Adapter gradients and a durable resumable checkpoint; step 10 took
  `4.7256` seconds. This is execution readiness, not the 2,000-step result or
  historical-baseline internal-evaluation promotion.
- Decision `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION` accepts a
  mathematics-preserving refactor of the Qwen3 native representation group
  builder to remove CPU/GPU bubbles measured in RP-17 and RP-18. For one
  same-image group, the source image may be decoded and processed once; later
  action transcripts and readout transcripts must derive their expanded native
  input IDs, masks, visual grids, and positions from that exact processor-owned
  source expansion. This reuse must preserve the canonical transcript, target
  span, source/D visual-token blocks, main D, every D-DeepStack branch, and all
  objective values and gradients. Redundant normal-path device-to-host
  validation synchronizations may be fused or deferred while retaining
  fail-closed diagnostics. Acceptance requires processor-call/image-instance
  count tests, exact derived-versus-processor transcript fixtures, existing
  representation contract tests, and a bounded two-GPU throughput/utilization
  comparison. A later `B16 x GA2` measurement may retain global batch 32 only
  after the same objective and gradient contracts pass; it is not accepted as
  a semantic batch change.
- Decision `RPI-20260720-CONFIG-BOUND-GPU-PAIR` accepts replacing the
  representation runner's historical physical-GPU `[2,3]` hardcode with an
  explicit two-device configuration identity. The pair must contain two
  distinct non-negative physical IDs, keep logical IDs `[0,1]`, and match
  `CUDA_VISIBLE_DEVICES` exactly and in order before distributed startup. This
  changes only launch placement; world size, FSDP2 mesh, model, objective,
  batch mathematics, determinism, and checkpoint identity remain unchanged.
  The initial authorized alternate pair is physical GPUs `[0,3]` for RP-17.
- Decision `RPI-20260720-BALANCED-MATRIX-CE` accepts a second selectable
  Matrix-CE cell score for the representation phase. Its public configuration
  name is `balanced`: for row `i` and complete candidate observation `j`
  (main D plus every D-DeepStack branch),
  `score_ij = -(evidence NLL sum / valid evidence-token count) / temperature`.
  Temperature is a fixed positive run-identity value and defaults to `1.0`.
  Row-wise diagonal-label cross entropy and the existing global valid-row
  reduction are unchanged, as is `L_gen`. The existing summed-log-likelihood
  score remains selectable as `legacy_summed_nll` and must retain exact value
  and gradient behavior. This task changes only the normal autograd and
  streaming/manual-gradient score paths and their CPU tests; it adds no margin,
  symmetric, embedding-contrastive, Norm, or other objective and requires no
  GPU smoke. The follow-up default decision makes `balanced` with temperature
  `1.0` the resolved default for new representation configuration schema v3
  objectives and direct new v3 objective construction. An experiment may still
  select `legacy_summed_nll` explicitly. Recorded earlier experiment identities
  retain their historical summed-NLL facts in the immutable ledger; this does
  not retain their unaccepted prompt/configuration path as an executable
  compatibility mode.

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

GRPO is the first online objective to validate. SDPO is also a required
architecture target from the initial skeleton, using
[`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) commit
`7c457fc1b1f636ae794eb0362ba37d4743b06fbc` as the fixed reference
implementation and arXiv `2601.20802v2` as the fixed paper revision.

This fixes the meaning and reference source, not our final objective contract.
The reference implementation's `loss_mode=sdpo` replaces the policy loss; it
does not define `GRPO + lambda * SDPO`. Pure GRPO, reference-style pure SDPO,
and any hybrid are therefore separate objective identities. Exact SDPO
equations, feedback/reprompt construction, teacher regularization, logit
approximation, importance weighting, and hybrid composition must be accepted
and parity-tested before use.

The accepted implementation must provide executable reference-style pure-SDPO
teacher context, token alignment, exact-observation teacher replay,
distillation targets/loss, teacher state, FSDP2 ownership, checkpoint, and
resume behavior. It may be implemented after the common trajectory/replay
substrate during I8H-20260719, but an interface-only placeholder is not a
completed result.

GRPO is also not identified by its name alone. Before any GRPO optimizer step,
the project must version the exact equations and conventions for group standard
deviation, advantage scaling, clipping, KL, per-token versus per-sequence
normalization, and masking, and verify them with a pure-tensor oracle. A
framework default is never the mathematical specification. Production choices
may remain open while framework code and non-production synthetic fixtures are
built.

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
The completed bounded compatibility task selected the exact upstream veRL
commit and vLLM-only backend, exercised the public integration surface,
transported a real Qwen3 two-call precomputed-latent fixture, and proved a tiny
two-rank FSDP2 checkpoint/resume path. Framework tests cover both
target-conditioning providers, exact behavior-logprob retention/replay, and
executable reference-style SDPO contracts. It was not a framework-selection
comparison. Full Qwen policy/reference replay, production topology, and the
Qwen2.5 family-specific representation/branch path remain later gates because
TGVF injects hidden visual embeddings, M-RoPE state, multimodal token types, and
D-DeepStack features rather than a PIL tool image.

The SDPO reference repository contains a modified veRL tree rather than a thin
plugin and does not identify a reproducible upstream veRL base commit. It is
therefore an algorithm/patch-surface reference, not the production framework
pin. Its current distillation path also rejects multimodal inputs, reconstructs
teacher prompts from a simplified final-user view, and does not provide the
teacher checkpoint/resume contract required here. The compatibility spike must
measure these gaps explicitly rather than inherit the fork.

### 2.5 Required Qwen families and target-conditioning providers

The primary policy/reference target is **Qwen3-VL-8B-Thinking**, using the
stable local path
`/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`. Per the user decision, the
project does not scan or hash every weight shard in that directory. The model
name and path are the accepted operational identity. Processor/tokenizer/chat
template behavior is still frozen through exact native transcript fixtures and
fixture hashes because it directly affects policy tokens.

The required secondary compatibility model is
**`Qwen/Qwen2.5-VL-7B-Instruct`**. Its local/runtime path remains `[TBD]`.
Model-family-specific processor, vision-tap, DeepStack, M-RoPE, native
transcript, and forward behavior live behind a versioned Qwen VLM adapter. The
Qwen3 representation checkpoint is not presumed portable to Qwen2.5-VL.

Both target-conditioning providers must be implemented and tested:

1. contextual hidden state from the exact sampled target span and trajectory;
2. target token embedding from the selected base model without new tokens.

An experiment selects one provider, but provider support itself is not an open
either/or architecture decision.

Decision `TCPI-20260719` makes that selection an explicit public interface:

- every run configuration contains one required target-conditioning provider
  configuration; there is no implicit default;
- provider identity is a closed enum with exactly `contextual_hidden_state` and
  `target_token_embedding` in schema v1;
- both implementations accept the same typed target-conditioning request and
  return the same `TargetConditioningOutput`/provenance schema;
- a repository-owned factory constructs the selected provider from the run
  configuration and injected model dependencies; callers do not branch on
  provider class names;
- contextual hidden-layer identity and base embedding identity are mutually
  exclusive, explicitly configured fields; invalid or incomplete combinations
  fail closed; and
- both selections must pass the same request, batching, TGVF Adapter handoff,
  provenance, and configuration-identity tests. Provider-specific Qwen-family
  numerical fixtures remain required before a real experiment is promoted.

Provider types are never mixed within one representation-training run or
batch. Contextual-hidden-state and target-token-embedding comparisons use
separate paired runs with the same retained sample IDs, same-image group order,
TGVF Adapter initialization, batch/accumulation plan, and random seed. Provider
identity is part of the experiment and checkpoint-artifact identity.

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

The tools are provided through `apply_chat_template(..., tools=[...])`. The
TGVF function contract remains equivalent to:

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

The policy RL tool set additionally contains:

```json
{
  "type": "function",
  "function": {
    "name": "image_zoom_in_tool",
    "description": "Crop one rectangular region from the original image.",
    "parameters": {
      "type": "object",
      "properties": {
        "bbox_2d": {
          "type": "array",
          "items": {"type": "integer"},
          "minItems": 4,
          "maxItems": 4
        }
      },
      "required": ["bbox_2d"],
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

Repeated and mixed tool calls are a first-class runtime requirement. A
trajectory may alternate among policy reasoning, `tgvf_focus_tool`, a new `D`
observation, `image_zoom_in_tool`, and an exact crop observation more than once
before the final answer. A configurable shared safety cap greater than one is
required; its exact value and stopping policy remain open.

The parser is fail-closed:

- exactly one complete tool-call object per assistant action turn;
- strict `json.loads`;
- an exact registered function name and exact per-function argument keys;
- a non-empty, non-generic TGVF target, or exactly four integer crop
  coordinates defining a non-empty box after source-bound clamping;
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
5. If it emits a valid `image_zoom_in_tool` call:
   - preserve the exact sampled call tokens and behavior log probabilities;
   - clamp the requested integer box to the immutable original-image bounds
     and reject an empty result;
   - materialize a contiguous RGB8 crop plus its content/provenance record;
   - append the native image-bearing tool response and retain its exact
     rollout-time processed visual state for replay.
6. Continue the same trajectory after either observation. The policy may
   answer or emit either registered tool call; repeat the applicable execution
   step until a final answer, a malformed action, or the shared safety limit
   terminates the loop.
7. Record the complete action mask, tool/environment spans, rewards, stop
   causes, and all identity fields needed for mathematically identical replay.
8. Replay policy/reference log probabilities only on policy-generated tokens
   from every assistant turn; template prefill and tool-observation tokens are
   environment output and receive no policy loss.
9. For every tool call, policy, old-policy, and frozen-reference replay all
   consume that call's same rollout-recorded observation. They may recompute
   logits on the recorded trajectory, but they never regenerate `Hq`, main `D`,
   or D-DeepStack from their own or updated parameters, and never recrop or
   substitute crop pixels.

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
build_teacher_context
replay_teacher_on_recorded_observations
compute_named_objective
```

All entries are present in the framework-neutral architecture from the
beginning. `build_teacher_context` and
`replay_teacher_on_recorded_observations` fail closed while SDPO is disabled.
`compute_named_objective` is a general registry: the approved GRPO path remains
independently usable, while `sdpo` and any hybrid mode fail closed until their
own contracts pass. Teacher replay consumes the recorded multimodal trajectory
and exact per-call observations; it must not reconstruct a text-only final
response or regenerate `D`.

## 5. Initial parameter policy

The initial policy is the unmodified original Qwen reasoning model plus a
newly defined policy adapter scope. The following are fixed or unresolved:

- Frozen reference: exact original Qwen model and prompt/tool schema.
- Policy initialization: exact original Qwen model; no Stage2 LoRA.
- Primary family: Qwen3-VL-8B-Thinking. Secondary compatibility family:
  `Qwen/Qwen2.5-VL-7B-Instruct`.
- TGVF Adapter: a newly trained native-format TGVF Adapter checkpoint, initially
  frozen during the first RL proof. The legacy checkpoint is not a direct
  initialization.
- Target conditioning: both contextual-hidden-state and target-token-embedding
  providers are required; the active provider is an experiment identity.
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

The optional external judge is `Qwen/Qwen2.5-72B-Instruct`, following the
pinned DeepEyes reference. Its provider interface is reserved, but it is
disabled for the first pilot. Before any later activation, its exact
local/service identity, prompt, sampling, calibration set, and scope must be
accepted. The answer judge, frozen RL reference, and SDPO self-teacher are three
separate roles and states.

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

The control-matching requirement above prevents scale or layout from becoming a
confounder; it does not accept a norm-training loss. Historical internal
representation tests and metrics must be reproduced according to the parity
inventory, with every exclusion or native-protocol adaptation stated rather
than silently dropped.

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
not a framework bake-off and it is not a training run. The completed bounded
task used upstream veRL and answered the following questions to the extent
recorded by its framework tests and smoke cells:

- Does the selected veRL commit support Qwen VLM policy/reference models?
- Does the family adapter run the primary Qwen3-VL-8B-Thinking fixture and a
  `Qwen/Qwen2.5-VL-7B-Instruct` compatibility fixture without leaking
  family-specific tensor assumptions across the boundary?
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
- Do maintained public dataflows expose group standard deviation, advantage,
  behavior/proximal ratio, clipping, reference/KL, masks, normalization, global
  denominators, and accumulation without silently overwriting project-owned
  values, so later accepted GRPO equations can be implemented and parity-tested?
- Is the intended SDPO implementation supported, or can its pure tensor loss
  and teacher lifecycle be added without replacing private trainer internals?
- Can the same framework-neutral trajectory support both target-conditioning
  providers and both GRPO and SDPO replay without changing token ownership or
  observation identity?
- Can SDPO teacher context preserve the complete native multi-call transcript,
  align every policy-sampled assistant turn, and consume the exact recorded
  original image, main `D`, D-DeepStack, layout, positions, masks, and cache
  contract?
- Can teacher/EMA or trust-region state be sharded with FSDP2 and strictly
  checkpointed/resumed alongside LoRA or full actor state? The reference SDPO
  repository does not prove this path.
- Are checkpoint/resume and dependency versions reproducible locally?

The accepted compatibility stack fixes upstream veRL commit
`e003163181731412595257a72ec173071efb125f`, its resolved Python 3.12 dependency
environment, vLLM `0.12.0` as the only backend, and the repository-owned public
adapter/plugin surface. Its live path requires
`VLLM_PLUGINS=tgvf_qwen3_precomputed`,
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`, and multimodal-encoder
`TORCH_SDPA`. This does not select a production actor/reference/teacher
placement, sharding plan, or parallel topology.

### 9.1 Accepted I8H compatibility and implementation task

The earlier proposed fixtures, numerical tolerances, and hard-failure rules are
retained as useful evidence requirements in
[`VERL_COMPATIBILITY_SPIKE_PLAN.md`](VERL_COMPATIBILITY_SPIKE_PLAN.md).
I8H-20260719 supersedes that document's proposal status, SGLang cells, stable-
comparison execution, seam-only SDPO outcome, and repeated human-approval
sequence.

The accepted upstream veRL compatibility revision is the exact `main` snapshot
`e003163181731412595257a72ec173071efb125f`, observed on 2026-07-19 JST. Official
`v0.8.0@7aed6b230776f963fa09509c10d9c3a767d1102c` remains historical comparison
material only and is not a required runtime cell. Only vLLM is implemented or
tested. The resolved lock selects vLLM `0.12.0`, Torch `2.9.0+cu128`, and
Transformers `4.57.6` under CPython `3.12.3`.

The bounded result is `111` passing CPU tests, `SC-20-R6` passing real Qwen3
TP=2 native two-call precomputed-latent transport, and `SC-30` passing a tiny
two-rank composable-FSDP2 strict checkpoint/resume with a bitwise-identical next
step. These results do not claim a trained Adapter, Qwen FSDP2 capacity,
production policy/reference replay parity, Qwen2.5 end-to-end support, or
production objective evidence.

Isolated dependency materialization and necessary public weights are
authorized. GPU work is authorized only on physical devices 2 and 3 after the
required ledger row is complete. Synthetic reference-style SDPO implementation
and its bounded parity smoke are part of the accepted goal; production
training, production objective selection, and real-data experiments are not.

### 9.2 Accepted Torch 2.11 compatibility re-spike

On 2026-07-20 JST the user accepted an isolated re-spike of the runtime stack
to evaluate the representation-throughput benefit of PyTorch 2.11. The
accepted control remains the complete I8H environment above; it must not be
modified in place. The first binary-compatible candidate is CPython 3.12,
PyTorch/TorchVision/TorchAudio `2.11.0/0.26.0/2.11.0` from the official CUDA
12.9 index, the official vLLM `0.23.0+cu129` release wheel, Transformers
`4.57.6`, and upstream veRL commit
`638b8ff84f279e054982f1f4633a546f3c6ced68`. CUDA 12.9 is selected for this
first re-spike because vLLM publishes a matching official binary while it does
not publish a CUDA 12.8 wheel; source-building vLLM is a separate fallback, not
an interchangeable result.

This re-spike may materialize a separate environment and adapt repository-owned
public integration code, but it may not overwrite
`requirements/compatibility.lock`, `.venv312`, or the accepted I8H artifacts.
The candidate is rejected unless all of the following pass in the same resolved
environment:

- dependency resolution, `pip check`, exact package/source identity, and the
  veRL public AgentLoop/DataProto/loss/FSDP/checkpoint surface;
- the repository's CPU contracts, with a newly audited vLLM 0.23 sampling and
  processed-logprob contract rather than a changed version constant;
- two-rank composable FSDP2 update plus strict checkpoint/reconstruct/resume;
- a bounded upstream-veRL FSDP2 actor-to-vLLM weight-synchronization and
  generation path; separate FSDP2 and vLLM processes are not sufficient
  evidence for this gate. This exact wheel candidate is restricted to
  `free_cache_engine=false`, `enable_sleep_mode=false`, and the colocated naive
  checkpoint engine; passing it does not claim the upstream patched sleep/wake
  path;
- the real local Qwen3-VL-8B-Thinking vLLM TP=2 native repeated-tool-call and
  precomputed main-`D`/D-DeepStack smoke;
- real Qwen3 representation FSDP2 forward/backward and the Qwen3 patch-embed
  performance probe; and
- no regression of tokenizer size, exact observation identity, deterministic
  replay state, both target-conditioning provider interfaces, or SDPO state
  transport.

GPU cells remain limited to physical devices 2 and 3 and require complete
`PLANNED` ledger identities before launch. Passing this task promotes a new
compatibility candidate only; changing the production/default lock is a
separate recorded decision. Failure leaves the I8H environment authoritative.

The candidate failed the mandatory combined veRL FSDP2-to-vLLM gate on
2026-07-20. FSDP2 checkpoint/resume passed and the real Qwen actor initialized,
but vLLM `0.23.0+cu129` selected its bundled decoder FlashAttention 2 kernel;
that kernel's PTX was rejected by the host's NVIDIA `570.195.03` driver
(`cudaErrorUnsupportedPtxVersion`). Selecting `TORCH_SDPA` for the multimodal
encoder did not change the decoder path, and this vLLM release treated the old
`VLLM_ATTENTION_BACKEND` environment key as unknown. Therefore this exact
candidate is rejected, the remaining candidate GPU cells are not run, and the
I8H Torch 2.9 environment remains authoritative. FlashAttention 2 and
FlashAttention 4 are explicitly deferred: neither may enter the production
dependency set until a separately approved compatibility spike proves the
chosen driver, CUDA build, vLLM, veRL, FSDP2, Qwen3, and replay path together.

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

### External reference boundary

The exact external commits and permitted topics are recorded in
`docs/EXTERNAL_REFERENCES.md`.

- `lasgroup/SDPO` is an algorithm and patch-surface reference. Its bundled veRL
  tree is not vendored or selected as the production framework.
- `Visual-Agent/DeepEyes` may inform multi-turn agent-loop, dynamic multimodal
  observation, tool registry, Qwen-VL M-RoPE/mask, data, reward, judge-service,
  tracing, evaluation interfaces, and the public `image_zoom_in_tool` /
  `bbox_2d` crop-call boundary pinned in `EXTERNAL_REFERENCES.md`.
- DeepEyes implementation code, rendered prompts, reward coefficients,
  asynchronous behavior, and logprob/replay conventions are not inherited.
- The legacy `revisit_vlm_clean` code is an eligible exact-file, read-only design
  reference under `docs/LEGACY_REFERENCE.md`; “clean” is not blanket reuse
  permission. Registered files may inform new interfaces, while wholesale
  trainer/replay/objective/runtime migration remains forbidden.

## 11. Planned repository shape

The interface roots in §0.2 are accepted for I8H-20260719. The intended shape
is:

```text
tgvf-e2e-rl/
  AGENTS.md
  README.md
  pyproject.toml                 # exact compatibility extras and plugin entry point
  docs/
    PROJECT_TASK.md
    LEGACY_REFERENCE.md
    EXTERNAL_REFERENCES.md
    EXPERIMENT_LEDGER.md
    OPEN_IMPLEMENTATION_CONTRACTS.md
    VERL_COMPATIBILITY_SPIKE_PLAN.md
    TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md
  src/tgvf_rl/
    representation/             # extracted TGVF Adapter and training path
    qwen/                       # Qwen3/Qwen2.5 family adapters and native VLM state
    conditioning/               # both required target-condition providers
    protocol/                   # schema, strict parser, transcript identity
    environment/                # TGVF tool execution
    trajectories/               # framework-neutral rollout records
    rewards/                    # decomposed executable rewards
    objectives/                 # framework-neutral GRPO/SDPO objective boundary
    judges/                     # optional versioned external answer judges
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

### Phase 0: freeze provenance and approve the bounded spike

- Archive the current old-repository dirty worktree under a user-approved
  commit/tag or immutable patch bundle.
- Record the accepted stable Qwen3 local path and native transcript identity;
  do not perform full weight-shard hashing. Pin the historical representation
  checkpoint only as a parity/reference identity.
- Finalize the exact extraction whitelist and file hashes.
- Retain the point-in-time SDPO and DeepEyes review commits in
  `docs/EXTERNAL_REFERENCES.md`; do not turn either repository into an implicit
  dependency.
- Accept the spike questions, candidate commits/backends, synthetic latent
  fixtures, PASS/FAIL rules, and any applicable `PLANNED` GPU ledger entry.

Gate: every imported idea or symbol has immutable provenance, and the isolated
compatibility spike is authorized without authorizing the main implementation.

### Phase 1: veRL/vLLM/FSDP2 compatibility and framework implementation

Status: **bounded framework portion complete**. The real Qwen3 transport and
tiny FSDP2 resume cells passed; full Qwen replay, Qwen2.5 end-to-end support,
and production objective/topology gates remain open.

- In the accepted isolated environment, test and then pin one upstream veRL
  commit with vLLM; do not adopt the SDPO or DeepEyes veRL trees.
- Validate Qwen3-VL-8B-Thinking policy/reference forward and the
  `Qwen/Qwen2.5-VL-7B-Instruct` family-adapter fixture.
- Validate a deterministic native `tgvf_focus_tool` synthetic-latent fixture
  with at least two calls, exact behavior logprobs, immutable observation
  replay, masks, positions, cache state, and checkpoint/resume.
- Validate FSDP2 execution/state handling and both target-conditioning provider
  seams.
- Map public ownership for every future GRPO dataflow—group formation/std,
  advantage, behavior/proximal ratio, clipping, reference/KL, masks,
  normalization, global denominators, and accumulation—and use sentinel fields
  plus a deterministic non-RL objective to prove that veRL does not silently
  overwrite or normalize project-owned values. Exact GRPO equations and
  loss/gradient/accumulation parity remain Gate G0 work and must close before
  any GRPO optimizer fixture, not before the infrastructure-only FSDP2 probe.
- Map the pinned SDPO patch surface onto maintained/public upstream veRL hooks,
  then implement reference-style pure SDPO with complete teacher context,
  per-turn alignment, exact recorded observations, distillation targets/loss,
  teacher lifecycle, FSDP2 ownership, and checkpoint/resume. A seam-only result
  does not pass this phase.
- Implement the accepted Qwen-family, dual-provider, trajectory,
  objective/teacher, judge, and veRL ownership interfaces in §0.2.

Gate: the framework-neutral code may be developed under §0 while evidence is
collected, but the veRL binding passes only if the selected stack works without
a private-trainer fork, PIL re-encoding of `D`, text-only teacher replay, or
loss of actual sampling logprobs, and real SDPO synthetic parity passes.

### Phase 2: representation-core extraction

- Extract only the selected TGVF classes/helpers.
- Export a minimal TGVF Adapter artifact without optimizer, scheduler, or
  protocol-token rows.
- Establish deterministic synthetic and real-row parity with the pinned old
  TGVF Adapter.

Gate: for fixed tensor inputs (`Hq`, pre-merge vision features, every DeepStack
branch, mask/config), main `D`, every D-DeepStack output, and required gradients
match the pinned Adapter. This gate tests Adapter mathematics, not legacy
Protocol-C serialization.

### Phase 3: native protocol and runtime

- Implement native tool schema, strict parser, target-span mapping, and native
  tool-response `D` injection.
- Establish a Qwen VLM family-adapter contract: Qwen3-VL-8B-Thinking is the
  primary executable path and `Qwen/Qwen2.5-VL-7B-Instruct` is the required
  compatibility fixture.
- Support repeated `tgvf_focus_tool` calls with a configurable safety cap
  greater than one and per-call immutable observation records.
- Support mixed repeated `image_zoom_in_tool` and `tgvf_focus_tool` calls under
  the same safety cap, with content-identified crop pixels and exact ordered
  replay of crop visual state and TGVF latent state.
- Prove no tokenizer resize, exact template-generated transcript/token round
  trip, correct ownership masks, and no duplicate opening `<think>` in either
  assistant turn.
- Measure untrained base prompt-trigger behavior without using it as SFT data.

Gate: native direct/tool trajectories execute deterministically and expose all
required policy/reference/teacher replay state through the already accepted
framework-neutral interfaces.

### Phase 4: native representation training and family compatibility

- Decision `RPI-20260720-WANDB-TELEMETRY` accepts a CPU-only W&B sidecar for
  user-authorized formal representation runs. It may read the immutable TOML
  identity and tail the runner-owned metrics JSONL, backfill existing metric
  events, and upload a small flattened parameter/metric set to a separately
  named project. It must not mutate training state, infer missing metrics,
  upload sample IDs, alter checkpoint/resume identity, or become a training
  dependency. The local JSONL remains the authoritative scientific record.
- Compare old Protocol-C and native-tool `Hq`/`D` behavior.
- Build the new native-format representation data/training/checkpoint pipeline.
- Reproduce the pinned same-image multi-target sampler and exact Matrix-CE
  tensor/gradient behavior before any general-purpose data-loader shuffle or
  distributed sampler is accepted.
- Implement `L_gen` as a separately configurable and logged readability term,
  including a controlled on/off ablation; do not infer its scientific
  necessity from historical use alone.
- Keep manifold optimization weight at zero. Implement the single accepted
  historical norm formula and scalar baseline weight from
  `RPI-20260719-NORM-EVAL`; do not add norm modes.
- Implement and test both contextual-hidden-state and target-token-embedding
  condition providers behind the same TGVF Adapter boundary. A selected
  provider's run does not require a paired real-GPU comparison.
- Reproduce the pinned historical internal representation tests and metrics in
  an executable real-Qwen/data runner, distinguishing exact tensor/reduction
  parity from native-protocol semantic adaptations.
- Train a new Qwen3-VL-8B-Thinking native-format TGVF Adapter checkpoint; do not
  initialize it directly from the historical TGVF Adapter checkpoint.
- Run target-sensitivity, readout, counterfactual flip, and free-continuation
  gates.
- Before claiming Qwen2.5-VL end-to-end support, configure the exact local or
  runtime identity for `Qwen/Qwen2.5-VL-7B-Instruct`; freeze
  its native transcript, processor, vision features, M-RoPE, mask/cache, and
  main-`D`/DeepStack-equivalent injection contract; train a separate
  family-specific representation artifact; and pass both-provider,
  multi-call, exact-replay, and objective fixtures. Missing equivalent
  DeepStack support is a recorded compatibility blocker, not permission to use
  silent zero/dummy branches or a Qwen3 artifact.

Gate: the primary family has target-specific, readable native `D`; a secondary
family is called supported only after its own artifact and full fixture suite
pass. No claim is based only on interface presence, formatting, or output
length.

### Phase 5: minimal GRPO proof

- Use a fixed, audited prompt/sample manifest.
- Train only the approved policy scope with the TGVF Adapter frozen.
- Evaluate direct/tool exploration, reward decomposition, D use, and reasoning
  retention at a checkpoint ladder rather than only at the endpoint.

Gate: final reward improves without losing high-budget original reasoning or
learning to ignore/misuse `D`.

### Phase 6: production-scale objective validation

- Scale rollout/replay only after Phase 5 correctness.
- Promote the already implemented reference-style pure-SDPO path only after its
  production mathematics/configuration is separately accepted against the
  pinned paper/implementation. Any GRPO+SDPO hybrid remains a distinct later
  research decision.
- Verify complete multi-call multimodal teacher context, exact-`D` replay,
  FSDP2 teacher lifecycle, LoRA/full parameter mapping, and strict resume.
- Consider constrained joint TGVF updates only as a named later experiment.

Gate: SDPO support is real rather than a placeholder, and its mathematical
identity, throughput, memory, and resume behavior are recorded and reproducible.

## 13. Open decisions requiring confirmation

1. Compact code/config labels for the representation phase and policy RL phase;
   the formal prose names remain in force until replacements are accepted.
2. Exact legacy representation dataset/code reuse whitelist and the new
   native-format training interface.
3. Whether the TGVF Adapter is frozen for all policy RL or only for
   the first proof; the first proof is fixed to frozen.
4. Policy LoRA/full-parameter scope and reference/KL contract.
5. Production actor/reference/rollout/teacher placement and concrete
   FSDP2/parallel topology. The compatibility veRL commit, dependency
   environment, and vLLM-only backend are already fixed.
6. Exact SDPO equations, feedback/reprompt contract, teacher regularization,
   approximation, and whether any separately named hybrid is scientifically
   required. The paper/repository identity is already fixed.
7. Tool-call safety cap greater than one, prompt wording, and initial
   exploration curriculum.
8. Original-image visibility and DeepStack mask scope after each `D`.
9. Training population, reward benchmarks, and held-out evaluation manifests.
10. Later reward activation conditions, prompt, sampling, calibration, and
    reward scope for `Qwen/Qwen2.5-72B-Instruct`. Its use as the fixed
    VLMEvalKit benchmark judge does not activate it as an RL reward.
11. Local/runtime path and family-specific representation artifact plan for the
    fixed `Qwen/Qwen2.5-VL-7B-Instruct` compatibility model.

No item above may be silently promoted into a production experiment. Under
I8H-20260719, implementation may expose it as an explicit unset configuration
or use a clearly named synthetic test identity with an oracle; real data,
reward, final prompt, production mathematics, and production configuration
remain open after the eight-hour goal.

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
