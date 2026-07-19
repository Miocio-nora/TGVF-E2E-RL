# Representation Phase Parity Inventory

Status: **bounded executable scaffold and parity fixtures implemented;
production/scientific promotion gates open**

Recorded: **2026-07-19 JST**

## Purpose

This file inventories the historical representation-training tests, internal
evaluations, metrics, and runtime invariants that must be reproduced in the new
native-format pipeline. It does not promote a historical protocol, checkpoint,
loss weight, or metric value into a new experiment default.

Every source below is pinned in [`LEGACY_REFERENCE.md`](LEGACY_REFERENCE.md).
Historical component names appear only as provenance.

## Accepted initial boundary

- Matrix CE and same-image multi-target grouping are required.
- Ordinary independent shuffle is invalid for Matrix-CE training.
- The accepted baseline retains both Matrix CE and `L_gen`, separately logged
  with explicit nonzero weights. `L_gen=off` remains a separately identified
  ablation; the exact baseline coefficient is still open.
- Manifold-loss optimizer weight is exactly zero.
- Norm-loss design is unresolved. No new mode, formula, target, or default is
  authorized by this inventory.
- Contextual-hidden-state and target-token-embedding providers run separately;
  paired comparisons share sample/group order, Adapter initialization, batch
  plan, and seed.
- A geometric or distance-based contrastive objective is a later named
  experiment, not part of baseline parity.

## Implemented bounded evidence

The current implementation has an executable Qwen3 representation path, not
only interface placeholders:

- `data.py` applies the strict retained-focus transform, validates the source
  hash and image paths, and emits immutable accepted/excluded/duplicate/leakage
  manifests plus four-key train/validation overlap reports;
- `native_pipeline.py` and `runtime.py` build the native Qwen tool trajectory,
  extract the strict raw target span, bind either contextual-hidden-state or
  target-token-embedding conditioning, capture main/three-branch Qwen3 visual
  features, and construct source-plus-`D` readout rows;
- `streaming.py` evaluates a same-image `K×K` matrix without retaining the full
  set of cell graphs, blocks original-image keys for post-`D` evidence queries,
  and backpropagates the exact Matrix-CE and `L_gen` gradients through main `D`
  and all D-DeepStack branches;
- `trainer.py` owns accumulation, global numerator/count normalization, frozen-
  Qwen checks, Adapter-only AdamW ownership, clipping, scheduling, metrics, and
  loss-excluded zero-gradient collective padding when different ranks receive
  four- versus five-candidate groups;
- `fsdp2.py`, `checkpoint.py`, `distributed_checkpoint.py`, and `config.py`
  implement composable-FSDP2 ownership, Adapter-only artifacts, strict training
  state/resume identities, content-bound per-rank distributed checkpoint
  schemas, and a no-default TOML contract for both providers.

The bounded evidence includes the separate real-local-processor golden in
`tests/representation/training/test_qwen3_representation_golden.py`, synthetic
native group-builder fixtures for both providers, streaming/trainer fixtures,
CPU bitwise next-step checkpoint parity, FSDP2 ownership/API tests, and the
independent FP32/BF16 functional output/gradient oracle in
`tests/representation/test_adapter_reference_parity.py`. These do not constitute
a trained Adapter, an accepted data split, a real 8B backward pass, or a
two-rank representation FSDP2 run.

## Exact mathematical parity items

### Matrix CE

Pinned sources:

- `src/revisit_vlm/tgvf_training.py` at SHA256
  `ebae9266cafc4f83685f6a7d46a5b0fc501111f3ba9ee257c9e530e636857dd4`;
- `src/revisit_vlm/tgvf_v3_stage1.py` at SHA256
  `78b465ec67d40c6d60863715c43171f373483b20c11447b13b56b6fe8e28384a`.

For a same-image group of size `K`, row `i` fixes the sample's target context,
original-image capture/layout, and `evidence_description`. Column `j` injects
the main `D` and all D-DeepStack branches produced for target `j`. The score is
the negative summed evidence NLL:

```text
score[i, j] = log p(evidence_i | target_i context, D_j, D-DeepStack_j)
            = -sum_t NLL(evidence_i,t)
```

The correct label for row `i` is column `i`. There is no historical
temperature. Cross entropy uses a sum reduction per matrix, followed by
division by the total number of valid rows across all matrices. The explicit
streaming score-gradient helper must match autograd.

At distributed/accumulated scale this is one global numerator and valid-row
denominator. Equal averaging of local row means is incorrect when ranks receive
different group sizes, such as the historical batch-size-5 path's mixture of
four- and five-row groups.

Required new fixtures:

- [x] diagonal-preferred value oracle;
- [x] multiple-group row-weighted reduction oracle;
- [x] score-gradient parity with autograd;
- [x] FP16/BF16 parity for the historical FP32-softmax-then-cast score-gradient;
- [x] unnormalized numerator/row-count terms and a four-row-versus-five-row global
  reduction oracle;
- [x] four-row-versus-five-row composable-FSDP forward/backward count alignment
  through loss-excluded, exact-zero-gradient Adapter padding;
- [x] synthetic live-forward sensitivity for main `D` and each ordered
  D-DeepStack branch under atomic whole-observation column swaps;
- [x] zero valid groups returns a scalar zero without creating a false training
  signal.

### `L_gen`

Historical `L_gen` is evidence-token mean NLL for each sample, followed by a
mean over samples. Matrix scores use negative summed NLL instead, so the two
reductions must not be conflated.

Across ranks and accumulation windows, the trainer therefore reduces the sum
of per-sample token means by the global sample count. It does not replace this
with a global token mean and does not equally average unequal local batch means.

`AD-03A` fixes the native label span. The phrase "assistant supervision" is not
a new module: it only describes which next-token positions receive
teacher-forcing labels. `evidence_description` is the reasoning content written
by Qwen after receiving the latent `D`; only the exact rendered
`evidence_description` token positions receive labels. Tool-response
serialization, tool-call JSON, prompt text, answer content, and chat-template
wrapper tokens remain ignored. Ownership comes from exact tokenizer offsets
against the rendered transcript: a token is evidence-owned when its start
offset lies inside the evidence span. This includes Qwen's sentence-final token
that also carries the following template newline; a token that starts before
and crosses into evidence remains an error.

Required new fixtures:

- [x] canonical exact label ownership and canonical-to-expanded-model visual
  mapping fixtures;
- [x] pinned real local-Qwen3 representation transcript/model-input golden
  fixture, using a clearly smoke-only prompt and deterministic 56×56 in-memory
  RGB image; it freezes action/evidence transcript identities, strict target
  span/IDs, processor-expanded IDs, two visual blocks, and evidence positions;
- [x] per-sample token-mean then sample-mean reduction, including unequal token
  counts;
- [x] negative summed-NLL Matrix score;
- [x] synthetic native streaming/trainer gradients reach main `D`, every
  D-DeepStack branch, and Adapter-owned parameters while frozen Qwen has no
  gradients; real-Qwen gradient parity remains a production gate;
- [x] baseline `L_gen=on` and ablation `L_gen=off` identities and separate
  Matrix/readability metrics.

## Same-image multi-target sampler parity

Pinned execution sources:

- `scripts/train_tgvf_v3_stage1.py` at SHA256
  `07428d9214e5662ee0477a037a3bee41e93fbf3caf5c90af395bd603d925dc26`;
- clean generic executor at SHA256
  `c54a2cbaee34d2f0d1a1e7a134eca666242c7bef263e0f083c43634d254dff30`.

The Golden reference run used the clean execution path, so its semantic
grouping is the parity baseline:

- group key: `image_id`, falling back to the image path;
- a whole image group belongs to one rank using
  `sha1(image_key) % world_size`;
- group order and group-member order are independently shuffled by a local RNG
  seeded from `seed + epoch`;
- a batch contains targets from exactly one image group;
- incomplete material that cannot form a permitted local group batch is
  dropped;
- the batch-size-5 legacy path permits both four- and five-row partitions under
  the pinned exact partition rule; other local batch sizes use their full size
  as the minimum.

The historical checkpoint did not persist enough sampler cursor/epoch state to
guarantee the exact next batch after arbitrary mid-epoch resume. The new
pipeline must preserve the observable grouping semantics while adding explicit
sampler epoch, materialized epoch order or equivalent RNG state, and batch
cursor to the training checkpoint. This is a documented correctness repair,
not claimed byte-for-byte resume parity with the old flaw.

For the pinned train file, the exact local batch-size-4 rule materializes
25,592 of 35,542 focus rows per epoch and drops 9,950 rows. In particular, all
groups with fewer than four targets and group remainders are unused. This is
reproduced for baseline parity and logged explicitly; smaller/variable group
sizes are a later data-efficiency comparison because they change both the
negative count and Matrix-CE gradient scale.

Required new fixtures:

- [x] no batch mixes image keys;
- [x] no image group is split across ranks;
- [x] deterministic group/member shuffle for a fixed seed and epoch;
- [x] incomplete-group behavior, including the batch-size-5 historical exception;
- [x] different epochs change order without changing membership;
- [x] checkpoint/resume emits the same next batch as uninterrupted execution.
- [x] exact duplicate target strings within one image group fail closed rather than
  becoming false negative columns; semantic near-duplicate handling remains a
  manifest-level decision.

The read-only legacy-data audit records 35,542 accepted train rows in 9,186
groups (6,398 groups with at least four targets), 14,480 excluded rows, and
3,004 accepted-row leakage warnings under manifest
`4160198e65268e33f1c36d050f74498f4f8fa35f3ac263202ee8bfdf5f5cd820`.
Validation has 1,382 accepted rows in 376 groups (226 groups with at least four
targets), 641 excluded rows, and 108 leakage warnings under manifest
`e44cbd6f86ff82879b3be312d9a23198b7267bccd710cbe7d1ecc1dc9954ea15`.
Group key, stable UID, and content hash are disjoint, but seven exact resolved
image paths overlap. No loader or sampler code silently removes them; an
accepted split decision and replacement manifest identities remain required.

## Historical unit-test inventory

Pinned tests:

- `tests/test_tgvf_training.py` at SHA256
  `e74aea92b65822f651c69c08796d21cb7ebc4d31223526a9dbf8c3d3832f3132`;
- `tests/test_tgvf_v3_stage1.py` at SHA256
  `f9010aa7d37143a4d8d98d3d0559868eab8018b62ad74287b0e9d08763bc0dcf`.

The new suite must cover or explicitly adapt:

- dataset required fields, focus-row filtering, and leakage warning semantics;
- same-image group/pair fallback to image path;
- Matrix-CE diagonal preference and manual-gradient parity;
- evidence readout label masking and `D`-position replacement;
- source-grid position/token-count validation;
- readout gradients reach `D` while Qwen remains frozen;
- TGVF Adapter parameters remain trainable under a frozen Qwen model;
- deployable Adapter checkpoint excludes Qwen weights;
- post-`D` evidence queries cannot attend to original-image keys while pre-tool
  queries retain their allowed context;
- D-DeepStack branch injection, ordering, shape checks, and gradients;
- attention entropy/top-k mass/coverage reduction;
- D and source-visual norm statistics as diagnostics only.

Tests tied to project-specific protocol tokens, historical tag rendering, or
token-row training are negative migration checks: the native implementation
must prove those paths are absent rather than reproduce them.

### Function-by-function migration map

`DONE` below means that the named pure semantic has a new fixture. `PARTIAL`
means only a lower-level primitive exists; it must not be reported as an
end-to-end reproduction. `OPEN` remains blocking. `PROPOSED_EXCLUSION` requires
explicit acceptance before Gate AD-13 can close.

| Pinned legacy test function | Status | New fixture or required disposition |
|---|---|---|
| `test_teacher_guide_dataset_loads_jsonl_and_warns_on_leakage` | `DONE` | strict retained-JSONL loader, immutable full-disposition manifest, leakage record/warning, source hash, and overlap fixtures in `test_data.py` |
| `test_default_loss_weights_are_conservative` | `PROPOSED_EXCLUSION` | historical defaults do not define the native objective; accepted weights need new identities |
| `test_training_config_defaults_readout_prompt_target_dropout` | `PARTIAL` | prompt text/hash and target visibility are explicit and no hidden dropout default exists; production prompt/control policy remains open |
| `test_token_direct_can_use_per_target_token_output_length` | `PROPOSED_EXCLUSION` | unused legacy Adapter variant; selected structure has image-layout-owned `D` length |
| `test_non_token_direct_rejects_none_num_foveated_tokens` | `PROPOSED_EXCLUSION` | unused legacy variant-builder validation |
| `test_visual_token_manifold_loss_returns_finite_scalar` | `PROPOSED_EXCLUSION` | manifold optimizer contribution is fixed to zero; no replacement loss is implemented |
| `test_attention_diagnostics_report_slot_entropy_mass_and_coverage` | `DONE` | `tests/representation/training/test_metrics.py` |
| `test_attention_diagnostics_average_sub_slots_to_fvt_slots` | `DONE` | `tests/representation/training/test_metrics.py` |
| `test_fvt_norm_diagnostics_and_summary_values` | `DONE` | diagnostic-only fixtures in `tests/representation/training/test_metrics.py` |
| `test_disabled_optional_losses_skip_cleanly` | `DONE` | native objective/config fixtures enforce manifold disabled at zero, norm unset, and separately identified Matrix-only ablation |
| `test_same_image_negative_groups_fallback_to_image_path` | `PARTIAL` | key fallback is covered; the older size-2 pair/cyclic path is not the selected Golden sampler and needs an accepted exclusion |
| `test_same_image_negative_matrix_ce_prefers_diagonal_scores` | `DONE` | pure Matrix-CE value fixture |
| `test_same_image_negative_matrix_ce_score_gradients_match_autograd` | `DONE` | FP32 autograd plus FP16/BF16 legacy-cast fixtures |
| `test_readout_inputs_mask_prompt_and_replace_only_fvt_positions` | `DONE` | exact evidence ownership, two-block processor mapping, and synthetic native readout integration in `test_transcript.py`, `test_native_pipeline.py`, and the real processor golden |
| `test_readout_inputs_can_use_source_image_grid_positions` | `PARTIAL` | native Qwen3 group builder constructs processor grid/M-RoPE positions and the real golden locks both visual expansions; real 8B forward parity remains open |
| `test_readout_inputs_reject_source_grid_token_count_mismatch` | `DONE` | typed runtime/readout/token-expansion contracts fail closed on grid, source-token, or visual-block disagreement |
| `test_readout_prompt_can_omit_target` | `DONE` | hashed prompt schema permits `{question}` without `{target}` and the native group fixture exercises that form; production matched-control evaluation is open |
| `test_readout_inputs_without_target_still_replace_fvt_positions` | `DONE` | native group-builder fixture omits target from the user prompt while preserving strict call-target extraction and both visual blocks |
| `test_readout_loss_backprops_to_fvt_not_frozen_lm` | `DONE` | synthetic streaming/trainer fixtures backpropagate through candidate `D` tensors and Adapter parameters while frozen Qwen receives no gradients; real 8B proof remains open |
| `test_tgvf_module_params_remain_trainable_when_qwen_is_frozen` | `DONE` | runtime, native group-builder, streaming, and trainer fixtures enforce frozen Qwen plus trainable Adapter ownership; real 8B FSDP execution remains open |
| `test_tgvf_checkpoint_saves_module_without_qwen_weights` | `DONE` | strict Adapter-only artifact and identity manifest exclude borrowed Qwen mergers; CPU save/load and tamper fixtures pass, distributed production export remains open |
| `test_build_tgvf_module_supports_target_slot_variant` | `PROPOSED_EXCLUSION` | unused legacy variant builder |
| `test_build_tgvf_module_supports_v2_dynamic_token_variants` | `PROPOSED_EXCLUSION` | unused legacy variants; only the selected bidirectional D-DeepStack structure is in scope |
| `test_v3_stage1_dataset_keeps_focus_and_skips_direct_rows` | `DONE` | strict retained-focus transform and every-row disposition fixture in `test_data.py` |
| `test_weak_strict_mask_blocks_only_post_tgvf_queries` | `DONE` | streaming native readout constructs and consumes the post-evidence source-key block; synthetic query-row fixture passes |
| `test_v3_stage1_readout_loss_backprops_to_d_not_frozen_qwen` | `DONE` | synthetic native streaming/trainer gradient and frozen-Qwen fixtures; real 8B gradient evidence remains open |
| `test_v3_stage1_readout_can_inject_d_deepstack_features` | `PARTIAL` | native pipeline plus streaming path carries Adapter-produced main `D` and all branches and consumes source-key blocking; real Qwen output/gradient and intended-layer mapping remain open |
| `test_qwen3_position_ids_unwraps_peft_like_model_but_uses_wrapper_embeddings` | `OPEN` | native Qwen-family/PEFT ownership replacement fixture |

The table records semantic migration only. Historical names remain provenance
labels and are not current package, configuration, or experiment identities.

## Historical internal evaluation inventory

Pinned suite and task sources:

- `eval/run_tgvf_v3_eval_suite.sh` at SHA256
  `7839d2e03c0e54b7de547b90ac904a44ab27f0055edf3a8ddff2ac2215c6dd6f`;
- readout task at SHA256
  `3df6f807d6b0b8d718d88e570451ac62be7eda5db311d48c311cde3622490e20`;
- query-sensitivity task at SHA256
  `3c241e3081dbcf760475d6837163cd0ff32c3019ea01d9bc32e07c60889bf791`;
- distribution task at SHA256
  `a84e525fc4ebe046c776f8da4f110280b7cd6c18718738cb9e580a31e3d9b4c5`;
- shared scoring/reduction sources recorded in `LEGACY_REFERENCE.md`.

### Readout metrics

Reproduce correct `D`, target-only/no-`D`, random `D`, wrong same-image `D`, and
wrong different-image `D` controls. Report:

- mean and median correct-`D` NLL;
- mean target-only and random-`D` NLL;
- mean correct advantage against every control;
- fraction of samples where correct `D` beats every control;
- grouped results by evidence type, source profile, answer type, and visual
  difficulty;
- proof that no second full Qwen forward path was used when that contract is
  selected.

The native suite additionally swaps main `D` and all D-DeepStack branches as
one observation and verifies matched layout, positions, masks, dtype, and
shape. Historical numeric results are comparison evidence, not automatic
promotion thresholds.

### Query-sensitivity metrics

For eligible same-image multi-target groups, save the full score matrix and
report:

- retrieval top-1 and top-2;
- mean reciprocal rank;
- mean and median diagonal gap;
- per-group and per-sample rows;
- grouped results by evidence type and source profile.

The old score convention is NLL-lower-is-better; training Matrix CE stores the
equivalent log-likelihood-higher-is-better score. Tests must make this sign
conversion explicit.

### Distribution and health diagnostics

Reproduce, without treating them as an accepted norm-training objective:

- finite rate;
- near-identical-token collapse rate and warning;
- `D` and source merged-visual tensor distribution statistics;
- average token norms and their ratio;
- attention entropy, top-k mass, and visual-token coverage;
- main/branch shapes, finite values, frozen-Qwen status, mask mode, position
  source, and no-second-full-forward invariant.

Historical manifold diagnostics were inactive for the 4096-dimensional `D`
versus 1152-dimensional pre-merge/source feature mismatch. Manifold optimizer
weight is zero in the new baseline.

## Native additions beyond historical parity

These are required because the new system changes the protocol or capability:

- [x] exact local-Qwen3 native tool transcript/token/processor round trip with
  no tokenizer growth, separately identified from the policy transcript fixture;
- [x] strict target span plus both contextual-hidden-state and target-token-
  embedding `Hq` paths in the synthetic native group fixture; a real 8B provider
  comparison remains open;
- [x] provider-specific configuration/checkpoint identity;
- [ ] paired provider runs with identical data/order/initialization/seed;
- [x] synthetic main `D` and every D-DeepStack branch causal/readability and
  post-`D` source-key-block controls;
- [ ] real-Qwen free continuation and counterfactual value-flip evaluation;
- [x] deterministic streaming score/recompute equality and exact source/candidate
  observation identity in bounded fixtures;
- [x] CPU checkpoint/resume including sampler state, provider/data/optimizer/
  scheduler/accumulation identities, RNG, and global step;
- [ ] two-rank representation FSDP2 optimizer/checkpoint/resume evidence.

## Golden-equivalence audit

The selected bidirectional attention, gate/residual, independent branch Adapter,
and visual-salience equations are now checked by an independent functional
oracle in `tests/representation/test_adapter_reference_parity.py`. For seeded
small dimensions, it compares main/three-branch outputs and gradients with
respect to target input, all four visual inputs, and all 104 Adapter-owned
parameters in FP32 (`atol=rtol=2e-6`) and BF16 (`atol=rtol=3e-2`). Borrowed
Qwen-merger parameters remain frozen and outside the owned artifact.

The native Qwen3 runtime binds exact model/component identities and the accepted
`4096/1152`, merge-size-2, branch-layer `(8,16,24)` architecture; the new
checkpoint path exports only Adapter-owned tensors and binds provider, model,
prompt, objective, data, sampler, optimizer, scheduler, precision,
accumulation, and initialization identities. The streaming readout consumes all
branches and blocks original-image keys for evidence prediction.

This is still not a Golden-checkpoint equivalence or trained-artifact claim. The
remaining blockers are:

- load the pinned historical 104-tensor state under the exact accepted Qwen3
  merger snapshot and compare real-dimension outputs plus input/parameter
  gradients under documented tolerances;
- prove on the real model that each captured/injected branch maps to the intended
  Qwen layer and that the complete native readout remains deterministic;
- execute both target-conditioning providers with real local Qwen3 state and
  record target-span/`Hq`, specificity, readability, and gradient evidence;
- pass a real two-rank representation FSDP2 optimizer/checkpoint/resume smoke;
- train and validate a newly initialized native-format Adapter rather than
  loading the historical checkpoint.

The functional oracle, real-processor golden, synthetic group/trainer fixtures,
and CPU checkpoint parity are supporting evidence, not substitutes for those
gates.

## Open scientific and production items

- accept a train/validation split policy that resolves the seven exact resolved-
  image-path overlaps without silent filtering; record dataset/image licenses
  and perform a perceptual near-duplicate audit;
- replace the smoke-only golden prompt with an accepted production prompt
  identity and bind the contextual provider's hidden layer;
- freeze Adapter initialization distribution/seed and final nonzero `L_gen` and
  Matrix-CE weights; define the required Matrix-only ablation size;
- decide whether any norm objective exists. No formula or default is currently
  implemented, and manifold optimization remains zero;
- freeze AdamW, scheduler, BF16, accumulation, clipping, validation cadence,
  checkpoint cadence, and metric logging in a production TOML artifact;
- run the real Qwen3 native group/readout backward for both providers and the
  two-rank representation FSDP2 optimizer/checkpoint/resume smoke;
- complete exact historical-checkpoint/real-merger output and gradient parity;
- set quantitative readout, retrieval, branch, causal-flip, free-continuation,
  and reasoning-retention promotion thresholds;
- build a separately identified representation artifact and complete native
  transcript/DeepStack/objective fixtures before claiming Qwen2.5-VL support.
