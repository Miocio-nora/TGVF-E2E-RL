# Representation Phase Parity Inventory

Status: **implementation inventory accepted; fixtures and native semantics in progress**

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
- [ ] pinned real representation transcript/model-input golden fixture;
- [x] per-sample token-mean then sample-mean reduction, including unequal token
  counts;
- [x] negative summed-NLL Matrix score;
- [ ] gradients reach main `D` and every D-DeepStack branch but not frozen Qwen;
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
| `test_teacher_guide_dataset_loads_jsonl_and_warns_on_leakage` | `OPEN` | retained-JSONL loader, immutable manifest, and leakage record |
| `test_default_loss_weights_are_conservative` | `PROPOSED_EXCLUSION` | historical defaults do not define the native objective; accepted weights need new identities |
| `test_training_config_defaults_readout_prompt_target_dropout` | `OPEN` | native target visibility/dropout decision and fixture |
| `test_token_direct_can_use_per_target_token_output_length` | `PROPOSED_EXCLUSION` | unused legacy Adapter variant; selected structure has image-layout-owned `D` length |
| `test_non_token_direct_rejects_none_num_foveated_tokens` | `PROPOSED_EXCLUSION` | unused legacy variant-builder validation |
| `test_visual_token_manifold_loss_returns_finite_scalar` | `PROPOSED_EXCLUSION` | manifold optimizer contribution is fixed to zero; no replacement loss is implemented |
| `test_attention_diagnostics_report_slot_entropy_mass_and_coverage` | `DONE` | `tests/representation/training/test_metrics.py` |
| `test_attention_diagnostics_average_sub_slots_to_fvt_slots` | `DONE` | `tests/representation/training/test_metrics.py` |
| `test_fvt_norm_diagnostics_and_summary_values` | `DONE` | diagnostic-only fixtures in `tests/representation/training/test_metrics.py` |
| `test_disabled_optional_losses_skip_cleanly` | `OPEN` | negative configuration fixture after the native objective schema is accepted |
| `test_same_image_negative_groups_fallback_to_image_path` | `PARTIAL` | key fallback is covered; the older size-2 pair/cyclic path is not the selected Golden sampler and needs an accepted exclusion |
| `test_same_image_negative_matrix_ce_prefers_diagonal_scores` | `DONE` | pure Matrix-CE value fixture |
| `test_same_image_negative_matrix_ce_score_gradients_match_autograd` | `DONE` | FP32 autograd plus FP16/BF16 legacy-cast fixtures |
| `test_readout_inputs_mask_prompt_and_replace_only_fvt_positions` | `OPEN` | native transcript/readout integration |
| `test_readout_inputs_can_use_source_image_grid_positions` | `OPEN` | native layout/M-RoPE readout integration |
| `test_readout_inputs_reject_source_grid_token_count_mismatch` | `OPEN` | native layout fail-closed fixture |
| `test_readout_prompt_can_omit_target` | `OPEN` | native matched-control semantics; not silently inherited |
| `test_readout_inputs_without_target_still_replace_fvt_positions` | `OPEN` | native matched-control readout integration |
| `test_readout_loss_backprops_to_fvt_not_frozen_lm` | `OPEN` | real readout gradient/frozen-Qwen integration |
| `test_tgvf_module_params_remain_trainable_when_qwen_is_frozen` | `PARTIAL` | synthetic Adapter/frozen-merger gradient exists; a real frozen-Qwen whitelist fixture is open |
| `test_tgvf_checkpoint_saves_module_without_qwen_weights` | `PARTIAL` | strict 104-tensor Adapter-owned subset excludes mergers; fresh file checkpoint and identity manifest are open |
| `test_build_tgvf_module_supports_target_slot_variant` | `PROPOSED_EXCLUSION` | unused legacy variant builder |
| `test_build_tgvf_module_supports_v2_dynamic_token_variants` | `PROPOSED_EXCLUSION` | unused legacy variants; only the selected bidirectional D-DeepStack structure is in scope |
| `test_v3_stage1_dataset_keeps_focus_and_skips_direct_rows` | `OPEN` | retained-data transform/filter fixture |
| `test_weak_strict_mask_blocks_only_post_tgvf_queries` | `PARTIAL` | pure key-block mask exists; native readout execution does not yet consume it |
| `test_v3_stage1_readout_loss_backprops_to_d_not_frozen_qwen` | `OPEN` | native readout integration |
| `test_v3_stage1_readout_can_inject_d_deepstack_features` | `PARTIAL` | synthetic live injection/sensitivity covers main `D` and each branch; original-image key blocking, real Qwen, Adapter-output provenance, and gradient parity remain open |
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

- exact Qwen native tool transcript/token round trip with no tokenizer growth;
- target-span and `Hq` identity under both conditioning providers;
- provider-specific artifact identity and paired-run comparison;
- main `D` and every D-DeepStack branch causal/readability controls;
- free continuation and counterfactual value-flip evaluation;
- deterministic forward and exact observation identity;
- representation checkpoint/resume including sampler state, provider identity,
  data manifest, optimizer, scheduler, accumulation, RNG, and global step.

## Golden-equivalence audit

The selected bidirectional attention, gate/residual, independent branch Adapter,
and restored visual-salience equations structurally match the pinned reference.
The opt-in Adapter-owned tensor-subset primitive now exports the 104 selected
TGVF tensors and strictly rejects borrowed Qwen-merger state. It is not yet
wired to a representation checkpoint writer and is not a complete artifact or
Golden-equivalence claim. The remaining blockers are:

- a production factory that binds the accepted Qwen3 main merger and three
  DeepStack mergers with dimensions `4096/1152`, merge size `2`, and branch
  layers `(8, 16, 24)` under exact model identities;
- fixed-reference output and input/parameter-gradient parity, including BF16;
- native representation readout using D-only DeepStack and post-D blocking of
  original-image keys;
- model-level proof that branch payload order maps to the intended Qwen layers;
- either masked variable-length batched target conditioning or a fail-closed
  execution contract that runs such Adapter inputs per sample.
- a checkpoint manifest/loader that binds model family, conditioning provider,
  architecture, projection identities, dtype, data, and training state and uses
  the Adapter-owned subset rather than ordinary `state_dict()` for deployment.

Synthetic shape/gradient tests and an Adapter-only state test are supporting
fixtures, not substitutes for those gates.

## Open items before implementation promotion

- exact retained train/validation manifests and image-disjoint split;
- the native post-tool evidence transcript and label span;
- Adapter initialization distribution and seed contract;
- final `L_gen` and Matrix-CE weights and required ablation size;
- norm-loss inclusion, if any;
- thresholds for readout, retrieval, branch, causal-flip, and free-continuation
  promotion;
- exact optimizer/scheduler/precision/accumulation contract.
