# Controlled Legacy Reference

Status: **bounded framework and representation-training port inventory recorded;
production data/artifact acceptance remains open**
Recorded: **2026-07-18 JST**
Updated: **2026-07-19 JST**

## Reference repository

```text
path:   /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
branch: research/reasoning-health-20260716
HEAD:   a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
state:  dirty; 52 porcelain entries at scaffold time
```

Because the worktree is dirty, `HEAD` alone does not identify arbitrary current
reference behavior. No unregistered code may be ported until the user approves
an archive commit/tag or an immutable patch/content-hash bundle. For
I8H-20260719, the exact committed paths and the separately recorded clean
working-file hashes below form the authorized bounded content-hash bundle; this
does not authorize any other file from the dirty worktree.

The pinned tree contains no `LICENSE`, `COPYING`, or `NOTICE` file. The user
explicitly authorized use of this legacy project, but no separate license or
ownership assertion is recorded here. This repository will therefore use
adapted, symbol-level reimplementations with recorded lineage; it will not copy
whole legacy files or comments into the public repository.

## Access rule

The old repository remains a sibling, read-only reference. It is not imported,
symlinked, vendored wholesale, or added as a submodule.

Inspect a committed file explicitly with:

```bash
git -C /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm \
  show <frozen-commit>:<path>
```

For a currently dirty candidate file, verify its working-tree SHA256 against
this document before discussing it, then replace that identity with the frozen
archive identity before porting.

Do not run broad searches from the parent `r-vlm` directory.

## Candidate exact-extraction whitelist

The following is a candidate whitelist, not permission to copy an entire file.

From `src/revisit_vlm/tgvf_foveal.py`:

- `FovealCrossAttentionOutput`;
- `TGVFv2Bidirectional`;
- `TGVFv2BidirectionalDDeepStack`;
- the minimal cross-attention/input-validation/metadata helpers they require.

The dependency closure must explicitly account for
`_merge_with_frozen_qwen_merger_module`,
`finalize_tgvf_output_with_frozen_qwen_merger`, `_visual_module`, and
`_infer_spatial_merge_size`. The broad multi-variant `build_tgvf_module` path
and every unused TGVF variant are forbidden.

Working-tree SHA256 at scaffold time:

```text
f2244980599510c976a20dbbe227523fde5af72f26e8253188e04c072456853f
```

This exact working-file identity is the registered mathematical lineage for the
selected TGVF Adapter. The new
`tests/representation/test_adapter_reference_parity.py` fixture evaluates an
independently written functional form of the selected attention, gate/residual,
main-projection, and three branch-projection paths. It compares outputs plus
target, visual-input, and all 104 Adapter-owned parameter gradients in FP32 and
BF16. This is bounded equation-level evidence; it does not load the historical
checkpoint or close real-dimension/real-Qwen merger parity.

The Golden 8B construction to verify is:

```text
d_lm=4096
d_v=1152
spatial_merge_size=2
attn_dim=None
branch_layers=(8, 16, 24)
```

## Thin-reference whitelist

These areas may be inspected and reimplemented behind new interfaces. They are
not whole-file copy candidates.

| Legacy source | Permitted reference purpose | Scaffold-time SHA256 |
|---|---|---|
| `revisit_vlm_clean/src/revisit_vlm_clean/deepstack.py` | DeepStack payload and mask semantics | `7499a3dbe1df2c654c8b6ff6a8d06d91ba2900bd9a76d7ce2e9b3058f2df0c5c` |
| `src/revisit_vlm/qwen3_vl_tgvf.py` | vision tap, source geometry, native visual append, M-RoPE reference only | `09401be77bc0a13fd48eb04681b8cfd00cbd2e5f33b59efbfe825e10ab163801` |
| `src/revisit_vlm/tgvf_v3_stage1.py` | historical representation losses, diagnostics, and candidate training logic for adapted reimplementation; not the pipeline as a whole | `78b465ec67d40c6d60863715c43171f373483b20c11447b13b56b6fe8e28384a` |

`qwen3_vl_tgvf.py` is in the dirty worktree and mixes legacy protocols,
Stage2 behavior, added-token logic, and useful low-level Qwen mechanics. It
must never be copied as a module.

## Reference-only model-path lookup

The user authorized a read-only lookup of the primary model path in the old
project. The following committed file was inspected directly; no broad legacy
search was run:

```text
legacy source: README.md
commit: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
content SHA256: ef18e93e8d4d8edbf969346801ebd8a4e4630a94a65595062a70be472ca9c377
inspection date: 2026-07-19 JST
role: specification only; model-path lookup
```

It records the default Qwen3 model used by that project as:

```text
/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
```

That exact directory was present when checked on 2026-07-19 JST. The path is a
stable operational model identity by user decision. No files inside it were
inspected by this lookup, and no full directory or weight-shard hash is required.
The new project still freezes processor/tokenizer/chat-template behavior through
exact native transcript fixtures and fixture hashes before rollout.

The committed README does not identify local paths for the fixed
`Qwen/Qwen2.5-VL-7B-Instruct` compatibility model or the reserved
`Qwen/Qwen2.5-72B-Instruct` judge. Those paths remain explicit `[TBD]` values
rather than guessed legacy paths.

## `revisit_vlm_clean` reference boundary

The user confirmed that the clean subtree may be used as a design reference.
This is exact-file, read-only authorization under the same provenance rule, not
permission to copy, import, or broadly inspect the subtree. The already recorded
`revisit_vlm_clean/src/revisit_vlm_clean/deepstack.py` entry is the first such
reference.

Potential additional specification-only topics include Qwen-family model
loading/configuration, target-conditioning helpers, multimodal layout/state,
and judge-service configuration. Before inspecting any of them, add the exact
path plus committed or working-tree content identity to this document. No path
for those topics has been approved or inspected yet.

Exact-file inspection of a clean RL/agent file may inform a new interface after
it is registered here. It does not authorize copying its trainer, replay,
objective, reward, executor, or scheduler implementation, and it does not make
its mathematics or behavior correct by assumption.

### Registered clean-subtree files for the framework goal

The user explicitly directed the implementation to prefer the embedded clean
subtree. The independently named sibling path `.../r-vlm/revisit_vlm_clean`
does not exist; the authoritative clean subtree is committed inside the pinned
legacy repository at `revisit_vlm/revisit_vlm_clean/`. The following exact
files are registered before content inspection or adaptation:

| Committed path at `a200437123afe6fbb481a6c9cf9b7ddf61ff36b8` | SHA256 | Permitted purpose |
|---|---|---|
| `revisit_vlm_clean/src/revisit_vlm_clean/deepstack.py` | `7499a3dbe1df2c654c8b6ff6a8d06d91ba2900bd9a76d7ce2e9b3058f2df0c5c` | adapted DeepStack payload/mask semantics |
| `revisit_vlm_clean/src/revisit_vlm_clean/training/stage1_executor.py` | `4f9106c772c20faa64a7dcf700c6a5294ed85efa30b797eb4925debef381ae5a` | specification-only TGVF construction, loss, and state boundaries; executor is not ported |
| `revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py` | `e19c388b9b2d033b4d3b176ba16484f3b3c8e6b6ef6dd88ac6f1d9ed3874f8d5` | specification-only representation configuration and artifact inputs; CLI is not ported |
| `src/revisit_vlm/tgvf_training.py` | `ebae9266cafc4f83685f6a7d46a5b0fc501111f3ba9ee257c9e530e636857dd4` | specification-only exact Matrix-CE, same-image grouping/sampling, representation auxiliary losses, and diagnostic semantics; only individually recorded helpers may be adapted |
| `revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py` | `c54a2cbaee34d2f0d1a1e7a134eca666242c7bef263e0f083c43634d254dff30` | specification-only representation execution, batching, optimizer, metric, checkpoint, and resume behavior; generic executor is not ported wholesale |
| `revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py` | `cac206c3620f417d81cf6a3af9c62c85b63db53bb27b2b13ab2629dc0cd4c430` | specification-only resolved representation configuration and batch identity |
| `revisit_vlm_clean/src/revisit_vlm_clean/defaults.py` | `a1f7c070cff26a03a9374065fc544aa9a86a89195295c7c6d3f1517585ea0901` | specification-only historical representation defaults; no value is accepted solely because it was a default |
| `tests/test_tgvf_training.py` | `e74aea92b65822f651c69c08796d21cb7ebc4d31223526a9dbf8c3d3832f3132` | test-specification reference for Matrix-CE, grouping, readout, loss, and diagnostic parity |
| `tests/test_tgvf_v3_stage1.py` | `f9010aa7d37143a4d8d98d3d0559868eab8018b62ad74287b0e9d08763bc0dcf` | test-specification reference for Qwen3 representation transcript/readout, D-DeepStack, masking, objective, and diagnostics |
| `scripts/train_tgvf_v3_stage1.py` | `07428d9214e5662ee0477a037a3bee41e93fbf3caf5c90af395bd603d925dc26` | specification-only historical representation sampler, execution loop, validation/internal generation, metrics, checkpoint, and resume semantics; the script and legacy protocol are not ported wholesale |
| `docs/TGVF_V3_STAGE1_TRAINING.md` | `3a2087a647a46825871ce6346703a35a4608aa31a841be865840ab82ad61d572` | specification-only inventory of historical representation training, diagnostics, evaluation commands, and known limitations |
| `eval/run_tgvf_v3_eval_suite.sh` | `7839d2e03c0e54b7de547b90ac904a44ab27f0055edf3a8ddff2ac2215c6dd6f` | specification-only inventory and invocation contract for historical readout, query-sensitivity, and distribution evaluation tasks |
| `eval/eval_v3_readout.py` | `3df6f807d6b0b8d718d88e570451ac62be7eda5db311d48c311cde3622490e20` | specification-only exact historical readout controls, per-sample records, aggregations, and metrics |
| `eval/eval_v3_query_sensitivity.py` | `3c241e3081dbcf760475d6837163cd0ff32c3019ea01d9bc32e07c60889bf791` | specification-only exact historical same-image score-matrix retrieval metrics and reports |
| `eval/eval_v3_fvt_distribution.py` | `a84e525fc4ebe046c776f8da4f110280b7cd6c18718738cb9e580a31e3d9b4c5` | specification-only exact historical D distribution, finiteness, collapse, and norm diagnostics |
| `eval/v3_common.py` | `6400e7ac9fb76b3b4c5951b33ca26bf4e5619d0e30e30769a10d8f3bf43dcb38` | specification-only shared historical representation evaluation sample preparation, controls, scoring, cache, and identity semantics |
| `eval/metrics.py` | `aa52ad8fca252ddbf7a36d4f47990e96dd7320898c76253fbc8e6c1c7404c794` | adapted metric-reduction semantics for readout, retrieval, and distribution parity fixtures |
| `revisit_vlm_clean/src/revisit_vlm_clean/stage3_grpo/model_prepare.py` | `6424a283a7281f6561ab638666d31216221fd61e42b03210821cc48e7c4aaefa` | specification-only Qwen/TGVF model-loading checks; legacy RL path is not ported |
| `revisit_vlm_clean/tests/test_deepstack.py` | `cfb9e6692ede29f0460cb251b48869f11499adb9ff62b2fa7ff182855c738685` | adapted semantic test cases |
| `revisit_vlm_clean/src/revisit_vlm_clean/tgvf_protocol.py` | `9af43f4c884f4a20b3a05c61d235ba4ed555f12154c3d4bbe460b791eefb955d` | negative/specification comparison only; legacy protocol implementation is forbidden |
| `revisit_vlm_clean/tests/test_tgvf_protocol.py` | `2284d34f5f980abfc7aaa3ec4e93cda063fbf0b78a580c5bc088445dcf26f64a` | negative test ideas only; no legacy serialization fixture becomes canonical |

The clean executor delegates the underlying TGVF mathematics to the already
registered `src/revisit_vlm/tgvf_foveal.py` symbols. Therefore the clean subtree
is the preferred orchestration/specification reference, while any mathematical
extraction still uses the narrow symbol whitelist and parity obligations above.

## Representation data reuse boundary

The legacy representation dataset is eligible for reuse only after its exact
manifest, sample identities, transforms, provenance, license, and split policy
are recorded here or in a linked immutable data artifact. It must be adapted to
the new native-format representation pipeline. The historical rendered format,
serialization, launcher, and resume state are not reused as the new pipeline.

The user selected the same historical representation data population for the
new pipeline. The following exact worktree files were identified by the pinned
training documentation and content-hashed before row inspection:

| Role | Exact legacy worktree path | SHA256 |
|---|---|---|
| candidate retained train rows | `data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl` | `8406f8f843f927642aa2d728f1896579f20c44ca7329b86cb35b42544f73f666` |
| candidate validation rows (not path-disjoint after audit) | `data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl` | `a228d28db76625d166dab874806c9034a244a683d41c7cecdc7f10f1aa754308` |

These files are not committed at the frozen legacy commit and remain external
data artifacts. Their hashes authorize bounded read-only schema/count/split
inspection and a new manifest that references them; they do not authorize
copying the JSONL or image assets into this public repository. Dataset-source
license, image-asset identities, duplicate/near-duplicate policy, and an
accepted split decision remain Gate A0 items.

The new `retained_focus_rows_v1` loader was run read-only against both exact
source hashes. It records every source-line disposition, resolved image path,
duplicate and historical target/short-answer leakage signal in an immutable
`representation_data_manifest_v1`; leakage is a warning/record, not an
automatic exclusion. The bounded audit produced:

| Audit value | Train | Validation |
|---|---:|---:|
| source rows | 50,022 | 2,023 |
| accepted representation rows | 35,542 | 1,382 |
| excluded rows | 14,480 | 641 |
| unique accepted image-group keys | 9,186 | 376 |
| groups with at least four accepted targets | 6,398 | 226 |
| accepted rows with the historical leakage signal | 3,004 | 108 |
| rows materialized per epoch by exact local batch-size-4 grouping | 25,592 | 904 |
| accepted rows unused per epoch by that grouping | 9,950 | 478 |

The canonical in-memory manifest identities for that exact audit are:

```text
train:      4160198e65268e33f1c36d050f74498f4f8fa35f3ac263202ee8bfdf5f5cd820
validation: e44cbd6f86ff82879b3be312d9a23198b7267bccd710cbe7d1ecc1dc9954ea15
```

The split audit found no overlap in `image_group_key`, `stable_image_uid`, or
`item_content_hash`, but it found **seven distinct exact resolved image-path
overlaps**. This corrects the earlier aggregate statement that all four keys
were disjoint. The loader and audit do not silently filter, reassign, or bless
those rows: both manifests describe the unmodified transformed populations.
Any training launch under the required-disjoint contract must fail until an
explicit split/exclusion policy is accepted and produces newly identified
manifests. The seven exact overlaps also do not substitute for a perceptual
near-duplicate audit.

The batch-size-4 parity sampler excludes about 28% of otherwise accepted train
rows in each epoch because groups of one to three never form a batch and group
remainders are dropped. Exact baseline parity preserves this behavior; using
smaller or variable group sizes is a separately named data-efficiency
experiment rather than a silent sampler change.

## Golden representation checkpoint

Historical candidate:

```text
/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/
outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/
stage1_micro4/clean_training_execution/checkpoint_step_2000.pt
```

Recorded SHA256:

```text
b119379fc13a3eee1d19fb347bda729262592599218ea1ff88733ba142cb0c0b
```

Audited semantics:

- no `qwen_lora`;
- `tgvf_module` contains 104 BF16 tensors and 72,055,808 parameters;
- main adapter plus D-DeepStack branches 8, 16, and 24;
- top-level checkpoint also contains optimizer, scheduler, global step,
  configuration, and learned `protocol_c_token_rows`.
- the state loads with 104/104 keys under the exact constructor above;
- the module consumes `Hq[T,4096]`, main pre-merge vision
  `V_pre[N,1152]`, and three DeepStack pre-merge branches `[N,1152]`, with
  `N % 4 == 0`;
- final main `D` and three D-DeepStack outputs are `[N/4,4096]`.

Only `tgvf_module` plus minimal architecture/provenance metadata is eligible
for a new representation artifact. Optimizer, scheduler, step state, and all
protocol-token rows are forbidden.

The `tgvf_module` state alone is not the complete numerical producer. Final
projection depends on the identical base-model snapshot's frozen
`visual.merger` and `deepstack_merger_list[0:3]`. Their identity and frozen
state must be pinned in the new representation manifest and parity tests.

The weights are 8B-specific and were trained under a focus-force prompt, fixed
pre-focus reasoning, learned Protocol-C boundaries, and Protocol-C target
hidden states. They are a parity, provenance, and diagnostic-comparison
reference only; they are not a direct initialization or runtime checkpoint for
the new native-format representation phase. The new native `Hq` contract must
pin the JSON-escaped value-span mapping, quote/syntax exclusion, hidden layer,
and token-time alignment before compatibility can be assessed.

## Explicitly forbidden legacy areas

- all legacy protocol implementations, including Protocol C, the historical
  apparently-native Protocol D, and Protocol E tokens, renderers, parsers,
  token-row PEFT, and tokenizer-resize paths;
- broad TGVF variant builders and all unused TGVF variants;
- the historical representation trainer as a whole pipeline, plus its
  serialization, launcher, executor, and resume state. Individually approved
  losses, diagnostics, and training helpers may be adapted only through the
  thin-reference and port-record rules above;
- all Stage2 trainers, fast trainers, runtime engines, adapters, checkpoints,
  data mixtures, reasoning replay, and Golden correction code;
- wholesale extraction, runtime reuse, or correctness-by-assumption for the
  `revisit_vlm_clean.stage3_grpo` implementation and its CLIs, executors,
  policy/reference replay, rewards, judges, and schedulers. A separately
  registered exact file may still be inspected read-only for specification
  purposes under the boundary above;
- legacy force/softforce training identities;
- current benchmark parser code until its final-answer extraction defects are
  fixed and independently tested;
- any runtime import from the old repository.

## Port record template

Before adding any implementation file to the new project, append a record:

```text
new path:
role: exact extraction | adapted extraction | specification only
legacy repository:
legacy frozen commit/tag:
legacy source path:
legacy source SHA256:
symbols/lineage used:
semantic differences:
parity fixture:
reviewed by:
date:
```

An exact extraction requires deterministic output and gradient parity. An
adapted extraction requires explicit semantic tests. A specification-only
reference contributes no copied implementation.

### I8H-20260719 port records

```text
new path: src/tgvf_rl/representation/adapter.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8 plus working-file SHA256 below
legacy source path: src/revisit_vlm/tgvf_foveal.py
legacy source SHA256: f2244980599510c976a20dbbe227523fde5af72f26e8253188e04c072456853f
symbols/lineage used: FovealCrossAttentionOutput, TGVFv2Bidirectional,
  TGVFv2BidirectionalDDeepStack, _cross_attention, _validate_inputs
semantic differences: project-native typed inputs/outputs; no historical stage,
  protocol, model lookup, builder, or output path; frozen merger supplied through
  an explicit projection port; the opt-in Adapter-owned tensor subset excludes
  all borrowed merger parameters while retaining the selected 104 TGVF tensors;
  the new artifact writer consumes that subset, while a newly trained production
  artifact remains a promotion gate
parity fixture: tests/representation/test_adapter.py and
  tests/representation/test_adapter_reference_parity.py (independent functional
  output plus input/104-owned-parameter gradient oracle in FP32/BF16); exact
  legacy-checkpoint and real-dimension Qwen-merger parity remain later gates
reviewed by: Codex I8H-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/deepstack.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: revisit_vlm_clean/src/revisit_vlm_clean/deepstack.py
legacy source SHA256: 7499a3dbe1df2c654c8b6ff6a8d06d91ba2900bd9a76d7ce2e9b3058f2df0c5c
symbols/lineage used: payload ordering and original-image key-block mask semantics
semantic differences: batch-aware typed implementation; D branches required;
  no legacy schema/runtime hooks or historical stage names
parity fixture: tests/representation/test_deepstack.py
reviewed by: Codex I8H-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/schema.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8 plus registered working-file identity
legacy source path: src/revisit_vlm/tgvf_v3_stage1.py
legacy source SHA256: 78b465ec67d40c6d60863715c43171f373483b20c11447b13b56b6fe8e28384a
symbols/lineage used: TGVFv3Stage1Sample data fields, focus-row identity, image-group key
semantic differences: native phase-neutral schema; no legacy protocol fields or rendering;
  immutable sample identity and validation are explicit
parity fixture: tests/representation/training/test_schema.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/__init__.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8 plus registered working-file identity
legacy source path: the individually recorded representation schema, sampler,
  loss, and metric sources listed in the adjacent port records
legacy source SHA256: see adjacent per-module records
symbols/lineage used: public representation-training semantic surface only
semantic differences: package export boundary only; no legacy runtime import,
  protocol renderer, trainer, or default objective values
parity fixture: import coverage in tests/representation/training/
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/sampling.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py;
  scripts/train_tgvf_v3_stage1.py
legacy source SHA256: c54a2cbaee34d2f0d1a1e7a134eca666242c7bef263e0f083c43634d254dff30;
  07428d9214e5662ee0477a037a3bee41e93fbf3caf5c90af395bd603d925dc26
symbols/lineage used: _SingleProcessSampleCursor same-image path; SameImageBatchSampler
semantic differences: explicit immutable sampler identity and exact next-batch
  checkpoint/resume state repair; no general random-sampling fallback
parity fixture: tests/representation/training/test_sampling.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/losses.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: src/revisit_vlm/tgvf_training.py;
  src/revisit_vlm/tgvf_v3_stage1.py
legacy source SHA256: ebae9266cafc4f83685f6a7d46a5b0fc501111f3ba9ee257c9e530e636857dd4;
  78b465ec67d40c6d60863715c43171f373483b20c11447b13b56b6fe8e28384a
symbols/lineage used: same_image_negative_matrix_ce_loss,
  same_image_negative_matrix_ce_score_gradients,
  compute_v3_stage1_lm_losses_batched
semantic differences: pure tensor, phase-neutral API; no legacy model/protocol
  execution; manifold and norm optimization losses are absent
parity fixture: tests/representation/training/test_losses.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/metrics.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: eval/metrics.py; eval/eval_v3_readout.py;
  eval/eval_v3_query_sensitivity.py; eval/eval_v3_fvt_distribution.py;
  src/revisit_vlm/tgvf_training.py
legacy source SHA256: aa52ad8fca252ddbf7a36d4f47990e96dd7320898c76253fbc8e6c1c7404c794;
  3df6f807d6b0b8d718d88e570451ac62be7eda5db311d48c311cde3622490e20;
  3c241e3081dbcf760475d6837163cd0ff32c3019ea01d9bc32e07c60889bf791;
  a84e525fc4ebe046c776f8da4f110280b7cd6c18718738cb9e580a31e3d9b4c5;
  ebae9266cafc4f83685f6a7d46a5b0fc501111f3ba9ee257c9e530e636857dd4
symbols/lineage used: mean, median, pct_positive, grouped_means,
  tensor_distribution_stats, attention_diagnostics, summarize_diagnostics,
  readout/query/distribution report reductions
semantic differences: typed pure-data reductions; branch-aware/native controls
  are supplied by callers; no file I/O, legacy model loading, or thresholds
parity fixture: tests/representation/training/test_metrics.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/data.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8 plus registered data hashes
legacy source path: src/revisit_vlm/tgvf_v3_stage1.py;
  scripts/train_tgvf_v3_stage1.py; docs/TGVF_V3_STAGE1_TRAINING.md
legacy source SHA256: 78b465ec67d40c6d60863715c43171f373483b20c11447b13b56b6fe8e28384a;
  07428d9214e5662ee0477a037a3bee41e93fbf3caf5c90af395bd603d925dc26;
  3a2087a647a46825871ce6346703a35a4608aa31a841be865840ab82ad61d572
symbols/lineage used: focus-row predicate, relative image resolution,
  target/short-answer leakage warning semantics, and retained metadata
semantic differences: strict typed focus metadata, source SHA validation,
  complete accepted/excluded/duplicate/leakage manifests, exact resolved-path
  split audit, and fail-closed duplicate handling; no row is silently filtered
  to repair a split overlap
parity fixture: tests/representation/training/test_data.py; exact legacy-data
  aggregate audit and manifest identities recorded above
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/native_pipeline.py;
  src/tgvf_rl/representation/training/runtime.py
role: adapted extraction plus new native-protocol implementation
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8 plus registered working-file identities
legacy source path: src/revisit_vlm/tgvf_foveal.py;
  src/revisit_vlm/qwen3_vl_tgvf.py; tests/test_tgvf_v3_stage1.py
legacy source SHA256: f2244980599510c976a20dbbe227523fde5af72f26e8253188e04c072456853f;
  09401be77bc0a13fd48eb04681b8cfd00cbd2e5f33b59efbfe825e10ab163801;
  f9010aa7d37143a4d8d98d3d0559868eab8018b62ad74287b0e9d08763bc0dcf
symbols/lineage used: selected TGVF construction, frozen-Qwen vision/merger
  capture geometry, and representation readout intent only
semantic differences: Qwen native tool schema/transcript, strict raw target
  span, no tokenizer growth, explicit hashed prompt, source plus latent visual
  blocks, provider-selected native Hq, and no legacy protocol/token rows
parity fixture: tests/representation/training/test_native_pipeline.py;
  tests/representation/training/test_runtime.py;
  tests/representation/training/test_qwen3_representation_golden.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/streaming.py;
  src/tgvf_rl/representation/training/trainer.py
role: adapted extraction
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: src/revisit_vlm/tgvf_training.py;
  revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py;
  scripts/train_tgvf_v3_stage1.py
legacy source SHA256: ebae9266cafc4f83685f6a7d46a5b0fc501111f3ba9ee257c9e530e636857dd4;
  c54a2cbaee34d2f0d1a1e7a134eca666242c7bef263e0f083c43634d254dff30;
  07428d9214e5662ee0477a037a3bee41e93fbf3caf5c90af395bd603d925dc26
symbols/lineage used: Matrix-CE score gradient, evidence-token L_gen,
  same-image optimizer loop, clipping, and metric reductions
semantic differences: memory-bounded cell recomputation, post-D source-key
  blocking, explicit global numerator/count normalization across accumulation
  and data-parallel ranks, and strict frozen-Qwen/Adapter-only ownership
parity fixture: tests/representation/training/test_streaming.py;
  tests/representation/training/test_trainer.py
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST

new path: src/tgvf_rl/representation/training/checkpoint.py;
  src/tgvf_rl/representation/training/distributed_checkpoint.py;
  src/tgvf_rl/representation/training/fsdp2.py;
  src/tgvf_rl/representation/training/config.py
role: specification-informed new implementation
legacy repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
legacy frozen commit/tag: a200437123afe6fbb481a6c9cf9b7ddf61ff36b8
legacy source path: revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py;
  revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py;
  revisit_vlm_clean/src/revisit_vlm_clean/defaults.py
legacy source SHA256: c54a2cbaee34d2f0d1a1e7a134eca666242c7bef263e0f083c43634d254dff30;
  cac206c3620f417d81cf6a3af9c62c85b63db53bb27b2b13ab2629dc0cd4c430;
  a1f7c070cff26a03a9374065fc544aa9a86a89195295c7c6d3f1517585ea0901
symbols/lineage used: optimizer/scheduler/sampler checkpoint intent,
  resolved run identity, and representation execution-state inventory
semantic differences: Adapter-only deployable artifacts, optimizer-boundary
  strict resume with RNG and complete identities, composable-FSDP2 ownership
  that excludes borrowed Qwen mergers, distributed checkpoint schema, and a
  strict TOML contract with no production scientific defaults
parity fixture: tests/representation/training/test_checkpoint.py;
  tests/representation/training/test_fsdp2.py;
  tests/representation/training/test_config.py; real two-rank representation
  optimizer/checkpoint execution remains a production gate
reviewed by: Codex RPI-20260719
date: 2026-07-19 JST
```
