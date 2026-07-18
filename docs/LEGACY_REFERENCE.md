# Controlled Legacy Reference

Status: **reference identified but not frozen**
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

No representation data path or manifest has been frozen by this document yet.

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
  an explicit projection port
parity fixture: tests/representation/test_adapter.py (synthetic output/gradient);
  exact legacy checkpoint parity remains a later representation gate
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
```
