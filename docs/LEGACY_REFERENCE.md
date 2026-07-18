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

Because the worktree is dirty, `HEAD` alone does not identify the current
reference behavior. No code may be ported until the user approves an archive
commit/tag or an immutable patch/content-hash bundle.

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
- the entire `revisit_vlm_clean.stage3_grpo` implementation and its CLIs,
  executors, policy/reference replay, rewards, judges, and schedulers;
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
