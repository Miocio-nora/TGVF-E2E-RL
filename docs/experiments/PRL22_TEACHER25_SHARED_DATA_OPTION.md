# PRL22: 25% Teacher Data Option

Status: implementation in progress

## Purpose

PRL22 adds one optional policy-RL dataset condition while preserving the
accepted tool-specific baselines.  The data implementation is shared once;
TGVF, Crop+TGVF, and Crop remain separate experiment projects with independent
run identities, outputs, checkpoints, and evaluation records.

The controlled variable is the prompt population only.  A PRL22 experiment
must not silently change the base model, RP67 artifact, Adapter freeze mode,
prompt/tool dialect, reward, rollout sampling, optimizer, batch size, or policy
update path inherited from its corresponding accepted baseline.

## Immutable parents

The existing policy population is:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/
  T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/
```

It contains 77,541 retained prompts from VStar, ArxivQA, and ThinkLite.  Its
fixed identities are:

```text
manifest_file_sha256 = 752ebe9ea5fced48773b9bc0babfbb6bc57a335dd1b580455f6962053d29fddf
content_sha256       = 5ab99622a2698a7c52c45795215fa5c467b741c103827a1a7dbe3800ff052934
samples_sha256       = 06e5b1b9039680111df5ef01f7f969b9cf3d8d0eaefa5774fd8d16169428611a
```

The supplementary population is:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_selection/teacher/
  tgvf-v4-rp66-rp67-train-t1-retained-v1/
```

It contains 24,779 T1-retained prompts from ChartQA, DocVQA, TextOCR,
TextVQA, and Visual Genome.  Its fixed identities are:

```text
manifest_file_sha256 = 254e09db547047e3e0788a85e56dc458f2437d87c4a6452771131ad6719a276b
content_sha256       = 796f580440662a43b732a8fc33e6380a4ea0c8c240e9023e24954de558607192
samples_sha256       = cf9b86cf11f1ca42c83eb94f1e3b418a193ce0cb1a5d84ec2c8bb5c2e6532880
```

## Mixture contract

The selected policy is an exact 3:1 prompt-group mixture over the same
20,480-row, 80-by-256 training horizon as the accepted PRL13 schedule:

- 15,360 existing prompts;
- 5,120 teacher prompts;
- 20,480 prompts total;
- no replacement within the materialized schedule;
- deterministic seed 42 ordering;
- repeating parent-role order `old, old, old, teacher`.

Consequently every aligned BS16 optimizer batch contains exactly 12 existing
and four teacher prompts, while every aligned BS256 batch contains exactly 192
existing and 64 teacher prompts.  The same artifact therefore supports both
the current small pilot and the 80-step BS256 scale point without changing the
data definition.  Within every 256-prompt macro batch the old-source counts are
90 VStar, 58 ArxivQA, and 44 ThinkLite, while 64 prompts come from the teacher
parent.  This is the 75%-scaled form of the accepted 120/77/59 PRL13 source
mixture, rather than a new natural-pool distribution.

Teacher rows retain `data_source = "teacher"` and their original
`source_dataset` provenance.  They must not masquerade as ArxivQA or another
legacy source.  Their explicit task kind controls MCQ versus open-answer
verification.  They are visual tasks without VStar ground-truth-region
semantics.

## Independent experiment projects

All three projects descend from the same PRL22 shared implementation commit
and bind the same immutable mixture artifact:

1. RP67 T-free Frozen TGVF;
2. RP67 T-free Frozen Atomic Crop+TGVF;
3. Crop T-free.

PRL22-B is the isolated Atomic Crop+TGVF arm.  Its launch-ready 8-step
configuration is:

```text
configs/policy/runs/
  prl_22_b_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_crop_tgvf_teacher25_8step_ws8.toml
```

It is a one-variable derivative of PRL20-R0: only the immutable teacher25
dataset binding, run identity, code binding, and output paths differ.  The
prompt/tool protocol, RP67 frozen Adapter, T-free reward, optimizer, world8
BS16 x n16 sampling, and capacity settings are unchanged.  Step 8 is the base
pilot boundary.  A later 8-to-16 continuation must use an experiment-owned
horizon-extension config and retain step 8 and step 16 as durable checkpoints;
it must not mutate this 8-step base config.

Only the TGVF project is scheduled for the first launch.  It starts from step
0 because changing the dataset changes the training identity.  The initial
pilot is BS16 x 16 rollouts for 16 optimizer steps, with durable checkpoints
at steps 8 and 16 and matched CoreDev-2511 evaluation after training.

The native Crop entry point accepts an experiment-owned overlay explicitly:

```bash
python tools/launch_prl21_crop_tfree16.py \
  --run-config /absolute/path/to/prl22-crop-teacher25.toml \
  --mode preflight
```

Omitting `--run-config` still selects the accepted PRL21 overlay byte-for-byte.
This keeps historical commands reproducible while allowing the isolated PRL22
Crop child project to bind its own data identity, run ID, and output root.

## Interpretation boundary

The teacher supplement is Stage1-train in-distribution for RP67.  It is useful
for testing whether policy RL benefits from questions aligned with the learned
representation, but improvements must still be established on held-out
external evaluation.  The parent populations share 587 exact image hashes but
no exact image-plus-question task; this overlap is reported in the output
manifest rather than hidden.
