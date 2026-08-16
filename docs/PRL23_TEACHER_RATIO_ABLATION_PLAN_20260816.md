# PRL23 Teacher-Ratio Policy-RL Ablation Plan

Date: 2026-08-16

Status: immutable data and launch identities bound; ready for tmux execution

## 1. Question

PRL22 showed that adding 25% Stage1-teacher data was strongly positive for the
RP67 T-free Frozen TGVF policy pilot. PRL23 asks whether that gain continues as
the teacher fraction rises to 50% and 100%, or whether removing the broader
legacy-policy distribution eventually hurts external generalization.

This is a one-variable ablation. The only scientific variable is the teacher
percentage in the policy prompt schedule. Teacher25 from PRL22-A is the accepted
control; PRL23-A and PRL23-B add Teacher50 and Teacher100 respectively.

## 2. Data contracts

All schedules contain 20,480 unique rows, use seed 42, disable dataloader
shuffle, and are materialized without replacement. The 16-step pilot consumes
the first 256-row macro, but the complete 80-by-256 schedule is retained so the
same data identity can scale to a later 80-step run.

| Arm | Existing rows | Teacher rows | Every BS16 | Role cadence |
|---|---:|---:|---:|---|
| PRL22-A Teacher25 control | 15,360 | 5,120 | 12 existing + 4 teacher | `old, old, old, teacher` |
| PRL23-A Teacher50 | 10,240 | 10,240 | 8 existing + 8 teacher | deterministic 1:1 cadence |
| PRL23-B Teacher100 | 0 | 20,480 | 16 teacher | all teacher |

The generic data kind is `policy_t1_teacher_ratio_mix`. Each PRL23 run config
binds `teacher_percentage` explicitly; neither the runtime nor the audit code
infers it from a directory or run name. Teacher selection uses deterministic
source-stratified hash prefixes, making the selected teacher populations nested
across 25%, 50%, and 100% wherever the source quota grows.

The exact Teacher50 existing-source totals are 4,800 VStar, 3,080 ArxivQA,
and 2,360 ThinkLite. Even-numbered 256-row macros contain 60/39/29 existing
rows plus 128 teacher rows; odd-numbered macros contain 60/38/30 plus 128.
Every BS16 follows `[base, teacher] x 8`. Its teacher-source totals are 909
ChartQA, 1,728 DocVQA, 1,633 TextOCR, 1,680 TextVQA, and 4,290 Visual Genome.

Teacher100 has no existing rows. It means that 100% of the fixed 20,480-row
schedule is teacher data: 20,480 rows are selected without replacement from
the 24,779-row teacher parent (82.65% of that parent), rather than consuming
all 24,779 parent rows. Its teacher-source totals are 1,818 ChartQA, 3,456
DocVQA, 3,266 TextOCR, 3,360 TextVQA, and 8,580 Visual Genome; every
BS16 and every 256-row macro are entirely teacher data. The selector retains
the PRL22 hash namespace, and materialization verifies the strict selected-set
nesting `Teacher25 subset Teacher50 subset Teacher100`, with successive set
differences of 5,120 and 10,240 rows.

Immutable artifact roots are:

```text
artifacts/data/policy_rl/PRL23-TEACHER50-MIXED-SCHEDULE-v1
artifacts/data/policy_rl/PRL23-TEACHER100-MIXED-SCHEDULE-v1
```

The materialized identities are:

| Arm | Manifest | Content | Samples | Iteration |
|---|---|---|---|---|
| Teacher50 | `0addf4741080dc922fbff84b85c01c4d2fd19f3669f84a7752e5524a8972cd97` | `2a80ee2d3cfb293e4eec5711964e9ab70dfc4620240f1e78fd1b5cc70a8a368a` | `33c77cdbc44a9f4a23bef2c1c885e32cb81c54f916c4e26bf9cb805987c65ae4` | `b9aa3b2187fd462cf86bb76e95a1a89c9c84f24eab818c18db706a9696e0a600` |
| Teacher100 | `5d1ba2ce7811cbefa05d2a66d54c13f43fbd39efda9326b3ba1ac1995d49b5bf` | `89bc5d5b15558db9b9bba1c80adfcc9404cb0fdb4036276604a4405d66f49a5c` | `3a624fe54e1922ac9695b9d1ca2eb088608cbdf537b7a6adc3bc6acaf06efee5` | `426fa00fd2fab63b98a3dc7bf168fa98195248395bd8f650761aa6a5e9675fc9` |

`tools/bind_prl23_teacher_ratio_launchers.py` validated each manifest and
samples file, wrote these identities plus shared code commit `6567677` into
the run configs, and refreshed each paired evaluation plan's policy-config
hash. No zero-valued launch blocker remains.

## 3. Fixed experimental contract

Relative to PRL22-A, the following are held exactly fixed:

- model: Qwen3-VL-8B-Instruct with native DeepStack and 1,003,520 max pixels;
- representation: RP67 step 2000, contextual hidden-state conditioning,
  frozen Adapter;
- tool protocol: TGVF-only prompt and `tgvf_focus_tool`, at most six calls;
- sampling: 16 rollouts per prompt, temperature 1, common master seed 42,
  maximum response length 20,480;
- reward: rule-first plus OpenRouter/DeepInfra Qwen2.5-72B text-only semantic
  fallback, answer reward, protocol penalty, and repeated-call penalty;
  T-free with focus and grounding rewards disabled;
- optimizer: AdamW, constant LR `1e-6`, weight decay `0.01`, grad norm `1.0`;
- update geometry: BS16 x n16, world size 8, micro batch 2 per rank, GA1;
- full-model Qwen policy update with FSDP2; RP67 remains frozen;
- checkpoints: durable Steps 8 and 16, with Step 8 used as the bound continuation
  boundary;
- evaluation: CoreDev-2511 official visible view, 2,240 supported single-image
  tasks plus 271 held multi-image tasks, Step 8 and Step 16 in parallel on four
  GPUs each, temperature 1, seed 42, common random numbers, and the pinned local
  Qwen2.5-72B benchmark judge.

The static test `tests/framework/test_prl23_teacher_ratio_launchers.py`
normalizes away only code, dataset, run/output, and evaluation identities and
requires every other run and evaluation field to equal PRL22-A.

## 4. Run identities

| Item | PRL23-A Teacher50 | PRL23-B Teacher100 |
|---|---|---|
| Run ID | `PRL-23-A-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-TEACHER50-8STEP-WS8` | `PRL-23-B-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-TEACHER100-8STEP-WS8` |
| W&B ID | `prl23at50` | `prl23bt100` |
| Output | `artifacts/policy/PRL-23-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher50-8step-ws8` | `artifacts/policy/PRL-23-B-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher100-8step-ws8` |
| Evaluation | `PRL23-A-FROZEN-RP67-TFREE-TEACHER50-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `PRL23-B-FROZEN-RP67-TFREE-TEACHER100-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` |
| tmux | `prl23_a_tgvf_teacher50` | `prl23_b_after_a` relay, then Teacher100 supervisor |

## 5. Automatic execution order

The total entry point is:

```bash
tools/launch_prl23_teacher_ratio_ablation_tmux.sh
```

It starts two detached tmux sessions. The first performs Teacher50 Step 0→8,
binds the exact Step-8 continuation, performs Step 8→16, and then runs the
paired Step-8/Step-16 CoreDev evaluation. The second session waits for the
canonical Teacher50 evaluation receipt, verifies the paired-summary SHA-256,
waits for the source process to exit and for two consecutive GPU-idle polls,
and only then starts the complete Teacher100 train-and-evaluate pipeline.

An absent or dead source process is not treated as success. Operational
interruptions are resumed from the committed checkpoint, up to eight training
attempts and six evaluation attempts. The following stop rather than blindly
retry: identity/schema/hash mismatches, incomplete distributed checkpoints,
non-finite values, OOM, frozen-Adapter drift, and API authentication, billing,
or model-availability failures. Sample-local evaluation failures remain
contained by the existing evaluator. `metrics.jsonl` is telemetry rather than
a recovery authority: duplicate or late-flushed rows cannot reject a valid,
identity-bound checkpoint receipt.

## 6. Time and resource estimate

PRL22-A measured 87.3 minutes for Step 0→8, 79.6 minutes for Step 8→16, and
about 50 minutes for paired CoreDev evaluation. Including process startup and
checkpoint boundaries, one PRL23 arm is expected to require 3.75–4 hours on
all eight B200 GPUs. The two controlled arms therefore require about 7.5–8
hours after binding and smoke validation. Initial shared-code integration,
artifact binding, and a low-cost functional smoke add roughly 1–2 hours.

The arms run serially because splitting them into two world-size-4 jobs would
change the accepted world-size-8 update geometry. Each completed arm is
expected to occupy approximately 320 GB before optional post-verification
cleanup; both fit within the currently available storage.

## 7. Current scope boundary

The two formal launch configs in this plan are pure-TGVF arms only. The shared
ratio artifact and veRL renderer are profile-driven, but plain native Crop and
Atomic Crop+TGVF still require their own matched configs and compose tests
before they can be called formal-ready at Teacher50/100. This document does
not claim that those additional arms have already been deployed.

## 8. Decision rule

The primary comparison is Macro* under the unchanged CoreDev measurement
contract at Steps 8 and 16, reported together with all seven components. The
ratio is considered promising when the gain is positive at both horizons or
when Step 16 clearly improves without a concentrated collapse on a major
component. A single temperature-1 run remains pilot evidence, not a claim of
statistical significance. Teacher100 is especially diagnostic: a regression
there would show that teacher alignment is useful as supplementation rather
than as a complete replacement for the diverse policy population.
