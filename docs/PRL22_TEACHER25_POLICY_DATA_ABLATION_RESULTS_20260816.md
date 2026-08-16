# PRL22 Teacher25 Policy-Data Ablation Results

Date: 2026-08-16
Status: complete; strongly positive pilot evidence

## Executive conclusion

Adding 25% retained Stage1 teacher data is **very positive in the current
small-batch policy-RL regime**.  Under the matched CoreDev-2511 protocol, the
Teacher25 schedule improves the same-step aggregate score in both tested
visual-tool lines:

| Tool line | Step 8, no Teacher | Step 8, Teacher25 | Delta | Step 16, no Teacher | Step 16, Teacher25 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| RP67 T-free Frozen TGVF | 56.1964 | **58.4655** | **+2.2691** | 58.1996 | **59.9590** | **+1.7595** |
| RP67 T-free Frozen Crop+TGVF | 62.1168 | **62.3719** | **+0.2550** | 60.9539 | **62.4974** | **+1.5434** |

The strongest result is pure TGVF, where Teacher25 is positive at both
endpoints and on six of the seven Macro* components at each endpoint.  In
Crop+TGVF, the Step-8 aggregate difference is small, but the Step-16 result is
clearly positive.  More importantly, Teacher25 changes the Step-8-to-Step-16
trajectory from a `-1.1629` point regression to a `+0.1255` point change.

The current decision is therefore:

> Teacher25 has passed the small-batch policy-RL pilot in two distinct visual
> tool protocols.  It is a strong default candidate for the next TGVF and
> Crop+TGVF pilots, rather than an optional data variant with no demonstrated
> benefit.

This is strong cross-protocol pilot evidence, not yet a multi-seed statistical
proof.  The remaining qualification matters because training has one seed,
evaluation uses stochastic temperature-1 decoding, and the treatment and
historical controls were not executed from a byte-identical code commit.

## 1. Teacher25 data contract

Teacher25 changes the policy prompt population while preserving the accepted
model, tool, reward, optimizer, rollout and evaluation recipes.

| Item | Value |
|---|---|
| Existing retained parent | 77,541 prompts from VStar, ArxivQA and ThinkLite |
| Teacher retained parent | 24,779 prompts from ChartQA, DocVQA, TextOCR, TextVQA and Visual Genome |
| Materialized schedule | 20,480 prompts, deterministic seed 42, no replacement |
| Existing / Teacher rows | 15,360 / 5,120 |
| Mixture | 75% existing + 25% Teacher |
| Every BS16 group | 12 existing + 4 Teacher |
| Every 256-prompt macro | 90 VStar + 58 ArxivQA + 44 ThinkLite + 64 Teacher |
| Ordered role pattern | `old, old, old, teacher` |

Teacher rows retain `data_source=teacher` and their original source-dataset
provenance.  They are Stage1-train in-distribution for RP67.  The two parent
populations share 587 exact image hashes, but contain no exact
image-plus-question task overlap.

## 2. Matched experiment and measurement contract

### 2.1 Fixed training settings

Both Teacher25 arms inherit their corresponding accepted no-Teacher control:

| Setting | Value |
|---|---|
| Base model | Qwen3-VL-8B-Instruct |
| Policy update | full Qwen policy update |
| Representation | RP67 Step-2000 Adapter |
| Adapter mode | frozen |
| Reward | T-free shaped reward; visual focus/grounding rewards disabled |
| Global prompt batch | BS16 |
| Rollouts per prompt | n16 |
| World size | 8 |
| Micro batch / gradient accumulation | micro2 / GA1 |
| Optimizer / learning rate | AdamW, constant `1e-6` |
| Rollout sampling | temperature 1 |
| Endpoints | optimizer Steps 8 and 16 |

The intended scientific variable is only the immutable Teacher25 prompt
schedule.  PRL17-R2 and PRL20-R0 were executed at historical control commit
`2c1039e`, while the shared Teacher25 implementation is bound to `37b99e2`.
The recipe and measurement protocol are matched, but this commit difference
means the comparison should be described as a matched-recipe A/B rather than a
byte-identical-code causal proof.

### 2.2 CoreDev-2511 contract

| Item | Value |
|---|---|
| Manifest | CoreDev-2511, 2,511 rows |
| Evaluated | 2,240 single-image rows |
| Held out | 271 multi-image rows |
| Sampling | temperature 1, `do_sample`, master seed 42 |
| Pairing | common random numbers per task and turn |
| Judge | local Qwen2.5-72B-Instruct where a judge is required |
| Status | all four Step-8/Step-16 treatment summaries passed |

`Macro*` is the unweighted mean of seven fixed components: VStar, HRBench
Average/all, BLINK single-image-180, OCR English/Chinese mean, MMMU
single-image-269, MathVista and the MathVerse five-version macro.  All scores
below are percentages; deltas are percentage points.

## 3. PRL22-A: pure TGVF

Direct control: PRL17-R2 RP67 T-free Frozen TGVF without Teacher data.

| CoreDev component | No Teacher S8 | Teacher25 S8 | Delta S8 | No Teacher S16 | Teacher25 S16 | Delta S16 |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 65.45 | **71.20** | **+5.76** | 64.92 | **69.11** | **+4.19** |
| HRBench Average/all | 60.00 | **61.00** | **+1.00** | 64.50 | **69.50** | **+5.00** |
| BLINK single-image-180 | 60.56 | **61.11** | **+0.56** | **63.33** | 60.56 | -2.78 |
| OCR English | 44.33 | **46.14** | **+1.80** | 43.83 | **45.83** | **+2.00** |
| OCR Chinese | 34.45 | **41.63** | **+7.19** | **37.95** | 36.23 | -1.72 |
| OCR mean | 39.39 | **43.89** | **+4.49** | 40.89 | **41.03** | **+0.14** |
| MMMU single-image-269 | **47.58** | 45.72 | -1.86 | 47.96 | **50.19** | **+2.23** |
| MathVista | 68.00 | **69.33** | **+1.33** | 70.00 | **73.33** | **+3.33** |
| MathVerse five-version macro | 52.40 | **57.00** | **+4.60** | 55.80 | **56.00** | **+0.20** |
| **Macro*** | 56.1964 | **58.4655** | **+2.2691** | 58.1996 | **59.9590** | **+1.7595** |

The common, untrained frozen-RP67 Step-0 Macro* is `57.0320`; it was not
rerun for Teacher25 because changing the training schedule does not change the
initial model.  Relative to this shared Step 0:

| Pure-TGVF trajectory | Step 8 - Step 0 | Step 16 - Step 0 | Step 16 - Step 8 |
|---|---:|---:|---:|
| No Teacher | -0.8356 | +1.1675 | +2.0032 |
| Teacher25 | **+1.4335** | **+2.9270** | +1.4935 |

This is a strong result rather than a single-benchmark accident: Teacher25 is
positive on six of seven Macro* components at Step 8 and again on six of seven
at Step 16.  It also turns the early Step-8 aggregate from below initialization
to above initialization.

## 4. PRL22-B: Atomic Crop+TGVF

Direct control: PRL20-R0 RP67 T-free Frozen Atomic Crop+TGVF without Teacher
data.

| CoreDev component | No Teacher S8 | Teacher25 S8 | Delta S8 | No Teacher S16 | Teacher25 S16 | Delta S16 |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 70.16 | **71.20** | **+1.05** | 71.73 | **72.77** | **+1.05** |
| HRBench Average/all | **73.00** | 69.00 | -4.00 | 68.00 | **70.00** | **+2.00** |
| BLINK single-image-180 | 65.00 | **67.22** | **+2.22** | 61.67 | **68.33** | **+6.67** |
| OCR English | 52.47 | **52.93** | **+0.45** | 51.53 | **53.05** | **+1.52** |
| OCR Chinese | **54.41** | 52.21 | -2.19 | **53.41** | 51.84 | -1.57 |
| OCR mean | **53.44** | 52.57 | -0.87 | **52.47** | 52.45 | -0.02 |
| MMMU single-image-269 | 47.96 | **51.67** | **+3.72** | **49.81** | 45.72 | -4.09 |
| MathVista | **69.67** | 69.33 | -0.33 | 67.00 | **71.00** | **+4.00** |
| MathVerse five-version macro | 55.60 | 55.60 | 0.00 | 56.00 | **57.20** | **+1.20** |
| **Macro*** | 62.1168 | **62.3719** | **+0.2550** | 60.9539 | **62.4974** | **+1.5434** |

| Crop+TGVF trajectory | Step 16 - Step 8 |
|---|---:|
| No Teacher | -1.1629 |
| Teacher25 | **+0.1255** |

The Step-8 treatment-control gap is too small to interpret by itself.  The
Step-16 result is more meaningful: five of seven Macro* components improve,
and the largest gain is BLINK at `+6.67` points.  Teacher25 also removes the
control's pronounced second-half aggregate regression.

## 5. Updated conclusions

1. **Teacher data is highly promising here.**  The same 25% treatment is
   positive at both pure-TGVF endpoints and at both Crop+TGVF endpoints.  The
   Step-16 gains of `+1.76` and `+1.54` points across different tool protocols
   are difficult to explain as a tool-specific quirk.
2. **The benefit is not only faster fitting.**  Pure TGVF improves early and
   remains ahead at Step 16.  Crop+TGVF shows the complementary effect: the
   treatment mainly stabilizes the later training interval.
3. **The result supports data-model alignment.**  Teacher questions are
   Stage1-train in-distribution for RP67, so their benefit is consistent with
   giving policy RL prompts whose visual-answer structure matches the learned
   representation.  The held-out CoreDev gain shows that the effect is not
   confined to reward on the added training rows.
4. **Teacher25 should become the default candidate for the next pilots.**  A
   no-Teacher arm remains useful as a scientific control, but the burden of
   evidence has shifted: new mainline TGVF and Crop+TGVF pilots should normally
   include Teacher25 unless they are explicitly testing the old data regime.
5. **The data are not uniformly beneficial on every benchmark.**  Pure TGVF
   loses BLINK at Step 16; Crop+TGVF loses MMMU at Step 16; OCR Chinese also
   declines in both Step-16 comparisons.  Teacher25 improves the aggregate and
   training trajectory, but does not solve every capability axis.

## 6. Evidence boundary and next confirmation

The result justifies the wording **strongly positive pilot evidence**.  It does
not yet justify a claim of statistically proven universal effectiveness,
because:

- there is one policy-training seed per condition;
- temperature-1 inference retains sampling variance, despite paired common
  random numbers;
- historical controls and Teacher25 treatments use matched recipes but not a
  byte-identical executable commit;
- the Teacher pool is deliberately aligned with RP67, so the result does not
  by itself establish the same gain for unrelated representations or models.

The cleanest confirmatory action is either a second training seed or a
same-current-commit no-Teacher rerun.  PRL22-C pure Crop would separately test
whether the benefit generalizes when no TGVF representation is available.

## 7. Artifact provenance

### PRL22-A

- Evaluation root:
  `artifacts/policy/PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8/evaluation/PRL22-A-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1`
- `paired-summary.json` SHA256:
  `06b7ccbc4c2d71d9bb338057a62da4b95e8bdf36b35f59eb5de9d3e101efd03e`
- `evaluation-complete` SHA256:
  `c4c49a7c21c970b563f221c863c549cb62b143971633a1db8014cdb184af2b20`

### PRL22-B

- Evaluation root:
  `artifacts/policy/PRL-22-B-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-8step-ws8/evaluation/PRL22-B-R0-FROZEN-RP67-TFREE-CROP-TGVF-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1`
- `paired-summary.json` SHA256:
  `dc07a11bbad97044294f634a188e5038b44a079fa964e34fc2cd63a9c61a5ee5`
- `evaluation-complete` SHA256:
  `4ea1f27251856b588f4ae2f8d52c6e1de9116759adbae0b1b013c52f8de871aa`

PRL22-B evaluation encountered a transient vLLM TCP-port collision and then
exposed a resume-validator namespace bug.  The validator was corrected in
commit `566a1fe`; evaluation resumed from the existing Step-8 and Step-16
materializations.  No training checkpoint was regenerated, and both official
summaries passed.
