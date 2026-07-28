# Policy RL Future Direction and Decision Report

Date: 2026-07-28 (Asia/Tokyo)

Status: direction fixed for implementation after the current rebuttal work

## 1. Purpose

This report freezes the conclusions reached after reviewing the failed current
Crop/TGVF Policy RL pilots and the successful legacy Stage3 GRPO experiment.
The team will be occupied with another project's rebuttal for the next several
days. No new formal RL run should be started merely to fill that interval. The
next implementation should follow the staged plan and acceptance gates below.

The current project direction is no longer to continue the Thinking-model
pilot unchanged. The primary policy, reference, and data-selection target is
Qwen3-VL-8B-Instruct. Data selection is part of the RL method rather than an
incidental preprocessing step.

## 2. Fixed high-level direction

There are two distinct experimental lines, in this order:

1. **Reproduce Crop RL.** Rerun the Crop arm after the coordinate/cropping bug
   fix and establish a valid Crop baseline. The previous Crop result is not
   scientific evidence about RL quality because most tool calls were corrupted
   by the environment bug.
2. **Improve TGVF reward.** Once the common Instruct/T1 data and optimization
   gates are stable, implement and test a better TGVF reward. This is not part
   of the first Crop reproduction.

The shared foundation for both lines is:

- Qwen3-VL-8B-Instruct rather than the previous Thinking checkpoint;
- policy-dependent data selection using eight rollouts per candidate;
- T1 as the initial useful-difficulty band: retain candidates with between one
  and seven correct rollouts out of eight, excluding both all-wrong and
  all-correct groups;
- language-decoder LoRA for the first controlled comparison;
- exact rollout/replay and resumable data/optimizer state;
- matched evaluation and immutable run identities.

## 3. What the failed pilots do and do not show

### 3.1 TGVF pilot

The prior Thinking-model TGVF pilot used 16 prompt groups per optimizer step,
eight trajectories per prompt, and 80 optimizer steps. Its on-policy answer
accuracy was 60.84%, tool attempt rate was 94.19%, and the conditional tool
bonus rate was 55.71%. Nevertheless, all seven held-out benchmark slices were
below the pre-RL baseline.

This rules out a simple explanation that training failed only because the
rollouts never answered correctly or never called the tool. More plausible
causes are reward-policy mismatch, aggressive optimization, insufficient
effective independent groups, excessive tool pressure, and TGVF hallucination
after observing the tool result.

About 30.7% of its prompt groups had identical rewards and therefore zero GRPO
advantage. Although there were 128 trajectories per update, the effective unit
of independent evidence was only about 11.1 nonzero-advantage prompt groups per
update. Trajectories within one group are correlated and must not be counted as
independent batch elements when judging optimizer stability.

### 3.2 Crop pilot

The previous Crop arm is invalid as an RL comparison. Among 4,504 parseable
crop calls, approximately 70.5% were clamped under the wrong coordinate
interpretation, 39.1% produced an empty crop, and about 2,500 calls failed. The
crop implementation has since been repaired, but the formal Crop experiment
has not yet been rerun. The immediate Crop goal is therefore reproduction, not
reward innovation.

### 3.3 Batch size

Small effective prompt batch can amplify noisy updates, but it cannot by itself
explain the difference between the failed current pilot and the successful
legacy Stage3 run. The legacy run used only eight global prompt groups and 64
rollouts per optimizer step, smaller than the failed pilot's 16 groups and 128
rollouts. Batch size remains a stability variable, especially together with a
large learning rate, but is not the sole failure cause.

### 3.4 Resolution and LoRA/full tuning

The legacy successful run used a 512 image-resolution cap and adapter-only
policy updates. The failed current pilots used a comparable 512-by-512 maximum
pixel budget and decoder LoRA. Therefore neither the resolution cap nor
LoRA-versus-full tuning is the leading explanation. Full tuning should remain
a later named experiment rather than a repair for an unverified reward/data
pipeline.

## 4. Legacy successful Stage3 evidence

The reviewed legacy run is:

```text
repository: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
run: outputs/stage3_grpo/
  final_datasetprop_optionalreward_no_block_frozenref_rewardgate_
  softprompt4_4gpu_g8_pb2_res512_200step_20260630_112809
runtime commit: 723bb850cfb04d20d9071518bdb6c595e34ed586
```

It started from a Qwen3-VL-8B-Thinking Stage2 checkpoint and completed 200
stepwise GRPO updates on four GPUs:

| Field | Legacy Stage3 setting |
|---|---:|
| Per-device prompt batch | 2 |
| Global prompt groups/update | 8 |
| Rollouts/prompt | 8 |
| Rollouts/update | 64 |
| Unique scheduled prompts | 1,600 |
| Total rollouts | 12,800 |
| Optimizer | manual SGD |
| Learning rate | `1e-6` |
| GRPO clip | 0.2 |
| Configured KL coefficient | 0.02 |
| Image resolution | 512 |
| Maximum tool calls | 1 |
| Policy scope | adapter/protocol-token parameters; base and TGVF frozen |

The data consisted of 20,000 SFT-image-disjoint prompts from Visual Genome,
TextVQA, DocVQA, and ChartQA. GPT-5.4 teacher triage supplied difficulty,
target, and tool-need hints. The final schedule sampled 1,600 unique prompts in
the dataset's tool-hint proportions.

The configured `frozen_stage2` reference and KL coefficient require an
important implementation caveat. Each one-step process loaded the preceding
Stage3 checkpoint and captured a new reference immediately before its single
update. Logged KL was zero throughout. This old job should be interpreted as a
sequence of single-step, group-normalized policy-gradient updates, not as
evidence that a globally frozen Stage2 KL penalty was effective.

### 4.1 Measured Stage2-to-Stage3 improvement

On the same CoreDev-2511 manifest and evaluation settings:

| Mode | Stage2 | Stage3 step 200 | Delta |
|---|---:|---:|---:|
| Free overall | 35.53% | 36.40% | +0.87 pp |
| Free macro | 39.67% | 40.57% | +0.90 pp |
| Softforce overall | 35.98% | 36.06% | +0.08 pp |
| Softforce macro | 39.90% | 40.26% | +0.36 pp |

The Free result improved all seven constituent benchmark slices over Stage2
Free. The gain is modest but credible and establishes that the legacy Stage3
did improve its Stage2 initialization.

## 5. Legacy reward audit

The legacy scalar reward was:

```text
R_total =
    2.0 * R_answer
  + 1.0 * R_tool_decision
  + 1.0 * R_focus_evidence
  + 1.0 * R_grounded_reasoning
  + R_protocol
```

The following table is computed from all 12,800 recorded rollout rewards. A
component's average contribution is not its GRPO influence by itself; the
second column measures whether it varied inside a prompt's eight-rollout group
and could therefore affect the group advantage.

| Component | Mean contribution | Groups with within-group variation | Main effect |
|---|---:|---:|---|
| Weighted answer | +0.782 | 41.5% | Task correctness |
| Tool decision | -0.125 | 67.8% | Call/no-call policy |
| Focus judge | +0.244 | 70.9% | Dense target-quality shaping |
| Grounding judge | +0.247 | 72.0% | Post-tool grounding/hallucination |
| Protocol penalty | -0.0058 | 4.3% | Structural safety only |

Answer plus tool decision produced variation in 74.7% of groups. The full
reward produced variation in 77.8%, leaving 22.2% zero-advantage groups. The
judges mainly reranked trajectories inside already-active groups; they only
made about 50 additional groups nonconstant.

### 5.1 Answer reward and required-tool gate

If a sample was labeled `tool_needed` and the rollout did not call the tool,
the answer reward was set to zero even when the final answer was correct.

- There were 4,405 `tool_needed + no tool` rollouts.
- 53.46% of them nevertheless guessed the answer correctly.
- Their mean total reward was `-1.01` because the correct answer could not
  bypass the required-tool decision.
- `tool_needed + used tool` rollouts had mean total reward `+2.74`.

This anti-shortcut gate is the strongest useful idea in the old reward. It is
also dangerous when the required-tool label is wrong, so current labels should
come from measured Instruct direct-versus-tool utility where possible, not an
unvalidated static teacher hint.

### 5.2 Tool-decision reward

All 12,800 actual labels came from teacher hints with weight 0.5; the configured
forced OFF/ON probe was not the source of the labels used by training.

The effective rewards were approximately:

```text
tool needed:       call +0.5, no call -1.0
tool optional:     call +0.25, no call 0
tool unnecessary:  no call +0.5, call about -0.3
```

This component supplied useful group variance, but its static label quality was
not independently established.

### 5.3 Focus judge

Among tool-using rollouts, 36.9% received `+1`, 61.0% received `+0.5`, and only
2.1% received zero. There was no negative focus reward. Thus 97.9% of tool
calls received a positive focus score. This was dense shaping, but it was too
close to an unconditional tool-use bonus and was weakly calibrated for bad
targets.

### 5.4 Grounding judge

Among tool-using rollouts, 82.7% received `+1`, 2.0% received `+0.5`, and 15.3%
received `-1`. This was the only legacy component that directly punished
reasoning unsupported by the tool observation. Its semantics are worth
carrying forward, although the judge was still relatively saturated.

### 5.5 Protocol penalty

Only 74 of 12,800 rollouts, 0.58%, received the `-1` protocol penalty. It was a
necessary defense against malformed tags, missing/generic targets, excessive
calls, loops, and parsing failures, but not a material source of task-learning
signal.

### 5.6 Reward defect: residual over-incentive to call tools

The combined mean reward was `0.223` without a tool call and `2.765` with a
tool call. More importantly, for `tool_unnecessary` samples:

| Behavior | Mean total reward |
|---|---:|
| Correctly do not call | 2.140 |
| Incorrectly call | 2.700 |

The focus and grounding judges supplied about `+1.36` per tool call on average
and overwhelmed the unnecessary-call penalty. Therefore the old reward should
not be copied numerically. Its useful structure was the required-tool answer
gate and negative groundedness judgment; its main defect was positive judge
shaping that implicitly rewarded calling the tool.

## 6. Learning-rate decision

The current `1e-5` learning rate was an internal Policy Pilot decision. It was
not inherited from DeepEyes, the successful legacy Stage3 run, or an official
veRL Qwen3-VL GRPO default. EasyR1 was reviewed for decoder LoRA target-module
selection, not as provenance for this learning rate.

Relevant comparable settings are:

| Reference | Learning rate |
|---|---:|
| Current failed Pilot | `1e-5` |
| Official veRL Qwen3-VL-8B GRPO example | `1e-6` |
| Official veRL Qwen3/Qwen2.5-VL LoRA tuning examples | `3e-6` |
| Successful legacy Stage3 | `1e-6` |

The combination of only 16 independent prompt groups, about 11 effective
nonzero groups, discrete high-variance rewards, and AdamW at `1e-5` is an
unnecessarily aggressive default. Future runs fix:

- `1e-6` as the primary formal baseline;
- `3e-6` as the higher-LR comparison arm;
- `1e-5` as a diagnostic high-LR arm only, not the default.

Before a full run, use the same selected data and reward to compare the three
rates for 5--10 optimizer steps. Compare update norm, policy/reference KL,
reward and advantage distribution, answer/tool behavior, reasoning health,
and held-out accuracy. Promote a rate only after this gate.

## 7. Reasoning-length decision

The legacy maximum generation length of 256 tokens must not be transferred to
the current Instruct line. Prior SFT experiments showed that training toward
short outputs can damage the base model's reasoning behavior. The current
project should preserve a sufficiently long response budget.

Length control should be diagnostic and structural rather than a reward for
shortness:

- measure generated-token distributions and truncation rate;
- detect loops and pathological repetition;
- preserve successful long reasoning trajectories;
- do not add a generic brevity reward;
- treat any future lower response cap as a separately evaluated ablation with
  explicit reasoning-health gates.

## 8. Reward direction for the current TGVF line

The old implementation is not the implementation base. The current exact
trajectory, verifier, replay, and checkpoint framework remains authoritative.
The legacy run contributes reward semantics only.

The target structure is:

```text
R = answer correctness
  + required-tool anti-shortcut gate
  + measured tool utility
  + zero-centered focus quality
  + zero-centered groundedness
  + protocol validity
  - unnecessary-call cost
```

Requirements:

1. **Answer remains primary.** Correctness must remain the largest interpretable
   outcome signal.
2. **Required-tool gate.** A candidate labeled genuinely tool-required cannot
   receive full answer credit by guessing without the tool.
3. **Outcome-based tool labels.** Prefer direct-versus-tool rollout utility from
   the current Instruct policy. Keep required, optional, unnecessary, and
   uncertain labels separate.
4. **Zero-centered focus reward.** A bad or generic target must be negative;
   an acceptable target should be near zero; only clearly useful targeting is
   positive. Tool use alone earns nothing.
5. **Grounding can be negative.** Unsupported OCR/visual claims, contradictions
   with the observation, and post-tool hallucinations must receive negative
   reward. Positive groundedness should be gated by, or at least calibrated
   against, final answer correctness.
6. **No tool-bonus leakage.** On tool-unnecessary examples, the maximum reward
   for an unnecessary call must be lower than the reward for a correct direct
   response. This is a pre-training counterfactual unit gate.
7. **Protocol is a boundary condition.** Keep deterministic malformed-action,
   missing-target, loop, and call-cap penalties, but do not expect them to
   supply the main learning signal.
8. **Bound and audit the scale.** Report every component, group variance,
   zero-advantage fraction, and reward by tool-need label. Do not tune only on
   mean total reward.

## 9. Implementation plan after rebuttal

### Phase A: finish and audit Instruct T1 data

- Complete resumable eight-rollout data selection.
- Materialize the accepted T1 manifest with model, prompt, seed, generation,
  scorer, and code identities.
- Confirm exclusion/leakage policy against training and held-out evaluation.
- Report source, answer type, difficulty, direct accuracy, and tool-need
  distributions.
- Audit random accepted, all-wrong, and all-correct samples before promotion.

Exit gate: immutable T1 manifest, reproducible selection counts, no unresolved
data corruption, and an approved difficulty/source distribution.

### Phase B: reproduce Crop RL

- Use the repaired crop coordinate implementation.
- Run deterministic unit and real-model crop observation gates first.
- Use Qwen3-VL-8B-Instruct and the fixed T1 data identity.
- Keep the first reward minimal and Crop-specific; do not introduce the new
  TGVF focus/grounding judge in this reproduction.
- Run the `1e-6` versus `3e-6` short LR gate.
- Promote one configuration to a formal matched Crop run.

Exit gate: crop success/empty/error telemetry proves valid observations, no
coordinate interpretation failures, stable optimization, and a matched
held-out comparison against the untrained Instruct baseline.

### Phase C: implement improved TGVF reward

- Implement decomposed component logging in the current reward framework.
- Build outcome-based tool-utility labels from current-policy rollouts.
- Calibrate focus and grounding scorers on a small human-audited set.
- Add counterfactual tests proving that unnecessary tool use cannot increase
  reward merely by activating judge terms.
- Run CPU/unit reward fixtures, one-step real rollout, then the 5--10-step LR
  gate.
- Launch a formal TGVF run only after reward and optimization gates pass.

Exit gate: correct reward ordering on required/optional/unnecessary fixtures,
acceptable judge calibration, bounded zero-advantage rate, preserved reasoning
health, and held-out improvement over the same Instruct baseline.

## 10. Required reporting for future formal runs

Every formal Crop or TGVF run should report at least:

- natural answer accuracy before reward gating;
- gated answer-reward rate;
- tool attempt, success, empty/error, and unnecessary-call rates;
- reward mean and distribution for every component;
- within-group component variance and zero-advantage fraction;
- effective nonzero prompt groups per update;
- optimizer LR, update norm, gradient norm, clipping, and current/reference KL;
- generated-token distribution, truncation, repetition, and reasoning-health
  diagnostics;
- metrics split by data source, T1 pass count, answer type, and tool-need label;
- matched held-out results versus the exact pre-RL Instruct checkpoint.

## 11. Decisions intentionally deferred

- Full-parameter policy training versus decoder LoRA after a valid LoRA proof.
- Jointly updating the TGVF Adapter during Policy RL.
- A lower generation cap or direct length reward.
- A larger production prompt batch beyond the minimum stability gate.
- Online judge inference in the optimizer loop.
- SDPO or a GRPO/SDPO hybrid.

These are later named experiments. They should not be mixed into the first
valid Crop reproduction or the first improved-reward TGVF comparison.
