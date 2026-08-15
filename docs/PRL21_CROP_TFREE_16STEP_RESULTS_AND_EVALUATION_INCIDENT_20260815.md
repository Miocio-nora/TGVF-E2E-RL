# PRL21 Crop T-free 16-step results and evaluation incident

Date: 2026-08-15 (Asia/Tokyo)

## Outcome

PRL21 is a positive Crop RL control. Under the established CoreDev-2511
temperature-1 measurement contract, Macro* reaches `61.1032` at Step 8 and
`61.0862` at Step 16. The major gain is already present by Step 8; Step 16 is
stable and differs by only `-0.0170 pp` from Step 8.

This run does **not** establish that 16 steps are materially better than eight.
The observed Step8-to-Step16 change is smaller than the previously measured
temperature-1 sampling variation, and the component movements are mixed.

## Training contract

- Model: `Qwen3-VL-8B-Instruct`
- Tool: native atomic Crop
- Training: full model; vision encoder, projector, and language model trainable
- Reward profile: `stage3-shaped-v1-tfree`
  - answer reward weight `2.0`
  - repeated-call penalty `0.05`
  - protocol-error penalty `1.0`
  - no target reward (`T-free`)
  - tool utility, focus, grounding, and positive-crop bonuses disabled
- Global prompt batch: `16`
- Rollouts per prompt: `16` (`256` trajectories per optimizer step)
- World size: `8`; actor micro batch per GPU: `32`; GA: `1`
- LR: `1e-6`, constant; KL coefficient: `0`
- Sampling temperature: `1.0`
- Retained checkpoints: Step 8 and Step 16

The immutable training overlay is
`configs/policy/runs/prl_21_r0_qwen3_instruct_full_crop_bs16_n16_tfree_16step_ws8.toml`.

## CoreDev-2511 result

All values are percentages. BLINK and MMMU use only the supported single-image
subsets (`180` and `269` rows). OCR mean averages English and Chinese once.
Macro* is the unweighted mean of VStar, HR, BLINK-single, OCR mean,
MMMU-single, MathVista, and the five-version MathVerse macro.

| benchmark | Step 8 | Step 16 | Step16 - Step8 |
|---|---:|---:|---:|
| VStarBench Overall | 75.39 | 75.92 | +0.52 |
| HRBench all / average | 71.00 | 73.50 | +2.50 |
| BLINK single-image (180) | 62.78 | 62.78 | +0.00 |
| OCRBench v2 English | 49.17 | 49.79 | +0.62 |
| OCRBench v2 Chinese | 53.59 | 57.66 | +4.07 |
| OCR EN/CN mean | 51.38 | 53.72 | +2.34 |
| MMMU-Pro single-image (269) | 46.84 | 45.35 | -1.49 |
| MathVista MINI | 65.33 | 65.33 | +0.00 |
| MathVerse five-version macro | 55.00 | 51.00 | -4.00 |
| **Macro\*** | **61.1032** | **61.0862** | **-0.0170** |

Context from the established measurement ledger:

| arm | Macro* |
|---|---:|
| Original Qwen3-VL-8B-Instruct | 55.36 |
| Crop clean Step 0 | 55.57 |
| prior Crop clean-final Step 8 | 59.72 |
| PRL21 Crop T-free Step 8 | **61.10** |
| PRL21 Crop T-free Step 16 | **61.09** |

The historical Step 0 comparison is not a paired-common-random-numbers run.
Consequently, the approximate `+5.53 pp` (Step 8) and `+5.52 pp` (Step 16)
deltas are strong positive pilot evidence but are not paired causal estimates.

## Training-health interpretation

The external score remains healthy, but Crop use decays late in training:

| step | reward mean | answer accuracy | tool-call rate | crop success rate | response mean tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7180 | 0.5078 | 0.7148 | 0.7070 | 1215.8 |
| 8 | 1.0924 | 0.6133 | 0.7422 | 0.7383 | 722.8 |
| 12 | 0.9439 | 0.5234 | 0.5000 | 0.4570 | 565.7 |
| 16 | 1.2803 | 0.6523 | 0.3008 | 0.2930 | 360.3 |

Thus PRL21 verifies that the T-free answer-centric reward can improve the Crop
policy, but it also exposes an answer-channel shortcut: late optimization can
retain answer accuracy while increasingly bypassing the tool. Step 8 is the
safer default checkpoint; Step 16 is retained as a platform/tail diagnostic.

## Evaluation validity

- Both checkpoints completed inference for all `2,240` supported single-image
  rows; the protocol explicitly held `271` unsupported multi-image rows.
- All 14 checkpoint-by-suite scoring jobs completed.
- Step 8 had zero deterministic judge parse failures.
- Step 16 had one parse failure among 2,511 rows; it was deterministically
  counted incorrect. This is below the frozen acceptance threshold and does not
  change the conclusion.
- Qwen2.5-72B-Instruct was served locally with tensor parallelism 2 and seed 42
  for judge-required suites. GPT fallback was disabled.

Result root:

```text
artifacts/policy/
  PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8/
    evaluation/
      PRL21-R0-CROP-TFREE-COREDEV2511-STEP8-STEP16-TEMP1-SEED42-V1/
```

## Evaluation incident and corrective action

The repeated failures were orchestration failures, not Crop inference failures.
PRL21 copied the old PRL14 experiment-specific supervisor rather than using the
newer manifest-driven paired evaluator. The copied shell retained four stale
assumptions:

1. an undiscoverable legacy VLMEvalKit run ID;
2. a hard-coded PRL13 repository checkout and Instruct/Thinking summary drift;
3. missing `reuse_aux=infer`, which allowed scoring to request inference reuse
   incorrectly;
4. hard-coded judge GPU/seed settings instead of the pinned judge service
   contract.

The recovered scores above use an isolated immutable scoring view, exact
inference reuse, the current scorer, and the pinned GPU2/3 seed-42 judge.

One provenance limitation remains: the materialized snapshots contain the
correct PRL21 checkpoint paths and weight SHA256 values, but the legacy
full-model format records the PRL13 policy contract as the checkpoint owner.
This does not change the executed weights or scores, but it prevents treating
the artifacts as clean owner-bound canonical records.

The permanent fix is to make the unified evaluator support both backends:

- `paired_tgvf` for adapter snapshots;
- `deepeyes_native_full_model` for Crop checkpoints.

Its plan must bind checkpoint owner identity separately from prompt/protocol
identity, freeze `mode=eval`, `reuse=true`, and `reuse_aux=infer`, load the judge
only from its pinned configuration, and reject all identity or coverage drift
before launching a GPU process. Experiment-specific evaluation shells are then
deprecated wrappers rather than independent implementations.
