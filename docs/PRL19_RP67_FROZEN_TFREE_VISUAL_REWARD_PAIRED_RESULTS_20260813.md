# PRL19 RP67 Frozen T-free Visual Reward Paired Results

Date: 2026-08-13  
Status: paired accuracy evaluation complete; primary held-out target/grounding audit pending
Experiment: `PRL-19-R0-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-VISUAL-API-8STEP-WS8`

## 1. Executive conclusion

The completed CoreDev evaluation measures downstream answer accuracy; it does
**not** directly measure the two properties that Focus/Target and Grounding
were introduced to improve.  Accuracy is therefore an auxiliary outcome in
this experiment, not the primary decision criterion.

- At Step 8, the visual-reward arm improves the canonical
  paired CoreDev-2511 Macro* from the matched no-visual control's `56.1964`
  to `57.8849`, a same-step gain of **`+1.6885 pp`**. It is also `+0.8529 pp`
  above the common Step-0 initialization.  This is an accuracy observation,
  not by itself evidence that targets became more image-dependent or that
  visual hallucinations decreased.
- At Step 16, the visual arm falls to `57.5422`. This remains `+0.5102 pp`
  above Step 0, but is `-0.3427 pp` below its own Step 8 and `-0.6573 pp`
  below the matched no-visual Step 16 (`58.1996`).
- The accuracy result is non-monotonic and does not support selecting a
  checkpoint on its own.  A primary held-out audit must compare target quality
  and visual grounding on paired trajectories before the visual reward can be
  called beneficial or harmful.
- Step 16 also develops a heavier extreme-length/repetition tail, especially
  on OCRBench. This is a model-health warning only: verbosity and repetition
  are not direct measurements of visual hallucination.

This is one paired `temperature=1`, seed-42 evaluation. The deltas are valid
within the common-random-numbers block, but remain single-seed evidence rather
than a statistical significance claim.

## 2. Controlled experiment

The direct scientific control is PRL17-R2 Frozen RP67 T-free. PRL19 preserves:

| Field | Fixed value |
|---|---|
| Policy | `Qwen3-VL-8B-Instruct` full training, including visual path |
| Representation | RP67 Step-2000 Adapter, frozen |
| Data | retained T1 mixed set, 77,541 rows, shuffle seed 42 |
| Batch | 16 prompts x 16 trajectories |
| Distributed | world8, micro2/rank, GA1 |
| Optimizer | AdamW, constant LR `1e-6`, no warmup |
| Sampling | temperature 1, top-p 1, top-k -1 |
| Response/tool limits | 20,480 response tokens; at most 6 tool attempts |
| Tool | `tgvf_focus_tool(target)` |
| Tool utility reward `T` | disabled |
| Evaluation | paired-seed-v1 CoreDev-2511, local Qwen2.5-72B scorer |

The only scientific treatment is enabling the two visual reward channels:

```text
R = 2*A - 0.05*max(0, tool_calls - 1) + F + G + P
```

- `A` is answer correctness.
- Focus/Target `F`: judge score `2/1/0` maps to `1/0.5/0`.
- Grounding `G`: judge score `2/1/0` maps to `1/0.5/-1`.
- `P=-1` for protocol/tool error; otherwise zero.
- `T` remains absent.

One gold-free OpenRouter call to pinned `qwen/qwen3-vl-32b-instruct` returns
both `F` and `G`. Its input contains the original image, question, ordered
successful targets, post-final-tool reasoning and final answer; it contains no
gold/reference answer.

Across 16 optimizer steps, the visual judge handled `3,143` applicable
trajectories and covered `3,132` (`99.65%`). Eleven bounded sample-local
provider failures received zero visual reward. Visual-judge cost was
`$0.5436`; answer-judge cost was `$0.7821`; combined judge cost was about
`$1.3257`.

## 3. Auxiliary paired accuracy result

All values are percentages. OCR mean is `(EN + CN) / 2` and contributes once
to Macro*. BLINK and MMMU use only their supported single-image subsets. The
five columns share the same manifest, policy prompt, tool schema, scorer,
paired seed namespace and initial RP67/Qwen state.

| Benchmark | Common S0 | No-visual S8 | Visual S8 | No-visual S16 | Visual S16 |
|---|---:|---:|---:|---:|---:|
| VStarBench Overall | 62.83 | 65.45 | 68.06 | 64.92 | 66.49 |
| HRBench Average / all | 58.50 | 60.00 | 63.50 | 64.50 | 60.00 |
| BLINK single-image (180) | 62.78 | 60.56 | 63.33 | 63.33 | 60.56 |
| OCRBench v2 English | 46.20 | 44.33 | 45.20 | 43.83 | 45.55 |
| OCRBench v2 Chinese | 40.87 | 34.45 | 38.77 | 37.95 | 41.75 |
| OCR EN/CN mean | 43.53 | 39.39 | 41.98 | 40.89 | 43.65 |
| MMMU-Pro single-image (269) | 50.19 | 47.58 | 44.98 | 47.96 | 48.70 |
| MathVista MINI | 68.00 | 68.00 | 67.33 | 70.00 | 69.00 |
| MathVerse five-version macro | 53.40 | 52.40 | 56.00 | 55.80 | 54.40 |
| **Macro\*** | **57.0320** | **56.1964** | **57.8849** | **58.1996** | **57.5422** |

The exact paired deltas are:

| Comparison | Macro* delta | Interpretation |
|---|---:|---|
| Visual S8 - common S0 | **+0.8529 pp** | auxiliary accuracy increase |
| Visual S8 - no-visual S8 | **+1.6885 pp** | same-step accuracy difference |
| Visual S16 - common S0 | +0.5102 pp | remains mildly above initialization |
| Visual S16 - Visual S8 | -0.3427 pp | visual arm does not scale monotonically |
| Visual S16 - no-visual S16 | -0.6573 pp | no-visual continuation wins at Step 16 |

At Step 8 the visual treatment improves VStar (`+2.62 pp`), HR (`+3.50`),
BLINK (`+2.78`), OCR mean (`+2.59`) and MathVerse (`+3.60`), while reducing
MMMU (`-2.60`) and MathVista (`-0.67`). At Step 16, relative to its own Step 8,
OCR, MMMU and MathVista improve, while VStar, HR, BLINK and MathVerse fall.
This is capability redistribution rather than uniform improvement.

## 4. Training-reward behavior

The visual judge itself remains active and mostly stable; it does not collapse
in the second half. However, the logged means include trajectories on which the
visual judge is not applicable.  Conditioning on covered, successful-tool
trajectories gives `F=0.7982 -> 0.8082` and `G=0.7286 -> 0.7717` from the first
to second half, while answer correctness falls `0.6860 -> 0.5991`.  These are
endogenous on-policy training signals over different prompt batches.  They
show what the training judge rewarded, but cannot establish held-out target or
hallucination improvement.

| Mean over optimizer steps | Steps 1-8 | Steps 9-16 |
|---|---:|---:|
| Answer component `2*A` | 1.3721 | 1.1982 |
| Focus `F` | 0.6182 | 0.6101 |
| Grounding `G` | 0.5642 | 0.5825 |
| `F+G` | 1.1824 | 1.1926 |
| Tool-attempt rate | 0.7773 | 0.7573 |
| Format-error rate | 0.0376 | 0.0322 |

The scalar still exposes a possible reward-ordering problem. A wrong answer can earn as much as
`F+G=2`, equal to the entire reward gap supplied by a correct answer
(`2*A=2`). In the latter eight steps, average `F+G` is almost exactly the same
size as average answer reward. The optimizer can therefore trade answer
correctness against judge-visible focus/grounding without a strong scalar
penalty. The Step-16 benchmark redistribution is consistent with this
mechanism, although the single run does not prove causality.

## 5. Output-health audit

The broad rollout distribution changes moderately, while its extreme tail
worsens.

| 2,240 supported rows per arm | Visual S8 | Visual S16 |
|---|---:|---:|
| Final-answer stop | 2,122 | 2,129 |
| Direct-answer stop | 79 | 67 |
| Max-token stop | 26 | 26 |
| Tool-call-cap stop | 13 | 18 |
| Mean final-answer chars | 1,792.5 | 1,982.2 |
| Median final-answer chars | 278.0 | 314.5 |
| Mean sampled trajectory tokens | 735.7 | 788.8 |
| Mean successful tool calls/sample | 1.6313 | 1.5906 |
| Tool-error samples/events | 113 / 173 | 130 / 205 |
| Sample-local evaluation failures | 0 | 0 |

The max-token rate remains exactly `26/2,240` at both checkpoints, and only
three capped sample IDs overlap. Thus the pathology is not a simple increase
in the count of capped responses. It is a heavier and more stochastic
repetition tail.

OCRBench shows the clearest effect:

| OCR prediction chars, 600 paired rows | Visual S8 | Visual S16 | No-visual S16 |
|---|---:|---:|---:|
| Mean | 1,979.3 | 2,829.6 | 1,893.8 |
| Median | 162.0 | 224.5 | 265.0 |
| P99 | 47,296.2 | 80,924.2 | 31,473.8 |
| Maximum | 100,190 | 134,775 | 91,822 |
| At least 50k chars | 6 | 11 | 6 |
| At least 100k chars | 1 | 4 | 0 |

The paired Step16-minus-Step8 OCR median is zero, but the mean is `+850.4`
characters. Excluding samples capped in either arm still leaves `+182.1`
characters, so the change is tail-amplified but not entirely tail-only. The
no-visual control also lengthens from Step 8 to 16, but only by `20.4%` in OCR
mean versus PRL19's `43.0%`; PRL19's distinctive change is its extreme tail.

One repeated sample reaches `134,775` characters. It caused the official
OCR scorer to take about `10m38s` at Step 16 versus `3m33s` at Step 8. The
official score completed, but the local summary reader then hit Python's old
default 128-KiB CSV field ceiling. This was an evaluation-reader boundary bug,
not a scorer or metric failure. The reader now permits fields up to the bound
of the already materialized artifact and restores the process-global setting
afterward in commit `c40030e`; 22 focused tests and the real
134,775-character artifact pass.

## 6. Decision and primary controlled audit

1. Do not select Visual Step 8 or Step 16 from Macro* alone.  Freeze both until
   the paired target/grounding audit is complete.
2. On the same held-out ordinals, compare common Step 0, no-visual Step 8/16,
   and visual Step 8/16. Report policy-level tool coverage separately, then
   paired `F/G` score distributions and win/tie/loss only on ordinals where
   both compared arms have a successful tool observation. Direct-answer rows
   are not zero-quality targets and must not be scored as zero.
3. Add a deterministic wrong-image calibration panel.  A healthy image-aware
   judge should lower Focus and Grounding when the original image is replaced
   by a donor image; otherwise apparent improvement may be a text-only judge
   shortcut.  Because the audit reuses the reward judge, its result remains a
   proxy and should be supplemented by blinded human review or an independent
   VLM on a stratified subset.
4. If the primary audit supports the treatment, the next experiment should
   restore strict answer dominance while changing
   only visual-reward composition. A theoretically clean constraint is to
   choose a visual coefficient whose maximum gain cannot compensate for a
   wrong answer, or to apply positive `F/G` shaping only within the same answer
   correctness class. This preserves learning from visual quality without
   allowing judge-facing verbosity to replace task success.
5. Keep output-length/repetition diagnostics in the report and reward audit.
   Do not silently truncate benchmark answers or change official scoring;
   truncation would change the measured policy rather than fix its behavior.
6. Any new reward variant should evaluate Step 8 first. Continue to Step 16
   only if target health, hallucination rate, auxiliary Macro*, and tail health
   jointly pass against this PRL19 baseline.

## 7. Provenance

Canonical PRL19 evaluation root:

```text
artifacts/policy/PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-8step-ws8/
  evaluation/PRL19-R0-FROZEN-RP67-TFREE-VISUAL-API-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
```

- `paired-summary.json` SHA256:
  `581c37c8e68d1dbe30e7be715e4dfd53fa8cf983b2b266b379fc1db16aaee156`
- completion receipt SHA256:
  `5d942cecb593137d504e04edd96ab135472b5e6a9f9136399eaa6757e42db587`
- Step-8 official summary SHA256:
  `3e621614fced0bd94b262c2e07fa3ccdeb49c77e5e0f0a0e80b6dd31e0625372`
- Step-16 official summary SHA256:
  `3339f3ab7a68b8380ffdadd0ecba52903d8700a2f985a8ab26209d5c6ccb693b`
- inference coverage: `2,240/2,240` per arm; 271 multi-image rows are explicit
  protocol holds, not failures
- official judge parse failures: Step 8 `2`, Step 16 `0`; all are bounded
  deterministic-incorrect cases
- evaluation sample-local failures: Step 8 `0`, Step 16 `0`

The common Step-0 and no-visual Step-8/16 values are taken from PRL17-R2's
accepted paired artifact. Original/Crop legacy-RNG values must not be mixed
into the deltas above.
