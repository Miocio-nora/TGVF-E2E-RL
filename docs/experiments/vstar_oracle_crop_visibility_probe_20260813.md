# VStar Oracle-Crop Visibility Probe (2026-08-13)

## Status and scope

This is a bounded internal diagnostic, not a new benchmark result and not a
change to the production Crop or TGVF protocols.  It tests one question:

> On VStar failures involving very small objects, would the same frozen policy
> answer correctly if the relevant source pixels were made directly visible?

The implementation is isolated on branch `diagnostic-vstar-oracle-crop`.
Production training and evaluation code is unchanged.

## Fixed sample

The probe uses 32 single-object `direct_attributes` questions from the pinned
191-row VStarBench snapshot:

- 27 primary rows whose annotated object occupies at most `0.02%` of the
  source-image area;
- 5 progressively larger medium-object controls.

The exact row membership is pinned in
`src/tgvf_rl/evaluation/vstar_oracle_crop_probe.py`.  All source images and
sidecar annotations are validated before generation.

## Paired arms

Every question is generated once in each arm using the same frozen PRL19
Step-8 Qwen policy weights, sampling seed, decoding settings, source image,
question, and options.

1. **Original**: source image only.
2. **Oracle crop**: source image plus a magnified crop selected from the held-out
   VStar GT box.
3. **Gray placebo**: source image plus a spatially uniform gray image with the
   exact same size and two-image prompt as the oracle crop.

The source `[x, y, width, height]` box is expanded to a square with side
`max(32, 4 * max(width, height))` and shifted inside the image without changing
the requested side length.  Qwen's normal multimodal processor then handles
resizing under the same `max_pixels=1,003,520` budget.

The model never receives `target_object`, the GT box coordinates, the gold
label, or any answer-bearing sidecar field.  GT is used only by the local image
crop operation.  Therefore the oracle-versus-placebo comparison controls for
the second-image channel and the prompt statement that a relevant region is
shown; their only intended difference is visible crop pixels.

## Runtime identity

- Policy: PRL19 frozen-RP67, T-free, visual-reward run, optimizer Step 8
- Policy weights SHA256: `44f9a7331edc2b71ae3f92b04b2b337687dc5c02d0551fafb6faf5866644844f`
- Model: `Qwen3-VL-8B-Instruct`, full-model checkpoint bundle
- Temperature: `1.0`
- Seed: `42`
- Maximum generated tokens: `256`
- Language attention: `TRITON_ATTN`
- Vision attention: `TORCH_SDPA`
- Scoring: exact final-option extraction; no API/VLM judge
- Probe manifest SHA256: `5f8ada7ab76da5a5b63be1a9ca9aa58a73e4b781986035150f98c592e92a7250`

The 96 generations took `7.0 s` after model initialization on one B200.  Engine
load and warm-up dominated total wall time.

## Results

| Stratum | N | Original | Gray placebo | Oracle crop |
|---|---:|---:|---:|---:|
| Tiny objects | 27 | 14/27 (51.9%) | 12/27 (44.4%) | 27/27 (100.0%) |
| Medium controls | 5 | 3/5 (60.0%) | 4/5 (80.0%) | 5/5 (100.0%) |
| **All** | **32** | **17/32 (53.1%)** | **16/32 (50.0%)** | **32/32 (100.0%)** |

Paired transitions provide the clearest diagnostic:

- Oracle versus original: 15 oracle-only correct, 0 original-only correct.
- Oracle versus placebo: 16 oracle-only correct, 0 placebo-only correct.
- Original versus placebo: 4 original-only correct, 3 placebo-only correct.
- All 15 original failures were rescued by the oracle crop.
- Twelve failures were rescued by oracle pixels but not by the placebo.

Examples of oracle-specific corrections include a green helmet, black cap, red
toothbrush, white cap, purple umbrella, white watch, green soda can, black trash
can, white glove, and black-and-white dog.  In multiple cases the original arm
reported that the tiny target was absent, while the crop arm selected the
correct option directly.

For context only, the already-completed natural-TGVF PRL19 Step-8 run received
16/32 under its official semantic scorer on this subset.  That trajectory uses
a different tool prompt and output protocol, so it is not treated as a fourth
formally paired arm here.

## Interpretation

This small probe is sufficient for its internal diagnostic purpose.  It shows
that these VStar errors are not primarily failures of answer knowledge: the
same frozen policy can answer every selected question once the annotated local
pixels are visible.  The near-identical original and gray-placebo aggregate
accuracy rules out a generic benefit from adding a second image or changing the
prompt.  The strong oracle-versus-placebo contrast instead identifies local
visual availability/resolution as the dominant bottleneck for this slice.

Accordingly, a wrong answer on these tiny-object cases must not automatically
be classified as unhealthy target hallucination.  Grounding/focus reward
analysis should distinguish at least:

1. an incorrect target or irrelevant observation;
2. a correct target whose returned representation still lacks resolvable local
   evidence;
3. visible evidence followed by an incorrect answer.

The result validates the value of an oracle-crop positive control.  It does not
show that the current TGVF representation performs an oracle crop, nor does it
estimate full-VStar accuracy.

## Artifacts and verification

Runtime artifacts are stored outside Git at:

`artifacts/evaluation/DIAG-VSTAR-ORACLE-CROP-STEP8-FG-V2/`

The directory contains the immutable manifest, all 96 result rows, a paired
summary, and the runtime log.  Before the GPU run, 19 targeted unit tests,
Ruff, `git diff --check`, and a real-data 32-case prepare pass succeeded.

