# LAS&T and MMAD texture benchmarks

This is the canonical project document for the LAS&T 2D Texture Retrieval and
MMAD benchmarks. It explains what each benchmark measures, records the exact
local dataset and evaluation protocol, reports the four-arm step-16 comparison
completed from 2026-08-13 through 2026-08-15, and provides the commands and
artifact identities needed to reproduce or extend it.

The project can evaluate the same immutable task manifest with four model
pipelines: `original`, `crop`, `tgvf`, and `tgvf_crop` (TGVF + crop). Dataset
preparation is independent of the model arm, so a paired run changes the
model/tool pipeline but never the questions or input-image bytes.

## What the two benchmarks measure

| Benchmark | Plain-language task | Main capability stressed | Evaluated set | Main view in this report |
| --- | --- | --- | ---: | --- |
| LAS&T - 2D Texture Retrieval | Given a four-panel image, find which of B, C, or D has the same texture identity as panel A. | Fine-grained texture matching when shape and background may agree or disagree. | 3,200 questions | Four-condition macro accuracy |
| MMAD | Compare an industrial query image with a normal reference and answer multiple-choice inspection questions. | Anomaly detection and explanation across seven industrial-inspection subtasks. | 39,670 questions | Four-dataset/seven-task macro with upstream-like permissive answer extraction and a fixed full denominator |

### LAS&T - 2D Texture Retrieval

The local LAS&T artifact is the 2D texture-retrieval part of the LAS&T release
([Zenodo record 16969919](https://zenodo.org/records/16969919), DOI
`10.5281/zenodo.16969919`). A quiz contains one anchor texture in panel A, one
positive view from the same texture identity, and two negatives in panels B-D.
The answer is the position of the positive view.

The eight evaluated source directories vary whether the rendered shape is the
same or different and whether the background is black or textured. They are
grouped into four canonical conditions:

1. same shape, black background;
2. different shape, black background;
3. same shape, textured background;
4. different shape, textured background.

This makes LAS&T useful for asking whether a model recognizes texture itself
instead of relying on silhouette, scene, or background shortcuts. The local
snapshot also contains `AllDifferent_Traininig`, but that training directory is
never part of evaluation. Each physical condition directory contains 500
texture identities with five rendered images per identity.

### MMAD

MMAD is a multiple-choice benchmark for multimodal large language models in
industrial anomaly detection ([paper](https://arxiv.org/abs/2410.09453),
[official code](https://github.com/jam-cc/MMAD),
[dataset](https://huggingface.co/datasets/jiang-cc/MMAD)). Its 8,366 annotated
query images come from GoodsAD, MVTec-AD/DS-MVTec, MVTec-LOCO, and VisA.
Questions cover seven evaluation tasks:

1. Anomaly Detection;
2. Defect Classification;
3. Defect Localization;
4. Defect Description;
5. Defect Analysis;
6. Object Classification;
7. Object Analysis.

In the one-shot setting used here, the model sees a normal example next to the
query. MMAD therefore tests more than binary defect recognition: it also asks
what the object is, where and what the defect is, and what consequence or
structural detail is visible. The benchmark's primary aggregation first forms
the seven task scores within each dataset, then averages the four datasets;
Anomaly Detection uses the mean of normal and abnormal accuracy.

## Dataset provenance and local snapshots

The defaults below point to the validated snapshots on this host. The
download archives have already been removed; preparation neither downloads
nor deletes anything.

| Dataset | Snapshot | Validated logical size | Inventory |
| --- | --- | ---: | --- |
| LAS&T - 2D Texture Retrieval | `/nvmesv/dredvpn009/datasets/benchmarks/las_t_2d_texture_retrieval/snapshot` | 2,358,118,555 bytes (2.358 GB) | 22,500 JPEGs, all 512 x 512 |
| MMAD | `/nvmesv/dredvpn009/datasets/benchmarks/mmad/snapshot` | 30,102,454,945 bytes (30.102 GB) | 63,217 regular files; 8,366 annotated query images and 39,670 questions |

Filesystem allocation can be larger than the logical byte counts above.
Source identity and archive checksums remain recorded in each dataset's
`DEPLOYMENT.json`.

The LAS&T Zenodo metadata says CC BY 4.0 while the paper text says CC0; this
deployment conservatively records CC BY 4.0 and retains DOI attribution. MMAD
is CC BY-NC-SA 4.0 for research use, and its component image datasets retain
their own license restrictions. The downloaded archives were removed only
after extraction and validation; the two source roots now contain zero archive
files.

MMAD is pinned to Hugging Face revision
`c4ed190dcb530f2f673ab293e575ad32054bb3cf`. Evaluation uses
`snapshot/mmad.json` with SHA-256
`639343b491bc67b2abb3c5d719f221ce27f83b2ed97948f4e88055aaa31f1c1e`,
not the annotation bundled with a newer GitHub checkout.
The public README advertises 39,672 questions, while this pinned annotation
contains 39,670 validated questions; every count and score in this report uses
the pinned local population.

## Default protocols

LAS&T uses the eight physical test directories and excludes
`AllDifferent_Traininig`. It deterministically samples 400 quizzes from each
directory (3,200 total), with seed `20260813` and prompt profile `neutral_v1`.
Each question is rendered as a lossless 1024 x 1024 four-panel PNG. The eight
physical directories map to four canonical shape/background conditions; their
separate source-directory identities are retained for diagnostics.

This is a reproducible project protocol, not a byte-for-byte replay of the
upstream evaluator. The paper supplement states 400 questions per test, while
the released `Test_All.py` currently says 200; this profile follows the paper's
400 count. It intentionally fixes discovery order and RNG, uses the neutral
prompt instead of selecting the best of several prompts, requires the two
negative identities to differ, labels with Pillow, stores lossless PNG, and
counts invalid answers as wrong. The upstream script uses filesystem order and
unseeded sampling, can independently sample the same negative identity twice,
uses OpenCV labels and API-specific encoding/retries, and its collected
`Correct ratio` excludes invalid answers. Consequently, scores from this
protocol must be labelled `derived`, not directly compared as official paper
reproductions.

MMAD creates one task per question and retains the annotation's option order.
The default is an official-derived one-shot protocol using
`random_templates[0]`. Because the policy boundary accepts one image, the
normal template and query are rendered into one labelled, lossless 1048 x 560
canvas: **NORMAL TEMPLATE is on the left and QUERY is on the right**. Each
source is fitted into its 512 x 512 panel with its aspect ratio preserved.
Segmentation masks are never read or supplied to the model. Zero-shot uses the
unaltered query image; `similar_templates[0]` is available as an explicit
non-default variant.

`--prefix N` selects a deterministic prefix for a smoke test. For `suite`, it
selects up to `N` rows from each already prepared component manifest. The suite
places LAS&T first and MMAD second, rewrites `ordinal` and `row_number` to one
contiguous range, and preserves every `sample_id`. The default full one-shot
suite therefore has 3,200 + 39,670 = 42,870 tasks.

## The 512 x 512 visual limit

The shared setting is `max_pixels = 512 * 512 = 262144`: it is an **area cap
inside the Qwen image processor**, not a request to rewrite every benchmark
asset to a 512 x 512 square. Native/composite assets remain unchanged,
`pre_resize_assets` is false, and aspect ratio is preserved. This distinction
matters for the 1024 x 1024 LAS&T composites and the 1048 x 560 MMAD one-shot
canvases. The pinned Qwen3-VL processor contract records `patch_size = 16`,
`merge_size = 2`, and therefore a spatial smart-resize factor of 32.

## Completed four-arm evaluation

### Scope and immutable configuration

The successful v2 inference run launched at 2026-08-13 06:52:36 JST. Original
finished at 08:18:06 and Crop finished at 11:34:58. It compared stock
Qwen3-VL-8B-Instruct with the PRL14 clean-final Crop policy at optimizer step
16 on every task in the combined suite.

PRL20-R0 atomic Crop+TGVF ran on the same immutable 42,870-task manifest from
2026-08-14 15:40:29 to 17:23:23 JST. An initial PRL17-R2 run that day used the
generic benchmark prompt instead of the training-matched TGVF prompt because
of a prompt-materializer routing error; it is deprecated and excluded from
every formal result below. The corrected PRL17-R2 TGVF-only run used
`build_tgvf_visual_messages` and ran from 2026-08-15 10:19:19 to 12:26:37 JST.
The accepted PRL20 and corrected PRL17 four-rank runs both completed on their
first attempt, passed final coverage and identity audits, and were scored under
the same strict and permissive contracts as the baseline.

| Common setting | Bound value |
| --- | --- |
| Task count | 42,870: 3,200 LAS&T + 39,670 MMAD |
| Task-manifest SHA-256 | `89d1510862c7cde5e0b3dd37b7d5fa345c7ed2c573a31e59a151cba3f9bb1ace` |
| Visual processing | `min_pixels=65536`, `max_pixels=262144`, preserve aspect ratio, no asset pre-resize |
| Qwen processor geometry | `patch_size=16`, `merge_size=2`; 262,144 is a total-area cap, not a forced square resize |

The arms are split across three immutable matrices because the completed
baseline predates the two RL-policy runs. The exact task-manifest SHA and visual
contract above are identical in all three, so results remain paired by task.

| Evaluation slice | Matrix | Matrix identity | GPU assignment | State |
| --- | --- | --- | --- | --- |
| Original + Crop step 16 | `configs/evaluation/texture_last_mmad_original_crop_prl14_step16_512_v1.json` | `a76be7017ae38dd34a738bafca114d60bf2ea4bb4cb7f2b72cd7c457fd500533` | Crop: B200 0-3; Original: B200 4-7 | Complete |
| PRL17-R2 TGVF step 16 | `configs/evaluation/texture_last_mmad_prl17_tgvf_step16_corrected_prompt_512_v1.json` | `feb796482ead0c8893782ac6070b8629ae882d1ff14ebfc4648fd0271c567700` | B200 0-3 | Complete |
| PRL20-R0 Crop+TGVF step 16 | `configs/evaluation/texture_last_mmad_prl20_crop_tgvf_step16_512_v1.json` | `25afd59c8602b1bd00751513fd6acb4ae79bd8571fa78fda2b68ec4203751ca1` | B200 4-7 | Complete |

Arm closures:

| Arm | Exact closure | Evaluation protocol and tool boundary |
| --- | --- | --- |
| Original | `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct`; model-tree SHA-256 `d4c62edf0fa6622ca511a6baa3b75b6314b8d702004f92556c46ca59dbaf8d73` | Stock one-pass generation; temperature 0; batch 8; at most 2,048 output tokens; content-addressed per-sample seeds |
| Crop step 16 | PRL14 clean-final full-model snapshot; policy-weights SHA-256 `50f5d9dd7ecdbf8d9baf46c00c13b1c3719de37b09f5aa91c40aabc758e06beb`; snapshot identity `d813b7511a35751380469516d58d2b55b1b1a933d86229210e92d95b6d7a928d` | `deepeyes_official_visible_native_crop_v1`; Crop-only; at most 6 calls; concurrency 8 per GPU |
| TGVF PRL17-R2 step 16 | Paired full-Qwen/RP66 closure under `artifacts/policy/PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`; run identity `9c64b840e540fccd98bab30e2f05783b15f43dddcc0e926a1778e713a1f74746`; destination snapshot identity `1445d9a755389c0e3275c09c3265a04199f890732f99e27bcf47fe860d10bbbf` | `training_run`; matched materializer `build_tgvf_visual_messages`; prompt bundle `e74bb5e1253af107ff27badfcfaca747b94574e19677d22cfe42b0b1c0ba5633`; tool profile `tgvf_only`; only `tgvf_focus_tool`; at most 6 calls; concurrency 8 per GPU |
| Crop+TGVF PRL20-R0 step 16 | Paired full-Qwen/RP66 closure under `artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/evaluation/PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`; run identity `190c3db105c5e35d8d8aab4ad837409b4b39b8ce7f22e7b2393249a5e1898457`; destination snapshot identity `6467170e1c01ce1d03357270913ed10e5ddcfb63c1ea3226557c04b8871c6345` | `training_run`; tool profile `crop_tgvf`; only atomic `tgvf_crop_tool`; at most 6 calls; concurrency 8 per GPU |

For the matched TGVF policy materializers, `template_tools_argument=[]` is
intentional: the frozen training prompt itself specifies the callable tool
protocol. The runtime dispatcher independently exposes only the tool allowed by
the bound profile. In the corrected PRL17 run, every one of the 42,859
successful calls was therefore executed as `tgvf_focus_tool`; model-generated
misspellings were rejected and never dispatched.

The Original/Crop-specific immutable details remain:

| Setting | Bound value |
| --- | --- |
| Original model | `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct` |
| Original model-tree SHA-256 | `d4c62edf0fa6622ca511a6baa3b75b6314b8d702004f92556c46ca59dbaf8d73` |
| Original generation | temperature 0; batch 8; max 2,048 output tokens; content-addressed per-sample seeds |
| Crop model | PRL14 clean-final step-16 full-model snapshot |
| Crop policy-weights SHA-256 | `50f5d9dd7ecdbf8d9baf46c00c13b1c3719de37b09f5aa91c40aabc758e06beb` |
| Crop snapshot identity | `d813b7511a35751380469516d58d2b55b1b1a933d86229210e92d95b6d7a928d` |
| Crop evaluation identity | `97e573bcea105caaea8b1f3aa8f87237776b7145167cf1f0ef258e9f93e30f8b` |
| Crop protocol | `deepeyes_official_visible_native_crop_v1`; at most 6 crop calls; concurrency 8 per GPU |
| GPU assignment | Crop on B200 GPUs 0-3; Original on B200 GPUs 4-7; `ordinal % 4` sharding |

In the completed Original/Crop run, the four rank files in each of those two
arms contain exactly 10,718, 10,718, 10,717, and 10,717 rows. All eight
successful-v2 workers completed on attempt 1 with exit code 0. There were no
retries, OOMs, CUDA failures, engine failures, malformed JSON rows, duplicate
ordinals, missing ordinals, or identity drift in that accepted run.

The corrected PRL17 and accepted PRL20 policy arms have the same exact rank
counts: 10,718, 10,718, 10,717, and 10,717, totaling 42,870 each. Their eight
workers also completed on attempt 1 with exit code 0. Final scans found no fatal
exception, OOM, CUDA failure, or engine failure. Both completion audits found
all ordinals 0 through 42,869 exactly once, enforced `ordinal % 4` assignment,
and verified the manifest, evaluation, policy, result-row, and rank-file
identities.

The deprecated generic-prompt output remains on disk only as forensic evidence
under `TEXTURE-LAST-MMAD-CURRENT-3ARM-512-V1`; it must not be quoted as a TGVF
benchmark result.

### Main comparison

For scientific interpretation, this report leads with MMAD's upstream-like
permissive last-letter/fuzzy answer extraction. It correctly recovers choices
from outputs such as `B. Yes.` that the project's strict explicit-answer
parser rejects. The main permissive view keeps the same fixed 39,670-question
denominator for every arm, so the 107 completed Crop rows with no final answer
still count as wrong. LAS&T retains the project's predeclared four-condition
metric and strict parser; no post-hoc permissive LAS&T metric is promoted.

The saved permissive score contract has identity
`9653cc6c0d2d09a667f983ca1dc3118083fd7e1beda1d7fd49064ffb19167f60`.
Its schema is `tgvf-texture-parser-contract-v1`; it records
`last_answer_extraction=strict_explicit_choice_v1`,
`mmad_parser=upstream-like-permissive`,
`denominator=fixed_complete_task_manifest`, invalid answers counted as
incorrect, `drop_invalid_rows=false`, and `exact_upstream_evaluator=false`.

| Metric | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 | Crop - Original |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAS&T four-condition macro | **72.4167%** | **68.2708%** | **64.3021%** | **61.7500%** | -4.1458 pp |
| LAS&T micro | 71.2500% | 65.6562% | 61.4062% | 59.3438% | -5.5938 pp |
| MMAD official aggregation structure, permissive fixed-denominator reparse | **69.2635%** | **67.9939%** | **66.5669%** | **67.0747%** | -1.2696 pp |
| MMAD micro, permissive fixed-denominator reparse | 69.4404% | 68.0691% | 67.2851% | 66.8036% | -1.3713 pp |
| MMAD still unparsed | 0 / 39,670 | 107 / 39,670 | 0 / 39,670 | 1 / 39,670 | +107 |

Under the main views, neither trained tool arm improves on Original or Crop.
LAS&T ranks Original (72.4167%), Crop (68.2708%), corrected TGVF-only
(64.3021%), then Crop+TGVF (61.7500%). Relative to Crop, TGVF-only loses
3.9687 points and Crop+TGVF loses 6.5208. MMAD is much tighter: Original
(69.2635%), Crop (67.9939%), Crop+TGVF (67.0747%), and TGVF-only (66.5669%).
Relative to Crop, TGVF-only loses 1.4270 points and Crop+TGVF loses 0.9191;
Crop+TGVF is 0.5078 points above TGVF-only on the primary dataset/task macro,
although TGVF-only is 0.4815 points higher on MMAD micro. Thus the large
strict-score changes below primarily reflect answer-format compliance, while
the permissive fixed-denominator comparison shows a small MMAD regression and
a much larger LAS&T regression.

LAS&T condition breakdown:

| Canonical condition | Samples | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 | Original invalid | Crop invalid | TGVF invalid | Crop+TGVF invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Different shape, black background | 400 | 75.5000% | 77.2500% | 75.5000% | 72.0000% | 3 | 30 | 20 | 15 |
| Different shape, textured background | 800 | 60.0000% | 47.1250% | 44.1250% | 41.2500% | 11 | 186 | 120 | 126 |
| Same shape, black background | 800 | 88.0000% | 92.3750% | 85.2500% | 81.0000% | 2 | 20 | 29 | 39 |
| Same shape, textured background | 1,200 | 66.1667% | 56.3333% | 52.3333% | 52.7500% | 13 | 248 | 153 | 111 |

The two textured-background conditions drive most of the LAS&T degradation.
Corrected TGVF-only is 2.5521 points above Crop+TGVF on the four-condition
macro, but both trained TGVF arms still trail Crop in every condition.

MMAD per-dataset task macro under the permissive fixed-denominator reparse:

| MMAD dataset | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 |
| --- | ---: | ---: | ---: | ---: |
| GoodsAD | 64.9218% | 64.7045% | 63.3571% | 63.0390% |
| MVTec-AD (including DS-MVTec) | 80.8414% | 75.1589% | 73.5319% | 77.1655% |
| MVTec-LOCO | 62.9856% | 63.4273% | 62.6176% | 61.7720% |
| VisA | 68.3052% | 68.6847% | 66.7609% | 66.3224% |

MVTec-AD is the only MMAD dataset on which Crop+TGVF exceeds Crop, by 2.0066
points; corrected TGVF-only trails Crop there by 1.6270 points. Neither TGVF
arm wins a per-dataset column. Crop+TGVF's 0.5078-point aggregate-macro edge
over TGVF-only comes from mixed subtask changes across datasets rather than a
uniform advantage.

### PRL20 Step 8 versus Step 16 paired LAS&T

A separate LAS&T-only run compares PRL20-R0 Crop+TGVF Step 8 and Step 16 under
strict common-random-number pairing. Both arms contain all 3,200 tasks with no
missing or duplicate rows and bind the same task-manifest/seed namespace
`9f2d76585569e7c9892d6752af6b2e356d481b7d149e17107af5314f9c352ccd`, RNG
protocol `3b66222e506290de90e4394fb62878d3a58dd9c4558986e48172cd3d88a66538`,
prompt bundle `5efbd617f69ce9b3a6cb6b0c96bf7e24d8156b6e4dab9af55c9dfe5692c52e69`,
and `build_crop_tgvf_visual_messages` materializer.

| Paired LAS&T metric | PRL20 Step 8 | PRL20 Step 16 | Step 16 - Step 8 |
| --- | ---: | ---: | ---: |
| Four-condition macro | 56.8333% | 62.5208% | +5.6875 pp |
| Micro | 54.5000% | 60.2812% | +5.7812 pp |
| Correct | 1,744 / 3,200 | 1,929 / 3,200 | +185 |
| Invalid | 515 / 3,200 (16.0938%) | 184 / 3,200 (5.7500%) | -331 (-10.3438 pp) |
| Valid-only micro | 64.9534% | 63.9589% | -0.9945 pp |
| Mean tool calls per task | 2.5378 | 1.7422 | -0.7956 |
| Tool errors | 252 | 30 | -222 |
| Mean assistant turns | 3.6166 | 2.7516 | -0.8650 |
| Mean row wall time | 4.3942 s | 3.4589 s | -0.9353 s |

| Canonical condition | Step 8 | Step 16 | Delta |
| --- | ---: | ---: | ---: |
| Different shape, black background | 68.2500% | 71.5000% | +3.2500 pp |
| Different shape, textured background | 33.1250% | 44.8750% | +11.7500 pp |
| Same shape, black background | 76.3750% | 80.1250% | +3.7500 pp |
| Same shape, textured background | 49.5833% | 53.5833% | +4.0000 pp |

The Step 16 headline gain is driven mainly by better answer formatting and
tool-protocol compliance, not better accuracy among already parseable answers:
invalids fall by 331 while valid-only micro falls by 0.9945 points. In the
paired transitions, 235 samples move from invalid at Step 8 to correct at Step
16, while 62 move from correct to invalid, for a net +173 from those two flows;
correct-versus-valid-wrong flips contribute the remaining net +12. The largest
condition gain is on different-shape textured backgrounds.

The Step 8 and paired Step 16 matrix identities are
`b63433b3884d61064cd005326988d77871e6ad7965458e50c12332883110798b` and
`1752e2782af9c09a69604983bd79c43964b23b0b6be6cc1001ddf133a4799bc0`.
This paired Step 16 run uses a LAS&T-only manifest and paired seed namespace,
so its 62.5208% macro must not replace the 61.7500% PRL20 full-suite value in
the four-arm table above.

### Strict-parser compliance diagnostic

The baseline `score.json` and policy-arm `score-strict-v2.json` artifacts use
the project's strict parser. Those numbers remain useful as a
protocol-compliance diagnostic, but they are not the main accuracy view in
this report:

| Strict diagnostic | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 | Crop - Original |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAS&T invalid | 29 / 3,200 | 484 / 3,200 | 322 / 3,200 | 291 / 3,200 | +455 |
| LAS&T valid-only micro | 71.9016% | 77.3564% | 68.2766% | 65.2802% | +5.4548 pp |
| MMAD official aggregation structure | 69.2624% | 32.7951% | 40.1800% | 54.2418% | -36.4673 pp |
| MMAD micro | 69.4379% | 31.4898% | 38.7749% | 52.2687% | -37.9481 pp |
| MMAD invalid | 4 / 39,670 | 21,600 / 39,670 | 17,457 / 39,670 | 8,676 / 39,670 | +21,596 |
| MMAD valid-only micro | 69.4449% | 69.1312% | 69.2477% | 66.9000% | -0.3137 pp |
| Overall strict micro | 69.5731% | 34.0401% | 40.4642% | 52.7968% | -35.5330 pp |
| Overall invalid | 33 / 42,870 | 22,084 / 42,870 | 17,779 / 42,870 | 8,967 / 42,870 | +22,051 |

The 32.7951% Crop strict value is dominated by answer-format noncompliance, not by
42,870 missing or failed inferences. The policy commonly returns a choice plus
option text, for example `B. Yes.` or `C. Irregular, jagged edges.`. The strict
parser accepts a bare terminal choice, a supported answer wrapper, or one
unambiguous explicit answer marker; it deliberately does not guess a choice
from arbitrary prose. As a result, 21,493 MMAD rows are
`ambiguous_or_unmatched` and 107 have no final answer after reaching the tool
call cap.

The two TGVF policies are more compliant than Crop but still format-sensitive:
corrected TGVF-only has 17,457 strict-invalid MMAD rows and Crop+TGVF has
8,676, versus 21,600 for Crop. Their strict MMAD macros are 40.1800% and
54.2418%, while their permissive macros are 66.5669% and 67.0747%. Corrected
TGVF-only's 69.2477% valid-only micro is close to the baselines despite its low
strict fixed-denominator score, so strict-parser gains must not be presented as
task-accuracy gains.

The permissive view reuses upstream-like answer extraction, but it is not an
exact run of the released evaluator: the main table retains the fixed full
denominator, whereas upstream removes invalid records. If those 107 Crop rows
are removed exactly as upstream does, Crop becomes 68.1732% macro and 68.2532%
micro; Original remains 69.2635% macro and 69.4404% micro. This report leads
with the slightly more conservative fixed-denominator values so missing
answers never improve a score. The permissive parser can also select the last
isolated capital letter or a fuzzy option match from long reasoning.

The two views answer different questions:

- the strict score measures answer correctness **and** explicit protocol
  compliance;
- the main permissive view estimates choice accuracy after forgiving the
  dominant output-format error.

Any comparison or paper table must name which parser it uses. Reporting only
the 32.7951% value without its 54.4492% MMAD invalid rate would be misleading;
the saved strict scores are retained for audit even though the report
emphasizes the permissive view.

### Runtime and agent behavior

| Runtime observation | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 |
| --- | ---: | ---: | ---: | ---: |
| Four-GPU wall time through last rank | 1 h 25 min 30 s | 4 h 42 min 22 s | 2 h 7 min 18 s | 1 h 42 min 54 s |
| Model interaction | One pass | Multi-turn agent loop | Multi-turn agent loop | Multi-turn atomic-tool agent loop |
| Mean tool calls per task | 0 | 1.3077 overall | 0.9997 overall | 1.6739 overall |
| Mean assistant turns per task | 1 | 2.3103 overall | 2.0014 overall | 2.7205 overall |

LAS&T was the expensive part of Crop inference: it averaged 3.5775 crop calls
and 4.5913 assistant turns per question, with median per-row wall time 45.94 s
and P95 89.13 s. MMAD averaged 1.1246 calls and 2.1263 turns, with median 3.50 s
and P95 9.66 s. In contrast, 42,864 of 42,870 Original rows generated only one
choice token plus the stop token.

Crop recorded 112 recoverable out-of-range crop attempts across 75 rows and
130 `tool_call_cap` stops. These are model/tool-protocol outcomes, not worker
or engine failures; every row was durably recorded and scored.

Corrected TGVF-only made 42,859 successful focus-tool calls and averaged
2.0014 assistant turns. Every successful call was to `tgvf_focus_tool`.
LAS&T had median/P95 row times of 2.13/14.97 s, while MMAD had 1.96/2.34 s.
Across the suite, 75 rows reached `max_tokens`; all 73 recorded tool errors,
spread across 35 rows, were recoverable model-generated call-format errors.
No invalid tool name was executed and there were zero non-recoverable tool or
infrastructure errors.

Crop+TGVF made 71,761 atomic tool calls and averaged 2.7205 assistant turns.
Its LAS&T median/P95 row times were 2.79/5.45 s and its MMAD values were
2.25/5.11 s. It recorded 1,998 recoverable tool errors, largely call-limit and
tool-execution/parse outcomes, with zero non-recoverable errors. These counts
explain agent behavior; all 42,870 rows in both policy arms remain in the fixed
denominator.

### Result artifacts and integrity

Durable output root:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-ORIGINAL-CROP-PRL14-STEP16-512-V1
```

| Artifact | SHA-256 |
| --- | --- |
| Original merged `results.jsonl` | `6063bc0b2852863726cc83bdf64e6b7bffb413302211731431a80cebb356d63f` |
| Original `score.json` | `7eec5eb382b1b9bbaea21ceef1d707602bfcaf50153393e1e4cd81a285a5a779` |
| Original `score-permissive.json` | `f7404060a8e3d967eaace0237ef495af3b80e66b21fdf0a80ce5b5e6ab01489c` |
| Crop `score.json` | `71453305d4abeb1df4caf6c83adb4ae3a4d843ae41f8e6c8ee8b5ca9380d920e` |
| Crop `score-permissive.json` | `ada6f178f8d0d26babd33b7eb56bcd54de62feaab8506ff91745c31cbe0b9f4e` |
| Final `supervisor-summary.json` | `5c3730b5abd64d06729f831456b5e2974e751d87177de514c3eb439f40399ca4` |

The two baseline permissive artifacts are, respectively:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-ORIGINAL-CROP-PRL14-STEP16-512-V1/original-qwen3-vl-8b-instruct/score-permissive.json
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-ORIGINAL-CROP-PRL14-STEP16-512-V1/crop-prl14-cleanfinal-step16/score-permissive.json
```

They are schema-v2, provenance-bound reparses of the immutable result rows and
both carry the permissive parser-contract identity listed above. The original
strict `score.json` files remain immutable schema-v1 evidence.

The completed policy-arm output roots are:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-PRL17-TGVF-STEP16-CORRECTED-PROMPT-512-V1/tgvf-prl17-r2-rp67-step16-corrected-prompt
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-PRL20-CROP-TGVF-STEP16-512-V1/tgvf-crop-prl20-r0-rp67-step16
```

| Immutable artifact | TGVF PRL17 SHA-256 | Crop+TGVF PRL20 SHA-256 |
| --- | --- | --- |
| `policy-benchmark-config.json` | `b6d3c5292da47515c502d82a04c0c0f4dbfdf81ea26c1211cb6643efa44172dd` | `7003a90c36ce54f6489ee7d3abe3ca38e4ebc4b155858fa07cff8179ed994b6f` |
| `evaluation-identity.json` | `185398b7871bd51c3477c63ec1e138b0ac1d847b1b3b644bc1dd02352a8ad566` | `fe2750270e5f890016c0c6a583745bb3e03c15b1701b63c57e9096ce8cf9fe07` |
| rank 0 JSONL, 10,718 rows | `84d235cfc12dbe89f1709d174b41ee19d9e6ee5785eaaa1f62f8893141ec49a1` | `dde998acfefc8c71d7d1003dc238ea5bb127fdb34ff2665f2650fdb64e60f729` |
| rank 1 JSONL, 10,718 rows | `8ad04f39ba3a1862589ab14f6fc857be2848c521c7f0f6cbcc1234182149962d` | `87eba240446c2c5300df0bdccd091321da7de7ea148bfbf6761c3ad04ed382fa` |
| rank 2 JSONL, 10,717 rows | `bcb54468d67979f966a620bb1fa1de068152de0195382ae3ddfe0434c509558a` | `32f984c9d10c1b24869c91239772dc5c96b896f7f99ae53a4d2e141b0958b15f` |
| rank 3 JSONL, 10,717 rows | `b01fd689adfcb050fd160ef4357f60fb70489aa832c549365289ee91113911e6` | `5607fbbc68c72917d736fe0246fd6675d3871143c048466b709cf8586e2614bc` |
| `score-strict-v2.json` | `7842421223c0a3ed98c3cfa5f91e5cd669fdfdc42a0c2f83bec85714e6668671` | `8038f371a51d4616c547c77e76f6033ea9ceb94624a48ac493b8e58b41a3ea68` |
| `score-permissive.json` | `1e166f70e944fd6a6186ff714de26b0cfca2c2c4154b5a85e88ac6e2a7eea8da` | `7bf86270e00d571212774d46e9d3755d2462cf2a017a1634d07ebfef42aaca8e` |
| `evaluation/completion-audit.json` | `97c567c6945cfdcfe2b65cfc048883a0a44ccaadbec020dc2134b474c08f57b8` | `65a8c6062c67993bf35cb85369100b6c35eb3f4b450b5c5210dcc161b4379745` |
| Supervisor log | `5a7fdc3fce6e32a197e1a7ecc9909720e75729d5cff454da9574553aa162dd3b` | `6daf9522b4644227a2bb074f8637bc2c41bca3ac51d3788c5b45a136e284e9f4` |

The TGVF and Crop+TGVF evaluation identities are, respectively,
`ec0a009e61268afc2d1966656476d5214da220e049ffaeda88e0e25e1e91c175`
and `154681b9c6d76ff78fbde97da805a51d2e4bb4687483839383b04198bda730a3`;
their policy-weight hashes are
`95e4f3e54b51b4f56237247853c365bf4f89d16b897c13038f9c19a6328e3a37`
and `703afc57b68b26c8ca5bea12cc2b424595715db1b4ef2f8bec4d504256182780`.
The completion-audit identities are
`3adec20fc38d3089ec33ec4178e7fa94ddf263930f5d78a78da77bb467a6ccaf`
and `f89f1ae84fccb2dc230a3862a31b637a01645e5b732da280af421f3766916a67`.
All four permissive score artifacts carry parser-contract identity
`9653cc6c0d2d09a667f983ca1dc3118083fd7e1beda1d7fd49064ffb19167f60`.

The PRL20 paired LAS&T artifacts described above are rooted at:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-PRL20-CROP-TGVF-STEP8-512-V1/tgvf-crop-prl20-r0-rp67-step8
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-PRL20-CROP-TGVF-STEP16-LAS-PAIRED-512-V1/tgvf-crop-prl20-r0-rp67-step16-las-paired
```

| Paired LAS&T artifact | PRL20 Step 8 SHA-256 | PRL20 Step 16 SHA-256 |
| --- | --- | --- |
| `score-strict.json` | `bcc89496e8602e49541a406d834c3f3711e57422383de86d44d1bc67637204ed` | `47b65dcff255367dcb4a30f2cda843d6edfff275538685175c0ff094b10aabd6` |
| `completion-audit.json` | `9cf483716dcc14abc2e04f6d2d21ef8f76070dd77787a6937e1a0c40521ac5c9` | `5efc157a5c510358d71284bc966d22a7019f5abfe212dd5219bdf5b48c10ecf9` |

Each full-suite policy completion audit locks and reparses all four rank files, requires
exact task coverage and rank assignment, recomputes canonical row identities,
and verifies every saved task, manifest, evaluation, policy, and rank hash.
Both full-suite audits report `complete: true`, `missing_count: 0`, and
`duplicate_count: 0`.

There is one deliberate limit to the durable audit surface. At every TGVF RPC
boundary, runtime code fail-closes on the conditioning `provider` and, for
PRL20, `preprocessed_visual_sha256`; the full observation also contributes to
`trajectory_sha256`. Rank JSONL rows retain that trajectory checksum and the
observation count, but do not project every observation field. The offline
completion audit can therefore verify saved row identity and checksums, but
cannot independently reconstruct and recheck the provider or preprocessed
visual hash for every RPC. This is a persistence-granularity limitation, not a
runtime validation failure.

The completion summary is at
`runtime/original-crop-step16-supervisor-v2/final/supervisor-summary.json`
beneath that output root and records `complete: true`. Independent final audit
reparsed every JSONL row, checked global ordinal and result-identity uniqueness,
verified `ordinal % 4`, recomputed canonical row identities and all referenced
artifact hashes, and found no blocking discrepancy.

The final summary was published at 12:10:59 JST after interrupted supervisor
finalization was resumed idempotently; this did not rerun or alter inference.

The Original arm publishes one merged `results.jsonl`. Crop scoring binds its
four immutable rank JSONLs directly; there is intentionally no top-level
merged Crop result file.

An earlier v1 launch wrote zero result rows and stopped during vLLM startup
because its durable temporary directory made the ZeroMQ Unix-socket path
longer than Linux's 107-byte limit. No output from that attempt was accepted.
The supervisor now uses short private `/tmp/t2a-*` paths; all numeric
Original/Crop results in this document come from the successful v2 run
described above.

For that completed baseline, the relevant targeted CPU regression was
`107 passed, 1 deselected, 0 failed` with two deprecation warnings. The one
deselected test requires retired external step-80 snapshot directories and is
unrelated to this benchmark. Ruff lint, Ruff formatting, shell syntax, JSON
syntax, and `git diff --check` passed for that accepted baseline revision.

For the final four-arm closure, the current texture, policy-benchmark,
completion-audit, preparation, runner, scoring, worker-failure, and supervisor
test selection reports `65 passed, 0 failed` with two deprecation warnings.
Final artifact-to-document checks and `git diff --check` also pass.

### Interpretation and limitations

- All four step-16 arms are complete and identity-audited on the same task
  manifest and visual contract. No partial-run metric appears in the tables.
- LAS&T uses a deterministic project-derived population and metric, not a
  byte-for-byte reproduction of the upstream API evaluator or paper table.
- MMAD uses the official dataset/task aggregation, but the one-shot pair is a
  single left/right canvas. This report emphasizes the upstream-like
  permissive reparse while retaining the predeclared strict score artifact as
  a compliance diagnostic.
- The 262,144-pixel setting is a processor area cap, not a literal square
  resize. It is shared across all four arms and all images.
- Strict valid-only scores diagnose formatting but cannot be interpreted as
  unbiased full-set accuracy. The main permissive MMAD view instead keeps the
  complete fixed denominator.
- Overall micro accuracy is a diagnostic dominated by MMAD's 39,670 rows; use
  the separate benchmark headline metrics for comparisons.

## Commands

Run from the repository root. Dataset preparation only needs the lightweight
Python 3.11/3.12 environment below. Model evaluation needs the project's full
environment, including vLLM, Ray, and the local verl checkout. The system
`python` is 3.10 and is not supported. Each tool bootstraps this worktree's
local `src` tree and prints a JSON summary.

Prepare the two default full manifests and combine them:

```bash
TEXTURE_DATA_PY=/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py last
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py mmad
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py suite
```

The default outputs are:

```text
/nvmesv/dredvpn009/datasets/benchmarks/texture_evaluation_v1/last-paper400-neutral-v1/
/nvmesv/dredvpn009/datasets/benchmarks/texture_evaluation_v1/mmad-1shot-random/
/nvmesv/dredvpn009/datasets/benchmarks/texture_evaluation_v1/suite-last-paper400-mmad-1shot-random/
```

Validated local prepared sizes are 4,396,300,140 logical bytes for LAS&T
(3,200 generated PNGs), 5,566,983,102 bytes for MMAD one-shot (8,366
content-addressed canvases), 69,015,571 bytes for the MMAD zero-shot manifest,
and 91,258,398 bytes for the combined suite manifest. The suite references the
component images and does not duplicate them.

Each directory contains canonical `tasks.jsonl` and `identity.json`. MMAD
one-shot output also contains content-addressed canvases; LAS&T output contains
the generated quiz images. Manifest and identity writes are create-or-identical:
repeating the same command is safe, while attempting to replace an existing
artifact with different bytes fails.

Prepare explicit eight-row component smoke tests and a two-by-eight suite:

```bash
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py last \
  --prefix 8 \
  --output /tmp/texture-bench-smoke/last
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py mmad \
  --shot 1 \
  --template-kind random \
  --prefix 8 \
  --output /tmp/texture-bench-smoke/mmad
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py suite \
  --last-manifest /tmp/texture-bench-smoke/last/tasks.jsonl \
  --mmad-manifest /tmp/texture-bench-smoke/mmad/tasks.jsonl \
  --prefix 8 \
  --output /tmp/texture-bench-smoke/suite
```

Prepare the full zero-shot MMAD alternative:

```bash
"$TEXTURE_DATA_PY" tools/prepare_texture_benchmarks.py mmad --shot 0
```

Use `--source` and `--output` to override source or destination roots. The
`--allow-unpinned-source` MMAD option exists only for development fixtures; a
reported benchmark should retain the default pinned-source verification.

## Four-arm execution and scoring

A matrix JSON binds one exact combined `tasks.jsonl`, one output root, and the
shared visual contract. It may stage any non-empty subset with at most one arm
of each kind, so available arms can run before every trained snapshot exists.
A final four-way comparison must contain exactly `original`, `crop`, `tgvf`,
and atomic `tgvf_crop`; pass `--require-complete-arms` during final validation.
Every model or policy path must be absolute. `gpu_ids` is an explicit ordered
four-GPU physical map (default `[0, 1, 2, 3]`); the emitted materializer and
worker commands use that same map.
Tool arms bind an exact policy config, optimizer step, and one of: a LoRA
pointer; a full-model snapshot manifest plus materialization receipt; or the
current paired full-Qwen/RP66 closure (`paired_qwen_model_path`, optional
step-zero/required-nonzero `paired_rp66_pointer_path`, and a destination
`paired_snapshot_receipt_path`).

### Evaluated and staged model closures

As of 2026-08-15, exact closures and complete texture results exist for all
four pipeline kinds under the same task and 512-area-cap contract:

- `original`: `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct` (34 files,
  17,545,917,781 bytes, tree SHA-256
  `d4c62edf0fa6622ca511a6baa3b75b6314b8d702004f92556c46ca59dbaf8d73`).
- `crop`: this report evaluated canonical PRL14 clean-final step 16 under
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/step16/runtime/`.
  Its source checkpoint is
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-cleanfinal-16step-ws8/checkpoints/global_step_16/actor/huggingface`.
  Use `frozen-policy-config.toml` together with
  `frozen-full-model-state/snapshot-manifest.json` and
  `frozen-full-model-state/materialization-receipt.json`; set optimizer step 16
  and protocol `deepeyes_official_visible_native_crop_v1`.
- `tgvf`: PRL17-R2 RP67 T-free paired-seed step 16 under
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`.
  Use `frozen-policy-config.toml`, `qwen-only-bundle/model`, and
  `rp66-state/step-00000016-pointer.json`; set optimizer step 16 and protocol
  `training_run`. Its `tgvf_only` profile exposes only `tgvf_focus_tool`, with
  at most six calls. The source run identity is
  `9c64b840e540fccd98bab30e2f05783b15f43dddcc0e926a1778e713a1f74746`.
  The paired receipt path is a destination beneath the matrix output rather
  than a pre-existing input.
- `tgvf_crop`: PRL20-R0 RP67 T-free paired-seed step 16 under
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/evaluation/PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`.
  Use `frozen-policy-config.toml`, `qwen-only-bundle/model`, and
  `rp66-state/step-00000016-pointer.json`; set optimizer step 16 and protocol
  `training_run`. Its `crop_tgvf` profile exposes only the atomic
  `tgvf_crop_tool`, with at most six calls. The source run identity is
  `190c3db105c5e35d8d8aab4ad837409b4b39b8ce7f22e7b2393249a5e1898457`.
  Its paired summary and `evaluation-complete` marker close the source CoreDev
  evaluation; the texture run uses a new destination receipt and identity.

The PRL14 step-8 Crop closure under the corresponding `step8/runtime/`
directory remains staged in the general-purpose three-arm matrix but is not
part of the completed result table. The old matrix's PRL17 output is also
excluded because it was materialized with the generic benchmark prompt rather
than `build_tgvf_visual_messages`. The step-16 Crop snapshot's frozen base
contract retains a historical PRL13 `run_id`.
This is a non-blocking provenance label mismatch: the source checkpoint path,
optimizer step, snapshot and weight identities, materialization receipt,
model tree, and evaluation identity all bind the PRL14 step-16 closure used in
the run.

The PRL17 execution matrix is
`configs/evaluation/texture_last_mmad_prl17_tgvf_step16_corrected_prompt_512_v1.json`,
with identity
`feb796482ead0c8893782ac6070b8629ae882d1ff14ebfc4648fd0271c567700`.
Its TGVF outputs are rooted at
`/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-PRL17-TGVF-STEP16-CORRECTED-PROMPT-512-V1/tgvf-prl17-r2-rp67-step16-corrected-prompt`.
The PRL20 one-arm matrix is
`configs/evaluation/texture_last_mmad_prl20_crop_tgvf_step16_512_v1.json`, with
identity
`25afd59c8602b1bd00751513fd6acb4ae79bd8571fa78fda2b68ec4203751ca1`.
Its outputs are rooted at
`/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-PRL20-CROP-TGVF-STEP16-512-V1/tgvf-crop-prl20-r0-rp67-step16`.
Both matrices bind the full 42,870-task manifest and `max_pixels=262144`.

Set the full evaluation environment once per shell:

```bash
TEXTURE_EVAL_PY=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
TEXTURE_EVAL_PYTHONPATH="$PWD/src:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl"
```

Validate the matrix, including all task images:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/run_texture_benchmark.py validate \
  --matrix /absolute/path/to/texture-matrix.json
```

Add `--require-complete-arms` for the publishable four-way readiness gate. A
staged matrix reports its missing kinds in `missing_pipeline_kinds`.

Run the stock-Qwen original arm:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/run_texture_benchmark.py original \
  --matrix /absolute/path/to/texture-matrix.json \
  --batch-size 8
```

The runner hashes the complete local model tree, uses deterministic
content-addressed request seeds, and writes immutable `results.jsonl` and
`run-identity.json`. It does not load any visual-tool policy code. For a long
run, use one durable worker per GPU instead. Tasks are assigned by
`ordinal % world_size`; each completed batch is appended and `fsync`ed to
`inference/rank-N.jsonl`. A rank lock prevents two processes from owning the
same shard, and restart validates every existing row before skipping it.

The following example keeps the matrix's policy arms on GPUs 0--3 while
placing original on physical GPUs 4--7. The same `--gpu-ids`, batch size,
token limit, and engine kwargs are part of the execution identity and must be
passed to workers, status, and finalization:

```bash
for rank in 0 1 2 3; do
  gpu=$((rank + 4))
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
    tools/run_texture_benchmark.py original \
    --matrix /absolute/path/to/texture-matrix.json \
    --rank "$rank" --world-size 4 --gpu-ids 4 5 6 7 \
    --no-verify-images &
done
wait

PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/run_texture_benchmark.py original-status \
  --matrix /absolute/path/to/texture-matrix.json \
  --world-size 4 --gpu-ids 4 5 6 7 --no-verify-images

PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/run_texture_benchmark.py original-finalize \
  --matrix /absolute/path/to/texture-matrix.json \
  --world-size 4 --gpu-ids 4 5 6 7 --no-verify-images
```

`original-finalize` refuses partial, duplicate, moved, or identity-drifted
rows, then publishes the same immutable `results.jsonl`/`run-identity.json`
boundary as the legacy single-process mode. The stock runner owns
`mm_encoder_attn_backend=TORCH_SDPA`: this avoids the non-portable bundled
FlashAttention ViT path on Blackwell while leaving decoder attention under
vLLM's GPU-specific selection. It cannot be overridden through engine kwargs.
The `--no-verify-images` optimization above is appropriate only because the
preceding matrix validation already checked every image byte; workers still
bind and validate each task's recorded image identity while loading it.

For a tool arm, emit argv-safe commands that bind its snapshot to the same
task manifest, pass `--image-max-pixels 262144`, prepare/validate the frozen
evaluation, and launch four rank workers:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/run_texture_benchmark.py policy-command \
  --matrix /absolute/path/to/texture-matrix.json \
  --arm tgvf_crop
```

`tgvf_crop` is one atomic sampled action: the trusted client crops the
immutable source, preprocesses that exact crop under the shared pixel cap,
then a single colocated worker RPC materializes the crop visual state and runs
the TGVF Adapter on it. Source/crop hashes, source-space box, dimensions,
sampled target tokens, Adapter output, and the complete observation record are
bound together. It is not reported as two sequential independent calls.

After every policy worker has exited successfully, publish the exact-coverage
audit before scoring:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/audit_texture_policy_completion.py \
  --config /absolute/path/to/policy-benchmark-config.json \
  --world-size 4 \
  --output /absolute/path/to/evaluation/completion-audit.json
```

The audit refuses an incomplete, duplicate, mis-sharded, malformed, or
identity-drifted result set and records the immutable rank hashes used by both
score artifacts.

Score a complete result set locally:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/score_texture_benchmark.py \
  --tasks /absolute/path/to/tasks.jsonl \
  --results /absolute/path/to/results.jsonl \
  --output /absolute/path/to/score.json
```

Publish the report's main MMAD view as a separate, provenance-bound artifact:

```bash
PYTHONPATH="$TEXTURE_EVAL_PYTHONPATH" "$TEXTURE_EVAL_PY" \
  tools/score_texture_benchmark.py \
  --tasks /absolute/path/to/tasks.jsonl \
  --results /absolute/path/to/results.jsonl \
  --mmad-parser upstream-like-permissive \
  --output /absolute/path/to/score-permissive.json
```

The default strict parser counts missing or ambiguous answers as wrong. LAS&T
uses the project-defined `four_condition_macro_accuracy`, plus
physical-directory and micro diagnostics; the upstream paper does not define
this aggregate as an official headline metric. MMAD merges `DS-MVTec` into
`MVTec-AD`,
merges Object Structure/Details into Object Analysis, uses balanced accuracy
for anomaly detection, macros over its seven tasks within each dataset, and
then macros over the four datasets. The upstream-like permissive/fuzzy parser
is an explicit read-only reparse and is the main MMAD interpretation used in
this report. It writes a separate `score-permissive.json` and never mutates the
saved strict score or result rows. The score embeds the parser contract and its
identity; a report must not infer that contract from the filename alone. Every
result row must bind both the exact `sample_id` and task-manifest SHA-256;
ordinal-only or cross-manifest result files are rejected.

MMAD is pinned to the annotation SHA listed above. A newer local Git checkout
has the same 8,366 images and 39,670 questions but changes 23 question records,
including two gold labels, so this suite must not be described as using the
latest annotation. Its one-shot left/right canvas, strict parser, and 262,144
total-area limit are deliberate single-image/project adaptations; the dataset
merges, seven-task means, four-dataset mean, and balanced anomaly-detection
cell follow the pinned official scorer.
