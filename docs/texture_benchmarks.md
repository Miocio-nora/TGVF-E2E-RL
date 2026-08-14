# LAS&T and MMAD texture benchmarks

This is the canonical project document for the LAS&T 2D Texture Retrieval and
MMAD benchmarks. It explains what each benchmark measures, records the exact
local dataset and evaluation protocol, reports the completed 2026-08-13
Original-versus-Crop step-16 baseline, tracks the in-progress 2026-08-14 TGVF
and atomic Crop+TGVF step-16 extension, and provides the commands and artifact
identities needed to reproduce or extend it.

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

## Four-arm evaluation: completed baseline and in-progress extension

### Scope and immutable configuration

The successful v2 inference run launched at 2026-08-13 06:52:36 JST. Original
finished at 08:18:06 and Crop finished at 11:34:58. It compared stock
Qwen3-VL-8B-Instruct with the PRL14 clean-final Crop policy at optimizer step
16 on every task in the combined suite.

On 2026-08-14, two additional optimizer-step-16 policy arms were bound to the
same immutable 42,870-task manifest and visual-processing contract: PRL17-R2
TGVF-only and PRL20-R0 atomic Crop+TGVF. Their four-rank evaluations are in
progress. All cells marked `pending run completion` below are intentional and
must be replaced only after complete rank coverage, scoring, and final identity
audit; they do not denote zero accuracy or a failed run.

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
| PRL17-R2 TGVF step 16 | `configs/evaluation/texture_last_mmad_current_3arm_512_v1.json` | `23f2a9850518e6a0774b8767c221f102b62c1321f43e9618335a8875d9ada156` | B200 0-3 | In progress |
| PRL20-R0 Crop+TGVF step 16 | `configs/evaluation/texture_last_mmad_prl20_crop_tgvf_step16_512_v1.json` | `25afd59c8602b1bd00751513fd6acb4ae79bd8571fa78fda2b68ec4203751ca1` | B200 4-7 | In progress |

Arm closures:

| Arm | Exact closure | Evaluation protocol and tool boundary |
| --- | --- | --- |
| Original | `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct`; model-tree SHA-256 `d4c62edf0fa6622ca511a6baa3b75b6314b8d702004f92556c46ca59dbaf8d73` | Stock one-pass generation; temperature 0; batch 8; at most 2,048 output tokens; content-addressed per-sample seeds |
| Crop step 16 | PRL14 clean-final full-model snapshot; policy-weights SHA-256 `50f5d9dd7ecdbf8d9baf46c00c13b1c3719de37b09f5aa91c40aabc758e06beb`; snapshot identity `d813b7511a35751380469516d58d2b55b1b1a933d86229210e92d95b6d7a928d` | `deepeyes_official_visible_native_crop_v1`; Crop-only; at most 6 calls; concurrency 8 per GPU |
| TGVF PRL17-R2 step 16 | Paired full-Qwen/RP66 closure under `artifacts/policy/PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`; run identity `9c64b840e540fccd98bab30e2f05783b15f43dddcc0e926a1778e713a1f74746`; destination snapshot identity `71e5866848dfd17cd74527cf7929adfaf8089a6e948567677a6849277ebe8b6f` | `training_run`; tool profile `tgvf_only`; only `tgvf_focus_tool`; at most 6 calls; concurrency 8 per GPU |
| Crop+TGVF PRL20-R0 step 16 | Paired full-Qwen/RP66 closure under `artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/evaluation/PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/step16/runtime/`; run identity `190c3db105c5e35d8d8aab4ad837409b4b39b8ce7f22e7b2393249a5e1898457`; destination snapshot identity `6467170e1c01ce1d03357270913ed10e5ddcfb63c1ea3226557c04b8871c6345` | `training_run`; tool profile `crop_tgvf`; only atomic `tgvf_crop_tool`; at most 6 calls; concurrency 8 per GPU |

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
| LAS&T four-condition macro | **72.4167%** | **68.2708%** | pending run completion | pending run completion | -4.1458 pp |
| LAS&T micro | 71.2500% | 65.6562% | pending run completion | pending run completion | -5.5938 pp |
| MMAD official aggregation structure, permissive fixed-denominator reparse | **69.2635%** | **67.9939%** | pending run completion | pending run completion | -1.2696 pp |
| MMAD micro, permissive fixed-denominator reparse | 69.4404% | 68.0691% | pending run completion | pending run completion | -1.3713 pp |
| MMAD still unparsed | 0 / 39,670 | 107 / 39,670 | pending run completion | pending run completion | +107 |

The completed baseline result is therefore much closer than the strict artifact
initially suggests: Crop trails Original by 1.2696 percentage points on the
MMAD macro after permissive answer extraction. The larger remaining degradation
is on LAS&T, where Crop trails by 4.1458 points on the four-condition macro. No
four-way conclusion is made until both in-progress arms are complete.

LAS&T condition breakdown:

| Canonical condition | Samples | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 | Original invalid | Crop invalid | TGVF invalid | Crop+TGVF invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Different shape, black background | 400 | 75.5000% | 77.2500% | pending run completion | pending run completion | 3 | 30 | pending run completion | pending run completion |
| Different shape, textured background | 800 | 60.0000% | 47.1250% | pending run completion | pending run completion | 11 | 186 | pending run completion | pending run completion |
| Same shape, black background | 800 | 88.0000% | 92.3750% | pending run completion | pending run completion | 2 | 20 | pending run completion | pending run completion |
| Same shape, textured background | 1,200 | 66.1667% | 56.3333% | pending run completion | pending run completion | 13 | 248 | pending run completion | pending run completion |

MMAD per-dataset task macro under the permissive fixed-denominator reparse:

| MMAD dataset | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 |
| --- | ---: | ---: | ---: | ---: |
| GoodsAD | 64.9218% | 64.7045% | pending run completion | pending run completion |
| MVTec-AD (including DS-MVTec) | 80.8414% | 75.1589% | pending run completion | pending run completion |
| MVTec-LOCO | 62.9856% | 63.4273% | pending run completion | pending run completion |
| VisA | 68.3052% | 68.6847% | pending run completion | pending run completion |

### Strict-parser compliance diagnostic

The immutable `score.json` artifacts use the project's predeclared strict
parser. Those numbers remain useful as a protocol-compliance diagnostic, but
they are not the main accuracy view in this report:

| Strict diagnostic | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 | Crop - Original |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAS&T invalid | 29 / 3,200 | 484 / 3,200 | pending run completion | pending run completion | +455 |
| LAS&T valid-only micro | 71.9016% | 77.3564% | pending run completion | pending run completion | +5.4548 pp |
| MMAD official aggregation structure | 69.2624% | 32.7951% | pending run completion | pending run completion | -36.4673 pp |
| MMAD micro | 69.4379% | 31.4898% | pending run completion | pending run completion | -37.9481 pp |
| MMAD invalid | 4 / 39,670 | 21,600 / 39,670 | pending run completion | pending run completion | +21,596 |
| MMAD valid-only micro | 69.4449% | 69.1312% | pending run completion | pending run completion | -0.3137 pp |
| Overall strict micro | 69.5731% | 34.0401% | pending run completion | pending run completion | -35.5330 pp |
| Overall invalid | 33 / 42,870 | 22,084 / 42,870 | pending run completion | pending run completion | +22,051 |

The 32.7951% Crop strict value is dominated by answer-format noncompliance, not by
42,870 missing or failed inferences. The policy commonly returns a choice plus
option text, for example `B. Yes.` or `C. Irregular, jagged edges.`. The strict
parser accepts a bare terminal choice, a supported answer wrapper, or one
unambiguous explicit answer marker; it deliberately does not guess a choice
from arbitrary prose. As a result, 21,493 MMAD rows are
`ambiguous_or_unmatched` and 107 have no final answer after reaching the tool
call cap.

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
the saved strict score is retained for audit even though the report emphasizes
the permissive view.

### Runtime and agent behavior

| Runtime observation | Original | Crop step 16 | TGVF PRL17 step 16 | Crop+TGVF PRL20 step 16 |
| --- | ---: | ---: | ---: | ---: |
| Four-GPU wall time through last rank | 1 h 25 min 30 s | 4 h 42 min 22 s | pending run completion | pending run completion |
| Model interaction | One pass | Multi-turn agent loop | Multi-turn agent loop | Multi-turn atomic-tool agent loop |
| Mean tool calls per task | 0 | 1.3077 overall | pending run completion | pending run completion |
| Mean assistant turns per task | 1 | 2.3103 overall | pending run completion | pending run completion |

LAS&T was the expensive part of Crop inference: it averaged 3.5775 crop calls
and 4.5913 assistant turns per question, with median per-row wall time 45.94 s
and P95 89.13 s. MMAD averaged 1.1246 calls and 2.1263 turns, with median 3.50 s
and P95 9.66 s. In contrast, 42,864 of 42,870 Original rows generated only one
choice token plus the stop token.

Crop recorded 112 recoverable out-of-range crop attempts across 75 rows and
130 `tool_call_cap` stops. These are model/tool-protocol outcomes, not worker
or engine failures; every row was durably recorded and scored.

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

The two completed permissive artifacts are, respectively:

```text
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-ORIGINAL-CROP-PRL14-STEP16-512-V1/original-qwen3-vl-8b-instruct/score-permissive.json
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-ORIGINAL-CROP-PRL14-STEP16-512-V1/crop-prl14-cleanfinal-step16/score-permissive.json
```

They are schema-v2, provenance-bound reparses of the immutable result rows and
both carry the permissive parser-contract identity listed above. The original
strict `score.json` files remain immutable schema-v1 evidence. The TGVF and
Crop+TGVF result, score, rank-file, and completion hashes remain pending until
their runs finish; no partial hash is promoted here.

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

### Interpretation and limitations

- Original versus Crop step 16 is complete. The exact trained PRL17 TGVF-only
  and PRL20 atomic Crop+TGVF step-16 closures are available and their full-suite
  evaluations are in progress; pending cells are not yet scientific results.
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

As of 2026-08-14, exact closures exist for all four pipeline kinds. Original and
Crop have complete texture results; PRL17 TGVF and PRL20 atomic Crop+TGVF are
materialized against the same task and 512-area-cap contract and are being
evaluated:

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
part of the completed result table. Only that matrix's PRL17 TGVF arm is used
for the in-progress TGVF row. The step-16 Crop snapshot's frozen base contract
retains a historical PRL13 `run_id`.
This is a non-blocking provenance label mismatch: the source checkpoint path,
optimizer step, snapshot and weight identities, materialization receipt,
model tree, and evaluation identity all bind the PRL14 step-16 closure used in
the run.

The PRL17 execution matrix is
`configs/evaluation/texture_last_mmad_current_3arm_512_v1.json`, with identity
`23f2a9850518e6a0774b8767c221f102b62c1321f43e9618335a8875d9ada156`.
Its TGVF outputs are rooted at
`/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/TEXTURE-LAST-MMAD-CURRENT-3ARM-512-V1/tgvf-prl17-r2-rp67-step16`.
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
