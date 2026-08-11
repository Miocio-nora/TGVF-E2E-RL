# PRL15 R1 step-0/step-8 paired evaluation handoff

This is the post-training evaluator for the active four-rank mathematical-
equivalence PRL15 lineage.  It is deliberately a separate tmux watcher rather
than an in-training validation callback: evaluation starts only after the
durable step-8 Qwen/RP66 pair exists and the training process has released the
requested GPUs.

## Bound pair and protocol

The immutable plan is
`configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json`.
It binds the active
`prl_15_r1_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_math_equiv_8step_ws4.toml`
run contract and compares:

- step 0: base Qwen3-VL-8B-Instruct plus the immutable stage-1 RP66 state;
- step 8: the Qwen parameters reconstructed from the four
  `global_step_8/actor/model_world_size_4_rank_*.pt` FSDP shards, plus the
  content-addressed RP66 step-8 state referenced by
  `latest-lora-snapshot.json`.

Both arms use the training-run TGVF prompt, schema, six-call limit, sampling
contract, tasks, and canonical four-rank partition.  A Qwen-only or RP66-only
state is rejected by the paired snapshot backend.

The FSDP shards contain two parameter namespaces: the trainable full Qwen
state and `tgvf_adapter.*`.  The embedded `actor/huggingface` directory holds
configuration/tokenizer files, not merged model weights.  Before step-8
inference, the evaluator therefore wraps the pinned veRL FSDP merger, verifies
that every tensor belongs to either the exact base-Qwen key closure or the
exact RP66 snapshot closure, removes only `tgvf_adapter.*`, and atomically
writes `step8/runtime/qwen-only-bundle/model/`.  RP66 is then loaded once from
its separate same-step snapshot.  The bundle receipt makes this CPU-side merge
resumable and prevents accidental double embedding of RP66.

## CoreDev coverage and scorer

The source manifest remains the canonical seven-source CoreDev-2511 snapshot,
but the current TGVF schema has no multi-image selector.  The executable
common-support tranche is therefore exactly 2,240 rows:

| Slice | Evaluated single-image rows | Held multi-image rows |
|---|---:|---:|
| VStarBench | 191 | 0 |
| HRBench4K | 200 | 0 |
| BLINK | 180 | 240 |
| OCRBench_v2 | 600 | 0 |
| MMMU-Pro 10-choice | 269 | 31 |
| MathVista MINI | 300 | 0 |
| MathVerse MINI | 500 | 0 |
| **Total** | **2,240** | **271** |

The 271 rows are an explicit unsupported hold.  They are never evaluated by
taking the first image or by silently compositing images.  The official
VLMEvalKit views retain all 2,511 source rows and use fail-closed invalid
sentinels for held rows so source indices and official scorer behavior remain
auditable.  Any BLINK/MMMU method comparison must report the common supported
single-image tranche; the paired summary separately records 2,240 evaluated
and 271 held rows.

Scoring follows the accepted PRL14 route:

1. materialize seven immutable official scoring views from the 2,240 durable
   trajectory records and pinned CoreDev source TSVs;
2. run pinned VLMEvalKit commit `7055d301...` for all seven datasets;
3. use the local BF16 Qwen2.5-72B-Instruct judge, TP2, at the pinned endpoint;
4. fail closed on judge unavailability, exhausted calls, random/exact-match
   fallback, missing rows, or missing judge evidence.  GPT fallback is
   forbidden.

The old generic MCQ scorer is not valid here: the official-visible task
manifest intentionally omits gold answers/options and includes 271 multi-image
rows.  The paired executor uses `materialize_policy_coredev_scoring.py` and the
official seven-suite scorer instead.

## Automatic tmux watcher

The watcher may be started before training finishes.  It first waits for the
step-8 tracker, complete FSDP model shards, paired checkpoint markers, and
RP66 pointer, then requires two consecutive free-memory polls on the requested
GPUs before constructing either vLLM arm.  No `CUDA_VISIBLE_DEVICES` value
should be set on the watcher itself; each inference worker and the judge
receive an explicit physical binding.

This command is the matched scientific evaluation, not the cheap code-function
smoke.  New implementation changes should first pass a tiny, low-cost smoke;
only then should this complete step-0/step-8 matched protocol be scheduled.
For the current allocation, request GPUs 0--3 only so unrelated work on 4--7
cannot block the watcher.

```bash
cd /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-rp66-matched

tmux -L prl15_r1_eval new-session -d -s paired_eval \
  "env \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 \
    PYTHONPATH=$PWD/src:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl \
    /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python \
    tools/run_prl15_paired_evaluation.py \
      --plan configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json \
      --mode run \
      --wait-for-step8 \
      --wait-for-gpus \
      --gpu-ids 0 1 2 3 \
    > .runlogs/prl15_r1_step0_step8_paired_eval.log 2>&1"
```

With eight GPUs, step 0 uses 0--3 and step 8 uses 4--7 concurrently.  After
all eight inference workers exit, the pinned judge uses GPUs 2/3 and the two
arms are scored sequentially against the same live judge service.  With four
GPUs the two inference arms run sequentially; the selected set must include
the pinned judge devices 2/3.

Historical same-protocol runs imply approximately 40--50 minutes for both
inference arms in parallel and 10--25 minutes for materialization/judging, or
about 55--75 minutes total on eight GPUs.  Four-GPU execution runs the two arms
sequentially.  Including the one-time CPU FSDP-to-HF merge, budget about
2 hours to 2 hours 25 minutes until the paired summary; this merge allowance
is an estimate until measured on the final step-8 shards.  Durable rank JSONLs,
the reusable Qwen-only bundle, immutable paired snapshot receipts, VLMEvalKit
`--reuse`, and accepted arm summaries avoid repeating completed work.

The final output is below the R1 training root at
`evaluation/PRL15-R1-RP66-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-V1/` and must
contain `paired-summary.json` with accepted step-0/step-8 summaries and the
explicit 2,240/271 coverage boundary.
