# Isolated answer-utility experiment

This package is a removable Stage1 research branch. It does not modify the
accepted representation-training config, objective, streaming, trainer, or
checkpoint schemas. Its artifacts are intentionally private and cannot be
loaded as production Stage1 artifacts without an explicit promotion step.

## Frozen experiment grid

| Cell | Answer context | Answer loss | Zero/wrong comparison | Interpretation |
|---|---|---:|---:|---|
| E0 | none | no | no | Existing RP66 objective; evaluation-only |
| E0-continuation | none | no | no | Trainable matched-budget RP66 continuation |
| E1 | source + D + gold evidence | yes | no | Leakage/bypass diagnostic |
| E2 | fresh D-only | yes | no | Clean absolute answer supervision |
| E3 | source + D + gold evidence | yes | yes | Leakage diagnostic with controls |
| E4-v1 | fresh D-only | yes | yes | Primary causal utility experiment |

The table describes the separate answer branch. E1--E4 all retain the existing
gold-evidence Matrix/evidence/norm branch as an auxiliary; E4 therefore tests
the requested combination of the existing evidence transcript, a clean answer
branch, and correct/zero/wrong comparisons without putting gold evidence in the
answer context.

The clean branch contains question, oracle target/tool call, one focused-D
tool response, and the short answer. It contains no source-image tokens, source
DeepStack, teacher evidence, or image-seen KV cache. Labels own exactly the
short-answer tokens and final native `<|im_end|>` token.

E4-v1 uses two controls while holding the native text/M-RoPE layout fixed:

- an atomic all-zero main-D plus DeepStack 8/16/24 bundle;
- a real D from another target in the same image group whose normalized
  `short_answer` differs from the current answer.

The second rule prevents the counterfactual loss from penalizing a D that
supports the same answer. All planned E3/E4 groups are checked before GPU
allocation. Cross-image same-target grounding negatives are deliberately not
claimed by v1 and require a later, separately named schema.

`E0-continuation` is intentionally distinct from E0. It warm-starts the same
RP66 Adapter but trains the unchanged Matrix/evidence/norm objective with the
same data, batch, learning rate, and optimizer-step budget as a formal answer-
utility cell. This avoids comparing an updated E2/E4 Adapter against an
unequally trained E0 checkpoint.

Formal matched-budget cells share an answer-safe sampler. A raw same-image K4
batch is deterministically skipped when all four normalized short answers are
identical, because no row in that batch can receive a valid different-answer
wrong D. The inherited sampler cursor records skipped batches, so exact resume
and the E0-continuation/E2/E4 accepted sample sequence remain identical.

## RP66 smoke settings

The checked-in run sidecars warm-start the final Qwen3-VL-8B-Instruct RP66
Adapter at step 2000. Qwen, vision, mergers, and decoder remain frozen; only
Adapter-owned parameters are optimized. The starting smoke uses:

- one visible GPU;
- 4 rows per same-image group and 4 accumulation groups, or 16 rows/update;
- constant AdamW learning rate `1e-6`;
- 80 optimizer steps;
- checkpoints every 5 steps and a checkpoint at every requested stop boundary.

## RP66 matched 500-step settings

The formal E0-continuation/E2/E4 sidecars all load the exact RP66 step-2000
Adapter, then create a fresh AdamW optimizer at constant learning rate `1e-6`.
They use one B200, 16 rows/update, 500 new optimizer steps, and durable
checkpoints every 25 steps. The private artifact records experiment step 500;
it must not be described as a production RP66 step-2500 checkpoint because the
RP66 optimizer and cosine-scheduler states are not continued.

Run a no-GPU validation first:

```bash
.venv312/bin/python tools/run_representation_answer_utility.py \
  --config configs/representation/experiments/answer_utility/rp66_e4_clean_answer_counterfactual_v1_run_gpu0.toml \
  --validate-only
```

Run only through an optimizer boundary, for example step 5:

```bash
.venv312/bin/python tools/run_representation_answer_utility.py \
  --config configs/representation/experiments/answer_utility/rp66_e4_clean_answer_counterfactual_v1_run_gpu0.toml \
  --stop-after-global-step 5
```

Resume from the latest private checkpoint:

```bash
.venv312/bin/python tools/run_representation_answer_utility.py \
  --config configs/representation/experiments/answer_utility/rp66_e4_clean_answer_counterfactual_v1_run_gpu0.toml \
  --resume-checkpoint /absolute/output/checkpoints/answer-utility-step-00000005.pt \
  --stop-after-global-step 10
```

Resume validates the source/config/code/environment identities, private state
digests, checkpoint filename and step, latest-checkpoint ownership, metrics
history, Adapter/optimizer/sampler/RNG state, and output isolation before the
8B model is loaded. The launcher re-executes once so CUDA visibility and Python
hash determinism are fixed at interpreter startup.

## Removal boundary

If the experiment is rejected, remove only:

- `src/tgvf_rl/representation/experiments/answer_utility/`;
- `configs/representation/experiments/answer_utility/`;
- `tests/representation/experiments/answer_utility/`;
- `tools/run_representation_answer_utility.py`;
- `artifacts/representation_experiments/answer_utility/`.

No production Stage1 or RL module needs a revert.
