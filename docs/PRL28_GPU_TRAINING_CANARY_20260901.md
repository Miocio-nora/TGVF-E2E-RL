# PRL28 GPU training canary

Status: **prepared but not run**. On 2026-09-01 all eight local B200 GPUs were
owned by another active job, and the user deferred GPU execution. No training,
checkpoint-resume, or evaluation result is claimed by this document.

The bounded canary covers Crop and Atomic at 512 squared pixels with BS4,
`n=2`, WS4, and an S1 teardown followed by an integrity-bound S1-to-S2 resume.
It uses the local Qwen2.5-72B answer judge and does not call a paid API.

## Inputs

- Crop: `configs/policy/runs/prl_28_gpu_canary_b_crop_pixel512_s1to2_bs4_n2_teacher25_ws4.toml`
- Atomic: `configs/policy/runs/prl_28_gpu_canary_e_atomic_pixel512_s1to2_bs4_n2_teacher25_ws4.toml`
- Training-runtime implementation: `89dfd52decece90cbb2d3923cc9dc65c8eb6f0d4`
- Judge placement: physical GPUs 0-1, port 8013
- Policy placement: physical GPUs 2-5, exposed to veRL as logical GPUs 0-3

Do not start while any selected GPU is owned by another process.

## S1 launch

```bash
CANARY_REPO_ROOT=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
CANARY_PYTHON="$CANARY_REPO_ROOT/.venv312/bin/python"
CANARY_CONFIG="$CANARY_REPO_ROOT/configs/policy/runs/prl_28_gpu_canary_b_crop_pixel512_s1to2_bs4_n2_teacher25_ws4.toml"
export PYTHONPATH="$CANARY_REPO_ROOT/src"

"$CANARY_PYTHON" -m tgvf_rl.cli validate-policy-config "$CANARY_CONFIG"
"$CANARY_PYTHON" -m tgvf_rl.cli run-policy "$CANARY_CONFIG"
```

The Atomic lane uses the corresponding Atomic config above. Each base config
stops at S1, writes checkpoints 0 and 1, uses console-only logging, and leaves
the scheduler horizon at S2 for a controlled continuation.

## S1-to-S2 resume

After the S1 artifact audit succeeds, materialize a continuation beside that
run and resume through the normal config-driven entry point:

```bash
CANARY_RUN_ROOT=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-28-GPU-CANARY-B-crop-pixel512-s1to2-bs4-n2-teacher25-ws4
CANARY_EXTENSION="$CANARY_RUN_ROOT/continuations/step1-to2.json"

"$CANARY_PYTHON" "$CANARY_REPO_ROOT/tools/materialize_policy_horizon_extension.py" \
  --config "$CANARY_CONFIG" \
  --output "$CANARY_EXTENSION" \
  --extension-id PRL28-GPU-CANARY-B-S1-TO-S2 \
  --target-step 2 \
  --checkpoint-step 0 --checkpoint-step 1 --checkpoint-step 2 \
  --code-commit 89dfd52decece90cbb2d3923cc9dc65c8eb6f0d4

"$CANARY_PYTHON" -m tgvf_rl.cli run-policy "$CANARY_CONFIG" \
  --horizon-extension "$CANARY_EXTENSION"
```

The materializer requires an exact S1 tracker, a contiguous metrics prefix,
complete rank shards, and matching project-checkpoint/runtime-pointer
identities. A future full-Qwen GPU evaluation canary remains separate: it must
first provide a non-destructive Qwen-only merger and a receipt proving the
merged model originated from the named checkpoint.
