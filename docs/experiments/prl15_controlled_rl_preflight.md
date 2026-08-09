# PRL15 controlled-RL preflight

This note records the implementation boundary for the matched RP66 pilot and
its native-Crop control. No formal run should start from an uncommitted tree.

## Frozen experimental shape

Both arms use Qwen3-VL-8B-Instruct full-model training, the retained mixed T1
schedule, 16 prompts per optimizer step, 16 trajectories per prompt, four
FSDP ranks, one PPO epoch, AdamW at `1e-6`, and the released DeepEyes GRPO
reward weights. The text judge uses Qwen2.5-72B through OpenRouter with the
same proven service contract as Crop: concurrency 16, four bounded attempts,
0.25--2.0 second backoff, transient exhaustion scored as zero and audited,
and authentication/model-identity failures aborting the run.

The scientific arm difference is the visual action and its trainable state:

- Crop uses the native DeepEyes crop prompt/tool and has no RP66 Adapter.
- TGVF uses the matched TGVF prompt/tool and jointly updates full Qwen plus the
  RP66 Adapter, publishing both to rollout after every optimizer step.

## Why the world4/micro1 Crop control is required

`deepeyes_official_micro_token_mean` reproduces the released reduction: it
computes a token mean independently inside each fixed local actor micro-batch,
then gives every micro-batch equal weight. With unequal response lengths,
changing actor micro-batch size changes the effective trajectory/token
weighting. It is not a generic veRL requirement, but it is part of this named
objective.

Therefore the earlier world8/micro32 Crop run is useful historical evidence,
but it is not a strict scalar-loss control for PRL15 world4/micro1. The new
launcher `tools/launch_prl15_crop_ws4_micro1_control.py` constructs a Crop arm
and programmatically proves the common runtime fields equal the RP66 formal
plan before composition or launch.

The comparison declaration is data, not Python code. It lives at
`configs/policy/controls/prl15_crop_rp66_matched.json` and has two lists:

- `required_equal`: copied from the current RP66 launch plan into Crop, then
  checked for equality;
- `arm_specific`: expected scientific/runtime differences between Crop and
  TGVF.

Every other observed difference is preserved in
`unclassified_differences`, but is informational rather than fatal. Therefore
changing a later batch size, micro batch, capacity, scheduler, or other common
variable requires editing the run config and, only if its scientific role has
changed, this declaration. It does not require editing a hard-coded Python
allowlist. Missing `required_equal` fields and unequal declared controls remain
fatal because those would make the named comparison false.

## Runtime isolation and reference diagnostic

Smoke metrics are forced to either the legacy `output.root/smoke/metrics.jsonl`
or a labeled closure `output.root/smoke/<smoke-id>/metrics.jsonl`; formal
metrics are forced to `output.root/metrics.jsonl`. A labeled smoke also owns
its own checkpoints and logs. The TaskRunner accepts only a safe lowercase
label inside this layout, preventing a successful smoke from contaminating the
formal history while allowing repeated engineering canaries without moving or
overwriting old evidence.

Both mathematical KL coefficients are zero. PRL15 consequently disables the
extra frozen-reference forward and uses an ActorRollout worker role. This
removes one full reference forward over 256 trajectories and its model/runtime
headroom. The legacy default remains enabled so old experiment identities do
not silently change. This switch does not alter reward, advantages, policy
loss, or checkpoint weights because the diagnostic coefficient was already
zero.

## Step 0 versus step 8 benchmark contract

The paired CoreDev-2511 plan is pinned in
`configs/evaluation/prl15_rp66_step0_step8_coredev2511_plan.json`. Both arms
must use the same task manifest, rank partition, TGVF prompt/tool protocol,
call cap, and sampling contract.

The state being compared is a pair, not a single model:

- step 0 = base Qwen weights + the immutable RP66 stage-1 artifact;
- step 8 = materialized full-Qwen step-8 checkpoint + the immutable RP66
  step-8 content-addressed snapshot.

The existing full-model CoreDev backend is Crop-only (`native_pixels=True`)
and the older TGVF backend assumes a policy LoRA. Neither is a valid shortcut
for this paired full-Qwen/RP66 state. The dedicated
`full_model_trainable_rp66` backend now freezes and loads both members. It
rejects a Qwen HF closure containing `tgvf_adapter.*` keys, installs the exact
RP66 state through the vLLM Adapter update RPC, requires the worker ACK, and
uses a combined Qwen-tree/RP66-state policy identity in every result row.

`tools/run_prl15_paired_evaluation.py` is the resumable executor. It reads the
plan and active run config rather than duplicating paths or hyperparameters.
With eight GPU IDs it runs step0 and step8 concurrently as two four-rank arms;
with four it runs them sequentially. `--wait-for-step8` makes it safe to start
the evaluator before training: it waits for checkpoint step 8, the embedded HF
tree, and the fixed RP66 pointer, then prepares, runs, resumes, and scores both
arms. The plan remains marked `awaiting_formal_step8_paired_snapshot` until the
formal closure exists.

The intended unattended command is:

```bash
python tools/run_prl15_paired_evaluation.py \
  --mode run --wait-for-step8 --gpu-ids 0 1 2 3 4 5 6 7
```

For a new engineering canary, use an explicit label and keep W&B disabled:

```bash
python -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config configs/policy/runs/prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_matched_8step_gpu0123.toml \
  --mode smoke --smoke-id actor-rollout-only-v1
```

## Git execution identity

The run TOML binds an execution-code commit rather than the later config-only
binding commit. The execution commit must be an ancestor of the checked-out
launch commit, the tree must be clean, and the branch must be pushed before
formal launch. This avoids a circular self-hash while preserving an exact,
reconstructable code identity.
