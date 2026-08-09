# PRL15 TGVF versus PRL14 Crop-16 controlled pilot

PRL15 is a treatment of the completed PRL14 Crop experiment. The control is
not regenerated from the TGVF plan. Its immutable source is the existing
`completion.json` with SHA-256
`3907b310642aa542cf7ffcb6dec12c2a23d87634cdd9f696b5f984eacb1f70f1`.
The comparison uses PRL14's permanently retained step-8 checkpoint.

## Scientific question and the only intended treatment

The question is whether replacing native Crop with trainable RP66/TGVF helps
under the already successful Crop-16 RL recipe. Both Qwen3-VL-8B-Instruct and
the RP66 Adapter are updated in the TGVF arm. The intended arm differences are:

- Crop prompt/tool and Crop observation construction versus TGVF prompt/tool
  and RP66 observation construction;
- no RP66 state in Crop versus RP66 initialized from the stage-1 step-2000
  artifact and updated jointly with Qwen in TGVF;
- the minimum dataset, AgentLoop, rollout publication and paired-checkpoint
  plumbing required to execute those two visual actions.

Reward mathematics is not a treatment. Both arms use the source-aware
DeepEyes equations: visual samples use
`0.8*accuracy + 0.2*format + 1.2*I(correct and successful tool)`, while
ThinkLite uses `1.2*accuracy + 0.4*format`. Both use the same Qwen2.5-72B
OpenRouter/DeepInfra service record (concurrency 16 and the same bounded retry
policy), whose file SHA-256 is
`fff705c59408f4863244ff28df3443176e85de83147344df6a2350859c233021`.
TGVF's async trajectory scorer is a transport adapter around the same official
extractors, judge prompts, weights and failure behavior.

## Matched effective training configuration

The two composed Hydra configurations match on the causal training controls:

- eight GPUs / eight FSDP ranks, TP1, and eight AgentLoop workers;
- 16 prompts and 16 rollouts per prompt (256 trajectories per update);
- actor micro-batch 32 per rank, rollout/ref log-prob micro-batch 32, one PPO
  epoch, and no gradient accumulation;
- full-model Qwen training, including vision tower and projector;
- AdamW at `1e-6`, betas `(0.9, 0.999)`, weight decay `0.01`, effective epsilon
  `1e-8`, constant LR, and grad clip `1.0`;
- response length 20480, prompt length 8192, token caps 16384, vLLM model/batch
  capacity 32768, max sequences 1024, and memory utilization 0.65;
- gradient checkpointing and remove-padding enabled, fused Torch kernels,
  text FlexAttention, vision SDPA, FSDP resharding and torch.compile enabled;
- token-mean DeepEyes actor loss, GRPO advantage normalization, and zero KL.

PRL14 continued from step 8 to step 16, whereas the first PRL15 pilot stops at
step 8. This does not change updates 1--8: the scheduler is constant and the
effective optimizer horizon is `-1` in both composed configs. The comparison
is therefore PRL14 step 8 versus PRL15 step 8, not PRL14 step 16.

The declaration is
`configs/policy/controls/prl15_crop_rp66_matched.json`. Ordinary equal fields
are checked directly. Semantically equal values stored under different
framework paths, such as the judge SHA, are compared through explicit
control/treatment paths. The audit command is:

```bash
python tools/audit_prl15_against_prl14_crop16.py \
  --crop-contract /absolute/path/to/prl_13_a_...toml \
  --rp66-config /absolute/path/to/prl_15_r0_...ws8.toml
```

`--launch` is deliberately forbidden because PRL14 already exists.

## Smoke and formal lifecycle

The earlier `actor-rollout-only-v1` one-step smoke used four GPUs with
micro-batch 1 and GA4. It remains useful evidence that joint Qwen/RP66
backpropagation, publication and checkpoint recovery execute, but it is not a
valid Crop-16 control and must not authorize formal training.

A new one-step smoke must use the formal eight-GPU/micro32 execution shape.
Smoke stays console-only and writes below
`output.root/smoke/<smoke-id>/`; it cannot contaminate formal metrics or W&B.
Formal launch remains blocked until that smoke completes with finite reward,
finite Qwen gradients, a changed RP66 state, eight Qwen/optimizer shards, and
a resumable paired checkpoint. Formal execution must use tmux.

## Evaluation

`configs/evaluation/prl15_rp66_step0_step8_coredev2511_plan.json` compares:

- step 0: base Qwen plus immutable RP66 stage-1 state;
- step 8: Qwen step-8 checkpoint plus the content-addressed RP66 step-8 state.

The evaluator must use the same CoreDev-2511 task/protocol and sampling
contract for both states. It loads both members of the Qwen/RP66 pair and
requires the RP66 update acknowledgement; a Qwen-only checkpoint is invalid.

## Git identity

Formal launch requires a clean, pushed branch. The run TOML binds an execution
commit that must be an ancestor of the final binding commit. Configuration,
comparison declaration, launcher, runtime and tests are committed together;
the old world4 experiment identity is retained only as historical smoke
evidence.
