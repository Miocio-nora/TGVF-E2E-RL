# Experiment Ledger

No experiments have been launched in this repository.

The repository is currently a documentation-only scaffold. GPU training,
evaluation, dependency installation, and checkpoint conversion are not
authorized by the scaffold itself.

## Compatibility-spike status

`docs/VERL_COMPATIBILITY_SPIKE_PLAN.md` is a proposed task, not an experiment
entry. There is currently no `PLANNED` compatibility-spike cell and therefore
no GPU cell is authorized. Exact hardware, environment/image digest, model
runtime identity, command, output path, tolerances, and stop conditions must be
filled here before each proposed S2/S3/S4 GPU process.

CPU-only static review under S0 does not create an experiment identity. Any
dependency installation or model download still requires the separate
approval recorded by the spike plan.

## Required entry template

```text
### <ID>

- Cell/matrix ID and mandatory/diagnostic class:
- Spike-plan git revision and A0/A1/A2 approval references:
- Lifecycle status: PLANNED | RUNNING | COMPLETE | CANCELLED
- Result: PENDING | PASS | FAIL | BLOCKED_NOT_RUN | INVALID | SIDE_RESULT
- Question:
- Baseline and exact output path:
- Model and processor identity:
- Representation checkpoint identity:
- N/A fields and justification:
- Policy/reference initialization:
- Rollout policy version and allowed asynchronous staleness:
- Code commit and worktree state:
- Repository adapter/patch surface and hash:
- Dataset/manifest, hashes, sample rule, and n:
- Native prompt/tool schema hash:
- Chat-template/token-fixture hash and token-ownership masks:
- D/DeepStack/position/mask identity:
- Observation materialization/artifact identity used by all replays:
- RL framework/version/environment lock:
- Objective equations and normalization:
- Rollout/replay forward mode and adapter dropout/RNG contract:
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention:
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh:
- Logit/logprob/loss/gradient parity tolerances:
- World size, microbatch, accumulation, and global batch:
- GPUs:
- Start/end timestamps, elapsed time, and session/process identity:
- Actual GPU-hours and peak scratch use:
- Command:
- Outputs:
- Scorer/parser identity:
- Metrics:
- Conclusion:
```

Before any launch, every applicable field must be resolved from files rather
than inferred from a script name or prior conversation.
