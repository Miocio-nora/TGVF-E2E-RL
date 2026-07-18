# Experiment Ledger

No experiments have been launched in this repository.

The repository is currently a documentation-only scaffold. GPU training,
evaluation, dependency installation, and checkpoint conversion are not
authorized by the scaffold itself.

## Required entry template

```text
### <ID>

- Status: PLANNED | RUNNING | COMPLETE | INVALID | SIDE_RESULT
- Question:
- Baseline and exact output path:
- Model and processor identity:
- Representation checkpoint identity:
- Policy/reference initialization:
- Rollout policy version and allowed asynchronous staleness:
- Code commit and worktree state:
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
- Logit/logprob/loss/gradient parity tolerances:
- World size, microbatch, accumulation, and global batch:
- GPUs:
- Start/end timestamps, elapsed time, and session/process identity:
- Command:
- Outputs:
- Scorer/parser identity:
- Metrics:
- Conclusion:
```

Before any launch, every applicable field must be resolved from files rather
than inferred from a script name or prior conversation.
