# Canonical runtime migration plan — 2026-08-30

Status: active, CPU-only stabilization work. Experiment execution remains
frozen.

## Why this migration exists

`origin/main` at `43f2295c` is the documentation line, not the complete runtime
used by the recent policy experiments. The current NoTool, Crop, TGVF and
Atomic implementations evolved on the PRL15–PRL27 side lineage. The final
runtime source snapshot on that lineage is `c448e583`; the later PRL27-B
training and evaluation heads add only experiment-specific configuration,
tests and control scripts.

The canonical line therefore uses `c448e583` as a reviewed source donor. It
does not merge or replay the complete experiment branch. Result artifacts,
watchers, relays and run-specific launchers are not runtime dependencies.

## Migration rules

1. Port cohesive file groups from the donor's final source tree, not a long
   chain of partially superseded commits.
2. Preserve the stabilization line's explicit observation and action-boundary
   contracts when resolving overlaps.
3. Preserve `origin/main` representation ablations while adding the trainable
   borrowed-projection interfaces required by the current policy runtime.
4. Keep historical schemas readable, but never promote a historical config to
   a new canonical run implicitly.
5. Produce method-specific v2 run schemas before any new four-arm experiment.
6. Do not start training or evaluation during migration.

## Batch 1 — collection and hermetic prerequisites

Required outcomes:

- restore the complete immutable LoRA snapshot-pointer provider used by the
  evaluator;
- restore the missing direct-answer prompt identity with its byte/hash tests;
- split pure launch-plan construction from live compiler/header preflight;
- make Python 3.12 headers an explicit, hash-bound runtime prerequisite rather
  than an ignored file under the current worktree;
- collect the full suite with no import errors.

The runtime preflight remains fail-closed, while unit tests use temporary
fixtures and never require a sibling worktree's `.deps` directory. The v1
strict JSON manifest binds the minimum declared inputs (`gcc`, `g++`,
`Python.h`, and `pyconfig.h`) by absolute path, size, executable bit and
SHA-256. It deliberately does not claim to close recursive Python includes or
the compiler's assembler, linker, built-ins and runtime. That residual remains
a launch blocker. `run-policy` must preflight this contract before consuming a
one-time authorization; `plan-policy` may omit it only to render the blocker.

## Batch 1B — immutable runtime and worker startup

Independent review found that path/config authorization alone is insufficient.
Before Batch 2 can launch anything, this batch must:

- build and content-bind an immutable runtime package from the reviewed commit;
  the shared `.venv312` editable install currently resolves `tgvf_rl` to the
  separate base worktree and is not a launchable code identity;
- execute the verified Python inode through an already-open descriptor instead
  of re-opening its venv symlink;
- bind the exact worker argv and allowed environment mapping into the consumed
  authorization, verify them before runtime imports, and clear authorization
  variables immediately afterwards;
- issue a one-shot claim for the Policy worker and topology/rank-bound member
  claims for representation workers;
- publish freeze-override, launch-token, and launcher-liveness state as one
  failure-atomic authority transaction;
- replace the representation evaluator's `weights_only=False` pickle boundary
  with a safe typed artifact loaded from already-verified bytes.

Until those requirements and their race/replay tests pass,
`runtime_closure.launch_enabled` remains false. A freeze override cannot change
that value or bypass these blockers.

## Batch 2 — four-method training substrate

Port and reconcile these cohesive layers:

- environment/replay: agent loop, Crop, Crop+TGVF, TGVF focus, native
  appender, Qwen3 materializers/layouts and source-visual state;
- observations/protocol: record/store schemas, strict parser and tool schemas;
- policy: current config/run-config, NoTool, Crop, TGVF target-guide, Atomic,
  trainable replay, metrics and Qwen replay;
- veRL: exact/fused replay, trainable Crop/TGVF engines, live runtime, task
  runner, checkpoint lifecycle/weight sync, launcher and vLLM tool runtime;
- data/reward: Teacher25/teacher-ratio bindings and the matched reward routes;
- Qwen/representation: Qwen3 family ports and the manually merged adapter and
  DeepStack layers.

`representation/adapter.py` is a mandatory three-way merge. It must retain
MatrixCE, unidirectional/bidirectional interaction, and pre-/post-merger
injection ablations while adding the donor's borrowed-projection ownership
boundary.

## Batch 3 — evaluation, action boundary, pixels and RNG

The evaluation contract binds, at minimum:

- training and evaluated weight identities;
- task-manifest SHA and coverage counts;
- training, declared evaluation and effective processor pixel caps;
- prompt, tool schema, parser, observation renderer and action-boundary IDs;
- generation budget, scorer and RNG identities.

Canonical action semantics are
`qwen-native-action-boundary-single-terminal-tool-call-v2`: exactly one
terminal tool call is executable. A suffix after `</tool_call>`, multiple calls
in one assistant turn, or malformed tags fail closed. Historical
answer-over-action/last-call behavior remains available only under the named
legacy v1 identity.

For a true `@512` row, all three checks must pass:

- training `image_max_pixels == 262144`;
- effective evaluation processor cap `== 262144`;
- processor evidence proves represented image area `<= 262144`.

The existing method RNG streams are not automatically paired: policy identity,
prompt tokens or protocol identity differ. A future common-random-number study
must preregister the projected intervention axes. Until then, cross-method
rows remain `standalone` even when their aggregate task IDs match.

## Batch 4 — method-specific v2 contracts

Only after the implementation commit is fixed, materialize four new run
contracts:

| Method | Observation contract | Action contract | Target endpoint |
|---|---|---|---|
| NoTool | explicit `none` | direct-only | S32 @512 |
| Crop | matched60 Crop | strict terminal v2 | S32 @512 |
| TGVF full target-guide | generic native TGVF | strict terminal v2 | S32 @512 |
| Atomic Crop+TGVF | generic native Atomic | strict terminal v2 | S32 @512 |

All four bind Teacher25, seed policy, S32, 512², one task manifest, one scorer
contract and the reviewed implementation commit. Historical PRL26/PRL27 files
may be used as references only; their IDs and artifacts are not inherited.

## Explicit non-migration set

Do not port:

- PRL25/26/27 waiters, relays, supervisors, launch binders or status writers;
- hard-coded artifact/output paths and tmux environment mutation;
- PRL27-A live60/replay86 checkpoints or claims;
- the PRL26 generic86 evaluator as a canonical path;
- seed43, oracle-Crop, forced-counterfactual or raw-direct diagnostics as
  four-arm training dependencies;
- hand-written result claims in place of the typed result registry.

## Required gates

Each batch must pass its focused tests and Ruff before the next batch. Final
promotion requires:

```text
Python 3.12 full test collection: 0 errors
Full CPU suite: pass
Ruff: pass
Launch-control audit: pass
Result-registry artifact verification: pass
Canonical runtime closure: enabled with zero blockers
GPU execution policy: still frozen
```

The four canonical configs are a separate follow-up commit so their
`code.commit` fields can point to the already reviewed implementation commit.
