# Canonical runtime migration plan — 2026-08-30

Status: superseded for source promotion on 2026-09-01. The strict
launch-security design below is retained as historical context, but it is no
longer the normal experiment path or a `main` promotion gate. Ordinary Policy
work uses the config-derived `run-policy` entry point without a repository
authorization token; `strict-run-policy` is an explicit opt-in compatibility
path.

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

### 2026-08-31 resolved Representation startup architecture

The current Representation route is not the future trusted route. It still
executes the audited Python as `python -m torch.distributed.run`, then asks
torchrun to start `python -m tgvf_rl.cli run-representation`. Both layers load
substantial mutable package code before inherited worker authority is checked.
The existing worker bootstrap is diagnostic-only and deliberately unwired.

The reviewed target architecture is one authorization-first, repo-owned
launcher followed by Torch Elastic with a string/binary entrypoint equal to the
exact Python executable. Torch must be imported only after launcher-stage
verification. Each member must be a fresh `-B -I -S` interpreter. Callable
Elastic entrypoints, `--run-path`, the current module-mode torchrun route, and
`PYTHON_EXEC` wrappers are not acceptable substitutes: installed Torch 2.9
either loads Torch too early, uses an in-process callable context, or cannot
express the required interpreter flags.

The proposed Linux candidate uses a small, authorization-bound `-c`
stdlib-only trampoline. Before any repository or sealed-package byte executes,
the trampoline must verify the live launcher PID and start ticks, reopen the
declared proc-fd, and check regular-file metadata, exact seals, size, and whole
payload SHA-256. It may then rebase the retained local fd as
`/proc/self/fd/<n>` and import a sealed stage-0/package. Directly passing a
launcher proc-fd as the Python script is only a mechanical compatibility probe:
Python executes bytes before those bytes can authenticate their own locator.

Commit `50f5de7` adds only the dependency-light sealed-memfd byte-capability
scaffold needed to study this candidate. It creates a Linux memfd with
write/grow/shrink/final seals, a fixed 64 MiB ceiling and streaming verification;
binds owner PID/start ticks, fd, device/inode, mode, length, seals and SHA-256 in
canonical JSON; retains a private guard fd; and can reopen and retain the same
sealed inode through a live owner's proc-fd. It imports no project runtime or
Torch and performs no process creation, authority consumption, target import,
mint, or dispatch.

This scaffold does not close immutable packaging. A canonical package still
requires the verified trampoline and outer exec handoff, deterministic sealed
archive construction and provenance, zip-origin and `__file__` migration,
descendant FD transport, procfs/hidepid/LSM/dumpable preflight, and separate
native Torch/CUDA/dependency closure. Same-process hostile raw-close, `dup2`, or
concurrent fd rebinding also remains outside the leaf's guarantee. Consequently
the seven experiment blockers and `launch_enabled=false` remain unchanged.

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

## Source-promotion gates (revised 2026-09-01)

The earlier runtime-closure and repository-freeze gates below proved too
coupled to machine state and made ordinary parameter changes and interrupted
workflow recovery unnecessarily difficult. Source promotion now requires:

```text
Python 3.12 full test collection: 0 errors
Default CPU suite: pass
Core config/matrix/action/replay/checkpoint focused suites: pass
Ruff: pass
```

The retired launch-security/control-plane suites and dedicated CUDA parity
probe remain available for explicit maintenance work, but are not collected by
default and are not source-promotion gates. GPU/Ray step-0 and checkpoint-resume
smokes are separate experiment-runtime acceptance checks; they do not block a
source-only promotion.

The five canonical configs are a separate follow-up commit so their
`code.commit` fields can point to the already reviewed implementation commit.
