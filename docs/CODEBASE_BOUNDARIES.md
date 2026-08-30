# Codebase boundaries

This repository separates reusable implementation from historical experiment
evidence. The boundary is enforced by a CPU-only, fail-closed audit; it is not
only a naming convention.

The measured debt and ordered reduction plan are tracked in
[`CODEBASE_CONSOLIDATION_PLAN_20260830.md`](CODEBASE_CONSOLIDATION_PLAN_20260830.md).
The frozen worktree/ref review is recorded separately in
[`WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md`](WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md).
That inventory is read-only and does not authorize any deletion.

## Code ownership boundary

Neutral source is reusable code whose identity does not depend on a particular
RP or PRL run. New implementation belongs under neutral modules in
`src/tgvf_rl` or in reusable, neutrally named tools. Neutral Python source must
not import a run-specific RP/PRL module.

Some historical source and tool paths include RP or PRL identifiers. They are
quarantined compatibility or evidence code, not canonical extension points.
The boundary policy records every such path in an exact allowlist. A new or
renamed run-specific source/tool path is a violation; it must not be accepted
merely because it resembles an existing historical path.

## Configuration ownership boundary

All future canonical experiment configurations have exactly these homes:

- `configs/canonical/policy` for policy training and rollout contracts;
- `configs/canonical/representation` for representation training contracts;
- `configs/canonical/evaluation` for evaluation contracts.

`configs/ops` contains repository operations and control policies, including
the boundary policy itself. It is not a home for scientific run
configurations.

These five legacy roots are frozen evidence-only inventories:

- `configs/evaluation`;
- `configs/overnight`;
- `configs/policy`;
- `configs/representation`;
- `configs/smoke`.

Their many historical files are deliberately not promoted one by one. The
policy pins each root's file count, hash of its sorted relative-path inventory,
and a content-tree SHA-256 over every relative path and the exact bytes of the
corresponding file. The content-tree stream uses domain separation, then an
unsigned 64-bit big-endian file count and sorted path/content records whose
UTF-8 path and byte content are each prefixed by an unsigned 64-bit big-endian
length. Concatenation therefore cannot make two different trees equivalent.
Additions, removals, renames, or a one-byte change to an existing file fail the
audit. Existing files remain historical evidence and are not templates for new
work. Any intentional historical correction or migration requires explicit
review and a deliberate policy revision; ordinary development must leave these
roots unchanged.

Canonical configuration roots must exist as real directories in a clean
checkout and must not be symbolic links. Canonical configuration paths must be
portable and neutrally named. They must not contain RP/PRL run identifiers or
machine-specific absolute paths.

The historical policy-evaluation v1 schemas are readable evidence records but
are not launch contracts: they predate explicit pixel, success-observation,
action-boundary and complete snapshot bindings. Newly materialized policy
evaluations use the v2 constants owned by
`evaluation/policy_evaluation_config.py` and compatibility re-exported by
`policy_coredev.py`, plus the schema aid at
`configs/canonical/evaluation/policy_evaluation_v2.schema.json`. A v1 file is
never upgraded in place, and a new measurement must use a new evaluation ID
and output root.

## Evidence ownership boundary

Typed registries and future preregistration records live under `evidence/`;
they are neither executable run configurations nor raw experiment artifacts.
For example, policy-result classifications are owned by
`evidence/policy/result_registry_v2.json`. Registry artifact paths are
repository-relative bindings to immutable evidence, while the usually large
artifact trees themselves remain outside this source migration and are never
silently copied into `configs/`.

Generated manuscript tables must be materialized from a typed registry after
its artifact checks pass. A document is a consumer of evidence, not an
alternative score database.

## Artifact primitive ownership

Canonical JSON bytes and hashes with exact matching semantics belong to
`tgvf_rl.artifact_contracts`. Create-only and content-consistent immutable
publication belong to `tgvf_rl.immutable_publication`. Descriptor-bound
regular-file reads belong to `tgvf_rl.secure_file_read`, whose final-leaf,
absolute-chain, and already-bound-root contracts are deliberately separate.

Migration is semantic and incremental. Similarly named helpers must not be
merged when they differ in symlink, stability, error, ownership, or publication
semantics. At this snapshot the secure-reader production migration covers only
`evaluation/result_registry.py`; the existence of the shared leaf does not
claim that every historical reader has been migrated.

## Absolute-path debt

Machine-specific absolute paths under source and tools are prohibited for new
code. The current policy preserves only the already-known debt using the exact
repository-relative file path, SHA-256 of the literal, and occurrence count.
The literal values themselves are not copied into the portable policy.

The audit reports two separate classes:

- `debts` are exact, known historical matches. They remain visible even when
  the audit passes and should be removed over time.
- `violations` are new debt, boundary crossings, policy drift, missing expected
  debt, or unreadable/untrusted inputs. Any violation blocks the audit.

The allowlists are a ratchet, not a suppression mechanism. Never change the
baseline simply to absorb a new machine path, run-specific module, or legacy
configuration. Fix the new debt at its source. A baseline change is acceptable
only for an intentional boundary migration with explicit review and evidence.

## Production module size debt

Repository-boundary policy v3 fixes the normal production-module limit at
1,000 physical lines. Each of the 29 current exceptions names one exact Python
path below `src/tgvf_rl`, its stable subsystem owner, concrete reason, next
split seam, and a ceiling equal to its current line count. The candidate audit
fails on an unregistered oversized module, growth above a ceiling, slack after
a file shrinks, a missing/moved file, or an exception retained after the module
reaches the limit.

The boundary CLI also accepts an explicit baseline policy. In that mode the
candidate exception paths must be a subset of the baseline and no ceiling may
increase, so changing code and its allowance together cannot relax the
ratchet. The initial v3 inventory is a reviewed bootstrap; its 29 rows are
visible structural debt and do not declare those files well-factored.

## Running the audit

From the repository root, run:

```bash
.venv312/bin/python tools/audit_repository_boundaries.py \
  --repository-root . \
  --policy configs/ops/repository_boundary_policy.json
```

The command emits a structured report, exits successfully only when there are
no violations, does not import model runtimes, and requires no GPU. CI runs the
same audit with `CUDA_VISIBLE_DEVICES` empty before the test suite.

## Execution-surface ownership

`configs/ops/execution_surface_policy.json` is the machine-readable inventory
of every directly invocable repository entry point and reviewed script support
module. The control-plane audit includes every Python file below `tools` and
`spikes`, plus every `src` module selected by a module-main guard,
`__main__.py`, Python shebang, executable bit, or `project.scripts` console
entry. It also recursively includes every shell file below `tools` and
`spikes`. The discovered and manifested path sets must be exactly equal: an
omitted, extra, duplicate, moved, or newly added candidate blocks the audit.

Each manifest row binds the path, entry kind, classification, allowed and
blocked modes, authorization kind, conservative capability flags, workload
kinds, and SHA-256 of the exact file bytes. A one-byte implementation change
therefore requires review of both the code and its execution classification;
marker searches are not accepted as a completeness proof.

The `check_launch_gate.py` control row is bound to all eight real argparse
commands (`ready`, `authorize`, `override-freeze`, `consume`, `wait`, `status`,
`quarantine-legacy`, and `audit-control-plane`) and to its exact conservative
capability map. Parser-mode or policy drift blocks the audit.

The revision-3 stabilization inventory contains 79 unique paths, and each has
exactly one machine-checked disposition. It contains two canonical entries,
two control-audit utilities, 36 permanently quarantined Python entries, one
permanently quarantined shell entry, 13 mixed entries whose explicitly
read-only modes remain available, 21 bounded offline artifact materializers,
three read-only utilities, and one strictly import-only support module. A
materializer receives `bounded_artifact_write` only after its content-bound
implementation has been reviewed for write-once or content-consistent
idempotent publication; arbitrary overwrite and cross-worktree frozen-config
writers are quarantined. GPU,
training/evaluation, external-judge/network, arbitrary exec, and destructive
legacy or spike entries must be permanently quarantined or have a statically
verified mode guard before any business dispatch.

Permanent Python quarantine is structural rather than marker-based. Each of
the 36 entries has an exact top-level `os.execv` handoff through absolute
`/usr/bin/python3` with `-I` to the fixed `quarantine-legacy` controller. The
handoff uses only `os` and appears immediately after the optional module
docstring and `__future__` imports, before repository/runtime imports, path
injection, environment mutation, or other script-level side effects. Its
legacy `main` helper retains a second direct guard, and its final module
dispatch must be direct and uncaught. A guard hidden inside `try`/`except`, a
pre-guard import, or a post-catch dispatch fails the audit even if the file
hash is deliberately refreshed.

Python interpreter startup occurs before any script statement and is therefore
an explicit caller trust boundary. The executable legacy Python files use the
fixed `#!/usr/bin/python3 -I` shebang, and the isolated second hop is always
enforced. An explicit `python script.py` invocation is safe from caller-side
`sitecustomize` only when that caller also uses a trusted isolated interpreter,
for example `/usr/bin/python3 -I`; the repository does not claim to neutralize
hooks already executed by an ordinary hostile caller before its code starts.

Mixed read-only/quarantined Python entries must parse their mode through a
statically traced `argparse.ArgumentParser` receiver, call the mode guard as a
direct statement immediately after that parse, and end in a direct module
dispatch. Calls on an unrelated object named `parse_args`, caught guards, and
nested business dispatch do not satisfy this contract.

The remaining permanent shell entry is an exact byte-bound wrapper. It starts
with `/bin/bash -p`, derives its location using only Bash parameter expansion
and builtins, and `exec` absolute `/usr/bin/python3 -I` into the quarantine
controller. A symlink used as the final script path is rejected before path
derivation, while symlinked ancestor directories are resolved to their physical
directory with builtin `cd -P`. This makes the pre-guard path independent of
hostile `PATH`, `BASH_ENV`, and `PYTHONPATH`; comments and heredocs containing
the expected words cannot satisfy the audit.

Canonical public branches also use structural checks. Each branch begins with
an argument-free runtime-closure assertion and has an exact reviewed call
shape through preflight, one-time authorization consume, and dispatch; nested
argument side effects and extra statements are rejected. Those control symbols
must have one exact local definition or direct-import provenance and may not be
rebound, shadowed, or deleted. The internal worker has an equivalent exact
inherited-receipt/closure prefix. These checks are containment while runtime
closure remains disabled; they do not declare the public launch implementation
complete.

The C1 disposition condition is therefore satisfied, but canonical launch and
runtime closure are not. A machine classification is containment and ownership
metadata, not evidence that every retained surface is a public CLI command.

Three conservative classifications deserve explicit rationale:

- `materialize_policy_selection_sources.py` can transitively download source
  data and invoke an archive subprocess, so it is not an offline-only
  materializer in its current form;
- `validate_vlmevalkit_deployment.py` accepts a caller-supplied deployment file
  whose Python executable is then invoked by subprocess, so it is not a closed
  read-only probe until that input is restricted to a content-bound authority.
- `materialize_policy_t1_mixed_retained_pool.py` is quarantined because the
  checked-in final-scoring producer emits manifest v1 while its consumer accepts
  only manifest v2. No checked-in v2 producer exists, and synthetic v2 unit-test
  fixtures are not an end-to-end provenance chain. Re-enable it only with an
  explicit content-bound v1-to-v2 migrator or a native v2 producer plus a
  producer-to-consumer integration test.

`audit_policy_data_selection_t1_replay.py plan`, the default no-launch plan in
`verl_fsdp2_vllm_sync_smoke.py`, and
`compact_policy_checkpoint_storage.py inventory` remain usable as verified
read-only modes. Their GPU replay and compact/delete modes remain blocked.

Run the combined control-plane audit with no GPU visibility:

```bash
CUDA_VISIBLE_DEVICES='' .venv312/bin/python tools/check_launch_gate.py \
  audit-control-plane --repository-root .
```
