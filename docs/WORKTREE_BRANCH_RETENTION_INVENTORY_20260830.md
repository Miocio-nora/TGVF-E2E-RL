# Worktree and branch retention inventory — 2026-08-30

Status: read-only C4 snapshot. No worktree, branch, ref, or historical file was removed, moved, reset, stashed, or rewritten. This document proposes review actions only.

## Snapshot

- Captured: `2026-08-30T11:04:34+09:00`.
- Current: `stabilize/protocol-contract-v1-20260830` at frozen HEAD `05e32ea4e767e0230b6c9f54ac42c296e55a243e`.
- Remote default: `main` from read-only `git ls-remote --symref origin HEAD`; local `origin/HEAD` is absent.
- Comparison baseline: `origin/main` at `43f2295c15da2bbc14972af6b63f0038e7789c3f`.
- All 35 local `origin/*` refs exactly matched all 35 live remote heads by name and SHA. No fetch/ref mutation was performed.
- Repository shallow: `false`.
- Totals: 72 worktrees (59 attached, 13 detached, 7 dirty, 65 clean, 0 locked, 0 prunable); 62 local branches (59 checked out, 3 without a worktree, 8 contained in `origin/main`).
- This file was absent at capture. Status hashing excludes only `?? docs/WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md` so the inventory self-file is neutral.

## Artifact-reference limit and approval rule

Git reachability is machine-decidable. Artifact references are not: there is no authoritative reverse index from branch/worktree/commit to all artifacts, while `artifacts/`, `outputs/`, and `checkpoints/` are ignored roots. Zero literal matches cannot rule out ignored, external, or historical dependencies. Therefore **every row is fail-closed as `artifact refs = unknown`**.

Before any removal, close artifact/path references, confirm no process uses the path, and preserve the exact commit with a durable ref where needed. **Every worktree or branch deletion requires exact user approval for that exact path/ref. Action codes are not blanket approval.**

Worktree actions: **W0** keep active; **W1** keep dirty; **W2** clean attached review candidate (retain branch; exact-path approval required); **W3** clean detached reachable candidate (first verify/create exact archive ref; exact-path approval required); **W4** keep unanchored and tag exact SHA before any later individually approved removal.

Counts: W0=1, W1=7, W2=51, W3=11, W4=2.

## Worktrees (72/72)

| Exact path | Branch/detached | HEAD | Dirty | Recovery anchor | Artifact refs | Action |
|---|---|---|---|---|---|---|
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl` | `main` | `475f2b1f85a703a8c261b46baaa9f0f92e266b9b` | dirty: 38 tracked + 55 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-adapter-ablation` | `adapter-ablation-rp67-20260826` | `92b8a4aec3713c3657e3c24897b7efbffdfcdbaf` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-adapter-architecture` | `adapter-architecture-ablation-20260826` | `029ca33f688491434aa3557ab3f35d7a8d158564` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-atomic512` | detached | `e5e02879d1bec87779c59712330e01eb2b1a2d43` | clean | ancestor of named refs (15) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-eval512-HVPRWY` | detached | `a8e655f4cc73d9ab3b17de6648f00f23e2f15768` | clean | ancestor of named refs (16) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-full-prompt-rl16` | `full-prompt-rl16-plan-20260827` | `029ca33f688491434aa3557ab3f35d7a8d158564` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-notool-s32` | `neurips-notool-rl-s32` | `e1e0e1754828944110f567f9359e4ef2f218f969` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-oracle-crop` | `diagnostic-vstar-oracle-crop` | `a434ee91b8dbc24978e250906f761e80c29dec2b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl14-run` | detached | `63826c8269808d557dfce7cfb4696116a9917e9f` | clean | ancestor of named refs (96) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-checkpoint-fix` | `prl15-checkpoint-every-step` | `4443fcc070f29b06dd18d59524ac92173c7706ed` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-eval-handoff` | `prl15-eval-handoff` | `4b1e1c9c92c74cef21a341245ac4e4fc06b5eb63` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-eval-live` | `prl15-eval-live` | `e06450d9a5f30e1d213e64ff8cbf933ce1ccfe4c` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-placeholder-fix` | `prl15-placeholder-protocol-fix` | `16dc184847b38662613e278f771b5e8ad0cb4c4b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-resume-schedule-fix` | `prl15-resume-runtime-checkpoint-gate` | `f3f5527ec4bc7844930d278177d3f6cdedc3a9b0` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-rp66-matched` | `prl15-rp66-matched-control` | `7324577ad1f8236cef49a73a0e67eb90aaf42861` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-termination-fix` | `prl15-termination-native-eos-fix` | `a6a4c149398608329579312b55f4057d1461467c` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-tool-attempt-fix` | `prl15-tool-attempt-metrics-fix` | `4bf8a7ddd4801864c5b74c02d424afa9a9ba3471` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-f1-eval` | `prl16-f1-eval` | `074fda6d81884180e905114e8101e34b30b482a9` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-judge-failsoft` | `prl16-judge-failsoft` | `e03461229a6196e7ce0f1c9dca2b645ca164b763` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-judge-rng-correctness` | `prl16-judge-rng-correctness` | `3f89161a848bb9a48ffff9364ac71381d682883b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-live-sync-parity` | `prl16-live-sync-parity` | `3f89161a848bb9a48ffff9364ac71381d682883b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-qwen-sync-integrity` | `prl16-qwen-sync-integrity` | `3f89161a848bb9a48ffff9364ac71381d682883b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-a-tgvf-teacher25` | `prl22-a-tgvf-teacher25` | `0b2292a469ecf0864b4e4ec85f545ea834ea1fb3` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-b-crop-tgvf-teacher25` | `prl22-b-crop-tgvf-teacher25` | `566a1fe4150ea61725944b134061209c5f8cb99c` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-c-crop-teacher25` | `prl22-c-crop-teacher25` | `3610f2dde2491194b3ada4fd0b1bee974035c10d` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-teacher25-shared` | `prl22-teacher25-shared` | `37b99e2b01f9459e0ee65f6b86e2950bf60d4417` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl23-teacher-ratio-launchers` | `prl23-teacher-ratio-launchers` | `02444c358f5d8bca78be7a206954bed14d4a548b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl23-tgvf-teacher-ratio` | `prl23-tgvf-teacher-ratio` | `0bf01727bceadae87ae8e488ae6075f3f10f3aad` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-a-bs64` | `prl24-a-bs64` | `2f07d89f8291113dddeb20f88c7cf843f3bf86a3` | dirty: 0 tracked + 3 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-d-crop-bs64-fmt2` | `prl24-d-crop-bs64-fmt2` | `c5bc2eab3cfa199ff96c515d816226c8a78c829e` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-d-crop-bs64-fmt2-sp2` | `prl24-d-crop-bs64-fmt2-sp2` | `c5bc2eab3cfa199ff96c515d816226c8a78c829e` | dirty: 7 tracked + 0 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-s24-s32-eval` | `prl24-s24-s32-paired-eval` | `27fd1ebeaa79f6a5e1750b15ef1e0a277386596b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-supervisor-refactor` | `prl24-supervisor-config-driven` | `2bf1b835beb9b1394b2d795b5d22cd5b78418443` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl25-crop-aligned` | `prl25-c-tgvf-80step` | `e43662998a6b76741fe6f36ab0d34d376b436040` | dirty: 6 tracked + 13 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-eval` | `prl26b-generic86-matched-eval-20260830` | `1dc0d1ec5eb233397b38bba9adf0f01812a79845` | dirty: 1 tracked + 0 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-historical` | `prl26b-generic86-historical-eval-20260830` | `001838b65a0655008fa066a08934f32bbc9bffb6` | dirty: 1 tracked + 3 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27-crop-canary` | `prl27-real-crop-canary-20260830` | `4fe6b3d931aedc19a4a4d9a7775a7850c4aa7f2b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27-replay-fix` | `prl27-crop-replay-renderer-fix-20260830` | `a779625c5ef309aed77d68ab202a96a8ad11830e` | dirty: 2 tracked + 0 untracked | exact local branch | unknown | **W1** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-eval-runtime` | `prl27b-eval-runtime-20260830` | `b6fdb73cfe926b6131b86e0a5c97a4af52940e3e` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-eval-waiter` | `prl27b-eval-waiter-20260830` | `a9e4b67c6e36e1efdefd1cb513b0aa86939b254d` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-relaunch` | `prl27b-corrected-crop-relaunch-20260830` | `300a1ee33a4df8a68ba3add835097f61d91f63eb` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-training-runtime` | `prl27b-training-runtime-20260830` | `f50fe3c66c719dd10f5dc5522e5142594831038b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-stabilize-protocol-v1` | `stabilize/protocol-contract-v1-20260830` | `05e32ea4e767e0230b6c9f54ac42c296e55a243e` | clean | exact local branch | unknown | **W0** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-target-v2` | `target-visual-descriptor-rl16-20260827` | `c0cebefd1ebc9ca2ddd91482a340f0d4b755e0b7` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-texture-bench` | `texture-benchmarks` | `913123e3539a6440219fb43163d56e221cb0900c` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-texture-bench-current` | `texture-benchmarks-current` | `abd652f6e98cec04fe912dfa5381124dcc3e08c4` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-673c0e4` | detached | `673c0e4cdcf97b74feb5b0bae944d75f85988520` | clean | ancestor of named refs (97) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-fb647` | detached | `fb64769c1dc04bf1c7fab793be951e9cbd37b257` | clean | ancestor of named refs (97) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-ffa4146` | detached | `ffa41467613b24376374e53d99be5bf29e9d0a0b` | clean | ancestor of named refs (97) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-rp70-20260802` | detached | `2d61b07995b1d5b90c221fe1faf5090e8d985fef` | clean | UNANCHORED | unknown | **W4** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.main-d-only-dev` | `main-d-only-dev` | `fe5ea776e6db817db55411fb1bfef53750904cde` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.mask-runtime-5ad8c10` | detached | `5ad8c10941633f7e0e2fa5449db9efa42d509a7c` | clean | ancestor of named refs (97) | unknown | **W3** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl16-frozen-rp66` | `prl16-frozen-rp66-control` | `0ab45b4b132b15a67ee5683c9b2a9e22adfe1c16` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r1-rp67-shaped` | `prl17-r1-rp67-shaped-reward` | `e177150f9b859004698017cb0f1760fe39b16374` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-rp67-calibrated-t` | `prl17-r2-rp67-calibrated-t` | `0d738adbc4ab257044f0ff3d9013e4371e7e113b` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-step16-paired-eval` | `prl17-r2-step16-paired-eval` | `a239c7d88f2bc8a4ec38c5c34e5fcb5dd5f965a3` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-tfree-step16` | `prl17-r2-tfree-step16-continuation` | `c4e24f74e9d603ed2cbad79640f032fc62edad68` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-rp66-shaped` | `prl17-rp66-shaped-reward` | `e7047d6979f0a2eaef6ed64838b2dc052ea54a3c` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl18-r0-unfrozen-rp67` | `prl18-r0-unfrozen-rp67` | `d4ab515d4e2eb2efc23fad91dfeb198dc0132cb0` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl19-r0-rp67-frozen-tfree-visual-api` | `prl19-r0-rp67-frozen-tfree-visual-api` | `9333fd429ecc7075258ef0d8f97cea5fefc49da9` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-crop-tgvf` | `prl20-r0-rp67-frozen-tfree-crop-tgvf` | `fc5e5dd99819dcb923ebf207d496be6f5fcb540a` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-eval` | `prl20-r0-step8-step16-eval` | `52f40853b71ec4a24efd0304cdb5036cd462f39f` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-step16` | `prl20-r0-step16-continuation` | `6b38a03c11c893e714bee6d0716cae9f73735357` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl21-r0-crop-tfree-16step` | `prl21-r0-crop-tfree-16step` | `b3d08b4ed657479250e449e57376781bd2cd8c77` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/texture-prl20-integration` | `texture-benchmarks-prl20-integration` | `5015ed3dd1b06651b74f37273b9e35a9d5bc709d` | clean | exact local branch | unknown | **W2** |
| `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/texture-prl21-step16` | `texture-benchmarks-prl21-step16` | `989fd038e2931fe928a2d2ac4c7f590f9f9838f9` | clean | exact local branch | unknown | **W2** |
| `/tmp/prl26-ab-config-fix-20260829-LgWEms` | detached | `5baddc64983039ac88a2fbb2cd6d95ed43601b20` | clean | ancestor of named refs (15) | unknown | **W3** |
| `/tmp/prl26-ab-release-gate-20260829` | detached | `001838b65a0655008fa066a08934f32bbc9bffb6` | clean | exact named ref (1) | unknown | **W3** |
| `/tmp/prl26-cd-prereq-hardening-20260829` | detached | `9e4d59f8b97a0b8d495d0a6bd761fe9f55852509` | clean | ancestor of named refs (15) | unknown | **W3** |
| `/tmp/prl26-doc-main-20260829` | detached | `999a87020eea5600a02085c1d9611cec7d82e65b` | clean | ancestor of named refs (4) | unknown | **W3** |
| `/tmp/prl26-gap-fix-NNVbZQ/worktree` | detached | `151701fdb13c5ecf6fac6c0f67760e51c427c277` | clean | UNANCHORED | unknown | **W4** |
| `/tmp/tgvf-true1m-docs-3wDBGl` | `docs-true1m-contract-20260828` | `c6bf780891efaa819c4888d5f3dff9b78c4da322` | clean | exact local branch | unknown | **W2** |

The two W4 rows are not reachable from any named local branch, remote-tracking ref, tag, or stash ref. They must be tagged at their exact SHAs before removal is considered; otherwise they may eventually become garbage-collection eligible.

## Local branches (62/62)

For each `L/R` pair, L is commits unique to the named baseline/upstream and R is commits unique to the local branch (`git rev-list --left-right --count BASE...BRANCH`). `contained` means fully merged into the baseline; `contains`, `equal`, and `diverged` are graph relations.

Branch actions: **B0** keep primary/active; **B1** keep dirty; **B2** remote-backed review candidate (separate exact-ref approval required); **B3** contained/no-upstream review candidate (archive branch-name semantics, then exact-ref approval required); **B4** keep/archive local-only or divergent history.

Counts: B0=2, B1=6, B2=28, B3=2, B4=24.

| Local branch | Tip | Worktree | Dirty | Upstream | Upstream U/B | Current C/B + status | origin/main M/B + status | Artifact refs | Action |
|---|---|---|---|---|---:|---|---|---|---|
| `adapter-ablation-rp67-20260826` | `92b8a4aec3713c3657e3c24897b7efbffdfcdbaf` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-adapter-ablation` | clean | `origin/adapter-ablation-rp67-20260826` | `0/0` | `58/0` contained | `38/0` contained | unknown | **B2** |
| `adapter-architecture-ablation-20260826` | `029ca33f688491434aa3557ab3f35d7a8d158564` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-adapter-architecture` | clean | `origin/adapter-architecture-ablation-20260826` | `0/0` | `53/0` contained | `33/0` contained | unknown | **B2** |
| `codex/framework-implementation` | `983657100848af3d35603e40315f9d888fbc3bde` | — | n/a | `origin/codex/framework-implementation` | `0/0` | `522/0` contained | `502/0` contained | unknown | **B2** |
| `diagnostic-vstar-oracle-crop` | `a434ee91b8dbc24978e250906f761e80c29dec2b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-oracle-crop` | clean | `origin/diagnostic-vstar-oracle-crop` | `0/0` | `143/112` diverged | `123/112` diverged | unknown | **B2** |
| `docs-true1m-contract-20260828` | `c6bf780891efaa819c4888d5f3dff9b78c4da322` | `/tmp/tgvf-true1m-docs-3wDBGl` | clean | `origin/main` | `16/0` | `36/0` contained | `16/0` contained | unknown | **B2** |
| `full-prompt-rl16-plan-20260827` | `029ca33f688491434aa3557ab3f35d7a8d158564` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-full-prompt-rl16` | clean | `origin/main` | `33/0` | `53/0` contained | `33/0` contained | unknown | **B2** |
| `main` | `475f2b1f85a703a8c261b46baaa9f0f92e266b9b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl` | dirty: 38 tracked + 55 untracked | `origin/main` | `42/0` | `62/0` contained | `42/0` contained | unknown | **B0** |
| `main-d-only-dev` | `fe5ea776e6db817db55411fb1bfef53750904cde` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.main-d-only-dev` | clean | — | — | `379/0` contained | `359/0` contained | unknown | **B3** |
| `neurips-notool-rl-s32` | `e1e0e1754828944110f567f9359e4ef2f218f969` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-notool-s32` | clean | `origin/neurips-notool-rl-s32` | `25/0` | `114/258` diverged | `94/258` diverged | unknown | **B2** |
| `prl15-checkpoint-every-step` | `4443fcc070f29b06dd18d59524ac92173c7706ed` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-checkpoint-fix` | clean | — | — | `143/39` diverged | `123/39` diverged | unknown | **B4** |
| `prl15-eval-handoff` | `4b1e1c9c92c74cef21a341245ac4e4fc06b5eb63` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-eval-handoff` | clean | — | — | `143/39` diverged | `123/39` diverged | unknown | **B4** |
| `prl15-eval-live` | `e06450d9a5f30e1d213e64ff8cbf933ce1ccfe4c` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-eval-live` | clean | `origin/prl15-eval-live` | `0/2` | `143/51` diverged | `123/51` diverged | unknown | **B4** |
| `prl15-placeholder-protocol-fix` | `16dc184847b38662613e278f771b5e8ad0cb4c4b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-placeholder-fix` | clean | — | — | `143/45` diverged | `123/45` diverged | unknown | **B4** |
| `prl15-resume-runtime-checkpoint-gate` | `f3f5527ec4bc7844930d278177d3f6cdedc3a9b0` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-resume-schedule-fix` | clean | — | — | `143/41` diverged | `123/41` diverged | unknown | **B4** |
| `prl15-rp66-matched-control` | `7324577ad1f8236cef49a73a0e67eb90aaf42861` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-rp66-matched` | clean | `origin/prl15-rp66-matched-control` | `0/0` | `143/50` diverged | `123/50` diverged | unknown | **B2** |
| `prl15-termination-native-eos-fix` | `a6a4c149398608329579312b55f4057d1461467c` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-termination-fix` | clean | — | — | `143/47` diverged | `123/47` diverged | unknown | **B4** |
| `prl15-tool-attempt-metrics-fix` | `4bf8a7ddd4801864c5b74c02d424afa9a9ba3471` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl15-tool-attempt-fix` | clean | — | — | `143/41` diverged | `123/41` diverged | unknown | **B4** |
| `prl16-f1-eval` | `074fda6d81884180e905114e8101e34b30b482a9` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-f1-eval` | clean | `origin/prl16-f1-eval` | `0/0` | `143/69` diverged | `123/69` diverged | unknown | **B2** |
| `prl16-frozen-rp66-control` | `0ab45b4b132b15a67ee5683c9b2a9e22adfe1c16` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl16-frozen-rp66` | clean | `origin/prl16-frozen-rp66-control` | `0/14` | `143/76` diverged | `123/76` diverged | unknown | **B4** |
| `prl16-judge-failsoft` | `e03461229a6196e7ce0f1c9dca2b645ca164b763` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-judge-failsoft` | clean | — | — | `143/63` diverged | `123/63` diverged | unknown | **B4** |
| `prl16-judge-rng-correctness` | `3f89161a848bb9a48ffff9364ac71381d682883b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-judge-rng-correctness` | clean | — | — | `143/62` diverged | `123/62` diverged | unknown | **B4** |
| `prl16-live-sync-parity` | `3f89161a848bb9a48ffff9364ac71381d682883b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-live-sync-parity` | clean | — | — | `143/62` diverged | `123/62` diverged | unknown | **B4** |
| `prl16-qwen-sync-integrity` | `3f89161a848bb9a48ffff9364ac71381d682883b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl16-qwen-sync-integrity` | clean | — | — | `143/62` diverged | `123/62` diverged | unknown | **B4** |
| `prl17-r1-rp67-shaped-reward` | `e177150f9b859004698017cb0f1760fe39b16374` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r1-rp67-shaped` | clean | — | — | `143/86` diverged | `123/86` diverged | unknown | **B4** |
| `prl17-r2-rp67-calibrated-t` | `0d738adbc4ab257044f0ff3d9013e4371e7e113b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-rp67-calibrated-t` | clean | `origin/prl17-r2-rp67-calibrated-t` | `0/1` | `143/92` diverged | `123/92` diverged | unknown | **B4** |
| `prl17-r2-step16-paired-eval` | `a239c7d88f2bc8a4ec38c5c34e5fcb5dd5f965a3` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-step16-paired-eval` | clean | `origin/prl17-r2-step16-paired-eval` | `0/0` | `143/92` diverged | `123/92` diverged | unknown | **B2** |
| `prl17-r2-tfree-step16-continuation` | `c4e24f74e9d603ed2cbad79640f032fc62edad68` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-r2-tfree-step16` | clean | `origin/prl17-r2-tfree-step16-continuation` | `0/0` | `143/94` diverged | `123/94` diverged | unknown | **B2** |
| `prl17-rp66-shaped-reward` | `e7047d6979f0a2eaef6ed64838b2dc052ea54a3c` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl17-rp66-shaped` | clean | — | — | `143/83` diverged | `123/83` diverged | unknown | **B4** |
| `prl18-r0-unfrozen-rp67` | `d4ab515d4e2eb2efc23fad91dfeb198dc0132cb0` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl18-r0-unfrozen-rp67` | clean | `origin/prl18-r0-unfrozen-rp67` | `0/0` | `143/98` diverged | `123/98` diverged | unknown | **B2** |
| `prl19-r0-rp67-frozen-tfree-visual-api` | `9333fd429ecc7075258ef0d8f97cea5fefc49da9` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl19-r0-rp67-frozen-tfree-visual-api` | clean | `origin/prl19-r0-rp67-frozen-tfree-visual-api` | `0/0` | `143/111` diverged | `123/111` diverged | unknown | **B2** |
| `prl20-r0-rp67-frozen-tfree-crop-tgvf` | `fc5e5dd99819dcb923ebf207d496be6f5fcb540a` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-crop-tgvf` | clean | `origin/prl20-r0-rp67-frozen-tfree-crop-tgvf` | `0/0` | `143/115` diverged | `123/115` diverged | unknown | **B2** |
| `prl20-r0-step16-continuation` | `6b38a03c11c893e714bee6d0716cae9f73735357` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-step16` | clean | `origin/prl20-r0-step16-continuation` | `0/0` | `143/119` diverged | `123/119` diverged | unknown | **B2** |
| `prl20-r0-step8-step16-eval` | `52f40853b71ec4a24efd0304cdb5036cd462f39f` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl20-r0-eval` | clean | `origin/prl20-r0-step8-step16-eval` | `0/0` | `143/116` diverged | `123/116` diverged | unknown | **B2** |
| `prl21-r0-crop-tfree-16step` | `b3d08b4ed657479250e449e57376781bd2cd8c77` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/prl21-r0-crop-tfree-16step` | clean | `origin/prl21-r0-crop-tfree-16step` | `0/0` | `143/129` diverged | `123/129` diverged | unknown | **B2** |
| `prl22-a-tgvf-teacher25` | `0b2292a469ecf0864b4e4ec85f545ea834ea1fb3` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-a-tgvf-teacher25` | clean | `origin/prl22-a-tgvf-teacher25` | `0/0` | `143/133` diverged | `123/133` diverged | unknown | **B2** |
| `prl22-b-crop-tgvf-teacher25` | `566a1fe4150ea61725944b134061209c5f8cb99c` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-b-crop-tgvf-teacher25` | clean | `origin/prl22-b-crop-tgvf-teacher25` | `0/0` | `143/134` diverged | `123/134` diverged | unknown | **B2** |
| `prl22-c-crop-teacher25` | `3610f2dde2491194b3ada4fd0b1bee974035c10d` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-c-crop-teacher25` | clean | `origin/prl22-c-crop-teacher25` | `0/0` | `143/131` diverged | `123/131` diverged | unknown | **B2** |
| `prl22-teacher25-shared` | `37b99e2b01f9459e0ee65f6b86e2950bf60d4417` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl22-teacher25-shared` | clean | `origin/prl22-teacher25-shared` | `0/0` | `143/130` diverged | `123/130` diverged | unknown | **B2** |
| `prl23-teacher-ratio-launchers` | `02444c358f5d8bca78be7a206954bed14d4a548b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl23-teacher-ratio-launchers` | clean | — | — | `143/134` diverged | `123/134` diverged | unknown | **B4** |
| `prl23-tgvf-teacher-ratio` | `0bf01727bceadae87ae8e488ae6075f3f10f3aad` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl23-tgvf-teacher-ratio` | clean | `origin/prl23-tgvf-teacher-ratio` | `0/0` | `143/137` diverged | `123/137` diverged | unknown | **B2** |
| `prl24-a-bs64` | `2f07d89f8291113dddeb20f88c7cf843f3bf86a3` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-a-bs64` | dirty: 0 tracked + 3 untracked | `origin/prl24-a-bs64` | `0/0` | `143/155` diverged | `123/155` diverged | unknown | **B1** |
| `prl24-d-crop-bs64-fmt2` | `c5bc2eab3cfa199ff96c515d816226c8a78c829e` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-d-crop-bs64-fmt2` | clean | — | — | `143/166` diverged | `123/166` diverged | unknown | **B4** |
| `prl24-d-crop-bs64-fmt2-sp2` | `c5bc2eab3cfa199ff96c515d816226c8a78c829e` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-d-crop-bs64-fmt2-sp2` | dirty: 7 tracked + 0 untracked | — | — | `143/166` diverged | `123/166` diverged | unknown | **B1** |
| `prl24-s24-s32-paired-eval` | `27fd1ebeaa79f6a5e1750b15ef1e0a277386596b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-s24-s32-eval` | clean | — | — | `143/144` diverged | `123/144` diverged | unknown | **B4** |
| `prl24-supervisor-config-driven` | `2bf1b835beb9b1394b2d795b5d22cd5b78418443` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-supervisor-refactor` | clean | — | — | `143/143` diverged | `123/143` diverged | unknown | **B4** |
| `prl25-c-tgvf-80step` | `e43662998a6b76741fe6f36ab0d34d376b436040` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl25-crop-aligned` | dirty: 6 tracked + 13 untracked | `origin/prl25-c-tgvf-80step` | `0/0` | `114/200` diverged | `94/200` diverged | unknown | **B1** |
| `prl25-crop-exact-replay-alignment` | `2bbb5f9309d6b1d9dca25a81bd1855dbd100859e` | — | n/a | `origin/prl25-crop-exact-replay-alignment` | `0/0` | `120/187` diverged | `100/187` diverged | unknown | **B2** |
| `prl26b-generic86-historical-eval-20260830` | `001838b65a0655008fa066a08934f32bbc9bffb6` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-historical` | dirty: 1 tracked + 3 untracked | — | — | `114/253` diverged | `94/253` diverged | unknown | **B1** |
| `prl26b-generic86-matched-eval-20260830` | `1dc0d1ec5eb233397b38bba9adf0f01812a79845` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-eval` | dirty: 1 tracked + 0 untracked | `origin/prl26b-generic86-matched-eval-20260830` | `0/0` | `114/274` diverged | `94/274` diverged | unknown | **B1** |
| `prl27-crop-replay-renderer-fix-20260830` | `a779625c5ef309aed77d68ab202a96a8ad11830e` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27-replay-fix` | dirty: 2 tracked + 0 untracked | `origin/prl27-crop-replay-renderer-fix-20260830` | `0/0` | `114/283` diverged | `94/283` diverged | unknown | **B1** |
| `prl27-real-crop-canary-20260830` | `4fe6b3d931aedc19a4a4d9a7775a7850c4aa7f2b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27-crop-canary` | clean | — | — | `114/260` diverged | `94/260` diverged | unknown | **B4** |
| `prl27b-corrected-crop-relaunch-20260830` | `300a1ee33a4df8a68ba3add835097f61d91f63eb` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-relaunch` | clean | — | — | `114/261` diverged | `94/261` diverged | unknown | **B4** |
| `prl27b-corrected-crop-relaunch-squashed-20260830` | `6217a2a305350d37b3704dcfeaf09255c405b6e5` | — | n/a | — | — | `114/260` diverged | `94/260` diverged | unknown | **B4** |
| `prl27b-eval-runtime-20260830` | `b6fdb73cfe926b6131b86e0a5c97a4af52940e3e` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-eval-runtime` | clean | `origin/prl27b-eval-runtime-20260830` | `0/0` | `114/265` diverged | `94/265` diverged | unknown | **B2** |
| `prl27b-eval-waiter-20260830` | `a9e4b67c6e36e1efdefd1cb513b0aa86939b254d` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-eval-waiter` | clean | — | — | `114/262` diverged | `94/262` diverged | unknown | **B4** |
| `prl27b-training-runtime-20260830` | `f50fe3c66c719dd10f5dc5522e5142594831038b` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27b-training-runtime` | clean | `origin/prl27b-training-runtime-20260830` | `0/0` | `114/263` diverged | `94/263` diverged | unknown | **B2** |
| `stabilize/protocol-contract-v1-20260830` | `05e32ea4e767e0230b6c9f54ac42c296e55a243e` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-stabilize-protocol-v1` | clean | `origin/stabilize/protocol-contract-v1-20260830` | `0/13` | `0/0` equal | `0/20` contains | unknown | **B0** |
| `target-visual-descriptor-rl16-20260827` | `c0cebefd1ebc9ca2ddd91482a340f0d4b755e0b7` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-target-v2` | clean | `origin/prl25-c-tgvf-80step` | `2/49` | `114/247` diverged | `94/247` diverged | unknown | **B4** |
| `texture-benchmarks` | `913123e3539a6440219fb43163d56e221cb0900c` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-texture-bench` | clean | — | — | `140/0` contained | `120/0` contained | unknown | **B3** |
| `texture-benchmarks-current` | `abd652f6e98cec04fe912dfa5381124dcc3e08c4` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-texture-bench-current` | clean | `origin/texture-benchmarks-current` | `0/0` | `143/101` diverged | `123/101` diverged | unknown | **B2** |
| `texture-benchmarks-prl20-integration` | `5015ed3dd1b06651b74f37273b9e35a9d5bc709d` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/texture-prl20-integration` | clean | `origin/texture-benchmarks-prl20-integration` | `0/0` | `143/124` diverged | `123/124` diverged | unknown | **B2** |
| `texture-benchmarks-prl21-step16` | `989fd038e2931fe928a2d2ac4c7f590f9f9838f9` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.worktrees/texture-prl21-step16` | clean | `origin/texture-benchmarks-prl21-step16` | `0/0` | `143/140` diverged | `123/140` diverged | unknown | **B2** |

## Other refs

| Ref | Object | Peeled commit / subject | Remote state |
|---|---|---|---|
| `stabilization-c0-20260830` | `f60b6af853816a48d1f4e343c717ae05232af677` | `87f70ba44acd9e359fa4e2f83bd24ef8d5f4f9ce`; C0 audited stabilization baseline before codebase consolidation | exact object and peeled commit verified on origin |
| `stash` | `3a17e73b0d74d1f7f3178cd9ac7e102b999669bc` | On texture-benchmarks: texture-benchmark-transfer-20260813 | local preservation ref; not a cleanup candidate |

## Recoverable sequence

1. Keep W0, every W1/B1, every W4, and every B4 unchanged.
2. Request approval to create exact annotated archive tags for both W4 SHAs; verify locally and remotely.
3. Build an authoritative reverse index across checked-in and ignored/external artifacts; until then keep `artifact refs = unknown`.
4. Present exact W2/W3 worktree paths with owner/process evidence; each removal requires exact user approval and retains a branch/tag anchor.
5. Re-run this inventory, then separately request exact approval for each B2/B3 local ref.
6. Never combine worktree, local-branch, and remote-branch deletion into implied approval.

No cleanup action was executed.

## Reproduction and integrity

```bash
git worktree list --porcelain
git -C <exact-worktree> status --porcelain=v1 --untracked-files=normal
git for-each-ref --format='%(refname:short)|%(objectname)|%(upstream:short)|%(worktreepath)' refs/heads
git rev-list --left-right --count <frozen-current-sha>...<branch>
git rev-list --left-right --count origin/main...<branch>
git for-each-ref --points-at <detached-head>
git for-each-ref --contains <detached-head>
git ls-remote --symref origin HEAD
git ls-remote --heads origin
```

- heads/ref digest: `637fa1eceeecab4d509802ddc7827d456a2fea867610f8fbf3b219b8cad487e4`
- worktree topology digest: `f8b4d1d9030cd755b62e449a08a3d6de12b5b426fa959df208687de7968b8d99`
- ordered status digest (self-file excluded): `351149001f7de70d765c970e018ffb2585c1e4f54fbe5359ff3ffea730127ee7`
- live origin-heads digest: `cdda32434005b04008d1b8097b24a03c3e14db6cd66b844f5721adcb2ec1fc49`

These digests matched on the pre-publication recheck.

