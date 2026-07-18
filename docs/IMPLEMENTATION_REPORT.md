# I8H-20260719 Framework Implementation Report

Status: **in progress**
Branch: `codex/framework-implementation`
Base commit: `168b3ab`
Authorization commit: `2ffa28e`

## Implemented evidence so far

- framework-neutral identity, token ownership, actual behavior-logprob,
  immutable tensor, focused observation, trajectory, and strict checkpoint
  contracts;
- content-addressed observation store with bit-preserving BF16 round trip;
- native `tgvf_focus_tool` schema, strict parser, repeated-call state machine,
  native Qwen3 transcript golden hashes, and no tokenizer growth;
- batch-aware TGVF Adapter with main `D`, required D-DeepStack branches, frozen
  projection ports, and both target-conditioning providers;
- Qwen3 exact recorded-`D` replay forward and a separately scoped Qwen2.5
  main-`D` family path;
- framework-neutral multi-call rollout loop and decomposed fail-closed reward,
  judge, and run configuration boundaries.

## Compatibility finding under validation

The selected upstream veRL source is pinned at
`e003163181731412595257a72ec173071efb125f`. Static inspection found a version
identity inconsistency in that exact tree:

- `setup.py` declares `vllm>=0.8.5,<=0.12.0` and resolves veRL version
  `0.9.0.dev0`;
- `docker/Dockerfile.stable.vllm` declares vLLM `0.23.0`, Torch `2.11.0`, and
  installs veRL `v0.7.1` before uninstalling it.

The first isolated import smoke therefore uses the package-metadata path:
vLLM `0.12.0`, Torch `2.9.0`, Transformers `4.57.6`, and the exact local veRL
checkout. No production pin will be claimed until import/public-hook/FSDP2
evidence is recorded.

## Deferred items

Real data, reward, prompt, production objective mathematics, long training,
and the 72B judge remain deliberately unset. See
[`DEFERRED_DECISIONS.md`](DEFERRED_DECISIONS.md).
