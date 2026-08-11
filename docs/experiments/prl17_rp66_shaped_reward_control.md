# PRL17: RP66 matched shaped-reward control

## Question

Does the previously implemented Stage3-shaped reward improve the frozen-RP66
TGVF policy when it is tested at the same larger-batch protocol as the PRL16
F2 baseline?

This is a reward-only control. It is not a new reward design and must not be
interpreted as a test of a new RP66 representation.

## Matched control

PRL17 inherits the PRL16 F2 execution contract:

- base model: `Qwen3-VL-8B-Instruct`;
- representation: RP66 step 2000, frozen during RL;
- global prompt batch: 16;
- trajectories per prompt: 16;
- world size: 8, prompt micro-batch per rank: 2, GA: 1;
- optimizer steps: 8, constant learning rate `1e-6`;
- at most six TGVF calls;
- the same TGVF prompt, stratified T1 schedule, sampling contract, async
  Qwen2.5-72B answer judge, checkpoints, and post-training benchmark protocol.

The only intended experimental variable is the trajectory reward scalar.

## Active reward

The complete Stage3 kernel is

\[
R = 2A_{\mathrm{gated}} + T + F + G + P.
\]

For this control, Focus and Grounding are explicitly disabled, so the executed
equation is

\[
R = 2A_{\mathrm{gated}} + T + P.
\]

- `A_gated`: a verified correct answer receives one before the factor of two.
  A `needed` sample with no successful TGVF observation gates this component
  to zero.
- `T`: the immutable counterfactual label is `needed`, `optional`, or
  `unnecessary`. With the established confidence 0.5, the used/not-used
  contributions are respectively `+0.5/-1.0`, `+0.25/0.0`, and
  `-0.25/+0.5`. Every attempted call after the first contributes `-0.05`.
- `P`: any protocol/tool error contributes `-1.0`; multiple error codes do not
  stack additional protocol penalties.
- `F` and `G`: exactly zero and no visual judge request is made.

If Focus or Grounding is tested later, it must use an API visual judge. A local
32B visual judge is not part of PRL17.

The answer judge keeps the matched Qwen2.5-72B model, DeepInfra-only route,
run-global concurrency 16 (deterministically sharded as two permits in each of
the eight AgentLoop worker processes), four attempts, prompts, sampling, and
sample-local zero fallback. Its transient-health window is tightened to 16 requests at 25%:
up to four transient failures remain sample-local, while the fifth failure in
one window aborts before an optimizer update. This prevents a provider outage
from silently turning an entire answer component into zeros without changing
the reward of any successfully judged trajectory.

## Utility-label artifact

The historical shaped-reward sidecar follows the old sequential small-batch
schedule and cannot be reused: only 4 of the first 128 PRL17 samples are
covered. PRL17 therefore materializes the exact first 128 sample IDs from the
verified DeepEyes stratified schedule and reruns the established eight-attempt
forced-RP66 counterfactual procedure.

The completed sidecar contains 128 rows: 36 `needed`, 53 `optional`, and 39
`unnecessary`. Its payload SHA-256 is
`1bd0bcda438bb542f9378c36a3800351318cfbf008fa5af1b9564c1067d8b3d2` and
its manifest SHA-256 is
`f967af7f3e1ebe15a5a8ffdf7724d7e3a48a7acdcf9923a8d9459bd1b02dd04c`.

For each sample,

\[
\Delta = p_{\mathrm{TGVF}} - p_{\mathrm{full}}.
\]

The established thresholds are `needed` for `Delta >= 0.25`, `unnecessary` for
`Delta <= -0.25`, and `optional` otherwise. The new artifact is create-only;
the historical sidecar is not modified.

## Required gates

1. Unit tests prove that legacy Stage3 configs retain the visual-quality path,
   while PRL17 makes no Focus/Grounding judge call.
2. The 128-row sidecar must match the exact training sample IDs and order.
3. A CPU config/identity preflight must pass.
4. A minimal GPU smoke must emit all five component metrics, with Focus and
   Grounding identically zero and no visual-judge usage.
5. Only then may the matched 8-step run start; smoke runs do not upload to
   Weights & Biases.
