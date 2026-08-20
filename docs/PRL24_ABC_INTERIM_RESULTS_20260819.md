# PRL24 A/B/C 阶段性结果与 BS64 暂停决策

更新日期：2026-08-20。

当前 A/B/C 可比较结果均使用 Qwen3-VL-8B-Instruct、RP67 Step-2000、
Teacher25、BS64 × n16、world8、full-policy update、constant LR `1e-6` 和
FMT2。A 为 Frozen/T-free，B 仅改为 Joint/Unfrozen Adapter，C 相对 A 开启
API Focus/Target 与 Grounding reward。

## FMT1 → FMT2 的版本决策

PRL24 最初的 A 使用 FMT1：发生 protocol/format/tool error 时罚分 `-1`。随后发现
format error rate 会随训练步数继续升高，`-1` 不足以约束这种退化，因此新建 FMT2
recipe，将同一错误事件的罚分改为 `-2`：

```text
FMT1: R = 2*A - 0.05*max(0, N_attempt-1) - 1*[protocol_or_tool_error]
FMT2: R = 2*A - 0.05*max(0, N_attempt-1) - 2*[protocol_or_tool_error]
```

这不是对旧 A 的静默改写：FMT1 A 保持独立历史身份，不能与 FMT2 checkpoint 拼接，
也不能把其结果混入下表。A/B/C 的正式横向结论全部以重新运行的 FMT2 身份为准；从
2026-08-20 起，已执行的 D-S1 以及后续第三期新训练统一采用 FMT2，除非另立明确的
reward-ablation ID。PRL24-E/F 最终未启动。

## 统一 CoreDev-2511 结果

单位为 `%`。Macro* 严格采用 VStar、HR Average/all、BLINK 单图 180、OCR
EN/CN mean、MMMU 单图 269、MathVista、MathVerse five-version macro 七项等权均值。

| 指标 | A S4 | A S8 | A S12 | A S16 | B S8 | C S4 | C S8 | C S12 | C S16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VStar | 72.77 | 73.82 | 68.06 | 71.73 | 63.87 | 73.30 | 73.30 | 69.63 | 68.59 |
| HR Average/all | 62.50 | 65.00 | 65.50 | 65.50 | 64.50 | 62.50 | 65.50 | 68.00 | 64.00 |
| BLINK single-180 | 58.33 | 66.67 | 65.00 | 60.56 | 61.11 | 65.00 | 68.33 | 67.78 | 66.67 |
| OCR EN/CN mean | 42.95 | 42.34 | 42.17 | 42.15 | 40.61 | 43.89 | 42.51 | 39.44 | 40.12 |
| MMMU single-269 | 46.10 | 50.19 | 50.56 | 49.07 | 47.96 | 47.96 | 50.19 | 49.44 | 49.07 |
| MathVista | 68.33 | 70.33 | 73.00 | 71.00 | 70.33 | 70.00 | 68.00 | 72.33 | 73.00 |
| MathVerse five-version | 57.80 | 56.80 | 54.80 | 55.80 | 54.00 | 58.60 | 56.40 | 54.60 | 55.00 |
| **Macro\*** | **58.3983** | **60.7348** | **59.8701** | **59.4007** | **57.4834** | **60.1777** | **60.6036** | **60.1754** | **59.4924** |

## 已落实的结论

- A 在 S8 达到当前峰值 `60.7348`，S12/S16 回落；BS64 带来较快的早期提升，
  但没有消除后期平台与回落。
- B S8 比 matched A S8 低 `3.2514 pp`，且 6/7 分量下降。这个负结果与上一轮
  BS16 Joint pilot 的方向一致，已经足以支持当前工程决策：**保留 Frozen RP67
  Adapter**。B 因而在 S8 有意停止，没有训练到 S16；这是一次有记录的资源分配/
  预注册偏离，不得把 B 写成“已完成 S16”，也不再把补跑 B-S16 当作 D 的前置条件。
- C 已训练并评测至 S16。C−A 在 S4/S8/S12/S16 分别为
  `+1.7794/-0.1312/+0.3053/+0.0917 pp`：F/G 有明确早期正信号，但没有形成持续的
  endpoint accuracy 增益，因此不升级为默认 reward。其 foveation/hallucination
  价值仍应由专项健康度审计回答。
- 这些均是 `temperature=1` 单次 paired-seed 结果；小于约 `1 pp` 的差异默认
  视为不确定。

## A0 到底是什么

A0 不是现有结果，也不是第三期之前必须补跑的 arm。它只是一个“严格 batch 因果”
确认项：如果要声称差异**只**来自 BS16→BS64，就需在同一 commit、同一 FMT2 recipe
下补一个 BS16 A0。历史 PRL22 BS16 与当前运行在代码身份及 FMT penalty 上并不完全
matched，因此目前可以报告 recipe-level scaling evidence，但不能把历史对比写成
严格的 batch-only causal effect。若论文未来确需该强措辞，再单独安排 FMT2/BS16 A0；
当前暂停 BS64 后，A0 不再是近期优先项。

## BS64 暂停与下一步

现有 TGVF recipe-level 对照没有显示明确 BS64 质量增益：等 exposure 的 A-S4
`58.3983` 相对历史 BS16 Teacher25 TGVF-S16 `59.9590` 为 `-1.5607 pp`；A 的最佳
S8 `60.7348` 只比历史 BS16 最佳点高 `+0.7758 pp`，低于约 `1 pp` 的默认不确定
区间；A-S16 又低 `-0.5583 pp`。由于 FMT/执行 commit 不完全 matched，这不是严格
batch-only 因果结论，但不足以支持继续优先投入 BS64。

PRL24-D 的优化执行体完成了一个 native Crop、Teacher25、BS64 × n16、FMT2 的正式
S1 optimizer checkpoint，并在 S2 update 前停止。该 step 从 rollout 开始到 checkpoint
完整落盘约 `2 h 32 min`，actor update 约 `2 h 22 min`，完整 checkpoint 约
`140.3 GB`。S1 rollout 由更新前 S0 policy 产生，而且停止早于最终 metrics/trajectory
publication，因此 D-S1 不是 efficacy 结果；不得声称 Crop 已改善或退化。E/F 未启动。

当前决策是暂停 PRL24 BS64，转入
[第三期 PRL25 BS16 Teacher25 80-step 统一计划](PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md)。
第三期包含 Crop conditional/T-free、pure TGVF、Atomic Crop+TGVF、TGVF F+G 五个 arm，
全部从 S0 新训，不续接 PRL24 checkpoint。

## 口径警告

不得直接使用 `official_summary.primary_metric` 计算本文 Macro*。尤其不得使用 HR
cycle-0、BLINK/MMMU zero-padded 全集、OCR Chinese-only 或 MathVerse
Text-Dominant 代替上述冻结口径。

## Canonical artifacts

```text
FMT1 A (historical only):
  artifacts/policy/PRL-24-A-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-16step-ws8/
FMT2 A:
  artifacts/policy/PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8/
  evaluation/*STEP4-STEP8*/paired-summary.json
  evaluation/*STEP12-STEP16*/paired-summary.json
FMT2 B (intentionally stopped at S8):
  artifacts/policy/PRL-24-B-FMT2-JOINT-qwen3-instruct-full-joint-rp67-bs64-n16-tfree-teacher25-8step-ws8/
  evaluation/*STEP8*/paired-summary.json
FMT2 C (completed through S16):
  artifacts/policy/PRL-24-C-FMT2-FG-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8/
  evaluation/*STEP4-STEP8*/paired-summary.json
  evaluation/*STEP12-STEP16*/paired-summary.json
FMT2 D (stopped after complete S1; non-efficacy checkpoint):
  artifacts/policy/PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8-sp1/
  checkpoints/global_step_1/
```
