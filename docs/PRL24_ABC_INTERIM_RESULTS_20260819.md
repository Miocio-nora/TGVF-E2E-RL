# PRL24 A/B/C 阶段性结果

更新日期：2026-08-19。

三组均使用 Qwen3-VL-8B-Instruct、RP67 Step-2000、Teacher25、BS64 × n16、
world8、full-policy update、constant LR `1e-6` 和 FMT2（protocol/format error
penalty `-2`）。A 为 Frozen/T-free，B 仅改为 Joint/Unfrozen Adapter，C 相对 A
开启 API Focus/Target 与 Grounding reward。

## 统一 CoreDev-2511 结果

单位为 `%`。Macro* 严格采用 VStar、HR Average/all、BLINK 单图 180、OCR
EN/CN mean、MMMU 单图 269、MathVista、MathVerse five-version macro 七项等权均值。

| 指标 | A S4 | A S8 | A S12 | A S16 | B S8 | C S4 | C S8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStar | 72.77 | 73.82 | 68.06 | 71.73 | 63.87 | 73.30 | 73.30 |
| HR Average/all | 62.50 | 65.00 | 65.50 | 65.50 | 64.50 | 62.50 | 65.50 |
| BLINK single-180 | 58.33 | 66.67 | 65.00 | 60.56 | 61.11 | 65.00 | 68.33 |
| OCR EN/CN mean | 42.95 | 42.34 | 42.17 | 42.15 | 40.61 | 43.89 | 42.51 |
| MMMU single-269 | 46.10 | 50.19 | 50.56 | 49.07 | 47.96 | 47.96 | 50.19 |
| MathVista | 68.33 | 70.33 | 73.00 | 71.00 | 70.33 | 70.00 | 68.00 |
| MathVerse five-version | 57.80 | 56.80 | 54.80 | 55.80 | 54.00 | 58.60 | 56.40 |
| **Macro\*** | **58.3983** | **60.7348** | **59.8701** | **59.4007** | **57.4834** | **60.1777** | **60.6036** |

## 当前结论

- A 在 S8 达到当前峰值 `60.7348`，S12/S16 回落，说明 BS64 加快了早期提升，
  但没有消除后期平台与回落。
- B S8 比 matched A S8 低 `3.2514 pp`，且六个七分量下降；当前证据继续支持
  **冻结 RP67 Adapter**。
- C S4 比 A S4 高 `1.7794 pp`，但 C S8 比 A S8 低 `0.1312 pp`。F/G 有早期
  正信号，最终 accuracy 基本持平，暂不作为默认 reward；其 foveation/hallucination
  价值仍需专项审计。
- B 目前只有 S8 正式外评；C 目前只有 S4/S8，不能写成已完成预注册 S16。
- 这些均是 `temperature=1` 单次 paired-seed 结果；小于约 `1 pp` 的差异默认视为不确定。

## 口径警告

不得直接使用 `official_summary.primary_metric` 计算本文 Macro*。尤其不得使用 HR
cycle-0、BLINK/MMMU zero-padded 全集、OCR Chinese-only 或 MathVerse
Text-Dominant 代替上述冻结口径。

## Canonical artifacts

```text
A: artifacts/policy/PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8/
   evaluation/*STEP4-STEP8*/paired-summary.json
   evaluation/*STEP12-STEP16*/paired-summary.json
B: artifacts/policy/PRL-24-B-FMT2-JOINT-qwen3-instruct-full-joint-rp67-bs64-n16-tfree-teacher25-8step-ws8/
   evaluation/*STEP8*/paired-summary.json
C: artifacts/policy/PRL-24-C-FMT2-FG-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8/
   evaluation/*STEP4-STEP8*/paired-summary.json
```
