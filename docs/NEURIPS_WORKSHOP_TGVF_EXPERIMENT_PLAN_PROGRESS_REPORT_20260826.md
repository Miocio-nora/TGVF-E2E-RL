# NeurIPS Workshop：TGVF 文章实验计划、推进台账与阶段报告

更新时间：2026-08-29 16:35 JST（Asia/Tokyo）

> **当前权威口径：** 旧 Crop processor-default S32/S80 `61.6699/59.1785` 不是 true-1M，
> 仅作历史记录。Crop S32/S80 已以有效 `max_pixels=1,003,520`、fixed `</tool_call>`
> action boundary、`2,240/2,240` 支持集和七项 scorer 重跑闭合。Original raw-direct true-1M
> 也已完成 2,511 行正式评分并通过验收，Macro* 为 `61.3147`。No-Tool S0/S8/S16/S32 的
> corrected true-1M V2 现已全部闭合；事前冻结 S32 Macro* 为 `63.7520`。冻结定义、完整证据与实时状态见
> [项目级 true-1M 唯一合同](COREDEV2511_TRUE1M_UNIFIED_MEASUREMENT_CONTRACT_20260828.md)。

## Stage golden result under the unified true-1M contract

这是本阶段统一输入预算下最具参考价值的五方端到端比较：五行评测都使用 `1,003,520`、
同一 `2,240` 条支持集、七项统计与 scorer；No-Tool、Crop、TGVF 和 Atomic 的 policy RL 训练也
使用 `1,003,520`。No-Tool S32 是事前冻结的唯一 no-tool headline。单位为 `%`。

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct | 61.3147 | 72.7749 | 66.5000 | 63.8889 | **59.7877** | 37.9182 | **75.3333** | 53.0000 |
| No-Tool RL S32 | **63.7520** | 70.1571 | 65.5000 | **71.1111** | 49.3928 | **53.9033** | 75.0000 | **61.2000** |
| Crop S32 | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| TGVF S64 | 59.8086 | **74.3455** | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | **73.5000** | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |

Results 层面，No-Tool S32 `63.7520` > Atomic S16 `63.0827` > Original `61.3147` > Crop S32
`61.0706` > TGVF S64 `59.8086`，相邻差值为 `0.6693 / 1.7680 / 0.2442 / 1.2619 pp`。
No-Tool 在 Macro*、BLINK-180、MMMU-269 与 MathVerse 最高；Original 在 OCR mean 与 MathVista
最高；TGVF 在 VStar 最高；Atomic 在 HR 最高。加入 No-Tool 后的 59-slice 胜出数与六方法
pairwise W/L/T 已在 Appendix A 机械重算；该统计只作描述性能力地图，不应解读为 59 个独立假设检验。

Discussion 层面，这一统一像素合同排除了旧 Crop/No-Tool processor-default 运行的主要混杂，
但仍不是严格单变量因果消融：Original 是 raw-direct；No-Tool 使用训练匹配的 no-tool prompt 与
direct-only loop；三条工具 RL 方法的 prompt、工具协议与 checkpoint step 也不同。No-Tool S32
领先 Atomic/Crop/TGVF `0.6693/2.6814/3.9434 pp`，只支持统一像素下的端到端排序，不支持
“关闭工具本身因果更优”。同一 No-Tool 合同内 S32−S0 仅 `−0.0098 pp`，因此 32-step RL 的
aggregate 结果近似持平。RP67 adapter 预训练仍为 `512²`，但 TGVF/Atomic policy RL 与本表评测
均为 true-1M。

Crop S80 true-1M 也已闭合：Macro* `59.6463`，七项为
`73.8220/69.5000/59.4444/54.0426/44.9814/63.3333/52.4000`。它低于 Crop S32，因此
S32 取代 S80 进入当前 golden 主表；S80 保留为 80-step 优化动态证据。

### 有效 true-512 四方表

| Method / effective contract | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct@512 | 55.3556 | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| Crop S80@512, boundary-fix V2 | **62.0967** | **86.9110** | **65.5000** | 64.4444 | **54.4299** | 45.7249 | 65.6667 | 52.0000 |
| TGVF S64@512 | 55.4067 | 53.4031 | 60.5000 | 62.7778 | 40.7642 | 46.4684 | 72.3333 | 51.6000 |
| Atomic S16@512 | 57.2762 | 57.0681 | 59.5000 | 61.6667 | 47.7713 | **48.3271** | 71.0000 | **55.6000** |

四行的实际像素上限均为 `262,144`，但 prompt 与 agent 协议不统一：Original 是 raw-direct，
其余三行是各自训练匹配的工具协议。跨行差值因此只是端到端观测。旧 boundary Crop S80@512
`61.5591` 已被 boundary-fix V2 `62.0967` supersede；Atomic S16@512 的七项评分已完成。

### Method-specific resolution response

| Method | true-1M Macro* | true-512 Macro* | Δ (512 − 1M) | 配对边界 |
|---|---:|---:|---:|---|
| Crop S80 | 59.6463 | 62.0967 | **+2.4504 pp** | RNG namespace 不同；只作 descriptive response |
| TGVF S64 | 59.8086 | 55.4067 | **−4.4019 pp** | same run/step/frozen config/RP67/prompt/tool/task/RNG/scorer；snapshot semantic equality 未闭合 |
| Atomic S16 | 63.0827 | 57.2762 | **−5.8065 pp** | same run/step/frozen config/RP67/prompt/tool/task/RNG/scorer；snapshot semantic equality 未闭合 |

Results 层面，TGVF 和 Atomic 在偏离 policy-RL 1M rollout 尺度的 512 评测上分别回落
`4.4019/5.8065 pp`。Discussion 层面，该高度匹配观测 **consistent with** TGVF-family 对 policy-RL
训练/评测尺度对齐的依赖，但不是因果证明。RP67 adapter 预训练为 `512²`，所以此处
“alignment”只指 policy rollout/evaluation distribution。两臂同 run/step 并绑定同 frozen policy config
SHA 与 RP67 state，但分别 materialize 的 Qwen snapshot 具有不同 tree/combined-weight byte SHA，
receipt 未闭合 tensor-semantic equality。因果确认需要复用同一 snapshot 或提供 semantic tensor hash，
并执行 512-only 或 multi-scale TGVF/Atomic 训练 × 512/1M 评测 factorial。

Crop 的 `+2.4504 pp` 因两臂 RNG namespace 不同而只作 descriptive。在 true-512 四方表中，
Crop 比 Original/TGVF/Atomic 高 `6.7411/6.6900/4.8205 pp`，并在 Macro*、VStar、HR 和 OCR mean
排名第一。这支持 Crop 系统在 constrained-pixel setting 下的 end-to-end effectiveness，但不能
把跨方法差值孤立归因给 crop tool。

### Crop true-1M 工具行为与审计键

| Crop checkpoint | 至少一次成功工具调用的题数 | 成功调用率 | 实际调用次数 | `invalid_crop` | `tool_call_cap` |
|---|---:|---:|---:|---:|---:|
| S32 | 1,423 / 2,240 | 63.53% | 1,769 | 92 | 11 |
| S80 | 2,006 / 2,240 | 89.55% | 2,043 | 102 | 0 |

fixed action boundary 已阻止旧 answer-over-action 路径。S80 的结构审计另发现 `1` 个未闭合的第二
`<tool_call>` opener；它未被执行为合法调用，但作为语法边界明示保留。S32/S80 的 audit
receipt identity 分别为 `0109f7f4f602106bf71bca50309019ae248962387dd7aa33f2b8e12c65042581`
与 `fac02bd313a2c9786b93f7927dccb25403f7b6421aba3256e21d40a97757bbe9`；对应
`paired-summary.json` SHA256 分别为 `a3aa1befedb509e48b69edf4822d7e0467bca346b866da6fd78378d3265ea87d`
与 `715ae0aca71532804942ef5f301afb31c99080d8ec78768b9a10d2be8f448ae5`。

状态：**Original、No-Tool、Crop、TGVF 与 Atomic 的 true-1M stage-golden 比较已闭合。旧
No-Tool processor-default 数值不进入 true-1M 主表，只保留为 historical。**

### PRL26 unified Train@512/Eval@512 live status

文首 true-1M 表仍是当前已闭合的 stage golden result。PRL26 是尚未产出 benchmark 分数的新一轮
fresh-S0 对照，用于回答从 RL 开始即固定 `262,144` pixels 时，No-Tool、Crop 与 TGVF 的表现及
prompt sensitivity 是否改变；在正式 scorer 闭合前，训练 reward 和工具调用统计只作健康诊断。

| Arm | 状态（2026-08-29 16:35 JST） | 已验收边界 | 下一自动动作 |
|---|---|---|---|
| No-Tool Train@512 | **S32 完成** | S8/S16/S24/S32 permanent receipts；S32 run identity `c079c678...` | 等待 Crop S32 后进入 aligned/matched Eval@512 |
| Crop Train@512 | **S25 完成，S26 运行中** | S24 permanent receipt；S25 为 256/258 次成功调用、2 execution failures | S32 后触发 A/B Eval@512 |
| TGVF Short Train@512 | **已冻结、未启动** | prompt、tool、RP67、@512 processor 与 S32 合同已预检 | A/B 评测完成且 GPU/Ray 释放后启动 |
| TGVF Target-guide-v2 Train@512 | **已冻结、未启动** | 仅增加 teacher-aligned Target 定义与视觉案例 | Short S32 后启动，再做 prompt-axis paired eval |
| Atomic Train@512 | **clean 自动接力已挂起、未开始训练** | `e5e0287` static admission accepted；独立 fresh-S0、matched Atomic prompt、@512；当前不占 GPU | 完整 C/D paired Eval@512 验收后自动 C0→S32→Eval@512 |

Crop S25 的 answer reward、format error、工具尝试题率和端到端时长分别为
`0.609375 / 3.90625% / 93.359375% / 799.56 s`，258 次尝试中 256 次成功，另有 2 次
`tool_execution_failed`。S23 出现 4 次
`incomplete_tool_call` 与 1 次 `invalid_json`，但仍有 273/278 次成功调用；S24 随即为 222/222、
0 error。结合 `include_stop_str_in_output=true`，当前只把 S23 记作稀疏 generation/parser failure，
没有证据把它升级为旧 answer-over-action action-boundary 缺陷回归。S1--S25 累计 6,400 条轨迹、
7,954 次工具尝试、7,842 次成功 observation，累计 answer reward `0.68671875`、format error
`5.515625%`；训练 reward 仅作链路诊断，不是 benchmark 结果。

最近 S18--S25 八个完整 step 平均 `13.77 min/step`，Crop S32 预计约 `18:07 JST`；这是速度估算，
不是验收承诺。`prl26-train512-s32-eval`、`prl26-cd-tgvf-prompt-s32` 与
`prl26-e-atomic-train512-s32` 三个常驻 supervisor 均存活：依次等待 Crop S32、A/B 正式评测和
C/D 完整 paired 评测。后两条等待链不占 GPU，不会与当前 Crop 训练争用资源。

A/B handoff 的 S32 前独立审计发现，旧 evaluator 在看到两个 permanent receipt 后会直接进入
bind/prepare/inference，却没有显式等待 trainer 的 vLLM/Ray teardown；已完成的 No-Tool 线路中，
receipt 比 trainer pane 真正退出早 17 秒。修复 commit
`bd31ac5e1ce299d44efbead30f553781b8f274fc` 在任何 checkpoint bind、模型 materialization 或 GPU
worker 前增加 fail-closed admission：GPU 0--7 均无 compute PID、显存不高于 32 MiB、无 Ray
进程，并连续通过至少两次探针。35 项直接 A/B→C/D→Atomic handoff 回归通过。A/B 与缓存其 pane
identity 的 C/D 已受控重挂到新 pane `%179/%180`；Crop 训练未中断，Atomic `%178` 也未重启，
三个 failed marker 均为空。

进度查看：本报告同步到 main 工作区
`docs/NEURIPS_WORKSHOP_TGVF_EXPERIMENT_PLAN_PROGRESS_REPORT_20260826.md`。在推理完成、评分完成、
审计包生成和文章结论更新等关键节点同步；运行中的计数只作为状态快照，不提前当作结果。按当前
授权，每个关键节点只提交这一个报告文件并 push 到 `origin/main`，不带入 main 的其他工作区改动。

## 0. 历史勘误：Crop `</tool_call>` action boundary（2026-08-28）

> 本节保留修复过程；当前 true-1M 与 true-512 结果只以文首 stage-golden 与 true-512 表为准。
> 本节中旧 processor-default 运行的分析不再是当前主结果。

**本节只记录历史 action-boundary 缺陷与修复过程。** 当前 Crop headline 是文首 true-1M
golden 表中的 S32 `61.0706`；S80 true-1M `59.6463` 作为 80-step 动态终点。有效
boundary-fix V2@512 为 `62.0967`，旧 boundary `61.5591` 已 supersede。历史 S8/S16/S48/S64
的分数、sub-benchmark 和工具行为仍为 provisional；Original、Pure TGVF 和 Atomic 不因 Crop
action-boundary 缺陷降级。

根因在 Crop-only 的 `full_model + deepeyes_official_visible_native_crop_v1` 路径：
full-model snapshot 重建 vLLM sampling request 时丢失了
`stop=["</tool_call>"]`、`stop_token_ids=[151645]` 和
`include_stop_str_in_output=true` 等 run-bound sampling 字段。模型因此可在合法
`</tool_call>` 后同轮继续生成 plain final，旧 evaluator 又会按 answer-over-action
接受尾文，使已请求的 Crop 没有执行。历史六点中同轮 `tool_call + final` 的题数为：

| Crop checkpoint | S8 | S16 | S32 | S48 | S64 | S80 |
|---|---:|---:|---:|---:|---:|---:|
| Same-turn `tool_call + final` | 150 | 213 | 336 | 618 | 512 | 483 |

这些题不能用离线规则反推修复后准确率，因而必须重跑。S32/S80 已重跑；其余四个
历史 step 仍只作训练动态记录。对照审计中，Pure TGVF S64
和 Atomic S16 的同类 same-turn 计数均为 **0**；两者使用另一条 paired
`training_run` evaluator 路径，从完整 owner config 传入 stop 契约，因此无需因本缺陷
重跑 TGVF S64 或 Atomic S16。

修复与复测状态：

- 修复 commit `5e37a77` 已在现有 `prl25-c-tgvf-80step` 分支落地，没有新建修复
  分支。full-model sampling identity 现在显式绑定完整 stop/action-boundary；正常生成
  在 `</tool_call>` 处硬停，若后端仍返回 mixed turn，evaluator 则以
  `tool_call_terminal_suffix` fail-closed，不再接受尾文为 final。
- Corrected S32/S80 formal rerun 使用 plan commit `e436629`，顶层 evaluation ID 为
  `PRL25-B-CROP-EXACT-COREDEV2511-S32-S80-TOOL-BOUNDARY-FIX-V2`。两臂均完成
  `2,240/2,240` 条受支持单图 trajectory、`7/7` 官方 slice，summary 状态均为
  `pass`；`paired-summary.json` 与 completion marker 已落盘。
- 修正前后的七项统一对比如下（单位 `%`）：

| Crop run | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S32 / old boundary | 63.5377 | 80.1047 | 73.0000 | 64.4444 | 54.8108 | 49.0706 | 71.3333 | 52.0000 |
| S32 / fixed boundary, native-default historical | **61.6699** | 80.6283 | 67.5000 | 61.6667 | 53.8465 | 44.9814 | 69.6667 | 53.4000 |
| S32 fixed − old | **−1.8678** | +0.5236 | −5.5000 | −2.7778 | −0.9643 | −4.0892 | −1.6667 | +1.4000 |
| S80 / old boundary | 62.2288 | 81.6754 | 74.5000 | 58.8889 | 55.3358 | 46.4684 | 67.3333 | 51.4000 |
| S80 / fixed boundary, native-default historical | **59.1785** | 78.5340 | 61.5000 | 57.7778 | 55.4948 | 44.6097 | 66.3333 | 50.0000 |
| S80 fixed − old | **−3.0503** | −3.1414 | −13.0000 | −1.1111 | +0.1589 | −1.8587 | −1.0000 | −1.4000 |

以下逐题与工具行为分析全部限定在这组 processor-default historical paired rerun，不是当前
true-1M S32/S80 comparison；当前 true-1M headline 与优化动态只以文首 golden 表和 Appendix A
为准。

- S32/S80 的 executed calls 为 `1,677 / 2,010`，successful observations 也精确为
  `1,677 / 2,010`；有效使用题为 `1,385 / 1,977`（`61.83% / 88.26%`）。
  两臂同轮 `tool_call + final` 均为 `0`，所有工具 turn 均以 `</tool_call>` 结束。
- 在当时冻结的 processor-default six-point analysis 中，S80 是事先指定的 80-step headline，
  不因修正后 S32 高 `2.4914 pp` 而 post-hoc 重选 checkpoint；该历史合同下 S80 只在 OCR mean
  上高于 S32（`+1.6483 pp`）。当前统一 true-1M 主表则按事前阶段口径使用 Crop S32。
- 对该 `S32 > S80` 现象做了严格逐题审计：两臂的 `2,240/2,240` 个 `sample_id`、题目与
  `paired_rng_stream_identity_sha256` 全部一致，因此这里没有题集或采样流混杂。在除 OCR
  连续部分分外的六个二值评分 set（`n=1,640`）中，S32→S80 的转移为：

| S32→S80 outcome | 题数 |
|---|---:|
| correct→wrong | 174 |
| wrong→correct | 123 |
| both correct | 823 |
| both wrong | 520 |

  净差为 S32 `+51` 题；按 set 分别为 VStar `+4`、HRBench `+12`、BLINK-single `+7`、
  MMMU-single `+1`、MathVista `+10`、MathVerse `+17`。加上 OCR 后，S32 在七项中的六项
  更高，只有 OCR mean 低 `1.6483 pp`，故不是某一个 subset 单独造成的偶然反转。
- 工具策略同时发生显著漂移：successful-use question 从 S32 的 `1,385/2,240=61.83%`
  增至 S80 的 `1,977/2,240=88.26%`，任意尝试率从 `64.24%` 增至 `92.46%`。有 `620`
  题由 “S32 不调用” 转为 “S80 调用”；在其中属于六个二值 set 的 `559` 题上，
  correct→wrong / wrong→correct 为 `65/42`，S32 仍净胜 `23` 题。双方都调用的六-set
  `888` 题上对应为 `96/69`，S32 又净胜 `27` 题。因此退化不仅是调用覆盖面变化；S80 在
  相同工具使用条件下的回答质量也回落。
- S80 的错误 trajectory 为 `141`（S32 `87`），其中 `context_limit` 从 `18` 增至 `41`，
  而 `invalid_crop` 基本不变（`109→112`）。S80 输出反而更短：assistant sampled-token
  均值/中位数为 `366.7/107`，S32 为 `493.8/141`；所以不能解释为 S80 plain output 更冗长，
  更符合高频 Crop 与多轮图像上下文带来额外 terminal/compound failure 的现象。
- 当前最强、但仍限于诊断性的解释是 **late-RL over-optimization**：S32→S80 学到了更激进的
  Crop 策略，却没有同步提高工具条件下的最终答题质量，并增加了 context-limit failure。
  该配对审计不单独证明“过调用”是全部因果机制；它只排除分辨率、题集、逐轮 RNG 和输出长度
作为 S32 优势的替代解释。该 processor-default historical paired observation 已闭合；当前
true-1M 结果另见文首 golden 表。与 true@512 的严格因果解释仍受 RNG/protocol 是否配对的边界
约束。
- boundary-fix V2 Crop S80@512 已闭合。fixed-boundary plan 在现有
  `neurips-notool-rl-s32` 分支 commit `eb37ad9` 冻结并完成：`2,240/2,240` 条受支持
  trajectory、`7/7` 官方 slice、summary `status=pass`、judge parse failure `0`，Macro*
  为 `62.0967`；当前合同复核将其确认为有效 `262,144` 像素上限。旧 boundary 的
  `61.5591` 已被该 V2 结果 supersede，不再进入任何当前主表。

## 1. 文章当前主线

本文不以宽泛的“互补能力与优化动态”作为唯一叙事。当前更可检验、也更有证据支撑的主线是：

> Under the unified true-1M evaluation contract, the frozen no-tool RL control leads the
> aggregate score while remaining flat relative to its matched S0 control. Atomic is the
> strongest tool policy in aggregate, whereas target-conditioned latent evidence retains
> target-specific utility and method-specific advantages only in selected visual-reasoning
> regimes. Aggregate tool superiority is therefore not established.

正文将文首 true-1M stage-golden 结果作为当前五方端到端主对照，以事前冻结的 **No-Tool S32**
作为 RL-only control、**Crop S32** 作为 native Crop 代表、**Pure TGVF** 作为机制主线、
**Original true-1M** 作为同像素预算、同 scorer 和同 sample reference 的 raw-direct 参考。
**Atomic Crop+TGVF** 目前仍列探索性扩展；其正文层级等待真正的
target-only 稳健性与无偏 target 合格率审计。已完成的广义 full-prompt stress test 同时改变了
多项 prompt/observation 合同，不能单独用于决定 Atomic 的正文层级。

## 2. 固定术语和比较口径

| 简称 | 本文固定含义 | checkpoint / run | 解释边界 |
|---|---|---|---|
| **Original raw-direct true-1M** | 原始 Qwen3-VL-8B-Instruct；无视觉工具、无自定义 system prompt；`max_pixels=1003520` | `PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1` | 进入当前统一主表和 59-slice 表；像素、scorer、sample reference 对齐，但与 PRL25 matched 行仍跨 prompt/agent protocol，只是端到端 direct reference |
| **No-Tool RL** | 同一 Qwen3-VL-8B-Instruct 做 full-model RL，但没有 Crop、TGVF、RP67、工具 schema 或工具调用 | `PRL-25-F-...-NO-TOOL-RL-...-32STEP-WS8`；S32 为事前冻结主终点 | corrected true-1M V2 S0/S8/S16/S32 已闭合；S32 Macro* `63.7520` 进入统一主表，S16 只作探索性动态；raw-direct@512 只回答 transfer，不得改称 Original |
| **Crop** | PRL25-B native RGB Crop | S32 stage golden；S80 动态终点，seed42 | true-1M S32 `61.0706` 进入当前 golden 主表；S80 true-1M `59.6463` 作优化动态；不补 seed43 |
| **TGVF** | PRL25-C Pure TGVF，Frozen RP67 | S64，seed42；seed43 仅作所选 checkpoint 复测 | 文章机制主线 |
| **Atomic** | PRL25-D Atomic Crop+TGVF，Frozen RP67 | S16，seed42；seed43 仅作所选 checkpoint 复测 | 探索性扩展，不能在审计前声称已稳定学会高质量 target |
| **matched prompt** | 80-step 训练与既有 CoreDev 评测使用的简化、训练匹配 prompt | 历史结果 | 用于现有主表 |
| **full prompt** | 详细说明 target、bbox、关系与禁止答案泄漏的 Instruct prompt；可见与运行时上限均为 6 次 | `full_visual_tool_prompt_v5_instruct_cap6` | 只在冻结 S64/S16 上评测，不重选 checkpoint；衡量 prompt shift robustness |
| **matched@512 control** | 保留各方法的 matched prompt、工具 schema、agent loop、checkpoint、seed42 与七项任务，评测上限为 `262144` | Crop S80 / TGVF S64 / Atomic S16 均已完成 | 像素上限对齐，但不同方法之间 prompt 和 agent 协议仍不同 |
| **Crop processor-default historical runs** | 旧 flat override 路径下的 fixed-boundary Crop S32/S80 | Macro* `61.6699 / 59.1785` | 两行均不是 true-1M，只保留在历史勘误区 |
| **Macro\*** | 七个百分比组件的无权平均 | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均 | 只在相同测量合同内比较 |

固定排除项：**不补 Crop seed43**。它不是本文结论的必要验证，也不用于构造三方法对称性。

### 2.1 输入分辨率与测量合同

`max_pixels` 表示保持长宽比时允许的最大图像像素面积，不是把所有图片强制缩放成正方形。
`262,144 = 512²`，而 `1,003,520` 的等面积正方形边长约为 `1,002` pixels，后者的像素面积预算
约为前者的 `3.83×`。

训练侧的 Crop、TGVF、Atomic 与 No-Tool 四条 PRL25 RL 路径均已确认有效
`image_max_pixels=1,003,520`。旧 Crop/No-Tool evaluator 的 flat override 曾回退到
`size.longest_edge=16,777,216`，因此它们的旧分数已降级。当前 Crop S32/S80 重跑则使用经
processor/grid receipt 证实的顶层 `mm_processor_kwargs.size.longest_edge=1,003,520`，已与
TGVF/Atomic 对齐为 true-1M。Original 未经过这轮 RL；其 true-1M raw-direct 正式评分已闭合，
历史 direct eval 的 `262,144` 路径仍只作为 true-512 参考。

| 比较臂 | Prompt / agent contract | 最大图像像素面积 | 当前用途 |
|---|---|---:|---|
| Original raw-direct true-1M | 无 system prompt、无工具、direct generation | 1,003,520 | 当前统一主表与 59-slice 的 direct reference；Macro* `61.3147` |
| Original raw-direct@512 | 无 system prompt、无工具、direct generation | 262,144 | 历史端到端参考；也是 S32 raw-direct@512 的严格 base comparator |
| Crop S32/S80 true-1M | native Crop prompt 与 fixed-boundary agent loop | 1,003,520 | S32 为 stage golden；S80 为优化动态终点 |
| Crop S80 boundary-fix V2@512 | 同一类 Crop prompt 与 fixed-boundary agent loop | 262,144 | 有效 true-512；Macro* `62.0967` |
| TGVF S64 / Atomic S16 matched true-1M | 各自训练匹配的工具 prompt 与 agent loop | 1,003,520 | stage-golden 五方表的两个 latent-evidence 方法 |
| TGVF S64 / Atomic S16 matched@512 | 与上一行逐方法相同，只降低 evaluator 像素上限 | 262,144 | 两臂均已闭合 |
| No-Tool S0/S8/S16/S32 corrected true-1M V2 | 训练匹配的 no-tool prompt 与 direct-only loop | 1,003,520 | 已闭合；S32 frozen headline `63.7520`，S32−S0 `−0.0098 pp` |
| No-Tool S0/S8/S16/S32 processor-default historical | 同上，但旧 flat evaluator override 失效 | processor default 16,777,216 | `64.4712/66.1132/66.9028/66.6853` 只保留 historical，不进入当前主表 |
| No-Tool S32 raw-direct@512 | 与历史 Original 相同的 raw-direct 配置，只替换 model path | 262,144 | 已完成；Macro* `54.3543`，比 Original 低 `1.0013 pp` |

因此，文首 Original/No-Tool/Crop/TGVF/Atomic 表在评测像素上限、scorer 和 sample reference
上对齐，四条 RL 方法的训练像素上限也对齐；但方法间仍跨 prompt、agent loop 与 checkpoint step。
Original 与任何 PRL25 工具/matched 行的绝对差值还会混入这些合同差异，不能写成 prompt-only
gain，更不能直接归因给 RL 或工具。

## 3. 历史 mixed-contract 参考表（已被文首 golden 主表 supersede）

下表保留先前文章决策的历史上下文，不再作当前主表。Original 的精确七项均值为
`55.3556`，按其既有 raw-direct@512 测量合同报告为 `55.36`。No-Tool S32 raw-direct 行与
Original 使用同合同；TGVF 与 Atomic 使用有效 matched@1M，而 Crop 和表内 No-Tool matched
行都实际回退到 processor default。该表用于完整展示，不构成所有行之间的严格 paired ranking。

| Method | Macro* | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.3556 | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| No-Tool RL S32 raw-direct@512 | 54.3543 | −1.0013 | 51.3089 | 59.0000 | 60.0000 | 47.9377 | 39.0335 | 74.0000 | 49.2000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | +3.8229 | 78.5340 | 61.5000 | 57.7778 | **55.4948** | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 | 59.8086 | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | +7.7271 | 71.7277 | **73.5000** | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |
| No-Tool RL S32 | **66.6853** | **+11.3297†** | **84.2932** | 69.0000 | **70.5556** | 50.8528 | **55.7621** | **75.3333** | **61.0000** |

`†` No-Tool S32 matched 与 Original 的差值同时跨越 prompt/protocol 和最大图像像素面积
（processor default `16,777,216` vs `262,144`），只是历史端到端参考。旧 No-Tool
processor-default 同协议 S32−S0 为 `+2.2141 pp`，但已被 corrected true-1M V2 的
`−0.0098 pp` supersede。在 Original 同测量合同下，S32 raw-direct@512 低 `1.0013 pp`；
因此旧 matched 增益既不能当作当前 RL effect，也不能外推为 raw-direct transfer。

只可作为历史上下文保留的事实边界：

- 三个工具方法的历史 Macro* 都高于 raw-direct@512 Original；该描述同时跨越 prompt/agent
  contract 和像素上限，只能作为端到端观察。TGVF/Atomic 的像素上限对齐，但 Crop 使用
  processor default，不能再声称三个工具线路内部像素对齐。Crop S80 是按用户指定报告的
  80-step 历史终点，不再使用 post-hoc 最优 S32。
- 旧 No-Tool RL S32 是该 mixed-contract 表最高 Macro*，比 Crop S80 historical、TGVF S64、
  Atomic S16 分别高 `7.5068 / 6.8767 / 3.6026 pp`；这些差值跨像素合同，不再支撑当前排序。
- 旧 No-Tool S0/S32 为 `64.4712/66.6853`、差 `+2.2141 pp`，只说明旧 processor-default
  测量轨迹；当前 RL 动态必须使用 corrected true-1M V2 的 `−0.0098 pp`。
- No-Tool S32 raw-direct@512 相对 Original 的分项差值为 VStar `+0.5236`、HR `0.0000`、
  BLINK-180 `−5.5556`、OCR mean `−0.2471`、MMMU-269 `+0.0035`、MathVista `−0.3333`、
  MathVerse `−1.4000 pp`。净结果为 `−1.0013 pp`，不支持 raw-direct transfer gain。
- TGVF 不是全榜最优方法，因此文章不能写成“通用性能支配”。它相对 Original 的主要整体
  增益在 VStar、HRBench 和 MMMU，并在一组更细的关系/数学任务中形成集中优势。
- Atomic 的 Macro* 比 Crop S80 native-default 历史运行高 `3.9042 pp`，但两者的有效像素合同
  不同；该差值只能描述既有端点，不能支持公平的方法排序或“Crop+TGVF 存在因果 synergy”。
- Original 在部分视觉强度与 OCR 切片上仍有优势；旧 No-Tool processor-default S32 的
  MathVista 比 Original@512 高 `1.0000 pp`，而 raw-direct@512 S32 低 `0.3333 pp`。两者均不
  替代当前 true-1M 五方表。

## 4. 历史 mixed-contract sub-benchmark 面板（superseded）

> 本节保留旧 Original@512 / Crop processor-default / TGVF/Atomic@1M / No-Tool
> processor-default 的分析轨迹，不再是当前跨方法证据。当前统一 true-1M 的 59-slice
> 完整结果见 Appendix A。

### 4.1 预冻结选择规则

主图只使用官方 scorer 已提供、样本定义稳定且可对五种方法对齐的 sub-benchmark。候选项必须
满足以下至少一项：

1. 对应方法相对 Original 有正增益，且能映射到明确的视觉能力；
2. 对应方法在 Crop/TGVF/Atomic 中形成方法特异性领先；
3. 是会限制论文主张的重要反例。

当前面板是根据 matched-prompt 结果形成的**探索性解释面板**，不是事前注册的 confirmatory
endpoint。本文在 target-only 结果揭晓前冻结其 v2 版本；此后不因 target-only 结果重选切片。
正文展示精简优势面板，补充材料同时报告同一 family 的完整切片，避免只报有利项。

### 4.2 正文候选：扩展优势面板 v2

表中粗体表示五种方法的行最优；最后一列仍是 `max(TGVF, Atomic) − Original`，专门保留
target-conditioned 方法的能力增益。No-Tool 列作为控制进入完整面板，不参与该列的定义。
Crop 列来自 fixed-boundary native-default 历史运行，不是 matched@1M；它与其他列任务 ID
对齐，但视觉像素合同不对齐，因此只作探索性上下文，不承担公平像素预算下的方法排序。

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 | Best TGVF/Atomic Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStar / direct attributes | 115 | 48.70 | 82.61 | 70.43 | 69.57 | **86.09** | **+21.74** |
| VStar / relative position | 76 | 53.95 | 72.37 | 80.26 | 75.00 | **81.58** | **+26.32** |
| HRBench / cross-image aggregate | 100 | 59.00 | 40.00 | 64.00 | **68.00** | 66.00 | **+9.00** |
| HRBench / single-image aggregate | 100 | 59.00 | **80.00** | 69.00 | 79.00 | 72.00 | **+20.00** |
| BLINK / Counting | 30 | 66.67 | 63.33 | 73.33 | 73.33 | **80.00** | **+6.67** |
| BLINK / Object Localization | 30 | 56.67 | 60.00 | 70.00 | **73.33** | 70.00 | **+16.67** |
| BLINK / Relative Depth | 30 | 83.33 | 83.33 | 86.67 | 80.00 | **90.00** | **+3.33** |
| BLINK / Relative Reflectance | 30 | 50.00 | 40.00 | 46.67 | **70.00** | 60.00 | **+20.00** |
| MathVista / numeric commonsense | 36 | 47.22 | 44.44 | **58.33** | 50.00 | **58.33** | **+11.11** |
| MathVista / arithmetic reasoning | 104 | 65.38 | 56.73 | **72.12** | 63.46 | 71.15 | **+6.73** |
| MathVista / visual question answering | 42 | 54.76 | 57.14 | **64.29** | 50.00 | 61.90 | **+9.52** |
| MathVista / math word problem | 63 | 77.78 | 60.32 | **84.13** | 79.37 | 82.54 | **+6.35** |
| OCR EN / text recognition | category | 60.49 | 73.05 | 55.05 | **73.38** | 65.54 | **+12.89** |
| OCR EN / visual text understanding | category | 75.00 | **83.27** | 80.00 | 83.21 | 78.33 | **+8.21** |
| OCR CN / visual text understanding | category | 35.00 | 45.00 | **65.00** | 55.00 | 60.00 | **+30.00** |
| MathVerse / Text Lite | 100 | 53.00 | 52.00 | 58.00 | 59.00 | **65.00** | **+6.00** |
| MathVerse / Vision Only | 100 | 28.00 | 48.00 | 42.00 | 51.00 | **55.00** | **+23.00** |

建议主图分成三块，并把 No-Tool 作为贯穿每块的控制列：

- **target-conditioned shared gains**：VStar、HR、Counting、Object Localization；
- **TGVF-concentrated gains**：relative position/depth、MathVista 四项、OCR CN visual text
  understanding；
- **Atomic-concentrated gains**：HR cross、Relative Reflectance、OCR EN text recognition、
  MathVerse Vision Only。

正文同时保留一个小型 boundary companion panel：BLINK IQ Test、Spatial Relation、MathVista
geometry reasoning、MathVerse Vision Intensive。这样主图可以彰显优势，但不会暗示全面支配。

### 4.3 MathVista MINI 低于 Original 的归因

这不是测试 subset 不同造成的。五种方法都在同一份完整 `MathVista_MINI` 上评分，逐题核对后
均为相同的 300 个 `index`；Original 答对 223 题，而 Crop native-default historical、TGVF、Atomic、
No-Tool S32 分别答对 199、217、209、226 题。按官方判分函数逐题配对得到：

| Method | Correct / 300 | Δ correct vs Original | Gained: method correct, Original wrong | Lost: Original correct, method wrong |
|---|---:|---:|---:|---:|
| Original | 223 | — | — | — |
| Crop S80 / fixed boundary, native-default historical | 199 | -24 | 12 | 36 |
| TGVF S64 | 217 | -6 | 26 | 32 |
| Atomic S16 | 209 | -14 | 19 | 33 |
| No-Tool S32 | 226 | +3 | 23 | 20 |

对 TGVF，5 个互斥官方 task 的净变化为：textbook question answering `-7` 题、figure
question answering `-4`、geometry problem solving `-3`，被 math word problem `+4` 和
visual question answering `+4` 部分抵消，合计 `-6`。按互斥 question type 看，TGVF 在
free-form 上 `+1` 题，在 multi-choice 上 `-7` 题。因此当前证据更准确的表述是：TGVF 在
视觉数学中的 word problem / visual QA 子域有集中收益，但没有保住 Original 在 textbook、
figure QA、geometry solving 和多选题上的优势；不能归结为一个更难的工具方法专属 subset。

Appendix A 完整列出各 benchmark 的 aligned sub-benchmark。MathVista 的 12 行是官方 scorer
输出的 5 个互斥 task 加 7 个可重叠 skill；重叠 skill 的增减不能相加来解释总体 6 题差值。
当前逐题数据只能定位回退发生在哪里，尚不能在没有单变量消融时断言原因是工具干扰、prompt
shift 或答案格式。

### 4.4 历史 mixed-contract 工具调用面板（superseded）

#### 4.4.1 统计口径与整体结果

调用统计来自三个 matched-prompt 最佳 checkpoint 的 `rank-0..3.jsonl` trajectory。三者在完全
相同的 `2,240` 个受支持单图 ID 上各有一条记录，其中 BLINK 使用共同单图 `n=180`，
MMMU-Pro 使用共同单图 `n=269`。本文区分以下口径：

这里的 Crop 调用行为来自 S80 fixed-boundary native-default 历史运行。调用计数本身有效，但
不能解释为 matched@1M 行为，亦不能与 nominal @512 历史运行构造像素效应。

- **attempted question**：`tool_calls + tool_errors > 0`，即模型至少生成过一次工具尝试；
- **successful-use question**：`tool_calls > 0`，即至少一次调用被执行并返回 observation；
- **executed calls**：trajectory 中 `tool_calls` 的总数；
- **invalid attempts**：trajectory 中 `tool_errors` 的总数，不计入有效工具调用；
- **repeat-use question**：有效调用次数至少为 2。

Original 没有工具接口；下表的 Original 行是在同一 `2,240` 个 ID 上按协议定义的零调用参照，
不是从 Original trajectory 反推的统计。Atomic 的 `tgvf_crop_tool` 是一次调用内联合产生
Crop 与 TGVF observation 的原子工具，不能拆成两次工具调用。

| Method | Tool function | Attempted questions | Successful-use questions | Executed calls | Invalid attempts | Execution yield | Calls/question | Calls/used question | Repeat-use questions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | none | 0 (0.00%) | 0 (0.00%) | 0 | 0 | — | 0.000 | — | 0 (0.00%) |
| No-Tool RL S32 | none | 0 (0.00%) | 0 (0.00%) | 0 | 0 | — | 0.000 | — | 0 (0.00%) |
| Crop S80 / fixed boundary, native-default | `image_zoom_in_tool` | 2,071 (92.46%) | 1,977 (88.26%) | 2,010 | 153 | 92.93% | 0.897 | 1.017 | 21 (0.94%) |
| TGVF S64 | `tgvf_focus_tool` | **2,012 (89.82%)** | **2,010 (89.73%)** | 2,011 | 3 | **99.85%** | 0.898 | 1.000 | 1 (0.04%) |
| Atomic S16 | `tgvf_crop_tool` | 1,866 (83.30%) | 1,863 (83.17%) | **2,300** | 36 | 98.46% | **1.027** | **1.235** | **221 (9.87%)** |

`Execution yield = executed calls / (executed calls + invalid attempts)`。三个方法的 `2,010`、
`2,011`、`2,300` 次 executed calls 均产生了 observation；小于 100% 的 execution yield 来自
另外记录的无效尝试，而不是已执行工具返回失败。

#### 4.4.2 不同 benchmark set 的调用覆盖率

每格为 `successful-use questions / n (rate)`。该表回答“模型在多少题上实际使用了工具”，不把
无效尝试算作成功使用。

| Set | n | Crop S80 native-default | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|
| VStarBench | 191 | 185/191 (96.86%) | 186/191 (97.38%) | **188/191 (98.43%)** |
| HRBench4K | 200 | 196/200 (98.00%) | **199/200 (99.50%)** | 198/200 (99.00%) |
| BLINK single-image | 180 | 169/180 (93.89%) | **178/180 (98.89%)** | **178/180 (98.89%)** |
| OCRBench v2 | 600 | 530/600 (88.33%) | 556/600 (92.67%) | **560/600 (93.33%)** |
| MMMU-Pro single-image | 269 | **250/269 (92.94%)** | 232/269 (86.25%) | 243/269 (90.33%) |
| MathVista MINI | 300 | 264/300 (88.00%) | **268/300 (89.33%)** | 222/300 (74.00%) |
| MathVerse MINI | 500 | 383/500 (76.60%) | **391/500 (78.20%)** | 274/500 (54.80%) |

#### 4.4.3 不同 benchmark set 的调用次数与频率

每格为 `executed calls / calls per eligible question`。分母始终是该 set 的全部受支持题目，而
不是只包含工具调用题，因此不同方法可直接比较调用强度。

| Set | Crop S80 native-default | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|
| VStarBench | 185 / 0.969 | 186 / 0.974 | **191 / 1.000** |
| HRBench4K | 198 / 0.990 | 199 / 0.995 | **218 / 1.090** |
| BLINK single-image | 170 / 0.944 | 178 / 0.989 | **307 / 1.706** |
| OCRBench v2 | 553 / 0.922 | 556 / 0.927 | **728 / 1.213** |
| MMMU-Pro single-image | 254 / 0.944 | 232 / 0.862 | **287 / 1.067** |
| MathVista MINI | 267 / 0.890 | 268 / 0.893 | **281 / 0.937** |
| MathVerse MINI | 383 / 0.766 | **392 / 0.784** | 288 / 0.576 |

#### 4.4.4 每题有效调用次数分布与无效尝试

| Method | 0 calls | 1 call | 2 calls | 3 calls | 4 calls | 5+ calls |
|---|---:|---:|---:|---:|---:|---:|
| Original | 2,240 (100.00%) | 0 | 0 | 0 | 0 | 0 |
| No-Tool RL S32 | 2,240 (100.00%) | 0 | 0 | 0 | 0 | 0 |
| Crop S80 / fixed boundary, native-default | 263 (11.74%) | 1,956 (87.32%) | 14 (0.63%) | 5 (0.22%) | 0 | 2 (0.09%) |
| TGVF S64 | 230 (10.27%) | 2,009 (89.69%) | 1 (0.04%) | 0 | 0 | 0 |
| Atomic S16 | 377 (16.83%) | 1,642 (73.30%) | 120 (5.36%) | 44 (1.96%) | 23 (1.03%) | 34 (1.52%) |

无效尝试的错误构成也不同：Crop S80 native-default 历史运行的 153 次包括 `invalid_crop=112`、
`context_limit=41`；
TGVF S64 只有 `tool_parse.invalid_tool_name=3`；Atomic S16 的 36 次包括
`tool_call_limit_exceeded=21`、`tool_parse.invalid_bbox=8`、
`tool_parse.incomplete_tool_call=5`、`tool_parse.invalid_tool_name=2`。它们必须与 executed
calls 分开报告，因为错误尝试没有产生工具 observation。

#### 4.4.5 可写结论与边界

- Crop S80 native-default 历史运行表现为**高覆盖、仍以单次为主**的调用：整体成功使用率
  `88.26%`，调用强度 `0.897 calls/question`；`1,956/1,977` 个工具使用题
  只有一次有效调用。历史边界缺失曾系统性低估 Crop 实际执行率。
- TGVF S64 表现为**高覆盖、近乎固定一次**调用：整体成功使用率 `89.73%`，只有 1 题发生
  两次有效调用；VStar、HRBench 和 BLINK 的覆盖率均超过 `97%`。
- Atomic S16 表现为**按 set 改变调用强度**：成功使用率 `83.17%`，但 `9.87%` 的全部题目
  出现重复调用，使整体达到 `1.027 calls/question`；重复检索主要集中在 BLINK 和 OCR。
- No-Tool RL S32 的七个 set 合计 `2,240` 条 trajectory 中，有效调用、工具错误、observation
  和结构化工具文本泄漏均为 `0`；全部 set 的 calls/question 均为 `0.000`。作为学习动态审计，
  S0/S8 曾分别出现 `16/5` 条结构化工具样式输出并以 `invalid_format` fail-closed，S16/S32 均为
  `0`，没有任何一次被解析或执行为工具调用。
- 调用率和调用次数是行为描述，不是因果 utility。不能由“调用更多”直接推出“工具更有效”；
  后续应按相同 ID 报告 `0/1/2+` 调用组的正确率与置信区间，并明确其受 policy 自选择混杂。

target-only 版本完成后，必须用同一定义追加一张对照表；在此之前不得把 matched-prompt 与
target-only 的调用统计合并。

## 5. 三项必须补齐的验证

### 5.1 RP67 image + D matched utility

目的：验证 RP67 提供的是 target-specific answer utility，而不只是额外 token 或任意 latent
扰动。

#### 既有 RP67 验证库存与口径纠正

必须先纠正一个容易混淆的历史口径：`POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md`
中 first-200 十臂表的 `image-only 72.5%`、`image + correct-D 83.5%`、zero/wrong
`76.0% / 77.0%` 来自 **RP66**，不能当作 RP67 结果。RP67 Step-2000 自身此前已经有以下
互补验证；它们不能简单相加，也不能把重复使用同一 867-row population 的结果当作独立复现。

| 层级 | RP67 既有验证 | 主要结果 | 能支持什么 | 关键边界 |
|---|---|---|---|---|
| 训练与表示健康 | 2,000-step 训练记录；main + 3 个 D-DeepStack branch | image-axis train top-1 `25% → 100%`，gap `−0.265 → 9.938`；内部评测全部 finite、collapse rate 0 | 优化闭合，D 没有数值崩溃或 token collapse | train 指标不是 held-out answer utility；S2000 validation 只有 8 rows |
| teacher-forced target retrieval | S2000 INT-DIAG，200 rows / 46 image groups | query top-1/top-2 `90% / 98%`，MRR `94.63%`；184 个完整控制样本中 correct-D 胜过全部 control `94.57%` | D 在固定 reader/readout 下具有很强的 image/target 可辨识性 | oracle target、teacher-forced NLL；不等价于自由生成或自主 target 选择 |
| image-conditioned answer utility | S2000 full-867 两臂 | image-only `644/867 = 74.28%`；image + correct-D `737/867 = 85.01%`，描述性差值 `+10.73 pp` | correct-D 在原图上下文上具有互补答题价值 | diagnostic semantic overlay；旧 artifact 未给 paired CI，且未含 zero/wrong-D |
| standalone / replacement 六臂 | S2000 first-200 | D-only correct/zero/wrong `36% / 34% / 37%`；direct replacement `35% / 21% / 32.5%` | 相对 zero 的 content signal 在 direct replacement 中存在 | D-only specificity 不成立；说明结论强烈依赖原图上下文，不能宣称 D 可替代图像 |
| earlier checkpoint wrong-image stress | S500 full-867 两臂 | image + correct-D `85.70%`；same-target wrong-image-D `2.19%` | D 明确绑定图像内容，错误图像 D 会强烈干扰 reader | 是 S500 而非所选 S2000；control 过强且主动破坏，只作补充诊断 |
| native causal/free-continuation | S2000 INT-DIAG | 9 对 continuation accuracy `50%`、expected-direction flip `55.56%`；36 对 target-presence actual-direction accuracy `0%`、continuation accuracy `2.78%` | 暴露 teacher-forced readout 到 native generation 的断层 | **负/未闭合证据**，不能作为 RP67 有效性的正面论据 |
| downstream external utility | Frozen-RP67 TGVF / Atomic 的 CoreDev-2511 | 已在主表及 sub-benchmark 表与 Original、Crop 同列 | 表示可嵌入端到端 policy 并在若干视觉推理 regime 形成优势 | 与 policy、prompt、训练 recipe 纠缠，不是 RP67 单变量消融 |

文章证据层级据此固定为：本次 full-867 三臂 paired utility 是 RP67 机制主证据；旧 full-867
correct-D vs image-only 与 INT-DIAG 是支持证据；first-200 六臂和 native continuation 作为
限制证据；CoreDev 结果负责外部效度。这样既能使用已有验证，也不会把 RP66 数值错归给 RP67，
或把 teacher-forced readout 误写成 autonomous tool-use 证明。

#### RP67 既有指标详细附录

训练与完成身份：RP67 Step-2000 run ID 为
`RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01`，run identity SHA256
为 `0b53d04cf8e4c8b665e76279da1df8d1e6ebabee63318c644a3bff5bad099b44`，最终 Adapter
manifest SHA256 为 `2ea098967ba36671d6975a17e3830778d441149c27f5f80e43e78daf818933b1`。
训练日志共有 200 个每 10-step 记录，Step 10 与 Step 2000 的同批次训练指标如下；它们用于
确认优化方向与数值健康，不当作 held-out 泛化结果。

| train metric | Step 10 | Step 2000 |
|---|---:|---:|
| image-axis top-1 | 25.00% | 100.00% |
| image-axis score gap | −0.2649 | 9.9379 |
| image-axis loss | 0.8818 | 0.00029 |
| Matrix-CE | 1.3904 | 0.4155 |
| evidence `L_gen` | 5.0440 | 1.3605 |
| total loss | 7.3622 | 1.8532 |

Step-2000 周期性 validation snapshot 只有 8 rows / 2 image groups：Matrix-CE `0.2018`、
`L_gen 1.2983`、Norm `0.7245`、total `1.5726`。小规模 validation 只能作执行 gate；主要
held-out 表示证据来自下述固定 200-row INT-DIAG。

INT-DIAG 的 teacher-forced NLL 越低越好。correct-D mean NLL 为 `1.2694`；相对每个控制的
结果为：

| Control | available n | control mean NLL | correct-D win rate | correct-D mean advantage |
|---|---:|---:|---:|---:|
| target only | 200 | 4.1035 | 99.50% | 2.8341 |
| random D | 200 | 3.9661 | 100.00% | 2.6967 |
| wrong D, same image | 200 | 6.5265 | 95.00% | 5.2571 |
| wrong D, different image | 184 | 9.9689 | 100.00% | 8.6801 |

同一 INT-DIAG 的 46 个 image group / 200 rows query matrix 达到 top-1 `90.00%`、top-2
`98.00%`、MRR `94.625%`、mean diagonal gap `3.0327`。184 条具备全部控制的样本中，
correct-D 同时胜过所有控制为 `174/184 = 94.57%`。main D 与 layer 8/16/24 branches 的
joint finite rate 均为 100%，near-identical-token collapse rate 均为 0；D/source mean-token-
norm ratio 分别为 `1.327 / 2.872 / 2.734 / 2.543`。这些 norm 只用来排除崩溃，不把绝对
scale 大小解释成语义质量。

不同 checkpoint / injection context 的历史 semantic overlay 不能混成一张同质 ablation，
但并列表明了结论边界：

| checkpoint / context | correct | zero / image-only | matched wrong | 读法 |
|---|---:|---:|---:|---|
| S500 first-200, D-only | 48.50% | 35.00% | 47.00% | content `+13.5 pp`，specificity 仅 `+1.5 pp` |
| S500 first-200, direct replacement | 37.00% | 22.00% | 35.00% | content `+15 pp`，specificity 仅 `+2 pp` |
| S2000 first-200, D-only | 36.00% | 34.00% | 37.00% | content `+2 pp`，specificity `−1 pp` |
| S2000 first-200, direct replacement | 35.00% | 21.00% | 32.50% | content `+14 pp`，specificity `+2.5 pp` |
| S500 full-867, image + D | 85.70% | — | 2.19% wrong-image D | 强 image binding stress；不是 S2000 control |
| S2000 full-867, image + D / image-only | 85.01% | 74.28% | — | 描述性互补效应 `+10.73 pp` |
| S2000 full-867, 本次 paired 三臂 | 84.89% | 77.74% zero-D | 63.32% same-image wrong-target D | 当前机制主结果；paired CI 见下表 |

S500 → S2000 并没有在所有 free-generation context 上单调改善：尤其 D-only specificity 从
微弱正差变成 `−1 pp`。另一方面，S2000 的 teacher-forced retrieval/readout 极强，且原图 + D
三臂结果支持 content utility 与 specificity。这种不一致本身是重要发现：RP67 学到的 D 更适合
作为原图上的条件残差，不应描述成独立视觉替代物；teacher-forced 可读性也不能自动推出 native
free-generation 因果性。

native 诊断保留原始负结果：9 对 counterfactual 的 continuation accuracy `50.00%`、
expected-direction flip `55.56%`、healthy termination `77.78%`；36 对 target-presence 的
actual-direction accuracy `0%`、continuation accuracy `2.78%`、healthy termination `93.06%`。
这些小样本探针不推翻 image+D 主结果，但明确阻止“RP67 已在 native continuation 中稳定控制
答案方向”的表述。

权威 artifact 索引：

- 训练与 INT-DIAG：
  `artifacts/representation/RP-67-qwen3-instruct-balanced-t1-image-axis-grounded-2000-gpu01/`
  下的 `metrics.jsonl` 与 `int-diag-step2000.json`；
- S2000 全验证闭合 receipt：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_all_validations_complete_v2.json`；
- S2000 first-200 六臂：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_first200_6arm_semantic_20260801/summary.json`；
- S2000 full-867 image-only / correct-D：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_full867_acc_main_semantic_v2_20260801/summary.json`；
- S500 first-200 六臂与 full-867 wrong-image stress：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step0500_first200_6arm_semantic_20260731/summary.json`
  和 `rp67_step0500_full867_2arm_semantic_20260731/summary.json`。

上述 free-generation accuracy artifact 的固定 `claim_scope` 均为
`diagnostic_semantic_overlay_not_formal_pilot`；正文或附录不得删去这一标签。

固定三臂：

| Arm | 定义 | 作用 |
|---|---|---|
| `image_correct_D` | 原图 + 正确 target 的 RP67 D | treatment |
| `image_target_zero_D` | 原图 + 同位置零 D | 测 D content utility 的 control |
| `image_matched_wrong_D` | 原图 + 同图、不同 target、答案安全 donor 的 D | 测 target specificity 的 control |

主要量：

- `Δcontent = Acc(image_correct_D) - Acc(image_target_zero_D)`；
- `Δspecificity = Acc(image_correct_D) - Acc(image_matched_wrong_D)`；
- 辅助报告 `Acc(wrong) - Acc(zero)`、paired wins/losses、95% CI。

当前状态：

- [x] 完整 867 样本预检；203 个 image group 均可构造 answer-safe wrong mapping；
- [x] 8-way 可恢复正式生成完成；867 × 3 = 2,601 条记录，无缺失或重复；
- [x] 对 deterministic unresolved 完成固定 Qwen2.5-72B blind semantic overlay；
- [x] 汇总三臂 accuracy、paired delta、bootstrap/Wilson CI。

结果：

| Arm | Correct / 867 | Accuracy | 95% Wilson CI |
|---|---:|---:|---:|
| `image_correct_D` | 736 | **84.89%** | [82.35, 87.12] |
| `image_target_zero_D` | 674 | 77.74% | [74.85, 80.38] |
| `image_matched_wrong_D` | 549 | 63.32% | [60.06, 66.46] |

| Paired contrast | Δ accuracy | 95% paired bootstrap CI | wins / losses | exact McNemar p |
|---|---:|---:|---:|---:|
| correct − zero (`Δcontent`) | **+7.15 pp** | [4.73, 9.57] | 92 / 30 | `1.65e-8` |
| correct − matched-wrong (`Δspecificity`) | **+21.57 pp** | [18.22, 24.91] | 222 / 35 | `1.95e-34` |
| matched-wrong − zero | −14.42 pp | [−17.99, −10.84] | 71 / 196 | `1.04e-14` |

解释：correct D 同时显著优于零 D 和同图错误 target 的 D，支持 RP67 observation 携带
**内容相关且 target-specific** 的 answer utility。wrong D 仍有 63.32% accuracy，说明原图和
语言先验仍能解出大量题目；但 wrong 显著低于 zero，也说明不匹配 D 会主动干扰 reader，不能
把 wrong arm 当作“无信息”基线。

评分边界：原始 deterministic scorer 对 verbose answer 保守地返回 unresolved，因此最终结果
使用固定哈希的 Qwen2.5-72B blind semantic overlay。共 2,055 个唯一 judge 请求，零重试、零
长度截断。项目 schema 将它标为 `diagnostic_semantic_overlay_not_formal_pilot`；正文必须称为
diagnostic semantic accuracy，并在附录同时给 deterministic lower bound 与完整 judge 合同。
三臂 deterministic strict lower bound 分别只有 `1.04% / 52.71% / 0.81%`；其巨大差异主要
反映 correct/wrong 输出远比 zero verbose、从而更常 unresolved，不能把这组三臂 lower bound
当成可比较的 accuracy 估计。

生成根目录：
`artifacts/representation_experiments/answer_utility/evaluation/rp67_step2000_full867_three_arm_20260826_v1/`

语义 overlay：
`artifacts/representation_experiments/answer_utility/evaluation/rp67_step2000_full867_three_arm_semantic_20260826_v1/`

### 5.2 冻结最佳 checkpoint 的 prompt 补充评测

目的：检查现有优势是否依赖过度简化的 matched prompt。

#### 5.2.1 已完成的广义 full-prompt stress test

2026-08-26 的逐字节协议审计发现，matched prompt 本身已经要求 `<think>...</think>`、按需
tool call 和 plain-text final；已完成的 `full_visual_tool_prompt_v5_instruct_cap6` 不只细化
target，还改变了逐轮 reasoning/final-only 指令、native schema 注入方式，并在成功 observation
中回显 target。因此下述结果保留为有效的**广义 prompt bundle stress test**，但不得写成“只增加
target 定义”的单变量结果，也不得把下降归因于 target 细化。

该 stress test 的固定设计：

- Pure TGVF：S64；
- Atomic：S16；
- checkpoint 在看 full-prompt 结果前冻结，不允许重新选 step；
- 仍使用 CoreDev-2511、相同官方 scorer、temperature 1、seed42；
- prompt 详细规定 target 必须包含“看什么 + 提取什么证据/关系”，Atomic 还规定 bbox、关系实体
  覆盖、禁止猜测答案；
- 可见工具上限与运行时均为 6；成功 observation 回显 target；
- 这是同一 benchmark 上的 prompt-shift robustness，不是 held-out dataset confirmation。

当前状态：

- [x] 新增独立协议 `full_visual_tool_prompt_v5_instruct_cap6`；
- [x] TGVF/Atomic prompt bundle、system prompt、tool schema、target echo 和 RNG 协议均绑定 SHA256；
- [x] S64/S16 两份计划通过真实训练配置和冻结 RP67 绑定预检；
- [x] 相关协议与计划测试 18/18 通过；
- [x] 修复并回归测试已完成训练中“历史最优 checkpoint + 历史 RP67 manifest”的等待边界；
- [x] TGVF S64 与 Atomic S16 full-prompt 生成均完成并通过完整性核验：每臂 2,240 rows、
  2,240 个唯一 sample / trajectory / result identity、零重复；
- [x] TGVF S64 七套官方 scorer 全部完成：7/7 slices，summary contract 通过，judge parse
  failure 为 0；
- [x] Atomic S16 七套官方 scorer 全部完成：7/7 slices，summary contract 通过，judge parse
  failure 为 0；
- [x] 两臂完整结果与 matched prompt、Original、Crop native-default 历史参考同表报告。

完成快照（2026-08-26 02:40 JST）：TGVF 与 Atomic 均为 `2,240 / 2,240`，两份
completion receipt 均已生成。TGVF stop 分布为 2,005 final-answer、156 direct-answer、78
max-tokens、1 call-cap，累计 41 次工具错误；Atomic 为 2,003 final-answer、225 direct-answer、
12 max-tokens，累计 10 次工具错误。错误计数保留为质量诊断，但不影响各自 2,240-row 唯一覆盖。

两条线路的 full-prompt 文章口径结果如下。Original 使用已闭合的 raw-direct true-1M reference；
它与 TGVF/Atomic 行对齐像素、scorer 和 sample reference，但仍不是严格 paired prompt control。
Crop native-default historical 行不共享像素合同，其 `Δ vs Original` 仅为描述性差值。
`Δ vs matched` 才是同 checkpoint、同任务和同 seed 下的 prompt-shift 稳健性量。

| Reference / run | Macro* | Δ vs matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct true-1M | 61.3147 | — | — | 72.7749 | 66.5000 | 63.8889 | **59.7877** | 37.9182 | **75.3333** | 53.0000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | n/a | −2.1362 | **78.5340** | 61.5000 | 57.7778 | 55.4948 | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 / matched prompt | 59.8086 | reference | −1.5061 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / full prompt | 58.5138 | −1.2949 | −2.8009 | 71.7277 | 64.5000 | **66.1111** | 39.4659 | 45.7249 | 68.6667 | 53.4000 |
| Atomic S16 / matched prompt | **63.0827** | reference | **+1.7680** | 71.7277 | **73.5000** | **66.1111** | 54.2720 | **51.3011** | 69.6667 | 55.0000 |
| Atomic S16 / full prompt | 60.4684 | **−2.6142** | −0.8463 | 70.6806 | 67.0000 | 63.3333 | 50.4349 | 46.0967 | 70.3333 | **55.4000** |

full prompt 相对 matched prompt 的七项 delta 依次为 VStar `−2.6178`、HR `−2.0000`、
BLINK-180 `+0.5556`、OCR mean `−5.0787`、MMMU-269 `+0.7435`、MathVista
`−3.6667`、MathVerse `+3.0000 pp`。因此当前结论不是“prompt 完全无影响”，而是：TGVF
full prompt 的 Macro* 比 true-1M Original 低 `2.8009 pp`，七项中 BLINK、MMMU 和 MathVerse
三项高于 Original；prompt shift 相对自身 matched 又造成 `1.2949 pp` 总体回落，损失主要集中在
OCR、MathVista 和 VStar。sub-benchmark 上，full prompt 仍保留 Relative Depth `83.33%`、HR cross aggregate
`62.00%`、MathVista numeric commonsense `55.56%` 和 visual QA `59.52%` 等定位，同时
Relative Reflectance 从 matched 的 `46.67%` 升至 `66.67%`；原先 arithmetic 和 math-word
优势分别回落至 `63.46% / 74.60%`。这些变化必须作为能力迁移而不是单一“更强/更弱”报告。

Atomic full prompt 的 Macro* 为 `60.4684`，比自身 matched prompt 低 `2.6142 pp`、比
true-1M Original 低 `0.8463 pp`，但比 TGVF full prompt 高 `1.9546 pp`。七项相对自身 matched 的
delta 为 VStar `−1.0471`、HR `−6.5000`、BLINK-180 `−2.7778`、OCR mean
`−3.8370`、MMMU-269 `−5.2045`、MathVista `+0.6667`、MathVerse `+0.4000 pp`。
这说明组合工具在广义完整 prompt 下保留若干分项价值，但未超过同像素 raw-direct 总体基线，
也不能由此单独决定其正文层级：预冻结的 Atomic
favorable regimes 中，HR cross 从 `68.00%` 降到 `56.00%`，Relative Reflectance 从
`70.00%` 降到 `46.67%`；只有 MathVerse Vision Only 从 `51.00%` 保持到 `52.00%`，而原先的
负面边界 Vision Intensive 从 `49.00%` 回升到 `53.00%`。这些数值支持 prompt sensitivity，
但不能替代下一小节已经闭合的 target-only 单变量评测。

OCR 生成行为给出了一个额外的协议边界。600-row OCR 切片上，TGVF full prompt 的答案长度
p95 / p99 为 `24,324 / 89,469` 字符，Atomic 为 `3,849 / 19,641`；TGVF 的极长输出尾部在
完整 prompt 下没有被消除，并与其 OCR mean 下滑同时出现。该相关性只作 termination/verbosity
诊断，不能据此单独断言准确率下降的因果来源。

full-prompt 权威 summary：

- TGVF：`artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-S64-FULL-PROMPT-V5-CAP6-COREDEV2511-SEED42-V1/step64/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
- Atomic：`artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-S16-FULL-PROMPT-V5-CAP6-COREDEV2511-SEED42-V1/step16/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。

计划文件：

- `configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s64_full_prompt_v5_cap6_coredev2511_plan.json`
- `configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s16_full_prompt_v5_cap6_coredev2511_plan.json`

#### 5.2.2 主要验证：target-only matched prompt

用户原定改动只有一项：补充 target 的详细定义和使用案例。新协议
`target_detailed_matched_prompt_v1` 因此冻结以下最小差异合同：

- 逐字保留 matched 的 `USER_PROMPT_V2`，包括已有的 `<think>...</think>`、按需 tool call 和
  plain-text final；不增加逐轮 reasoning 或分类别 final-only 规则；
- 保留同一 tool name、可执行 schema、Hermes call frame、`tools=[]` chat-template 路径、6 次
  运行上限和 matched success renderer；observation **不**回显 target；
- 仅在 matched system prompt 的 `<tools>` 与调用格式说明之间插入一个 target guide：定义
  target 必须同时包含 referent/entity 与所需视觉证据，并给出 3 个 valid 和 3 个 invalid 示例；
- checkpoint、CoreDev-2511、official scorer、temperature 1 和 seed42 不变；seed namespace
  逐字复用原 matched evaluation，使每题每轮随机流与原 matched run 对齐；
- S64/S16 在查看结果前冻结，不重新选 step。

实现与状态：

- [x] target-only prompt identity、system/user SHA256、protocol SHA256 与差异白名单冻结；
- [x] 81 项相关测试通过；白名单证明 matched system prompt 只有一个连续 target-guide 插入块，
  user suffix 完全相同，且不存在 full prompt 新增的逐轮/final-only 文本；
- [x] 两份真实 snapshot 预检通过：TGVF/Atomic 分别使用 matched 的
  `render_qwen_native_matched_tgvf_success_environment_text` /
  `render_qwen_native_matched_crop_tgvf_success_environment_text`；
- [x] 两臂均完成 2,240 条共同支持单图推理；七套官方 scorer 均通过，共覆盖各自完整的
  2,511-row CoreDev 输入合同；TGVF / Atomic judge parse failure 为 `4 / 2`，均按固定
  deterministic-incorrect 策略处理且低于阈值；
- [x] 与 matched、广义 stress、Crop S80 native-default 历史参考、Original 同表回填。

本表 Original 使用 true-1M reference；TGVF/Atomic 与其对齐像素、scorer 和 sample reference。
Crop native-default historical 行不共享像素合同，其 `Δ vs Original` 仅为描述性差值。

| Arm | Macro* | Δ vs own matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct true-1M | 61.3147 | — | — | 72.7749 | 66.5000 | 63.8889 | **59.7877** | 37.9182 | **75.3333** | 53.0000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | n/a | −2.1362 | **78.5340** | 61.5000 | 57.7778 | 55.4948 | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 / matched | 59.8086 | reference | −1.5061 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / target-only | 58.1788 | −1.6298 | −3.1359 | 75.3927 | 61.5000 | 63.8889 | 40.1453 | 45.7249 | 68.0000 | 52.6000 |
| Atomic S16 / matched | 63.0827 | reference | +1.7680 | 71.7277 | **73.5000** | **66.1111** | 54.2720 | **51.3011** | 69.6667 | 55.0000 |
| Atomic S16 / target-only | 60.8253 | −2.2574 | −0.4894 | 71.7277 | 70.5000 | 61.6667 | 49.3089 | 46.8401 | 69.3333 | **56.4000** |

target-only 改动没有带来总体提升；相对 true-1M Original，TGVF 与 Atomic 的 Macro* 分别低
`3.1359 / 0.4894 pp`。TGVF 的主要回退在
HR `-5.00 pp`、OCR mean `-4.40 pp` 和 MathVista `-4.33 pp`，同时 VStar、MMMU-269、
MathVerse 分别提高 `+1.05 / +0.74 / +2.20 pp`。Atomic 的主要回退在 OCR mean `-4.96 pp`、
MMMU-269 `-4.46 pp`、BLINK-180 `-4.44 pp` 和 HR `-3.00 pp`，MathVerse 提高 `+1.40 pp`。
因此“更详细 target 定义本身普遍提高准确率”不被支持；可以支持的是两种方法在该单变量 prompt
干预下存在明确的 benchmark-specific sensitivity，且 Atomic 总体接近、但没有超过同像素
raw-direct reference。

target-only prompt 同时改变了工具调用行为。下表沿用 4.4 节定义；`Execution yield` 为
`executed calls / (executed calls + invalid attempts)`，coverage、calls/question 与 repeat-use
rate 的分母为共同的 2,240 条 single-image trajectory，calls/used question 则只以成功使用题为分母。

| Method / prompt | Attempted questions | Successful-use questions | Executed calls | Invalid attempts | Execution yield | Calls/question | Calls/used question | Repeat-use questions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TGVF S64 / matched | 2,012 (89.82%) | 2,010 (89.73%) | 2,011 | 3 | 99.85% | 0.898 | 1.000 | 1 (0.04%) |
| TGVF S64 / target-only | 2,118 (94.55%) | 2,116 (94.46%) | 2,120 | 6 | 99.72% | 0.946 | 1.002 | 4 (0.18%) |
| Atomic S16 / matched | 1,866 (83.30%) | 1,863 (83.17%) | 2,300 | 36 | 98.46% | 1.027 | 1.235 | 221 (9.87%) |
| Atomic S16 / target-only | 2,089 (93.26%) | 2,088 (93.21%) | 2,420 | 21 | 99.14% | 1.080 | 1.159 | 174 (7.77%) |

逐 set 的 successful-use coverage 与调用强度如下。每格依次为
`successful-use questions / n (rate); executed calls / question`。

| Set | n | TGVF matched → target-only | Atomic matched → target-only |
|---|---:|---:|---:|
| VStarBench | 191 | 186/191 (97.38%); 0.974 → 191/191 (100.00%); 1.000 | 188/191 (98.43%); 1.000 → 189/191 (98.95%); 0.990 |
| HRBench4K | 200 | 199/200 (99.50%); 0.995 → 200/200 (100.00%); 1.000 | 198/200 (99.00%); 1.090 → 200/200 (100.00%); 1.050 |
| BLINK single-image | 180 | 178/180 (98.89%); 0.989 → 180/180 (100.00%); 1.000 | 178/180 (98.89%); 1.706 → 180/180 (100.00%); 1.572 |
| OCRBench v2 | 600 | 556/600 (92.67%); 0.927 → 571/600 (95.17%); 0.952 | 560/600 (93.33%); 1.213 → 576/600 (96.00%); 1.115 |
| MMMU-Pro single-image | 269 | 232/269 (86.25%); 0.862 → 252/269 (93.68%); 0.948 | 243/269 (90.33%); 1.067 → 263/269 (97.77%); 1.134 |
| MathVista MINI | 300 | 268/300 (89.33%); 0.893 → 285/300 (95.00%); 0.953 | 222/300 (74.00%); 0.937 → 261/300 (87.00%); 1.100 |
| MathVerse MINI | 500 | 391/500 (78.20%); 0.784 → 437/500 (87.40%); 0.874 | 274/500 (54.80%); 0.576 → 419/500 (83.80%); 0.868 |

Results 层面，target-only prompt 将 TGVF/Atomic 的 successful-use coverage 分别提高
`4.73/10.04 pp`，其中 Atomic 在 MathVerse 的覆盖率提高 `29.00 pp`；但两者 Macro* 同时下降
`1.6298/2.2574 pp`。因此更多调用没有转化为总体准确率增益。Atomic 的总调用强度上升，但
repeat-use questions 反而从 `221` 降至 `174`，说明变化主要来自更多题转为一次调用，而不是普遍
增加多轮检索。

target 长度和整条输出长度也没有呈现统一方向。对两臂都成功调用工具的共同样本，仅比较第一次
调用时，Atomic（n=1,840）的 target 平均从 `14.22` 增至 `15.21` 个空白分词（`+0.99`），
TGVF（n=1,977）则从 `14.29` 降至 `13.78`（`−0.50`）。按 trajectory 汇总所有 assistant turn
的 `sampled_token_count`，并以经验顺序统计量取分位数，TGVF 的 mean/P50/P95 从 `812.1/116/3,376` 变为
`1,092.0/111/5,888`，Atomic 从 `529.7/155/1,937` 变为 `501.3/147/1,960`。TGVF 的均值上升
主要来自更长的尾部，而 Atomic 均值下降；现有证据不支持用“target 或输出统一变长”解释性能回退。
这些均为生成行为统计，不是 target 语义合格率；后者仍由 5.3 节盲审决定。

计划文件：

- `configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s64_target_detailed_matched_v1_coredev2511_plan.json`
- `configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s16_target_detailed_matched_v1_coredev2511_plan.json`

权威 summary：

- TGVF：`artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-S64-TARGET-DETAILED-MATCHED-V1-COREDEV2511-SEED42-V1/step64/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
- Atomic：`artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-S16-TARGET-DETAILED-MATCHED-V1-COREDEV2511-SEED42-V1/step16/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。

### 5.3 Atomic 无偏 target 合格率审计

目的：把“协议可解析”与“语义 target 合格”分开，确定 Atomic 是否能在正文中被描述为稳定地产生
高质量 target。

固定抽样与盲审设计：

- matched prompt 与 target-only prompt 各抽取 200 条**实际使用工具**的 trajectory；广义
  full-prompt stress arm 只作附加诊断；
- 以 `dataset × arm` 分层，层内按 SHA256 排序抽样；每条 trajectory 只审第一次工具调用，防止
  多调用样本过度加权；
- 标注者不看到 arm 名、checkpoint、最终答案、正确性、reward 或 benchmark score；
- 两名独立标注者；分歧由第三人裁决；报告 agreement 和逐标准 Wilson CI。

逐条全部通过才计为合格：

1. referent 可识别；
2. 明确说明要取得的视觉证据/属性/关系，不只是物体名；
3. 不泄漏猜测答案或后续计算结果；
4. bbox 与 target 一致；
5. 关系/比较任务包含必要实体；
6. target 是视觉查询，而不是含糊的后续推理指令。

当前状态：

- [x] 审计 rubric 和正文纳入门槛冻结；
- [x] 在 matched S16 的 1,863 条工具 trajectory 上实测物化 200 条盲审样本；七个数据集按
  population-proportional quota 覆盖，review view 未泄漏 arm、dataset、sample ID、correct、
  reward、score 或 final answer；
- [x] 既有 matched/广义-full blind audit pack 已生成：每臂 200，共 400 条；review view 仅含 bbox、
  call index、image path/hash、question、opaque review ID、schema 和 target；400 个 review ID
  唯一，未出现 arm、dataset、sample ID、correct、reward、score 或 final answer 字段；
- [x] 基于已闭合的 target-only inference 物化新的 matched/target-only 正式盲审包：两臂各
  200 条，共 400 条；matched/target-only 可用工具 trajectory population 为 `1,863/2,088`，
  均按七个 dataset 的 population-proportional largest-remainder quota 抽样；
- [x] reviewer-facing 字段与图片路径盲化复核：400 个 review ID 唯一，400/400 图片 SHA256
  复现；匿名图片路径不含 dataset，review items 不含 arm、dataset、sample ID、checkpoint、
  final answer、correctness、reward 或 score；
- [ ] 双人盲标、裁决、Wilson CI 与 agreement；
- [ ] 根据审计结果界定 Atomic 探索性分析可使用的 target-quality 表述。

正式盲审根目录：
`artifacts/evaluation/neurips-workshop-atomic-target-audit-matched-target-only-20260828-v2/`。
manifest SHA256 为
`a352a0692834b3f556226ee2bccf656d668adebce6754095c9b65964064035b7`，状态为
`ready_for_two_reviewer_blind_annotation`；review image tree identity 为
`81936292aa6c9708f87bb13181b07f0dfe7e4aa42e678676631f7964bab523b5`。generator、输入 rank
文件 SHA、quota、review image inventory 与 coordinator key 均由 manifest 固定。独立 pack
复核通过 6 个绑定文件、400 条 review row、400 张匿名图片和 `200/200` source split。旧
`neurips-workshop-atomic-target-audit-20260826-v1` 仅保留 matched/广义-full stress 的历史诊断，
不替代当前主要审计。人工标注完成前不得报告 target 合格率或据此升级 Atomic 的正文地位。

### 5.4 No-Tool RL：corrected true-1M RL-only control

这条新线路回答一个独立于工具机制的问题：当前 Crop、TGVF 和 Atomic 相对 raw Original 的
增益中，有多少可以由**同等训练预算下的 full-model RL 本身**解释。为避免把 raw base 与训练后
模型混为一谈，固定名称为 **No-Tool RL**；`Original` 始终只表示未经本文 RL 的原始
Qwen3-VL-8B-Instruct。

#### 5.4.1 事前冻结的训练合同

| 项目 | 冻结设置 |
|---|---|
| 正式主终点 | **S32**；不根据中途结果延长或改选更优 step |
| 保存与评测点 | S0、S8、S16、S32 |
| base / update | 与 PRL25 三条工具线相同的 Qwen3-VL-8B-Instruct；full-model Qwen update |
| 数据顺序 | 同一不可变 Teacher25 schedule `PRL22-TEACHER25-MIXED-SCHEDULE-v1` 的前 `32 × 16 = 512` 个 prompt |
| rollout 预算 | global batch 16、每 prompt 16 rollouts，共 `8,192` 个 rollout |
| 优化设置 | seed42、temperature 1、learning rate `1e-6`，其余 optimizer / GRPO 设置与 PRL25 对齐 |
| 奖励 | 保留相同 answer judge / correctness reward；无 TGVF utility reward；工具与重复调用惩罚因无工具而恒为零 |
| 明确移除 | Crop、TGVF、RP67 representation、tool schema、tool observation、tool agent loop |

No-Tool 的 visual-row 训练 prompt 只保留图像、canonical question、`<think>...</think>` 推理要求
和 plain-text final 要求；不加入工具定义、target 定义、bbox 案例或工具使用指令。ThinkLite 行保持
Teacher25 各条线路已有的原始 no-tool 合同不变。任何结构化工具调用文本都按 protocol error
处理，不能被当作有效动作。

#### 5.4.2 固定评测矩阵

S0/S8/S16/S32 checkpoint 均执行 **matched no-tool@1M**：使用与 No-Tool RL visual-row
训练一致的 no-tool prompt，作为优化动态以及 `No-Tool RL ↔ TGVF/Atomic RL` 的主要像素对齐
控制。Crop/No-Tool 的 corrected true-1M 重跑现已闭合，因此文首五方表已消除 effective pixel
contract 的主要混杂。协议复核发现历史 Original 实际使用 `max_pixels=262144`，而上述 matched 评测使用
`image_max_pixels=1003520`。因此在不改变冻结 checkpoint 的前提下，新增 **S32
raw-direct@512**：其配置与历史 Original 逐字段一致，只替换模型路径。无需重跑 S0 raw-direct，
因为历史 Original 本身就是同一 base model 的 S0 raw-direct@512 结果。

| 评测 | Checkpoint | Pixel cap | 主要问题 |
|---|---|---:|---|
| matched no-tool | S0 / S8 / S16 / S32 | 1,003,520 | RL 学习动态与 PRL25 matched 控制 |
| raw-direct transfer | S32 | 262,144 | 在 Original 合同下，32-step RL 后性能是否仍提升 |
| historical raw-direct | Original = base/S0 | 262,144 | 上一行的严格 base comparator |

matched no-tool 使用同一 CoreDev2511 七项、Macro*、完整 aligned sub-benchmark 和逐 set 输出；
No-Tool RL 的工具调用率定义上应为 0，另做结构化工具文本泄漏审计。S32 是唯一正式 headline，
S8/S16 只呈现学习动态，不能用于 post-hoc checkpoint 选择。

#### 5.4.3 可支持的结论与边界

- 当前 frozen No-Tool S32 超过三个工具方法，因此不支持工具方法的 aggregate superiority；
  工具证据必须转向预先界定的能力切片、matched utility 与行为机制。
- 同协议 S32−S0 近似为零，因此 corrected true-1M 并不支持“32-step no-tool RL 自身带来总体
  增益”；No-Tool 相对 Original 的差值同时包含 prompt/direct-loop 合同差异。
- matched no-tool 去除了工具 schema，因此它是当前最强的 RL-only control，但仍不可消除
  “存在工具 schema / agent loop”这一协议差异。已有 Original 的 raw direct 结果也不能替代
  matched RL-only 比较；S32 raw-direct@512 已显示 `−1.0013 pp` transfer，不替代 matched 工具对照。

#### 5.4.4 正式结果与学习动态

下表全部使用 matched no-tool 协议。S32 是事前冻结的唯一 headline；S8/S16 只用于描述动态，
不能因为 S16 的 Macro* 略高而把正式 checkpoint 改选为 S16。首先报告 corrected true-1M V2；
旧 processor-default 表随后单列，仅用于保留实验史。

##### Corrected true-1M V2（当前正式结果）

| Step | Macro* | Δ vs S0 | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse | Parse fail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 63.7619 | — | **72.2513** | 64.0000 | 66.6667 | 48.5015 | 54.6468 | **77.6667** | 62.6000 | 0 |
| S8 | 63.7354 | −0.0264 | 71.2042 | 66.0000 | 66.1111 | 48.8990 | **56.1338** | 75.0000 | **62.8000** | 0 |
| S16 exploratory | **63.9307** | **+0.1688** | 71.7277 | **68.0000** | 69.4444 | 47.6957 | 54.6468 | 76.0000 | 60.0000 | 1 |
| **S32 frozen** | 63.7520 | −0.0098 | 70.1571 | 65.5000 | **71.1111** | **49.3928** | 53.9033 | 75.0000 | 61.2000 | 2 |

S16 的唯一 judge parse failure 是
`hr_bench_4k_800/hr_bench_4k_snapshot_hr_bench_4k_parquet/700_000700`；S32 的两个 failure 是
`vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/110_000110` 与
`vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/167_000167`。三条均按冻结
fail-closed 规则确定性记错；四份 summary 仍为 `status=pass`。

##### Processor-default historical（已降级，不进入当前主表）

| Step | Macro* | Δ vs S0 | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 64.4712 | — | 79.5812 | 61.0000 | 65.5556 | 48.8072 | 57.6208 | 77.3333 | 61.4000 |
| S8 | 66.1132 | +1.6420 | 83.7696 | 66.0000 | 65.5556 | 48.9899 | 56.8773 | **78.0000** | **63.6000** |
| S16 | **66.9028** | **+2.4317** | 82.1990 | **69.5000** | **71.1111** | 50.6324 | 56.8773 | 75.0000 | 63.0000 |
| **S32 frozen** | 66.6853 | +2.2141 | **84.2932** | 69.0000 | 70.5556 | **50.8528** | 55.7621 | 75.3333 | 61.0000 |

以上四个旧值实际回退到 processor default `16,777,216`，不能与 corrected true-1M V2 混用，
也不能据此声称 S32−S0 获得 `+2.2141 pp` 的当前有效增益。

同合同 raw-direct transfer 结果如下。OCR EN/CN 分别为 `49.5995 / 46.2760`；MathVerse 五版本
为 Text Dominant `70.0`、Vision Only `30.0`、Text Lite `48.0`、Vision Intensive `48.0`、
Vision Dominant `50.0`。

| Method | Macro* | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct@512 | 55.3556 | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| No-Tool S32 raw-direct@512 | 54.3543 | −1.0013 | 51.3089 | 59.0000 | 60.0000 | 47.9377 | 39.0335 | 74.0000 | 49.2000 |

观察与解释边界：

- Corrected true-1M 的 S32−S0 为 `−0.0098 pp`，应描述为 aggregate-flat。S16 的探索性峰值
  仅为 `+0.1688 pp`，不触发 post-hoc checkpoint 重选。
- Frozen S32 比 Atomic、Original、Crop S32 与 TGVF 分别高
  `0.6693 / 2.4373 / 2.6814 / 3.9434 pp`。这些是统一像素、scorer 与 sample reference 下的
  端到端差值；其中 No-Tool 与 Original 仍跨 prompt/loop，与工具方法还跨工具 schema 和 agent loop。
- 近似不变的 Macro* 掩盖了分项重分配：S32 相对 S0 的 BLINK 上升 `4.4444 pp`，但 VStar、
  MMMU 与 MathVista 分别下降 `2.0942/0.7435/2.6667 pp`。
- 该比较仍不是“只切换工具开关”的严格因果消融：matched no-tool 同时去除了工具 schema 和
  agent loop。Original raw direct 必须保留，但其 prompt/protocol 又不同，只作端到端参考。

#### 5.4.5 输出长度、格式与零工具行为

四步各 2,240 条 accepted rank JSONL 的 sampled-token 统计如下。P50/P90/P95 使用 empirical
order-statistic 口径；空 final 包含达到 token 上限或 invalid-format 后没有可评分 final 的行。

| Step | Mean sampled tokens | P50 | P90 | P95 | `max_tokens` | `invalid_format` | Empty final | Tool calls / errors / observations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1005.88 | 171 | 3068 | 5443 | 3 | 3 | 6 | 0 / 0 / 0 |
| S8 | 1024.41 | 163 | 3270 | 5808 | 4 | 0 | 4 | 0 / 0 / 0 |
| S16 | 944.06 | 145 | 2822 | 5416 | 4 | 0 | 4 | 0 / 0 / 0 |
| S32 | 839.73 | 127 | 2065 | 5065 | 5 | 0 | 5 | 0 / 0 / 0 |

S32 的 mean/P50/P90 均低于 S0，说明训练后输出总体变短；但仅凭长度与分数共同变化不能建立
因果关系。四步工具调用、工具错误与 observation 均为零，符合 direct-only no-tool 合同。

#### 5.4.6 正式验收与 finalizer 边界修复

Aggregate scoring receipt 的 `status=complete`，identity 为
`69328dfe889366650e96a0582eaa86a27df1ac349751ee40493d83f47c92a955`，文件 SHA256 为
`0fc8bcf3865c4d6ee360206da318c7913afb8c4cf659ea4bcb29104ad132db59`；它固定每步
`max_pixels=1,003,520`、2,511 rows 与 7 slices。S0/S8/S16/S32 accepted summary SHA256 为
`37c7b376cfc53dcd9b3d47101348fcb25fcb05c74f9a7d6d446ae4219edf0bbe`、
`b28d941746592ef80cdd8157bf077c058bf1253a76cf1ccd17e4bc8af5bc9223`、
`13bf456370d23540c8130b855f30adaae8b4b6c21f24dabcd9a37ff058390d72`、
`bd6057d798c37791714033c4f9a734f435264ec1242fc841fdaad4fd064d897a`。

闭合过程中，aggregate finalizer 的二次身份检查曾把 worker 在结果 hash 之后附加的
`wall_seconds` telemetry 错误纳入语义 hash。对齐正式 worker validator 后，四步共
`8,960/8,960` 行全部通过；rank-file SHA/size 仍绑定包含 telemetry 的精确字节。这是验收器的
身份边界修复，不是实验失败，推理与评分均无需重跑。

当前状态：

- [x] 名称、S32 主终点、训练预算、matched no-tool 评测协议和解释边界已冻结；
- [x] 独立干净分支上的 no-tool prompt/data route、空工具 schema、run config、S32 守护脚本与
  CPU compose/回归测试；
- [x] 真实 1-step canary：首轮在 Step 0 前发现共享 termination builder 仍把
  `</tool_call>` stop 当作全路径硬条件；修复后 canary 已闭合 Step 1 且零工具调用；
- [x] 正式 32-step 训练：S8/S16/S32 永久 checkpoint 与最终 supervisor acceptance 均已闭合；
- [x] S0/S8/S16/S32 matched no-tool 的 `4 × 2,240` 条单图推理；
- [x] S0/S8/S16/S32 matched no-tool 的七项正式评分；
- [x] corrected true-1M V2 结果回填主表、学习动态、行为表和 claim ledger；
- [x] Appendix A 加入 frozen S32 后的 59-slice winner 与 pairwise W/L/T 机械重算。
- [x] S32 raw-direct@512 的七个完整 slice、共 2,511 条推理；
- [x] S32 raw-direct@512 的七项官方评分、180/269 单图 headline 对齐和 Original 同合同差值；
  Macro* `54.3543`，即 `−1.0013 pp`。

执行与审计快照（2026-08-26 07:18 JST）：

- 正式 run ID：
  `PRL-25-F-QWEN3-INSTRUCT-FULL-NO-TOOL-RL-BS16-N16-TFREE-TEACHER25-32STEP-WS8`；
  canary 为独立 `BS4 × n2 / world4 / 1-step` 功能门，不进入正式 lineage。
- 执行分支 `neurips-notool-rl-s32` 已 push 到远端；当前授权恢复 commit 为
  `7645fe4a5bb095cb9e0c9cbb9f428abd98ef9aae`，核心 termination 修复 commit 为
  `8140ce05b2fbb6d5c9c9fd0594eb9f5b5984f6aa`。
- 回归证据：no-tool protocol/launcher/runtime/vLLM 相关测试 `129 passed`，native-agent-loop
  补充测试 `14 passed`；formal 与 canary CPU compose 均通过。
- 首轮失败发生在 rollout 前，因此没有污染恢复 canary。修复把 tool-boundary interpretation
  显式关闭于 No-Tool schema：工具样式文本仍完整保留并由 direct-only loop 判为 protocol error，
  既不解析/执行，也不让单条坏输出击穿整次运行。
- 恢复 canary 在 `137.21 s` 内完成 Step 1：4 prompts、8 trajectories、658 generated policy
  tokens；`successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。world4 rolling checkpoint 的 model/optimizer/extra-state 四组 shards、
  paired project state、tracker 与 metrics 均已验收。canary 模式按框架合同不生成 permanent 副本。
- formal S32 supervisor 已通过 canary gate 与最终 CPU preflight，并在 tmux
  `prl25_f_no_tool_s32` 启动 fresh-S0 训练；正式运行使用 8 卡并保存、逐一验收 S8/S16/S32 的
  world8 模型、优化器和 extra-state shards。W&B 凭证缺失时只切换为本地 offline telemetry，
  不改变训练合同。
- 2026-08-26 07:23 JST 实时节点：world8 FSDP 与 8 个 vLLM 实例已完成初始化，trainer 显示
  `Training Progress 0/32`，首个 BS16 × n16 rollout batch 正在执行；tmux 存活，8 卡均已被该
  正式任务占用，日志中没有 traceback。此时 tracker/metrics 尚未出现是因为 Step 1 仍未提交，
  不能把 `0/32` 误写成已完成一个 optimizer step。
- 2026-08-26 07:44 JST 实时节点：正式训练已提交 **Step 2/32**，累计 32 prompts、512
  trajectories、217,722 generated policy tokens；累计 mean answer reward 为 `0.6855`，format
  error rate 为 `0.2266`。按 no-tool 合同，`successful_tgvf_observations=0`、
  `tool_call_attempt_rate=0.0`、`mean_tool_call_attempts=0.0`。Step 1/2 分别耗时约
  `626.8 / 592.2 s`；Step 2 rolling checkpoint 已验收 8 份 model、8 份 optimizer 与 8 份
  extra-state shard。tmux/supervisor 存活且无 traceback；采样时 GPU 正处于 checkpoint 写盘后的
  短暂空闲窗口。按当前均速粗估，S8 约在 1 小时后、S32 约在 5 小时后到达；该估时不作为实验结果。
- 2026-08-26 08:52 JST 关键节点：正式训练已提交并验收首个保留点 **S8/32**。累计
  128 prompts、2,048 trajectories、928,070 generated policy tokens；累计 mean answer reward
  为 `0.5962`，format error rate 为 `0.2378`，这些是在线训练遥测而非 benchmark 结果。
  No-Tool 合同持续满足：`successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。S8 rolling checkpoint 与 permanent checkpoint 均存在；后者含
  permanent receipt、8 份 model、8 份 optimizer、8 份 extra-state shard 及配对 project state，
  共 40 个文件。tmux/supervisor 存活且无 traceback，训练已继续向 S16/S32 推进；按当前约
  `11.1 min/step` 的运行均速粗估，S16 约还需 1.5 小时、S32 约还需 4.5 小时。
- 2026-08-26 13:19 JST 完成节点：正式训练已闭合 **S32/32**，supervisor 写入
  `step32-accepted` 后正常退出。最终累计 512 prompts、8,192 trajectories、2,826,178
  generated policy tokens；累计 mean answer reward 为 `0.6182`，format error rate 为
  `0.1564`，S32 单步对应 `0.7227 / 0.1836`。这些数值只描述训练遥测，不替代 CoreDev
  benchmark。No-Tool 合同全程满足：8,192 条 trajectory 中
  `successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。S8、S16、S32 permanent checkpoint 均包含 receipt、8 份
  model、8 份 optimizer、8 份 extra-state shard 与 paired project state，各 40 个文件。
  Step 9 后 answer judge 曾出现一次 HTTP 429 transient-window failure；守护器保留已提交
  checkpoint，冷却后从同一正式 lineage 自动恢复，最终 prompts/trajectories 计数与冻结合同
  精确一致。训练成功后的 vLLM teardown 输出 `pure virtual method called`，发生在最终 metrics、
  permanent receipt 与 100% progress 之后，不影响 S32 验收。下一步只执行事前冻结的
  matched no-tool 评测，不依据训练 reward 改选 checkpoint。
- 2026-08-26 13:37 JST 评测启动节点：matched no-tool 的 full-model 原生像素 evaluator、
  S0/S8/S16/S32 snapshot/merge 流水线与可恢复 supervisor 已通过 CPU 回归（相关既有路径与
  新增 no-tool 专项合计 `26 passed`），执行分支推送至 `36702a1`。tmux
  `prl25_f_notool_dual_eval` 已启动，当前正在为 S0 构建并绑定 immutable full-model snapshot；
  随后自动执行单题真机 smoke、四卡完整 S0 推理，并在其运行期间合并 S8 FSDP 权重。S8 与
  S0 分配到不同四卡组，后续 S16/S32 复用对应组并保持相同 CoreDev2511 task manifest、
  paired seed namespace 与 matched no-tool prompt。此节点表示评测已进入工件准备/执行流水线，
  尚不表示任何 CoreDev 分数已经产生。
- 2026-08-26 13:54 JST 文章范围决策：在任何 No-Tool RL CoreDev 结果产生前，作者取消
  raw-direct transfer 复测，只保留 S0/S8/S16/S32 matched no-tool。raw-direct 从未启动，当前
  supervisor 本身也只执行 matched 路径，因此无需中断或重启在跑任务。已有运行目录和 tmux
  名称中的 `DUAL` / `dual` 是启动时的历史标签，不表示第二套协议产生了数据。
- 2026-08-26 16:44 JST 配置级分辨率复核与范围修订节点：逐字段核对历史 Original resolved
  config 后确认其 `max_pixels=262144`；Crop/TGVF/Atomic/No-Tool matched 的冻结配置均记录
  `image_max_pixels=1003520`。当时因此把四者视作同一有效像素合同；2026-08-28 processor audit
  后该判断只对 TGVF/Atomic/No-Tool 成立，Crop 配置值被 fast processor 忽略。13:54 的“不测
  raw-direct transfer”决策仍被 Original/No-Tool 的可比性证据
  覆盖，新增且只新增冻结 S32 的 raw-direct@512。七个 CoreDev 完整 slice 共 2,511 条已按
  dataset 隔离并行启动；配置除 S32 model path 外复现 Original 的生成参数、seed namespace 与
  raw-direct prompt。该节点只是运行状态，不提前报告 Macro*。
- 2026-08-26 14:34 JST 推理节点：S0 matched no-tool 已闭合全部 `2,240/2,240` 条单图任务，
  completion marker 与状态文件均已落盘；CoreDev manifest 中另有 271 条多图任务按既有协议显式
  hold，不计作未完成。S8 已生成 `1,900/2,240` 条单图结果（`84.82%`）；S16 真机 smoke 已通过，
  四个正式 worker 正在初始化；S32 full-model checkpoint merge 已并行启动。tmux 与 supervisor
  均存活，当前未发现致命错误。此节点只证明推理工件进度，S0 尚未执行七项评分，因此不产生
  Macro*、sub-benchmark 优劣或论文结论。
- 2026-08-26 15:23 JST 推理闭合与评分启动节点：S0、S8、S16、S32 均已完成各自全部
  `2,240/2,240` 条单图推理，四臂 completion marker 与总 `matched-inference-complete` 均已落盘。
  严格评分守护器已在独立 tmux `prl25_f_notool_matched_score` 启动，执行实现已推送至
  `daf1b26`。四臂均已通过 scoring-view 物化校验：每臂 2,240 条观测单图答案、271 条显式
  fail-closed 多图任务、2,511 条官方总计及七个 slice；本地 Qwen2.5-72B-Instruct judge 正在
  GPU0/1 初始化。此时尚无任何 slice 完成正式评分，也尚无 Macro* 或 sub-benchmark 结论。
- 2026-08-26 15:46 JST 评分 fail-closed 与恢复节点：首轮 28 个 scorer 在 15:25 因执行分支
  wrapper 同时收到上游互斥的 `--config` 与 `--data/--model` 参数而全部在正式评分前退出；已保存
  推理、四臂 scoring view 和 source manifest 均未改变，也没有产生可误用的正式分数。恢复实现
  移除该冲突，并对每个 step/dataset 显式绑定不可变 source run ID、source manifest、`--reuse`
  与 `--reuse-aux infer`；相关 pinned-reuse/参数合同测试 `21 passed`，修复已推送至 `4bbddf4`。
  重启后本地 Qwen2.5-72B-Instruct judge 已通过 health gate，28/28 个 slice scorer 均已真实启动，
  暂未再次出现参数断言。此节点仍是运行状态，不表示任何 slice 或四臂汇总已经完成。
- 2026-08-26 15:49 JST 并发评分隔离节点：恢复批次证明 pinned-reuse 参数链有效，但四个并行
  OCRBench scorer 共用执行 checkout 下的 `.vlmeval` 临时目录，发生 ground-truth 文件竞争；
  OCR slice fail-closed，其他 dataset 不受该临时目录问题影响。截至该快照已有 19 个 slice 写入
  通过 wrapper postflight 的 pinned-reuse receipt，另有 8 个 scorer 仍在运行。修复将当前工作目录
  隔离到各 step/dataset，并在恢复时跳过已有 receipt、只重跑未闭合 slice；CPU 合同测试
  `34 passed`，修复已推送至 `9744dca`。自动恢复 waiter 已启动，将在当前批次自然退出后接管；
  四臂 summary 和 Macro* 仍未生成。
- 2026-08-26 16:04 JST 正式评分闭合节点：恢复流程已生成并校验 `28/28` 个 pinned-reuse
  receipt、四个 step 的 `coredev-2511-eval-summary.json` 和总 `matched-scoring-complete` marker；
  四份 summary 均为 `status=pass`、`slice_count=7`、`sample_count=2,511` 且
  `judge_parse_failure_count=0`。最终汇总改为从 receipt 固定的 destination eval ID 精确读取，
  避免旧失败 OCR eval 被“latest”规则误选；修复和结果生成代码已推送至执行分支 commit
  `abee6a4`。当时得到的 S32 `66.6853` 后续已确认属于 processor-default historical，不能进入
  当前 true-1M 主表；当前只使用 corrected V2 S32 `63.7520`。
- 2026-08-26 17:38 JST `@512` 控制节点：冻结 S32 raw-direct 的七个完整 slice 推理均已
  完成。MathVerse judge 行已产生，但历史 raw-direct TSV 缺少官方聚合所需的
  `metadata.problem_version`；恢复实现只从固定 MathVerse `testmini.json` 补入该元数据，并逐行
  验证 `prediction` 完全不变，随后重新执行官方聚合。七个推理 source run 已显式绑定；本地
  Qwen2.5-72B-Instruct judge 正在 GPU0/1 初始化。与此同时，Crop S80、TGVF S64 与 Atomic S16
  的 nominal matched@512 计划已由自动 supervisor 排队，待 S32 评分闭合后依次使用 GPU4–7；
  后续 processor audit 证明其中只有 TGVF 与 Atomic 真正应用了 262,144，Crop 计划不是有效
  单变量控制。
  实现与计划已推送到执行分支 `neurips-notool-rl-s32` 的 `81cec97`，source-run 选择修复为
  `12b4ca1`；相关 targeted 回归为 `79 passed`。该节点仍不提前报告任何新 Macro*。
- 2026-08-26 17:50 JST S32 raw-direct@512 正式闭合节点：七项 accepted summary 为
  `status=pass`、`sample_count=2,511`、`slice_count=7`。MathVerse 只补入固定 source JSON 的
  `problem_version`，500 条 prediction 逐值不变；BLINK/MMMU 使用冻结 CoreDev task manifest
  按 index 生成只读 coverage view，原 prediction、hit 和 scorer 字段逐值不变，只将多图 reference
  从 180/269 单图 headline 排除。最终 Macro* 为 `54.3543`，比 Original raw-direct@512 低
  `1.0013 pp`。汇总与 coverage 实现已推送至执行分支 `e54b851`，相关新增回归 `62 passed`。
  完成 marker 触发三方法顺序队列，Crop S80 nominal @512 已于 17:50 JST 开始准备 snapshot。
- 2026-08-26 18:35 JST 工具方法 nominal `@512` 并行接管节点：Crop S80 推理以 `2,240/2,240`
  单图记录完整闭合；原顺序 barrier 被移除。既有 TGVF S64 推理进程及已写记录无损保留并继续
  使用 GPU0–3，Atomic S16 已立即在 GPU4–7 开始 checkpoint materialization/validation；TGVF
  释放 GPU0/1 后将与 Atomic 推理重叠执行 Crop、TGVF 的 TP=2 judge，随后评分 Atomic。接管后
  无 OOM、Traceback 或失败 marker；可恢复 supervisor 修复已在执行分支提交并推送为
  `4fdd724`。本节点只报告运行进度，不提前报告 `@512` 分数。
- 2026-08-26 18:55 JST TGVF 推理闭合/评分接力节点：TGVF S64 的四个 rank 已完整写入
  `2,240/2,240` 条单图记录并通过 supervisor 行数 gate，`tgvf-s64-inference-complete`
  marker 已落盘；Crop S80 score 随即自动启动，Qwen2.5-72B TP=2 judge 从所给
  GPU0–3 池中实际选择 GPU2–3 加载。同期 Atomic S16 保持 GPU4–7 推理并持续写入。无失败
  marker；Crop nominal `@512` 与 TGVF true@512 的分数仍须等待各自 accepted summary，不能从
  未完成评分外推。
- 2026-08-26 19:01 JST Crop S80 nominal `@512` 历史评分节点：当时 summary 为
  `status=pass`、Macro* `61.5591`；后续确认该 full-model evaluator 同样缺失
  `</tool_call>` action boundary，因而该数字已作废，只作运行历史。
- 2026-08-26 19:16 JST TGVF S64 `@512` 正式闭合节点：七个 slice accepted summary 为
  `status=pass`，冻结 `extract_coredev_macro_star` 由相同 scorer artifact 提取 Macro*
  `55.4067`，比自身 matched@1M 的 `59.8086` 低 `4.4019 pp`。v2 plan 的 accepted summary
  不内嵌 `headline` 字段，但七个 slice、2,511 样本及 extractor 输入完整；这是 artifact schema
  差异，不是改用另一统计标准。Atomic S16 同期完成 `2,240/2,240` 条推理并自动进入评分。
- 2026-08-26 19:25 JST Atomic S16 `@512` 正式闭合：`7/7` slice、`2,511`
  条 official coverage、judge parse failure `0`；统一 extractor 得到 Macro* `57.2762`，
  比自身 matched@1M 低 `5.8065 pp`。
- 2026-08-28 03:05 JST Crop S80 fixed-boundary nominal `@512` 运行闭合：四个 rank 完整写入
  `2,240/2,240` 条受支持单图 trajectory，七项 accepted summary 为 `status=pass`、
  `sample_count=2,511`、`slice_count=7`、judge parse failure `0`，Macro* 为 `62.0967`。
  `2,014` 次 executed calls 与 successful observations 精确相等，same-turn mixed 为 `0`；
  runner、judge 和全部 GPU 随后正常退出。当前合同将该 boundary-fix V2 运行确认为有效
  true-512 结果；旧 boundary `61.5591` 已 supersede。
- 2026-08-28 中间 processor-override 判断（**superseded**）：曾将 `62.0967` 降级为
  processor-default。该中间判断不再是当前口径；当前只使用文首 true-512 表与项目级
  统一合同。Crop S80 true-1M/true-512 两臂 RNG namespace 不同，因此当前 `+2.4504 pp`
  差值仍只作 descriptive response。

### 5.5 Method-specific resolution response 与 policy-RL scale alignment

本节比较同 run/optimizer step 的 policy 在 true-1M 与 true-512 下的逐方法响应。TGVF S64 和
Atomic S16 的两种分辨率运行保持 frozen policy config SHA、RP67 state、prompt、tool contract、
task、seed namespace、common per-turn RNG 与 scorer 相同，因此是高度匹配的 resolution response。
但 Qwen snapshot 分别 materialize，tree/combined-weight byte SHA 不同，且 receipt 没有证明 tensor-semantic
equality。Crop S80 的两臂另外使用不同 RNG namespace，因此 Crop 差值只作描述性观察。

| Method | true-1M Macro* | true-512 Macro* | Δ (512 − 1M) | 解释范围 |
|---|---:|---:|---:|---|
| Crop S80 | 59.6463 | 62.0967 | **+2.4504 pp** | RNG namespace 不同；descriptive only |
| TGVF S64 | 59.8086 | 55.4067 | **−4.4019 pp** | highly matched；snapshot semantic equality 未闭合 |
| Atomic S16 | 63.0827 | 57.2762 | **−5.8065 pp** | highly matched；snapshot semantic equality 未闭合 |

TGVF 与 Atomic 在偏离 policy-RL 1M rollout 尺度后均回落。该证据 **consistent with**
TGVF-family 依赖 policy-RL train/eval scale alignment，但不是因果证明。RP67 adapter
pretraining 本身使用 `512²`，所以这里的 “alignment” 只指 policy rollout/evaluation
distribution。要确认因果，需要复用同一 snapshot 或提供 semantic tensor hash，并以
512-only 或 multi-scale TGVF/Atomic 训练 × 512/1M 评测形成完整 factorial 交叉。

Crop 在 true-512 四方表中取得最高 Macro*，且 VStar、HR 与 OCR mean 最高；其 Macro* 分别比
Original、TGVF 和 Atomic 高 `6.7411/6.6900/4.8205 pp`。这支持 Crop 系统在
constrained-pixel setting 下的 end-to-end effectiveness。由于方法间 prompt、tool contract
和 agent loop 不同，且 Crop 的两种分辨率运行未共享 RNG namespace，这一优势不能孤立归因给
crop tool，也不能解释为低分辨率对 Crop 的因果增益。

### 5.6 PRL26：fresh-S0 Train@512/Eval@512 与 TGVF prompt-axis 对照

历史 true-512 四方表只降低冻结 checkpoint 的评测像素上限，不能回答 policy 从第一步 RL rollout
开始就在 `262,144` pixels 下优化时会发生什么。PRL26 因此从同一 Original
Qwen3-VL-8B-Instruct S0 启动新的 Train@512 线路，并在相同像素上限下评测。第一阶段先闭合
No-Tool 与 corrected native Crop；第二阶段在释放 GPU/Ray 后比较 Pure TGVF Short 与
Target-guide-v2。Atomic fresh-S0@512 保持为独立 arm，不与当前 TGVF prompt-axis arm 混写；
它的 clean 自动接力现已挂起，并在前两阶段及完整 C/D paired evaluation 闭合后自行启动。

| 冻结项 | PRL26 合同 |
|---|---|
| Base / update | 同一 Original Qwen3-VL-8B-Instruct S0；full-model Qwen update |
| Pixel contract | train rollout 与正式 eval 均为 `max_pixels=262144` |
| Budget | 32 optimizer steps；global batch 16；每 prompt 16 rollouts；S0/S8/S16/S24/S32 保存 |
| Data / sampling | 同一 Teacher25 行序、seed42、temperature 1、constant LR `1e-6` |
| Reward | 沿用现有 answer/format/tool-penalty 合同；不因本轮对照改变 reward 权重 |
| Monitoring | 只在 S16/S24/S32 或异常时汇报；“稀疏”不表示训练监督、reward、数据或 sampling 稀疏化 |
| Crop boundary | 生成在 `</tool_call>` 处硬停并 include-stop；mixed answer-over-action fail-closed |

No-Tool 已完成 S32，permanent checkpoint receipt 的 `optimizer_step=32`，pair/project/run 三项
identity 均已落盘。Crop 已完成 S25；最近的 S24 永久 receipt 的 run/pair/project SHA256 分别为
`b064bf21674b6dea2d932c67e40f63e3aab7fad306245dd0bd4d92ad356d5d92`、
`f48c6e267df64c98e82898a9d6b551724fd1a86093d0894d5d8704e5c89607fb` 与
`c739c3437bc29c7a6a4d20d9bd0fa970d5fa13b1a89ee2689c8c6006b5d3b8f6`。S24 为 222/222 次调用
成功且 typed error 为空。S25 的 258 次尝试中 256 次成功，另有 2 次 execution failure。S23 的
4 次 incomplete 与 1 次 invalid JSON 被保留为稀疏解析失败，不据此宣称 action-boundary 回归。
这些数值只证明训练链路健康，不作为 CoreDev-2511 benchmark 结果，也不用于提前选择 checkpoint。

TGVF 的两个训练 arm 只允许一个 prompt 变量。Short 保留既有 user suffix、
`<think>...</think>`、plain-text final、Hermes parser、observation renderer、tool schema、最多 6 次
调用与 `</tool_call>` include-stop action boundary。Target-guide-v2 在此基础上只增加 Target 的
详细定义和 teacher-aligned 视觉描述案例，例如
`small circular gauge, its needle position, and surrounding scale markings`；它不新增逐轮 reasoning
规则、不收紧 final-only 格式，也不改 observation 文本。删除这一连续 guide block 后必须逐字恢复
Short prompt。两份 prompt bundle SHA256 分别为
`e74bb5e1253af107ff27badfcfaca747b94574e19677d22cfe42b0b1c0ba5633` 与
`77ed3a597d2a58e748b70bafe37882760944e293723a28008818a96aad025d0d`。

No-Tool/Crop 的 A/B evaluator 已常驻等待 Crop S32；两臂使用各自训练匹配的 prompt/agent contract、
相同 2,240 个支持 ID、七项 CoreDev-2511 scorer 与 @512 processor。两臂都是 temperature 1、
master seed42，但 seed namespace 不同，因此这是 aligned/matched end-to-end comparison，不能用来
估计逐样本 common-random-number paired effect。A/B 完成后，第二个 relay 先
并行运行 Short/Target-guide-v2 C0，再顺序训练 Short S32、Target-guide-v2 S32，最后执行正式 V6
`target_prompt_pair_v1`。V6 逐题逐轮 RNG 只投影实验变量 `prompt_sha256`，共享 seed protocol
SHA256 为 `4cbfd3cf698cb47b0c9594ca9f9e146ca09932d62bdb93d0877e59f9a85bee9c`。真实 processor 证明两臂
均表示 239,616 pixels、234 visual tokens；action boundary 均为 token `151645`，
`include_stop=true`。最终报告七项 Macro*、完整 sub-benchmark，以及每个 set 的工具使用题率、调用
总数/强度、成功 observation、错误、stop 分布与输出 token p50/p95/p99，并逐题核对 2,240 个 ID
和 RNG stream identity。

PRL26 的合同、自动接力与 164 项综合测试首先固定在
`origin/neurips-notool-rl-s32@efeaf1b48a18dc7481732712d326d2da8cdf2338`。运行前复核发现 A/B
成功标记由 `touch` 生成零字节 regular file，而 C/D relay 原先以 `-s` 等待非空文件，会在 A/B
成功后永久等待。修复 commit
`c0cebefd1ebc9ca2ddd91482a340f0d4b755e0b7` 将等待条件改为 `-f`，并增加回归断言；4 项 handoff
测试与 shell/diff 检查通过。旧等待型 tmux 在尚未产生任何 C/D 输出时以状态 130 退出，新 tmux
已在 clean `c0cebef` 上重启并通过 static contract audit。随后 S32 handoff 审计又识别出
receipt-to-resource-release race：No-Tool 的 receipt→trainer-exit 实测间隔为 17 秒，而旧 A/B
supervisor 没有 GPU/Ray admission gate。`bd31ac5e1ce299d44efbead30f553781b8f274fc` 增加至少两次
连续 all-GPU/Ray clean 探针，并通过 35 项直接 handoff 回归。等待中的 A/B 与 C/D 被受控停止后
分别在 pane `%179/%180` 重挂；Crop 与 Atomic 均未中断。
当前执行顺序严格为
`Crop S32 → No-Tool/Crop aligned Eval@512 → TGVF Short C0/S32 → Target-guide-v2 C0/S32 → paired
Eval@512 → Atomic C0/S32 → Atomic Eval@512`；训练与评测不会并发争用同一组 GPU。任何中间
reward、S16/S24 checkpoint 或单个 subset 分数都不得替代事前冻结的 S32/七项正式结果。

Atomic 运行实现固定于 `8e6b3d647d3a94c7768e3d8718b69d544010841e`，配置、自动接力、审计与
回归测试固定于 `e5e02879d1bec87779c59712330e01eb2b1a2d43`，均已 push 到
`origin/neurips-notool-rl-s32`。它从 Original Qwen S0 启动，使用 frozen RP67 S2000、既有 matched
Atomic prompt SHA256 `5efbd617f69ce9b3a6cb6b0c96bf7e24d8156b6e4dab9af55c9dfe5692c52e69`、
一个 `tgvf_crop_tool`、最多 6 次调用、corrected `</tool_call>` include-stop boundary，以及
train/eval `max_pixels=262144`。形式合同为 Teacher25 seed42、BS16×n16、world8、temperature 1、
constant LR `1e-6` 和 S0/S8/S16/S24/S32；reward 保持与其余 PRL26 parity arms 相同的
answer+protocol 口径，不追加 tool-utility reward。接力必须逐项验收 C/D 的 2,240 样本、七项覆盖、
paired RNG、result/summary hash 与 completion marker，不能仅凭 C/D checkpoint 启动。

## 6. Atomic 纳入正文的决策门槛

Atomic 进入正文核心方法必须同时满足：

1. target-only Macro* 和主要 Atomic favorable sub-benchmark 不发生足以推翻当前定位的崩塌；
2. target audit 的 all-pass rate 及逐标准结果可被透明报告；
3. 文章只声称观察到的任务条件优势，不声称未被单变量消融证明的 synergy；
4. Original、No-Tool、Crop、TGVF 和 Atomic 五列同时出现，且 MathVerse Vision Intensive 等
   负面切片保留。

如果任一条件不满足，Atomic 降级到 exploratory analysis / appendix。Pure TGVF + RP67 utility
仍作为主机制线，Crop 作为强基线。

**当前决策：性能门槛部分闭合，Atomic 仍维持 exploratory。** target-only Macro* 为
`60.8253`，比 true-1M Original 低 `0.4894 pp`、比自身 matched 低 `2.2574 pp`；其中 HR cross
从 `68.00%` 降至 `60.00%`，Relative Reflectance 从 `70.00%` 降至 `56.67%`，Vision Only
保持 `51.00%`。总体接近 raw-direct reference，却没有超过它；matched favorable regimes 也有
明显衰减，尚不足以升级为稳定核心方法。第 2 项正式 target audit 仍未完成，因此当前只保留
探索性正文/附录定位。

## 7. Claim–evidence–boundary 台账

| Claim | 当前证据 | 必须保留的边界 | 状态 |
|---|---|---|---|
| TGVF 改善一组 target-conditioned reasoning regimes | Relative Depth；MathVista arithmetic、word、numeric、visual QA；逐题案例 | 不是总体最优；OCR 和精细像素读取弱；BLINK 切片小 | 可写，待 CI |
| RP67 D 具有内容 utility 和 target specificity | correct−zero `+7.15 pp`；correct−wrong `+21.57 pp`；两者 95% CI 不跨零 | diagnostic semantic overlay；oracle target 不测自主工具选择 | 已支持 |
| Atomic matched prompt 下在跨图、反射率和 Vision Only 上显示优势 | matched HR cross、BLINK reflectance、MathVerse Vision Only | target-only 下 HR cross / reflectance 优势明显收窄，Vision Only 保持；target 合格率未闭合 | 探索性，核心门槛部分验证 |
| 广义 prompt bundle 下工具总体增益仍存在 | TGVF / Atomic stress-test Macro* 为 `58.5138 / 60.4684`，分别比 true-1M Original 低 `2.8009 / 0.8463 pp` | 相对各自 matched prompt 下降 `1.2949 / 2.6142 pp`；非 target-only；不是新 benchmark 泛化 | **总体 claim 不支持；保留分项稳健性** |
| 只补充 target 定义与案例的稳健性 | TGVF / Atomic target-only Macro* `58.1788 / 60.8253`，分别比 true-1M Original 低 `3.1359 / 0.4894 pp` | 相对 own matched 分别下降 `1.6298 / 2.2574 pp`；不支持“详细 target 定义普遍增益” | 已支持 benchmark-specific sensitivity，不支持总体增益 |
| 工具方法整体优于 raw direct Original | true-1M 表中仅 Atomic 高于 Original `1.7680 pp`；Crop/TGVF 分别低 `0.2442/1.5061 pp`。true-512 表中三者均高于 Original | Original 是 raw-direct，其余三行使用方法特定 prompt/tool/agent contract；不能把差值孤立归因给工具 | **整体 claim 不支持；仅作分辨率内端到端描述** |
| 工具方法的总体增益超越 RL-only/no-tool control | Corrected true-1M frozen No-Tool S32 为 `63.7520`，高于 Atomic/Crop S32/TGVF `0.6693/2.6814/3.9434 pp`；S32−S0 为 `−0.0098 pp` | 方法间仍跨 prompt、工具 schema 与 agent loop；这是端到端 control，不是纯工具开关消融 | **总体 claim 不支持；转向能力切片与机制证据** |
| Crop S80 在 1M→512 下的 Macro* 上升 | 有效 true-1M/true-512 为 `59.6463/62.0967`，差 `+2.4504 pp` | 两臂 RNG namespace 不同；只是 descriptive response，不是低分辨率因果增益 | **已观测，不作因果宣称** |
| TGVF-family 依赖 policy-RL train/eval scale alignment | TGVF/Atomic highly matched 512−1M 为 `−4.4019/−5.8065 pp` | RP67 adapter 预训练为 512²；snapshot semantic equality 未闭合；需复用同 snapshot 或 semantic tensor hash，再做 train-cap × eval-cap factorial | **consistent with，非因果证明** |
| 三种方法形成不同工具调用行为 | Crop S32/S80 true-1M 成功调用率 `63.53/89.55%`，实际调用 `1,769/2,043`；TGVF/Atomic 另表 | 描述性统计；调用更多不等于 utility 更高；policy 自选择混杂 | 已支持 |

## 8. 论文实验部分建议结构

1. **Unified true-1M stage-golden comparison.** 先报告 Original、frozen No-Tool S32、Crop S32、
   TGVF S64 与 Atomic S16 的统一 `1,003,520` 表与完整 sub-benchmark；明确它是
   prompt/agent contract 不同的端到端比较。
2. **How much does RL itself add?** 报告 No-Tool S0/S8/S16/S32 corrected true-1M 动态；S32−S0
   `−0.0098 pp` 为 aggregate-flat，旧 processor-default `+2.2141 pp` 只作历史线索。
3. **Where target-conditioned evidence remains useful.** 在总体 claim 收缩后，用完整
   sub-benchmark 图定位 TGVF/Atomic 的关系、深度、视觉数学与 OCR 条件优势，同时保留
   No-Tool 控制列和负面切片。
4. **How do the policies use their tools?** 报告整体及逐 set 的调用覆盖率、调用强度、重复调用和
   无效尝试，并与正确率分析分开。
5. **Does RP67 carry target-specific answer utility?** 867 样本 correct/zero/wrong 三臂与
   paired effect。
6. **Robustness to target specification.** 以 target-only matched arm 为主验证；广义 full
   prompt 只作 supplementary stress test。
7. **Exploratory Atomic Crop+TGVF.** target-only 与 target audit 闭合后再决定正文或附录层级。
8. **Method-specific resolution response.** 报告 TGVF/Atomic 的 highly matched 回落与 Crop 的 descriptive 上升，
   将 scale alignment 写为待 factorial 证实的解释。
9. **Qualitative mechanisms and failures.** 复用已有真实 trajectory 与 bbox 案例，但把机制语言
   限定为 behavior-level inference。

## 9. 当前推进顺序

1. [已完成] RP67 semantic overlay 和 CI，机制主张已锁定；
2. [已完成] 广义 full-prompt stress test 及七项官方评分；
3. [已完成] Crop S32/S80 true-1M 与 TGVF/Atomic matched-prompt 整体、逐 set 调用率、
   调用次数和错误类型审计；
4. [已完成] TGVF S64、Atomic S16 target-only matched prompt 推理与七套官方评分；
5. [已完成] No-Tool RL 训练与 corrected true-1M V2 S0/S8/S16/S32 推理、评分、processor/grid
   proof 和 aggregate receipt 已闭合；旧 matched 评测仅保留 processor-default historical；
6. [已完成] Crop S80、TGVF S64、Atomic S16 的 true-512 结果与四方表已闭合；
7. [已完成] Crop S32/S80 true-1M 推理、评分、processor/grid audit 与 receipt 已闭合；
8. [已完成] Original raw-direct true-1M 2,511 行正式评分、七项 headline、completion receipt
   与 59-slice 回填均已闭合；
9. [已完成] Appendix A 加入 frozen No-Tool S32 后机械重算 59-slice winner 与 pairwise W/L/T；
10. [已完成] 从 matched/target-only inference JSONL 物化正式 Atomic blind audit pack，
    review view、图片哈希、分层 quota 与 source split 均已复核；
11. [部分完成] target-only 调用行为对照已回填；正式 audit 仍待双人盲标、第三人裁决、
    Wilson CI 与 agreement；
12. [已完成] PRL26 No-Tool fresh-S0 Train@512 到 S32，permanent receipt 已审计；
13. [运行中] PRL26 Crop fresh-S0 Train@512 已完成 S25 并继续向 S32；最近的 S24 permanent
    checkpoint 已验收，S25 为 256/258 次成功调用、2 次 execution failure；
14. [已排队] Crop S32 后自动执行 No-Tool/Crop matched Eval@512；
15. [已排队] A/B 评测闭合后自动执行 TGVF Short 与 Target-guide-v2 的 C0、两个独立 S32 训练
    和 prompt-axis paired Eval@512；
16. [已排队] Atomic fresh-S0 Train@512/Eval@512 clean 接力已通过 static admission 并挂起；
    完整 C/D paired evaluation 闭合后自动执行 C0、S32 与正式七项 Eval@512；
17. [待写作] 形成英文 Experiments/Discussion 初稿；
18. [明确不做] Crop seed43。

## 10. 证据来源

- Original 定义和 Macro* 合同：
  `docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md`
- 80-step 三线路数值与 checkpoint 选择：
  `docs/PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md`
- 三方法逐题轨迹与失败案例：
  `docs/PRL25_CROP_TGVF_ATOMIC_QUALITATIVE_CASE_ANALYSIS_20260825.md`
- 工具调用行为：Crop S32/S80 true-1M 及 TGVF S64/Atomic S16 的
  `inference/rank-0..3.jsonl` 中 `tool_calls`、`tool_errors` 与
  `successful_observation_count`；各臂均为 2,240 个唯一共同 ID。
- Crop S32 true-1M summary：
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP32-TRUE1M-RESOLUTION-RNG-EXTENSION-V1/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。
- Crop S80 true-1M summary：
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TRUE1M-TRUE512-RESOLUTION-PAIR-V1/pixel1003520/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。
  两个 true-1M receipt identity 与 paired-summary SHA 见文首审计键。
- TGVF S64 / Atomic S16：对应 six-point evaluation 的 `step64` / `step16` 官方 summary 与
  `paired-summary.json`。
- Original raw-direct true-1M：
  `artifacts/evaluation/PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1/scoring/coredev-2511-eval-summary.json`；
  summary SHA256 `f8dc31b5353c36d2e764096ee2f2a1f0da0ca3d28fb4525c2fb660829c705904`。
  同一 evaluation 根目录下的
  `runtime/scoring-supervisor/original-true1m-scoring-complete.json` 固定 Macro* `61.3147`、
  `max_pixels=1,003,520`、2,511 rows 与 7 slices；judge parse failure 为 0。
- No-Tool RL S0/S8/S16/S32 processor-default historical summary：
  `artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1/matched/step{0,8,16,32}/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
  28 个精确评分来源由同级各 dataset 的 `pinned-reuse-receipt.json` 固定；这些分数不进入
  true-1M 主表。
- No-Tool RL corrected true-1M V2 aggregate receipt：
  `artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2/runtime/scoring-supervisor/matched-scoring-complete.json`；
  receipt identity `69328dfe889366650e96a0582eaa86a27df1ac349751ee40493d83f47c92a955`，
  文件 SHA256 `0fc8bcf3865c4d6ee360206da318c7913afb8c4cf659ea4bcb29104ad132db59`。
- `@512` 执行计划：
  `configs/evaluation/prl25_{b_crop_exact_step80,c_frozen_rp67_tfree_teacher25_s64_matched,d_atomic_crop_tgvf_s16_matched}_pixel512_coredev2511_plan.json`；
  并行、可恢复 supervisor 为 `tools/supervise_prl25_bcd_selected_pixel512_evaluation.sh`。
- Crop S80 true-512 fixed-boundary V2 计划：
  `configs/evaluation/prl25_b_crop_exact_step80_pixel512_tool_boundary_fix_v2_coredev2511_plan.json`，
  现有 `neurips-notool-rl-s32` 分支 commit `eb37ad9`；accepted summary 与总合同分别为
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-PIXEL512-TEMP1-SEED42-TOOL-BOUNDARY-FIX-V2/step80/scoring/coredev-official-v1/coredev-2511-eval-summary.json`
  和同级 evaluation 根目录的 `paired-summary.json` / `evaluation-complete`。该 artifact 为
  当前有效 true-512 Macro* `62.0967`；它与 S80 true-1M 运行不共享 RNG namespace，所以
  分辨率差值只作 descriptive。
- No-Tool S32 raw-direct@512 accepted summary：
  `artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-RAW-DIRECT-512-S32-V1/scoring/coredev-2511-eval-summary.json`。
- PRL26 No-Tool / Crop Train@512 roots：
  `artifacts/policy/PRL-26-{A-train512-s32-parity-notool,B-train512-s32-parity-crop}-qwen3-instruct-bs16-n16-teacher25-ws8`；
  permanent receipt 分别见 `global_step_32` 与当前已验收的 `global_step_24`。
- PRL26 No-Tool/Crop Eval@512 输出根：
  `artifacts/evaluation/PRL26-TRAIN512-S32-PIXEL512-COREDEV2511-V1`；常驻 tmux session 为
  `prl26-train512-s32-eval`。S32 receipt 后的双稳定 GPU/Ray release gate 固定于
  `origin/neurips-notool-rl-s32@bd31ac5e1ce299d44efbead30f553781b8f274fc`；通过后的探针将写入
  上述输出根的 `runtime/training-resource-probe-*.json` 与 `training-resources-released.json`。
- PRL26 TGVF prompt-axis V6 plan：
  `configs/evaluation/prl26_cd_tgvf_target_prompt_pair_s32_pixel512_coredev2511_plan.json`，固定于
  `origin/neurips-notool-rl-s32@c0cebefd1ebc9ca2ddd91482a340f0d4b755e0b7`；控制根为
  `artifacts/control/PRL-26-tgvf-prompt-parity-20260829`，常驻 tmux session 为
  `prl26-cd-tgvf-prompt-s32`。
- PRL26 Atomic Train@512/Eval@512：运行实现 commit `8e6b3d647d3a94c7768e3d8718b69d544010841e`，
  接力与审计 commit `e5e02879d1bec87779c59712330e01eb2b1a2d43`；控制根为
  `artifacts/control/PRL-26-E-atomic-train512-s32-20260829`，常驻 tmux session 为
  `prl26-e-atomic-train512-s32`。正式合同另记于 `docs/EXPERIMENT_LEDGER.md` 的
  “PRL26-E train@512 Atomic Crop+TGVF parity rerun”。

注意：主仓当前 qualitative 文档可能尚未进入本 worktree 的提交历史；本文只把它作为只读证据
来源，不覆盖主仓未提交内容。

## Appendix A. Unified true-1M aligned sub-benchmark appendix

这里的 aligned 严格限定为六个已完成方法共同支持的 2,240 个 single-image items：VStar 191、HRBench 200、BLINK 180、OCRBench 600、MMMU 269、MathVista 300、MathVerse 500。所有 subgroup 都只从这组共同 support IDs 统计；full scorer 中的 unsupported multi-image 零分占位项一律排除。Original 使用正式 raw-direct true-1M 结果，No-Tool 使用事前冻结的 S32；没有混入任何 @512 数值。除特别说明外，数值单位均为百分比。

| Method | Macro* | VStar | HR | BLINK-180 | OCR EN/CN mean | MMMU-269 | MathVista | MathVerse-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Crop S32 | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| Crop S80 | 59.6463 | 73.8220 | 69.5000 | 59.4444 | 54.0426 | 44.9814 | 63.3333 | 52.4000 |
| TGVF S64 | 59.8086 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | 73.5000 | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |
| Original true-1M | 61.3147 | 72.7749 | 66.5000 | 63.8889 | 59.7877 | 37.9182 | 75.3333 | 53.0000 |
| NoTool S32 | 63.7520 | 70.1571 | 65.5000 | 71.1111 | 49.3928 | 53.9033 | 75.0000 | 61.2000 |

以下胜出统计覆盖共同 2,240 supported items 上的 59 个 aligned slices。含并列胜出会给每个并列方法各记一次，因此该列不要求合计为 59；并列按比例分摊列的合计为 59。六种方法全部参加统计；按表中四位小数判断并列，并已机械重算每行 winner、胜出计数与 pairwise W/L/T。

| Method | 含并列胜出 | 独占胜出 | 并列按比例分摊 |
|---|---:|---:|---:|
| Crop S32 | 9 | 4 | 5.5667 |
| Crop S80 | 10 | 2 | 4.9333 |
| TGVF S64 | 13 | 5 | 7.4667 |
| Atomic S16 | 17 | 4 | 9.1333 |
| Original | 17 | 7 | 10.7667 |
| NoTool S32 | 32 | 14 | 21.1333 |

### VStar-2

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direct_attributes | 115 | 75.6522 | 73.9130 | 70.4348 | 69.5652 | 71.3043 | 70.4348 | Crop S32 |
| relative_position | 76 | 69.7368 | 73.6842 | 80.2632 | 75.0000 | 75.0000 | 69.7368 | TGVF S64 |

### HR-2

这里采用官方 scorer 的 `cycle=Average`；single 与 cross 各 100 个样本。此前附录将部分
per-cycle 单元格误当成 aggregate，现已按六份 accepted 200-row result TSV 统一修正。

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single | 100 | 77.0000 | 79.0000 | 69.0000 | 79.0000 | 65.0000 | 68.0000 | Crop S80 / Atomic S16 |
| cross | 100 | 59.0000 | 60.0000 | 64.0000 | 68.0000 | 68.0000 | 63.0000 | Atomic S16 / Original |

### BLINK-6

这是统一合同中的六个 single-image 子集，各 30 个样本；不使用完整 scorer 中 unsupported multi-image 的零分占位项。

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Counting | 30 | 73.3333 | 60.0000 | 73.3333 | 73.3333 | 66.6667 | 76.6667 | NoTool S32 |
| IQ Test | 30 | 23.3333 | 33.3333 | 23.3333 | 10.0000 | 20.0000 | 33.3333 | Crop S80 / NoTool S32 |
| Object Localization | 30 | 60.0000 | 53.3333 | 70.0000 | 73.3333 | 60.0000 | 73.3333 | Atomic S16 / NoTool S32 |
| Relative Depth | 30 | 73.3333 | 83.3333 | 86.6667 | 80.0000 | 83.3333 | 80.0000 | TGVF S64 |
| Relative Reflectance | 30 | 63.3333 | 40.0000 | 46.6667 | 70.0000 | 56.6667 | 66.6667 | Atomic S16 |
| Spatial Relation | 30 | 86.6667 | 86.6667 | 93.3333 | 90.0000 | 96.6667 | 96.6667 | Original / NoTool S32 |

### MathVista-12

MathVista skill 标签可重叠，因此各行 n 不能相加作为总样本数；总样本仍为 300。

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| algebraic reasoning | 75 | 70.6667 | 64.0000 | 74.6667 | 76.0000 | 81.3333 | 80.0000 | Original |
| arithmetic reasoning | 104 | 65.3846 | 54.8077 | 72.1154 | 63.4615 | 71.1538 | 72.1154 | TGVF S64 / NoTool S32 |
| figure question answering | 96 | 66.6667 | 67.7083 | 69.7917 | 69.7917 | 73.9583 | 73.9583 | Original / NoTool S32 |
| geometry problem solving | 49 | 77.5510 | 73.4694 | 85.7143 | 87.7551 | 85.7143 | 87.7551 | Atomic S16 / NoTool S32 |
| geometry reasoning | 63 | 71.4286 | 66.6667 | 79.3651 | 79.3651 | 82.5397 | 84.1270 | NoTool S32 |
| logical reasoning | 12 | 25.0000 | 33.3333 | 16.6667 | 25.0000 | 16.6667 | 33.3333 | Crop S80 / NoTool S32 |
| math word problem | 63 | 76.1905 | 60.3175 | 84.1270 | 79.3651 | 82.5397 | 85.7143 | NoTool S32 |
| numeric commonsense | 36 | 55.5556 | 44.4444 | 58.3333 | 50.0000 | 50.0000 | 47.2222 | TGVF S64 |
| scientific reasoning | 37 | 56.7568 | 56.7568 | 54.0541 | 56.7568 | 64.8649 | 64.8649 | Original / NoTool S32 |
| statistical reasoning | 111 | 76.5766 | 77.4775 | 81.9820 | 81.0811 | 85.5856 | 83.7838 | Original |
| textbook question answering | 50 | 58.0000 | 54.0000 | 56.0000 | 56.0000 | 70.0000 | 66.0000 | Original |
| visual question answering | 42 | 61.9048 | 57.1429 | 64.2857 | 50.0000 | 61.9048 | 57.1429 | TGVF S64 |

### MathVerse-5

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Text Dominant | 100 | 71.0000 | 67.0000 | 64.0000 | 66.0000 | 70.0000 | 70.0000 | Crop S32 |
| Text Lite | 100 | 57.0000 | 57.0000 | 58.0000 | 59.0000 | 50.0000 | 61.0000 | NoTool S32 |
| Vision Dominant | 100 | 51.0000 | 49.0000 | 46.0000 | 50.0000 | 54.0000 | 60.0000 | NoTool S32 |
| Vision Intensive | 100 | 47.0000 | 48.0000 | 42.0000 | 49.0000 | 57.0000 | 60.0000 | NoTool S32 |
| Vision Only | 100 | 45.0000 | 41.0000 | 42.0000 | 51.0000 | 34.0000 | 55.0000 | NoTool S32 |

### OCR EN/CN

| Slice | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| English Overall | 400 | 51.5033 | 51.4573 | 47.2443 | 53.7774 | 59.0441 | 46.2998 | Original |
| Chinese Overall | 200 | 56.2175 | 56.6280 | 41.8449 | 54.7665 | 60.5313 | 52.4859 | Original |

### MMMU subject 全表

下表从六份 accepted MMMU coverage-result TSV 取共同的 269 个 `coverage=single_image_evaluated` sample IDs，再按 subject 对 hit 求均值。31 个 unsupported/excluded multi-image 零分占位项已完全排除，因此 subject n 为 5–10，总和为 269；这与 headline MMMU-269 使用完全相同的支持集合。

| Subject | n | Crop S32 | Crop S80 | TGVF S64 | Atomic S16 | Original | NoTool S32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Accounting | 10 | 50.0000 | 30.0000 | 60.0000 | 60.0000 | 20.0000 | 60.0000 | TGVF S64 / Atomic S16 / NoTool S32 |
| Agriculture | 10 | 30.0000 | 30.0000 | 40.0000 | 30.0000 | 30.0000 | 30.0000 | TGVF S64 |
| Architecture and Engineering | 10 | 20.0000 | 50.0000 | 50.0000 | 60.0000 | 40.0000 | 60.0000 | Atomic S16 / NoTool S32 |
| Art | 10 | 20.0000 | 30.0000 | 30.0000 | 30.0000 | 30.0000 | 30.0000 | Crop S80 / TGVF S64 / Atomic S16 / Original / NoTool S32 |
| Art Theory | 7 | 42.8571 | 57.1429 | 57.1429 | 57.1429 | 57.1429 | 57.1429 | Crop S80 / TGVF S64 / Atomic S16 / Original / NoTool S32 |
| Basic Medical Science | 10 | 30.0000 | 30.0000 | 20.0000 | 50.0000 | 20.0000 | 40.0000 | Atomic S16 |
| Biology | 7 | 28.5714 | 28.5714 | 0.0000 | 14.2857 | 14.2857 | 28.5714 | Crop S32 / Crop S80 / NoTool S32 |
| Chemistry | 5 | 60.0000 | 60.0000 | 60.0000 | 60.0000 | 20.0000 | 60.0000 | Crop S32 / Crop S80 / TGVF S64 / Atomic S16 / NoTool S32 |
| Clinical Medicine | 9 | 33.3333 | 22.2222 | 22.2222 | 22.2222 | 11.1111 | 44.4444 | NoTool S32 |
| Computer Science | 10 | 50.0000 | 50.0000 | 40.0000 | 50.0000 | 70.0000 | 50.0000 | Original |
| Design | 10 | 60.0000 | 50.0000 | 70.0000 | 60.0000 | 70.0000 | 70.0000 | TGVF S64 / Original / NoTool S32 |
| Diagnostics and Laboratory Medicine | 10 | 0.0000 | 20.0000 | 10.0000 | 20.0000 | 20.0000 | 40.0000 | NoTool S32 |
| Economics | 8 | 50.0000 | 37.5000 | 25.0000 | 37.5000 | 50.0000 | 50.0000 | Crop S32 / Original / NoTool S32 |
| Electronics | 10 | 60.0000 | 80.0000 | 50.0000 | 70.0000 | 30.0000 | 60.0000 | Crop S80 |
| Energy and Power | 10 | 30.0000 | 30.0000 | 30.0000 | 30.0000 | 50.0000 | 60.0000 | NoTool S32 |
| Finance | 10 | 60.0000 | 50.0000 | 70.0000 | 70.0000 | 10.0000 | 60.0000 | TGVF S64 / Atomic S16 |
| Geography | 8 | 75.0000 | 37.5000 | 25.0000 | 37.5000 | 50.0000 | 25.0000 | Crop S32 |
| History | 8 | 62.5000 | 62.5000 | 62.5000 | 62.5000 | 50.0000 | 75.0000 | NoTool S32 |
| Literature | 8 | 75.0000 | 62.5000 | 75.0000 | 75.0000 | 75.0000 | 75.0000 | Crop S32 / TGVF S64 / Atomic S16 / Original / NoTool S32 |
| Manage | 10 | 40.0000 | 40.0000 | 40.0000 | 50.0000 | 50.0000 | 70.0000 | NoTool S32 |
| Marketing | 9 | 66.6667 | 66.6667 | 88.8889 | 100.0000 | 44.4444 | 77.7778 | Atomic S16 |
| Materials | 8 | 37.5000 | 25.0000 | 50.0000 | 75.0000 | 12.5000 | 62.5000 | Atomic S16 |
| Math | 10 | 30.0000 | 30.0000 | 40.0000 | 40.0000 | 60.0000 | 60.0000 | Original / NoTool S32 |
| Mechanical Engineering | 10 | 40.0000 | 40.0000 | 40.0000 | 40.0000 | 0.0000 | 50.0000 | NoTool S32 |
| Music | 7 | 42.8571 | 57.1429 | 42.8571 | 42.8571 | 42.8571 | 42.8571 | Crop S80 |
| Pharmacy | 7 | 71.4286 | 42.8571 | 57.1429 | 71.4286 | 14.2857 | 42.8571 | Crop S32 / Atomic S16 |
| Physics | 10 | 80.0000 | 60.0000 | 40.0000 | 60.0000 | 30.0000 | 40.0000 | Crop S32 |
| Psychology | 8 | 50.0000 | 62.5000 | 25.0000 | 62.5000 | 50.0000 | 37.5000 | Crop S80 / Atomic S16 |
| Public Health | 10 | 70.0000 | 80.0000 | 80.0000 | 60.0000 | 40.0000 | 100.0000 | NoTool S32 |
| Sociology | 10 | 50.0000 | 40.0000 | 50.0000 | 50.0000 | 70.0000 | 50.0000 | Original |

### 六方法 pairwise W/L/T 与能力边界

W/L/T 均按左侧方法视角，以表中四位小数判断相等：

| Left method | Right method | W | L | T |
|---|---|---:|---:|---:|
| Crop S32 | Crop S80 | 27 | 18 | 14 |
| Crop S32 | TGVF S64 | 22 | 27 | 10 |
| Crop S32 | Atomic S16 | 15 | 30 | 14 |
| Crop S32 | Original | 25 | 27 | 7 |
| Crop S32 | NoTool S32 | 11 | 37 | 11 |
| Crop S80 | TGVF S64 | 20 | 29 | 10 |
| Crop S80 | Atomic S16 | 11 | 32 | 16 |
| Crop S80 | Original | 23 | 31 | 5 |
| Crop S80 | NoTool S32 | 11 | 38 | 10 |
| TGVF S64 | Atomic S16 | 13 | 29 | 17 |
| TGVF S64 | Original | 26 | 25 | 8 |
| TGVF S64 | NoTool S32 | 11 | 36 | 12 |
| Atomic S16 | Original | 23 | 25 | 11 |
| Atomic S16 | NoTool S32 | 16 | 30 | 13 |
| Original | NoTool S32 | 15 | 32 | 12 |

- NoTool S32 的 `32` 个含并列胜出 slice 由 MMMU 小 subject（17）、MathVista 重叠标签（7）、
  MathVerse（4）、BLINK（4）共同构成；其中独占胜出为 14。它没有在 VStar、HR 或 OCR slice
  上胜出，不能解释为 32 个独立显著优势。
- Crop S32 相对 TGVF / Atomic / Original / NoTool 分别为 `22/27/10`、`15/30/14`、
  `25/27/7`、`11/37/11`；相应 Macro* 差值为 `+1.2619 / −2.0121 / −0.2442 / −2.6814 pp`。
- Crop 较稳定的有利方向是局部视觉读取与文本主导任务：相对 TGVF，OCR CN 高
  `14.3725 pp`、OCR EN 高 `4.2589 pp`、MathVerse Text Dominant 高 `7.0000 pp`；VStar
  direct attributes 相对 TGVF/Atomic 分别高 `5.2174/6.0870 pp`。
- Crop 的不利方向集中于几何/统计推理、相对位置和跨图交互：相对 TGVF，VStar relative
  position 低 `10.5263 pp`、BLINK Relative Depth 低 `13.3333 pp`；MathVista 仅 3/12
  slices 严格优于 TGVF。
- 因而可以将 Crop 的优势定位为部分局部视觉读取、文本主导和属性识别，但不能表述为全面优于
  TGVF、Atomic、Original 或 NoTool；尤其 Original 在 OCR EN/CN 均领先。
- MMMU subject 只有 n=5–10，MathVista skill 标签彼此重叠；59-slice winner 与 pairwise
  W/L/T 都是能力定位用的描述性统计，不等同于 59 个独立检验，也不替代 headline Macro*。

### Accepted scorer 数据源与 SHA256

- Crop S32 summary:
  artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP32-TRUE1M-RESOLUTION-RNG-EXTENSION-V1/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json
  SHA256: f8ac95e39873ea03bf1ea822e096178d92a058a0c678a1b17508f31419971cbd
- Crop S80 summary:
  artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TRUE1M-TRUE512-RESOLUTION-PAIR-V1/pixel1003520/scoring/coredev-official-v1/coredev-2511-eval-summary.json
  SHA256: 530a23ba1468e9c8b31bdfe83b9085bf49aaf02ee375182b2d69d5b66458b811
- TGVF S64 summary:
  artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1/step64/scoring/coredev-official-v1/coredev-2511-eval-summary.json
  SHA256: 0e1986ca84eb7102c21dc58750175d74b943930514626f592d9edccf15ce2720
- Atomic S16 summary:
  artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1/step16/scoring/coredev-official-v1/coredev-2511-eval-summary.json
  SHA256: 9ead00a25cd123a12b6bc5dff5b9067714716d3c31c301216a4ba531023501f2
- Original raw-direct true-1M summary:
  artifacts/evaluation/PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1/scoring/coredev-2511-eval-summary.json
  SHA256: f8dc31b5353c36d2e764096ee2f2a1f0da0ca3d28fb4525c2fb660829c705904
- NoTool S32 true-1M summary:
  artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2/matched/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json
  SHA256: bd6057d798c37791714033c4f9a734f435264ec1242fc841fdaad4fd064d897a

六份 summary 均满足 status=pass、sample_count=2511、slice_count=7，并固定于 VLMEvalKit commit 7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f。2511 是 scorer materialization 总行数；本附录的 aligned 分母只使用六方法共同的 2240 supported items。Original 的 judge parse failure 为 0；NoTool S32 为 2，均位于 VStar，并已按 deterministic-incorrect 规则纳入 accepted 分数。对 MMMU，六份 coverage-result TSV 都恰为 269 个 single_image_evaluated 加 31 个 unsupported/excluded multi-image，且六份 269-ID 集合完全相同；本附录只读取前者的 hit。

NoTool true-1M V2 aggregate completion receipt 为
`artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2/runtime/scoring-supervisor/matched-scoring-complete.json`，文件 SHA256 为 `0fc8bcf3865c4d6ee360206da318c7913afb8c4cf659ea4bcb29104ad132db59`，内嵌 identity 为 `69328dfe889366650e96a0582eaa86a27df1ac349751ee40493d83f47c92a955`。S32 单步 receipt SHA256 为 `ba48a5742509238dc46e21bb6cae30e3cf68b8d79bfd7cb2b837ffbbb5a3717a`。

Original scoring completion receipt 为
`artifacts/evaluation/PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1/runtime/scoring-supervisor/original-true1m-scoring-complete.json`，文件 SHA256 为 `8f5881aa227ec350f4839c9a46779c20ecc64de2092e9012345717387a39a366`；其内嵌 `summary_sha256` 与上列 accepted summary SHA 完全一致，并固定 `max_pixels=1,003,520`、2,511 rows 与 7 slices。

Crop true-1M audit receipt:

- Crop S32 receipt SHA256: 40a82f79b73eb58ba078f35a0e99e6e6d936677ccc587af72cccfb1ddea172bf
- Crop S80 receipt SHA256: ed476beb74113421a8bdab171d5f22a518cbcfd197ea7eb752eabaaaab80b341

两份 receipt 均为 status=accepted，记录 evaluation_image_max_pixels=1,003,520、maximum_observed_visual_token_count=980、maximum_observed_represented_pixel_area=1,003,520，以及 2240 个受支持的 single-image 推理结果。

MMMU-269 过滤后的总命中数为 Crop S32 125/269、Crop S80 121/269、TGVF S64 121/269、Atomic S16 138/269、Original 102/269、NoTool S32 145/269，分别精确复现 headline 的 46.4684、44.9814、44.9814、51.3011、37.9182、53.9033。

按 sample_id 字典序排序、以 LF 分隔并保留末尾 LF 后，共同 269-ID 集合的 SHA256 为 3b7beed9691569723c023caa1d8080957f20a49bec69d6e792467bfcc95ba38f。六份 accepted MMMU coverage-result TSV 的 SHA256 依次为：

- Crop S32: a0a36780118e5f03b9107ea89cb980f9632ef02a86f1bedb07ca6b0a61289528
- Crop S80: 47b8ba4e5467270b89bf0d20aab2c603c11c5dc28e3fe6d44cdcf1d318e17a51
- TGVF S64: 3cbd0731bd2b59db1fcdc3e62d53e2e006b1a141fecaffca57d3ed5fdfa22323
- Atomic S16: 9ae8907f93ebef755c03a604d671778eafa5c0d558094efa87da001bbedd3f1d
- Original: 08ea4d45f9e1d7a561bb4dc0552ce26e399cfeadc5764de9550d94cd9beb2271
- NoTool S32: e362ce65dde083c0828a99320909d294482c5400adc7133b4bf493c9fbdf9822

六方法共同的 2,240 supported-ID 集合 SHA256 为
`1754796867842d7ae78ee8c0616b4a035efc5fe853df8333bb24eefb3a42bb85`。NoTool S32 的
HR accepted result TSV SHA256 为 `7ade24ffa765b5da8517933d8c522a80516f99b552241efe579669050b14bae7`。

### 最小复核方式

~~~bash
sha256sum "$SUMMARY"
jq '{
  status,
  sample_count,
  slice_count,
  vlmevalkit_commit,
  headline,
  slices: [.slices[] | {dataset, sample_count, metrics}]
}' "$SUMMARY"

jq '{
  status,
  arm,
  rows,
  processor_proof
}' "$EVAL_ROOT/runtime/true1m-audit-receipt.json"
~~~

MMMU subject 的共同支持集合和分数可直接从六份 accepted coverage-result TSV 复核：

~~~bash
python - "$CROP_S32_MMMU_TSV" "$CROP_S80_MMMU_TSV" \
  "$TGVF_S64_MMMU_TSV" "$ATOMIC_S16_MMMU_TSV" \
  "$ORIGINAL_MMMU_TSV" "$NOTOOL_S32_MMMU_TSV" <<'PY'
import json
import sys

import pandas as pd

names = [
    "Crop S32",
    "Crop S80",
    "TGVF S64",
    "Atomic S16",
    "Original",
    "NoTool S32",
]
frames = {
    name: pd.read_csv(path, sep="\t")
    for name, path in zip(names, sys.argv[1:], strict=True)
}
supported = {
    name: set(
        frame.loc[
            frame["extra_records"].map(
                lambda value: json.loads(value)["coverage"]
                == "single_image_evaluated"
            ),
            "sample_id",
        ]
    )
    for name, frame in frames.items()
}
common = set.intersection(*supported.values())
assert len(common) == 269
assert all(ids == common for ids in supported.values())

for subject in sorted(frames[names[0]]["subject"].unique()):
    values = {}
    counts = set()
    for name, frame in frames.items():
        selected = frame[
            frame["sample_id"].isin(common)
            & frame["subject"].eq(subject)
        ]
        counts.add(len(selected))
        values[name] = 100.0 * selected["hit"].astype(float).mean()
    assert len(counts) == 1
    print(subject, counts.pop(), values)
PY
~~~
