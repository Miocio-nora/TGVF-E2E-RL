# NeurIPS Workshop：TGVF 文章实验计划、推进台账与阶段报告

更新时间：2026-08-28（Asia/Tokyo）

> **Crop 双重勘误（2026-08-28）：action boundary 已修复，但此前宣称的 Crop
> 1M→512 像素控制无效。** fixed-boundary S32/S80 的 Macro* 为 `61.6699 / 59.1785`；另一次
> nominal `@512` 运行的 Macro* 为 `62.0967`，三臂 same-turn mixed 均为 `0`。然而处理器审计
> 证明后两次 S80 运行的 initial visual token counts 完全相同，均实际使用 fast processor 默认
> `size.longest_edge=16,777,216`，而不是 `1,003,520 / 262,144`。因此两项数值只保留为
> native-default 历史运行，`+2.9182 pp` 不具有像素效应含义。Crop true@1M/true@512 正在等待
> corrected common-RNG rerun。旧 boundary 的 nominal S80@512 `61.5591` 仍作废；Pure TGVF
> 与 Atomic 不受 Crop action-boundary 或该 Crop processor-override 缺陷影响。详见第 0 与 5.5 节。

状态：**实验进行中；RP67 三臂验证、广义 full-prompt stress test、严格 target-only matched
prompt 与 No-Tool RL S0/S8/S16/S32 matched no-tool 评测均已闭合。事前冻结的 No-Tool S32
Macro* 为 `66.6853`，同协议 S0 为 `64.4712`，即 32-step RL 增益为 `+2.2141 pp`；S32 同时
高于 Crop S80 native-default 历史运行、TGVF S64 和 Atomic S16。该结果不支持“工具方法的总体增益超越当前 RL-only
control”，文章据此收缩总体工具优势 claim，并把工具价值限定到 RP67 内容 utility、特定
sub-benchmark 和策略行为。协议复核同时发现：历史 Original raw direct 的图像上限为
`262,144` pixels，而 TGVF、Atomic 与 No-Tool matched 各臂为 `1,003,520` pixels；Crop
official-visible 路径后来确认实际落到 processor default。冻结 S32 的 raw-direct@512
Macro* 为 `54.3543`，比同合同 Original 低 `1.0013 pp`；这进一步否定“32-step RL 在原始
raw-direct 合同上带来普遍增益”的解释。Crop S32/S80 fixed-boundary 主复测已完成，但 nominal
Crop S80@512 并未真正应用 512² pixel cap：其 Macro* `62.0967` 与 `59.1785` 只能作为两次
native-default 历史运行，二者 RNG namespace 与 protocol 又不相同，差值不可归因。TGVF
S64@512 与 Atomic S16@512 的路径经复核正确，Macro* `55.4067 / 57.2762` 及相对各自
matched@1M 的 `−4.4019 / −5.8065 pp` 仍有效。下一步优先完成 Crop true@1M/true@512 的
corrected common-RNG rerun，再完成正式 Atomic matched/target-only 盲审、target-only 调用行为
对照与英文 Experiments/Discussion 初稿。**

进度查看：本报告同步到 main 工作区
`docs/NEURIPS_WORKSHOP_TGVF_EXPERIMENT_PLAN_PROGRESS_REPORT_20260826.md`。在推理完成、评分完成、
审计包生成和文章结论更新等关键节点同步；运行中的计数只作为状态快照，不提前当作结果。按当前
授权，每个关键节点只提交这一个报告文件并 push 到 `origin/main`，不带入 main 的其他工作区改动。

## 0. 紧急勘误：Crop `</tool_call>` action boundary（2026-08-28）

**Crop S32/S80 fixed-boundary 复测已完成，本文当前 Crop headline 使用 S80 的
native-default 历史运行；此前 nominal S80@512 不能作为像素控制。** 历史 S8/S16/S48/S64
的准确率、sub-benchmark 和工具使用率仍为 provisional；旧 boundary 的 nominal S80@512
结果已作废。Original、Pure TGVF 和 Atomic 不因 Crop action-boundary 缺陷降级；其中 TGVF
和 Atomic 的 pixel-cap override 路径经复核有效。

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
| S32 / fixed boundary | **61.6699** | 80.6283 | 67.5000 | 61.6667 | 53.8465 | 44.9814 | 69.6667 | 53.4000 |
| S32 fixed − old | **−1.8678** | +0.5236 | −5.5000 | −2.7778 | −0.9643 | −4.0892 | −1.6667 | +1.4000 |
| S80 / old boundary | 62.2288 | 81.6754 | 74.5000 | 58.8889 | 55.3358 | 46.4684 | 67.3333 | 51.4000 |
| S80 / fixed boundary, native-default historical | **59.1785** | 78.5340 | 61.5000 | 57.7778 | 55.4948 | 44.6097 | 66.3333 | 50.0000 |
| S80 fixed − old | **−3.0503** | −3.1414 | −13.0000 | −1.1111 | +0.1589 | −1.8587 | −1.0000 | −1.4000 |

- S32/S80 的 executed calls 为 `1,677 / 2,010`，successful observations 也精确为
  `1,677 / 2,010`；有效使用题为 `1,385 / 1,977`（`61.83% / 88.26%`）。
  两臂同轮 `tool_call + final` 均为 `0`，所有工具 turn 均以 `</tool_call>` 结束。
- S80 仍是事先指定的 80-step headline，不因修正后 S32 高 `2.4914 pp` 而 post-hoc
  重选 checkpoint。S80 只在 OCR mean 上高于 S32（`+1.6483 pp`）。
- nominal Crop S80@512 同样受旧边界缺陷影响。fixed-boundary plan 已在现有
  `neurips-notool-rl-s32` 分支 commit `eb37ad9` 冻结并完成：`2,240/2,240` 条受支持
  trajectory、`7/7` 官方 slice、summary `status=pass`、judge parse failure `0`，Macro*
  为 `62.0967`。该运行可验证 action boundary，但不能验证 512² pixel cap：processor audit
  显示它与 Macro* `59.1785` 的 S80 运行均使用默认 `size.longest_edge=16,777,216`，样例
  `2250×1500` 图像均产生 `3,290` 个 initial visual tokens。旧 `61.5591` 只作历史无效记录；
  true@1M/true@512 需使用 nested `images_kwargs.size` 后按 common RNG 重跑。

## 1. 文章当前主线

本文不以宽泛的“互补能力与优化动态”作为唯一叙事。当前更可检验、也更有证据支撑的主线是：

> Under the matched protocols, no-tool RL provides the strongest aggregate control,
> but its gain does not transfer to the raw-direct@512 contract. Target-conditioned
> latent evidence retains target-specific utility and method-specific advantages in
> selected visual-reasoning regimes, without establishing aggregate superiority of
> tool-augmented policies.

正文先以 **No-Tool RL** 收缩总体 claim，再以 **Pure TGVF** 为机制主线、**Native Crop** 为
native-default 历史工具基线、**Original** 为 raw direct 端到端参考。Crop 的公平 1M/512 基线
位置等待 corrected rerun。**Atomic Crop+TGVF** 目前仍列探索性扩展；其正文层级等待真正的
target-only 稳健性与无偏 target 合格率审计。已完成的广义 full-prompt stress test 同时改变了
多项 prompt/observation 合同，不能单独用于决定 Atomic 的正文层级。

## 2. 固定术语和比较口径

| 简称 | 本文固定含义 | checkpoint / run | 解释边界 |
|---|---|---|---|
| **Original raw-direct@512** | 原始 Qwen3-VL-8B-Instruct；无视觉工具、无自定义 system prompt；`max_pixels=262144` | `PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4` | 必须进入所有主表和 sub-benchmark 表；与 PRL25 matched 行同时跨越 prompt/agent protocol 和输入像素上限，只是端到端 direct reference |
| **No-Tool RL** | 同一 Qwen3-VL-8B-Instruct 做 full-model RL，但没有 Crop、TGVF、RP67、工具 schema 或工具调用 | `PRL-25-F-...-NO-TOOL-RL-...-32STEP-WS8`；S32 为事前冻结主终点 | matched S0/S8/S16/S32 回答同协议优化动态；新增 S32 raw-direct@512 回答训练后模型在 Original 合同下的 transfer，不得改称 Original |
| **Crop** | PRL25-B native RGB Crop | S80，seed42 | 80-step 终点工具基线；当前 `59.1785` 是 fixed-boundary native-default 历史运行，不补 seed43 |
| **TGVF** | PRL25-C Pure TGVF，Frozen RP67 | S64，seed42；seed43 仅作所选 checkpoint 复测 | 文章机制主线 |
| **Atomic** | PRL25-D Atomic Crop+TGVF，Frozen RP67 | S16，seed42；seed43 仅作所选 checkpoint 复测 | 探索性扩展，不能在审计前声称已稳定学会高质量 target |
| **matched prompt** | 80-step 训练与既有 CoreDev 评测使用的简化、训练匹配 prompt | 历史结果 | 用于现有主表 |
| **full prompt** | 详细说明 target、bbox、关系与禁止答案泄漏的 Instruct prompt；可见与运行时上限均为 6 次 | `full_visual_tool_prompt_v5_instruct_cap6` | 只在冻结 S64/S16 上评测，不重选 checkpoint；衡量 prompt shift robustness |
| **matched@512 control** | 保留原 matched prompt、工具 schema、agent loop、checkpoint、seed42 与七项任务，并把评测上限真正改为 `262144` | 当前仅 TGVF S64 / Atomic S16 已有效完成 | 单变量输入分辨率控制；Crop nominal @512 因 override 未生效而排除，true@1M/true@512 pending |
| **Crop native-default historical runs** | fixed-boundary Crop S80 的两次历史运行；processor 均使用默认 `size.longest_edge=16,777,216` | Macro* `59.1785 / 62.0967` | 后者虽在 plan 中 nominal 标为 @512，但不是有效 pixel-cap control；两次运行 RNG/protocol 不同，差值不可归因 |
| **Macro\*** | 七个百分比组件的无权平均 | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均 | 只在相同测量合同内比较 |

固定排除项：**不补 Crop seed43**。它不是本文结论的必要验证，也不用于构造三方法对称性。

### 2.1 输入分辨率与测量合同

`max_pixels` 表示保持长宽比时允许的最大图像像素面积，不是把所有图片强制缩放成正方形。
`262,144 = 512²`，而 `1,003,520` 的等面积正方形边长约为 `1,002` pixels，后者的像素面积预算
约为前者的 `3.83×`。

训练侧的 Crop、TGVF、Atomic 与 No-Tool 四条 PRL25 RL trainer 日志均记录
`mm_processor_kwargs.max_pixels=1003520`。但“配置中记录了该值”不等于所有 evaluator 都实际
应用了该值。TGVF、Atomic 与 No-Tool 的冻结 matched 路径正确解析该上限；Crop 的
`policy_official_visible` 路径把 `max_pixels` 作为 fast processor 不读取的顶层 kwarg 传入，
实际回退到 `size.longest_edge=16,777,216`。因此现有 Crop S80 不能再标成 matched@1M，也不能
与 nominal @512 组成像素消融。Original 未经过这轮 RL，其历史 direct eval 的 262,144 路径仍
有效。

| 比较臂 | Prompt / agent contract | 最大图像像素面积 | 当前用途 |
|---|---|---:|---|
| Original raw-direct@512 | 无 system prompt、无工具、direct generation | 262,144 | 历史端到端参考；也是 S32 raw-direct@512 的严格 base comparator |
| Crop S80 fixed-boundary historical | native Crop prompt 与 agent loop | processor default 16,777,216 | Macro* `59.1785`；不是 matched@1M，保留为历史性能记录 |
| Crop S80 nominal @512 fixed-boundary | 同一类 Crop evaluator；plan nominal 请求 262,144 | processor default 16,777,216 | Macro* `62.0967`；override 未生效，invalid pixel control |
| TGVF S64 / Atomic S16 matched | 各自训练匹配的工具 prompt 与 agent loop | 1,003,520 | 两个 latent-evidence 方法之间的有效 matched 上限 |
| TGVF S64 / Atomic S16 matched@512 | 与上一行逐方法相同，只降低 evaluator 像素上限 | 262,144 | 两臂均已闭合且 override 路径有效 |
| No-Tool S0/S8/S16/S32 matched | 训练匹配的 no-tool prompt 与 direct-only loop | 1,003,520 | S0→S32 的同协议 RL 动态；与 TGVF/Atomic matched 像素上限对齐，不与现有 Crop 历史运行对齐 |
| No-Tool S32 raw-direct@512 | 与历史 Original 相同的 raw-direct 配置，只替换 model path | 262,144 | 已完成；Macro* `54.3543`，比 Original 低 `1.0013 pp` |

因此，No-Tool matched 的 S32−S0 `+2.2141 pp` 不受分辨率混杂；TGVF、Atomic 与 No-Tool
matched 的像素上限也相同。Crop 当前历史结果使用 processor default，不能加入这一像素对齐集合。
Original 与任何 PRL25 工具/matched 行的绝对差值同时混入 prompt、agent loop 和输入分辨率，
不能写成 prompt-only gain，更不能直接归因给 RL 或工具。

## 3. 当前主结果：Original 必须在场

下表全部为当前选定 checkpoint 的 seed42 结果，单位为 `%`。Original 的精确七项均值为
`55.3556`，按其既有 raw-direct@512 测量合同报告为 `55.36`。No-Tool S32 raw-direct 行与
Original 使用同合同；TGVF、Atomic 与 No-Tool matched 行使用有效的 matched@1M 合同，而 Crop
行是 processor-default 历史合同。该表用于完整展示，不构成所有行之间的严格 paired ranking。

| Method | Macro* | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.3556 | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| No-Tool RL S32 raw-direct@512 | 54.3543 | −1.0013 | 51.3089 | 59.0000 | 60.0000 | 47.9377 | 39.0335 | 74.0000 | 49.2000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | +3.8229 | 78.5340 | 61.5000 | 57.7778 | **55.4948** | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 | 59.8086 | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | +7.7271 | 71.7277 | **73.5000** | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |
| No-Tool RL S32 | **66.6853** | **+11.3297†** | **84.2932** | 69.0000 | **70.5556** | 50.8528 | **55.7621** | **75.3333** | **61.0000** |

`†` No-Tool S32 matched 与 Original 的差值同时跨越 prompt/protocol 和最大图像像素面积
（`1,003,520` vs `262,144`），只是端到端参考，不能解释为同协议 RL effect。同协议 RL effect
应使用 matched@1M 的 S32−S0，即 `+2.2141 pp`。在 Original 同测量合同下，S32 raw-direct@512
反而低 `1.0013 pp`；因此 matched 增益不能外推为 raw-direct transfer。

可直接写入正文的事实边界：

- 三个工具方法的历史 Macro* 都高于 raw-direct@512 Original；该描述同时跨越 prompt/agent
  contract 和像素上限，只能作为端到端观察。TGVF/Atomic 的像素上限对齐，但 Crop 使用
  processor default，不能再声称三个工具线路内部像素对齐。Crop S80 是按用户指定报告的
  80-step 历史终点，不再使用 post-hoc 最优 S32。
- No-Tool RL S32 是全表最高 Macro*，比 Crop S80 native-default 历史运行、TGVF S64、Atomic S16 分别高
  `7.5068 / 6.8767 / 3.6026 pp`。这直接否定“当前工具方法总体优于 RL-only 对照”的写法。
- No-Tool S0 已达到 `64.4712`，而 S32−S0 只有 `+2.2141 pp`。因此 prompt/schema/agent
  protocol 是主要替代解释之一；matched no-tool 去除了工具 schema，仍不是工具存在性的严格
  单变量消融。
- No-Tool S32 raw-direct@512 相对 Original 的分项差值为 VStar `+0.5236`、HR `0.0000`、
  BLINK-180 `−5.5556`、OCR mean `−0.2471`、MMMU-269 `+0.0035`、MathVista `−0.3333`、
  MathVerse `−1.4000 pp`。净结果为 `−1.0013 pp`，不支持 raw-direct transfer gain。
- TGVF 不是全榜最优方法，因此文章不能写成“通用性能支配”。它相对 Original 的主要整体
  增益在 VStar、HRBench 和 MMMU，并在一组更细的关系/数学任务中形成集中优势。
- Atomic 的 Macro* 比 Crop S80 native-default 历史运行高 `3.9042 pp`，但两者的有效像素合同
  不同；该差值只能描述既有端点，不能支持公平的方法排序或“Crop+TGVF 存在因果 synergy”。
- Original 在部分视觉强度与 OCR 切片上仍有优势，必须作为负面边界一起报告；No-Tool matched
  S32 在 MathVista headline 上比 Original 高 `1.0000 pp`，但 raw-direct@512 S32 低
  `0.3333 pp`。

## 4. 用于彰显优势的 sub-benchmark 面板

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

### 4.4 三种工具方法的调用覆盖率与调用强度

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

两条线路的 full-prompt 文章口径结果如下。Original 仍是 raw direct reference，不把它误写成
严格 paired prompt control；`Δ vs matched` 才是同 checkpoint、同任务和同 seed 下的 prompt-shift
稳健性量。

| Reference / run | Macro* | Δ vs matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw direct | 55.3556 | — | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | n/a | +3.8229 | **78.5340** | 61.5000 | 57.7778 | **55.4948** | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 / matched prompt | 59.8086 | reference | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / full prompt | 58.5138 | −1.2949 | +3.1582 | 71.7277 | 64.5000 | 66.1111 | 39.4659 | 45.7249 | 68.6667 | 53.4000 |
| Atomic S16 / matched prompt | **63.0827** | reference | **+7.7271** | 71.7277 | **73.5000** | **66.1111** | 54.2720 | **51.3011** | 69.6667 | 55.0000 |
| Atomic S16 / full prompt | 60.4684 | **−2.6142** | +5.1128 | 70.6806 | 67.0000 | 63.3333 | 50.4349 | 46.0967 | 70.3333 | **55.4000** |

full prompt 相对 matched prompt 的七项 delta 依次为 VStar `−2.6178`、HR `−2.0000`、
BLINK-180 `+0.5556`、OCR mean `−5.0787`、MMMU-269 `+0.7435`、MathVista
`−3.6667`、MathVerse `+3.0000 pp`。因此当前结论不是“prompt 完全无影响”，而是：TGVF
总体优势经较完整 prompt 后仍保留，Macro* 仍比 Original 高 `3.1582 pp`，七项中有五项高于
Original；但 prompt shift 造成 `1.2949 pp` 总体回落，损失主要集中在 OCR、MathVista 和
VStar。sub-benchmark 上，full prompt 仍保留 Relative Depth `83.33%`、HR cross aggregate
`62.00%`、MathVista numeric commonsense `55.56%` 和 visual QA `59.52%` 等定位，同时
Relative Reflectance 从 matched 的 `46.67%` 升至 `66.67%`；原先 arithmetic 和 math-word
优势分别回落至 `63.46% / 74.60%`。这些变化必须作为能力迁移而不是单一“更强/更弱”报告。

Atomic full prompt 的 Macro* 为 `60.4684`，比自身 matched prompt 低 `2.6142 pp`，但仍比
Original 高 `5.1128 pp`，并比 TGVF full prompt 高 `1.9546 pp`。七项相对自身 matched 的
delta 为 VStar `−1.0471`、HR `−6.5000`、BLINK-180 `−2.7778`、OCR mean
`−3.8370`、MMMU-269 `−5.2045`、MathVista `+0.6667`、MathVerse `+0.4000 pp`。
这说明组合工具在广义完整 prompt 下仍有总体价值，但不能由此单独决定其正文层级：预冻结的 Atomic
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

| Arm | Macro* | Δ vs own matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.3556 | — | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 / fixed boundary, native-default historical | 59.1785 | n/a | +3.8229 | **78.5340** | 61.5000 | 57.7778 | **55.4948** | 44.6097 | 66.3333 | 50.0000 |
| TGVF S64 / matched | 59.8086 | reference | +4.4531 | 74.3455 | 66.5000 | **65.5556** | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / target-only | 58.1788 | -1.6298 | +2.8233 | 75.3927 | 61.5000 | 63.8889 | 40.1453 | 45.7249 | 68.0000 | 52.6000 |
| Atomic S16 / matched | 63.0827 | reference | +7.7271 | 71.7277 | **73.5000** | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |
| Atomic S16 / target-only | 60.8253 | -2.2574 | +5.4697 | 71.7277 | 70.5000 | 61.6667 | 49.3089 | 46.8401 | 69.3333 | **56.4000** |

target-only 改动没有带来总体提升，但也没有抹去相对 Original 的整体优势。TGVF 的主要回退在
HR `-5.00 pp`、OCR mean `-4.40 pp` 和 MathVista `-4.33 pp`，同时 VStar、MMMU-269、
MathVerse 分别提高 `+1.05 / +0.74 / +2.20 pp`。Atomic 的主要回退在 OCR mean `-4.96 pp`、
MMMU-269 `-4.46 pp`、BLINK-180 `-4.44 pp` 和 HR `-3.00 pp`，MathVerse 提高 `+1.40 pp`。
因此“更详细 target 定义本身普遍提高准确率”不被支持；可以支持的是两种方法在该单变量 prompt
干预下仍保持高于 Original 的 Macro*，同时存在明确的 benchmark-specific sensitivity。

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
- [ ] 基于已闭合的 target-only inference 物化新的 matched/target-only 正式盲审包；既有
  广义-full pack 不替代该主要审计；
- [ ] 双人盲标、裁决、Wilson CI 与 agreement；
- [ ] 根据审计结果界定 Atomic 探索性分析可使用的 target-quality 表述。

正式盲审根目录：
`artifacts/evaluation/neurips-workshop-atomic-target-audit-20260826-v1/`。manifest SHA256 为
`5af64da18e03d7455ab523e011887baf7d1361e50a59d19914f10380bbd165cc`；状态为
`ready_for_blind_annotation`。matched/full 可用工具 trajectory population 分别为
`1,863 / 2,009`，按各自七套数据 population-proportional 抽样 200 条。coordinator key 与
review view 分离；人工标注完成前不得报告 target 合格率或据此升级 Atomic 的正文地位。

### 5.4 No-Tool RL：RL 本身的正式对照

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
控制。与现有 Crop 历史运行的比较仍跨 effective pixel contract，需等待 Crop true@1M。
协议复核发现历史 Original 实际使用 `max_pixels=262144`，而上述 matched 评测使用
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

- 若工具方法超过 matched No-Tool RL，可把差额解释为与工具化训练合同相关的增益候选；只有在
  其他合同严格一致且置信区间支持时，才进一步归因到视觉工具。
- 若 matched No-Tool RL 已接近工具方法，正文必须把“RL 本身”列为主要替代解释，不能把全部
  增益归给 TGVF/RP67。
- matched no-tool 去除了工具 schema，因此它是当前最强的 RL-only control，但仍不可消除
  “存在工具 schema / agent loop”这一协议差异。已有 Original 的 raw direct 结果也不能替代
  matched RL-only 比较；S32 raw-direct@512 已显示 `−1.0013 pp` transfer，不替代 matched 工具对照。

#### 5.4.4 正式结果与学习动态

下表全部使用 matched no-tool 协议。S32 是事前冻结的唯一 headline；S8/S16 只用于描述动态，
不能因为 S16 的 Macro* 略高而把正式 checkpoint 改选为 S16。

| Step | Macro* | Δ vs S0 | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 64.4712 | — | 79.5812 | 61.0000 | 65.5556 | 48.8072 | 57.6208 | 77.3333 | 61.4000 |
| S8 | 66.1132 | +1.6420 | 83.7696 | 66.0000 | 65.5556 | 48.9899 | 56.8773 | **78.0000** | **63.6000** |
| S16 | **66.9028** | **+2.4317** | 82.1990 | **69.5000** | **71.1111** | 50.6324 | 56.8773 | 75.0000 | 63.0000 |
| **S32 frozen** | 66.6853 | +2.2141 | **84.2932** | 69.0000 | 70.5556 | **50.8528** | 55.7621 | 75.3333 | 61.0000 |

同合同 raw-direct transfer 结果如下。OCR EN/CN 分别为 `49.5995 / 46.2760`；MathVerse 五版本
为 Text Dominant `70.0`、Vision Only `30.0`、Text Lite `48.0`、Vision Intensive `48.0`、
Vision Dominant `50.0`。

| Method | Macro* | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct@512 | 55.3556 | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| No-Tool S32 raw-direct@512 | 54.3543 | −1.0013 | 51.3089 | 59.0000 | 60.0000 | 47.9377 | 39.0335 | 74.0000 | 49.2000 |

观察与解释边界：

- 同协议 S32−S0 为 `+2.2141 pp`，说明 32-step No-Tool RL 带来温和净增益；学习曲线在 S16
  达到 `66.9028` 后轻微回落 `0.2175 pp`，但不触发 post-hoc checkpoint 重选。
- S0 已达到 `64.4712`，高于 Crop S80 native-default 历史运行、TGVF S64 和 Atomic S16 的
  headline Macro*。因此 no-tool prompt/protocol 本身是主要替代解释，当前实验不能把工具线路
  相对 raw Original 的总增益全部归给工具，也不支持工具方法在 Macro* 上超过当前 RL-only
  control；其中与 Crop 的比较还额外跨越有效像素合同。
- S32 相对 Crop native-default 历史运行、TGVF、Atomic 的描述性差值分别为
  `+7.5068 / +6.8767 / +3.6026 pp`。工具证据必须转向特定切片、RP67 三臂 utility 与行为机制；
  Crop true@1M 结果产生前，第一项尤其不能作为像素对齐的方法差值。
- 该比较仍不是“只切换工具开关”的严格因果消融：matched no-tool 同时去除了工具 schema 和
  agent loop。Original raw direct 必须保留，但其 prompt/protocol 又不同，只作端到端参考。

当前状态：

- [x] 名称、S32 主终点、训练预算、matched no-tool 评测协议和解释边界已冻结；
- [x] 独立干净分支上的 no-tool prompt/data route、空工具 schema、run config、S32 守护脚本与
  CPU compose/回归测试；
- [x] 真实 1-step canary：首轮在 Step 0 前发现共享 termination builder 仍把
  `</tool_call>` stop 当作全路径硬条件；修复后 canary 已闭合 Step 1 且零工具调用；
- [x] 正式 32-step 训练：S8/S16/S32 永久 checkpoint 与最终 supervisor acceptance 均已闭合；
- [x] S0/S8/S16/S32 matched no-tool 的 `4 × 2,240` 条单图推理；
- [x] S0/S8/S16/S32 matched no-tool 的七项正式评分；
- [x] 结果回填主表、sub-benchmark、调用行为表和 claim ledger。
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
  `abee6a4`。正式 S32 Macro* 为 `66.6853`，所有 No-Tool 结果现已进入本文主表、完整
  sub-benchmark、调用行为表与 claim ledger。
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
  runner、judge 和全部 GPU 随后正常退出。该节点只闭合 action-boundary 与评分工件，后续
  processor audit 已撤回其 `@512` 像素控制身份。
- 2026-08-28 Crop processor-override 勘误节点：逐条比较两次 fixed-boundary S80 的 initial
  visual token counts 后确认它们完全相同，均走 fast processor 默认
  `size.longest_edge=16,777,216`。此前传入的顶层 `max_pixels` kwarg 被忽略。对一张
  `2250×1500` 样例图，两次历史运行均产生 `3,290` tokens；使用正确的 nested
  `images_kwargs.size` 后，true@512 与 true@1M 分别产生 `247 / 950` tokens。两次历史运行
  的 RNG namespace 与 protocol 不同，因此 `62.0967−59.1785=+2.9182 pp` 只是不可归因的
  跨运行差异。Crop true@1M/true@512 common-RNG rerun 尚待执行；TGVF/Atomic 的 override
  路径和 paired RNG 复核通过，既有两组 pixel-cap 结果不撤回。

### 5.5 有效的 TGVF/Atomic matched@512 控制与 Crop pixel-cap 勘误

本节原计划对 Crop、TGVF 与 Atomic 做 matched@1M→matched@512 单变量控制。运行后处理器审计
确认，TGVF 与 Atomic 的 override 路径正确，且各自 1M/512 运行逐字复用了同一 paired RNG
namespace。Crop 的 `policy_official_visible` 路径则把 `max_pixels` 放在 fast processor 忽略的
顶层 kwarg 中；两次 S80 运行均实际使用 processor default
`size.longest_edge=16,777,216`。因此本节只保留 TGVF/Atomic 的像素效应结论，Crop true@1M
与 true@512 仍为 pending corrected common-RNG rerun。

| Arm | Frozen checkpoint | Nominal request | 实际 processor 合同 | 状态 |
|---|---|---:|---:|---|
| Crop historical A | S80 | 1,003,520 | default 16,777,216 | Macro* `59.1785`；native-default 历史运行，不是 true@1M |
| Crop historical B | S80 | 262,144 | default 16,777,216 | Macro* `62.0967`；nominal @512 invalid control，只验证 fixed boundary |
| TGVF | S64 | `1,003,520 → 262,144` | `1,003,520 → 262,144` | **有效且已完成** |
| Atomic | S16 | `1,003,520 → 262,144` | `1,003,520 → 262,144` | **有效且已完成** |

Crop 的判定来自实际 visual-token 证据，而不是配置文件名。两次历史运行的逐题 initial visual
token counts 完全相同；一张 `2250×1500` 样例图在两次运行中均为 `3,290` tokens。将 cap
正确放入 nested `images_kwargs.size` 后，同图 true@512 与 true@1M 分别为 `247 / 950`
tokens，证明旧调用链没有应用计划中的 cap。历史 A/B 还使用不同 RNG namespace 与 protocol，
所以 `62.0967−59.1785=+2.9182 pp` 只是不可归因的随机运行差异，不能解释成分辨率效应、调用
策略效应或更低分辨率增益。

当前可报告的结果如下。Original 始终保留为 raw-direct@512 端到端参考；`Δ vs Original` 只
描述总系统差值，不把跨 prompt/tool/agent contract 的差异归因给工具。Crop 两行只作历史记录，
不进入 pixel-cap delta。

| Arm / effective pixel contract | Macro* | Δ vs own true@1M | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct / 262,144 | 55.3556 | — | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 historical A / native-default | 59.1785 | n/a | +3.8229 | 78.5340 | 61.5000 | 57.7778 | **55.4948** | 44.6097 | 66.3333 | 50.0000 |
| Crop S80 historical B / native-default; nominal @512 invalid | 62.0967 | n/a | +6.7411 | **86.9110** | 65.5000 | 64.4444 | 54.4299 | 45.7249 | 65.6667 | 52.0000 |
| TGVF S64 / 1,003,520 | 59.8086 | reference | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / 262,144 | 55.4067 | −4.4019 | +0.0511 | 53.4031 | 60.5000 | 62.7778 | 40.7642 | 46.4684 | 72.3333 | 51.6000 |
| Atomic S16 / 1,003,520 | 63.0827 | reference | +7.7271 | 71.7277 | **73.5000** | **66.1111** | 54.2720 | **51.3011** | 69.6667 | **55.0000** |
| Atomic S16 / 262,144 | 57.2762 | −5.8065 | +1.9206 | 57.0681 | 59.5000 | 61.6667 | 47.7713 | 48.3271 | 71.0000 | 55.6000 |

Crop historical B 的行为统计仍能证明 fixed action boundary 生效，但不能称为 @512 行为。该运行
共执行 `2,014` 次 Crop，全部产生 observation，same-turn mixed 为 `0`；它与 historical A 的
调用差异同样不能归因于 pixel cap。逐 set 历史行为如下：

| Set | Supported questions | Successful-use questions | Executed calls | Calls / question | Tool errors |
|---|---:|---:|---:|---:|---:|
| VStarBench | 191 | 183 | 183 | 0.958 | 5 |
| HRBench4K | 200 | 196 | 198 | 0.990 | 36 |
| BLINK single-image | 180 | 171 | 177 | 0.983 | 8 |
| OCRBench v2 | 600 | 527 | 556 | 0.927 | 52 |
| MMMU-Pro single-image | 269 | 253 | 258 | 0.959 | 2 |
| MathVista MINI | 300 | 263 | 266 | 0.887 | 8 |
| MathVerse MINI | 500 | 376 | 376 | 0.752 | 13 |
| **Overall** | **2,240** | **1,969** | **2,014** | **0.899** | **124** |

每题有效调用次数分布为 `0:271 / 1:1,943 / 2:14 / 3:7 / 4:4 / 5:0 / 6:1`；仅
`26/2,240` 题发生重复有效调用。完整推理还记录 `4,314` 个 assistant turns、`964,186`
sampled tokens，以及 `final/context-limit/cap = 2,206/32/2` 的终止分布。所有 2,240 条推理、
七个 scorer 和 Qwen2.5-72B judge 均无 OOM/traceback，judge parse failure 为 `0`。

TGVF 当前有效的 @512 结果回退 `4.4019 pp`，主要来自 VStar `−20.9424 pp`、
HR `−6.0000 pp`、
OCR mean `−3.7804 pp` 和 BLINK-180 `−2.7778 pp`；MathVista 不变，MMMU-269 与 MathVerse
分别上升 `1.4870 / 1.2000 pp`。TGVF@512 的 Macro* 比 Original raw-direct@512 只高
`0.0511 pp`，而二者仍跨 agent/tool 协议，不能据此声称显著优于 Original。OCR scorer 还暴露
出低分辨率下的生成病态：个别 final answer 达 `117,574` 字符，导致官方本地字符串指标耗时
显著增加；这些回答被原样评分，没有后处理截断。

Atomic 的 @512 回退为 `5.8065 pp`，主要来自 VStar `−14.6596 pp`、HR
`−14.0000 pp`、OCR mean `−6.5007 pp` 和 BLINK-180 `−4.4444 pp`；MathVista 与
MathVerse 分别上升 `1.3333 / 0.6000 pp`。Atomic@512 比 Original raw-direct@512 高
`1.9206 pp`，但同样跨 prompt/agent 合同，不是严格方法优势。

## 6. Atomic 纳入正文的决策门槛

Atomic 进入正文核心方法必须同时满足：

1. target-only Macro* 和主要 Atomic favorable sub-benchmark 不发生足以推翻当前定位的崩塌；
2. target audit 的 all-pass rate 及逐标准结果可被透明报告；
3. 文章只声称观察到的任务条件优势，不声称未被单变量消融证明的 synergy；
4. Original、Crop、TGVF 和 Atomic 四列同时出现，且 MathVerse Vision Intensive 等负面切片保留。

如果任一条件不满足，Atomic 降级到 exploratory analysis / appendix。Pure TGVF + RP67 utility
仍作为主机制线，Crop 作为强基线。

**当前决策：性能门槛部分闭合，Atomic 仍维持 exploratory。** target-only Macro* 为
`60.8253`，仍比 Original 高 `5.4697 pp`，但比自身 matched 低 `2.2574 pp`；其中 HR cross
从 `68.00%` 降至 `60.00%`，Relative Reflectance 从 `70.00%` 降至 `56.67%`，Vision Only
保持 `51.00%`。这不构成整体崩塌，却说明 matched favorable regimes 有明显衰减，尚不足以升级
为稳定核心方法。第 2 项正式 target audit 仍未完成，因此当前只保留探索性正文/附录定位。

## 7. Claim–evidence–boundary 台账

| Claim | 当前证据 | 必须保留的边界 | 状态 |
|---|---|---|---|
| TGVF 改善一组 target-conditioned reasoning regimes | Relative Depth；MathVista arithmetic、word、numeric、visual QA；逐题案例 | 不是总体最优；OCR 和精细像素读取弱；BLINK 切片小 | 可写，待 CI |
| RP67 D 具有内容 utility 和 target specificity | correct−zero `+7.15 pp`；correct−wrong `+21.57 pp`；两者 95% CI 不跨零 | diagnostic semantic overlay；oracle target 不测自主工具选择 | 已支持 |
| Atomic matched prompt 下在跨图、反射率和 Vision Only 上显示优势 | matched HR cross、BLINK reflectance、MathVerse Vision Only | target-only 下 HR cross / reflectance 优势明显收窄，Vision Only 保持；target 合格率未闭合 | 探索性，核心门槛部分验证 |
| 广义 prompt bundle 下工具总体增益仍存在 | TGVF / Atomic stress-test Macro* 为 `58.5138 / 60.4684`，分别比 Original 高 `3.1582 / 5.1128 pp` | 相对各自 matched prompt 下降 `1.2949 / 2.6142 pp`；非 target-only；不是新 benchmark 泛化 | 已支持，带退化边界 |
| 只补充 target 定义与案例的稳健性 | TGVF / Atomic target-only Macro* `58.1788 / 60.8253`，仍比 Original 高 `2.8233 / 5.4697 pp` | 相对 own matched 分别下降 `1.6298 / 2.2574 pp`；不支持“详细 target 定义普遍增益” | 已支持，带退化边界 |
| 工具方法整体优于 raw direct Original | 三个选定 checkpoint 的历史 Macro* 均高于 55.36 | Original 非 paired control；TGVF/Atomic 为 1,003,520，Crop 为 processor default 16,777,216，而 Original 为 262,144；不能把差值归因给工具 | 仅可作跨合同端到端描述 |
| 工具方法的总体增益超越当前 RL-only/no-tool control | No-Tool matched S32 `66.6853`，高于 Crop native-default historical/TGVF/Atomic `7.5068/6.8767/3.6026 pp`；S0 `64.4712`；S32−S0 `+2.2141 pp` | no-tool 与工具线路的 schema/agent-loop 仍不同；与 Crop 还跨有效像素合同。S32 raw-direct@512 为 `54.3543`，比 Original 低 `1.0013 pp` | **当前不支持，必须撤回总体 claim** |
| Crop 在 1M→512 下获得 `+2.9182 pp` 像素增益 | 两次 historical Macro* 为 `59.1785 / 62.0967`，但 initial visual token counts 完全相同；`2250×1500` 样例均为 3,290 tokens | 两次均走 default 16,777,216，且 RNG namespace/protocol 不同；差值不可归因 | **已撤回；true@1M/true@512 pending corrected common-RNG rerun** |
| TGVF/Atomic 降低输入像素上限会降低当前 checkpoint 的 Macro* | 有效 matched@512 相对 matched@1M 为 `−4.4019 / −5.8065 pp` | 两臂各自 1M/512 paired RNG 完全一致；只支持所选 checkpoint 的逐方法消融，不证明训练像素最优性 | **已支持，限 TGVF/Atomic** |
| 三种方法形成不同工具调用行为 | Crop native-default historical/TGVF/Atomic successful-use rate `88.26/89.73/83.17%`；calls/question `0.897/0.898/1.027`；逐 set 表 | 描述性统计；Crop 不属于 1M 对齐集合；调用更多不等于 utility 更高；policy 自选择混杂 | 已支持，带 Crop 像素合同边界 |

## 8. 论文实验部分建议结构

1. **Comprehensive comparison and control result.** Original、No-Tool RL 与三个工具方法的七套
   benchmark 主表；先报告 No-Tool S32 总体最高及其协议边界，Original 永不缺席，Crop
   native-default 历史列显式标注为未与 1M 对齐。
2. **How much does RL itself add?** 报告冻结 No-Tool S32 与同协议 S0/S8/S16 动态，明确
   S32−S0 `+2.2141 pp`，并把 prompt/schema/agent protocol 列为替代解释。
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
8. **Qualitative mechanisms and failures.** 复用已有真实 trajectory 与 bbox 案例，但把机制语言
   限定为 behavior-level inference。

## 9. 当前推进顺序

1. [已完成] RP67 semantic overlay 和 CI，机制主张已锁定；
2. [已完成] 广义 full-prompt stress test 及七项官方评分；
3. [已完成] matched-prompt 三方法整体、逐 set 调用率、调用次数和错误类型审计；Crop 行已补充
   native-default effective pixel contract 边界；
4. [已完成] TGVF S64、Atomic S16 target-only matched prompt 推理与七套官方评分；
5. [已完成] No-Tool RL：合同、实现、CPU 回归、真实 canary、正式 S32 训练、S0/S8/S16/S32
   matched no-tool 推理、28 个 slice 评分、四臂汇总与文章结果回填均已闭合；
6. [已完成] S32 raw-direct@512 及 TGVF S64、Atomic S16 的有效 matched@512 已闭合；
   Crop Macro* `59.1785 / 62.0967` 已降级为 native-default 历史运行，nominal @512 control
   无效；
7. [待执行] 使用正确 nested `images_kwargs.size` 完成 Crop true@1M/true@512 common-RNG rerun；
8. [待执行] 从 matched/target-only inference JSONL 物化正式 Atomic
   blind audit pack；
9. [待回填] target-only 调用行为对照与正式 audit；
10. [待写作] 形成英文 Experiments/Discussion 初稿；
11. [明确不做] Crop seed43。

## 10. 证据来源

- Original 定义和 Macro* 合同：
  `docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md`
- 80-step 三线路数值与 checkpoint 选择：
  `docs/PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md`
- 三方法逐题轨迹与失败案例：
  `docs/PRL25_CROP_TGVF_ATOMIC_QUALITATIVE_CASE_ANALYSIS_20260825.md`
- 工具调用行为：三个 matched 最佳 checkpoint 各自 `step80/step64/step16/inference/rank-0..3.jsonl`
  的 `tool_calls`、`tool_errors` 与 `successful_observation_count`；三臂均为 2,240 个唯一共同 ID。
- Crop S32/S80 fixed-boundary 官方 summary：
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-S32-S80-TOOL-BOUNDARY-FIX-V2/step{32,80}/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
  总合同为同级 `paired-summary.json`。其中 S80 Macro* `59.1785` 只标作 native-default
  historical，不标作 true@1M。
- TGVF S64 / Atomic S16：对应 six-point evaluation 的 `step64` / `step16` 官方 summary 与
  `paired-summary.json`。
- No-Tool RL S0/S8/S16/S32 matched summary：
  `artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1/matched/step{0,8,16,32}/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
  28 个精确评分来源由同级各 dataset 的 `pinned-reuse-receipt.json` 固定。
- nominal `@512` 执行计划：
  `configs/evaluation/prl25_{b_crop_exact_step80,c_frozen_rp67_tfree_teacher25_s64_matched,d_atomic_crop_tgvf_s16_matched}_pixel512_coredev2511_plan.json`；
  并行、可恢复 supervisor 为 `tools/supervise_prl25_bcd_selected_pixel512_evaluation.sh`。其中
  TGVF/Atomic 的 override 有效；Crop 计划只作无效控制的 provenance。
- Crop S80 nominal @512 fixed-boundary 历史计划：
  `configs/evaluation/prl25_b_crop_exact_step80_pixel512_tool_boundary_fix_v2_coredev2511_plan.json`，
  现有 `neurips-notool-rl-s32` 分支 commit `eb37ad9`；accepted summary 与总合同分别为
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-PIXEL512-TEMP1-SEED42-TOOL-BOUNDARY-FIX-V2/step80/scoring/coredev-official-v1/coredev-2511-eval-summary.json`
  和同级 evaluation 根目录的 `paired-summary.json` / `evaluation-complete`。该 artifact 的
  Macro* `62.0967` 只证明运行与 fixed-boundary 评分闭合，不证明 pixel cap 生效。
- Crop processor audit：两次 fixed-boundary S80 的逐题 initial visual token counts 完全一致；
  processor default 为 `size.longest_edge=16,777,216`。`2250×1500` 样例在两次历史运行均为
  `3,290` tokens，正确 nested `images_kwargs.size` 下 true@512/true@1M 为 `247/950`。
  corrected common-RNG rerun 尚无 accepted summary，不能提前填入结果表。
- No-Tool S32 raw-direct@512 accepted summary：
  `artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-RAW-DIRECT-512-S32-V1/scoring/coredev-2511-eval-summary.json`。

注意：主仓当前 qualitative 文档可能尚未进入本 worktree 的提交历史；本文只把它作为只读证据
来源，不覆盖主仓未提交内容。

## Appendix A. 完整 aligned sub-benchmark 清单

本附录不只保留有利切片，而是完整列出当前五种方法在共同任务 ID 上可稳定对齐的
sub-benchmark。这里的“对齐”只指样本与 scorer；所有 Crop 列均来自 S80 fixed-boundary
native-default 历史运行，并不与 TGVF/Atomic/No-Tool 的 1,003,520 pixel contract 对齐。Crop
true@1M/true@512 完成前，这些列只作探索性历史上下文。除 MMMU
subject 重算外共有 40 行；其中 TGVF 或 Atomic 在 `22/40` 行、No-Tool S32 在 `27/40` 行上
高于 Original（另有 2 行持平）。这些计数仅用于
清点覆盖面：MathVista skill 等标签彼此重叠，不能把这些行数当作独立样本上的显著性检验或
新的综合指标。小样本切片（尤其 BLINK `n=30` 和 MMMU subject `n=5–10`）只作能力定位，
不能单独承担论文核心结论。

### A.1 VStar、HRBench 与 BLINK 共同单图切片

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 |
|---|---:|---:|---:|---:|---:|---:|
| VStar / direct attributes | 115 | 48.70 | 82.61 | 70.43 | 69.57 | **86.09** |
| VStar / relative position | 76 | 53.95 | 72.37 | 80.26 | 75.00 | **81.58** |
| HRBench / cross-image aggregate | 100 | 59.00 | 40.00 | 64.00 | **68.00** | 66.00 |
| HRBench / single-image aggregate | 100 | 59.00 | **80.00** | 69.00 | 79.00 | 72.00 |
| BLINK / Counting | 30 | 66.67 | 63.33 | 73.33 | 73.33 | **80.00** |
| BLINK / IQ Test | 30 | **40.00** | 13.33 | 23.33 | 10.00 | 26.67 |
| BLINK / Object Localization | 30 | 56.67 | 60.00 | 70.00 | **73.33** | 70.00 |
| BLINK / Relative Depth | 30 | 83.33 | 83.33 | 86.67 | 80.00 | **90.00** |
| BLINK / Relative Reflectance | 30 | 50.00 | 40.00 | 46.67 | **70.00** | 60.00 |
| BLINK / Spatial Relation | 30 | **96.67** | 86.67 | 93.33 | 90.00 | **96.67** |

VStar 和 HRBench 已列出各自全部官方对齐切片。BLINK 官方 full-420 中其余 8 个类别为多图
输入，对当前工具方法不受支持并被填零；本表完整列出五种方法共同支持的 6 个单图类别。

### A.2 MathVista MINI 全部 12 个官方 Task&Skill 标签

前 5 行 task 互斥并完整覆盖 300 题；后 7 行 skill 可重叠，同一题可能进入多个 skill。

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 |
|---|---:|---:|---:|---:|---:|---:|
| MathVista task / figure question answering | 96 | **73.96** | 69.79 | 69.79 | 69.79 | **73.96** |
| MathVista task / geometry problem solving | 49 | **91.84** | 83.67 | 85.71 | 87.76 | 89.80 |
| MathVista task / math word problem | 63 | 77.78 | 60.32 | **84.13** | 79.37 | 82.54 |
| MathVista task / textbook question answering | 50 | **70.00** | 58.00 | 56.00 | 56.00 | 66.00 |
| MathVista task / visual question answering | 42 | 54.76 | 57.14 | **64.29** | 50.00 | 61.90 |
| MathVista skill / algebraic reasoning | 75 | **84.00** | 74.67 | 74.67 | 76.00 | 81.33 |
| MathVista skill / arithmetic reasoning | 104 | 65.38 | 56.73 | **72.12** | 63.46 | 71.15 |
| MathVista skill / geometry reasoning | 63 | **87.30** | 76.19 | 79.37 | 79.37 | 84.13 |
| MathVista skill / logical reasoning | 12 | **41.67** | 33.33 | 16.67 | 25.00 | 33.33 |
| MathVista skill / numeric commonsense | 36 | 47.22 | 44.44 | **58.33** | 50.00 | **58.33** |
| MathVista skill / scientific reasoning | 37 | 62.16 | 56.76 | 54.05 | 56.76 | **64.86** |
| MathVista skill / statistical reasoning | 111 | 82.88 | 79.28 | 81.98 | 81.08 | **83.78** |

五种方法均使用完整的相同 300 个 `index`。作为互斥的附加诊断，free-form / multi-choice 的
样本数为 `171 / 129`：Original 为 `64.91 / 86.82`，Crop S80 native-default historical 为
`53.80 / 82.95`，TGVF
S64 为 `65.50 / 81.40`，Atomic S16 为 `61.40 / 80.62`，No-Tool S32 为
`67.84 / 85.27`。这定位出 TGVF 总体回退主要落在 multi-choice，而不是评测 subset 改变。

### A.3 MathVerse MINI 全部 5 个版本

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 |
|---|---:|---:|---:|---:|---:|---:|
| MathVerse / Text Dominant | 100 | 69.00 | 66.00 | 64.00 | 66.00 | **71.00** |
| MathVerse / Text Lite | 100 | 53.00 | 52.00 | 58.00 | 59.00 | **65.00** |
| MathVerse / Vision Dominant | 100 | 51.00 | 44.00 | 46.00 | 50.00 | **61.00** |
| MathVerse / Vision Intensive | 100 | 52.00 | 40.00 | 42.00 | 49.00 | **53.00** |
| MathVerse / Vision Only | 100 | 28.00 | 48.00 | 42.00 | 51.00 | **55.00** |

### A.4 OCRBench v2 全部 13 个官方语言类别

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 |
|---|---:|---:|---:|---:|---:|---:|
| OCR EN / text recognition | category | 60.49 | 73.05 | 55.05 | **73.38** | 65.54 |
| OCR EN / text detection | category | 29.00 | 22.97 | 28.87 | **36.05** | 28.00 |
| OCR EN / text spotting | category | 0.00 | 5.00 | 16.50 | **18.90** | 4.10 |
| OCR EN / relationship extraction | category | 89.05 | **91.99** | 76.63 | 88.14 | 73.16 |
| OCR EN / element parsing | category | **43.89** | 38.25 | 29.27 | 40.25 | 23.78 |
| OCR EN / mathematical calculation | category | 39.25 | **45.51** | 34.14 | 32.82 | 42.01 |
| OCR EN / visual text understanding | category | 75.00 | **83.27** | 80.00 | 83.21 | 78.33 |
| OCR EN / knowledge reasoning | category | 62.44 | **63.34** | 57.50 | 57.47 | 52.50 |
| OCR CN / text recognition | category | 59.82 | 68.48 | 22.17 | 67.80 | **80.21** |
| OCR CN / relationship extraction | category | 49.31 | **83.37** | 55.82 | 58.81 | 53.29 |
| OCR CN / element parsing | category | 27.75 | 33.01 | 20.69 | **33.14** | 28.17 |
| OCR CN / visual text understanding | category | 35.00 | 45.00 | **65.00** | 55.00 | 60.00 |
| OCR CN / knowledge reasoning | category | **60.51** | 60.48 | 45.55 | 59.07 | 57.21 |

OCR 类别分数来自官方 rule-based scorer；各类别内部样本数和计分尺度不同，因此不对这 13 行
再做无权平均。

### A.5 MMMU-Pro 共同 269 单图的完整 subject 审计

MMMU 官方 subject 表混有工具方法不支持的 31 个多图样本，不能直接与 Original 比。下表以
三个工具方法与 No-Tool S32 result TSV 的 `extra_records.coverage` 支持标记确定 269 个单图
ID，再与 Original result TSV 按相同 `index` 对齐并按 subject 聚合 `hit`。这是由官方逐题判分导出的 aligned
diagnostic，不是 MMMU 官方 subject headline。

| Sub-benchmark | n | Original | Crop S80 native-default | TGVF S64 | Atomic S16 | No-Tool S32 |
|---|---:|---:|---:|---:|---:|---:|
| MMMU / Accounting | 10 | 20.00 | 50.00 | **60.00** | **60.00** | 50.00 |
| MMMU / Agriculture | 10 | **40.00** | 30.00 | **40.00** | 30.00 | **40.00** |
| MMMU / Architecture and Engineering | 10 | 40.00 | 50.00 | 50.00 | **60.00** | **60.00** |
| MMMU / Art | 10 | **50.00** | 40.00 | 30.00 | 30.00 | 30.00 |
| MMMU / Art Theory | 7 | 57.14 | 57.14 | 57.14 | 57.14 | **71.43** |
| MMMU / Basic Medical Science | 10 | 30.00 | 20.00 | 20.00 | **50.00** | 30.00 |
| MMMU / Biology | 7 | **42.86** | 0.00 | 0.00 | 14.29 | 28.57 |
| MMMU / Chemistry | 5 | 20.00 | **60.00** | **60.00** | **60.00** | **60.00** |
| MMMU / Clinical Medicine | 9 | 11.11 | 11.11 | **22.22** | **22.22** | **22.22** |
| MMMU / Computer Science | 10 | **70.00** | 50.00 | 40.00 | 50.00 | 50.00 |
| MMMU / Design | 10 | **70.00** | 60.00 | **70.00** | 60.00 | **70.00** |
| MMMU / Diagnostics and Laboratory Medicine | 10 | 20.00 | 20.00 | 10.00 | 20.00 | **30.00** |
| MMMU / Economics | 8 | 50.00 | 50.00 | 25.00 | 37.50 | **62.50** |
| MMMU / Electronics | 10 | 40.00 | **70.00** | 50.00 | **70.00** | **70.00** |
| MMMU / Energy and Power | 10 | 40.00 | 20.00 | 30.00 | 30.00 | **70.00** |
| MMMU / Finance | 10 | 10.00 | 40.00 | 70.00 | 70.00 | **80.00** |
| MMMU / Geography | 8 | **50.00** | **50.00** | 25.00 | 37.50 | 37.50 |
| MMMU / History | 8 | 50.00 | **62.50** | **62.50** | **62.50** | **62.50** |
| MMMU / Literature | 8 | 75.00 | **87.50** | 75.00 | 75.00 | 75.00 |
| MMMU / Manage | 10 | 50.00 | 20.00 | 40.00 | 50.00 | **70.00** |
| MMMU / Marketing | 9 | 33.33 | 66.67 | 88.89 | **100.00** | 66.67 |
| MMMU / Materials | 8 | 12.50 | 37.50 | 50.00 | **75.00** | 50.00 |
| MMMU / Math | 10 | **60.00** | 50.00 | 40.00 | 40.00 | 50.00 |
| MMMU / Mechanical Engineering | 10 | 0.00 | 40.00 | 40.00 | 40.00 | **60.00** |
| MMMU / Music | 7 | 42.86 | **57.14** | 42.86 | 42.86 | **57.14** |
| MMMU / Pharmacy | 7 | 14.29 | 42.86 | 57.14 | **71.43** | 57.14 |
| MMMU / Physics | 10 | 30.00 | 50.00 | 40.00 | **60.00** | **60.00** |
| MMMU / Psychology | 8 | 50.00 | 25.00 | 25.00 | **62.50** | **62.50** |
| MMMU / Public Health | 10 | 40.00 | 80.00 | 80.00 | 60.00 | **90.00** |
| MMMU / Sociology | 10 | 50.00 | 50.00 | 50.00 | 50.00 | **70.00** |

TGVF 或 Atomic 在 `15/30` 个 subject、No-Tool S32 在 `21/30` 个 subject 上高于 Original；
其中 Marketing、Materials、Finance、Pharmacy、Accounting 的工具方法描述性增益最大。由于
每个 subject 仅 `n=5–10`，这些行只用于提出能力假设和挑选案例，不进入 Macro*。
