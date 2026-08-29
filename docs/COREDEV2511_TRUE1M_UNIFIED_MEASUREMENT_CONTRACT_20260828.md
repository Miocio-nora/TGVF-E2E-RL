# CoreDev-2511 true-1M 统一测量合同与重测台账

更新时间：2026-08-29 22:47 JST

状态：**像素合同已冻结；Original raw-direct、No-Tool S0/S8/S16/S32、TGVF S64 与 Atomic S16
的 true-1M 结果已闭合。Crop S32/S80 的像素测量虽闭合，但已因 post-tool continuation mismatch
隔离为 historical。本文仍是项目级 true-1M 像素口径，但不再授予旧 Crop aligned-golden 身份。**
在本文标为
`accepted` 之前，任何仅在配置中写有 `1,003,520`、但没有真实 processor/grid 证据的结果，均不得
进入统一主表。

> **Crop continuation 更正（2026-08-29）：** PRL25-B/PRL26-B Crop-only 训练在成功 Crop 后
> 使用 generic observation reminder，旧评测使用 crop image + `USER_PROMPT_V2`；初始 prompt
> 相同不等于 post-tool continuation matched。旧 Crop 权重不能通过重测修复，既有 Macro*、
> sub-benchmark 和调用统计均只作历史测量。修复 commit
> `ecddc379d392d154c91783d7651528b20d40afba` 与新 PRL27-A 配置/binder commit
> `122db51e5a865878fd482be70dd99cfb49608d71` 已上传，corrected fresh-S0 S32@512 尚未启动。
> No-Tool、TGVF、Atomic 不受
> 该 Crop-only renderer bug 直接影响。

## 1. 结论先行

PRL25 的 Crop、TGVF、Atomic 和 No-Tool RL 训练均使用
`image_max_pixels = 1,003,520`。`16,777,216 = 4096²` 是 Qwen3 fast image processor
保存的默认 `size.longest_edge`，不是这批 RL 训练允许的最大面积。

这里的“RL 训练”专指 PRL25 policy RL。TGVF/Atomic 所加载的 RP67 representation adapter
预训练使用 `image_max_pixels = 262,144 = 512²`；因此不能把 adapter 预训练与后续 policy RL
混称为同一个分辨率合同。两阶段都不是 `16,777,216`。

评测侧曾有两条独立缺陷：旧 Crop official-visible evaluator 与旧 No-Tool matched evaluator
均把像素覆盖值放在 processor 不读取的 flat kwarg 中，因而实际回退到 `16,777,216`。所以旧
Crop fixed-boundary S32/S80 与旧 No-Tool S0/S8/S16/S32 都不是 true-1M 结果。TGVF 与 Atomic
使用另一条已验证的 nested preprocessing 路径，不受该缺陷影响。Crop 已在顶层
`mm_processor_kwargs.size` 和 fixed action boundary 下重跑闭合；No-Tool 四个冻结 step 也已用
有效顶层像素覆盖值重跑并通过正式 aggregate receipt 验收。

### Unified true-1M stage table（Crop row quarantined）

下表是当前统一输入预算下的五方端到端结果：五行评测都使用 `1,003,520`、同一
`2,240` 条支持集、七项统计与 scorer；No-Tool、Crop、TGVF 和 Atomic 的 policy RL 训练也使用
`1,003,520`。No-Tool 以事前冻结的 S32 进入主表；Crop S32 仅以 historical row 保留，等待
corrected PRL27-A 替换。No-Tool S16 即使数值略高，也不作 post-hoc checkpoint 替换。单位为 `%`。

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct | 61.3147 | 72.7749 | 66.5000 | 63.8889 | **59.7877** | 37.9182 | **75.3333** | 53.0000 |
| No-Tool RL S32 | **63.7520** | 70.1571 | 65.5000 | **71.1111** | 49.3928 | **53.9033** | 75.0000 | **61.2000** |
| Crop S32† | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| TGVF S64 | 59.8086 | **74.3455** | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | **73.5000** | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |

† 像素、支持集和 scorer 合同正确，但训练/评测 post-tool continuation 不一致；只作历史测量。

No-Tool S32 在 Macro*、BLINK-180、MMMU-269 与 MathVerse 上最高；Original 在 OCR mean 与
MathVista 最高；TGVF 在 VStar 最高；Atomic 在 HR 最高。Macro* 排序为 No-Tool S32 `63.7520` >
Atomic `63.0827` > Original `61.3147` > Crop S32 `61.0706` > TGVF `59.8086`，相邻差值依次为
`0.6693 / 1.7680 / 0.2442 / 1.2619 pp`。排序只描述表值，Crop 的相对位置不具 aligned 含义。
这是统一输入预算、scorer 和 sample reference 下的端到端测量，不是严格单变量因果消融：
Original 是 raw-direct；No-Tool 使用训练匹配的 no-tool
prompt 与 direct-only loop；三条工具 RL 方法之间的 prompt、工具协议与 checkpoint step 也不同。
RP67 adapter 预训练仍为 `512²`，但加载它的 TGVF/Atomic policy RL 及本表评测均为 true-1M。

## 2. 固定术语

| 术语 | 唯一定义 |
|---|---|
| **true-1M** | `max_pixels = 1,003,520`；保持长宽比的最大表示面积，不是固定正方形 resize |
| **true-512** | `max_pixels = 262,144 = 512²` |
| **processor-default cap (≤16.7M)** | `size.longest_edge = 16,777,216 = 4096²`；它是失效历史运行的上限，不表示每张图都被放大到 16.7M |
| **Original raw-direct true-1M** | 原始 Qwen3-VL-8B-Instruct，官方 raw prompt，无 system prompt、无工具，seed42，仅把历史 direct 的像素上限改为 `1,003,520` |
| **No-Tool RL matched true-1M** | PRL25-F 的训练匹配 user-only no-tool prompt 与 direct-only loop，S0/S8/S16/S32 |
| **Crop fixed-boundary true-1M historical** | PRL25-B native RGB Crop，严格 `</tool_call>` action boundary，但 post-tool continuation mismatch，S32/S80 |
| **TGVF matched true-1M** | PRL25-C Pure TGVF、Frozen RP67，S64 |
| **Atomic (TGVF+Crop) matched true-1M** | PRL25-D Atomic Crop+TGVF、Frozen RP67，S16 |
| **Macro\*** | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均的七项无权平均 |

## 3. 统一主表合同

统一主表固定使用 CoreDev-2511、seed42、temperature 1、相同七项官方 scorer、BLINK 180
单图 coverage、MMMU 269 单图 coverage 和相同 Macro* 聚合。每行必须同时满足：

1. 配置声明 `max_pixels = 1,003,520`；
2. prompt expansion 与 vLLM decode 都必须使 Qwen3 的有效
   `size.longest_edge = 1,003,520`：本地 HF/训练 preprocessing 可使用
   `images_kwargs.size.longest_edge`；vLLM 0.12 必须使用可哈希的
   `mm_processor_kwargs.size.longest_edge`；direct VLMEvalKit 使用等价的逐图 `max_pixels` 绑定；
3. 对至少一张大于上限的真实 processor probe，保存 source area、`image_grid_thw`、visual
   token count 与 represented area，并验证 represented area 不超过 `1,003,520`；
4. `2,240/2,240` 个支持的单图任务完成推理，七个官方 slice 全部评分闭合；
5. accepted summary、coverage view、evaluation identity 与 completion receipt 均可追溯。

像素预算统一只消除输入面积这一项混杂。Crop 另有 post-tool continuation mismatch；Original、
No-Tool、Crop、TGVF 与 Atomic 的 prompt、agent loop、工具 schema 和可调用工具也仍不同。因此
该表是**统一输入预算下的端到端测量**，不是严格单变量方法消融。

## 4. 当前主表状态

| 主表行 | checkpoint | true-1M 状态 | 当前 Macro* |
|---|---|---|---:|
| Original raw-direct | base Qwen3-VL-8B-Instruct | `accepted; raw-direct reference` | 61.3147 |
| No-Tool RL | S32 | `accepted; frozen headline` | 63.7520 |
| Crop | S32 | `historical; continuation-mismatched` | 61.0706 |
| Crop | S80 | `historical; continuation-mismatched` | 59.6463 |
| TGVF | S64 | `accepted` | 59.8086 |
| Atomic (TGVF+Crop) | S16 | `accepted` | 63.0827 |

No-Tool S0/S8/S16/S32 已全部补测；S32 是统一主表行，S0 是训练前控制行，S8/S16 只用于描述
优化动态。

当前 accepted true-1M 结果如下，单位为 `%`：

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct | 61.3147 | 72.7749 | 66.5000 | 63.8889 | 59.7877 | 37.9182 | 75.3333 | 53.0000 |
| No-Tool RL S32 | 63.7520 | 70.1571 | 65.5000 | 71.1111 | 49.3928 | 53.9033 | 75.0000 | 61.2000 |
| Crop S32† | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| Crop S80† | 59.6463 | 73.8220 | 69.5000 | 59.4444 | 54.0426 | 44.9814 | 63.3333 | 52.4000 |
| TGVF S64 | 59.8086 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | 73.5000 | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |

† Crop 行完成了 true-1M 像素测量，但不是 train/eval-aligned result。

Original completion receipt 与 accepted summary 分别为
`artifacts/evaluation/PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1/runtime/scoring-supervisor/original-true1m-scoring-complete.json`
和同一 evaluation 根目录下的 `scoring/coredev-2511-eval-summary.json`。completion 状态为
`complete`，summary 状态为 `pass`；两者固定 `max_pixels=1,003,520`、`sample_count=2,511`、
`slice_count=7`，summary 的 judge parse failure 为 `0`，summary SHA256 为
`f8dc31b5353c36d2e764096ee2f2a1f0da0ca3d28fb4525c2fb660829c705904`。
Original 的 OCR EN/CN 分别为 `59.0441/60.5313`，MMMU-269 为 `102/269`；MathVerse 五版本
Text Dominant / Text Lite / Vision Dominant / Vision Intensive / Vision Only 分别为
`70/50/54/57/34`，宏平均 `53.0000`。

### 4.1 No-Tool true-1M 学习动态与行为审计

四个 step 使用同一 matched no-tool prompt、direct-only loop、true-1M processor、任务与 scorer。
S32 是事前冻结的唯一 headline；粗体只标示描述性峰值，不改变 checkpoint 选择。

| Step | Macro* | Δ vs S0 | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse | Judge parse failure |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 63.7619 | — | **72.2513** | 64.0000 | 66.6667 | 48.5015 | 54.6468 | **77.6667** | 62.6000 | 0 |
| S8 | 63.7354 | −0.0264 | 71.2042 | 66.0000 | 66.1111 | 48.8990 | **56.1338** | 75.0000 | **62.8000** | 0 |
| S16 exploratory | **63.9307** | **+0.1688** | 71.7277 | **68.0000** | 69.4444 | 47.6957 | 54.6468 | 76.0000 | 60.0000 | 1 |
| **S32 frozen** | 63.7520 | −0.0098 | 70.1571 | 65.5000 | **71.1111** | **49.3928** | 53.9033 | 75.0000 | 61.2000 | 2 |

S32−S0 为 `−0.0098 pp`，在当前单次 seed42 评测中应描述为 aggregate-flat，而不是 RL 带来
总体提升。S16 的探索性峰值仅比 S0 高 `0.1688 pp`，且不得用于事后替换冻结 S32。RL 过程中
BLINK 提高而 VStar、MMMU 与 MathVista 发生不同程度回落，说明总体近似不变掩盖了组件重分配。

下表从四个 accepted rank JSONL 的 2,240 条支持集直接统计。P50/P90/P95 使用 empirical
order-statistic 口径；空 final 包含达到 token 上限或 invalid-format 后没有可评分 final 的行。

| Step | Sampled tokens mean | P50 | P90 | P95 | `max_tokens` rows | `invalid_format` | Empty final | Tool calls / errors / observations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1005.88 | 171 | 3068 | 5443 | 3 | 3 | 6 | 0 / 0 / 0 |
| S8 | 1024.41 | 163 | 3270 | 5808 | 4 | 0 | 4 | 0 / 0 / 0 |
| S16 | 944.06 | 145 | 2822 | 5416 | 4 | 0 | 4 | 0 / 0 / 0 |
| S32 | 839.73 | 127 | 2065 | 5065 | 5 | 0 | 5 | 0 / 0 / 0 |

S32 的 mean/P50/P90 均低于 S0，且四个 step 的工具调用、工具错误与 observation 全部为零。
这证明 no-tool 行没有执行视觉工具；长度收缩与分项变化是行为相关观测，不能据此建立因果解释。

正式 aggregate scoring receipt 为
`artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2/runtime/scoring-supervisor/matched-scoring-complete.json`。
其 `status=complete`、identity 为
`69328dfe889366650e96a0582eaa86a27df1ac349751ee40493d83f47c92a955`，文件 SHA256 为
`0fc8bcf3865c4d6ee360206da318c7913afb8c4cf659ea4bcb29104ad132db59`，并固定每步
`max_pixels=1,003,520`、2,511 rows 与 7 slices。S0/S8/S16/S32 summary SHA256 依次为
`37c7b376cfc53dcd9b3d47101348fcb25fcb05c74f9a7d6d446ae4219edf0bbe`、
`b28d941746592ef80cdd8157bf077c058bf1253a76cf1ccd17e4bc8af5bc9223`、
`13bf456370d23540c8130b855f30adaae8b4b6c21f24dabcd9a37ff058390d72` 与
`bd6057d798c37791714033c4f9a734f435264ec1242fc841fdaad4fd064d897a`。

闭合前的 aggregate finalizer 一致性检查暴露出一个 telemetry 边界偏差：worker 在生成
`result_identity_sha256` 后才附加 `wall_seconds`，而旧 finalizer 重算时错误地纳入该字段。修复后
四步共 `8,960/8,960` 行语义 hash 全部通过；rank-file SHA/size 仍绑定包括 `wall_seconds` 在内的
精确字节。因此这是一项验收器修复，不是推理或评分实验失败，也不需要重跑推理或 scorer。

### 4.2 Historical Crop true-1M 工具行为与边界审计

以下调用统计只描述 continuation-mismatched 的旧 policy，不得用于证明 Crop utility 或伤害。

| Crop checkpoint | 至少一次成功工具调用的题数 | 成功调用率 | 实际调用次数 | `invalid_crop` | `tool_call_cap` |
|---|---:|---:|---:|---:|---:|
| S32 | 1,423 / 2,240 | 63.53% | 1,769 | 92 | 11 |
| S80 | 2,006 / 2,240 | 89.55% | 2,043 | 102 | 0 |

fixed action boundary 使合法 `</tool_call>` 在当轮终止，评分不再接受同轮工具请求后的 plain
final。结构审计另外发现 S80 中 **1 个**样例在一个已闭合调用后又生成了未闭合的第二个
`<tool_call>` opener。该次未被解析或执行为合法调用，也未重现旧的 answer-over-action 接受路径；
它仍是一个需要保留的格式边界，因此不将 S80 描述为“所有工具语法都完全闭合”。

可追溯审计键如下：

| Crop checkpoint | true-1M audit receipt identity SHA256 | `paired-summary.json` SHA256 |
|---|---|---|
| S32 | `0109f7f4f602106bf71bca50309019ae248962387dd7aa33f2b8e12c65042581` | `a3aa1befedb509e48b69edf4822d7e0467bca346b866da6fd78378d3265ea87d` |
| S80 | `fac02bd313a2c9786b93f7927dccb25403f7b6421aba3256e21d40a97757bbe9` | `715ae0aca71532804942ef5f301afb31c99080d8ec78768b9a10d2be8f448ae5` |

## 5. true-512 历史表与更正

后续首轮视觉 token audit 证明 Crop boundary-fix V2 S80 的 processor override 未生效：
`1275/2240` 条超过 true-512 的 256-token 上界。因此该 Crop 行不是有效 true-512，且其训练还受
continuation mismatch 影响；本节保留旧数值只为追溯。Original、TGVF 与 Atomic 的 `262,144`
测量仍有效。单位为 `%`。

| Method / effective contract | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct@512 | 55.3556 | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| Crop S80 processor-default historical（旧称 @512） | 62.0967 | 86.9110 | 65.5000 | 64.4444 | 54.4299 | 45.7249 | 65.6667 | 52.0000 |
| TGVF S64@512 | 55.4067 | 53.4031 | 60.5000 | 62.7778 | 40.7642 | 46.4684 | 72.3333 | 51.6000 |
| Atomic S16@512 | 57.2762 | 57.0681 | 59.5000 | 61.6667 | 47.7713 | **48.3271** | 71.0000 | **55.6000** |

该表不能再称为四方同像素对照；有效的 Original/TGVF/Atomic 行仍跨 prompt、工具 schema 和
agent loop。Crop 行既不满足 true-512 像素合同，也不满足 train/eval continuation parity。

### 5.1 Method-specific resolution response

| Method | true-1M Macro* | true-512 Macro* | Δ (512 − 1M) | 可解释性 |
|---|---:|---:|---:|---|
| Crop S80 | 59.6463 | 62.0967 | `invalid` | nominal-512 override 未生效，且 continuation mismatch；不得解释 resolution response |
| TGVF S64 | 59.8086 | 55.4067 | **−4.4019 pp** | same run/step/frozen config/RP67/prompt/tool/task/RNG/scorer；snapshot semantic equality 未闭合 |
| Atomic S16 | 63.0827 | 57.2762 | **−5.8065 pp** | same run/step/frozen config/RP67/prompt/tool/task/RNG/scorer；snapshot semantic equality 未闭合 |

TGVF 和 Atomic 的高度匹配结果支持一个限定观测：当评测输入从 policy RL 的 1M rollout 尺度
偏离到 512 时，两个 TGVF-family checkpoint 都出现 Macro* 回落。这些证据 **consistent with**
TGVF-family 依赖 policy-RL 训练/评测尺度对齐，但不构成因果证明。RP67 adapter 预训练本身是
`512²`，因此这里的“alignment”严格特指 policy rollout/evaluation distribution，不是 adapter
pretraining resolution。两个像素臂虽来自同 run/optimizer step 并绑定同一 frozen policy config SHA 与
RP67 state，Qwen snapshot 却分别 materialize，tree/combined-weight byte SHA 不同；现有 receipt 没有
闭合 tensor-semantic equality proof。更强的单变量控制应复用同一 snapshot 或提供 semantic tensor hash，
并以 512-only 或 multi-scale TGVF/Atomic 训练 × 512/1M 评测做完整 factorial 交叉。

Crop 的旧 `+2.4504 pp` resolution-response 结论撤回：nominal-512 arm 实际使用 processor-default
上限，且两臂都来自 continuation-mismatched 训练。corrected Crop 的 resolution response 需等待
新训练合同下的明确实验。

## 6. 必须降级的历史结果

下列数值只保留为 `processor-default-cap historical`，不得再放入 true-1M 主表，也不得与 true-512
构成像素消融：

| 历史运行 | Macro* | 失效原因 |
|---|---:|---|
| Crop fixed-boundary S32 | 61.6699 | flat processor override 被忽略 |
| Crop fixed-boundary S80 | 59.1785 | flat processor override 被忽略 |
| No-Tool S0 | 64.4712 | flat processor override 被忽略 |
| No-Tool S8 | 66.1132 | flat processor override 被忽略 |
| No-Tool S16 | 66.9028 | flat processor override 被忽略 |
| No-Tool S32 | 66.6853 | flat processor override 被忽略 |

真实 Qwen3 processor 对 `2048×1536` 图像的审计结果为：

| 传参方式 | represented pixels | 结论 |
|---|---:|---|
| processor default | 3,145,728 | 未受 1M 限制 |
| flat `max_pixels=1,003,520` | 3,145,728 | 覆盖值被忽略 |
| direct HF：`images_kwargs.size.longest_edge=1,003,520` | 995,328 | 有效 true-1M；训练 preprocessing 使用 |
| vLLM 0.12：顶层 `mm_processor_kwargs.size.longest_edge=1,003,520` | 995,328 | 有效且 cache 可哈希；评测 runtime 使用 |

vLLM 0.12 不能直接使用 nested `mm_processor_kwargs.images_kwargs.size`：它虽然对裸 HF
processor 有效，但会在 vLLM processor cache 中触发 `unhashable dict`。本轮 Crop 首次启动在
第一条请求前即 fail-closed，未产生预测；现已改为顶层 `size`，须以重新启动后的 completion
receipt 为准。

历史 Original raw-direct 的 `55.3556` 使用有效 `262,144` 上限，是第 5 节 true-512 四方表的
Original 行，但不是统一 true-1M 表中的 Original 行。No-Tool S32 raw-direct@512 的 `54.3543`
同理只属于 512 合同。

## 7. 训练像素审计

训练链分为两层：

| 训练层 | 适用对象 | 生效像素上限 |
|---|---|---:|
| representation adapter 预训练 | RP67（随后供 TGVF/Atomic 加载） | `262,144 = 512²` |
| policy RL | PRL25-B/C/D/F | `1,003,520` |

RP67 的最终配置
`configs/representation/experiments/image_axis_grounding/rp67_qwen3_instruct_image_axis_grounded_2000_gpu01_finalize_step2000.toml`
（SHA256 `cbbfc146…`）绑定 `image_max_pixels = 262,144`；其 `metrics.jsonl` durable
completion 记录也保存同一值（run identity `0b53d04c…`，global step `2000`，adapter
SHA256 `13332865…`）。用历史 representation preprocessing 对 `2048×1536` RGB 复跑，得到
`image_grid_thw=[1,26,36]`、represented area `239,616`、merged visual tokens `234`，确认
RP67 adapter 预训练为 true-512。

PRL25-B/C/D/F 的冻结 run config 均绑定 `image_max_pixels = 1,003,520`。进一步按各自最终
launch provenance commit（B `08a9d8b4`、C `b87126ae`、D `017b5077`、F `7645fe4a`）回查
历史源码：在线 rollout 的 source image，以及 Crop/Atomic 的 crop observation，均调用
`preprocess_qwen3_rgb()`；该函数通过 Hugging Face 实际识别的
`images_kwargs.size.longest_edge=image_max_pixels` 先生成受限的
`pixel_values/image_grid_thw`，再把预展开视觉张量送入 vLLM。dataset prompt expansion 使用
同一有效绑定。

用训练模型的真实 Qwen3 processor 对 `2048×1536` RGB 复跑上述历史函数：processor 保存的
默认上限虽为 `16,777,216`，实际得到 `image_grid_thw=[1,54,72]`、represented area
`995,328`、merged visual tokens `972`，均满足 `1,003,520` 上限。因此这些 checkpoint 的
训练输入是 true-1M；`16,777,216` 仅是 override 缺失或失效时的 processor 默认值。

像素问题与 action-boundary 问题分开处理：Crop true-1M 重测同时要求 fixed
`</tool_call>` boundary；像素修复本身不改变 TGVF/Atomic 的 action boundary 结论。

## 8. 执行队列

- [x] 审计训练侧真实像素路径；确认 PRL25 RL 为 true-1M。
- [x] 撤销 Crop 与 No-Tool 旧 nominal-1M 身份。
- [x] 保留 TGVF S64 与 Atomic S16 accepted true-1M 结果。
- [x] Crop S32/S80：fixed boundary、有效顶层 `mm_processor_kwargs.size` cap、独立
  compile cache、`2,240/2,240` 推理、七项评分与 true-1M audit receipt 均已闭合；结果已因
  continuation mismatch 降级为 historical。
- [x] No-Tool S0/S8/S16/S32：顶层 `mm_processor_kwargs.size`、独立 compile cache、
  `4 × 2,240` 条支持集推理、七项评分与 aggregate receipt 均已闭合。
- [x] Original：raw-direct true-1M 的真实 processor probe、2,511 行推理、七项官方评分与
  completion receipt 均已闭合；Macro* `61.3147`，judge parse failure `0`。
- [x] 回填 Original 主表与 59-slice sub-benchmark；Crop 工具行为已回填。
- [x] 生成含 frozen No-Tool S32 RL-only 控制的最终完整主表。
- [x] 加入 No-Tool S32 后机械重算 59-slice 附录、winner 与 pairwise W/L/T；六方法结果见
  workshop 报告 Appendix A。
- [x] 把 Crop true-1M 历史测量同步到 NeurIPS workshop 报告并显式撤销 aligned-golden 身份。
- [x] 修复 Crop continuation runtime，配置全新 PRL27-A run/eval identity；尚未启动重训重测。

## 9. 文章 claim 边界

当前统一表的测量值排序是 No-Tool S32 `63.7520` > Atomic S16 `63.0827` > Original raw-direct
`61.3147` > historical Crop S32 `61.0706` > TGVF S64 `59.8086`。五行共享 true-1M、scorer 与
sample reference，但 Crop continuation 不匹配，prompt、agent contract 与 checkpoint step 也不
统一；因此 No-Tool 领先 Atomic/TGVF 的 `0.6693/3.9434 pp` 只是端到端合同差值，对 Crop 的
`2.6814 pp` 还只是历史测量差。它们都不能写成“关闭工具带来的纯因果增益”。同一 No-Tool
合同内的 S32−S0 仅为 `−0.0098 pp`，不支持 32-step RL 带来总体
提升；S16 的 `+0.1688 pp` 也只是探索性中间点，不触发 post-hoc checkpoint 重选。工具方法的
论文价值因此必须建立在特定能力切片、RP67 matched utility、工具行为与稳健性证据上，而不是
总体 Macro* 支配。Atomic 是否进入正文主线，仍由 target-only 稳健性、无偏 target 合格率与
统一 true-1M 主表共同决定。
