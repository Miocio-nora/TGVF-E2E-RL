# PRL24：BS64 Policy-RL Scaling 系列计划

日期：2026-08-16

状态：`PAUSED / A、C 已完成；B 在 S8 有记录停止；D 在 S1 后停止；E/F 未启动`

Decision ID：`POLICY-RL-BS64-SCALE-SERIES-20260816-v2`

执行修订 ID：`POLICY-RL-BS64-SCALE-SERIES-20260820-FMT2-v1`

暂停决策 ID：`POLICY-RL-BS64-PAUSE-TO-PHASE3-BS16-S80-20260820-v1`

本计划承接已经收官的
[BS16 small-batch pilot](POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md)，并使用
[CoreDev-2511 统一测量合同](POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md)
作为正式外部评测标准。Teacher 数据沿用
[PRL22 Teacher25 结论](PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md)。

## 0. 2026-08-20 执行修订（覆盖后续运行口径）

本文件最初是预注册计划；以下修订记录实际执行中已经发生的 recipe 决策与资源分配，
不得再用旧段落推断当前运行状态：

1. 最初的 FMT1 对 protocol/format/tool error 罚 `-1`。训练中观察到 format error
   随步数继续升高，因此新建 FMT2 并把同一错误罚分改为 `-2`。旧 FMT1 A 保留为
   独立历史实验，不与 FMT2 checkpoint 或结果混合。A/B/C 的正式横向表以及从 D
   开始的所有后续 PRL24 训练统一使用 FMT2；若再次改变罚分，必须另立实验身份。
2. A0 只服务于“严格 batch-only 因果效应”措辞，不是主序列必跑项。历史 BS16 与
   当前 FMT2/执行 commit 不完全 matched；未补 A0 时可以报告 recipe-level scaling
   evidence，但不能声称差异只由 batch 引起。可选的 A0 应是 same-commit FMT2 BS16，
   且不阻塞 D/E/F。
3. B 的 FMT2 S8 Macro* 比 matched A S8 低 `3.2514 pp`，6/7 分量下降，且方向与
   上一轮 BS16 Joint pilot 一致。因此 B 在 S8 有意停止，结论为当前保留 Frozen
   Adapter；没有把 B 伪装成已完成 S16，也不再为补 B-S16 占用 D 的前序资源。
4. C 已完成 S12/S16 训练与外评。C−A 在 S4/S8/S12/S16 为
   `+1.7794/-0.1312/+0.3053/+0.0917 pp`：早期有正信号，但无持续 endpoint
   accuracy 增益，F/G 暂不成为默认 reward。
5. D 使用 native Crop、Teacher25、BS64 × n16、world8、full Qwen、LR `1e-6`、
   FMT2 启动，但实际只完成 S1 optimizer checkpoint，并在 S2 update 前按资源决策停止。
   S1 从 rollout 开始到 checkpoint 完整落盘约 `2 h 32 min`，其中 actor update 约
   `2 h 22 min`；完整 checkpoint 约 `140.3 GB`。停止发生在最终 step metrics/trajectory
   publication 前，因此 D-S1 不是 efficacy 结果，不能声称 Crop 已改善或退化。
6. 现有 pure-TGVF recipe-level scaling evidence 没有显示明确 BS64 质量增益：等 exposure
   的 BS64-S4 相对历史 BS16-S16 为 `-1.5607 pp`；BS64 最佳点只高 `+0.7758 pp`，
   BS64-S16 低 `-0.5583 pp`。由于 FMT/commit 不完全 matched，这不是严格 batch-only
   因果结论，但不足以支持继续把 GPU 优先投入 BS64。
7. PRL24 从本修订起暂停。E/F 未启动，A0 不再是近期优先项；后续统一转入
   [第三期 PRL25 BS16 Teacher25 80-step 计划](PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md)，
   五个 arm 全部从 S0 新训，不续接任何 PRL24 checkpoint。

## 1. 执行摘要

下一阶段不再按 `BS16 → BS32 → BS64` 逐级试探，而是直接进行第一个四倍 prompt-batch
放大：

```text
BS16 × n16 = 256 trajectories / optimizer update
                         ↓ 4× independent prompts
BS64 × n16 = 1,024 trajectories / optimizer update
```

本系列要回答的不是“大 batch 理论上是否更平滑”，而是以下四个可被实验否证的问题：

1. **BS64 是否能把 BS16 下观察到的非单调、后期平台或回落，转化成更稳定且更强的
   外部能力提升？**
2. **BS16 下 Joint/Unfrozen RP67 变差，究竟是 Adapter 不应参与 RL，还是小 batch
   梯度噪声过大造成的破坏？**
3. **Focus/Target 与 Grounding visual reward 在更大 batch 下，能否稳定改善
   foveation、降低幻觉，并最终转化为外部准确率，而不是与 answer reward 发生替代？**
4. **TGVF、Crop、Atomic Crop+TGVF 在 BS64 下各自能否继续增强；旧的
   answer-gated conditional Crop reward 是否比当前 T-free Crop reward 更适合大 batch？**

本段以下内容保留为 2026-08-16 的原始预注册逻辑。2026-08-20 的当前决策已经改为：
**暂停 BS64，不进入 BS128/BS256；先回到 BS16，把统一 Teacher25/FMT2 配方从 S0
训练至 S80。** 训练 reward、format 合法率或 W&B 曲线变平滑本身都不能证明“大 BS
更强”；正式判断仍必须由统一外部 benchmark 和相应的视觉健康度审计共同给出。

## 2. 为什么需要新的 scaling 系列

### 2.1 BS16 已经证明“可学”，但没有证明“已经充分优化”

上一阶段固定为 `16 independent prompts × 16 rollouts`。`n16` 能改善同一道题内部的
相对 advantage 估计，却不能替代更多独立问题所提供的梯度方向多样性。已有结果不是
“所有线路都平台”，而是多条关键线路出现了不同形式的非单调、平台或后期回落：

| BS16 线路 | Step 8 Macro* | Step 16 Macro* | S16 − S8 | 观察 |
|---|---:|---:|---:|---|
| Crop clean-final，conditional reward | 59.7161 | 59.5502 | -0.1659 | 基本平台 |
| Crop T-free | 61.1032 | 61.0862 | -0.0170 | 外评平台，后期工具使用显著下降 |
| RP67 Frozen + Focus/Grounding | 57.8849 | 57.5422 | -0.3427 | 后期回落 |
| RP67 Frozen Crop+TGVF，无 Teacher | 62.1168 | 60.9539 | -1.1629 | 明显回落 |
| RP67 Frozen TGVF，无 Teacher | 56.1964 | 58.1996 | +2.0032 | 反例：后半程仍提升 |
| RP67 Frozen TGVF，Teacher25 | 58.4655 | 59.9590 | +1.4935 | 反例：后半程仍提升 |

因此，“BS16 的独立 prompt 太少，更新噪声限制了后期收益”是目前最有力的工作假设，
但还不是因果结论。PRL24 的首要任务就是用 matched larger-batch ablation 检验它。

### 2.2 DeepEyes scale 说明仍存在明显 batch 维度

当前 pilot 每次更新使用 256 条 trajectories；DeepEyes 的公开规模是
`256 prompts × n16 = 4,096 trajectories/update`。PRL24 的 BS64 为 1,024 条，仍只有
该参照规模的四分之一。因此 BS64 是一个可承受、可继续放大的中间 scale gate：它足以
显著增加独立 prompt diversity，同时不会把第一次验证直接扩成不可承受的大型训练。

### 2.3 Teacher25 固定为本系列默认数据

PRL22 中，Teacher25 在纯 TGVF 与 Atomic Crop+TGVF 的 Step 8/16 四个 endpoint 上均
为正向。当前选择 25%，不是因为 50%/100% 已被证明无效，而是因为 Teacher25 已有跨
两条工具线的积极证据，同时更保守地保留原 T1 分布。

PRL24 不再把 Teacher 比例作为实验变量。所有 arm 固定为：

```text
Teacher25 schedule = 20,480 prompts, seed 42, no replacement
Mixture            = 75% retained T1 + 25% retained Stage1 teacher
BS64 composition   = 48 retained T1 + 16 teacher prompts
```

每个 BS64 batch 必须由 canonical schedule 中连续四个 BS16 slice 拼接而成，保持顺序、
来源和 provenance 不变。这样可以建立严格的 prompt-exposure 对齐，而不是重新随机抽取
一份“近似 25%”数据。

## 3. 预注册假设

### H1：更大 prompt batch 改善稳定性与外部能力

在 LR、n、数据、模型、reward 和 optimizer 均不变时，BS64 应降低跨问题梯度方向的
方差。若该解释正确，BS64 在等暴露量 endpoint 上应优于或至少不弱于 BS16，并且在
继续暴露后避免明显平台或回落。

反例条件包括：

- 只有 W&B reward/gradient 曲线变平滑，CoreDev-2511 不升；
- 等暴露量外评与 BS16 的差异落在已知采样波动内；
- 外评上升只由单一 benchmark 拉动，其他能力明显坍塌；
- tool call、输出长度或 answer-channel shortcut 出现新的病态行为。

### H2：大 batch 可能使 Joint RP67 变得可行

BS16 paired control 中，Frozen 与 Joint 在 Step 8 几乎相同，但 Frozen Step 16 比
Joint 高 `1.6713 pp`；Joint Step 16 还低于共同 Step 0。这只证明当前 BS16 下应冻结，
不能外推到 BS64。

BS64 下只改变 Adapter 是否接收 policy-RL gradient。Joint 只有同时满足以下条件才可
晋级：

1. 外部 Macro* 不弱于 Frozen；
2. `D` 的视觉健康度没有下降；
3. 不再出现 norm 增长、same-target wrong-image 高分或 answer-channel shortcut 等
   representation pathology。

### H3：大 batch 可能让 F/G visual reward 的 credit 更可靠

BS16 下，F/G 相对 no-F/G 的 treatment difference 为：

| endpoint | F/G − no-F/G Macro* |
|---|---:|
| Step 8 | +1.6885 |
| Step 16 | -0.6573 |

同时，人工样本检查显示看图 reasoning 更具体、更贴近图像。这意味着 F/G 既不能被
判定为无效，也不能仅凭 judge reward 上升宣布成功。BS64 将检验：更多独立问题是否能
减少 visual judge credit 的噪声，并避免 `F+G` 与 answer correctness 互相替代。现有
`F+G` 的最大正向幅度为 2，与正确答案主体的 reward 处于相同量级，因此这种 reward
substitution 风险必须显式检查。

### H4：conditional Crop reward 可能更需要大 batch

当前 Crop T-free 主要奖励答案正确，并惩罚重复调用和工具/协议错误。它在 BS16 的
Step 8→16 外评基本持平，但 tool-call rate 从 `0.7422` 降至 `0.3008`，crop success
从 `0.7383` 降至 `0.2930`。这表明 policy 可能在保持答案通道的同时逐渐绕开工具。

旧 DeepEyes-style conditional reward 只在答案正确时奖励成功 Crop，因此不会给
“错误但频繁调用工具”正奖励。需要在**当前相同 Crop prompt、runtime、数据和代码**上
重做 reward A/B，判断 BS64 的 group diversity 是否能让这种 answer-gated tool credit
更稳定。这里只移植 reward 公式，不复活旧 prompt、旧 `<answer>` dialect 或旧 runtime。

## 4. 固定训练合同

除每个 arm 明确列出的单一变量外，以下设置全部固定：

| 字段 | PRL24 固定值 |
|---|---|
| base policy | `Qwen3-VL-8B-Instruct` |
| policy update | full Qwen：vision encoder + merger + language model |
| Stage1 representation | A/B/C/F：RP67 Step 2000；D/E native Crop：N/A |
| default Adapter mode | A/C/F：Frozen；B：Joint/Unfrozen；D/E：N/A |
| data | canonical Teacher25 schedule |
| global prompt batch | 64 independent prompt groups |
| rollout count | n16 / prompt |
| trajectories / update | 1,024 |
| distributed | world8，prompt micro2/rank，GA4，FSDP2 |
| optimizer | AdamW，weight decay `0.01` |
| learning rate | constant `1e-6`，不 warmup |
| PPO epochs / KL | 1 / 0 |
| gradient clipping | `1.0` |
| current format/protocol penalty | FMT2：发生错误时 `-2`；FMT1 `-1` 仅作历史身份 |
| rollout sampling | temperature 1 |
| final dialect | plain final；不使用 `<answer>...</answer>` |
| tool-call / response cap | 所有工具线路最多 6 次调用；response cap 20,480 tokens |
| evaluation endpoints | matched diagnostic：Step 2、4；intermediate：Step 8、12；primary final：Step 16 |
| formal run length | 默认训练至 Step 16；B 是有记录的 S8 资源例外 |

关键原则：**第一轮不随 batch 线性放大 LR。** 否则观察到的差异无法归因于 batch。
`world8 / micro2 / GA4` 的数学含义是每卡每个 micro-block 处理 2 个 prompt groups，8 卡
合计 16 prompts，累计 4 个 micro-block 后进行一次包含 64 prompts 的 optimizer update。
loss 必须按完整 global prompt batch 正确归一化，不能因 GA4 把有效梯度放大四倍。

### 4.1 低成本实现检查

BS64 正式训练前只做能保护科学比较所必需的检查：

1. 配置解析与 schedule composition 单元测试；
2. CPU 上核对 GA4 loss normalization；
3. 一个低成本 GPU functional smoke，验证 rollout、reward、反传、optimizer step、保存和
   resume；
4. smoke 不上传 W&B，不进入正式结果；
5. 正式任务使用 `tmux`，自动保存、自动接力评测，并在失败时保留可恢复状态。

这些检查应是活的配置记录与快速 preflight，不引入逐文件哈希扫描或与科学变量无关的
fail-closed 审计。如果 BS64 需要修改训练数学/执行代码而不是纯 config 与 schedule
overlay，则已有 PRL22-A 只能作为 historical anchor，需补一个 same-commit BS16
control；若代码路径和训练数学完全不变，则不重复消耗资源重跑 BS16。

## 5. PRL24 实验矩阵

| ID | Tool line | Adapter | Reward | 相对 control 的唯一变量 | 主要问题 |
|---|---|---|---|---|---|
| **PRL24-A** | pure TGVF | Frozen RP67 | FMT2 T-free | BS16 → BS64 | 大 batch 是否更强、更稳定 |
| **PRL24-B** | pure TGVF | Joint/Unfrozen RP67 | FMT2 T-free | 仅解冻 Adapter | 大 batch 下 Joint 是否仍破坏表示 |
| **PRL24-C** | pure TGVF | Frozen RP67 | FMT2 T-free + F + G | 仅开启 F/G | visual reward 是否转化为健康 foveation 与外评 |
| **PRL24-D** | native Crop | — | FMT2 T-free | 建立 Teacher25 BS64 Crop control | 当前 Crop 线路在大 batch 下的表现 |
| **PRL24-E** | native Crop | — | FMT2 conditional Crop | 相对 D 只换 scalar reward | DeepEyes-style conditional credit 是否更合适 |
| **PRL24-F** | Atomic Crop+TGVF | Frozen RP67 | FMT2 T-free | 相对 A 改成组合工具协议 | 组合工具在大 batch 下能否持续增强 |

比较边界如下：

- 历史 PRL22-A BS16 Teacher25 可作 recipe-level anchor，但因当前采用 FMT2 且执行
  commit 不完全相同，不能直接称为严格 batch effect；仅在确需 batch-only 因果措辞时
  补 **PRL24-A0（same-commit FMT2 BS16）**，A0 不阻塞后续 arm；
- A vs B 是 Frozen/Joint 严格 A/B；
- A vs C 是 F/G off/on 严格 A/B；
- D vs E 是 Crop reward 严格 A/B；
- A、D、F 可描述三条工具线路各自的学习幅度与最终能力，但 prompt/tool schema 不同，
  不能把横向分数差直接写成严格 synergy 因果效应；
- Teacher25 下尚无完整的 pure native Crop 正式训练，因此 D 是新 baseline，不是对既有
  Teacher25 Crop 结果的“复现”。

### 5.1 Reward 合同

TGVF、Crop+TGVF 以及 PRL24-D 的 T-free 主体为：

```text
R_tfree = 2 × AnswerCorrect
        − 0.05 × max(0, ToolCallCount − 1)
        + ProtocolOrToolErrorPenalty

ProtocolOrToolErrorPenalty = -2 if an error occurred, otherwise 0  # FMT2
```

FMT1 的对应值为 `-1`，只属于早期历史 A。FMT2 是 A/B/C、已执行的 D-S1，以及原计划
E/F 的统一 recipe；不得把两种 penalty 的 checkpoint 接续训练或放入同一 matched 表。

PRL24-C 只在同一主体上增加现有 gold-free visual rewards：

```text
R_visual = R_tfree + FocusTarget + Grounding
```

VLM judge 固定使用 API 方式的 `qwen/qwen3-vl-32b-instruct`，输入不包含 gold answer。
Focus/Target 与 Grounding 的现有映射、prompt、并发、重试和失败策略全部保持冻结。API
错误先做有上限的自动重试；重试耗尽后采用预先冻结的 sample-local fallback、记录失败与
judge coverage，不因单个样本失败中止整条训练，也不得静默换模型或改变 answer reward。

PRL24-E 的视觉 Crop 样本采用历史 answer-gated 公式：

```text
R_crop_conditional = 0.8 × AnswerCorrect
                   + 0.2 × FormatScore
                   + 1.2 × AnswerCorrect × HasSuccessfulCrop
```

历史公式中的非法 format 为 `-1`；PRL24-E 实现时必须按本次统一修订使用 FMT2，
即非法 format 为 `-2`，其余 conditional 系数不变。ThinkLite direct/no-tool 分支沿用
历史路由，但同样采用 FMT2 format penalty：

```text
R_ThinkLite = 1.2 × AnswerCorrect + 0.4 × FormatScore
```

Teacher25 新增的 16 个 teacher prompts 均按 visual、crop-enabled 样本处理，在 E 中采用
上述 conditional 视觉公式。D 与 E 必须共享完全相同的 tool eligibility、prompt routing
和 batch schedule；两者唯一差异是 scalar reward。若实现审计发现当前 D 的 teacher
rows 使用了其他路由，应在正式训练前统一，而不是让 E 单独改变路由。

该 arm 的意义不是恢复所有旧训练行为，而是把 conditional reward 公式移植到当前 clean
Crop 实现中，与 D 做单变量比较。

## 6. Exposure-matched checkpoint 设计

直接比较“相同步数”会把 batch 和总数据暴露量混在一起。PRL24 同时采用两种视角：

| endpoint | prompts seen | trajectories seen | 对应的 BS16 endpoint | 用途 |
|---|---:|---:|---|---|
| BS64 Step 2 | 128 | 2,048 | BS16 Step 8 | 第一组等 prompt-exposure 比较 |
| BS64 Step 4 | 256 | 4,096 | BS16 Step 16 | 第二组等 prompt-exposure 比较 |
| BS64 Step 8 | 512 | 8,192 | 无现成 BS16 Step 32 | 判断大 batch 在更多暴露后能否继续增强 |
| BS64 Step 12 | 768 | 12,288 | 无现成 BS16 Step 48 | 补充中后程曲线，定位 S8→S16 回落 |
| BS64 Step 16 | 1,024 | 16,384 | 无现成 BS16 Step 64 | 正式终点；检验后半程平台、回落与持续收益 |

因此：

- **S2/S4 回答 batch 本身是否改善同等数据预算下的学习；**
- **S4→S8→S12→S16 回答 BS64 是否仍快速平台，以及后半程收益能否持续；**
- 不能用 S8/S16 相对 BS16-S16 的差值单独证明 batch 优势，因为它们分别多看一倍/三倍
  prompts；
- BS64-S8 与 BS16-S8 的相同 update-count 对照可作为工程 scaling 视角，但因前者多看
  四倍数据，不能替代 matched-exposure 结论；
- 默认正式 arm 运行至 S16；唯一已执行的例外是 B：S8 强负结果与上一轮 BS16 Joint
  pilot 同向，已按本文件第 0 节记录停止，以便把 GPU 转入 D。

正式任务每 step 可做 rolling recovery save，但 **S2 必须临时 pin**，不能被后续 rolling
retention 淘汰；S4、S8、S12、S16 永久保留。S2 在 checkpoint 评测、receipt 与结果表完成后可
删除大体积权重，仅保留 metrics、evaluation 和 provenance。各工具协议的 Step 0 只需
评测一次：A/B/C 共享 pure-TGVF S0，D/E 共享 Crop S0，F 使用自己的 Crop+TGVF S0。
若已存在完全相同协议与执行身份的可信 S0，可直接复用，不机械重测。

正式训练启动后默认不因 S2/S4/S8/S12 的暂时性能高低提前结束；只有数值爆炸、协议失效、不可恢复
运行错误或资源安全问题才允许在 S16 前停止；B 是上述 2026-08-20 有记录的资源例外。
为避免中途评测抢占训练 GPU，默认连续训练
并保存至 S16，再自动依次评测 S2/S4/S8/S12/S16；若有完全独立的评测资源，可在不影响训练的
前提下并行评测。

## 7. 统一评测与健康度指标

### 7.1 正式 accuracy

全部正式 endpoint 使用冻结的 CoreDev-2511 合同：

| 字段 | 固定值 |
|---|---|
| manifest | 2,511 rows |
| 实际推理 | 2,240 single-image rows |
| 显式 hold | 271 multi-image rows |
| sampling | temperature 1、top_p 1、每题一次 |
| RNG | `paired-seed-v1`，master seed 42，common random numbers |
| scorer | VLMEvalKit；需要语义 judge 时使用本地 Qwen2.5-72B-Instruct |
| headline | 七个分量等权 Macro* |

七个分量为 VStar、HRBench Average/all、BLINK single-image、OCR EN/CN mean、
MMMU-Pro single-image、MathVista、MathVerse five-version macro。必须同时报告逐 benchmark
结果；Macro* 不能掩盖单项 collapse。

已知 temperature-1 单次评测存在约 `1 pp` 量级波动，因此预注册解释为：

- `|delta Macro*| < 1.0 pp`：默认不确定；
- `delta >= 1.5 pp` 且至少 4/7 分量同向、无严重单项 collapse：强正向证据；
- 介于两者之间或由单一分量主导：边界结果，补第二个 paired evaluation seed；
- 第二个 evaluation seed 优先于立即新开更多 training arm。

阈值是 pilot 决策规则，不是统计显著性声明。

### 7.2 所有 arm 的运行健康度

训练期记录以下诊断，但不把它们当成外部泛化证据：

- answer reward、total reward、format/protocol error；
- tool call、成功率、重复调用和 tool error；
- response length 均值、长尾、截断和机械重复；
- zero-advantage prompt-group 比例；
- policy loss、gradient norm、clip fraction、ratio/KL diagnostic；
- GA4 四个 micro-block 的 loss/gradient scale 是否一致。

Teacher rows与 RP67 Stage1 数据同分布，因此 teacher training reward 更高、format 更合法
或调用更顺利都只能说明训练运行状态，不能单独证明 Teacher25 或 BS64 提升泛化。

### 7.3 Joint RP67 的 representation-health gate

PRL24-B 在 S4/S8/S16 增加固定的 D-health audit：

- `D` norm/RMS 与训练前 RP67 的偏移；
- correct-image vs donor-image、same-target separation；
- correct-D / zero-D / wrong-D counterfactual；
- image + D 的 clean-answer utility；
- answer-channel shortcut 与“由文本指令直接控制 decoder”的风险。

first-200 用作快速 gate；只有外评和表示健康度都通过的 checkpoint 才做 full-867。
默认对 S4/S8/S16 做 first-200，full-867 至少覆盖统一主终点 S16；其他 endpoint 仅在
通过快速 gate 且确有比较价值时补做。
如果 Joint 与 Frozen 的 Macro* 差异不超过噪声，默认仍保留更稳妥的 Frozen，而不是仅因
“没明显变差”就解冻。

### 7.4 F/G 的视觉健康度 gate

PRL24-C 额外报告：

- visual judge coverage、失败率、成本、Focus/Target 和 Grounding conditional mean；
- F/G 与 answer correctness 的相关/交换关系；
- 固定 held-out foveation、wrong-image sensitivity 和 hallucination audit；
- 同题 qualitative examples，尤其是工具后 reasoning 是否真正引用新视觉信息。

若 F/G 上升但 answer accuracy 或 CoreDev 下降，判定为 reward substitution，不因 judge
分数好看而晋级。

### 7.5 Conditional Crop 的工具使用 gate

PRL24-D/E 除 Macro* 外还比较：

- crop call 与 crop success；
- `P(correct | successful crop)` 与 answer-only bypass；
- Step 4→8→16 的工具使用是否坍塌；
- 重复调用、invalid bbox、长度与 protocol error。

conditional reward 如果只提高调用率、但没有保持或提高外评，不算成功。

## 8. 分阶段执行与停止规则

### Phase 0：实现与低成本 smoke

完成 BS64 schedule/config overlay、数学归一化检查、1-step functional smoke、resume 与自动
evaluation dry-run。该阶段不产生可汇报的科学结果。

### Phase I：先跑 PRL24-A，验证 BS64 总假设

训练 Frozen RP67 FMT2 T-free TGVF 至 S16，并评测 S2/S4/S8/S12/S16。若出现明确外评退化、梯度 scale
错误、长度/工具 pathology 或无法稳定恢复，先诊断 BS64/LR/GA，不把同一问题复制到其他
五个 arm。

H1 的支持条件是：

1. 历史 PRL22-A 只作 recipe-level anchor；若最终需要严格 batch-only 因果措辞，再补
   same-commit FMT2 BS16 A0，A0 不阻塞主序列；
2. S2/S4 的 matched-exposure 结果相对有效 BS16 control 同向改善，且改善超过采样噪声；
3. S4→S8→S16 没有重现不可接受的后期回落，并明确记录是否仍存在平台；
4. 至少 4/7 Macro* components 同向，且无关键 benchmark collapse；
5. 训练与工具行为健康。

如果 matched deltas 小于约 1 pp 且分项混杂，则结论为“尚无明确 batch gain”，而不是
强行归因。若 A 明确失败，暂停本系列并检查是否需要 LR、采样或数据上的新实验；这些
改动必须进入新的变量身份，不能在 PRL24-A 中途静默调整。

### Phase II：Adapter 与 visual reward（已形成当前决策）

以 A 为复用 control，依次运行 B（Joint）和 C（F/G）。不另跑重复的 Frozen/no-FG
control。实际执行中 B 按第 0 节在 S8 有意停止并保留 Frozen；C 已完成至 S16，F/G
未形成持续 endpoint accuracy 增益，仍需以 foveation/hallucination audit 判断其专项价值。

### Phase III：Crop、conditional reward 与组合工具（暂停）

原计划先运行 D，再与 E 成对比较，最后运行 F。实际 D 因 native Crop 在 BS64 下单步约
`2 h 32 min`，在完整 S1 checkpoint 后停止；E/F 未启动。本阶段不再在 BS64 身份下继续。
Crop conditional/T-free、Atomic Crop+TGVF 等问题迁移到 PRL25 的 BS16/Teacher25/S80
统一新训矩阵，仍保留 reward A/B 与跨协议措辞边界。

### Phase IV：决定 BS128 或 BS256（暂停，不执行）

下表保留为原预注册决策树；当前不会进入 BS128/BS256：

| BS64 结果 | 下一步 |
|---|---|
| 明确正向、吞吐与显存允许 | 优先评估是否直接 BS256，接近 DeepEyes prompt scale |
| 明确正向但直接 BS256 风险/成本过大 | 先 BS128 验证 scaling curve |
| 只有边界增益 | 补 evaluation seed 或 winner 的训练 seed，不机械放大 |
| 无增益或出现 pathology | 停止 batch scaling，转向 reward/data/optimization 诊断 |

这不是必须逐级完成 `64 → 128 → 256` 的固定阶梯。

## 9. 时间与资源预算

以下表格是启动前的历史规划，已被实测推翻，不能再作为运行 ETA。PRL24-D 实测 S1
约 `2 h 32 min`，而不是表中的 `38--45 min/step`；原样 S16 约需 `40 h`。这与前述
质量增益不明确共同构成暂停 BS64 的资源理由。

原始估计为：

| 项目 | 规划估计 |
|---|---:|
| BS64 单 step | 约 38--45 min |
| 16-step training | 约 10--12 h |
| S2/S4/S8/S12/S16 五个 CoreDev evaluation | 约 1.6--2.5 h，首个 arm 实测后校准 |
| 单 arm 总 wall time | 约 11.5--14 h；Joint/F-G/Combo 等较慢 arm 可到 13--17 h |
| 六 arm 全串行 | 约 75--95 h |

实际执行只完成 A/B/C 及 D-S1；E/F 未启动。PRL24 不再追求六臂 S16 完成，后续资源转入
PRL25。任何未来恢复 BS64 的 ETA 都必须从同一工具线的实测完整 step 重新校准。

2026-08-21 补充：PRL25 已把 pure Crop 对齐到与 Atomic 路径相同原则的 actual behavior
logprobs、exact replay、live current-vision/recorded-reference replay 和 full-Qwen 版本同步。
BS4 × n2 canary 以 `262.27 s` 完成一个含 checkpoint 的真实 optimizer step。该结果说明
PRL24-D 的 `2 h 32 min` 不能解释为 Crop 算法的固有速度，但它也不改写已经发生的 BS64
历史 wall time，更不能替代 BS64 × n16 benchmark；PRL24 仍保持暂停。

存储采用 rolling checkpoint：除已停止的 B 外，其余 arm 至少保留 S2/S4/S8/S12/S16 到评测、结果 receipt 与
必要轨迹抽样完成；随后每个 arm 的指标与 provenance 永久保留，大体积权重只长期保留
S16、关键 endpoint 和 winner，避免六条线累积占满磁盘。

## 10. 预期产出与结论模板

原计划要求 PRL24 完成后形成以下产出；由于系列暂停，未完成项迁移到 PRL25，不得把它们
写成已经存在：

1. BS16/BS64 matched-exposure 总表；
2. BS64 S2/S4/S8/S12/S16 继续性曲线；
3. Frozen vs Joint paired 表与 D-health 审计；
4. F/G off/on paired 表、成本和 foveation/hallucination 审计；
5. Crop T-free vs conditional reward paired 表；
6. TGVF、Crop、Crop+TGVF 的协议内 S0→S2/S4/S8/S12/S16 表；
7. 是否进入 BS128/BS256 的明确 decision record。

最终只允许使用以下层级的措辞：

- **“BS64 更强”**：matched-exposure 外评超过噪声、分项广泛同向、继续性和健康度均
  通过；
- **“BS64 更稳定但未证明更强”**：梯度/reward 更平滑，但外评差异不明确；
- **“BS64 仅靠更多 exposure 获益”**：S2/S4 不优于 BS16，只有 S8/S16 在额外数据后提高；
- **“BS64 未改善当前 recipe”**：matched 与 continuation 均无收益或出现 pathology。

## 11. 来源与证据索引

- [Policy RL 小批量 Pilot 收官](POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md)：
  BS16 的主要结论、reward、训练设置与证据边界。
- [CoreDev-2511 统一测量合同与基线](POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md)：
  paired-seed-v1、Macro* 定义和 canonical 结果总表。
- [PRL22 Teacher25 Policy-Data Ablation](PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md)：
  Teacher25 数据合同和两条工具线的积极 pilot 证据。
- [PRL21 Crop T-free 16-step 结果](PRL21_CROP_TFREE_16STEP_RESULTS_AND_EVALUATION_INCIDENT_20260815.md)：
  当前 Crop T-free 外评与后期 tool-bypass 现象。
- [PRL13 Crop RL 8-step 成功报告](PRL13_CROP_RL_8STEP_SUCCESS_REPORT_20260808.md)：
  历史 answer-gated conditional Crop reward 的来源。

---

本文件最初是预注册计划；当前实际状态与偏离以第 0 节及阶段性结果文档为准。任何正式
启动仍须在独立实验身份下记录实际 commit、配置、数据 schedule、训练 receipt、checkpoint
和 evaluation receipt；不得在运行中无记录地改变 LR、reward、batch、prompt 或工具协议。

当前 A/B/C 阶段性实测结果见
[PRL24 A/B/C 阶段性结果](PRL24_ABC_INTERIM_RESULTS_20260819.md)。
