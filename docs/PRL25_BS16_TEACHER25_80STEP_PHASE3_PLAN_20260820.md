# 第三期（PRL25）：BS16 Teacher25 80-step 统一实验计划

日期：2026-08-20；执行状态更新：2026-08-24（Asia/Tokyo）

状态：`RUNNING / PRL25-B/C 六点曲线已完成；PRL25-D 正在 fresh-S0 自动准入`

Decision ID：`POLICY-RL-PHASE3-BS16-TEACHER25-80STEP-20260820-v1`

## 1. 决策摘要

第三期暂停继续扩展 PRL24 的 BS64 序列，统一回到 `BS16 × n16`，把训练长度从
S16 扩展到 S80。所有正式 arm 都从同一个基础模型状态重新开始，固定 Teacher25、FMT2、
优化器、采样与评测合同，不续训任何 PRL21/22/24 checkpoint。

本期要检验的核心假设是：此前多个 S8/S16 结果出现的平台或回落，可能主要来自 optimizer
step 数量不足，而不是工具或 reward 已经达到能力上限。S16 在本期只是学习曲线中的诊断点，
S80 才是预注册主终点；除数值爆炸、协议失效或不可恢复错误外，不因 S16 暂时平台而早停。

## 2. 为什么暂停 BS64

PRL24 暂停在当前证据边界，不把它写成完整收官的六臂实验：

- pure TGVF 的 recipe-level 对照没有显示明确 BS64 质量增益。等 exposure 的
  `BS64 S4 = 58.3983` 相对历史 `BS16 S16 = 59.9590` 为 `-1.5607 pp`；BS64
  最佳点 `60.7348` 只比 BS16 最佳点高 `+0.7758 pp`，低于当前约 `1 pp` 的不确定
  区间；BS64 S16 还低 `-0.5583 pp`。由于 FMT/commit 不完全 matched，这些是
  recipe-level evidence，不是严格 batch-only 因果结论，但已不足以支持继续优先投入 BS64。
- PRL24-D native Crop 的正式 S1 从开始 rollout 到 checkpoint 完整落盘约
  `2 h 32 min`，其中 actor update 约 `2 h 22 min`。按该执行体原样外推，S16 约需
  `40 h`，资源效率不可接受。
- D 只完成 S1 optimizer checkpoint，并在 S2 update 前停止。S1 rollout 来自更新前的
  S0 policy，且停止发生在最终 metrics/trajectory publication 前，因此 D-S1 不是 efficacy
  结果，不能用于声称 Crop 已改善或退化。
- PRL24-E/F 尚未启动。A/C 已完成，B 在 S8 有记录停止，D 保留可恢复 S1，但这些结果与
  checkpoint 均不续接到第三期。

BS64 不是永久否定；只有当 BS16-S80 先建立可信的长程最佳点，且后续确有 batch-only
因果问题时，才以同 commit、同 reward、同 exposure 的独立实验重新评估。

## 3. 五个正式 arm

| ID | 工具线路 | Adapter | Reward | 与 matched control 的变量 | 主要问题 |
|---|---|---|---|---|---|
| **PRL25-A** | native Crop | — | **FMT2 DeepEyes-style conditional Crop** | A/B 只换 scalar reward | answer-gated Crop credit 在长训练下是否保持工具使用并提高外评 |
| **PRL25-B** | native Crop | — | **FMT2 T-free（自研主线）** | A/B 只换 scalar reward | answer-centric reward 在 S16 后能否继续学习 |
| **PRL25-C** | pure TGVF | Frozen RP67 | **FMT2 T-free** | C/E 的 no-visual-reward control | 纯 TGVF 的长程学习曲线与 S80 能力 |
| **PRL25-D** | Atomic Crop+TGVF | Frozen RP67 | **FMT2 T-free** | 相对 C 改工具协议 | 组合工具在长训练下是否保持其描述性优势 |
| **PRL25-E** | pure TGVF | Frozen RP67 | **FMT2 T-free + Focus/Target + Grounding（F+G）** | C/E 只开启 F+G | Target/Ground visual credit 是否需要更多 step 才转化为外评 |

执行优先级修订（2026-08-20）：第一批只排自研 T-free 主线，顺序为
`PRL25-B Crop → PRL25-C TGVF → PRL25-D Atomic Crop+TGVF`。三条全部完成前，
PRL25-A conditional Crop 与 PRL25-E F+G 不占用正式训练档；smoke/preflight 不计入
科学训练，也不得改变每条 arm 从相同 S0 fresh start 的要求。

术语约定：本计划中的 “TGVF Target Ground reward” 指已经在 PRL19/PRL24-C 使用的
`Focus/Target + Grounding` 两项 gold-free visual reward（F+G），不是新建另一套未定义
的 reward。若未来只开启 F 或只开启 G，必须另立 arm，不能混入 PRL25-E。

严格比较边界：

- A vs B 是当前 native Crop runtime 上的 reward A/B；
- C vs E 是相同 pure-TGVF 协议上的 F+G off/on A/B；
- C、D 与 Crop arms 使用不同工具/prompt schema，只能形成能力地图，不能把横向差值称为
  严格 synergy；
- 所有 arm 独立从 S0 训练，不能把前一 arm 的 winner 当后一 arm 初始化。

## 4. 统一训练合同

| 字段 | 第三期固定值 |
|---|---|
| base policy | `Qwen3-VL-8B-Instruct`，五臂共享同一初始权重身份 |
| policy update | full Qwen：vision encoder、merger/projector、language model 均更新 |
| representation | TGVF/Crop+TGVF：RP67 Step-2000，Frozen；native Crop：N/A |
| data | canonical Teacher25 schedule，seed 42，无放回 |
| BS16 composition | 每 step 12 条 retained T1 + 4 条 retained teacher prompt |
| horizon exposure | 80 steps = 1,280 prompts = 20,480 trajectories |
| global prompt batch | 16 independent prompt groups |
| rollout count | n16 / prompt，即 256 trajectories / optimizer step |
| distributed | 8 × B200，world8，prompt micro2/rank，GA1，FSDP2 |
| optimizer | AdamW，weight decay `0.01`，constant LR `1e-6`，无 warmup |
| PPO / KL | PPO epoch 1；KL reward 0；actor KL loss off |
| gradient clipping | `1.0` |
| sampling | temperature `1.0`，top-p `1.0` |
| length / calls | prompt 8,192；response 20,480；最多 6 次工具调用 |
| format contract | **FMT2：protocol/format/tool error 统一罚 `-2`** |
| primary endpoint | Step 80；Step 16 不再作为训练停止点 |

Teacher25 在五臂中共享同一 canonical sample 顺序和 75%/25% 来源组成。由于工具能力不同，
prompt 的工具说明与可用 tool schema 必然按 arm 渲染；“数据一致”指 sample identity、顺序、
teacher 比例与 supervision 来源一致，不伪称不同工具协议的输入字节完全相同。

Fresh start 的含义是：新的 run/output/W&B identity、新 optimizer、新 dataloader cursor、
新训练轨迹，从相同 S0 权重开始。PRL24-D S1 以及历史 PRL21/22 权重不得作为初始化。

### 4.1 PRL25-B Crop 执行对齐与准入结果（2026-08-21）

PRL25-B 不复用 PRL21 的旧纯 Crop 执行路径。当前 exact-Crop 路径传输 rollout 时真实的
behavior logprobs，在 actor update 中以原始预处理 pixels 对当前可训练 Qwen vision 做
live differentiable replay，并以 rollout 时记录的 features 做 frozen-reference replay。
vision encoder、merger/projector 与 language model 均沿 Crop 路径更新；本 arm 不加载 RP67、
TGVF 或 policy LoRA。每次成功的 full-Qwen upstream sync 都发布绑定 run identity 与
optimizer step 的版本 receipt，下一轮 rollout 只消费已发布版本。

实现分支为 `prl25-crop-exact-replay-alignment`；正式配置提交为
`e5344b36501d36a8aff612cb18477d37a221b61a`，其代码身份绑定
`2ae994a7cd71fefb9a4dc2c92dd52fc59865e7f4`。BS4 × n2、world4 的非科学
1-step canary 在 2026-08-21 00:28 JST 正常退出：4 prompts、8 trajectories、9 次成功
Crop observation，`grad_norm=24.2336`，step-1 full-Qwen receipt 与 checkpoint 均完整。
这只证明功能、梯度、同步和恢复链路，不是质量结果，也不是 BS16 × n16 的吞吐 benchmark。

### 4.2 PRL25-B S39 中断与恢复边界（2026-08-21）

正式 scientific lineage 已连续完成并发布 S1–S39，`metrics.jsonl` 与 latest checkpoint
tracker 均停在 39；S40 尚未发生 optimizer update。S39 单步用时 `1,722.74 s`，answer
accuracy 为 `55.8594%`，FMT2 format-error rate 为 `2.7344%`，成功 Crop observations
为 179。S39 后的下一轮 rollout 因外部 OpenRouter/DeepInfra DeepEyes judge 连续 HTTP 429
超过 bounded transient window 而 fail-closed 停止；该事件不表示 Crop replay、GPU 或 CPU
训练路径失败，S39 checkpoint 完整可恢复。

首次 clean-process auto-resume 已在全部八个 rank 成功载入 S39 model、optimizer、RNG 与
scheduler，但随后暴露 full-Qwen operational receipt 的恢复时序缺口：worker 初始化先发布
S0 receipt，checkpoint bridge 在载入后仍读取该旧标记，因而把实际 S39 误判为 S0 并退出。
恢复提交 `705112574b789ac04ccfc69e4206e1998448ab6f` 只在 upstream checkpoint 成功载入
且重新计算的 run/step/base identity 与 project state 完全一致后，才把 bootstrap receipt
更新到 checkpoint step；严格 mismatch 保护与 LoRA tensor 校验保持不变。该变更通过 68 个
focused tests 与 Ruff，允许同一 scientific run 从 S39 继续到原定 S80，不改变 reward、数据、
优化器或 arm identity。

### 4.3 PRL25-B S80 完成与主终点评测（2026-08-22）

PRL25-B 已完成全部 80 个 optimizer steps；`metrics.jsonl` 连续包含 S1–S80，latest tracker、
runtime policy receipt 和永久 S80 checkpoint 均闭合。S80 单步为 answer `66.80%`、FMT2
format error `0.39%`、mean reward `1.328`、成功 Crop trajectory 比例 `81.25%`，用时
`23.31 min`。最后八步 S73–S80 的均值为 answer `71.39%`、format error `0.59%`、
mean reward `1.416`、成功 Crop trajectory 比例 `81.98%`、`25.04 min/step`。训练内部
S65–S72 answer 均值 `76.61%` 高于最后八步，因此不能只用训练 reward 宣称长程改善，必须
以外部 CoreDev 结果判定 S80，并把较早 checkpoint 仅作为预注册主终点之外的补充曲线。

S80 CoreDev-2511 主终点评测于 `2026-08-22 17:30 JST` 启动。评测 ID 为
`PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TEMP1-SEED42-UNIFIED-V1`，固定 native-Crop
visual prompt、`image_zoom_in_tool`、Hermes parser、最多 6 次调用、temperature 1、master
seed 42 和七项 Macro* 官方评分。checkpoint owner 由 PRL25-B config、S80 永久 receipt、
8-rank FSDP pair 与连续 80 行 metrics 联合证明；协议 owner 使用 Teacher25 native-Crop
合同，prompt 哈希与训练完全一致。接入代码/计划提交为
`9e977c6b6b8a4714e2057ba5fe010afb33995bdb`。正式评测已完成并通过：Macro* 为
`62.2288`，七项分量依次为 VStar `81.6754`、HRBench `74.5000`、BLINK single-image
`58.8889`、OCR mean `55.3358`、MMMU-Pro single-image `46.4684`、MathVista `67.3333`、
MathVerse five-version macro `51.4000`。2,511-row summary 仅有 1 次 judge parse failure，
按预注册规则确定性计错且未超过阈值。`paired-summary.json` SHA256 为
`166153c701cabd2684dbc2ffce54de06b58d2fd0014de40fb70e52794cc557ee`，
`evaluation-complete` SHA256 为
`50c88724fa67c7fa5f0a2d61471e119f94369520c84d8ba89f97f71e30c5a047`。

为避免 10 个永久 checkpoint 全量外评造成不必要的时间开销，PRL25-B 的常规长程曲线固定为
`S8 / S16 / S32 / S48 / S64 / S80`：S8/S16 提供早期和历史平台基线，S32/S48/S64 是
长程补充点，S80 保持预注册主终点；S24/S40/S56/S72 仅在主曲线出现异常时按诊断需要补评。物理机器
实际为 8 张 GPU（编号 0--7）。S80 在 GPU 0--3 继续运行的同时，S16 于
`2026-08-22 18:06 JST` 以 inference-only / deferred-scoring 模式在 GPU 4--7 启动，避免
重启 S80 或因共用 GPU 2/3 的 72B judge 产生冲突。该调度模式由提交
`2bc2886f660880ee4eca87a8b9bfc9d1d12118ff` 记录；S16 生成完成后再进入同一官方 scorer，
不能把 inference-complete receipt 当作外评完成或结果 receipt。

启动前复核发现初版 S16 计划使用了独立的 `step16` RNG namespace；当时 S16 尚为
`0/2,240`，因此在没有丢失正式生成的情况下停止准备并修正。提交
`43106f480f282c5854a82be91e6003e984468b36` 将全部学习曲线节点绑定到 S80 已冻结的
namespace，使 checkpoint 间真正共享逐题逐 turn 随机流。剩余 S8/S32/S48/S64 的四臂
执行计划固定在
`configs/evaluation/prl25_b_crop_exact_step8_step32_step48_step64_full_model_coredev2511_plan.json`，
最终执行提交为 `bd6ad16eb599e52f2c3dab6acf34114a67769f30`；8 卡运行时按两臂一批并发生成。
初版四臂计划误把 owner completion 指向 S80，而该计划最后 arm 为 S64；运行在 GPU 启动和
样本生成前 fail-closed。上述提交改为 S64 永久 receipt 并增加 runtime regression test。

`2026-08-22 18:54 JST` 已补齐自动接力：S16 inference-only 保持运行；四臂 supervisor
并行进行 checkpoint materialization，待 S16 释放 GPU 4--7 后自动以 0--7 执行
S8/S32、再执行 S48/S64，并完成四臂官方评分；四臂 `evaluation-complete` 发布后，独立
handoff supervisor 自动对已有 S16 inference 执行正式评分。任一前序进程若退出而没有对应
completion receipt，接力会 fail-closed，不会静默跳过 checkpoint。

S16 正式评分于 `2026-08-22 19:39 JST` 完成并通过，Macro* 为 `62.0842`；VStar、
HRBench、BLINK single-image、OCR mean、MMMU-Pro single-image、MathVista、MathVerse
macro 分别为 `77.4869/77.0000/58.8889/52.8325/44.9814/69.0000/54.4000`。相同逐题
随机流下，S80−S16 Macro* 仅 `+0.1446 pp`；分量 delta 为
`+4.1885/-2.5000/0.0000/+2.5034/+1.4870/-1.6667/-3.0000 pp`，因此当前不能把
延长到 S80 写成整体能力超过 1 pp 的改善。S16 paired-summary SHA256 为
`45f2082a54eabb64971161046a9650b8f6f536a26742c232661ec88bbb9cfac4`，
evaluation-complete SHA256 为
`5a529107c3bd396d6038ca8edd59849ea35e22fdba8a289ba8d83a0f7e9148eb`。

四臂 full-model materialization 与静态验证均已完成；首批 S8（GPU 0--3）与 S32
（GPU 4--7）的八个正式 worker 于 `2026-08-22 19:56 JST` 启动。该批完成后同一
supervisor 自动运行 S48+S64，再以单次加载的 TP2 72B judge 并发评分四个 checkpoint。

### 4.4 PRL25-B 六点学习曲线闭环（2026-08-22）

S8/S32/S48/S64 四臂生成于 `2026-08-22 22:04 JST` 全部完成，每个 checkpoint 均覆盖
2,240 个支持的单图任务；统一官方评分与四臂 `evaluation-complete` 于 `22:21 JST` 完成。
至此 PRL25-B 的默认 `S8/S16/S32/S48/S64/S80` 六点曲线全部闭合。各节点使用完全相同的
CoreDev-2511 任务、native-Crop 协议、temperature 1、master seed 42 和 S80 frozen RNG
namespace。单位均为 `%`：

| Checkpoint | Macro* | VStar | HRBench | BLINK single | OCR mean | MMMU single | MathVista | MathVerse macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S8 | 59.0269 | 76.4398 | 65.0000 | 60.5556 | 47.4135 | 42.3792 | 67.0000 | **54.4000** |
| S16 | 62.0842 | 77.4869 | **77.0000** | 58.8889 | 52.8325 | 44.9814 | 69.0000 | **54.4000** |
| S32 | **63.5377** | 80.1047 | 73.0000 | **64.4444** | 54.8108 | **49.0706** | **71.3333** | 52.0000 |
| S48 | 61.5559 | 76.4398 | 74.0000 | 59.4444 | 53.7285 | 47.2119 | 68.6667 | 51.4000 |
| S64 | 61.9993 | 80.1047 | 73.5000 | 61.6667 | **56.2758** | 44.9814 | 65.6667 | 51.8000 |
| S80 | 62.2288 | **81.6754** | 74.5000 | 58.8889 | 55.3358 | 46.4684 | 67.3333 | 51.4000 |

S32 是六点中的 post-hoc 最佳 checkpoint，Macro* 相对 S16 为 `+1.4535 pp`，七项中
VStar、BLINK、OCR、MMMU 和 MathVista 五项上升，HRBench 与 MathVerse 两项下降。这超过
第 7 节的 `1 pp` 中程改善门槛，但比预先定义的 `>=1.5 pp` 强证据线低 `0.0465 pp`。
此后 S48/S64/S80 相对 S16 分别为 `-0.5283/-0.0849/+0.1446 pp`，没有保持 S32 的
改善。因此本 arm 支持“更多 step 可在中程产生有限改善”，但不支持“能力随训练长度持续
单调提高”；S80 仍是预注册主终点，S32 只能作为完整曲线中明确标注的补充最佳点。

四臂评测 ID 为
`PRL25-B-CROP-EXACT-COREDEV2511-STEP8-STEP32-STEP48-STEP64-TEMP1-SEED42-UNIFIED-V1`。
`paired-summary.json` SHA256 为
`0f8287298a4d467cdb52918957cb84e307f76728d75a9249649e7fd1f62451c6`，
`evaluation-complete` SHA256 为
`2b89fbbc47cf41d0adc5bb9b288e695bb88fe25275cc3b9b4178d9f3d4661eb9`。

本次评分暴露了资源调度缺口：四个 checkpoint 共用一个位于 GPU 2--3 的 TP2 72B judge，
其余六张 GPU 在评分阶段空闲；judge 服务从启动到四臂 completion 约 `15 min`。这不影响
结果身份或数值，但没有利用可并行的四个 TP2 实例。提交
`2bbb5f9309d6b1d9dca25a81bd1855dbd100859e` 将后续多臂评分改为最多四个并发 judge：
GPU `0--1/2--3/4--5/6--7` 分别绑定端口 `8012/8013/8014/8015`，各 arm 只访问自己的
服务；所有实例先并发启动再等待 readiness，历史单端口结果仍可严格恢复验收。Ruff 与 60 个
focused tests 通过。该修复不重算或改写本轮正式结果；约 8 分钟目标需在下一次多臂评分中
实测确认。

### 4.5 PRL25-C pure-TGVF 正式启动与自动评测合同（2026-08-22）

PRL25-C 已于 `2026-08-22 22:47 JST` 从 fresh S0 启动。正式 run 为
`PRL-25-C-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-TEACHER25-80STEP-WS8`，
run identity SHA256 为
`272d6209be1247582c8c4b2f616609b55203646c2a4b26a4b075aad3960c9b02`。
它使用 pure TGVF、Frozen RP67 Step-2000、BS16 × n16、Teacher25、full-Qwen update、
constant LR `1e-6` 和 FMT2 `protocol_error_penalty=2.0`；没有 Crop、F+G visual reward
或 policy LoRA。CPU preflight 复核了 GA1、每 step 4/16 teacher 和完整 80-step 调度；
身份闸门通过后，8 个 actor/rollout worker 已在物理 GPU 0--7 初始化。启动 smoke 不混入
scientific lineage。

实现与监督器位于分支 `prl25-c-tgvf-80step`；代码/监督器提交为
`b100d3d462bead2f5f2a0b4a365b4a38e59f5d2d`，正式配置绑定提交为
`b87126ae291758545f929c4f06ffc098dd4a7886`，最终六点评测计划绑定提交为
`9f4104b78ebcf8ba90c81d401fc591ab8ed3945a`。训练监督器在中断后只从同一 canonical checkpoint 自动恢复；只有 S80 永久
checkpoint receipt、data、FSDP config 以及每类 8 个 model/optim/extra-state shard 全部完整，
才自动交接评测。

自动评测固定覆盖 `S8/S16/S32/S48/S64/S80`，共用同一 CoreDev-2511 任务、TGVF 协议、
temperature 1、master seed 42 和逐题逐 turn paired RNG namespace。生成阶段使用全部 8 卡，
每批并发两个四卡 checkpoint arm；评分阶段最多同时启动四个 TP2 Qwen2.5-72B judge，分别
绑定 GPU `0--1/2--3/4--5/6--7`。任一 checkpoint 身份、冻结 RP67 或 completion receipt
不匹配均 fail-closed；不会把 inference-complete 当作正式评分完成。

#### 4.5.1 S80 长输出的进行中诊断（2026-08-24）

截至 `2026-08-24 03:51 JST`，六点评测仍在生成最后一批 S64/S80，因此本段不是最终
CoreDev 分数。S8/S16/S32/S48 已各完成 2,240 个受支持样本；S64/S80 分别完成
`628/2,240` 与 `108/2,240`。在 S80 已完成的 108 个 sample identity 上做严格配对后，
S64 仅 `6/108 = 5.56%` 触及 `max_tokens`，平均每条约 `1,213` 个 sampled tokens；S80
则有 `81/108 = 75.00%` 触及 `max_tokens`，平均约 `15,785` tokens。四个 S80 rank 均持续
写入，日志无 OOM、traceback 或 runtime error，故当前慢速解释为 policy 长输出退化，而非
CPU/GPU hang。

训练内指标提供同向证据：S64 batch 为 answer `67.58%`、FMT2 error `1.95%`、平均
`440.6 tokens/trajectory`；S72 一度达到 answer `84.77%`、FMT2 error `0.78%`；S80
batch 则为 answer `51.95%`、FMT2 error `23.44%`、平均 `1,337.2 tokens/trajectory`。
S65--S72 窗口均值为 answer `77.39%`、FMT2 error `2.05%`，S73--S80 为 answer
`68.21%`、FMT2 error `9.23%`。这些数据支持“晚期退化”的进行中诊断，但不能替代完整
2,240-sample 生成和七项正式评分；评测不得通过临时截短 response、修改 stop contract 或
挑选 S72 来掩盖预注册 S80 结果。

#### 4.5.2 六点正式结果与结论（2026-08-24）

PRL25-C 于 `2026-08-24 06:01 JST` 完成全部六点评测。每个节点均有完整的
`2,240` 条受支持任务生成、`2,511` 条 CoreDev 记分覆盖、7/7 数据集 receipt 和
`PASS` summary；六点合计 42/42 receipt。所有节点使用同一 temp1/seed42 paired-RNG
合同，故表中差异可直接解释为同一训练轨迹的 checkpoint 曲线。

| Step | Macro* | VStar | HRBench | BLINK-single | OCR-mean | MMMU-single | MathVista | MathVerse-macro | Judge parse failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S8 | 59.6852 | 68.0628 | 64.5000 | 67.2222 | 41.8819 | 50.9294 | 70.0000 | 55.2000 | 0 |
| S16 | 59.1474 | 72.2513 | 63.0000 | 63.3333 | 39.0893 | 50.5576 | 72.0000 | 53.8000 | 1 |
| S32 | 59.5262 | 72.7749 | 65.5000 | 66.6667 | 39.8045 | 49.0706 | 69.6667 | 53.2000 | 1 |
| S48 | 59.1325 | 71.7277 | 67.5000 | 66.1111 | 43.1063 | 40.1487 | 70.3333 | 55.0000 | 2 |
| S64 | **59.8086** | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 | 1 |
| S80 | 57.7907 | 70.1571 | 65.0000 | 64.4444 | 35.4185 | 44.9814 | 72.3333 | 52.2000 | 1 |

S64 是六点中的 post-hoc 最高点，但只比 S8 高 `0.1235 pp`；S16/S32/S48 也没有形成
随训练长度稳定增加的趋势。因此，本轮不支持“pure TGVF 只因步数不足、延长至 S80 就会
明显改善外评”的假设。预注册主端点 S80 反而是六点最低值，比 S64 低 `2.0180 pp`，其中
OCR-mean 从 `44.5446` 降至 `35.4185`。这与 S73--S80 的 answer/FMT2 恶化和 S80 长输出
退化一致；它是正式结果的一部分，不能以后验选择 S64 代替 S80 主结论。

paired summary SHA256 为
`15acd55a77ec1f59b3610806d74ebc0bd5e8ca0717ec859e9c443da7fa2ba83d`，
evaluation-complete SHA256 为
`8ffc749fbb41482f35f6323c305d6b32eb78eea495e3ea60d485c7d92b432534`。

### 4.6 PRL25-D Atomic Crop+TGVF 自动交接合同（2026-08-24）

用户于 `2026-08-24 03:37 JST` 冻结下一条正式线路：PRL25-C 六节点评测完成后，立即从
fresh S0 启动 PRL25-D Atomic Crop+TGVF 80-step；PRL25-E Focus/Target + Grounding 不在
本次自动队列中，留待 D 完成后另行启动。等待监督器会检查 C 的正式
`evaluation-complete`，随后把全部物理 GPU `0--7` 交给 D，不以 inference-complete 或部分
arm 完成作为交接条件。

PRL25-D 正式 run 为
`PRL-25-D-QWEN3-INSTRUCT-FULL-FROZEN-RP67-BS16-N16-TFREE-CROP-TGVF-TEACHER25-80STEP-WS8`，
run identity SHA256 为
`a0b8183875c38482da476e90a676c9b2ad401b100d452c73361d37b53157990c`。
固定合同为 full-Qwen update、Frozen RP67 Step-2000、BS16 × n16、Teacher25、constant LR
`1e-6`、最多 6 次 `tgvf_crop_tool` 调用和 FMT2
`protocol_error_penalty=2.0`；Focus/Target 与 Grounding reward 均关闭。它使用已验证的
Atomic 单工具协议，不切换到 native-pixel Crop backend，也不把 RP67 误作 policy LoRA。

配置与监督器提交为 `ec0555b`（执行分支 `prl25-c-tgvf-80step`），配置文件 SHA256 为
`37a4f2bb64821dfcd9918d92432bd1e341020efd15bc936b0704e6570e187b1e`。CPU-only preflight
已通过 run-config compose、Atomic protocol、FMT2、GA1 和前 80 个 BS16 slice 的
Teacher25 `4/16` 检查；不另建独立 GPU smoke lineage，首个 GPU step 即正式 S1。训练监督器
允许同一身份断点恢复，并在完整 S80 门禁后自动评测
`S8/S16/S32/S48/S64/S80`；生成与四路 TP2 judge 继续使用全部八张 GPU。

## 5. Reward 合同

自研 T-free 主体（PRL25-B/C/D）为：

```text
R_tfree = 2 × AnswerCorrect
        − 0.05 × max(0, ToolCallCount − 1)
        − 2 × 1[ProtocolOrToolError]
```

PRL25-A 只把 scalar reward 换成当前协议上的 DeepEyes-style conditional Crop credit：

```text
R_crop_conditional = 0.8 × AnswerCorrect
                   + 0.2 × FormatScore
                   + 1.2 × AnswerCorrect × HasSuccessfulCrop
```

其中非法 format 使用本期通用 FMT2 `-2`；其余 conditional 系数保持冻结。这里只迁移
reward 公式，不恢复旧 prompt、旧 `<answer>` dialect、旧 runtime 或旧数据。

PRL25-E 为：

```text
R_target_ground = R_tfree + FocusTarget + Grounding
```

F/G judge 的模型、prompt、gold-free 输入、映射、并发、重试和 fallback 全部沿用并冻结
PRL24-C 身份。F/G 上升但 answer accuracy 或 CoreDev 下降时，按 reward substitution 处理，
不因训练 reward 更高而判定成功。

## 6. Checkpoint、metrics 与评测

- 每步必须原子发布 reward/health metrics；至少记录 answer reward、total reward、FMT2 error、
  tool call/success/error、重复调用、response length、zero-advantage group、loss、gradient norm、
  clip/ratio/KL diagnostics。轨迹审计数据必须在允许人工停止前完成 publication，避免重现
  PRL24-D “checkpoint 有效但 S1 metrics 未落盘”的信息缺口。
- **每 1 step** 写一次完整 optimizer recovery checkpoint，活跃训练期间 rolling 只保留最近
  2 个；`S8/S16/S24/S32/S40/S48/S56/S64/S72/S80` 每 8 step 暂存完整快照，S0 另做共享
  evaluation，S1 作为早期恢复/速度校准点。训练与既定评测闭环后，每个保留 step 都转为
  实测约 `15.89 GiB` 的可独立测评 compact checkpoint；每个完成 S80 的正式 arm 只额外
  保留一个完整 S80 recovery。S80 同时保留 compact 与 full recovery，不再为 intermediate
  或 post-hoc winner 永久保留 optimizer/FSDP state。完整门禁与 2026-08-24 现存对象清单见
  [checkpoint 存储缩减守则](POLICY_CHECKPOINT_STORAGE_COMPACTION_20260824.md)。
- 永久保留不等于默认全部外评。常规外评节点为 `S8/S16/S32/S48/S64/S80`；其余永久节点
  用于恢复、异常定位和按需补评。S8/S16 是早期学习与平台基线，不得因已有历史 Crop
  S8/S16 而跳过本次 same-run、same-protocol 的对应节点。
- 全部 endpoint 使用冻结 CoreDev-2511 `paired-seed-v1`、master seed 42、temperature 1
  合同，同时报告七项 Macro* 分量。A/B 共享 exact Crop S0；C/E 共享 exact pure-TGVF S0；
  D 使用自己的 Atomic Crop+TGVF S0。共享 S0 evaluation 不等于共享训练 lineage。
- S80 是预注册主结果；best checkpoint 只能作为补充，并必须同时展示完整学习曲线，不能
  用 post-hoc best 掩盖 S80 回落。

## 7. 假设判定

支持“过去主要是 step 太少”需要同时满足：

1. S32/S48/S64/S80 至少一个长程 endpoint 相对相同 arm 的 S16 提升超过约 `1 pp`，
   强证据仍要求 `>=1.5 pp`、至少 4/7 分量同向且无关键 collapse；
2. 提升不只来自 reward/format 曲线，而在 CoreDev-2511 外评中出现；
3. 工具调用、格式、长度和视觉健康度没有发生不可接受的退化；
4. 第二个 paired evaluation seed 优先用于确认边界结果，而不是只挑最高 checkpoint。

若五臂普遍在 S16 后持平或回落，则“step 太少”假设不受支持，应转向 reward credit、
数据分布或优化效率，而不是继续机械延长到更多 step。

## 8. 顺序与资源口径

正式启动前先冻结同一实现 commit，并分别做不进入科学结果的一步 smoke。按本期最新资源
优先级，前三条正式顺序为 `B → C → D`：先完成三条自研 T-free 主线，再进入 F/G 与
conditional reward ablation。2026-08-24 的执行修订明确仅自动续接 D；PRL25-E Grounding
留待 D 完成后另行启动，不提前占用训练档。该修订不改变五臂均从 S0 开始的要求，也不改变
A/B、C/E 的 matched 比较定义。

历史 BS16 实测只保留为旧 recipe 的资源背景，不作为新 exact-Crop 路径的保证：

| 执行体 | 实测 | 当前解释 |
|---|---:|---|
| PRL21 旧 pure Crop，BS16 × n16 | `39.68 min/step` | 含旧 logprob/replay 与 checkpoint 路径；不再用作 PRL25-B ETA |
| PRL25-B exact-Crop canary，BS4 × n2 | `262.27 s/step` | publication 前 `162.04 s`、full-Qwen sync `4.09 s`、checkpoint `96.14 s`；仅功能 gate |
| PRL22 pure TGVF，BS16 × n16 | `10.43 min/step` | PRL25-C 的历史 recipe-level 容量锚点 |
| PRL22 Atomic Crop+TGVF，BS16 × n16 | `14.03 min/step` | PRL25-D 的历史 recipe-level 容量锚点 |

PRL25-B 正式 run 于 2026-08-21 00:32:18 JST 启动，run identity 为
`8749332d6031ed87b18c08a91c0cb0590ea7a14c4729300bfe812b3aa44eaca1`。正式首个
BS16 × n16 step 完整发布前，不给出 80-step 窄区间 ETA；canary 与正式 step 的 trajectory
数量相差 32 倍，GPU batching、序列长度和固定 checkpoint 成本又非线性，不能直接乘 32。
PRL25-C/D 也仍在各自首个正式 step 后重新校准；F/G 另计 visual judge 成本。

## 9. 与既有文档的关系

- [PRL24 BS64 计划](PRL24_BS64_POLICY_RL_SCALE_SERIES_PLAN_20260816.md) 保留为原始
  预注册与已完成 A/B/C 的记录，但从本决策起暂停后续 D/E/F。
- [PRL24 A/B/C 阶段结果](PRL24_ABC_INTERIM_RESULTS_20260819.md) 记录 BS64 已有证据与
  D-S1 的非 efficacy 边界。
- [BS16 Crop/TGVF/Crop+TGVF 资料页](BS16_CROP_TGVF_REWARD_ALIGNED_ANALYSIS_20260820.md)
  是历史基线；其中 PRL21/22 使用 FMT1，不能当成本期 FMT2 matched control。
