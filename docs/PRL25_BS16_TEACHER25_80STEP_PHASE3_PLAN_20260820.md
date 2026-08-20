# 第三期（PRL25）：BS16 Teacher25 80-step 统一实验计划

日期：2026-08-20（Asia/Tokyo）

状态：`PLANNED / 尚未启动训练`

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
- **每 1 step** 写一次完整 optimizer recovery checkpoint，rolling 只保留最近 2 个；长期
  固定保留/转存的评测 endpoint 为
  `S0/S8/S16/S24/S32/S48/S64/S80`。中间 endpoint 评测/审计完成后可只保留 model-only
  snapshot 与 receipt；S80 和最终 winner 保留完整 optimizer state，控制约 140 GB/完整
  checkpoint 的存储压力。
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
优先级，正式顺序为 `B → C → D → E → A`：先完成三条自研 T-free 主线，再进入 F/G 与
conditional reward ablation。顺序不改变五臂均从 S0 开始的要求，也不改变 A/B、C/E 的
matched 比较定义。

历史 BS16 实测仅用于容量规划，不作为保证。下列时间包含历史配方当时的 step checkpoint
成本，不包含第三期 CoreDev endpoint 推理/评分：

| 第一批 arm | 历史均值 | 80-step 线性基线 | 排期预留 |
|---|---:|---:|---:|
| PRL25-B Crop T-free | `39.68 min/step` | `52 h 54 min` | `53--60 h` |
| PRL25-C pure TGVF T-free | `10.43 min/step` | `13 h 55 min` | `15--17 h` |
| PRL25-D Atomic Crop+TGVF T-free | `14.03 min/step` | `18 h 42 min` | `20--22 h` |

三条纯训练串行线性基线共约 `85 h 31 min`（3 d 13 h 31 min），实际排期预留约
`88--99 h`（3.7--4.1 天），另加 smoke、失败恢复与 endpoint 评测。每条首个完整 step
结束后只使用本线路实测值更新 ETA，不再从其他工具线做窄区间外推。五臂全部串行的旧
6--7 天估计仍只可作为粗略下限；F/G 还需另外计入 visual judge。

## 9. 与既有文档的关系

- [PRL24 BS64 计划](PRL24_BS64_POLICY_RL_SCALE_SERIES_PLAN_20260816.md) 保留为原始
  预注册与已完成 A/B/C 的记录，但从本决策起暂停后续 D/E/F。
- [PRL24 A/B/C 阶段结果](PRL24_ABC_INTERIM_RESULTS_20260819.md) 记录 BS64 已有证据与
  D-S1 的非 efficacy 边界。
- [BS16 Crop/TGVF/Crop+TGVF 资料页](BS16_CROP_TGVF_REWARD_ALIGNED_ANALYSIS_20260820.md)
  是历史基线；其中 PRL21/22 使用 FMT1，不能当成本期 FMT2 matched control。
