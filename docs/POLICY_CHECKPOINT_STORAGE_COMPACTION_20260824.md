# Policy-RL checkpoint 存储缩减守则与现存对象盘点

日期：2026-08-24（Asia/Tokyo）

状态：`DECIDED / INVENTORIED / NO DELETION EXECUTED`

## 1. 最终存储口径

checkpoint 的科学身份继续永久保留，但永久保留的默认形态改为约 `16 GiB` 的可独立测评对象，
不再是包含 optimizer/FSDP 恢复状态的约 `95--127 GiB` 完整训练 checkpoint。

- 每个决定保留的 step 都必须有一个 compact evaluation checkpoint。当前 Qwen-only 实测为
  `15.890--15.892 GiB`；“16 GiB”是目标量级，不是逐 byte 上限。
- 每个完成 S80 的正式 arm 额外只保留 **一个**完整 `S80 recovery checkpoint`。S80 因此同时有
  compact evaluation 版本和 full recovery 版本；rolling/permanent 两个路径即使是 hardlink，
  也只能代表同一个物理 recovery 对象。
- 活跃训练期间仍允许每步写 full recovery，并 rolling 保留最近两个；每八步的完整永久快照也可
  在训练和既定评测尚未闭环时暂存。训练、评测闭环后，除 full S80 外全部转为 compact。
- 非科学 smoke、canary、lifecycle gate、明确 invalid/failed 的 full checkpoint 不需要生成
  16 GiB 模型；保留日志、metrics、配置、receipt 和失败证据后，可以删除其训练权重。
- 本守则保留的是 checkpoint 的可测评身份，不承诺保留每个历史 optimizer 状态，也不允许用
  compact checkpoint 恢复训练。

## 2. Compact 对象的最低合同

释放来源 full checkpoint 前，compact 对象必须同时满足：

1. 包含可由正式 evaluator 直接加载的 merged policy weights、config、tokenizer/processor；
2. 保存 run ID、optimizer step、来源 checkpoint 路径与 digest、policy-weight digest、base model
   revision、prompt/tool/reward/protocol 身份；
3. 对 Frozen-TGVF arm 绑定共享只读 RP checkpoint；对 joint arm 额外保存该 step 的可训练
   adapter 权重。共享 frozen RP 不复制进每个 16 GiB Qwen bundle；
4. 通过独立目录加载和至少一条确定性生成 smoke，并通过对应 Crop/TGVF/Atomic 工具协议 preflight；
5. 写入 immutable compaction receipt，记录源/目标 byte 数、文件清单和 SHA-256；
6. 已完成或明确取消所有需要 full source 的既定评测，且没有进程继续打开该 run；
7. 对 S80 arm，唯一 full S80 recovery 已完成 rank/shard、optimizer、scheduler、RNG、data cursor
   和项目 receipt 校验。

现有 PRL25-B `runtime/full-model-hf` 与 PRL25-C `runtime/qwen-only-bundle` 都已证明约 16 GiB
merged Qwen 模型可供 evaluator 加载，但只有补齐上述 provenance、digest 和独立加载 receipt 后，
才可晋升为 canonical compact checkpoint。删除来源 full state 是后续独立的破坏性步骤，不能和
materialize 命令合并执行。

## 3. 2026-08-24 全库盘点结论

盘点根目录为 `artifacts/policy/`。full-training 对象通过
`actor/model_world_size_*_rank_0.pt` 识别；rolling/permanent hardlink alias 按 rank-0 model
shard inode 去重，大小由每个独立 checkpoint 目录的 `du -B1` 统计。因而下列“对象数”不是路径数。

| 类别 | 独立对象 | 当前 full-state 占用 | 当前动作 |
|---|---:|---:|---|
| 正式或诊断身份，转 compact | 103 | 9.918 TiB | 92 个可开始；PRL25-C 的 11 个暂缓释放 |
| smoke/canary/lifecycle/invalid，删除候选 | 22 | 2.114 TiB | 只登记；审批后另行删除 |
| 合计 | 125 | 12.032 TiB | 不含非 checkpoint 日志、评测结果和数据 |

当前文件系统约 `75 TiB`，已用约 `72 TiB`，可用约 `3.2 TiB`（96% used）。进程核对仅发现
PRL25-C 六点评测仍在访问本盘点中的 run；因此当前可进入 compact materialization 队列的是
`103 - 11 = 92` 个对象。这里的“可开始”不等于允许立即删除来源 full state，仍须逐个通过第 2 节门禁。

若 103 个对象全部按当前实测 15.89 GiB 转换，并仅为已完成的 PRL25-B/C 各保留一个 full S80，
其 checkpoint 主体约为 `1.60 TiB compact + 0.19 TiB recovery`。这是按现存对象计算的容量口径，
不包含未来 PRL25-D/E/A，也不把已经存在的 compact bundle 重复计算为新增需求。

## 4. 当前 PRL25-B/C 的直接执行口径

两个 80-step arm 各有 11 个独立 full checkpoint 身份：
`S8/S16/S24/S32/S40/S48/S56/S64/S72/S79/S80`。S80 的 rolling 与 permanent 目录是同一物理
对象。两臂合计 22 个身份、2.050 TiB full state。

| Arm | full 身份 | 已有约 16 GiB bundle | 尚需 materialize | full state 释放门禁 |
|---|---:|---:|---:|---|
| PRL25-B exact Crop | 11 | 6：S8/S16/S32/S48/S64/S80 | 5：S24/S40/S56/S72/S79 | 可实施；最终只留 full S80 |
| PRL25-C pure TGVF | 11 | 6：S8/S16/S32/S48/S64/S80 | 5：S24/S40/S56/S72/S79 | 六点评测正在运行，暂不释放任何 full state |

两臂完成后目标占用约为 `349.6 GiB` compact checkpoints 加 `190.8 GiB` 两个 full S80，合计
约 `540.4 GiB`。与当前 full state 加已有 12 个 compact bundle 相比，新增 10 个 compact
bundle 后，预计净释放约 `1.71 TiB`；删除 20 个非 S80 full state 的毛释放量约 `1.86 TiB`。

## 5. 可缩减对象清单

下表共 103 个对象；大小已按 hardlink 去重。除 PRL25-C 外，盘点时未发现进程使用这些 run。

| Run | 对象数 | Full 占用 | Steps |
|---|---:|---:|---|
| `PRL-13-A ... pilot-bs16` | 1 | 127.02 GiB | S8 |
| `PRL-14-A ... crop-t1-cleanfinal` | 4 | 508.10 GiB | S5/S6/S8/S16 |
| `PRL-15-R1 ... crop16-math-equiv` | 2 | 192.11 GiB | S7/S8 |
| `PRL-16-F0 ... frozen-rp66` | 2 | 191.07 GiB | S7/S8 |
| `PRL-16-F1 ... exact-matched` | 2 | 191.07 GiB | S1/S2 |
| `PRL-16-F2 ... lifecycle-fix` | 5 | 477.68 GiB | S4/S5/S6/S7/S8 |
| `PRL-17-R0 ... rp66-shaped` | 5 | 477.68 GiB | S4/S5/S6/S7/S8 |
| `PRL-17-R1 ... rp67-shaped` | 5 | 477.68 GiB | S4/S5/S6/S7/S8 |
| `PRL-17-R2 ... rp67-tfree` | 6 | 573.22 GiB | S4/S5/S6/S8/S15/S16 |
| `PRL-18-R0 ... joint-rp67` | 3 | 288.18 GiB | S8/S15/S16 |
| `PRL-19-R0 ... visual-api` | 3 | 286.61 GiB | S8/S15/S16 |
| `PRL-20-R0 ... crop-tgvf` | 6 | 573.22 GiB | S4/S5/S6/S8/S15/S16 |
| `PRL-21-R0 ... full-crop` | 4 | 508.10 GiB | S2/S3/S8/S16 |
| `PRL-22-A ... teacher25` | 3 | 286.61 GiB | S8/S15/S16 |
| `PRL-22-B-R0 ... crop-tgvf-teacher25` | 3 | 286.61 GiB | S8/S15/S16 |
| `PRL-23-A ... teacher50` | 3 | 286.61 GiB | S8/S15/S16 |
| `PRL-23-B ... teacher100` | 3 | 286.61 GiB | S8/S15/S16 |
| `PRL-24-A-FMT2 ... BS64` | 5 | 477.68 GiB | S4/S8/S12/S15/S16 |
| `PRL-24-A ... FMT1 BS64` | 7 | 668.76 GiB | S2/S4/S8/S16/S24/S31/S32 |
| `PRL-24-B-FMT2-JOINT ... BS64` | 3 | 288.18 GiB | S4/S7/S8 |
| `PRL-24-C-FMT2-FG ... BS64` | 5 | 477.68 GiB | S4/S8/S12/S15/S16 |
| `PRL-24-D-FMT2 ... sp1` | 1 | 127.02 GiB | S1（runtime/non-efficacy 证据） |
| `PRL-25-B ... exact-crop-80step` | 11 | 1,047.98 GiB | S8/S16/S24/S32/S40/S48/S56/S64/S72/S79/S80 |
| `PRL-25-C ... pure-tgvf-80step` | 11 | 1,050.90 GiB | S8/S16/S24/S32/S40/S48/S56/S64/S72/S79/S80 |
| **合计** | **103** | **9.918 TiB** |  |

PRL24-D S1 只有 runtime/checkpoint 证据、没有 efficacy 结论；依照“不删除正式 checkpoint
身份”的最新决定，它仍归 compact，而不是 smoke 删除候选。

## 6. Smoke/canary 可删除候选

下表只统计 full checkpoint 权重。删除后仍保留小体积的 config、log、metrics、receipt、错误摘要
与本清单；本轮没有删除任何文件。

| Run/子目录 | 对象数 | Full 占用 | 理由 |
|---|---:|---:|---|
| `PRL-13-A/smoke` | 1 | 127.02 GiB | 一步工程 smoke |
| `PRL-15-R0/smoke` | 2 | 96.08 GiB | 两个非科学实现 smoke |
| `PRL-15-R1/smoke` | 1 | 96.05 GiB | matched-horizon smoke |
| `PRL-16-F1/smoke` | 1 | 95.54 GiB | exact-offload smoke |
| `PRL-17-R0/smoke` | 2 | 191.07 GiB | reward-switch smoke v1/v4 |
| `PRL-18-R0/smoke` | 1 | 96.06 GiB | joint-RP67 full-step smoke |
| `PRL-19-R0-C0/canary` | 1 | 95.53 GiB | visual-API canary |
| `PRL-19-R0/smoke` | 1 | 95.54 GiB | full-step smoke |
| `PRL-20-R0-C0/canary` | 1 | 95.53 GiB | Atomic Crop+TGVF canary |
| `PRL-21-R0/smoke-integration` | 1 | 127.02 GiB | Crop integration smoke |
| `PRL-24-A-C0/canary` | 1 | 95.53 GiB | FMT2 preflight canary |
| `PRL-24-B-FMT2-JOINT-C0/canary` | 1 | 96.05 GiB | joint preflight canary |
| `PRL-24-D` 三个 `smoke-integration` | 3 | 381.06 GiB | 三次 Crop 跑通尝试；不含正式 sp1 S1 |
| `PRL-25-B-C0/canary` | 1 | 95.27 GiB | 已由正式 run 取代的功能 gate |
| `PRL-25-B-F2` | 2 | 190.54 GiB | 账本明确的 full-size lifecycle gate，非科学结果 |
| `PRL-25-B ... INVALID-PRE-RESYNC` | 2 | 190.54 GiB | S2/S3 明确 invalid、禁止 resume |
| **合计** | **22** | **2.114 TiB** |  |

另有 `PRL-20-R0-C0 ... failed-fsdp-collective-mismatch` 目录约 `0.13 GiB`，没有形成 full
checkpoint，可作为低优先级失败目录清理项，不计入上述 22 个对象。

## 7. 实施顺序与删除门禁

1. 先把 PRL25-B 已有六个 merged 目录晋升为 canonical compact，并 materialize
   S24/S40/S56/S72/S79；独立加载验证全部 11 个对象。
2. 验证 PRL25-B 唯一 full S80 recovery 后，才释放 S8--S72 与 S79 的 full model/optimizer
   state；保留 S80 compact 与一个 full recovery。
3. PRL25-C 六点评测 `evaluation-complete` 前，不 compact-in-place、不删除、不移动任何 full
   checkpoint。评测闭环后按与 B 相同的 11 compact + 1 full-S80 规则处理。
4. 旧正式/诊断 run 按第 5 节逐个 materialize、校验，再释放原 full state。一次只处理一个
   checkpoint，先记录空间和 digest，再删除来源；失败立即停止，不批量越过错误。
5. 第 6 节删除候选需再次确认没有打开文件或外部依赖，先保存 audit receipt，再单独执行删除。
   不使用模糊 glob，不删除整个 `artifacts/policy` run root。

本文件是当前 storage source of truth；各实验计划只引用本规则，不再各自定义不同的永久
checkpoint 保留方式。
