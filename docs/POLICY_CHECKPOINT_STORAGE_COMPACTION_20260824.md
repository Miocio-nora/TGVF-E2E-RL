# Policy-RL checkpoint 存储缩减守则与现存对象盘点

日期：2026-08-24（Asia/Tokyo）

状态：`COMPLETE/PASS / 非 PRL25-B/C：81/81 个科学对象已 compact 并通过全量复核；17 个非科学 payload 与 1 个 failed tree 已删除`

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
才可晋升为 canonical compact checkpoint。删除来源 full state 是显式的破坏性阶段；执行器可以
在同一条逐对象命令中完成 materialize 与 prune，但必须先原子发布并复核 compact receipt、完整
BF16 parameter closure、文件 SHA-256 和双次确定性 CPU generation，任何门禁失败都不得进入 prune。

## 3. 2026-08-24 全库盘点结论

盘点根目录为 `artifacts/policy/`。full-training 对象通过
`actor/model_world_size_*_rank_0.pt` 识别；rolling/permanent hardlink alias 按 rank-0 model
shard inode 去重，大小由每个独立 checkpoint 目录的 `du -B1` 统计。因而下列“对象数”不是路径数。

| 类别 | 独立对象 | 盘点时 full-state 占用 | 当前动作 |
|---|---:|---:|---|
| 正式或诊断身份，转 compact | 103 | 9.918 TiB | 81 个非 B/C 对象执行中；PRL25-B/C 共 22 个严格排除 |
| smoke/canary/lifecycle/invalid，删除候选 | 22 | 2.114 TiB | 17 个非 B/C payload 已删除；5 个 PRL25-B payload 严格排除 |
| 合计 | 125 | 12.032 TiB | 不含非 checkpoint 日志、评测结果和数据 |

初始盘点时文件系统约 `75 TiB`，已用约 `72 TiB`，可用约 `3.2 TiB`（96% used）。用户随后
要求整个 PRL25-B/C family 暂不处理，因此本次 destructive allowlist 不是旧的 92 个“空闲对象”，
而是同时排除 B/C 共 22 个正式对象后的 **81 个**。截至 `2026-08-24 04:02 JST`，34 个 compact
receipt 已落盘；同期文件系统可用空间约 `6.8 TiB`。每个对象仍须逐个通过第 2 节门禁，不能用
累计数量代替单对象验证。

若 103 个对象全部按当前实测 15.89 GiB 转换，并仅为已完成的 PRL25-B/C 各保留一个 full S80，
其 checkpoint 主体约为 `1.60 TiB compact + 0.19 TiB recovery`。这是按现存对象计算的容量口径，
不包含未来 PRL25-D/E/A，也不把已经存在的 compact bundle 重复计算为新增需求。

## 4. 当前 PRL25-B/C 的直接执行口径

两个 80-step arm 各有 11 个独立 full checkpoint 身份：
`S8/S16/S24/S32/S40/S48/S56/S64/S72/S79/S80`。S80 的 rolling 与 permanent 目录是同一物理
对象。两臂合计 22 个身份、2.050 TiB full state。

| Arm | full 身份 | 已有约 16 GiB bundle | 尚需 materialize | full state 释放门禁 |
|---|---:|---:|---:|---|
| PRL25-B exact Crop | 11 | 6：S8/S16/S32/S48/S64/S80 | 5：S24/S40/S56/S72/S79 | 本轮整个 family 严格排除，未处理 |
| PRL25-C pure TGVF | 11 | 6：S8/S16/S32/S48/S64/S80 | 5：S24/S40/S56/S72/S79 | 本轮整个 family 严格排除；六点评测仍在运行 |

两臂完成后目标占用约为 `349.6 GiB` compact checkpoints 加 `190.8 GiB` 两个 full S80，合计
约 `540.4 GiB`。与当前 full state 加已有 12 个 compact bundle 相比，新增 10 个 compact
bundle 后，预计净释放约 `1.71 TiB`；删除 20 个非 S80 full state 的毛释放量约 `1.86 TiB`。

## 5. 初始可缩减对象清单与本轮 allowlist

下表是初始盘点的 103 个对象；大小已按 hardlink 去重。当前执行器的显式 allowlist 只包含前
22 个 run 的 **81 个**对象，最后两行 PRL25-B/C 共 22 个对象全部不在程序 allowlist 中。
截至 `2026-08-24 04:02 JST`，81 个目标中已有 34 个写出 canonical compact receipt 并释放
对应 source aliases；两个分片 worker 正继续处理其余对象。

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

下表是初始的 22 个 full-checkpoint 删除候选。删除后仍保留小体积的 config、log、metrics、
receipt、错误摘要与本清单。依照“整个 PRL25-B/C 暂不处理”的后续指令，前 14 行共 17 个非 B/C
payload 已删除，最后 3 行共 5 个 PRL25-B payload 保持原样。

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
checkpoint，不计入上述 22 个对象；该 failed tree 已与 17 个非 B/C payload 一并删除。不可恢复
删除 receipt 位于
`artifacts/policy/checkpoint-storage-compaction-20260824/non-scientific-deletion-receipt.json`，
记录 18 个 target、apparent `1.695879 TiB`、allocated `1.648638 TiB`，inventory SHA-256 为
`e78abf39c3ebfd0ea885020afa843ae888c5f2863611fde0b99fcf8a7cbf0ed9`。

## 7. 实施顺序与删除门禁

1. 当前只处理第 5 节前 22 个 run 的 81 个非 B/C 正式/诊断对象；逐个 materialize、校验并在
   immutable receipt 落盘后释放 source aliases。两个 worker 使用互斥分片位置，不处理模糊 glob。
2. PRL25-B/C 的 22 个正式对象和 5 个非科学对象均不在当前程序 allowlist；即使显式传入相关
   路径，执行器也按 family 名称 fail-closed。之后是否处理必须另立批次和门禁。
3. 17 个非 B/C smoke/canary/lifecycle payload 与一个 failed tree 已完成不可恢复删除；删除前
   保存逐文件 inventory 和 receipt，未删除任何 run root、日志、metrics 或评测结果。
4. 全部 81 个对象完成后，重新扫描 receipt、BF16 closure、确定性 generation、TGVF sidecar、
   source-alias 消失和剩余磁盘占用；验证失败的对象必须单独返工，不能用总数掩盖。
5. PRL25-C 六点评测和 PRL25-D 训练/评测优先；若缩减 worker 尚未在 D 启动前完成，应暂停存储
   作业，避免其 CPU/I/O 与正式训练竞争。

本文件是当前 storage source of truth；各实验计划只引用本规则，不再各自定义不同的永久
checkpoint 保留方式。

## 8. 最终执行与复核（2026-08-24 05:09 JST）

本轮非 PRL25-B/C 处理已闭环：

- 显式科学 allowlist 的 `81/81` 个对象均已发布 canonical compact receipt，其中
  `10` 个为 full-Qwen，`71` 个为 full-Qwen+TGVF，没有将 Qwen policy-LoRA 误展开为
  full model。compact Qwen 总大小为 `1,421,573,818,442 bytes`。
- 四分片只读终审对 81 个 bundle 全部重新执行 safetensors index/parameter closure、
  全 BF16 dtype、config/processor 加载以及实际 model-tree SHA-256 与 receipt 比对；
  `81/81` 个 success record 且四个 worker 全部以退出码 0 结束。证据在
  `artifacts/policy/checkpoint-storage-compaction-20260824/final-audit/`。
- `71/71` 份 TGVF protocol sidecar 的 manifest/tensor SHA-256、tensor bytes、parameter count、
  optimizer step 和 weights identity 均与 receipt 一致；哈希了 94 个按 inode 去重的实际
  sidecar 文件。全部 81 个对象均有双次确定性 CPU generation 证据；其中 2 个旧
  receipt 通过绑定 receipt/model-tree identity 的 `post-compaction-validation.json` 补齐。
- 81 个对象的 source aliases 剩余数为 `0`。反向扫描确认 PRL25-B 和 PRL25-C 仍各保留
  11 个独立 full checkpoint（S8/S16/S24/S32/S40/S48/S56/S64/S72/S79/S80），两个 family
  均没有 compact receipt；PRL25-B 的 5 个受保护 C0/F2/invalid checkpoint 全部存在，删除
  receipt 中的 B/C target 数为 `0`。
- 文件系统可用空间从初始盘点的约 `3.2 TiB` 上升到终审时约 `12 TiB`（`85%`
  used）。这是全共享文件系统的观测值，不将其全部归因于本任务；本任务的可审计删除量
  仍以第 6 节 receipt 记录的 bytes 为准。
