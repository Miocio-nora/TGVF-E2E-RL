# 主代码整合与审计进度（2026-08-31）

本文档是当前论文实验代码的专属整合台账。它记录“统一分支上真实可用的代码”，不把历史 worktree 中出现过、但尚未移植和验证的实现算作完成。

当前工作分支：`stabilize/protocol-contract-v1-20260830`。

## 当前结论

仓库已经完成了一部分结构性整理，但尚未形成 NoTool、Crop、TGVF、Atomic 四方法共享的可运行 `train -> pause -> resume -> eval` 主链。当前最主要的问题不是缺少启动安全检查，而是：

1. 四方法核心实现仍散落在多个历史 donor 分支；
2. canonical policy 目录没有可运行的四方法配置；
3. NoTool 和 Atomic 的 live runtime 路由缺失；
4. Crop/TGVF 的 full-model replay、checkpoint 和 weight-sync 实现尚未完整移植；
5. 恢复身份和评测拓扑混入过多运行参数，导致正常开发和中断恢复被拒绝。

因此，现阶段主线目标是恢复科研代码的可运行性和一致性。runtime ZIP、trampoline、一次性 token、父进程 liveness 等启动封锁工作已经暂停，不再作为主线前置条件。

暂停代码已可恢复地保存在：

`stash@{0}: paused-runtime-zip-trampoline-overdesign-20260831`

## 四方法整合矩阵

| 方法 | 环境工具 | 精确 Prompt | 数据/Schema | Live runtime | 训练 replay/engine | 当前判定 |
|---|---|---|---|---|---|---|
| Original / NoTool | 不需要工具 | 正在整合 direct-only prompt | 缺统一 @512 schema | 正在整合 | 缺统一 full-Qwen 路径 | 未闭环 |
| Crop | Crop 环境层基本存在 | 历史 DeepEyes prompt 存在 | 缺统一 canonical 配置 | 部分存在 | full-model Crop replay/engine 待移植 | 未闭环 |
| TGVF | TGVF 环境层基本存在 | Short 与 Target-guide-only 已移植 | Teacher25/schema 待接入 | 部分存在 | trainable TGVF replay/engine 待移植 | 未闭环 |
| Atomic | Atomic 环境层存在 | matched prompt/schema 待移植 | Teacher25/schema 待接入 | 当前显式拒绝 | Atomic RPC 与训练接线缺失 | 未闭环 |

“Original”在文中必须继续作为比较基线；代码层面将它作为 NoTool/direct-only 方法，而不是无身份的特殊分支。

## 已完成并进入当前分支的核心修复

### 1. 已完成实验配置可以再次审计

`validate-representation-config` 是只读命令。它现在允许目标内部评测报告已经存在，因此 RP71–RP74 等已完成实验的配置可以重新验证。

训练和 worker 入口仍保留“不得静默覆盖已有报告”的检查。

验证：四份 RP71–RP74 treatment 配置均已通过真实外部文件校验。

### 2. Representation 中断恢复接入 metrics WAL 回滚

仓库原先已经有 `recover_representation_metrics_history_prefix()`，但 runner 没有调用。现在 resume 时：

1. rank 0 先读取并验证 checkpoint metadata、run identity 和 global step；
2. 使用 checkpoint 绑定的 metrics identity 验证已提交前缀；
3. 将 checkpoint 之后未提交的 metrics suffix 完整归档；
4. active metrics JSONL 原子恢复到 checkpoint 前缀；
5. 将同一 identity broadcast 给所有 rank，再执行分布式 checkpoint restore。

合法中断可以恢复；已提交历史的字节漂移仍然硬拒绝。

验证：全部 representation training CPU 测试 `460 passed`。

### 3. Policy 评测不再错误地强制四张 GPU

评测结果 identity 本来就记录 world size，任务也按 world size 确定性分片。代码现在接受至少一张、互不重复且非负的 GPU ID。

论文 golden 配置仍可以固定四张 GPU；单卡开发评测不再被错误当成协议不合法。像素、subset、checkpoint、prompt、工具协议和评分身份没有因此放宽。

### 4. TGVF Target-guide-only prompt 已移植

当前实现明确区分：

- Short：原 matched TGVF prompt；
- Target-guide-only：只增加 Target 的视觉化定义与 teacher 风格示例。

Target-guide-only 不修改 `<think>`、final-only、observation 文本、工具调用次数或 action boundary。新增示例采用：

- `small circular gauge, its needle position, and surrounding scale markings`
- `printed text below the red warning symbol`
- `wide shared view containing the bicycle, the parked car, and the space between them`

测试证明移除 Target guide 后可逐字节恢复 Short system prompt，user message 和 tool parser identity 保持一致。

## 恢复与可复现边界审计

当前代码把三类不同概念混成一个 fail-closed identity：

- checkpoint 物理兼容性；
- 实验语义身份；
- 本次调用的运行环境与输出路径。

应继续硬拒绝的项目包括：损坏 checkpoint、模型/Adapter 结构不兼容、tensor shape 或 optimizer 拓扑不兼容、无法加载的 FSDP/world-size 拓扑、数据 cursor 不可应用、checkpoint pair 不完整。

不应阻止普通恢复的项目包括：物理 GPU ID、输出/日志路径、W&B project、checkpoint cadence、validation cadence、未超过 scheduler horizon 的目标 step、timeout/capacity、TOML 注释和空白。

会改变实验语义但物理上可加载的变化，例如 prompt、tool schema、reward、数据、采样和 optimizer 超参数，应作为显式 fork 记录 provenance，而不是伪装成相同实验。

计划采用两种边界：

- `dev`：可运行、可中断恢复；只对真正不兼容的 checkpoint 硬拒绝，并记录所有差异；
- `repro`：用于论文 golden 结果资格审计，严格绑定模型、数据、prompt、像素、工具协议、reward、seed、subset、scorer 和 checkpoint。

严格复现是结果资格检查，不再等同于“是否允许启动 Python”。

## 待整合的历史核心切片

以下 commit 只是 donor 来源，尚未因为出现在历史分支就自动算作当前主线能力。移植时按当前模块边界重写，不整块合并旧 launcher/supervisor：

| 目的 | donor commit |
|---|---|
| Teacher25 quarter-mix | `37b99e2` |
| NoTool direct-only substrate | `f9dff1f` |
| Crop full-model replay/engine | `762e43f` |
| TGVF replay/engine/checkpoint/weight-sync | `8a2a50d` |
| Atomic RPC/runtime | `eadae55` |
| T-free reward optional wiring | `2c1039e` |
| @512 method schemas | `e756546` |
| Target-guide-only prompt | `396a258`（核心 prompt 已移植） |
| Crop live/replay byte parity | `c448e583` |
| Atomic @512 schema/runtime binding | `8e6b3d6` |

## 下一阶段验收顺序

### P0：CPU 可组合

- NoTool/Crop/TGVF/Atomic 四份配置由同一 loader 解析；
- 统一绑定 `262,144 = 512^2` pixels、S32、seed 42 和 Teacher25；
- 四方法都明确记录 prompt、action boundary、observation 和 tool schema identity；
- Original/NoTool 保持零工具调用；
- 不依赖 OpenRouter 或外部 judge 才能完成配置和 runtime composition 测试。

### P1：工具与 replay 一致

- Crop/TGVF/Atomic 至少各有一次成功调用的 CPU 合约或最小 GPU canary；
- `</tool_call>` 后缀、多调用、畸形输出必须零执行、零工具奖励；
- Crop live 与 replay 使用相同图像字节和 token；
- TGVF/Atomic 保持 RP67 Adapter hash 不变，只更新并同步 Qwen 参数。

### P2：恢复与评测闭环

- step 1 checkpoint 可 teardown/resume；
- 只改变输出路径、日志、GPU placement 或 checkpoint cadence 不阻止兼容恢复；
- 模型、Adapter、tensor topology 或 checkpoint 损坏仍然拒绝；
- 同一 evaluator 支持 Original/NoTool、Crop、TGVF、Atomic，且记录完整 subset、pixel、prompt、工具调用次数/频率和 checkpoint provenance。

GPU 训练和正式 benchmark 只在 CPU 合约闭环后启动。
