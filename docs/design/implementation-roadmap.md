# GPT‑6.1 sol 逐项实施入口

**Integration update / 集成更新 — 2026-11-09:** All authorized increments are now merged; work items are done. See the [integration record](integration-2026-11-09.md) for actual PRs and verification. Earlier local-only statements below describe the recorded pre-integration checkpoint.

2026-11-09 实施更新：BUG-0010–0012 与 FEAT-0005–0012 已逐项本地实现并通过
独立审查，状态 review，未提交或合并。全部合同及发行名称/归属均已获用户批准，
记录为 ADR-011–019；FEAT-0010 另获批准采用根 family 块顺序，跨块重排明确拒绝。
TASK-0002 已实施发行拆分，最终产物复验与检查记录见[安装矩阵](installation-matrix.md)
和对应 handoff。见[确认材料](implementation-approval.md)。
以下基线段落与步骤是原始交接计划，执行证据见各项 handoff。

原始规划基线：`5df15dd`，2026-11-09。以下保留原始设计交接顺序；当前实现状态以上方更新为准。
先读 [设计修订](v1-2026-11-09.md) 与 [审查报告](review-2026-11-09.md)。
每个工作项都有 `en.md`、`zh-CN.md`、`meta.json` 和 `handoff.md`。

## 1. 实施顺序

工作项的唯一状态来源是对应 `meta.json`。下表是依赖和交付顺序，不替代状态。
预计 PR 数以一项一 PR 为默认；若接口评审要求切分，先更新双语 spec，再切分。

| 顺序 | 工作项 | 一次交付的结果 | 前置 / 开始条件 |
|---|---|---|---|
| 1 | [BUG-0010](../work/BUG-0010/zh-CN.md) | object 数组不丢形状/元素边界 | 已复现；可开始 |
| 2 | [BUG-0011](../work/BUG-0011/zh-CN.md) | structured 数组可重建或编码前明确拒绝 | BUG-0010；共享 codec 文件，顺序做 |
| 3 | [BUG-0012](../work/BUG-0012/zh-CN.md) | 缓存补值清理旧错误 | 已复现；可开始 |
| 4 | [FEAT-0005](../work/FEAT-0005/zh-CN.md) | 表格每格保留固定版本来源 | 三项修复 + §5.1 接口评审 |
| 5 | [FEAT-0006](../work/FEAT-0006/zh-CN.md) | Run 保存环境和输入位置事实 | §5.2 接口评审；不依赖 FEAT-0005 |
| 6 | [FEAT-0007](../work/FEAT-0007/zh-CN.md) | 外部文件引用与显式检查 | codec 修复 + §5.3 接口评审 |
| 7 | [FEAT-0008](../work/FEAT-0008/zh-CN.md) | 一个可组合的结果消费协议 | codec 修复 + §6.1 架构评审 |
| 8 | [FEAT-0009](../work/FEAT-0009/zh-CN.md) | 多目录下 halo/subhalo 元数据 | §6.2 身份/层级评审；不要求先写 loader |
| 9 | [FEAT-0010](../work/FEAT-0010/zh-CN.md) | 显式选择重建与执行 | FEAT-0006 + §6.3/Run 位置归属评审 |
| 10 | [FEAT-0011](../work/FEAT-0011/zh-CN.md) | 根快照 JSON job 重放 | FEAT-0006 + §7 清单合同评审 |
| 11 | [FEAT-0012](../work/FEAT-0012/zh-CN.md) | 串行研究报告含每格 Run/ref | FEAT-0005/0006 + §7 批处理合同评审 |
| 可提前 | [SPIKE-0001](../work/SPIKE-0001/zh-CN.md) | 可评审的完整/轻量发行方案 | 只做文档和临时构建实验，可开始 |
| 发布前 | [TASK-0002](../work/TASK-0002/zh-CN.md) | 批准方案的实际安装/平台矩阵 | SPIKE-0001 的发行名称和归属获批 |

“可开始”不表示已经执行。功能项合同现已由用户明确批准并记录 ADR，状态见 metadata；
没有实现的仍不能写成已实现。每项只完成自己的范围，
不要用某项完成代替整个 V1 完成。

## 2. 建议的里程碑

- **M0 数据可靠性**：BUG-0010/0011/0012，全套测试与边界回归通过。
- **M1 可追溯结果**：FEAT-0005/0006/0007。保存表格和 Run 后，数据库重开仍能定位
  固定结果与描述符，能解释本次调用的环境和文件缺失。
- **M2 可组合对象**：FEAT-0008/0009/0010。处理链保持一套 DAG，目录身份不串，
  可重建选择与不可重建历史 view 有明确区别。CatalogueObject loader 仍不在此里程碑。
- **M3 可重放研究**：FEAT-0011/0012。新进程执行同一根输入清单，逐格记录 Run，
  出错保留可恢复证据，不承诺 exactly-once。
- **M4 安装交付**：SPIKE-0001/TASK-0002。以实际 wheel/sdist 和平台证据验收。

V1 原始需求尚保留的目录发现、MergerTree、科学拟合器等见设计 §8；没有从产品愿景
删去，但也没有把缺少科学语义/数据 fixture 的工作包装成“拿去就能实现”的编码任务。

## 3. 复制给 GPT‑6.1 sol 的指令

先从下面这个已具备复现的任务开始：

```text
请在 pynbody-extras 实施 docs/work/BUG-0010。
先读 AGENTS.md、CONTEXT.md、docs/design/README.md，以及该项的 meta.json、
en.md、zh-CN.md、handoff.md、plan.md 和相关现有测试。
仅修复该项，不实施后续功能。保留现有测试；按 spec 添加失败回归，记录红灯原因，
再做最小修复。执行专项测试、make check、make test-all。
更新双语工作项、metadata、handoff，报告源码变更、红绿证据和剩余限制。
不要改写旧整体架构，不迁移数据库，不自动清理历史结果，不触碰无关未跟踪文件。
本条只授权本地实施；是否 commit/push/开 PR/merge 遵循本会话明确授权。
```

以后将工作项 ID 替换为下一个已满足前置的项目。功能项应另加：

```text
先核对该项的接口合同是否已经由负责人批准、对应 ADR 是否已记录。
如果已有本会话批准，不重复请求；将它记录到 ADR 和 handoff 后执行。
如果没有批准，先把该项的具体选择与验收材料整理供评审，不自行改架构。
每次只完成这一项，并验证所有依赖项已经合并到当前基线。
```

这处评审要求来自仓库 `AGENTS.md`：
“Architecture changes (`pynbodyext/core/`, `cpp/`, import layers) need human
review and a recorded decision.” 以及 “If a task is ambiguous about architecture,
stop and ask rather than guessing.” 它不要求对已获批准的同一决定重复确认，也不阻止
上述明确范围的既有合同 bug 修复。

## 4. 每项的交付清单

- [ ] 确认最新 Git 基线、工作区现有变更与前置状态，保留用户已有文件。
- [ ] 先运行相关已有测试，再添加行为回归；记录红灯的准确失败原因。
- [ ] 最小实现，既有公开合同和测试保持完整；架构变化有已审阅决定。
- [ ] 专项与仓库要求检查通过；真实数据/native/公共行为变化跑 `make test-all`。
- [ ] 更新双语 spec、`meta.json`、追加 handoff，生成索引并跑 `make docs-check`。
- [ ] PR 使用仓库模板，包含科学/身份语义、失败恢复、验证与限制。
- [ ] 只有合并且验证后才写 done；未验证平台、未批准功能不得写“已完成”。

没有 GitHub issue 编号的新增项以本地 spec 为任务起点，符合 AGENTS.md 的
“or from the spec that defines the requirement”。本次没有创建远端 issue/PR。
