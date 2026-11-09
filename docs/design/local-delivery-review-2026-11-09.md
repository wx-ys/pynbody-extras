# Local delivery review / 本地交付审查 — 2026-11-09

**Mode:** PR Review（本地未提交改动，无远端 PR）
**Scope:** `codex/sequential-review-fixes`，HEAD `5df15dd`；审查 tracked diff
及新增源码、测试、ADR、工作项和打包文件。改动超过 300 行，采用高风险路径抽样；
不将当前文件的全部内容归因于本轮任务。生成的 INDEX、uv.lock 和发行版本字段
核对一致性，不作人工风格审查。
**Health Score:** 100/100（仅表示本次抽样未确认问题，不是全库质量认证）。
**Trend:** 上次 PR Review 为 60，本次 100；两次范围不同，不能据此量化质量提升。

本次审查未发现需要阻止本地交付的具体问题。实施计划已完成，工作项保持 `review`；
后续整合以各项验收和依赖为单位。源码保持原样。

## Findings

未确认新的正确性、合同或可维护性问题。两名独立只读审查者分别覆盖 ORM/重放
与 core/打包边界；根审查者复核安装产物、集成执行、工作项状态与交付范围。

审查覆盖：类型化身份、Catalogue host 关系、Selection 的根位置与 family 顺序、
Job 的注册类型/版本/重建一致性、Study 的全轴预检和失败回执、Run 调用上下文、
SampleTable 固定引用、数组 codec、外部文件描述符、后处理器异常清理，以及
完整/轻量发行的文件归属和版本同步。测试抽查包括行为断言、异常后副作用、
数据库重开和历史值读取；未发现仅断言 mock 调用的新行为保护缺口。

## Verification / 验证证据

| 检查 | 本轮结果 | 证据 |
|---|---|---|
| `UV_CACHE_DIR=/tmp/task0002-cache UV_OFFLINE=1 make check` | 1679 passed、2 skipped、7 deselected；70.88 秒 | `/tmp/pynbody-next-step-check.log` |
| lint / mypy / docs / packaging synchronization | 通过；124 个类型检查文件，27 个工作项 | 同上 |
| `git diff --check` | 通过 | 本轮执行 |
| 已记录 staging 输入 SHA256 对照当前文件 | 34,144 项一致 | `/tmp/task0002-revalidated/source-sha256.json` |
| 最终 wheel/sdist SHA256 与字节数 | 8 项一致 | `/tmp/task0002-revalidated/artifact-hashes.json` |
| 仓库产物检查器 `pair` | core 126 个 import/resource 文件，full 0；无重叠；native 存在，sdist 含 20 个 C++ 输入 | `scripts/check_installation_artifacts.py`；[原产物矩阵](installation-matrix.md) |
| 仓库外 core / full 安装版 probe | 两者通过，native 可用，计算值 6；core 无 SQLAlchemy | `/tmp/task0002-revalidated/{light,full}-env/`，本轮执行 |
| 仓库外 full-wheel 联合流程 | 11 Runs、4 cells，顺序 `[2,0,2]`；值 12 → 24，旧引用仍读 12；外部文件检查为 changed | `/tmp/pynbody-next-step-installed-workflow.log` |
| 独立 ORM 专项复验 | jobs + study：160 passed，10.90 秒 | 独立审查者本轮执行 |
| 独立 structured-array 补充检查 | 12 个 JSON 往返通过，含嵌套字段、padding、空 dtype、零宽字段及不同形状 | 独立审查者本轮执行 |

2 个标准环境 skip 来自可选 pandas。此次未修改运行源码，未重复全量/原生构建；
上轮相同源码的全量结果为标准 1686 passed / 2 skipped，可选 pandas 环境
1688 passed / 0 skipped，C++ ALL PASS。详见安装矩阵与 TASK-0002 handoff。
本轮结果不扩展到 macOS/Windows、其他 Python、其他 SQL 方言或并发压力测试。
临时日志/环境位于 `/tmp`，不是持久交付物，不能用这些路径代替仓库内复现说明。

## Delivery inventory / 交付范围

下表是审阅分组；源码存在共享文件，实际拆分提交时须按 hunk 和前置合同处理，
每个切片重新验证，不能仅按文件路径拆分后沿用整体测试结果。

| 顺序 | 工作项与验收入口 | 核心变化 |
|---|---|---|
| 1 | [BUG-0010](../work/BUG-0010/en.md) / [handoff](../work/BUG-0010/handoff.md) | object 数组形状、元素边界与单位 |
| 2 | [BUG-0011](../work/BUG-0011/en.md) / [handoff](../work/BUG-0011/handoff.md) | structured 数组 NPY 往返及不可表示 dtype 拒绝 |
| 3 | [BUG-0012](../work/BUG-0012/en.md) / [handoff](../work/BUG-0012/handoff.md) | 缓存补值清理旧错误 |
| 4 | [FEAT-0005](../work/FEAT-0005/en.md) / [handoff](../work/FEAT-0005/handoff.md) | SampleTable 单元格固定版本引用 |
| 5 | [FEAT-0006](../work/FEAT-0006/en.md) / [handoff](../work/FEAT-0006/handoff.md) | Run 环境与选中输入上下文 |
| 6 | [FEAT-0007](../work/FEAT-0007/en.md) / [handoff](../work/FEAT-0007/handoff.md) | 显式本地外部文件描述符 |
| 7 | [FEAT-0008](../work/FEAT-0008/en.md) / [handoff](../work/FEAT-0008/handoff.md) | 现有 DAG 上的 PostprocessorBase |
| 8 | [FEAT-0009](../work/FEAT-0009/en.md) / [handoff](../work/FEAT-0009/handoff.md) | Catalogue / CatalogueObject 元数据 |
| 9 | [FEAT-0010](../work/FEAT-0010/en.md) / [handoff](../work/FEAT-0010/handoff.md) | 根相对 Selection 重放与 Run 关联 |
| 10 | [FEAT-0011](../work/FEAT-0011/en.md) / [handoff](../work/FEAT-0011/handoff.md) | 严格根输入 JobSpec JSON |
| 11 | [FEAT-0012](../work/FEAT-0012/en.md) / [handoff](../work/FEAT-0012/handoff.md) | 串行 study 与可恢复证据 |
| 12 | [SPIKE-0001](../work/SPIKE-0001/en.md) → [TASK-0002](../work/TASK-0002/en.md) / [handoff](../work/TASK-0002/handoff.md) | core/full 发行、真实安装/卸载矩阵 |

每组保留双语 spec、metadata、handoff、对应测试和 ADR-011–019。
`CONTEXT.md`、公共 exports、ORM repository/models、codec、SampleTable 和
README/INDEX 包含多个切片的改动。原有 `.develop_notes/`、`.github/`、`.planning/`、
`.test_code/` 与本地审查历史文件不按目录整体加入交付。

## Summary

本地交付审查已完成；下一交付动作是按上述依赖顺序准备提交/PR，再逐项合并。
目前仍保留未提交分支，工作项未标记 done。设计 §8 中的目录发现、对象加载、
MergerTree 和科学拟合器等仍须先满足各自启动条件。

已知安装限制沿用真实矩阵：只验证 Linux x86_64 / CPython 3.11.9；旧单发行
同版本直接 upgrade 未满足文件归属合同，已验证路径为先卸载旧包再安装新 pair。
外部文件的历史引用只固定描述符，不保存被覆盖的文件字节。
