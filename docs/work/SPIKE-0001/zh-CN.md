# SPIKE-0001: 决定正式的完整与轻量发行结构

## 背景与目标

在改变包归属和依赖策略前，准备可评审的 ADR-005 发行方案。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：无新增工作项依赖；从最新主分支开始。

## 范围与合同

比较轻量核心发行包+完整元包与现有 optional-store 单包。为保留 ADR-005 推荐前者，但不擅选/发布包名。在临时目录制作 wheel 元数据实验，列明每个 pynbodyext 路径、native 二进制、CSS 的归属，证明同时安装没有路径重叠。说明 lockfile、版本发布同步、sdist 行为。维持 optional ORM 须明确修改 ADR-005，extra 不能移除硬依赖。本任务不改正式打包、不发布。

## 文件范围

- `pyproject.toml (read only during spike)`
- `setup.py (read only during spike)`
- `docs/design/packaging-decision.md (create)`
- `tests/test_package_layers_contract.py (read only during spike)`

## 验收标准

- [x] 决策文档包含标为未保留的候选包名、文件归属图与安装/卸载矩阵。
- [x] 临时实验验证完整/轻量依赖元数据，不以 --no-deps 作为受支持路径。
- [x] 记录准确实验、网络/平台限制，并提交具体选择供评审。
- [x] 架构与发行名称批准前 TASK-0002 保留前置依赖。

## 执行步骤与验证

- [x] 读取前置决策与文件，逐项完成上列交付验收。
- [x] 记录准确实验命令和退出码，不把未运行平台标为通过。
- [x] 更新双语文档、metadata 与 handoff，供后续实现交接。

```bash
make docs-check
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 2026-11-09 — local spike / 本地实验

已整理具体[发行方案](../../design/packaging-decision.md)：未保留的候选名称、完整文件归属、依赖元数据、安装/卸载预期及真实临时 wheel/sdist 实验。离线安装因依赖缓存不足未完成；sdist 重建退出 0 但缺头文件导致 native 编译失败并被省略，已列为 TASK-0002 发布验收要求。未改正式打包。

Status: review; human approval pending. Source builds and metadata checks passed, isolated installation and uninstall not certified; TASK-0002 remains blocked. Logs: /tmp/pynbody-packaging-spike/.
