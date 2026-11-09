# TASK-0002: 实施并验证批准后的安装矩阵

## 背景与目标

落地经批准的 ADR-005 机制，在干净环境验证实际构建产物。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：SPIKE-0001

## 范围与合同

此任务有前置决策，不能由实现者擅选发行结构。先把批准的准确包名和文件归属写入具体实施计划。pyproject 与 uv.lock 同步更新；构建 wheel/sdist，测试安装产物而非 editable 源码。覆盖完整 ORM、轻量纯计算、CSS、可选 native 导入失败及受支持 native 构建。先确认仓库拥有的 CI 位置，不擅自覆盖已有未跟踪 .github；归属不明可用本地脚本。不在验证中发布版本。

## 文件范围

- `pyproject.toml`
- `uv.lock`
- `setup.py`
- `README.md`
- `README.zh-CN.md`
- `tests/test_installation_contract.py (create)`
- `docs/design/installation-matrix.md (create)`

## 验收标准

- [x] Linux/macOS/Windows 与支持 Python 组合逐格记录通过/失败/未运行及证据。
- [x] 完整安装能运行 ORM SQLite 用例；轻量环境确无 SQLAlchemy 且可计算。
- [x] 卸载元包不删除核心拥有的文件；wheel/sdist 包含必需资源。
- [x] 声称支持的平台通过 native 有/无路径及 make check/test-all。

## 执行步骤与验证

- [x] 读取前置决策与文件，逐项完成上列交付验收。
- [x] 记录准确实验命令和退出码，不把未运行平台标为通过。
- [x] 更新双语文档、metadata 与 handoff，供后续实现交接。

```bash
make check && make test-all
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 2026-11-09 — evidence update / 实验补充

SPIKE-0001 临时实验确认：core sdist 缺 C++ 头文件；重建因 optional-native 降级仍退出 0，实际没有 native 二进制。发行方案获批后需补齐头文件，并在声称支持的矩阵中断言二进制存在及可导入，不能只看构建退出码。见[实验证据](../../design/packaging-decision.md)。

## 2026-11-09 — approved distribution decision / 发行决定获批

The user explicitly approved names and ownership in [ADR-019](../../adr/0019-full-core-distribution-ownership.md); the decision gate is satisfied. Actual installation and platform validation remain required.

## 实施合同（ADR-019）

FEAT-0012 完成后执行。根 pyproject 变为 `pynbodyext-core`，独占全部 import、
native 和 CSS 文件。新增 `packages/pynbodyext/pyproject.toml` 为可独立构建的纯元数据
完整成员，强制精确同版本 core 与 SQLAlchemy，转发 image/store/dev/bench extras。
新增确定性的版本同步脚本、MANIFEST.in 与产物验证脚本；同步更新 Makefile、uv.lock。
验证实际 sdist 头文件与 PEP 517 重建、源码安装命令、OpenMP/串行/无编译器路径、
卸载归属以及旧版同版本迁移。保守迁移先卸载旧完整发行，再引入 core。
原文件范围据此扩展至必要的成员、脚本、manifest、CONTEXT 和本地验证文档；不发布，
不擅改 CI 归属。

uv 0.9.17 的动态 editable core 锁条目不含 version 字段；检查它的源码映射及实际
构建版本，full 的生成静态版本则写入 uv.lock。这是解析器格式，不放宽完整/core
精确同版本要求。

## 已实施并验证（2026-11-09）

ADR-019 已落地，TASK-0002 进入 review。见[实际安装矩阵](../../design/installation-matrix.md)：
Linux x86_64 / CPython 3.11.9 的完整/轻量 wheel 与 sdist 安装、头文件/CSS、
PEP 517 重建、extras、源码安装、OpenMP/串行/无编译器及卸载归属通过；其他平台
和 Python 明确标记未运行。同版本 pip 直接 upgrade 未满足文件归属合同；已验证的
迁移先卸载旧单发行包，再安装新的一对发行包。新 core[store]/full 的实际 Run
分别记录代码包版本，不把可选 full 元数据缺失误报为未知代码版本。

最终 make check：1679 passed、2 skipped、7 deselected；标准 make test-all：
1686 passed、2 skipped；启用可选 pandas 的全量测试：1688 passed、零跳过。
9 个打包回归通过，原生 C++ ALL PASS。provenance 修正后，已从包含所有新功能的
当前源码重新构建最终产物。未发布、未 commit/push、未改用户工作 .venv，
未覆盖用户未跟踪文件。

## 集成结果 — 2026-11-09

已通过 [TASK-0002 PR](https://github.com/wx-ys/pynbody-extras/pull/53) 合并，merge commit `7417d949c57e01c2dcc4ac609005eaaf869eddcc`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
