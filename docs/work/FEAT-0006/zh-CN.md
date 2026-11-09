# FEAT-0006: 为 Run 保存执行环境和选中输入事实快照

## 背景与目标

记录调用时的已知环境，并与被复用载荷的生产环境区分。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：无新增工作项依赖；从最新主分支开始。

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-012](../../adr/0012-run-execution-context.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

实现设计 §5.2：Run 增加带 schema 的 execution_context JSON；run/run_input/begin_run 增加 keyword execution_metadata，只接受 tool_versions 字符串映射与 random_seed 整数或空。执行前保存 Python/包版本和选中 URI/loader/revision；版本不可得用 null。不为记录 native 状态导入扩展。JSON 校验并复制调用者数据。repository API 不修改快照，直接 session 写入仍由调用者负责。不自动设种子、不改缓存键。

## 文件范围

- `pynbodyext/orm/models.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/environment.py`
- `tests/test_orm_environment.py`
- `tests/test_orm_loading.py`
- `CONTEXT.md`

## 验收标准

- [x] 成功、抛出失败、缓存命中均在 SQLite 重开后保留各自调用快照。
- [x] 后改 DataLocation.uri 或调用者字典不改变旧 Run 快照。
- [x] 缓存命中保存消费者环境与同一固定载荷引用，不声称未知的生产者环境。
- [x] 非法 execution_metadata 在 loader/Run 创建前失败；纯计算导入不引入 ORM。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，按模板整理单独 PR，保留红绿证据。

```bash
.venv/bin/python -m pytest tests/test_orm_environment.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_package_layers_contract.py -q
```

红绿证据和最终本地检查结果见交接记录。

## 已实现行为

`execution_context` 使用 schema `pynbodyext.orm.execution-context/v1` 和
`scope="invocation"`。只读属性每次从已保存的 JSON 文本解码独立副本；repository
完成、失败及恢复操作不替换它。包版本读取发行元数据，发行包不存在记为空；native
信息只读取 `sys.modules` 中已经加载的模块。ADR-019 拆包后，`packages` 除 NumPy
与 pynbody 外，分别记录代码拥有者 `pynbodyext-core` 和完整元包 `pynbodyext`；
core/store 安装记录已知 core 版本，缺少 full 时仅 full 字段为空。

三个入口均在 Run 创建或执行前校验并用 JSON 复制 `execution_metadata`。省略时
tool_versions 为空映射，random_seed 为空；布尔或浮点种子、非字符串工具版本映射及
未知字段抛 ValueError。不设置 RNG 状态，不修改计算器签名或缓存键。

`run_input` 在 loader IO 前保存选中的 URI/前缀、逻辑 loader_id 和 revision。
直接 `run` 以及未选位置的手动 `begin_run` 保存 URI 空值，已有快照文件名不意味着
选择了托管位置。缓存命中时快照属于消费者，固定载荷引用保持原值，生产者环境仍
未知。加载或执行失败保留快照；元数据收尾失败保留最初 running 快照，供显式恢复。
非法元数据、采集失败或初始提交失败均不加载、不计算，也不创建 Run。

## 本地验证 — 2026-11-09

- 红灯：环境与加载模块 **52 failed, 38 passed**，原因是缺少元数据／快照行为
  （`/tmp/feat0006-red.log`）。
- 绿灯：上面的完整专项命令 **121 passed**（`/tmp/feat0006-focused.log`）。
- `make check`：lint、类型与文档检查通过，**1223 passed, 2 skipped,
  4 deselected**（`/tmp/feat0006-check.log`）。
- `make test-all`：**1227 passed, 2 skipped**（`/tmp/feat0006-test-all.log`）。
- 状态：**review**。未执行 Git 修改、创建 PR 或远端合并。已验证 Linux/SQLite；
  其他平台和 SQL 方言未验证。不包含数据库迁移或自动重建生产者环境。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 集成结果 — 2026-11-09

已通过 [FEAT-0006 PR](https://github.com/wx-ys/pynbody-extras/pull/46) 合并，merge commit `e8bdfcb1adf75987d0e88366f823abb95396eaf4`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
