# FEAT-0012: 记录登记输入与计算器的串行批量研究

## 背景与目标

通过既有受追踪运行，为每格提供值、错误、Run 与固定引用报告。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：FEAT-0005, FEAT-0006

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-018](../../adr/0018-serial-recorded-studies.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

按 §7 提供 run_study(repo, input_ids, calculators, store_name, location_ids=None, reuse=True, errors="raise") -> StudyReport。calculators 是有序标签映射，input IDs 为不重复根，location_ids 按输入映射位置。执行前校验完整列表，串行调用 run_input。errors="record" 只继续 RunExecutionError 或返回的 failed RecordedRun；RunPersistenceError 始终停止，由 StudyPersistenceError 携带部分报告及原回执/异常链。StudyReport JSON 用 ValueCodec 与 pinned refs，load 只读。恢复是明确新调用和新 Run，不声称自动 exactly-once。不让 SampleTable 导入 ORM。

## 文件范围

- `pynbodyext/orm/study.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_study.py`
- `CONTEXT.md`

## 验收标准

- [x] 两个输入×两个计算器产生四个 Run 及正确对应单元格。
- [x] 重复研究新增四 Run 但复用载荷 ref；刷新及数据库重开后旧报告 ref 仍可读。
- [x] record 模式科学失败留下 error/run 单元格并继续。
- [x] 元数据完成失败停止后续单元格，暴露部分进度与回执，不自动关联/删除。
- [x] JSON 保留值、单位、typed refs、错误；重复输入/标签及未知位置映射预检拒绝。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，保留红绿证据。
- [ ] 父任务集成流程：获准发布时按模板整理独立 PR。

```bash
.venv/bin/python -m pytest tests/test_orm_study.py tests/test_orm_associations.py tests/test_sample_refs.py tests/test_package_layers_contract.py -q
```

## 本地实现与验证

ORM 层已实现 `run_study`、冻结的 `StudyCell`/`StudyReport` 和 `StudyPersistenceError`。完整列表只读预检覆盖有序标签计算器、唯一正数 exact-int 根输入 ID、loader/store/location 绑定与所有权、既有 Recipe 载荷冲突；通过前不创建 Run、不调用 loader、不写 Recipe。支持 stable 但不可重构的 live calculator 及其已配置选项，不套用 JobSpec 重构限制。空轴在校验请求后仍合法。

按输入优先、标签映射顺序串行委托 `run_input`，保存实际值、结构化错误、Run ID 与实际 pinned Result ref。重复调用创建新 Run 并可复用同一载荷 ref；刷新和数据库重开后旧值及旧 ref 保留。record 模式只继续已登记的执行失败或返回的 failed RecordedRun。任何 RunPersistenceError 均立即停止；StudyPersistenceError 保留原异常回执对象、原 Result/ref 别名、异常链、中断坐标和此前完成格。恢复仅由调用者显式执行，中断格不冒充已完成格。

版本化报告 JSON（`pynbodyext.orm.study/v1`）采用 ValueCodec 与 typed RecordRef JSON，保留单位、错误、None、嵌套 NumPy 值及完整请求轴，拒绝畸形/重复字段和不匹配/未固定 ref；load 不访问后端、不重构 calculator。checkpoint 不自动恢复或关联 Run；明确新调用仍产生新 Run。

红灯证据：**91 failed**，均为缺少新 API（`/tmp/feat0012-red.log`）；初次绿灯 **91 passed**（`/tmp/feat0012-green-attempt.log`）。另四个嵌套 ref JSON 畸形回归先 **4 red**（`/tmp/feat0012-ref-red.log`），加入严格嵌套 JSON 校验后转绿。最终源码专项 **193 passed, 1 skipped**，17.76s（`/tmp/feat0012-focused.log`），包含全部 **95 个 study 用例**、既有 ORM/loading/SampleTable refs 及分层/calculator 合同。专项跳过项需要可选 pandas。`make check` 的 lint/mypy（124 源文件）/docs 通过，**1670 passed, 2 skipped, 5 deselected**，70.78s（`/tmp/feat0012-check.log`）；`make test-all` **1675 passed, 2 skipped**，72.39s（`/tmp/feat0012-all.log`）。全量两个跳过项均需可选 pandas。SQLite/Linux 验证不代表其他 SQL 方言或平台认证。源码和测试已冻结，等待父任务审查；本地前项验证不代表远程合并，不声称 Git 或远程操作。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务验证本地前项，不假称远程合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。
