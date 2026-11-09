# FEAT-0010: 重载显式根相对粒子选择

## 背景与目标

凭保存的选择定义重建已登记视图，不从索引摘要逆推粒子集合。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：FEAT-0006

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-016](../../adr/0016-replayable-particle-selections.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

按 §6.3 定义 indices/family/filter 三类 SelectionSpec，绑定已登记根及预期粒子数。register_selection 接受 LoadedInput，登记时求值并校验，保存实际 provider 身份与定义。indices 为内建整数 JSON 列表，在遵循根 family 块顺序的条件下保序保重复，拒绝 bool、负值、越界。Filter 签名须 stable 且 constructible；只登记无外部变换根。load_selection(selection_id, location_id=None) 返回视图与根 provider 的 LoadedInput；run_selection 委托既有运行流程并记录失败。位置归属须检查根，这是对 Run 根加载约束的显式扩展，需要 ADR 和回归。既有任意 view 的 load_input 拒绝仍保留。

Selection 表保存 root_input_id、view_input_id、schema、definition_json、expected_count 和唯一规范描述符 key。不同定义可以共享同一实际 view DataObject，不互相覆盖定义。Run 增加可空 selection_id 以追溯实际采用的定义。已实现 API：SelectionSpec.indices(list[int])、SelectionSpec.family(str)、SelectionSpec.filter(calculator)；repo.register_selection(loaded_root, spec) -> Selection；repo.get_selection(selection_id)；repo.load_selection(selection_id, *, location_id=None)；repo.run_selection(calculator, selection_id, *, store_name, location_id=None, reuse=True, options=None, execution_metadata=None)。登记时从 loaded_root 得出 expected_count，不信任调用者填写。run_selection 在元数据预检后、加载前提交 running；完成和恢复时同时验证实际 view 与选用定义。

## 已批准的 family 顺序修订 — 2026-11-09

用户明确选择方案 A：“首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”。本修订在修改对应验收测试之前取代原先不限制跨 family 顺序的要求。索引逐项保序保重复，但所选 family 块必须遵循根的 family 顺序。以 dm 在前、star 在后且各两粒子的根为例，`[1,0,1,3,2,3]` 合法且原样保留；`[3,1,3,0]` 或 `[3,1]` 明确 ValueError，不排序、不写入元数据。若重放根的 family 布局使原序列变得不受支持，run_selection 在创建 Run 后记录 failed，且不执行分析。上游连续 family 视图约束见[已批准的决定](family-order-decision.md)。本项不实现自定义视图适配器。

## 文件范围

- `pynbodyext/orm/selections.py`
- `pynbodyext/orm/models.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/loading.py`
- `tests/test_orm_selections.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_associations.py`（已批准的元数据表预期）
- `tests/test_orm_loading.py`（保留既有回归）
- `CONTEXT.md`

已批准的 Selection 模型在 FEAT-0009 之后新增一张元数据表（9 增至 10）。更新 test_orm_associations.py 的表集合/数量预期，并保留全部“不包含载荷列”断言；Run.selection_id 是显式元数据关联。

## 验收标准

- [x] 重开元数据后恢复同一有序索引及类型化 provider 身份。
- [x] family 与可重建 filter 选中预期粒子；family 缺失或 mask 长度错误明确失败。
- [x] 根 revision/粒子数变化或实际选择摘要改变时在计算前拒绝。
- [x] Run 创建后的加载/选择失败记 failed；最后元数据失败保留回执，恢复校验根位置。
- [x] 无 revision 时不保证检测同数量重排/编辑，文档和测试只承诺已定义范围。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，保留红绿证据。
- [x] 父级流程：按模板整理单独 PR。

```bash
.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_loading.py tests/test_logical_snapshot_identity.py tests/test_orm_associations.py -q
```

## 本地实施证据

SelectionSpec 使用不可变规范 JSON，严格校验 schema/kind/字段和完整的 stable/constructible FilterBase 签名。indices 拒绝 bool、NumPy 整数、负数和越界，并遵循已批准的 family 块顺序修订。family 缺失、错误/错误长度 mask 和实际摘要改变均明确失败。Filter 在原坐标系通过既有 DAG 求值，拒绝 scoped transform，求值后重新验证根身份；不新增持久 mask 缓存。

Selection 分别保存根/视图关联、派生的预期根粒子数、定义和完整规范描述符 hash。不同定义可以共享同一 view 而不覆盖，完全相同的定义幂等登记。新 view 与 Selection 在同一事务插入，注入 Selection INSERT 失败后同时回滚。get_selection/load_selection 返回已加载直接根/视图层次的 detached 元数据；即使存在 Selection，load_input 仍只加载根。

run_selection 在 loader/selector IO 前提交视图 Run、准确 selection_id、选中根位置和复制的 invocation context，再委托 _execute_run。保留命中/刷新与 collected-errors 行为，并接受与 run_input 相同的 execution_metadata。仅完全匹配且通过验证的 Selection 允许 Run 使用根所属位置，不能仅凭 DataObject.parent 推断。finish/fetch/audit/unlinked_versions/恢复统一校验描述符、根/视图层次、准确的视图回执与根位置归属；相同固定回执恢复幂等，不猜 latest。audit 不加载快照、不解码结果值。

初始红灯：因缺少 SelectionSpec/选择 API，**58 项失败**（/tmp/feat0010-red.log）。按已批准 family 顺序修订，显式拒绝回归先有 **3 项红灯**（/tmp/feat0010-family-red.log）；严格签名 envelope 先有 **3 项红灯**（/tmp/feat0010-red-validation.log）。随后 scope=null 回归先复现 AttributeError，再变为明确 ValueError，红绿证据见 /tmp/feat0010-scope-red.log、/tmp/feat0010-scope-green.log。增加最后一项 scope 回归前，本模块已 **86 passed**（/tmp/feat0010-green.log）；最终验证包含 **87 项新增测试**，覆盖 pynbody.load 真实文件重放、原生 view、SQLite 约束、登记后数据/family 布局变化和元数据/载荷失败注入。

最终源代码上的完整套件：**1515 passed、2 skipped**，60.06s（/tmp/feat0010-all.log）；两项跳过均因可选 pandas 缺失。最终专项 **326 passed**，33.55s（/tmp/feat0010-focused.log）；make check 通过 lint、mypy（**122 个源文件**）、**1510 passed、2 skipped、5 deselected**，57.94s，并通过文档校验（27 个工作项），见 /tmp/feat0010-check.log。最终冻结代码的独立合同与质量审查均无发现。状态为 review；未执行 Git 操作，不宣称远程合并、PR 或 release。已验证范围为 SQLite/Linux；其他 SQL 方言/平台和支持任意 family 顺序的新视图不在本项内。不新增迁移/依赖、不改核心 DAG/provider、不新增自动内容指纹。无新 revision 的同数量编辑/重排可能保留原 key，测试明确要求 reuse=False 或新 revision 才保证更新计算。元数据表数量 9 增至 10，全部无载荷断言仍保留。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先验证本地前项，不假称远程已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 集成结果 — 2026-11-09

已通过 [FEAT-0010 PR](https://github.com/wx-ys/pynbody-extras/pull/50) 合并，merge commit `448fe1156f19d32aed42118cd108c1cfe0cb6c74`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
