# FEAT-0005: 将 SampleTable 单元格来源固定到保存版本

## 背景与目标

样本刷新或导出后，仍可准确追溯每格对应的结果版本。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：BUG-0010, BUG-0011, BUG-0012

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-011](../../adr/0011-sample-cell-references.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

实现设计 §5.1：SampleRow 尾部新增有默认值的 refs；SampleTable 增加 keyword-only 可空 store_name。成功后捕获 result.store_ref；失败或成功未保存时清理旧 ref；fill/where/save/load 保留引用。新 sample JSON schema 含 refs 与 store_name，RecordRef.from_json 恢复后校验 pinned、类型化行身份与列签名。load 不访问 backend。旧 schema 可读且 refs 为空，不臆造历史版本。DataFrame 来源以类型化身份索引，避免重复显示名覆盖。

## 已实现行为

`SampleRow.refs` 为尾部带默认值的字段。`add` 和缓存补值复制实际
`Result.store_ref`；重试失败（含 `errors="raise"`）或成功但未保存时移除旧引用。
读取 backend 失败也清除此单元格旧引用，同时保留原值和错误诊断。
fill 对非空格和缓存未命中的格保持不变。`from_store` 和 `where` 保留显式可空
`store_name`。

JSON schema `pynbodyext.sample.table/v2` 保存引用 JSON 字符串和 backend 标签。
导入拒绝非固定版本、错误类型化身份、错误签名和未知列引用。旧版 v1 保留值，
引用为空。load 仅绑定显式传入的 store，从不查询 backend。

DataFrame 元数据含 `store_name`、按行排序的 `identity_keys`，以及用
`sim_identity_key` 索引的 `identities`/`refs` 映射。身份由 `ValueCodec` 编码，
引用由 `RecordRef.to_json()` 编码，重名显示行不会覆盖彼此。
成功 `None` 仍与缺失值共用现有哨兵；本项未扩展为新的单元格状态机。

## 文件范围

- `pynbodyext/core/calculate/sample.py`
- `tests/test_sample_table.py`
- `tests/test_sample_refs.py`
- `CONTEXT.md`

## 验收标准

- [x] 值 6 刷新为 12 后，首张表的 ref 在数据库重开及缓存失效后仍读取 6。
- [x] 命中保留原 ref；无 store、session-only 与失败路径不遗留旧引用。
- [x] JSON/where/DataFrame 保留类型身份和 backend 标签，包括重名显示行。
- [x] 导入时拒绝错误身份/hash/非固定引用；不自动重绑定缺失 backend。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，保留红绿证据并置于 review；本次不提交、不推送、不创建 PR。

```bash
.venv/bin/python -m pytest tests/test_sample_refs.py tests/test_sample_table.py tests/test_result_versions.py -q
```

新增回归/API 测试先于源码改动写入；准确红绿命令、计数及可选 pandas 验证见 handoff。

## 首个回归用例

```python
import pynbody
from pynbodyext.core.calculate import SampleTable, InMemoryResultStore
from pynbodyext.properties import ParamSum


def test_sample_keeps_original_receipt_after_refresh(tmp_path):
    sim = pynbody.new(2)
    sim["mass"] = [6.0, 0.0]
    identity = lambda sim: ("sim", "sample-reference-fixture")
    calc = ParamSum("mass")
    store = InMemoryResultStore()
    first = SampleTable({"mass": calc}, store=store, store_name="results", sim_identity=identity)
    row = first.add(sim)
    pinned = row.refs["mass"]
    sim["mass"] = [12.0, 0.0]
    calc.run(sim, store=store, sim_identity=identity, reuse=False)
    path = tmp_path / "sample.json"
    first.save(path)
    loaded = SampleTable.load(path, store=store)
    assert loaded.rows[0].refs["mass"] == pinned
    assert float(store.fetch(pinned).value) == 6
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；按已批准且本地验证的前置变更顺序推进，不宣称前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。
