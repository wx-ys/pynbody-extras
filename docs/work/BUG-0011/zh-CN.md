# BUG-0011: 结构化 NumPy 数组无损往返，避免解析 dtype 展示字符串

## 背景与目标

修复结构化数值数组编码成功却无法解码的问题。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：BUG-0010

## 范围与合同

增加专门的带版本 structured-array NPY tag，使用 BytesIO，np.save/np.load 均设 allow_pickle=False。在普通原始字节分支前识别 dtype.fields。保留 shape、字段名/格式/offset/itemsize、字节序及适用时的单位。structured dtype 的 hasobject 在编码前明确拒绝，禁止保存对象指针字节。不能无损表示的 dtype 在编码时清晰 TypeError，不扩大 opaque pickle 行为。

dtype metadata（包括嵌套字段上的 metadata）本增量明确不支持，必须在 NPY 序列化前报错，不能静默丢弃。NPY 拒绝的重叠或乱序字段布局也在编码阶段失败。

## 文件范围

- `pynbodyext/core/calculate/store/codec.py`
- `tests/test_value_codec.py`
- `tests/test_sql_store.py`

## 验收标准

- [x] 简单/嵌套 records、subarray 字段、对齐后的 offsets/itemsize 经过 JSON、SQLite 重开仍一致。
- [x] 零维和空 structured 数组往返。
- [x] 含 object 字段在保存记录前拒绝。
- [x] 截断 NPY 和含 object 的 NPY 载荷失败且不调用 pickle；保留旧数值数组与 SimArray 测试。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [ ] 获得发布授权后开独立 PR；双语交接及红绿证据已在本地准备。

```bash
.venv/bin/python -m pytest tests/test_value_codec.py tests/test_sql_store.py tests/test_result_versions.py -q
```

专项预期：新增回归/API 测试先失败，实施后全绿；这是未来执行要求，不是本轮已经运行的结果。

具体步骤见 [实施计划](plan.md)。

## 首个回归用例

```python
import json
import numpy as np
from pynbodyext.core.calculate.store import ValueCodec


def test_structured_array_roundtrip():
    value = np.array([(1, 2.0)], dtype=[("id", "i4"), ("mass", "f8")])
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 本地实施 — 2026-11-09

新增关闭 pickle 的 structured_array_v1 NPY 编码，保留形状、字段布局、字节序与单位；在编码时拒绝 object 字段、嵌套 metadata 和无法读取的 header。用零元素数组校验 dtype，支持载荷大小范围内的大型 header。JSON、SQLite 重开、拒绝前无提交与损坏载荷回归均通过，复审无待修正问题。

- Red: 40 failed, 103 passed; reviewer regressions 3 failed, 1 passed (unreadable titles and wide header).
- Focused green: 147 passed; re-review no actionable findings.
- `make check`: 1151 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1155 passed, 1 skipped.

状态为 review：已本地实现并验证，尚未提交、发布或合并。不迁移数据库或改变无关架构。验证范围为 Linux/SQLite，不代表其他平台或后端通过。
