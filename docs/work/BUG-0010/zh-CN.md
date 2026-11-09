# BUG-0010: 保留 object 数组形状与元素边界

## 背景与目标

修复已复现的 ValueCodec JSON 往返失真，保持已有数组语义，不重建存储架构。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：无新增工作项依赖；从最新主分支开始。

## 范围与合同

使用带版本的 object-array tag，保存 shape、可选 units 和 C 顺序逐元素编码。按存储形状预分配 object 数组，再逐元素赋值，避免 np.array 从元素列表推断新维度。校验非负整数维度与元素数量完全匹配。旧 tag 保持可读及原有限制，不迁移数据库；普通数值数组不变。

## 文件范围

- `pynbodyext/core/calculate/store/codec.py`
- `tests/test_value_codec.py`
- `tests/test_result_store.py`
- `tests/test_sql_store.py`

## 验收标准

- [x] 包含两个列表的向量往返仍为 (2,)，每个列表仍是一个元素。
- [x] 零维 object 标量以及 (0,2)、(2,0,3) 空数组往返。
- [x] 嵌套 dict/tuple、非连续视图、object SimArray 的单位保留。
- [x] 非法维度或 flat 数量不符明确失败；内存和 SQLite 重开后形状、dtype、值一致。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 获得发布授权后开独立 PR；双语交接及红绿证据已在本地准备。

```bash
.venv/bin/python -m pytest tests/test_value_codec.py tests/test_result_store.py tests/test_sql_store.py -q
```

专项预期：新增回归/API 测试先失败，实施后全绿；这是未来执行要求，不是本轮已经运行的结果。

具体步骤见 [实施计划](plan.md)。

## 首个回归用例

```python
import json
import numpy as np
import pytest
from pynbodyext.core.calculate.store import ValueCodec


def object_vector():
    value = np.empty(2, dtype=object)
    value[0], value[1] = [1, 2], [3, 4]
    return value


@pytest.mark.parametrize("value", [object_vector(), np.array(7, dtype=object),
                                  np.empty((0, 2), dtype=object)])
def test_object_array_preserves_element_boundaries(value):
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.tolist() == value.tolist()
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 本地实施 — 2026-11-09

带版本的 object 数组载荷保存 shape、C 顺序元素和单位；解码预分配后逐元素赋值。补充 JSON、内存与 SQLite 重开边界回归，独立代码审查未发现需修正问题。

- Red: 19 failed, 56 passed: shape inference, scalar iteration, lost units and missing new-tag validation.
- Focused green: 75 passed.
- `make check`: 1103 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1107 passed, 1 skipped.

状态为 review：已本地实现并验证，尚未提交、发布或合并。不迁移数据库或改变无关架构。验证范围为 Linux/SQLite，不代表其他平台或后端通过。

## 集成结果 — 2026-11-09

已通过 [BUG-0010 PR](https://github.com/wx-ys/pynbody-extras/pull/42) 合并，merge commit `1b4f1b8f7c3187df2d7f3ac18e6ecdcfe8749e37`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
