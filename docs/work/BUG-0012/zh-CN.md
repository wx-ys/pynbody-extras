# BUG-0012: 缓存成功补值后清理单元格旧错误

## 背景与目标

通过存储结果恢复失败单元格时，让值与错误状态保持一致。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：无新增工作项依赖；从最新主分支开始。

## 范围与合同

fill_from_store 命中并成功取得值后，设置值并删除对应列错误。miss 保留错误；store 异常沿用既有传播行为，值未取得前不清除错误。不更改 None/missing 哨兵或错误策略。

## 文件范围

- `pynbodyext/core/calculate/sample.py`
- `tests/test_sample_recovery.py`
- `tests/test_sample_table.py`

## 验收标准

- [x] 失败记录后，另一运行填入缓存，补值后有结果且无旧错误。
- [x] 另一未命中列保留自身错误；非空列保持不变。
- [x] backend 读取失败时原诊断不被提前清除。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 获得发布授权后开独立 PR；双语交接及红绿证据已在本地准备。

```bash
.venv/bin/python -m pytest tests/test_sample_recovery.py tests/test_sample_table.py -q
```

专项预期：新增回归/API 测试先失败，实施后全绿；这是未来执行要求，不是本轮已经运行的结果。

具体步骤见 [实施计划](plan.md)。

## 首个回归用例

```python
from typing import ClassVar
import pynbody
from pynbodyext.core.calculate import InMemoryResultStore, PropertyBase, SampleTable


@PropertyBase.dataclass
class RecoverableForReview(PropertyBase[float]):
    signature_name = "tests.review.RecoverableForReview"
    failing: ClassVar[bool] = True

    def calculate(self, sim, params=None):
        if type(self).failing:
            raise RuntimeError("temporary failure")
        return 7.0


def test_store_fill_clears_only_recovered_error():
    sim = pynbody.new(1)
    identity = lambda sim: ("sim", "recovery-fixture")
    store = InMemoryResultStore()
    calc = RecoverableForReview()
    table = SampleTable({"value": calc}, store=store, sim_identity=identity)
    RecoverableForReview.failing = True
    try:
        row = table.add(sim, errors="record")
        assert "value" in row.errors
        RecoverableForReview.failing = False
        calc.run(sim, store=store, sim_identity=identity)
        table.fill_from_store()
        assert row.values["value"] == 7.0
        assert "value" not in row.errors
    finally:
        RecoverableForReview.failing = True
```

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 本地实施 — 2026-11-09

成功从缓存赋值后，仅清理该列旧错误。补充真实失败后恢复、未命中列、已有值与后端读取异常回归；保留现有缺失哨兵和异常传播合同，独立审查无问题。

- Red: 1 failed, 26 passed, 1 skipped: successful cache fill retained the old error.
- Focused green: 27 passed, 1 skipped; independent review no findings.
- `make check`: 1155 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1159 passed, 1 skipped.

状态为 review：已本地实现并验证，尚未提交、发布或合并。不迁移数据库或改变无关架构。验证范围为 Linux/SQLite，不代表其他平台或后端通过。

## 集成结果 — 2026-11-09

已通过 [BUG-0012 PR](https://github.com/wx-ys/pynbody-extras/pull/44) 合并，merge commit `1c35d9892a93f5d175b86cf432065c0f389d4c74`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
