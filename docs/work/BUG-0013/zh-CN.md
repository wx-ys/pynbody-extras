# BUG-0013 · 恢复 Pipeline 分支的变换作用域

## 复现方式

两个具名输出都用 `Disc(3.0, 0.1)` 选择恒星，其中一个先用 `AlignVec` 旋转快照。
分别在新快照上运行时，旋转分支选中的粒子在根快照中的索引为 `[3, 4, 5]`，
原坐标系分支选中 `[2, 3, 5]`。在 `c56120f` 上，先运行旋转分支会使两个输出
都得到 `[3, 4, 5]`。先运行原坐标系分支时，两组索引均正确，但调用方快照的
18 个位置分量中有 6 个在运行后发生了变化。

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py::test_branch_order_restores_entry_frame
```

## 第一阶段验收标准

- **坐标系。** 调换输出顺序，两组结果都应与各自在新快照上单独测量的结果一致。
  每个输出的默认可逆变换在筛选、输入检查、计算和结果物化期间保持生效。无论输出成功
  或失败，后续输出开始前都应将位置和速度恢复到该分支运行前的坐标系。绑定到 Pipeline
  本身的变换对它的全部输出生效。嵌套 Pipeline 在退出时恢复父级坐标系。对子集的变换
  作用于指定的粒子集合，调用方在进入 Pipeline 前持有的原生变换句柄仍由调用方控制。
- **测量结果。** 返回的粒子数组视图保留测量时的数值和单位，即使调用方随后撤销自己的
  变换也能读取原测量值，包括按不同长度分组的粒子数组。预先计算的 Bin 查询在变换后
  的坐标系中测量。逐箱 Calculator 和 Filter 选择器都在处理下一个子集前撤销自己的
  默认变换。稍后执行的惰性 Bin 查询读取届时的粒子数组。
- **变换的生效时间。** 独立运行 Property 或 Transform 时沿用各自的既有行为。在绑定的
  作用域、单个变换或变换链上显式设置 `NEVER`，后续输出和调用方都能看到保留的修改。
  独立运行 Postprocessor 时的异常清理由 [ADR-014](../../adr/0014-dag-value-postprocessors.md) 规定。
- **失败处理。** 逆序撤销已应用的可逆变换，包括变换链中途失败前已经完成的步骤。
  `collect_partial` 模式下，如果测量失败而清理成功，可以继续计算后续输出。如果句柄
  无法恢复坐标系，则停止后续输出。若测量同时失败，优先报告测量异常。失败的 Pipeline
  不会写入成功结果。
- **持久化。** 含有 Pipeline 的计算配方使用新的 `CalculatorSignature` 哈希，包括
  嵌套和自定义 Pipeline。升级后，Database 的 `auto` 模式在第一次请求时重新计算，
  再次请求时复用新版本。固定版本的引用和已冻结的 Study 列仍能读取历史值。独立
  Property 沿用原有键。见 [ADR-022](../../adr/0022-pipeline-transform-scopes.md)。

## 验证

在 `c56120f` 上，两种顺序的回归用例均失败，同批 52 项已有基线测试通过。
SQLite 回归用例先写入旧配方和旧结果，关闭并重新打开 Database，再检查缓存复用、
刷新后的版本历史及已冻结的 Study 列。用例和后续验证见[交接记录](handoff.md)。

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py tests/test_pipeline_scope_versions.py tests/test_postprocessors.py
```
