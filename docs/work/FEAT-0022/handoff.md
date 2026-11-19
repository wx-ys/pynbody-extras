# FEAT-0022

## 2026-11-20

Four initial query cases failed at the missing public discovery entry point. Regressions cover class and instance filters, ordered transforms, dependencies, read-only recipe fallback, concurrent refresh and pinned Study cells.

最初四项查询测试因公开结果发现入口缺失而失败。回归用例覆盖 Filter 类和实例、有序变换、依赖、只读旧库、并发刷新和固定的 Study 单元格。

A 1,000-halo query reproduced the SQLite 999-variable limit. A second regression used an eight-variable reader to interleave refresh between batches. Keys, recipes, result versions, reverse uses and composite host addresses are batched on the same read transaction. The scientific control is a six-star rotating sample with κrot = 1 and a projected half-light radius of 3.

1,000 个晕的查询复现了 SQLite 的 999 个绑定变量上限。另用上限为八的只读连接在分批查询之间插入刷新，验证对象键、配方、版本、反向用途和宿主复合键共享同一读事务。六星粒子的运动学对照给出 κrot = 1、投影半光半径为 3。

Integrated verification passed `make check` (1,920 fast tests) and `make test-all` (1,928 tests) with pandas and real data. The complementary empty selection produced its expected warning.

集成验证已通过 `make check`（1,920 项快速测试）和 `make test-all`（1,928 项完整测试），含 pandas 和真实快照。取补集后为空的选集触发了预期提示。
