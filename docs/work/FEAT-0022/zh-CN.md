# FEAT-0022 · 发现已存输出并反查粒子成员

所有条件必须命中同一个具名输出，返回固定版本引用。历史输出与实际成员来源支持双向追溯，只读库可按配方元数据查询。 [实施计划](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

最初四项查询测试因公开结果发现入口缺失而失败。回归用例覆盖 Filter 类和实例、有序变换、依赖、只读旧库、并发刷新和固定的 Study 单元格。

具名叶输出保留根结果版本及各自的输出路径。只读查询和固定的 Study 单元格在另一连接刷新时仍能返回原版本对应的成员证据。

[tests/test_database_result_queries.py](../../../tests/test_database_result_queries.py) · [Handoff](handoff.md)
