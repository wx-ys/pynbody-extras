# FEAT-0018 — 数据库研究和独立结果表

实现已获批准的 [ADR-021](../../adr/0021-database-studies-and-detached-tables.md) 及其[验收场景](../../design/database-study-2026-11-16.zh-CN.md#10-验收场景)。

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| 定义 | 新增数据库命名空间、只读打开、精确对象解析、活绑定和具名定义 | A05、A10、A19、A21、A27、A31 |
| 执行协调 | 单次与批量格共享执行、只读计划、缓存读快照、精确重试与恢复 | A01–A04、A06–A09、A11、A13–A18、A23、A28、A32 |
| 独立结果表 | 不可变观察、可移植原子导出、数据投影与导入合同 | A12、A20、A22、A24–A26、A29–A30 |

[数据库验收测试](../../../tests/test_database_study.py)和[结果表验收测试](../../../tests/test_study_table.py)按编号标记场景。既有 SampleTable、ORM StudyReport 和 Database 测试继续验证原有公开合同。
