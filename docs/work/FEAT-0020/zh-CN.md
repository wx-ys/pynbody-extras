# FEAT-0020 · 随结果确认事务发布精确成员来源

在现有保存与恢复流程中，把实际选集关联到已确认的固定输出版本。相同输入可共用索引，各目录对象保留各自的科学来源。 [实施计划](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

最初十三项测试因快照和目录对象缺少 members 入口而失败。用例涵盖延迟保存、并发写入、命中复用、未确认保存、嵌套来源和十个输出共享索引。

成功保存时，输出用途和固定版本回执一并发布。缓存命中保留原生产者的科学对象。十个输出消费同一序列的 12,000 个粒子时，共用一个持久索引块。

[tests/test_database_members.py](../../../tests/test_database_members.py) · [tests/test_database_member_recovery.py](../../../tests/test_database_member_recovery.py) · [Handoff](handoff.md)
