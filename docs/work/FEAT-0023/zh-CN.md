# FEAT-0023 · 按科学定位键连接子晕、宿主与 Study 测量

导出完整目录清单，保留未知成员和缺测对象，按类型化原始编号和宿主使用索引查询。 [实施计划](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

最初三项工作流在显式对象登记或分页导出处失败。另一项测试复现了全库查询遗漏已知宿主。五个子晕的完整清单以及 100 和 1,000 项目录的等长分页验证公开 API。

子晕和宿主两次连接均保留五行完整清单，可区分成功的零值、缺测和未知成员。100 项与 1,000 项目录的索引点查和等长分页使用相同数量的 SQL 查询。整数和浮点编号混合导出后可精确回查。

[tests/test_database_member_catalogue_workflows.py](../../../tests/test_database_member_catalogue_workflows.py) · [Handoff](handoff.md)
