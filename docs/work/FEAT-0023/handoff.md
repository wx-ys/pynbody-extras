# FEAT-0023

## 2026-11-20

Three initial catalogue workflows failed at explicit object registration or paginated export. A subsequent test showed that global discovery omitted known hosts. The five-object inventory and fixed-size pages for 100 and 1,000 entries exercise the public API.

最初三项工作流在显式对象登记或分页导出处失败。另一项测试复现了全库查询遗漏已知宿主。五个子晕的完整清单以及 100 和 1,000 项目录的等长分页验证公开 API。

Concurrent registrations of one new subhalo reproduced a uniqueness failure, and registering a known halo from an exact slice initially changed its result address. Both now preserve the same catalogue row and computation receipt. Mixed floating-point and large integer IDs survive dataframe export exactly; one Study collection pins host and subhalo measurements together.

并发登记同一个新子晕曾触发唯一约束冲突，用精确切片重新登记已知晕曾改变结果地址。修复后，两条路径分别复用目录行和计算回执。混合浮点与大整数编号导出后均能精确回查，一个 Study 可以同时固定宿主和子晕的测量版本。

Integrated verification passed `make check` (1,920 fast tests) and `make test-all` (1,928 tests) with pandas and real data. The complementary empty selection produced its expected warning.

集成验证已通过 `make check`（1,920 项快速测试）和 `make test-all`（1,928 项完整测试），含 pandas 和真实快照。取补集后为空的选集触发了预期提示。
