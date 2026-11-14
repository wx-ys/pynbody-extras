# FEAT-0014: 数据库缓存模式与局部默认配置

实施已批准 Database 设计的 B 阶段。用户已授权实施、提交、拉取和合并。

[Design / 设计](../../design/database-ux-2026-11-14.zh-CN.md) · [ADR-020](../../adr/0020-database-user-interface.md)

验收：满足对应阶段表及共用身份、失败和分层规则；保留既有测试；在此记录红绿与完整测试证据。

## Implementation / 实现

- [x] manual/auto/refresh、ContextVar 嵌套、旧 store 隔离、关闭资源和跨库归属遵循 ADR-020。
- [x] Existing behavior tests retained; package layer declarations extended under accepted ADR-020. / 保留既有行为测试；按 ADR-020 扩展包分层声明。

## Evidence / 验证证据

Initial Database tests: 10 expected missing-API failures, one existing import-isolation pass. ORM path tests initially failed on missing opt-in API. Independent review supplied additional red regressions for partial roots, corrupted recovery metadata, payload-free listing, uint64 halo IDs, alias locations, repeated loading and old-store interoperability; all addressed.

新入口初始 10 项因缺失 API 失败，1 项既有导入隔离通过；ORM 路径测试初始因缺失 opt-in 失败。独立评审补充的部分加载、损坏元数据恢复、免载荷列表、uint64 编号、多位置、重复加载及旧存储互操作回归均已修复。

- 52 new contract tests passed; 88 compatibility-focused tests passed.
- make check: 1731 passed, 2 skipped, 7 deselected; Ruff, mypy 130 modules, package/lock and docs checks passed.
- Real Tipsy quickstart passed, including reopened root/subset/halo cache hits with zero loader calls.
- Independent ORM/core/catalogue review approved after fixes.
- make test-all: 1738 passed, 2 skipped (81.08 s); includes data-backed and isolated distribution tests. / 完整测试 1738 项通过、2 项跳过，含数据与隔离发行测试。

Logs: /tmp/database-red.log, /tmp/database-review-red.log, /tmp/catalogue-review-red.log, /tmp/database-new-final.log, /tmp/database-compat-green.log, /tmp/database-make-check.log, /tmp/database-quickstart-final.log.
