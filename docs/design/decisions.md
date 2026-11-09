# Established decisions / 既有决策

These are reconciled records, not new approvals. ADR-001–005 were recorded in
the supplied V1 design, §21. Their identifiers are reserved; do not reuse them.

| ID | Constraint / 约束 | Current implementation / 现状 |
|---|---|---|
| ADR-001 | ORM owns metadata/references; ResultStore owns payloads / 分离关系与载荷 | FEAT-0003 implements the first association chain |
| ADR-002 | Adapt the existing Calculator DAG and EvalEngine / 渐进适配，不重写执行器 | Existing execution remains authoritative; public manifest is future work |
| ADR-003 | Explicit project logical identity, separate locations / 显式逻辑身份与位置解耦 | FEAT-0002, ADR-008 |
| ADR-004 | No default full content hashing / 默认不做完整内容哈希 | Revision optional; manual refresh, no fingerprint service |
| ADR-005 | Full default installation includes ORM; supported lightweight calculation path / 完整与轻量安装均有正式入口 | Still unresolved packaging mechanism; SQLAlchemy currently remains in `store` extra |
| [ADR-006](../adr/0006-typed-simulation-store-identity.md) | Typed simulation keys; no legacy migration / 类型化身份，无旧库迁移 | BUG-0009 |
| [ADR-007](../adr/0007-immutable-result-versions.md) | Immutable saved versions and independent current cache / 固定历史版本与当前缓存分离 | FEAT-0001 |
| [ADR-008](../adr/0008-configured-logical-snapshot-identity.md) | Optional revision and explicit loader identity / revision 可选、loader 语义由用户管理 | FEAT-0002 |
| [ADR-009](../adr/0009-orm-analysis-associations.md) | Separate commits, explicit recovery, no automatic GC / 分离提交、显式恢复、不自动清理 | FEAT-0003 |
| [ADR-010](../adr/0010-explicit-snapshot-loading.md) | Explicit trusted loaders, local root snapshots / 显式可信加载绑定、只加载本地根快照 | FEAT-0004 |

Python >=3.11 and pynbody >=2.8.0 are the current approved baseline. SQLite is
the verified relational backend; other dialects and platform combinations are
not certified by a Linux test run. No existing databases/caches require migration
in the current development scope. This does not imply that future released user
databases may be silently recreated.

For additions in the revised design, record a focused ADR **after human review**
and before architecture implementation. Review should explicitly approve the
public API, identity rules, serialization and failure behavior of that increment.
Do not infer approval for every future feature from permission to fix a bug.

## 2026-11-09 — newly accepted increments

User explicitly approved the complete contract review packet and packaging names/ownership in this chat.

- [ADR-011](../adr/0011-sample-cell-references.md): Pinned SampleTable cell references.
- [ADR-012](../adr/0012-run-execution-context.md): Invocation execution-context snapshots.
- [ADR-013](../adr/0013-local-external-file-references.md): Explicit local external-file result references.
- [ADR-014](../adr/0014-dag-value-postprocessors.md): Value-consuming postprocessors on the existing DAG.
- [ADR-015](../adr/0015-catalogue-object-metadata.md): Catalogue-scoped object metadata.
- [ADR-016](../adr/0016-replayable-particle-selections.md): Root-relative replayable particle selections.
- [ADR-017](../adr/0017-root-input-job-manifests.md): Versioned root-input job manifests.
- [ADR-018](../adr/0018-serial-recorded-studies.md): Serial studies through recorded runs.
- [ADR-019](../adr/0019-full-core-distribution-ownership.md): approved full/core names and exclusive file ownership.
