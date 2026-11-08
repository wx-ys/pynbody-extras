# ADR-007: Immutable result versions and mutable cache addresses

Date: 2026-11-08
Status: accepted for implementation under the user-approved incremental plan
Work item: FEAT-0001

## Decision

The user requested the next milestone after typed simulation identities and
approved the plan to preserve historical results. No old database/cache migration
is required. Keep one ResultStore, with distinct addressing modes:

- RecordRef without version_id is a cache address. Its cache_key is the typed
  simulation identity plus full CalculatorSignature hash. list_refs() lists these
  current addresses, retaining its one-result-per-computation contract.
- store() returns a RecordRef with a non-empty UUID version_id. Its key includes
  that version, so equality/hash distinguish historical versions. fetch() on this
  reference always returns that exact version, even after refresh/invalidation.
- get/has/load use current cache addresses; list_versions(ref) lists saved versions
  of that address in insertion order, including versions no longer in the cache.
- Result.store_ref exposes the saved or reused version on engine/store reads;
  results computed without persistence have None. Cache hits create no versions.
- RecordRef JSON round-trip preserves identity types and version, omitting the
  optional backend-local id. Callers still pair a reference with its backend.

Every successful write appends one immutable version and atomically updates the
current cache entry. SQL uses separate version and cache tables in one transaction;
memory keeps isolated encoded history plus current records. Every read decodes a
fresh value. UUIDs identify saves, not payload content: identical recomputations
can have distinct provenance and versions. No content deduplication is promised.

Delete invalidates the cache only. An unversioned ref removes the current entry;
a versioned ref removes it only if that version is still current. It never deletes
saved history and never promotes an older result to current. fetch/contains with
a versioned ref address history; their unversioned forms address current cache.
History grows until an explicit retention/GC protocol is designed later.

The store adds two primitives (_get_version, _list_versions). Shared read checks
verify simulation identity, signature hash/text and requested version before value
decoding. Record schema advances to v4 with a required version_id. SQL version
insertion and cache replacement failures roll back together; concurrent successful
writes retain both versions, while conflicting/failed transactions may raise.
No retry/scheduler, cross-backend transaction, ORM Run, automatic GC or migration.

## Validation and review

Keep existing tests intact. Add red-to-green tests on both stores for historical
reads, current reads, invalidation, detached values, typed reference JSON, failure
rollback, SQLite reopen and simultaneous writers. Verify engine receipts for
writes/hits/refresh/no-store and that failed runs save nothing. Run make check and
make test-all. The user has authorized push, PR and merge; do not bypass protection.

## 中文摘要

用户要求继续实现历史结果不可变的下一阶段，且确认不需要旧数据库兼容迁移。
无 version_id 的 RecordRef 是当前缓存地址；store() 返回带 UUID 版本号的固定引用。
fetch(固定引用) 始终读取当次保存版本，get/has/load 读取当前缓存；list_refs 仍只列
当前地址，list_versions 列历史版本。Result.store_ref 暴露实际写入或命中的固定引用。

写入原子地追加版本并更新缓存指向。删除仅取消缓存；带版本引用仅在仍为当前版本时
可取消，不影响后来的版本，不删除历史、不自动回退。JSON 引用保留类型和版本，需
调用方配合 backend 使用。ResultRecord 升为 v4。暂不实现自动 GC、ORM、迁移或去重；
历史存储会增长。两个后端、事务失败、并发成功写入及完整测试必须验证。
