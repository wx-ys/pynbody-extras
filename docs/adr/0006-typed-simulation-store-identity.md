# ADR-006: Typed simulation identities in result storage

Date: 2026-11-08
Status: accepted; maintainer authorized publication and merge on 2026-11-08
Work item: BUG-0009

## Context and decision

The maintainer explicitly confirmed that the project has no existing databases
or caches and needs no compatibility or migration. Adopt a new typed key format
and result-store schema. No legacy reads, dual writes or migration command.
This concerns result storage, not a replacement DAG or logical snapshot registry.

Encode identities as versioned canonical JSON with explicit tags. Support built-in
str, bool, int, finite float, None and recursive tuples. Distinguish types, including
bool/int and int/float; normalize signed floating zero. Reject mutable containers,
arbitrary objects, NumPy scalars and non-finite floats instead of stringifying or
pickling them. Providers must explicitly convert NumPy scalar IDs to built-ins;
snapshot_identity already converts halo numbers this way.

One leaf encoder in store/identity.py defines equality for RecordRef, both stores,
simulation listing and SampleTable row indexing. Preserve original identity tuples
as metadata. CalculatorSignature numerical equivalence remains unchanged. The
per-run cache stays unchanged: each run has a single simulation identity.

Bump ResultRecord to v3; reject missing/unknown record schemas on JSON load.
Existing SQL columns suffice for new databases. Snapshot/path identity, cache
refresh semantics, historical payload versions, ORM and dependencies are unchanged.

## Consequences and validation

Identities 1, "1", True and 1.0 coexist without overwriting, fetching or deleting
one another. Identical typed identities still reuse results. SampleTable must not
collapse rows through Python equality. Exercise SQLite reopen, filtered lists,
deletion, read-through, table save/load, invalid identities and deterministic keys.
Run make check and make test-all. The maintainer authorized merge after these checks.

## 中文决策摘要

维护者确认项目仍在开发中，不存在旧数据库或缓存，明确不需要兼容迁移。
采用带类型标签和版本号的 canonical JSON 身份键，将 ResultRecord 升为 v3；
缺失或未知记录版本直接拒绝。不增加旧格式读取、双写或迁移工具。

支持 Python 内建 str、bool、int、有限 float、None 及递归 tuple；不同类型分别寻址，
浮点正负零归一。拒绝可变容器、任意对象、NumPy 标量及非有限浮点；provider 应显式
转换 NumPy ID。统一 RecordRef、两个 store、模拟列表和 SampleTable 行索引规则。
保留原始 tuple 元数据，不改变 CalculatorSignature 数值等价、运行内缓存、snapshot
身份来源、覆盖策略或依赖。维护者已明确授权发布及合并。
