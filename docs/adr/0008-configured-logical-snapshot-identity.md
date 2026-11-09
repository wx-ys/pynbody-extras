# ADR-008: Configured logical snapshot identities

Date: 2026-11-09
Status: accepted for implementation; optional revision explicitly selected by user
Work item: FEAT-0002

## Context and decision

The supplied V1 ADR-003/004 already require user/project logical IDs, mutable
locations and no default full-content hash. The next increment uses the existing
SimIdentityProvider seam. It does not introduce a new execution or storage system.

LogicalSnapshotIdentity(project_id, simulation_id, timestep_id, revision=None,
loader_id="default") is a frozen value object. Required fields and loader_id are
non-empty strings; revision is None or a non-empty string. Whitespace-only values
are invalid. Literal values are preserved. Its tuple is:

("sim", "pynbodyext.logical-snapshot/v1", project_id, simulation_id,
 timestep_id, revision, loader_id)

ConfiguredSnapshotIdentity maps explicit local paths/prefixes to those values.
Normalize locations once using abspath, without realpath or existence checks.
Snapshot the supplied mapping. Repeated normalized paths with identical identities
are harmless; conflicting identities raise ValueError. Aliases are declarations
of equivalence, not checksums. A fresh provider replaces configuration explicitly.

Match the ancestor filename before applying existing halo/subset identity suffixes.
Reuse current transformation checks: an externally transformed root/view is
session-only even when configured. Unknown locations keep snapshot_identity
fallback; configurations do not silently bind unnamed in-memory snapshots.

JSON schema pynbodyext.snapshot-identity-config/v1 stores an entries list with
absolute paths and logical fields. Unknown/missing fields, unsupported schema,
invalid types/values, relative paths and conflicting normalized paths fail early.
No implicit file read, fingerprint, plugin loading, global registry or DB dependency.

The user explicitly chose optional revision on 2026-11-09. If omitted, contents
and particle order may change without a key change: reuse=False refreshes each
affected calculation. Supplied revisions and loader IDs separate cache namespaces.
This is an explicit data-version policy, not automatic invalidation or provenance
proof. Saved result version_id still identifies an individual save, independently.

## Alternatives

1. Explicit revision required: stronger reminder of versioning, but rejected by
   the user in favor of optional revision with manual refresh.
2. Optional revision and manual refresh: selected; small API and no scanning cost.
3. Size/mtime or content fingerprint: not selected for this increment. Fast file
   metadata cannot prove content equality; any future policy needs strength/scope.

## Boundaries and validation

Implement runtime/logical_identity.py, share existing snapshot-view addressing,
and export through the calculator facades. No import layer exceptions, new
dependencies, default-provider changes, ORM, migration or automatic GC.
SampleTable default row labels recognize the logical tuple: all five logical fields
are compact JSON (escaping literal IDs), followed by halo/subset suffixes. This
avoids treating the schema tag as a filename and naming every logical row v1;
row identity keys and explicit user names are unchanged.

Test relocation/reload, field separation, no-revision refresh with old pinned
receipts, subsets/order/halo IDs, transform fallback, unknown paths, configuration
conflicts and JSON across processes. Verify memory and SQLite reads, SQLite reopen,
SampleTable integration, original tests, make check and make test-all.

## 中文摘要

逻辑身份由项目、模拟、时间步、可选 revision 和加载设置标识组成，位置仅用于配置
查找。用户明确选择 revision 可省略；省略时通过 reuse=False 逐项刷新，不自动检测
内容或粒子顺序变化。原有路径/session fallback 保留，外部变换仍拒绝持久化身份。
配置冲突及非法 JSON 早期报错。无扫描、ORM、依赖变更或迁移；结果 version_id 与数据
revision 含义不同。身份配置只声明等价，不证明文件内容相同。
