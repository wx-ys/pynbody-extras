# ADR-012: Invocation execution-context snapshots

Date: 2026-11-09
Status: accepted — user explicitly approved FEAT-0005–0012 contracts on 2026-11-09
Work item: FEAT-0006

## Context and approval

The user reviewed docs/design/implementation-approval.md and answered
“批准全部 FEAT-0005–0012 合同”. This records that specific approval before
implementation. The existing Calculator DAG, ResultStore and ORM remain the
owning seams. Verified local predecessors may be used sequentially without
claiming remote merge. Existing user changes are preserved.

## Decision

Implement design §5.2. Add versioned execution_context JSON to Run and optional keyword execution_metadata to run/run_input/begin_run, accepting only tool_versions (string map) and random_seed (integer or null). Snapshot Python/package versions and selected URI/loader/revision before starting execution; unavailable package versions are null. Native state is recorded without importing the extension. Caller data is copied via JSON validation. Snapshot is immutable through repository APIs; existing direct session writes remain caller responsibility. No automatic random seeding or cache-key changes.

ADR-019 integration: `packages` records both `pynbodyext-core` (the code-owning
distribution) and `pynbodyext` (the full metadata package), independently and with
null for each missing distribution. Core with the store extra must retain its
known code version without claiming the absent full package is installed.


## Consequences and boundaries

Follow the full [English contract](../work/FEAT-0006/en.md) and
[中文合同](../work/FEAT-0006/zh-CN.md) and their acceptance matrices.
No additional executor, implicit binding, automatic history GC, legacy database
migration or external publication is authorized by this decision. API failure
semantics, typed identity and pinned references remain explicit. SQLite/Linux
verification does not certify other platforms or SQL dialects. Each increment
must retain red/green tests, bilingual handoff and honest review status.

## 中文摘要

用户已在本会话明确批准该项完整接口合同。先记录本决定，再按既有 DAG、结果存储与
ORM 分层逐项实施。具体身份、序列化、异常和恢复语义以上述双语合同为准；不重复
请求相同批准，不扩大为迁移、发布或自动清理授权。
