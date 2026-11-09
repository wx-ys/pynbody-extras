# ADR-011: Pinned SampleTable cell references

Date: 2026-11-09
Status: accepted — user explicitly approved FEAT-0005–0012 contracts on 2026-11-09
Work item: FEAT-0005

## Context and approval

The user reviewed docs/design/implementation-approval.md and answered
“批准全部 FEAT-0005–0012 合同”. This records that specific approval before
implementation. The existing Calculator DAG, ResultStore and ORM remain the
owning seams. Verified local predecessors may be used sequentially without
claiming remote merge. Existing user changes are preserved.

## Decision

Implement design §5.1: add refs as the final defaulted SampleRow field; add optional keyword-only store_name to SampleTable. Capture result.store_ref only after success, clear stale refs on failure/non-persisted success, preserve refs during fill/where/save/load. New sample JSON schema includes refs and store_name; restore with RecordRef.from_json, validate pinned ref against row typed identity and column signature, reject wrong-key/unpinned refs. Load never dereferences a backend. Keep legacy schema readable with empty refs, without inventing historical versions. Export dataframe references indexed by typed identity, not duplicate display names.


## Consequences and boundaries

Follow the full [English contract](../work/FEAT-0005/en.md) and
[中文合同](../work/FEAT-0005/zh-CN.md) and their acceptance matrices.
No additional executor, implicit binding, automatic history GC, legacy database
migration or external publication is authorized by this decision. API failure
semantics, typed identity and pinned references remain explicit. SQLite/Linux
verification does not certify other platforms or SQL dialects. Each increment
must retain red/green tests, bilingual handoff and honest review status.

## 中文摘要

用户已在本会话明确批准该项完整接口合同。先记录本决定，再按既有 DAG、结果存储与
ORM 分层逐项实施。具体身份、序列化、异常和恢复语义以上述双语合同为准；不重复
请求相同批准，不扩大为迁移、发布或自动清理授权。
