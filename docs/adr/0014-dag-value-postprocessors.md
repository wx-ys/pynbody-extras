# ADR-014: Value-consuming postprocessors on the existing DAG

Date: 2026-11-09
Status: accepted — user explicitly approved FEAT-0005–0012 contracts on 2026-11-09
Work item: FEAT-0008

## Context and approval

The user reviewed docs/design/implementation-approval.md and answered
“批准全部 FEAT-0005–0012 合同”. This records that specific approval before
implementation. The existing Calculator DAG, ResultStore and ORM remain the
owning seams. Verified local predecessors may be used sequentially without
claiming remote merge. Existing user changes are preserved.

## Decision

Implement design §6.1. PostprocessorBase subclasses the existing property/runtime contract with one source calculator dependency declared through existing Param mechanisms. The reusable adapter resolves source then calls process(value). Signature identity includes source and dataclass parameters, honors signature_version and reconstructibility. Initial source accepts a Calculator only; reject raw Result/RecordRef inputs explicitly. Expose through calculate facade. No fitting dependency, new global registry or custom scheduling.

The initial processor supports one dynamic field, source; other processor configuration fields are static parameters. The adapter owns calculate(sim, params) and calls process(params.source); scientific subclasses override process only. Do not silently accept additional dynamic configuration unless its resolved-value delivery contract is designed and tested in this same reviewed increment.

### Scoped exception cleanup implementation boundary

Implementation tests found that the existing engine applies a node's bound
scope but does not clean that scope after a property exception. Implement the
approved process-exception cleanup requirement locally in
`PostprocessorBase.execute`: delegate execution to the existing property runtime,
and on an exception clean only this postprocessor's own successfully applied
bound transform handle. Honor scope and transform revert policies and the raw
result's `revertible` flag. Use the existing transform cleanup phase/log pattern;
if cleanup itself fails, log it and preserve the original process exception.
Do not invoke cleanup for an ancestor's inherited scope or change generic
engine/property success behavior. This is callback ownership: existing native
pynbody compound handles can themselves cascade through earlier transforms when
reverted. A retained native `NEVER` step can prevent an earlier handle from
reverting; this is a logged cleanup failure with the original error preserved,
not a promise to isolate arbitrary handle effects. Failures while applying a transform or filter before entering
the adapter are outside this narrow process-exception cleanup path. This is an
implementation of the approved exception clause, not a new executor or scheduler.
Error diagnostics retain the engine's existing last-recorded-phase behavior:
cleanup can leave `ErrorInfo.phase` as `revert`, while the original exception
type, message, and traceback remain the reported error. Redesigning phase-origin
reporting is outside this increment.


## Consequences and boundaries

Follow the full [English contract](../work/FEAT-0008/en.md) and
[中文合同](../work/FEAT-0008/zh-CN.md) and their acceptance matrices.
No additional executor, implicit binding, automatic history GC, legacy database
migration or external publication is authorized by this decision. API failure
semantics, typed identity and pinned references remain explicit. SQLite/Linux
verification does not certify other platforms or SQL dialects. Each increment
must retain red/green tests, bilingual handoff and honest review status.

## 中文摘要

用户已在本会话明确批准该项完整接口合同。先记录本决定，再按既有 DAG、结果存储与
ORM 分层逐项实施。具体身份、序列化、异常和恢复语义以上述双语合同为准；不重复
请求相同批准，不扩大为迁移、发布或自动清理授权。
