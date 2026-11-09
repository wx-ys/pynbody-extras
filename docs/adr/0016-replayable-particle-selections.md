# ADR-016: Root-relative replayable particle selections

Date: 2026-11-09
Status: accepted — user explicitly approved FEAT-0005–0012 contracts on 2026-11-09
Work item: FEAT-0010

## Context and approval

The user reviewed docs/design/implementation-approval.md and answered
“批准全部 FEAT-0005–0012 合同”. This records that specific approval before
implementation. The existing Calculator DAG, ResultStore and ORM remain the
owning seams. Verified local predecessors may be used sequentially without
claiming remote merge. Existing user changes are preserved.

## Decision

Implement design §6.3. Add SelectionSpec schema for indices/family/filter, tied to a registered root and expected particle count. register_selection requires a LoadedInput to materialize and validate the selection once; persist the resulting provider identity plus definition. Indices are a JSON list of built-in integers, ordered and duplicate-preserving; reject bool, negatives and out-of-range values. Filter signature must be stable and constructible. Register only an untransformed root. Add load_selection(selection_id, location_id=None) returning LoadedInput for the view with root provider; run_selection delegates the existing run flow and records failures. Location ownership checks refer to the root; this is a deliberate extension of root-only Run location rules requiring an ADR and regression tests. Existing arbitrary view load_input rejection remains.

Persist Selection rows with root_input_id, view_input_id, schema, definition_json, expected_count and a unique canonical descriptor key. Different definitions may share the same materialized view DataObject; do not overwrite either definition. Add nullable Run.selection_id to retain which definition was used. Proposed API: SelectionSpec.indices(list[int]), SelectionSpec.family(str), SelectionSpec.filter(calculator); repo.register_selection(loaded_root, spec) -> Selection; repo.get_selection(selection_id); repo.load_selection(selection_id, *, location_id=None); repo.run_selection(calculator, selection_id, *, store_name, location_id=None, reuse=True, options=None). Registration derives expected_count from loaded_root rather than trusting caller input. run_selection commits running after metadata preflight, before loading; the selected view and definition are both validated during finalization and recovery.


## Approved family-order amendment — 2026-11-09

The user explicitly approved proposal A: “首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”. The original universal ordering promise is replaced by exact order and duplicate preservation within root family-block ordering. On `pynbody.new(dm=2, star=2)`, `[1,0,1,3,2,3]` is accepted unchanged; `[3,1,3,0]` and `[3,1]` fail explicitly without sorting or metadata writes. This follows the installed pynbody 2.8 IndexingViewMixin contiguous-family requirement; a new general view adapter is outside this increment. Replay encountering an incompatible family layout fails the already-created Run before analysis. See the [approved decision evidence](../work/FEAT-0010/family-order-decision.md). All descriptor, identity, context, location and receipt-recovery invariants remain in effect.

## Consequences and boundaries

Follow the full [English contract](../work/FEAT-0010/en.md) and
[中文合同](../work/FEAT-0010/zh-CN.md) and their acceptance matrices.
No additional executor, implicit binding, automatic history GC, legacy database
migration or external publication is authorized by this decision. API failure
semantics, typed identity and pinned references remain explicit. SQLite/Linux
verification does not certify other platforms or SQL dialects. Each increment
must retain red/green tests, bilingual handoff and honest review status.

## 中文摘要

用户已在本会话明确批准该项完整接口合同。先记录本决定，再按既有 DAG、结果存储与
ORM 分层逐项实施。具体身份、序列化、异常和恢复语义以上述双语合同为准；不重复
请求相同批准，不扩大为迁移、发布或自动清理授权。
