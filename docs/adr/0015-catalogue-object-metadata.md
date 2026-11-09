# ADR-015: Catalogue-scoped object metadata

Date: 2026-11-09
Status: accepted — user explicitly approved FEAT-0005–0012 contracts on 2026-11-09
Work item: FEAT-0009

## Context and approval

The user reviewed docs/design/implementation-approval.md and answered
“批准全部 FEAT-0005–0012 合同”. This records that specific approval before
implementation. The existing Calculator DAG, ResultStore and ORM remain the
owning seams. Verified local predecessors may be used sequentially without
claiming remote merge. Existing user changes are preserved.

## Decision

Implement design §6.2 as metadata only. Add Catalogue(root_input_id,catalogue_id,finder,revision,loader_id) with a typed canonical identity key for uniqueness including null revision. Add CatalogueObject(catalogue_fk,kind,typed_object_key,host_id optional). Repository register/get/list methods are idempotent and return detached immediate relations. Object keys accept built-in string/int/bool/finite float using sim_identity_key; cross-type keys differ. A host must be a halo in the same catalogue; only subhalos may have a host. Catalogue root must be a validated snapshot DataObject. Do not alter existing provider signatures, DataObject.parent semantics or run_input.

Proposed repository methods: register_catalogue(root_input_id, catalogue_id, *, finder, revision=None, loader_id="default") -> Catalogue; get_catalogue(catalogue_pk); catalogues(root_input_id); register_catalogue_object(catalogue_pk, object_key, *, kind="halo", host_id=None) -> CatalogueObject; get_catalogue_object(object_pk); catalogue_objects(catalogue_pk, *, kind=None, host_id=None). Validate nonempty string labels; catalogue identity includes root typed identity and all four labels. Changing finder/revision/loader creates another catalogue. The list methods order by primary key for deterministic results.


## Consequences and boundaries

Follow the full [English contract](../work/FEAT-0009/en.md) and
[中文合同](../work/FEAT-0009/zh-CN.md) and their acceptance matrices.
No additional executor, implicit binding, automatic history GC, legacy database
migration or external publication is authorized by this decision. API failure
semantics, typed identity and pinned references remain explicit. SQLite/Linux
verification does not certify other platforms or SQL dialects. Each increment
must retain red/green tests, bilingual handoff and honest review status.

## 中文摘要

用户已在本会话明确批准该项完整接口合同。先记录本决定，再按既有 DAG、结果存储与
ORM 分层逐项实施。具体身份、序列化、异常和恢复语义以上述双语合同为准；不重复
请求相同批准，不扩大为迁移、发布或自动清理授权。
