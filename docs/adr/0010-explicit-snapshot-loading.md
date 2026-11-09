# ADR-010: Explicit registered-snapshot loading

Date: 2026-11-09
Status: proposed — awaiting user review
Work item: FEAT-0004

## Context

ORM metadata, logical snapshot identities and pinned results are implemented.
DataLocation is inert metadata. Existing subset/halo index digests cannot rebuild
particle selection. The next increment should load registered roots and delegate
to the current calculator engine, retaining ADR-008/009 cache/recovery rules.

## Proposed decision

Keep loader_id -> callable bindings in the repository process, explicitly provided
by project code. Pre-bind options there; reopen with equivalent bindings. Store
logical loader identity and location relations, not executable loader code or
parameters in this increment. No implicit default binding or location fallback.
One location is unambiguous; multiple locations require explicit selection.
Load local absolute paths/prefixes only, without an isfile precheck.

Add load_input returning the root and configured identity provider; add run_input
which commits running before loader IO and records the selected location on Run.
Validate root type, filename, transformation state and resulting logical identity.
Reuse the existing calculator/store/finalization path. Loading failure is a failed
run; metadata failure retains the explicit inspection/recovery seam. No ORM
transaction stays open across loader IO. Cache hits still load in this increment.

## Alternatives and consequences

Persisting a versioned declarative loader specification can improve portability,
but needs a parameter codec and reconstruction contract not yet selected.
The proposed runtime mapping is smaller and matches named-store configuration;
reproducibility depends on project code retaining equivalent bindings. loader_id
is a user assertion, not an automatic callable/configuration fingerprint.

Selection reconstruction, catalogues, remote transport and skip-load cache reads
remain separate work. Optional revision/manual refresh and no legacy migration
remain accepted requirements. No new dependency or core reverse import.

Full API, error cases and acceptance criteria: [EN](../work/FEAT-0004/en.md),
[中文](../work/FEAT-0004/zh-CN.md).

## 中文摘要

建议显式绑定 loader_id 到项目代码中的加载函数，参数在代码中预绑定；重开数据库需
重新提供等价绑定。首版仅加载本地已登记根快照，多位置需指定。run_input 先登记运行
再加载，记录所选位置，沿用原计算/结果/恢复规则。声明式参数持久化作为另一方案待
用户选择；本 ADR 尚未接受，不作为已确认架构引用。
