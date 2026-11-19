# BUG-0013 handoff

## 2026-11-19 · Work item opened

Created the bilingual item against `c56120f`. The Phase 1 contract is in
[ADR-022](../../adr/0022-pipeline-transform-scopes.md).

## 2026-11-19 · Reproduction and review

- The two output orders in `test_branch_order_restores_entry_frame` failed on
  `c56120f`. Running the rotated branch first changed the native selection
  from `[2, 3, 5]` to `[3, 4, 5]`. Running the native branch first left 6 of
  the caller's 18 position components changed. The 52 existing tests run
  against that baseline passed. [Frame regressions](../../../tests/test_pipeline_branch_scopes.py)
- Reopening a SQLite Database containing the old result reproduced a cache
  collision. A corrected recipe with the old address returned the saved
  native selection from the rotated frame under `mode="auto"`. The Pipeline
  scope version changes the address of recipes containing a Pipeline. The
  regression also reads the old pinned receipt and frozen Study column and
  reuses an existing standalone Property result. [Persistence regressions](../../../tests/test_pipeline_scope_versions.py)
- The runtime and persistent-identity changes are under review. The bilingual
  item records transform ownership, measurements in nested frames, failure
  handling and standalone lifetimes. [English](en.md) · [中文](zh-CN.md)
- `make docs-index` regenerated the work-item row. `make docs-check` passed
  for 35 work items and 0 archived items.

两种输出顺序在基线分别复现了原坐标系分支选错粒子和调用方位置发生变化。SQLite
用例复现了升级后 `auto` 模式用旧键命中错误结果。阶段一的变换作用域和配方版本
变更已进入评审。

## 2026-11-20 · Phase 1 acceptance

- The eight acceptance modules passed all 139 tests. Both branch orders now
  select the same particles as their standalone controls and return the
  caller's original positions, velocities and active handle.
- Review reproduced an eager Bin changing the next subset's coordinates and a
  ragged array following the restored particle coordinates. Regression tests
  cover per-bin Calculators, Filter selection and public readback, nested Bin
  axes, and the units and values in a ragged result. An undefined alignment
  vector can be reported before mutation while other outputs finish.
- `make check` passed Ruff, mypy, distribution and lock checks, the 1,831 fast
  tests and the work-item index check. `make test-all` passed all 1,838 tests,
  including data-backed, native, source-distribution and optional pandas
  cases. The existing particle data and cached build and pandas packages were
  used offline.

阶段验收共 139 项。快速检查通过，全量 1,838 项全部通过。对同一个 Bin Filter
重复读取会复用已记录的子集。两个 Filter 使用不同的平移参数时能分别读回各自
的粒子。工作项等待核心运行时的人工评审。

## 2026-11-20 · Pre-merge review

Three indexed-subset tests reproduced periodic wrapping that skipped the forward mutation or failed before wrapping. Two alignment tests reproduced a lost healthy sibling after invalid axis input. All five new cases pass after write-through subset assignment and pre-mutation axis validation.

The reviewed tree passed `make check` (1,836 fast tests) and `make test-all` (1,843 tests) on November 20, including data, pandas, native and distribution tests.

A final review reproduced a custom transform chain losing its receipt and four borrowed-array containers losing their values or types. Both custom-chain paths and all four containers now pass. At the final tree, `make check` passed 1,842 fast tests and `make test-all` passed all 1,849. Callers sharing a mutable root across independent threads coordinate access as described in ADR-022.
