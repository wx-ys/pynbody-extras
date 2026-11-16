# FEAT-0018 — Database studies and detached tables

Implements the approved [ADR-021](../../adr/0021-database-studies-and-detached-tables.md) and its [acceptance scenarios](../../design/database-study-2026-11-16.md#10-acceptance-scenarios).

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| Definitions | Additive namespace, read-only open, exact object resolution, live bindings and named definitions | A05, A10, A19, A21, A27, A31 |
| Coordination | Shared single/cell execution, metadata-only plans, cache snapshots, exact retries and recovery | A01–A04, A06–A09, A11, A13–A18, A23, A28, A32 |
| Detached tables | Immutable observations, portable atomic export, data projections and import contracts | A12, A20, A22, A24–A26, A29–A30 |

[Database acceptance tests](../../../tests/test_database_study.py) and [table acceptance tests](../../../tests/test_study_table.py) identify the scenarios by number. The existing SampleTable, ORM StudyReport and Database tests continue exercising their original public contracts.
