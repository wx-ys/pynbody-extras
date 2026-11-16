# SPIKE-0002: Review unified Database studies and detached result tables

**Decision accepted.** Implementation and verification are tracked in [FEAT-0018](../FEAT-0018/en.md).

## Request and scope

The user requested review of the two target stories, focused on single/batch result sharing, failed attempts versus old values, and independent reopening of Study and result tables; then a formal design document, explicitly without implementation.

Deliverables: [formal design](../../design/database-study-2026-11-16.md), [Chinese pair](../../design/database-study-2026-11-16.zh-CN.md), and [proposed ADR-021](../../adr/0021-database-studies-and-detached-tables.md). Baseline: d862742; date: 2026-11-16.

## Review outcome

- Twelve findings R01–R12 resolve shared-key/execution requirements, mutable-table incompatibility, stale-success confusion, incomplete persistence, unavailable calculator reconstruction and database ownership.
- Recommend a detached StudyTable without changing legacy SampleTable/StudyReport contracts; retain the existing result-store key and Database execution ownership.
- Separate saved requests from frozen observations. Reopening a Study requires its database; opening a portable table does not require a database or project calculator code.
- Defer path-only registration, automatic particle-load sharing and not-applicable outcomes pending their own contracts.
- Define 32 future behavioral acceptance scenarios A01–A32. These are not implemented tests.

## Acceptance for this design task

- [x] Inspect owning source and public-contract tests, rather than infer current behavior from earlier conversation.
- [x] Document inconsistencies, alternatives and reduced scope in both languages.
- [x] Specify identity, state, reopen, readonly, recovery and serialization contracts with target code examples.
- [x] Record proposed layer/schema decisions without marking them accepted.
- [x] Validate documentation, local links, code-block syntax and bilingual contract coverage; exact results are in handoff.

Status review means the proposal is ready for human design review, not that implementation is complete or authorized. No runtime, test, dependency or current-contract changes belong to this item. No PR or publication is requested.
