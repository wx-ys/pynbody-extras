# FEAT-0004 handoff

## 2026-11-09 — Codex / proposed next-stage design

- Started from merged FEAT-0003 / PR #39, main 2d32336; user asked to continue.
- Audited ORM, identity providers, tests and original V1 phase C/D requirements.
- Proposed explicit runtime loader bindings and root-only load/run APIs.
- Key unresolved choice: runtime bindings vs persisted declarative parameters.
- No implementation or runtime contract change yet. User review of en/zh-CN spec
  and proposed ADR-010 is required before implementation.
- Existing no-migration, optional-revision, separate-store/recovery and publication
  decisions persist. Preserve unrelated untracked user directories.
- Next after design approval: red tests, implementation plan, implementation,
  independent review, make check/test-all, scoped PR and merge under authorization.

- Baseline before implementation: ORM, package-layer and logical-identity suites
  passed (78 tests, 6.57 s); log /tmp/pynbody-loading-baseline.log.
- docs-check passed with 14 work items. Clarified preflight rejects unstable
  calculator signatures; constructibility is not a new run requirement.
