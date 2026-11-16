# SPIKE-0002 handoff

## 2026-11-16 — Design review / 设计评审

User scope: review the two user stories and produce a formal design; do not implement. Baseline d862742. Reviewed CONTEXT, ADR-020, FEAT-0012/0017 records, Database/ORM/SampleTable/codecs and related public-contract tests.

Created paired design documents, proposed ADR-021 and a design-index entry. Findings R01–R12 and acceptance scenarios A01–A32 are aligned in both languages. Main correction to the earlier sketch: a new detached StudyTable preserves existing mutable SampleTable behavior; single and batch must share Database coordination, not wrap the older load-first ORM runner unchanged. Outcomes distinguish failed, incomplete and unattempted work; exact refs never claim stale success. Saved Study definitions need full signatures and durable database ownership, not live Python objects.

用户范围：评审两条使用故事并形成正式设计，暂不实现。已建立双语文档、ADR 提案与设计索引。关键修正是独立 StudyTable、共享 Database 协调、分清失败/未完成/未尝试，以及完整签名和持久归属。实施验收场景尚未转为测试。

Validation is recorded below after checks. Existing unrelated docs/README edits and untracked local/example files remain outside this task. No implementation approval, commit, PR or merge is claimed.

## 2026-11-16 — Verification / 验证

- `make docs-check`: passed; 33 work items, 0 archived.
- Local document inspection: 7 Markdown files, 59 relative links resolved; 8 Python blocks parsed successfully with `ast.parse` (syntax only, not executed).
- Both design languages contain exactly R01–R12 and A01–A32 in order; corresponding Python example ASTs match. No unresolved placeholder markers, unbalanced fences or trailing whitespace.
- `git diff --check`: passed. Runtime source, tests, dependencies and CONTEXT were not edited. Existing unrelated workspace changes were retained.
- Inline self-review tightened read-only preflight, portable-value limits, complete cell-grid serialization and example connection lifetime. No runtime tests were run because the change is documentation-only; no proposed API is claimed to execute.

文档检查通过；双语发现/验收编号及代码示例语法结构一致，链接与空白检查通过。仅完成设计合同的内联自审与文档验证，没有实现或运行目标 API。工作项保留 review，ADR-021 保留 Proposed，等待后续人工设计评审；用户明确要求暂不实施。

## 2026-11-16 — Decision accepted / 决策已批准

The user accepted ADR-021 and authorized test-first staged implementation and merge. Implementation evidence is tracked in [FEAT-0018](../FEAT-0018/en.md).

用户批准 ADR-021，并授权分阶段先写测试、实现与合并。验收证据见 [FEAT-0018](../FEAT-0018/zh-CN.md)。
