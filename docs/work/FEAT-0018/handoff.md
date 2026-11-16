# FEAT-0018 handoff

## 2026-11-16 — Implementation / 实施

The user approved ADR-021 and authorized staged test-first implementation,
commit, pull request and merge. Baseline `d862742`. Source and design edits
were isolated from the original checkout's unrelated changes.

- Definitions. Before implementation, five tests failed on the missing
  `Database.study` entry point and `read_only` argument.
- Coordination. Before implementation, tests failed on missing plan, collect
  and run methods. The final Database scenarios check compatible single/batch
  addresses, zero loads on hits, a concurrent refresh while collecting, exact
  selectors, real restart/interrupt outcomes and explicit receipt recovery.
- Detached tables. Before implementation, ten cases failed on missing run and
  collect support. The final tests exercise a clean process with no database
  or project calculator code, portable values, pinned refs, conditional
  overwrite and unit-aware pandas projection.
- Independent review reproduced mutation before binding, foreign handle
  access, nested opaque export, unconfirmed-cache reuse and loss of a receipt
  after a commit acknowledgment failure. Five dedicated regressions failed
  before the fixes and pass afterward.

用户批准 ADR-021 和分阶段先写测试、实施、提交及合并。以 `d862742`
为基线，在独立 checkout 实施。定义阶段先验证缺少 Study 入口及只读参数；
协调阶段先验证缺少 plan、collect 和 run；结果表阶段先验证尚无结果表入口。
独立评审复现的五项问题都有单独的先失败、后通过测试。

## 2026-11-16 — Verification / 验证

- Focused ADR-021 suite with pandas available: **37 passed**, covering A01–A32.
- Additional red-to-green review cases cover signed parameters changed in a
  loader, unsupported normalized parameter state, non-signature runtime
  memos, and custom containers that cannot round-trip portably.
- `make check`: **1,770 passed**, 7 deselected. Lint, 135 source modules
  type-checked, lockfile and bilingual work-item validation passed.
- `make test-all`: **1,777 passed**, including simulation data and both
  distribution builds. No skips.
- `git diff --check` passed. Both design documents contain A01–A32 and
  their Python blocks parse. Checked 46 local links across six documents.

补跑 pandas 投影后的 ADR-021 专项为 37 项通过，覆盖 A01–A32。
`make check` 的快速套件 1,770 项通过，7 项未选中。锁文件、lint、类型和
文档检查通过。`make test-all` 共 1,777 项通过，无跳过，含模拟数据与两个
发行包构建。检查了双语验收编号、示例语法与本地链接。

## 2026-11-16 — Pull request / 拉取请求

Implementation and review evidence are in [PR #56](https://github.com/wx-ys/pynbody-extras/pull/56).
实施与评审证据见 [PR #56](https://github.com/wx-ys/pynbody-extras/pull/56)。
