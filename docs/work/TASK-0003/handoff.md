# TASK-0003 handoff

## 2026-11-20 · Historical review

- Reviewed merged PRs #1 through #59 on baseline `60fc13f`. GitHub returned all pages of comments, reviews, threads and replies. The English and Chinese work items list the disposition of the 51 inline findings.
- The first sample and codec regressions produced five failures. Mismatched iterable lengths could partly compute a table, a duplicate name selected the first row, a complex column returned a list, and an oversized structured header was accepted. After fixing those paths, 227 sample, codec, median, dependency-contract and documentation tests passed with pandas available.
- A deep result card raised `RecursionError`, simple cards evaluated hidden content, four legacy parser cases failed to import, and a broken native library was reported as skipped. The focused suites passed 210 tests after the fixes.
- Signature tests reproduced object-array collisions, reuse of opaque node IDs, unchecked saved signatures, repeated execution of default factories and stale or incomplete Pipeline keys. Independently rechecked subsequent fixes against current storage and reconstruction behavior.
- Old median-filter reports remain covered by the finite-value, window-definition and bounded-batch tests. Verified structured scientific records round-trip in NPY formats 1.0, 2.0 and 3.0, including a 3,600-field record.
- Integration reproduced a stored recipe conflict and a rejected Study binding for matching NaN parameters. The store, ORM and Database now compare the same complete canonical representation. The 177 related tests include SQLite reopen, reuse and refresh. Portable Study definitions retain their strict JSON boundary.
- Both orderings of a mixed custom TransformChain failed. Borrowed container attributes also returned coordinates from the restored frame. The seven Pipeline, observation, membership and recovery suites passed all 160 tests after the fixes.
- A nested custom Pipeline repeated its children in its output bindings, constructor state and dependencies. The seven-layer audit case fell from 415,441 bytes to 1,739 bytes. A twelve-layer check visited twelve hooks. Rebuilding preserved the original hash. Shared output aliases were both discoverable in Database and survived the portable JobSpec round trip.

已核对第 1 至 59 个已合并 PR，逐项结果和依据见双语工作项。验证环境包含 Linux、Python 3.11、pynbody 2.8、SQLite、原生扩展、真实快照及 pandas。
