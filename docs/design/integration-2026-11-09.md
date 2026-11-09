# Integration record / 集成记录 — 2026-11-09

User authorized autonomous PR/merge after local delivery review. Each implementation slice was reconstructed from the approved final source, verified with `make check` and `make test-all`, then committed, pushed and merged with an exact head-commit match. Private untracked directories were excluded. No release was published.

| Item | PR | Merge commit | Scoped check | Scoped full suite |
|---|---|---|---|---|
| Contracts / ADR-011–019 | [#41](https://github.com/wx-ys/pynbody-extras/pull/41) | `674ea4f21b473df282ea2e67f1ec9f8426596eb8` | docs27 passed | Documentation only |
| BUG-0010 | [#42](https://github.com/wx-ys/pynbody-extras/pull/42) | `1b4f1b8f7c3187df2d7f3ac18e6ecdcfe8749e37` | 1103 passed, 1 skipped, 4 deselected in 35.12s | 1107 passed, 1 skipped in 33.92s |
| BUG-0011 | [#43](https://github.com/wx-ys/pynbody-extras/pull/43) | `8c34db56a53f4dfc2185d9398c2a19266cd71800` | 1151 passed, 1 skipped, 4 deselected in 30.58s | 1155 passed, 1 skipped in 30.26s |
| BUG-0012 | [#44](https://github.com/wx-ys/pynbody-extras/pull/44) | `1c35d9892a93f5d175b86cf432065c0f389d4c74` | 1155 passed, 1 skipped, 4 deselected in 29.36s | 1159 passed, 1 skipped in 29.52s |
| FEAT-0005 | [#45](https://github.com/wx-ys/pynbody-extras/pull/45) | `1fbde1c4aeca11844cb7ede0339731756a33832f` | 1170 passed, 2 skipped, 4 deselected in 29.73s | 1174 passed, 2 skipped in 29.84s |
| FEAT-0006 | [#46](https://github.com/wx-ys/pynbody-extras/pull/46) | `e8bdfcb1adf75987d0e88366f823abb95396eaf4` | 1225 passed, 2 skipped, 4 deselected in 34.33s | 1229 passed, 2 skipped in 34.17s |
| FEAT-0007 | [#47](https://github.com/wx-ys/pynbody-extras/pull/47) | `5103136ec2837ec428ea722743c24637be067616` | 1318 passed, 2 skipped, 4 deselected in 34.35s | 1322 passed, 2 skipped in 34.97s |
| FEAT-0008 | [#48](https://github.com/wx-ys/pynbody-extras/pull/48) | `83ae4b5b3c363ef151f179265157c1d03f799cf4` | 1357 passed, 2 skipped, 4 deselected in 35.02s | 1361 passed, 2 skipped in 34.77s |
| FEAT-0009 | [#49](https://github.com/wx-ys/pynbody-extras/pull/49) | `cb333cfacfb343357937378c7441437be649e980` | 1426 passed, 2 skipped, 4 deselected in 37.52s | 1430 passed, 2 skipped in 35.62s |
| FEAT-0010 | [#50](https://github.com/wx-ys/pynbody-extras/pull/50) | `448fe1156f19d32aed42118cd108c1cfe0cb6c74` | 1512 passed, 2 skipped, 5 deselected in 49.53s | 1517 passed, 2 skipped in 52.14s |
| FEAT-0011 | [#51](https://github.com/wx-ys/pynbody-extras/pull/51) | `a7d6f6a053ed41a439a55ce3e88e3c6b2f2f46cb` | 1577 passed, 2 skipped, 5 deselected in 55.11s | 1582 passed, 2 skipped in 54.59s |
| FEAT-0012 | [#52](https://github.com/wx-ys/pynbody-extras/pull/52) | `1843f7608a8b0255e1aa3b798380a017f42f420f` | 1672 passed, 2 skipped, 5 deselected in 60.39s (0:01:00) | 1677 passed, 2 skipped in 61.61s (0:01:01) |
| TASK-0002 | [#53](https://github.com/wx-ys/pynbody-extras/pull/53) | `7417d949c57e01c2dcc4ac609005eaaf869eddcc` | 1679 passed, 2 skipped, 7 deselected in 64.12s (0:01:04) | 1686 passed, 2 skipped in 68.65s (0:01:08) |

Work-item metadata records done only after the API confirmed MERGED. The initial planning, local implementation and delivery-review documents retain their dated evidence; this record supersedes their earlier publication/status statements. SPIKE-0001 is closed by the approved decision record in PR41 and the actual TASK-0002 installation matrix.

The final full/core ownership, Linux-only observed installation certification, supported old-package uninstall-first transition, root-only jobs/studies, family-block order restriction, descriptor-only external history and absence of automatic migration/GC remain as specified. Other platforms and SQL dialects are not certified.

Current integration is complete. Global V1 follow-on requirements in design §8 remain separate work.

## Final merged-source verification / 合并结果复验

Verified local main at `7417d949c57e01c2dcc4ac609005eaaf869eddcc`, matching origin/main after PR53:

- `make check`: **1679 passed, 2 optional pandas skips, 7 deselected**, 67.10 s; lint, mypy124, docs27, version and lock checks passed. Log: `/tmp/pynbody-publish-main-check.log`.
- Full suite with temporary optional pandas: **1688 passed, zero skipped**, 73.45 s. Log: `/tmp/pynbody-publish-main-all.log`; no project dependency or working environment modification.
- `make cpp-test`: **ALL PASS**. Log: `/tmp/pynbody-publish-main-cpp.log`.
- **145 artifact Python/CSS/C++ inputs** compared byte-for-byte with merged source: no mismatches. Existing final installation artifacts remain the ones documented in the installation matrix.
- This closure changes documentation/status only; `make docs-check`, local-link checks and staged whitespace checks validate the closure.
