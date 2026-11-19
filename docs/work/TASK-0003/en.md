# TASK-0003 — Audit Sourcery feedback on merged pull requests

## Scope

Review every Sourcery conversation comment, submitted review and inline thread on PRs #1–59 against main at `60fc13f`. Recheck resolved and outdated threads against the current implementation. Record the evidence for each disposition.

## Acceptance criteria

- A mismatch between sample objects and supplied names fails before calculating or writing a row. Duplicate display names survive export. A name-only lookup detects ambiguity. Numeric columns return NumPy arrays and preserve complex values.
- The structured-array codec enforces a fixed 1 MiB NPY header limit on both write and read. Supported wide records continue to round-trip with pickle disabled.
- Check the reported signature, storage, rendering and Pipeline issues against their public contracts. Exercise confirmed changes through their normal API.
- Preserve the cell sentinel and typed row identities specified by ADR-011. Verify the calculator layers, existing tests, data-backed tests and bilingual index before merging.

## Review record

GitHub returned 79 conversation comments (53 review guides and 26 quota or size notices) and 33 review submissions. The 51 inline findings spanned 18 PRs. At the audit date, 40 threads were open and 11 were marked resolved.

| Merged PR | Disposition on current main and in this change |
| --- | --- |
| [#3](https://github.com/wx-ys/pynbody-extras/pull/3) | The render tests import tracked helpers and assert the exact `RBelow` label. Both original reports have regression coverage. |
| [#4](https://github.com/wx-ys/pynbody-extras/pull/4) | Restored the deprecated `statistics.parse_stat_key` import for legacy queries. Typed Card blocks already escape labels and values. Tests cover literal markup in both renderers. The original raw HTML helpers and their stale docstrings have been removed. |
| [#7](https://github.com/wx-ys/pynbody-extras/pull/7) | Corrected the `RBelow.calculate` test helper's return annotation to a NumPy boolean mask. |
| [#14](https://github.com/wx-ys/pynbody-extras/pull/14) | The current positional-default label documentation agrees with the tested behavior when another argument is explicit. |
| [#16](https://github.com/wx-ys/pynbody-extras/pull/16) | Image annotation resolution uses the documented lazy `resolve_type_hints` helper to keep plotting imports out of startup. The artist union already accepts scatter `PathCollection` objects for colour bars. |
| [#17](https://github.com/wx-ys/pynbody-extras/pull/17) | Use initialized factory values and show non-dataclass recipe parameters. Build drill-down content on demand. Document asyncio style inheritance and the `InfoView` subclass migration from `_display_body()` to typed `_display_blocks()`. |
| [#18](https://github.com/wx-ys/pynbody-extras/pull/18) | Restrict Pipeline reconstruction to Pipeline classes and document its trusted-input contract. Distinguish object-array contents and opaque node instances, reject unchecked stored records, compare canonical identities for NaN, and prevent legacy session keys from writing durable results. Fix mutable Pipeline outputs and subclass identity. A plain Pipeline’s transform scope already participates in its signature. ADR-007 versions concurrent saves and ADR-006 explicitly excludes a short-key migration. |
| [#22](https://github.com/wx-ys/pynbody-extras/pull/22) | Validate the name count before computing any cell. Preserve duplicate display names and typed row identities, and report an ambiguous name-only lookup. Preserve complex numeric columns and explicitly test the NumPy return type. ADR-011 reserves `None` as the missing-cell sentinel in this table format, including successful `None` values. |
| [#23](https://github.com/wx-ys/pynbody-extras/pull/23) | Use an iterative traversal for result trees. A rich card can render the last node in a chain deeper than Python’s recursion limit. |
| [#25](https://github.com/wx-ys/pynbody-extras/pull/25) | The public `vcirc` error already names the native extension and retains its load failure. The native test gates now propagate a broken installation and skip an absent extension. |
| [#33](https://github.com/wx-ys/pynbody-extras/pull/33) | The work-item link already points to a commit-qualified repository URL that works from the PR and the Markdown file. |
| [#34](https://github.com/wx-ys/pynbody-extras/pull/34), [#35](https://github.com/wx-ys/pynbody-extras/pull/35) | Constant boundary windows already exclude out-of-image samples. Tests cover the pynbody minimum version, bounded median batches, extreme finite values and the documented even-window conventions. |
| [#37](https://github.com/wx-ys/pynbody-extras/pull/37) | Corrected the `EvalEngine.run` refresh documentation. Each refresh appends an immutable version and advances the current cache pointer. |
| [#43](https://github.com/wx-ys/pynbody-extras/pull/43) | Enforce a 1 MiB header limit before reading NPY header text. The encoder checks the same limit. A large data section with a small header still round-trips. |
| [#44](https://github.com/wx-ys/pynbody-extras/pull/44) | Use the public term “cell errors” in the recovery test names and module description. |
| [#57](https://github.com/wx-ys/pynbody-extras/pull/57) | Restore custom TransformChains that mix directly applied and delegated steps in receipt order. Detach borrowed arrays in tuple, list and dict subclass attributes, preserving aliases. ADR-022 assigns shared-root synchronization to callers and distinguishes recoverable preparation failures from apply hooks with unknown mutations. Tests and runnable examples use named-output terminology. |
| [#58](https://github.com/wx-ys/pynbody-extras/pull/58) | Joined worker transforms and lazy raw Filter consumption have coverage. Database-confirmed saves and recovery publish historical member origins and use sites accessible by pinned version. Value-only store hits explicitly expose `selection_observations is None`. A scoped input delivered to `check_input` counts as consumed under ADR-023. |

The Pipeline subclass fix also covers existing pickles and catalogues. Reconstructed recipes preserve output names, order and cache identity. Repeated aliases are individually discoverable in Database results. In an independently checked seven-layer recipe, storing each child once reduced the encoded size from 415,441 to 1,739 bytes.
