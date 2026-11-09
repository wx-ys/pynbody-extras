# FEAT-0005 implementation plan

Approved contract: ADR-011, implemented on verified local BUG-0010/11/12.
Existing checkout: `codex/sequential-review-fixes`; preserve unrelated changes.

1. Add behavioral tests for immutable cell receipts across refresh/reopen/cache
   invalidation, hits/fill, missing persistence and failures, and typed exports.
   Confirm failures are due to absent refs/store_name/schema behavior.
2. Append defaulted SampleRow.refs; capture actual successful Result.store_ref,
   clear stale receipts before retries, propagate store_name through filtering and
   store construction, and serialize v2 with pinned typed-key validation.
   Keep v1 readable with empty refs; load must never query/rebind a backend.
3. Export JSON-compatible DataFrame metadata keyed by sim_identity_key with
   identity-key order retained alongside duplicate display names.
4. Run focused regressions, make check, and make test-all; log exact results in
   /tmp/feat0005-*. Update both locales, review metadata and handoff, and the
   SampleTable glossary in CONTEXT.md. Do not commit or push.

Known limit: successful None values still share the existing missing-cell sentinel.
