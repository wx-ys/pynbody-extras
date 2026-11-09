# Work items and handoff

[English](README.md) | [简体中文](README.zh-CN.md)

This directory is the shared record for work that crosses devices, sessions, or
agents. A new contributor should be able to pick up a task without reading a
chat log.

For the current implementation review, revised V1 design and sequential agent
tasks, start with [Design entry point](design/README.md).

## Three records per work item

| Record | File | Purpose |
| --- | --- | --- |
| State | `meta.json` | Machine-readable identity and status. The single source of truth for `id`, `type`, `status`, `owner`, and `issue`. |
| Specification | `en.md` / `zh-CN.md` | Durable description of the requirement, with the same sections in both languages. |
| Handoff | `handoff.md` | Append-only operational log: what was done, what is next, commands, blockers. |

## Layout

```text
docs/work/
  INDEX.md                 # generated bilingual table
  _template/
    meta.json
    en.md
    zh-CN.md
    handoff.md
  BUG-0001/
    meta.json
    en.md
    zh-CN.md
    handoff.md
```

IDs are global and never reused: `REQ-*`, `FEAT-*`, `BUG-*`, `TASK-*`, `SPIKE-*`.
The directory name is the ID; the title lives in `meta.json`. This keeps paths
stable when a title changes.

## Create a work item

```bash
make work-item TYPE=bug \
  TITLE_EN="Short English title" \
  TITLE_ZH="简短中文标题"
```

Then fill in both language files and the acceptance criteria. The item starts as
`draft`.

## Status

```text
draft -> ready -> in-progress -> review -> done
                    |  ^
                    v  |
                  blocked
```

| Status | Meaning |
| --- | --- |
| `draft` | Idea; needs triage or detail. |
| `ready` | Acceptance criteria and test plan are complete; an agent can start. |
| `in-progress` | Someone is working on it. |
| `blocked` | Waiting on an external decision or dependency. |
| `review` | A pull request is open. |
| `done` | Merged and verified. |
| `archived` | No longer relevant. |

Status lives only in `meta.json`; do not duplicate it in `en.md` or `zh-CN.md`.

## Workflow

1. Create or link a GitHub issue. Put its number in `meta.json.issue`.
2. Create the work item and write both language files.
3. For a bug, reproduce it and record the actual and expected output.
4. Write the failing test before changing implementation code.
5. Implement the smallest fix; run `make check` and `make test-all` when needed.
6. Set `status` to `review`, append a `handoff.md` entry, and open a PR.
7. Update both languages in the same PR.

## Completing and archiving

1. Set `meta.json.status` to `done`, fill `completed`, `resolution`, and `pr`.
2. Add an `Outcome` section to both language files and tick every acceptance
   criterion.
3. Append a final `handoff.md` entry with the verification evidence.
4. Commit the completed item (**Commit A**). It must exist in that commit
   before archiving.
5. Keep it in `docs/work/` until the next release or for 30 days, so it can be
   reopened without restoring files.
6. Archive it:

   ```bash
   make archive-item ID=BUG-0001
   ```

   The command appends one row to `docs/history/YYYY-MM.md` and
   `docs/history/YYYY-MM.zh-CN.md`, pointing at Commit A, then deletes the
   work-item directory and regenerates `docs/work/INDEX.md`.
7. Review and commit the history row plus the deletion (**Commit B**).

The history link must point at Commit A, because Commit B no longer contains
the work item. IDs are never reused: a reopened item gets a new ID and lists the
old one in `related`.

## Handoff protocol

Append a new section; do not rewrite earlier entries:

```markdown
## 2026-10-20 14:30 +08:00 — agent-name @ device

- State: in-progress
- Done: ...
- Next: ...
- Commands: ...
- Failing test: ...
- Blockers: ...
- Files: ...
- Env: Python ... / uv ... / pynbody ...
```

Field names stay in English so tools can parse them; the prose may be in either
language. Never put tokens, absolute home paths, or machine-specific credentials
in a handoff.

## Bilingual policy

- Keep code identifiers, API names, and commands in English in both files.
- `en.md` and `zh-CN.md` must be updated in the same change.
- `docs-check` enforces pairing, IDs, status values, and index freshness; it
  cannot judge translation quality.
- When a requirement changes, update the acceptance criteria in both languages
  before changing implementation code.

## Commands

```bash
make work-item TYPE=bug TITLE_EN="..." TITLE_ZH="..."
make docs-index     # regenerate docs/work/INDEX.md
make docs-check     # validate pairs, metadata, and index freshness
make archive-item ID=BUG-0001
```
