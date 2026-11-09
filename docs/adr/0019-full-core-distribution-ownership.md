# ADR-019: Full and lightweight distribution ownership

Date: 2026-11-09
Status: accepted — user explicitly approved proposal and candidate names
Work items: SPIKE-0001, TASK-0002

## Decision

Use `pynbodyext-core` to own every `pynbodyext/**` Python/native/CSS path.
Use `pynbodyext` as a metadata-only full distribution requiring exactly the same
core version plus SQLAlchemy >=2.0. Keep Python >=3.11 and pynbody >=2.8.0.
Core retains optional store/image/dev/bench extras; full forwards those extras
to the exact core version. No two distributions own the same installed path.
Use one version source and synchronize pyproject/uv.lock with both members.

The user reviewed docs/design/packaging-decision.md and answered
“批准该发行方案与候选名称”. Names are approved candidates, not reserved or published.

## Consequences

Preserves ADR-005: the default full installation includes ORM; a documented core
installation supports calculations without SQLAlchemy. Extras add dependencies;
--no-deps is not a supported lightweight path. Validate actual wheel and sdist
installations outside the source tree. Include all C++ headers and assert actual
native presence/import because optional compilation may exit 0 after failure.
Test install/uninstall ownership and old single-distribution upgrade behavior.
Record each supported platform cell passed/failed/not-run with evidence.

No registry publication or untested cross-platform certification. See the
[approved proposal and experiment evidence](../design/packaging-decision.md).

## 中文摘要

已批准 pynbodyext-core 独占源码/native/CSS，pynbodyext 为强制依赖同版本 core 与
SQLAlchemy 的完整元包。保留并转发 extras，同步版本与锁文件。TASK-0002 在隔离
环境验证实际产物，修复 sdist 头文件缺失；发布与名称保留不在授权范围。
