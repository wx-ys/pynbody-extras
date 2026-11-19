# FEAT-0022 · Discover saved outputs and trace their particle members

Match all predicates on one named output and return its pinned receipt. Link a historical output to its member origins in both directions. Query recipe metadata in a read-only database. [Implementation plan](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

Four initial query cases failed at the missing public discovery entry point. Regressions cover class and instance filters, ordered transforms, dependencies, read-only recipe fallback, concurrent refresh and pinned Study cells.

Named leaves retain the root version and their individual output paths. Read-only discovery and pinned Study cells preserve the original member evidence across a concurrent refresh.

[tests/test_database_result_queries.py](../../../tests/test_database_result_queries.py) · [Handoff](handoff.md)
