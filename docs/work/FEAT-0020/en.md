# FEAT-0020 · Publish exact particle member origins with confirmed saves

Attach actual selections to their archived output versions in the existing save and recovery protocol. Group matching origins by exact input and retain separate scientific catalogue locators. [Implementation plan](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

Thirteen initial tests failed because snapshot and catalogue handles lacked members. The contracts cover deferred save, concurrent writers, cache reuse, incomplete publication, nested origins and ten outputs sharing one block.

A successful save atomically publishes output uses with the receipt. Cache hits retain the original scientific producer. Ten outputs that consume the same ordered sequence of 12,000 particles share one durable index block.

[tests/test_database_members.py](../../../tests/test_database_members.py) · [tests/test_database_member_recovery.py](../../../tests/test_database_member_recovery.py) · [Handoff](handoff.md)
