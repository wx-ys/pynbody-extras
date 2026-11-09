# ADR-009: ORM metadata associations with explicit recovery

Date: 2026-11-09
Status: accepted for implementation; user explicitly chose separate stores
Work item: FEAT-0003

Follow V1 ADR-001/002: metadata/relations in ORM, payloads in ResultStore, existing
DAG execution. Use a new pynbodyext.orm layer above calculate, imported explicitly;
core/calculator APIs never import ORM. SQLAlchemy 2 mapped classes use SQLite as
the tested deployment. Existing store extra remains until the separate ADR-005
packaging decision is implemented. No new dependency or database migration.

Simulation(project, logical ID) -> Timestep -> DataObject -> Run -> Recipe and
ResultLink form the first graph. DataObject uses exact typed logical snapshot/view
identity and points to root for views; revision and loader belong to this versioned
input, not to the shared Timestep. DataLocation is root location metadata, not an
identity proof. Recipe stores the existing full hash/signature DAG. ResultLink
stores a pinned RecordRef, schema and the Run's named backend; no duplicate payload.
One root input/output per run; pipeline named outputs remain in its payload.

The user selected separate-store recovery instead of forced shared transactions.
Persist running first, execute/cache/store, then atomically finalize metadata and
reference. Every invocation gets a new Run, including cache hits. Raised/collected
errors produce failed runs without links. Final metadata failure returns a typed
exception with run ID and saved receipt/result, leaving running for recovery.

finish_run validates exact input/recipe/pinned version/backend without decoding
payload, then conditionally transitions running->succeeded with link insertion in
one transaction. Repeat same completion is idempotent, conflicting terminal writes
fail. failed is terminal as well; retries are new runs. Explicit recovery can leave
cache reuse unknown. No claim that a matching candidate proves execution ownership.

Audit reports unfinished/missing/invalid associations and unreadable backends,
without decoding payload. unlinked_versions lists matching versions not referenced
in this metadata repository's named store; no automatic inference, relinking,
delete/GC, or scan claiming knowledge of other repositories. Hard interruption
may leave candidates; explicit user selection plus finish_run is the recovery seam.
There is no distributed exactly-once execution or cross-store transaction.

Foreign keys and uniqueness enforced, including enabling SQLite foreign keys.
Terminal writes use conditional UPDATE RETURNING (SQLite 3.35+); concurrent entity registration may raise
unique conflicts for caller retry. Query helpers eagerly load the immediate run
chain; ordinary SQLAlchemy sessions support collection navigation and joins.

Validation covers graph persistence, identities/views, cache hits vs refresh,
failed execution, metadata rollback after payload save, interruption/recovery,
unlinked candidates, invalid/missing refs and stores, FK enforcement, import
boundaries and full regression. Existing tests are not weakened: only the declared
package layer list expands by orm. No Catalogue, MergerTree, RunStep, scheduler or
fingerprint/loader implementation in this increment.

References: SQLAlchemy 2 [relationships](https://docs.sqlalchemy.org/en/20/orm/basic_relationships.html)
and [sessions](https://docs.sqlalchemy.org/en/20/orm/session_basics.html);
[SQLite RETURNING](https://www.sqlite.org/lang_returning.html).

## 中文摘要

用户选择元数据/载荷可分离并显式恢复。新 orm 层建立模拟→时间步→输入对象→运行→
流程/固定结果引用的关联；不复制载荷，不替换执行器。先登记 running，再计算保存，
最后原子提交运行终态和引用。失败/中断可查询，finish_run 校验后显式补关联；候选
版本不是归属证明，不自动挑选或删除。revision 可省略；位置不作为身份；SQLite
外键启用。完整安装打包方案继续作为独立事项，无旧数据库迁移。
