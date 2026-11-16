# ADR-021: Database studies and detached tables / 数据库研究与独立结果表

Date / 日期: 2026-11-16

Status / 状态: **Accepted / 已批准**.

The user approved this design for staged implementation on 2026-11-16. The additive schema, detached calculation-layer types and lazy public export are part of this decision.
用户于 2026-11-16 批准分阶段实施，包含新增元数据表、计算层独立类型与惰性公开导出。

## Context / 背景

ADR-020 provides single-object saving, browsing and lookup-before-load execution. ORM run_study predates that interface: it validates roots and loads before cache lookup. Mutable SampleTable and detached ORM StudyReport also have different state and serialization contracts. Merely wrapping them would not deliver a consistent, independently reopenable high-level Study.

ADR-020 已提供单对象保存、浏览和先查缓存后加载。更早的 ORM run_study 只接受根输入，且先加载再查缓存。可变 SampleTable 与独立 ORM StudyReport 的状态和序列化合同也不同，不能直接包装后声称得到统一、可独立重开的高层 Study。

## Decision / 决策

1. Keep the existing exact typed input × full CalculatorSignature result-store key. Study names, column labels and definition hashes do not replace it. Route both individual calls and Study cells through the Database coordinator, with the existing EvalEngine and ResultStore as owners.
2. Save Study as a frozen ordered input/signature definition. Inspect and reopen without rebuilding calculators; require code/loaders only for cells that must compute. Exact-signature live bindings are explicit and process-local.
3. Return one new detached, read-only StudyTable from high-level run and collect. Retain legacy SampleTable and ORM StudyReport contracts. Separate current-cache observations from invocation outcomes; failed/incomplete cells never claim an old or unconfirmed successful ref.
4. Give exported input and result references a persistent Database UUID namespace. Study reopens from its owning database; a portable table reopens from its own file without database, particles or project calculator code.
5. Add read-only opening and metadata-only query guarantees, including catalogue handles. Keep manual/auto/refresh and use raise/collect in the new high-level Study. Preserve old lower-level error spellings.

1. 保留精确类型化输入 × 完整 CalculatorSignature 的现有存储 key。Study 名称、列名和定义摘要不替代它；单次与批量格共用 Database 协调入口，EvalEngine 和 ResultStore 继续负责原有职责。
2. Study 保存固定、有序的输入与签名定义。重开与查看不重建 Calculator，只有必须计算的格才需要实现和加载器；活绑定必须精确匹配签名，且只在当前进程生效。
3. 高层 run 与 collect 统一返回新的独立只读 StudyTable。保留旧 SampleTable 与 ORM StudyReport 合同；区分当前缓存观察和本次调用结果，失败/未完成格不借旧引用或未确认回执冒充成功。
4. 导出的输入和结果引用携带持久 Database UUID 命名空间。Study 从所属数据库重开，可移植表从自身文件重开，不需要数据库、粒子或项目 Calculator 代码。
5. 增加只读打开和元数据查询保证，覆盖目录句柄。保留 manual/auto/refresh；新 Study 使用 raise/collect，旧低层错误拼写保持兼容。

Normative contracts, review findings R01–R12 and acceptance scenarios A01–A32 / 规范合同、评审发现与验收场景：
[English](../design/database-study-2026-11-16.md) · [中文](../design/database-study-2026-11-16.zh-CN.md).

## Layering and compatibility / 分层与兼容性

Database owns Study registration, binding and coordination above ORM. Detached table/cell/reference data and portable codec validation belong to the calculation layer, with no SQLAlchemy or upward Database import. A new lazy root StudyTable export must extend the explicit import contract. Namespace and Study metadata are additive; existing result addresses and payloads are not rewritten. Read-only opening never initializes or upgrades storage.

Database 在 ORM 上方管理 Study 登记、绑定与协调。独立表、单元格、引用数据及可移植编码验证位于计算层，不导入 SQLAlchemy 或上层 Database。新增惰性根 StudyTable 导出需要扩展明确的导入合同。命名空间和 Study 元数据以新增方式登记，不改写已有结果地址或载荷；只读打开不初始化或升级存储。

SampleTable v1/v2 and ORM study/v1 retain their existing loaders. The new detached format has its own schema and rejects opaque/pickle tags; it does not reinterpret ambiguous legacy None cells or invent missing Run/ref history. File backup preserves database namespace but does not guarantee every pinned version is present in a diverged copy.

旧 SampleTable v1/v2 和 ORM study/v1 保留原加载器。新独立表使用新 schema 并拒绝 opaque/pickle 标签，不重新解释旧的歧义 None，也不补造 Run/ref 历史。数据库备份保留命名空间，但分叉副本不保证仍包含所有固定版本。

## Alternatives and scope / 备选与范围

Rejected: implicit Pipeline fusion; input-cardinality-dependent Calculator.run return types; silently changing mutable SampleTable; a second execution engine. Deferred: path-only registration, shared live particle caching, not-applicable outcomes without a domain protocol, dynamic samples, cross-database/multi-input studies and distributed scheduling.

不采用隐式 Pipeline 融合、按输入数量改变 Calculator.run 返回类型、悄悄改变可变 SampleTable 或第二执行器。暂缓仅路径登记、共享活粒子缓存、缺少领域协议的 not-applicable 状态、动态样本、跨库/多输入研究及分布式调度。

## Review and implementation / 评审与实施

[SPIKE-0002](../work/SPIKE-0002/en.md) records the design review. [FEAT-0018](../work/FEAT-0018/en.md) records implementation and acceptance evidence.
[SPIKE-0002](../work/SPIKE-0002/zh-CN.md) 记录设计评审，[FEAT-0018](../work/FEAT-0018/zh-CN.md) 记录实施与验收证据。
