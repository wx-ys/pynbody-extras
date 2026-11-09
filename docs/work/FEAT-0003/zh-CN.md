# FEAT-0003: 通过 ORM 运行记录关联模拟输入、计算流程和保存结果

## 背景与目标

在 PR #38 逻辑身份和 PR #37 不可变结果引用上构建首条 ORM 关联链。遵循原 V1
ADR-001/002：ORM 保存元数据和引用，ResultStore 保存载荷，沿用现有 Calculator DAG。
用户于 2026-11-09 明确选择元数据/载荷可分离，提供校验和手动恢复。不兼容旧数据库；
revision 仍可省略，由用户 reuse=False 刷新。

## 实体与边界

新增 pynbodyext.orm，包含 SQLAlchemy 2 映射类及 AnalysisRepository。ORM 依赖
现有计算层，纯计算入口不反向导入 ORM/SQLAlchemy。

- Simulation：项目 ID + 模拟逻辑 ID 唯一。
- Timestep：所属模拟 + 时间步逻辑 ID 唯一。
- DataObject：按完整类型化输入身份唯一；关联 Timestep，区分 revision、loader
  和 halo/subset。视图关联根 DataObject，暂不推断 Catalogue 内父子关系。
- DataLocation：根 DataObject 的位置元数据，可有多个别名；同一路径可在不同历史
  revision 上出现，不自动读取文件。
- Recipe：完整 CalculatorSignature hash + JSON，流程组合和依赖仍由 DAG 签名表达。
- Run：UUID、输入及流程外键、store_name、起止时间、状态、异常信息和可空 reused。
  每次调用都建新记录，命中缓存也记录一次运行。
- ResultLink：每次运行一个根输出引用，固定 RecordRef JSON + store schema。
  Pipeline 命名输出保留在原载荷中，ORM 不重复存数组。

## 操作接口

AnalysisRepository(url_or_engine, stores={名称: ResultStore}, create=True)。
register_input(逻辑身份对象或 provider tuple, location=None) 幂等登记层级，仅接受
配置式逻辑身份及已有 halo/subset 后缀，不把路径/session 自动升格为科学对象。
register_recipe 校验稳定性；已有 hash 的规范参数必须一致。

begin_run(input_id, recipe_hash, store_name) 先持久化 running 记录。
run(calculator, sim, sim_identity=provider, store_name=..., reuse=True, options=None)
登记输入/流程后开始运行，调用现有 calculator.run，关联准确 result.store_ref，返回
RecordedRun(run, result)。收集模式的异常返回 failed 运行和原 Result；抛出模式通过
带 run_id 的异常保留原异常链。失败运行没有结果关联。

finish_run(run_id, ref, reused=None) 在关联前校验 backend、固定版本、输入和流程。
状态更新与引用插入在一个元数据事务中完成；同引用完成操作幂等，终态不同引用拒绝。
手工恢复若无法确认缓存来源，reused 留 None。fail_run 只允许 running 转 failed。
终态迁移使用带状态条件的更新，防止并发覆盖。

get_run/runs（按 input_id、recipe_hash、status 筛选）预加载输入→时间步→模拟、
流程及结果引用；session() 提供普通 SQLAlchemy Session 供关联查询。
fetch_result(run_id) 只读取固定历史版本，不退回最新缓存。

## 中断与恢复

即便两个 URL 指向同一数据库，也不承诺跨存储原子提交。结果保存后的元数据提交
失败时，RunPersistenceError 带回 run_id、result_ref 和内存 Result，之前的 running
记录保留。显式选择引用后通过 finish_run 校验并补关联，绝不自动挑最新值。
Audit 不解码载荷，报告未完成、缺 backend/引用/载荷、非法引用及读取异常。
unlinked_versions(run_id) 列出该命名 store 中同输入/流程、未被当前 repository 引用
的保存版本；仅是候选，不证明归属、不表示可以删除，也不自动关联或删除。
进程可能在任一步中断，不承诺自动恰好一次执行。

## 验收标准

- [x] SQLite 重开后及 SQLAlchemy join 可查询完整对象关联链。
- [x] 输入/流程幂等登记；模拟、时间步、revision、loader、视图隔离；位置单独记录。
- [x] 缓存命中也记录 Run，刷新后旧 Run 仍读取旧结果。
- [x] 抛出/收集异常均形成 failed 运行，无结果引用。
- [x] 执行前登记失败不计算；最后提交失败完整回滚到未完成，保留引用并可幂等恢复。
- [x] 拒绝不稳定流程、路径/session 身份、非固定/缺失/错身份引用、终态冲突及非法 backend。
- [x] Audit 不解码载荷即可识别缺失关联和中断；未关联版本仅作恢复候选。
- [x] SQLite 外键和唯一约束生效。
- [x] 新 orm 分层明确；无 SQLAlchemy 时纯计算仍可用；不迁移、不削弱旧测试。
- [x] 专项、make check、make test-all 通过，并记录红绿证据。

## 限制与后续工作

本阶段单根输入/输出、同步本地执行，验证 SQLite。不做分布式锁/重试、心跳、自动
恢复/GC、RunStep、完整 Catalogue/MergerTree、DAG 节点关系表、数据加载或 UI。
并发登记同一实体可能触发唯一约束，调用方可重试；每个 Session 对应一个工作单元。
显式配置内存 store 时，持久元数据仍可能比载荷长寿；重开需绑定同名 store 且版本
仍存在。替换名称绑定不搬迁数据。载荷信任边界沿用 ValueCodec。
ADR-005 的默认完整安装/轻量发行方案仍单独推进，本阶段使用已有 store extra，
不重新决定打包策略。

## 计划与进度

见 [ADR-009](../../adr/0009-orm-analysis-associations.md) 和 [计划](plan.md)。
已实现，基于 main a29081c。基线身份/存储测试 91 项通过。
红：20 项缺少 ORM API 失败，1 项纯计算检查通过；绿：21 项通过。
随后损坏引用形状/层级关系两项测试先失败，补上校验后通过。
最终 ORM 测试 26 项通过，包含并发冲突完成、载荷写入失败和失败状态登记中断。
独立审查无剩余问题，ORM/包分层独立检查 31 项通过。
make check：ruff、116 个模块 mypy、文档均通过，1046 通过、1 跳过、3 未选中；
make test-all：1049 通过、1 跳过。旧测试行为未削弱，包分层声明按 ADR-009 增加 orm。

条件 UPDATE RETURNING 需要 SQLite 3.35+，未声称已验证其他 SQL 方言。
发布与合并已获授权。
