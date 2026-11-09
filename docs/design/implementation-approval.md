# 后续实施合同确认 / Implementation contract review

2026-11-09，基线 `5df15dd`，实施分支 `codex/sequential-review-fixes`。

BUG-0010/0011/0012 已完成本地实现与验证，状态为 review，尚未合并。
最终 `make check`：1155 passed, 1 skipped, 4 deselected；
`make test-all`：1159 passed, 1 skipped。后续实施将使用这些已验证本地变更，
逐项保留验收与 handoff，不假称前项已经远端合并。

以下合同来自既有双语工作项与修订设计；本文件记录已由用户审阅批准的合同摘要。
接受决定已记录为 ADR-011–019；按路线图逐项实现、测试和审阅。
只涉及本地实施；发布、迁移、自动 GC、跨平台认证不在此确认范围。

| 工作项 | 已批准的具体合同 | 边界与失败语义 |
|---|---|---|
| [FEAT-0005](../work/FEAT-0005/zh-CN.md) | SampleRow 末尾增加默认 refs；SampleTable 增加可选 keyword-only store_name；新 JSON schema 保存 pinned RecordRef | 成功后记录实际 ref，失败/未持久化成功清理旧 ref；load 不读 backend；校验行类型化身份/列 hash/固定版本；旧 schema 引用留空；where/DataFrame 保留身份；不改 None 缺失哨兵 |
| [FEAT-0006](../work/FEAT-0006/zh-CN.md) | Run.execution_context 保存带版本 JSON；run/run_input/begin_run 接收 execution_metadata（仅 tool_versions 与 random_seed） | begin_run 前复制并验证；保存此次调用环境与输入 URI/loader/revision；未知版本为 null；失败也保留；不设置随机种子、不改变缓存键、不把命中环境当生产环境 |
| [FEAT-0007](../work/FEAT-0007/zh-CN.md) | frozen ExternalFileRef + 带版本 codec tag；绝对本地 path/URI、media_type、可选大小/SHA256；显式 inspect | encode/decode 不做 IO；inspect 返回 available/missing/changed，访问错误传播；拒绝远端 URI、相对路径及非法字段；文件始终调用者所有，不复制/删除 |
| [FEAT-0008](../work/FEAT-0008/zh-CN.md) | PostprocessorBase 复用 PropertyBase/Param 与现有 DAG；唯一动态参数 source；适配器调用 process(params.source) | source 只接受 Calculator，其他配置参数静态；拒绝 Result/RecordRef 输入；签名包含 source/静态参数并保留版本与可重建规则；不建新执行器或拟合依赖 |
| [FEAT-0009](../work/FEAT-0009/zh-CN.md) | Catalogue 与 CatalogueObject 及 register/get/list；类型化目录身份包含 root、catalogue_id、finder、revision、loader_id | 只登记元数据；同目录 halo 才能作 subhalo host；null revision 的唯一性由 canonical key 保证；不改 DataObject.parent、现有 view/provider 身份或 loader |
| [FEAT-0010](../work/FEAT-0010/zh-CN.md) | SelectionSpec(indices/family/filter)、Selection 关系与 Run.selection_id；register_selection(LoadedInput, spec)、load_selection、run_selection | 只登记未变换根，登记时实际验证；根相对索引有序可重复、拒绝 bool/越界；filter 稳定可重建；定义和物化 view 可多对一；Run 位置归属验证扩展到根；保留 arbitrary view 的 load_input 拒绝行为 |
| [FEAT-0011](../work/FEAT-0011/zh-CN.md) | frozen JobSpec；严格 JSON 字段/schema；from_calculator/to_json/from_json/execute(repo) | 仅根输入、默认 RunOptions；拒绝 unstable/nonconstructible signature；仅解析已注册计算类型；绑定不序列化，执行委托 run_input，不持久化 _EvaluationPlan |
| [FEAT-0012](../work/FEAT-0012/zh-CN.md) | run_study 串行调用 run_input；StudyReport 每格 value/error/run/ref；全列表先验证 | 唯一根输入 IDs；errors=record 仅继续执行错误；持久化错误抛 StudyPersistenceError 并携带 partial report/receipt/cause；JSON load 只读；恢复为新调用/新 Run，不承诺 exactly-once |

各项验收以链接中的完整双语标准为准，仍需各自测试红灯、最小实现、专项与全套验证。
FEAT-0005/0006/0007 是 M1；其后 M2 为 0008/0009/0010，M3 为 0011/0012。

发行方案单独见 [packaging-decision.md](packaging-decision.md)。推荐候选为
`pynbodyext-core` 独占全部 Python/native/CSS，`pynbodyext` 为强制依赖 ORM 的完整元包。
这些名称尚未保留或发布。SPIKE 只收集证据，TASK-0002 需名称与方案获批后才能实施；
源码包漏 C++ 头文件必须在该安装矩阵中修复并验证。

审批状态：2026-11-09 用户明确回复“批准全部 FEAT-0005–0012 合同”及
“批准该发行方案与候选名称”。批准覆盖全部八项及发行方案，不需重复确认。

English: This is a concrete review packet for the existing proposed contracts,
now recorded as accepted ADR-011–019 after explicit user approval. The first three bugs are locally tested,
unmerged increments. Approval permits focused ADRs and sequential local feature
implementation; publishing, migrations and platform certification are separate.
