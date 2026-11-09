# FEAT-0002: 配置与位置解耦的快照逻辑身份

## 背景

PR #37 已提供不可变结果引用，但默认快照身份仍使用绝对路径，搬迁后无法命中原缓存。
文件位置、结果保存版本、科学数据 revision 是三个不同概念。
原 V1 ADR-003/004 已要求用户/项目提供逻辑 ID、位置单独处理，默认不计算完整内容哈希。
用户于 2026-11-09 选择 revision 可省略，省略时数据变动由用户通过 reuse=False 刷新。

## 目标

通过现有 sim_identity 参数接入可选的配置式身份提供器。已配置快照搬迁后身份保持
不变；未配置对象沿用路径/session 行为。不重写执行与存储系统。

## 设计

见 [ADR-008](../../adr/0008-configured-logical-snapshot-identity.md)。
LogicalSnapshotIdentity 是冻结值对象：project_id、simulation_id、timestep_id 为
必填非空字符串；revision 默认为 None，可选非空字符串；loader_id 为非空字符串，
默认 "default"。不自动转字符串、不大小写归一化、不裁剪 ID；纯空白字段无效。
loader_id 由用户标识影响解释结果的加载设置，不自动收集 pynbody 配置或插件。

ConfiguredSnapshotIdentity 接受本地路径/文件前缀到逻辑身份的映射，构造时复制配置，
规范为绝对路径，不解析符号链接，也不访问文件。同一身份可绑定多个路径；同一规范化
路径的不同身份报 ValueError。调整位置或 revision 时新建 provider，无全局注册表。

按 ancestor.filename 匹配配置。输出包含所有逻辑字段的版本化稳定 tuple；子集/halo
继续附加原有索引摘要与 halo 编号。快照或视图在计算器外被变换时仍退回 session。
未匹配路径沿用 snapshot_identity；未配置的内存快照仍为 session。

配置 JSON 使用严格的版本化 schema，包含绝对路径与全部身份字段。读取时校验 schema、
字段和重复项；拒绝相对路径，避免换工作目录后配置含义改变。不在配置和身份查询中
加载模拟数据或读取文件。

## 验收标准

- [x] 显式逻辑身份在路径搬迁/增加别名后保持不变。
- [x] 项目、模拟、时间步、revision 或 loader 改变时身份不同。
- [x] revision 省略时允许复用；reuse=False 刷新，旧固定结果引用仍可读取。
- [x] 搬迁副本的相同子集/顺序身份一致；不同索引、顺序或 halo 编号区分身份；
  外部变换仍退回 session。
- [x] 未配置对象的原有路径/session 行为不变。
- [x] 规范化路径冲突、非法配置均拒绝。
- [x] 配置 JSON 能跨进程、SQLite 重开后复用。
- [x] Calculator.run、EvalEngine、SampleTable 直接使用现有 seam；内存/SQLite 均验证。
  SampleTable 默认行名包含逻辑字段，保留 halo/subset 后缀。
- [x] 原有测试保持不变；make check、make test-all 通过。

## 非目标及边界

不涉及 ORM 表、catalog/halo 逻辑登记、自动发现、指纹、文件 IO、数据加载、迁移、
打包调整或历史清理。配置是用户对数据等价性的声明，相同标签不证明内容相同。
不填 revision 时需逐项刷新受影响的计算（包含子集），不自动全局失效。
填写 revision 时通过更新 provider 让后续键使用新版本。加载解释改变时更新 loader_id。
子集索引身份依赖父数据粒子顺序不变，由 revision 或显式刷新负责管理。

## 测试计划

见 [实施步骤](plan.md)。先增加 tests/test_logical_snapshot_identity.py 并运行确认失败，
再实现。覆盖两个存储后端、SQLite 重开、非法配置、跨进程稳定性、子集及变换。

## 结果

已实现：codex/logical-snapshot-identity，基于 main aaf9b92。
初始 API 契约测试在实现前 45 项失败，实现后 45 项通过。随后增加两项默认行名回归，
确认失败后修正命名辅助函数；身份与 SampleTable 测试 70 通过、1 跳过。
此前集成/分层专项测试 150 通过、1 跳过。最终验证包含全部 47 项新增测试。

make check：ruff/mypy/文档检查通过，1020 通过、1 跳过、3 未选中。
make test-all：1023 通过、1 跳过。原有测试未修改。
独立审查发现并复核了 JSON 工厂的类型修正（改为明确构造当前配置类的静态工厂），
并确认行名修正后无剩余问题。发布与合并已获授权，无待决实现问题。
