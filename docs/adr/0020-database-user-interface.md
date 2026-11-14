# ADR-020: A unified Database user interface / 统一 Database 用户接口

Date / 日期: 2026-11-14

Status / 状态: **Accepted for implementation / 已批准实施**.

## Context / 背景

Users currently compose ResultStore, AnalysisRepository, logical input registration,
loader bindings and run/result links themselves. The local onboarding examples
make this cost visible. The requested interface is calculate, save, browse and
reload, with optional automatic reuse, and may require refactoring.

现有用户要自行连接结果存储、ORM、身份、加载器和运行关联。用户要求通过
计算、保存、浏览、重载完成常见工作，并接受为此重构。

## Proposed decision / 拟议决策

Adopt Database as a deep module above the existing calculator/store/ORM seams.
Retain cc.run and the existing Result type. Add Result.save and navigable database
objects. Use one SQLite file by default, existing codecs/signatures/versioning,
and existing EvalEngine execution. Add dependency-free core protocols for
coordination and save delegation, with context-local configuration and lazy
Database imports. Extend existing input metadata to support path identities;
do not invent logical Simulation IDs from locations.

采用统一 Database 模块，保留 cc.run、Result、现有执行器和存储协议；新增 save、
对象浏览及上下文。默认一个 SQLite 文件；core 只依赖无 SQLAlchemy 的协议，
Database 负责协调。扩展已有输入元数据以容纳路径身份，不从路径臆造逻辑身份。

Capture persistent input descriptions and calculator signatures at high-level run
time, including for later explicit saves. Default file-backed high-level run to
snapshot_identity, while direct EvalEngine keeps its session default. Known
database-object cache hits precede any particle load. Auto/refresh preserve Run
records; deferred manual saves record save operations without inventing old Runs.

高层 run 运行时固定身份和计算器描述，支持延迟 save；文件输入默认使用
snapshot_identity，直接 EvalEngine 默认行为不变。已知身份的数据库对象先查缓存
再加载。auto/refresh 记录 Run；manual 后补存储记录保存操作而不伪造历史 Run。

Detailed normative proposal / 详细接口提案:
[English](../design/database-ux-2026-11-14.md) /
[中文](../design/database-ux-2026-11-14.zh-CN.md).

## Relationship to current decisions / 与既有决策的关系

- ADR-006/007: preserve typed identity and immutable versions / 保留类型身份和固定版本。
- ADR-008/009: extend logical-only ORM input registration to path inputs; do not
  silently convert keys / ORM 将支持路径输入，不静默重写旧 key。
- ADR-010: add built-in loading at the Database layer; retain explicit custom
  loader bindings and replace load-before-hit for the new interface / 新入口提供
  默认加载器并允许先命中后加载，保留自定义绑定。
- ADR-015/016: connect catalogue metadata to verified replay definitions; do not
  assume metadata alone identifies particles / 目录对象需要可验证的粒子重放定义。
- ADR-019: preserve core/full ownership and optional SQLAlchemy / 保留发行与可选依赖合同。

## Alternatives / 备选方案

More helper functions retain the user's orchestration burden. Making db.run the
main entry point splits the calculation vocabulary. A second workspace/artifact
framework duplicates existing scientific identity and execution ownership.

继续加辅助函数不能消除用户组织流程的负担；改用 db.run 会分裂计算入口；
另建 Workspace/Artifact 框架会重复已有身份与执行职责。

## Consequences / 后果

This is an interface and metadata refactor, not merely shorter tutorials. It
requires new core contracts, path-input support, save provenance, catalogue
adapters, load-free lookup and recovery tests. A single file does not promise
cross-layer atomic transactions. Implementation follows the authorization below and is validated by the
red/green evidence in FEAT-0013 through FEAT-0017. Existing low-level defaults
remain compatible; the new behavior is scoped to the Database interface.

需要重构接口与元数据，并验证运行时身份、目录适配、免加载缓存和失败恢复。
单一文件不代表跨层事务自动原子化。本决策已按下述授权实施，红绿证据见
FEAT-0013 至 FEAT-0017。已有底层默认行为保持兼容，新行为由 Database 入口提供。

## Implementation authorization / 实施授权

On 2026-11-14 the user instructed implementation and autonomous commit, pull and merge. This accepts the detailed design and core/ORM changes. Existing direct AnalysisRepository callers retain logical-only registration by default; Database opts into path support with allow_path_inputs=True.

用户于 2026-11-14 明确授权实施与自主提交、拉取、合并。现有 AnalysisRepository 默认仍仅接受逻辑身份，Database 通过 allow_path_inputs=True 启用路径支持。

### Import-layer implementation rule

`database` is above ORM and the scientific calculator classes. The only approved
upward public export is `pynbodyext.__getattr__('Database') -> pynbodyext.database`;
it is lazy and covered by a fresh-process SQLAlchemy import test. This is an
explicit extension of the package-layer contract; no core-to-database import is
allowed. The architecture test must classify the new layer rather than treating
new modules as package-root files.

## Verification / 验证

Independent reviews approved the core, ORM and catalogue integration after fixes. `make check` passed (1731 passed, 2 skipped, 7 deselected); `make test-all` passed (1738 passed, 2 skipped). The 52 new contracts cover saving, identity, context modes, replay, recovery, metadata-only browsing and path input compatibility. The runnable Tipsy example verifies reopened root, selection and halo hits without loader calls.

核心、ORM 与目录集成经过独立评审并修复。快速检查与完整测试均通过，新增 52 项合同测试；真实 Tipsy 示例验证重开数据库后根快照、选择与 halo 的缓存命中不调用加载器。
