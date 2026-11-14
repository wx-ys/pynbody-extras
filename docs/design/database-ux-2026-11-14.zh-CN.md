# Database 使用体验设计

日期：2026-11-14。源码基线：`41dc264`。

状态：用户已授权实施以及自主提交、拉取、合并，实施与验证记录于 FEAT-0013–0017。
当前使用指南见 [docs/database.zh-CN.md](../database.zh-CN.md)；本文保留已接受的设计。
[English](database-ux-2026-11-14.md) · [架构决策草案](../adr/0020-database-user-interface.md)

## 1. 用户只需要记住的规则

Calculator 负责计算，Result 承载结果，Database 管理保存和查找。
点号访问固定属性，方括号从集合中选对象；数据库对象的 `.load()` 返回粒子。

```python
import pynbody
import pynbodyext as pe
from pynbodyext.properties import ParamSum

sim = pynbody.load('/data/snapshot_099')
cc = ParamSum('mass')
res = cc.run(sim)

db = pe.Database('analysis.sqlite')
ref = res.save(db, name='total_mass')

snap = db.snapshots['snapshot_099']
saved = snap.results['total_mass']
saved.value
saved.calculator
particles = snap.load()
db.close()
```

文中的新接口都是目标用法，不是现在可运行的教程。`save()` 返回固定版本的
`RecordRef`，并更新当前 Result 的 `store_ref`；Result 本身仍是原来的对象。
`Database` 接受本地 SQLite 文件路径；普通计算无需实例化它或安装 SQLAlchemy。
初版只认证 SQLite，不因此宣称认证其他 SQL 方言。

## 2. 自动计算与保存

```python
with pe.Database('analysis.sqlite') as db:
    with db.use(mode='auto'):
        first = cc.run(sim)
        second = cc.run(sim)
        fresh = cc.run(sim, mode='refresh')

    assert second.from_store
    snap = db.snapshots['snapshot_099']
    cached = cc.run(snap)  # 已绑定 db，默认 auto；命中时不加载粒子
```

| 模式 | 查持久缓存 | 未命中或重新计算后保存 |
| --- | --- | --- |
| `manual` | 否 | 否，显式调用 `res.save()` |
| `auto` | 是 | 是 |
| `refresh` | 否 | 是，新版本 |

普通 `cc.run(sim)` 无数据库、无上下文时默认 manual。显式
`cc.run(sim, database=db)` 和数据库对象输入默认 auto。`db.use()` 默认 auto。
单次显式 mode 优先于上下文模式；未知 mode 在计算前报错。

数据库选择顺序：显式 database、数据库对象自身、当前 `db.use()`。
数据库对象属于 A 时，显式指定 B 报归属错误；上下文 B 不将它改绑。
对象属于 A 而当前上下文属于 B 时，只采用 A 的默认模式，不借用 B 的模式。
嵌套上下文退出或异常后恢复外层配置，采用 ContextVar，不采用可变全局默认值。
上下文不是数据库事务，不在计算期间持有数据库写锁；退出 `use()` 不关闭 db。
`with Database(...)` 退出会关闭其资源；已解码的 Result 数值仍可使用。

`res.save()` 的目标顺序为显式参数、当前上下文、Result 已绑定的数据库；都没有则
报错并提示 `res.save(db)`。保存到另一个数据库是显式复制，输入身份不能重新解释。
结果绑定与对象运行绑定的优先级有意不同：save 的上下文表示本次保存目的地。
绑定数据库关闭后报错，不隐式重开，也不自动转存其他数据库。

既有 `store=` / `reuse=` 接口保留。显式低层 store 调用不采用环境数据库配置；
与显式 database/mode 或数据库对象输入混用则在执行前报错。`reuse=False` 的既有
语义保留；新入口统一使用 mode，避免同一次调用出现两个缓存策略。

## 3. 浏览规则与名字

```python
db.snapshots
snap = db.snapshots['snapshot_099']
snap.results
snap.results[cc]                 # 按完整 CalculatorSignature 精确查询
snap.results['total_mass']       # 按显式别名查询
snap.load()

# 只有显式登记了模拟归属，才出现在此层级中。
db.simulations['TNG50-1'].snapshots['099']
```

所有集合只展示已登记条目，不隐式扫描文件系统或完整 halo catalogue。
repr 显示有上限的元数据表及总数，不加载粒子、解码大结果或重建计算器。
结果表至少展示名字、计算器摘要、hash 摘要、值类型/形状/单位、当前版本与保存时间。

Snapshot 是一个时刻的数据，Simulation 是整套模拟。自动路径登记的快照可以没有
Simulation/Timestep 归属；不得从父目录或 basename 猜科学意义上的模拟身份。
默认快照展示名取 basename，但 basename 不参与唯一身份。每个快照提供不可歧义的
`key`；`db.snapshots[name]` 同名时抛歧义异常并列出候选 key/路径，用户可按 key 选择。
不存在时抛 KeyError，不返回 None，也不创建条目。

显式结果别名在一个对象内唯一，绑定完整 calculator hash。重复绑定同一 hash
允许；绑定不同 hash 则报冲突，不偷偷替换。显示名/pretty key 可重复，但没有显式
别名时不自动变成可查询的字符串 key。始终可用 `results[cc]` 查找。
结果名字不影响缓存；依赖、参数、signature version 和 Pipeline 输出名影响缓存。

`results[...]` 返回当前已保存版本的 Result；无结果则抛 KeyError，绝不偷偷运行。
刷新后重新查询获得新版本；之前取得的 Result/store_ref 仍指向原版本。
每次成功调用 `save()` 都追加一个版本，与当前 ResultStore 合同一致。
`auto` 命中不创建新结果版本。Pipeline 保存完整根结果和 named outputs，不把
每个输出伪装成可独立重建的 Calculator。

`res.calculator` 沿用当前 Result 类型：可重建时返回 calculator，否则为 None。
其结构化 signature/表达式仍可查看，数值读取不以成功重建 calculator 为条件。
这是对上一轮“重建失败时属性抛错”建议的收敛，避免破坏现有字段合同。

## 4. 身份与延迟保存

`res = cc.run(sim); res.save(db)` 必须和直接 auto 模式生成相同缓存键。
高层 run 对常规文件输入默认采用已有 snapshot_identity；直接 EvalEngine 的
session 默认值、无文件对象和外部活动变换的 session 语义保持不变。
这是高层 run 的默认行为变更，先记录决策、再增加验收测试；不能靠保存时读取
活的 sim/calculator 反推计算发生时的身份。

运行前捕获不可变输入描述及 CalculatorSignature。描述包含根位置、身份、loader
标识、可用的单位加载约定、根粒子数和必要的根相对选择。结果不持有整份 sim 的
强引用。计算中捕获 DAG 自己的变换，不把临时变换后的坐标写成根输入。
运行后修改 calculator 参数、对象标签或当前工作目录，不能改变该次结果的 key。
save 序列化调用时 Result 中的值，与既有存储语义一致；用户修改 Result.value
不会自动重算，因此不能将修改后的值宣称为未修改的原始计算输出。

| 输入 | 自动规则 |
| --- | --- |
| 常规本地根快照 | 规范化绝对路径身份，内建 pynbody loader；多文件前缀也允许 |
| 已登记逻辑快照 | 使用显式逻辑身份、revision、loader，不退回路径身份 |
| 根相对粒子视图 | 根身份 + 既有选择摘要，保存可重放选择 |
| 无文件内存对象或外部活动变换 | 可以 manual 计算，自动持久化/显式 save 明确失败 |
| 不能可靠识别加载约定的对象 | 不宣称可重载；给出缺少的 loader 信息 |

进阶登记入口为 `db.register(sim, simulation='TNG50-1', snapshot='099',
revision='r1', loader='default') -> Snapshot`；项目命名空间取 Database 配置，默认
`default`。它显式声明逻辑等价性，登记完成后通过返回的 Snapshot 或 db.use 上下文
计算。此操作不改写早先路径身份的结果，也不迁移或合并缓存。
同一路径映射到多个 revision 时，只能通过选定的 Snapshot 运行；原始 sim 输入
不猜哪个 revision 是当前版本。

默认不哈希全部粒子数据。原地改数组不会自动改变身份；用 refresh 更新当前结果，
需要区分输入版本时用 revision。refresh 不等于粒子数据归档，也不能恢复旧文件。
路径移动在默认模式下产生新身份；跨位置复用使用显式逻辑登记。
若一个 Result 没有运行时捕获的持久身份，save 不把 session id 强行升级为稳定身份。

## 5. Halo、Subhalo 与粒子加载

```python
snap = db.snapshots['snapshot_099']
halo = snap.halos[0]
halo.results[cc]
particles = halo.load()
res = cc.run(halo, mode='auto')

# 多个目录时显式选目录
halo = snap.catalogues['subfind'].halos[0]
subhalo = snap.catalogues['subfind'].subhalos[12]
```

方括号内是原 catalogue ID，不是行号；保留 typed ID 的区别。只有唯一目录或
已设定默认目录才允许 snap.halos；否则报歧义，不任选一个。
`db.set_default_catalogue(snap, 'subfind')` 显式设置并持久保存默认目录。
物理 host/subhalo 关系与粒子视图的根归属分开保存。

保存真实 Halo 的结果时，自动登记该对象及根；能可靠识别 catalogue 时记录完整
目录命名空间，包括 finder/revision/loader。不能仅凭 `_halo_number` 猜目录。
未知目录的可重放视图登记到 `snap.selections[key]`，不伪造 catalogue 或 halo 归属。
未知目录信息会作为可见登记提示返回，不丢失可保存的数值及选择。
通过 `db.register_catalogue(snap, catalogue, name='subfind', revision='v1')`
显式登记受支持的 pynbody catalogue，可在不读取所有对象粒子的前提下导入对象元数据。
目录适配器不支持时明确报错；初版不承诺适配所有第三方 finder。

`.load()` 必须有可执行的重放描述：受支持 catalogue 的定位，或已复制的根相对
particle indices。不能从 index digest 重建粒子。重放沿用 Selection 的顺序、
重复粒子及 family-block 限制，不静默排序。
第一次成功 materialize 的 catalogue 对象固定其输入身份与选择；目录版本改变
必须产生不同 catalogue 身份，不能让同一个缓存地址指向新粒子集合。
只有元数据、尚无精确输入身份的对象第一次运行必须加载，不能声称免 I/O 命中。
加载后验证根身份、预期数量、选择摘要；同粒子数但内容变化仍需用户管理 revision。

常规文件走内建 `pynbody.load`。进阶 loader 通过
`db.bind_loader('custom', callable)` 绑定项目代码，callable 接收位置并返回根快照；
参数在项目代码中预绑定。数据库重开后重新绑定。数据库不序列化 callable/凭据。
普通 `pynbody.load(..., custom_option=...)` 的调用历史无法可靠倒推，因此不能承诺
自动恢复任意加载参数；内建加载约定不适用时要求显式登记 loader。
多位置 `.load(location_id=...)` 必须选定；失效时不自动换副本。

## 6. 免加载缓存与失败行为

数据库对象先解析精确身份与 signature，再查当前版本。命中时不调用 snapshot
loader、halo finder 或 selector；甚至粒子文件暂时离线也能返回已保存结果。
它证明的是登记身份对应的结果存在，不证明磁盘数据未变。
未命中或 refresh 时才加载并验证，再交给唯一的 EvalEngine 执行。
raw SimSnap 已由用户加载，不承诺替用户省去这部分 I/O。

完整数值、单位、named outputs 和固定版本引用通过同一个 ValueCodec/ResultStore
保存；损坏记录、解码失败、signature 不匹配不能被当成 miss 后悄悄重算。
加载失败、计算失败、编码失败均不能更新当前成功结果或制造成功关联。
auto/refresh 每次调用都有独立 Run，包括命中；manual 纯计算不写 Run。
manual 结果后来 save 只记录保存操作和已捕获来源，不伪造过去并未记录的 Run。
auto 命中记录本次调用环境，不能把它当作历史结果的生产环境。

单一 SQLite 文件不意味着所有现有 store/repository 事务已经原子化。
初版保留先写载荷、后关联元数据的恢复模型；关联失败抛包含固定 receipt 的异常，
并提供 `db.recover(ref)` 验证输入、流程、保存操作后完成缺失关联。恢复不重算、不产生
新结果版本、不猜最新 receipt；底层无法验证来源时拒绝恢复并保留诊断。
保存操作在写载荷前登记 recovery 所需描述；失败后不能只留下不可识别的 receipt。

## 7. 实现职责与明确排除项

新增 `pynbodyext/database/` 负责用户入口、对象查询、名字、loader 绑定及运行协调。
在现有 ORM 中扩展路径输入与浏览关系，继续复用现有结果表；不为每个快照创建表，
不建立第二套结果存储、Calculator DAG 或执行器。
core 增加无 SQLAlchemy 依赖的运行/保存协议和上下文；Database 实现协议，core
不反向 import ORM/Database。顶层 pe.Database 延迟导出。

已有逻辑身份、不可变版本、codec、选择重放、运行恢复保持唯一实现归属。
新旧入口可以共享同一个结果身份；旧明确 store/reuse 行为不因环境配置变化。
现有测试保持完整，改变高层默认身份的测试只有在 ADR 接受后按真实需求变更。

本轮不加入远端下载、粒子文件归档、自动内容指纹、自动科学身份推断、分布式调度、
任意 SQL 后端认证、旧数据库迁移、自动删除历史版本。并发首次 miss 可以重复计算，
但不得串库、串 key 或损坏版本；不承诺全局 exactly-once 执行。

## 8. 分阶段交付与验收

| 阶段 | 用户可见交付 | 必须证明的验收场景 |
| --- | --- | --- |
| A | 根快照 run/save/reopen/results/load | 真文件加载、单位往返；无需手建 schema；延迟保存身份不随活对象变化；同名路径不串结果 |
| B | manual/auto/refresh 与上下文 | 命中计数为零；refresh 新版本且旧引用可读；嵌套/异常恢复；跨库归属拒绝 |
| C | 数据库对象直接 run | 命中时 loader/selector 调用次数严格为零；文件离线仍读结果；miss/refresh 才加载 |
| D | halo/subhalo/selection 统一浏览和加载 | catalogue 隔离；原 ID；重开后粒子顺序/重复/单位一致；缺选择不伪装可加载 |
| E | 逻辑登记、自定义 loader、错误恢复及教程 | 多 revision 不猜测；缺绑定仍读旧结果；载荷后关联失败可恢复；完整入门脚本可执行 |

每阶段先创建双语 work item，写公开入口验收测试并记录预期 red，再实施最小改动。
详细实施计划在本设计评审后编写；这里是交付顺序，不是已经执行的计划。
代码验证需要 make check；公共行为和数据路径变化还需 make test-all。
保留低层身份、版本、catalogue、selection、安装分层测试。免 I/O 要测真实调用计数，
不能只检查 from_store。文档示例最终必须使用真实 pynbody 写入/加载的临时快照，
不能只通过修改 `_filename` 假装有可重载文件。

## 9. 本次交接

新增设计双语对照、ADR-020 草案和设计入口链接。未改包代码、测试、现有 work item
状态或现有入门示例；未提交、推送或新建 PR。上一轮文档示例的未提交修改原样保留。
