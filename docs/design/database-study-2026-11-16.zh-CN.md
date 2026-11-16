# 数据库研究与独立结果表设计

[English](database-study-2026-11-16.md) · [ADR-021](../adr/0021-database-studies-and-detached-tables.md) · [SPIKE-0002](../work/SPIKE-0002/zh-CN.md)

日期：2026-11-16。**已批准**。实施与验证见 [FEAT-0018](../work/FEAT-0018/zh-CN.md)。
设计评审基线：`d862742`。已实现接口的用法见[数据库指南](../database.zh-CN.md)。

本设计覆盖两个工作流：单对象计算、保存与重开；固定样本计算、重试与导出。两者使用同一结果存储地址和执行职责。Study 保存研究请求，StudyTable 保存一次观察或执行的结果。读取两者都不会启动计算。

## 1. 评审发现与取舍

以下是对前一轮使用提案的评审，并结合当前源码与测试核实；不表示目标行为已经实现。

| ID | 不一致或不必要的复杂性 | 设计决定 |
| --- | --- | --- |
| R01 | 直接包装 ORM `run_study` 仍会在命中时加载粒子，而且不接受粒子选择。`run_input` 明确先加载再执行。 | 批量与单对象共用 Database 的先查缓存、后加载协调逻辑；EvalEngine 和 ResultStore 继续负责原有职责。不能直接包装旧的根输入执行器。 |
| R02 | 按 Study 整体建缓存 key，或自动把所有列组合成 Pipeline，都会让新增或重命名列破坏复用。 | 每格按现有精确输入 × 完整 CalculatorSignature 存储。Study 名称和表格布局不进入计算 key。 |
| R03 | `SampleTable.compute/add/fill_from_store` 会原地修改值，缺失哨兵还与成功的 None 重叠。称它为不可变报告会改变已有合同。 | 新增一个脱离数据库、只读的 **StudyTable**，供高层 run 和 collect 共用。保留旧 SampleTable 和 ORM StudyReport 接口，适当复用编码与展示工具。 |
| R04 | 刷新失败后重新 collect，可能读到旧成功值，掩盖本次失败。 | run 表记录本次执行，collect 表记录选中的当前缓存版本。失败单元格绝不借旧值补成成功。 |
| R05 | 已经开始、但无法确认完成的调用不能标为 not_run。 | 结果表增加 incomplete；恢复回执与成功结果引用分开存放。 |
| R06 | 保存 signature 不保证可重建；活 Calculator 的运行选项和 loader 函数不会自动跨进程保存。 | 签名按数据保存，只有需要计算时才重建；允许显式绑定签名完全匹配的活 Calculator。加载器保持进程内绑定。 |
| R07 | 路径或 `store_name="default"` 无法证明导出对象、结果引用属于哪个数据库。 | 增加持久 Database UUID 命名空间，保留类型化精确输入 key 和已有固定 RecordRef。路径只是位置提示。 |
| R08 | 为了扩展保存过的 Study，如果读取 `study.calculations` 就重建 Calculator，会使普通查看依赖项目代码。 | 返回只读的列名 → CalculatorSignature 映射；工厂也接受签名。查看时不重建。 |
| R09 | frozen dataclass 仍可暴露可修改数组或字典，使表中数值偏离固定来源。 | 结果表通过只读存储或防御性复制保护值，不提供隐式刷新或原地编辑接口。 |
| R10 | `errors="record"`、`errors="collect"`、`on_error` 对相似高层意图使用不同拼写。 | 新 Study 只接受 errors="raise" 或 "collect"。现有低层拼写保留，不另外增加同义入口。 |
| R11 | not_applicable 尚无明确科学结果协议。仅凭路径登记、自动共享粒子加载、审核表后提交缓存也会显著扩大范围。 | 暂缓这些能力。成功 None 是有效结果，科学异常是失败，不猜测适用性。第一版使用已登记的精确输入和串行隔离执行。 |
| R12 | 无限制使用 ValueCodec 读取表文件可能反序列化 opaque pickle，与“不需要项目代码”的重开承诺冲突。 | 新表文件只接受可移植、非 opaque 编码，在解码前拒绝 pickle/opaque 标签。旧格式保持不变。 |

源码依据：[Database 执行](../../pynbodyext/database/database.py)、[对象集合](../../pynbodyext/database/objects.py)、[ORM 执行](../../pynbodyext/orm/repository.py)、[ORM studies](../../pynbodyext/orm/study.py)、[SampleTable](../../pynbodyext/core/calculate/sample.py)、[ValueCodec](../../pynbodyext/core/calculate/store/codec.py)。相关已有测试：[database](../../tests/test_database.py)、[catalogues](../../tests/test_database_catalogues.py)、[studies](../../tests/test_orm_study.py)、[sample references](../../tests/test_sample_refs.py)。

## 2. 用户概念与范围

| 对象 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| Database | 登记、持久研究定义、输入解析、执行协调和结果访问 | 科学计算或第二套 Calculator DAG |
| 输入句柄 | 已登记的精确根输入或视图，可以带目录来源 | 粒子数组本身、自动发现的演化历史 |
| Calculator / Result | 科学定义 / 一次计算的值与来源 | 样本成员 |
| Study | 有序固定对象 × 有序具名签名 | 最新数值、动态查询成员或后台作业 |
| StudyTable | 独立的数值、状态、实际版本引用与调用信息 | 数据库连接、可执行 Calculator 或恢复执行操作 |

`db.study(...)` 从所属数据库已登记的句柄或独立引用，在内存中构造请求；不接受原始粒子快照，也不隐式登记新数据。`study.save(name=...)` 保存定义，`db.studies[name]` 重开定义。run 和 collect 都返回新的 StudyTable。保存表文件不把数值写入计算缓存。本轮不引入额外的 Sample、Workspace 或调度器。

目标 API：

```python
db = pe.Database("analysis.sqlite", read_only=False)
optional = obj.results.get(cc)               # 仅结果缺失时返回 None
study = db.study(objects=objects, calculations={"mass": cc})
definition_id = study.save(name="sample-v1")
study = db.studies["sample-v1"]
plan = study.plan(mode="auto")
table = study.collect(missing="raise")       # 或 "keep"
table = study.run(mode="auto", errors="raise", cells=None)
study = study.bind_calculators({"mass": live_cc})  # 可选的精确绑定
table.save("sample-v1.json")
table = pe.StudyTable.load("sample-v1.json")
```

`pe.StudyTable` 是拟新增的惰性根导出，由不依赖 ORM 的计算层数据类型实现。`sample-v1` 是用户命名的固定定义，不是会移动的 latest 别名。同名保存相同内容幂等；同名保存不同内容报错；不同名称可以指向同一定义。Study 定义 ID 是内容摘要，不是新的计算缓存 key。

## 3. 身份与单对象、批量结果共享

唯一计算地址继续使用：

```text
K = (sim_identity_key(精确输入 signature), 完整 CalculatorSignature hash)
保存值 = (所属 Database UUID, 固定 RecordRef(K, version_id))
```

缓存命中仍校验完整签名载荷。显示名、列名、Study 定义 ID、目录浏览路径都不能代替 K。逻辑数据 revision、loader 身份和粒子选择沿用现有身份规则；不自动对文件内容或内存数组修改做指纹检测。

独立对象引用包含 Database UUID、精确类型化输入签名、规范对象定位信息和固定显示名。目录来源由定位信息保留，但计算 key 仍是精确输入身份。SQL 自增主键和文件 basename 都不能单独成为可移植引用。工厂接受本库句柄或这些独立引用，并校验归属，不按名称偷偷绑定外库对象。

第一版要求每行精确输入 key 不重复。同一输入被选择两次，包括来自两个目录但解析为同一精确输入的句柄，均明确报重复，不静默合并。重复抽样和跨目录对照表不在本版范围。列名是非空唯一标签；允许多个标签使用同一签名。串行 auto 下，后一个相同计算可以复用前一个保存值；refresh 则按先行后列顺序为每个请求单元格生成新版本。不承诺全局恰好计算一次。

工厂复制并固定签名 JSON、输入引用和轴顺序。定义摘要包含 schema、所属 Database UUID、固定对象描述、有序列名及完整签名载荷，不包含保存名、时间戳或活文件位置提示。`study.objects` 是只读引用元组；`study.calculations` 是只读 CalculatorSignature 映射。工厂的列值接受 Calculator 或稳定 CalculatorSignature。保存定义不依赖活对象引用。

活 Calculator 在请求开始前及对应格执行前，必须仍与固定签名匹配。修改参数后报错，不能悄悄改变已保存定义；新的科学定义应创建新的 Study。诊断用运行选项不作为科学身份序列化；会改变科学含义的配置，包括相关随机性，必须进入 Calculator signature 或输入身份。mode/errors 按每次调用解析并单独记录。

无并发写入时，单对象保存 → 批量 auto、批量保存 → 单对象 auto、两个不同名称的 Study，对相同 K 应返回相同固定引用。发生并发刷新时，每个返回值必须对应自己的固定引用；两次独立的“当前缓存”读取不保证选中同一版本。

## 4. 读取、预览与执行

### 4.1 只读路径

`results[cc]`、`results.get(cc)`、`fetch(ref)`、查看 Study、plan 和 collect 都不能写数据库。仅解析目录元数据时，不能调用会登记粒子选择的物化函数。受支持的已保存归属描述，可以不加载粒子而导出精确 key。若无法从已登记元数据确定精确输入，构造 Study 前明确拒绝；后续适配器可扩展覆盖。

get 仅在对应结果不存在时返回 None。歧义、非法签名、损坏载荷和解码失败继续抛错。成功返回 None 的 Result 仍然是 Result。

`plan(mode=...)` 读取有限的元数据，不读取科学载荷。每格动作是 reuse、compute 或 blocked；原因区分缺少 Calculator/loader 绑定与已知元数据错误。没有执行 IO，就不能声称文件可用、内存足够或计算一定成功。计划只是预览；run 重新验证，不预留缓存版本或计算资源。

collect 先在本地数据库的一个读快照内固定所请求网格的当前版本引用，然后读取这些精确版本。不创建 Run，不重建 Calculator。missing="raise" 报告缺失坐标；"keep" 保留 missing 单元格。其他错误始终抛出，不能用 missing 策略容忍损坏。之后发生的并发刷新不会替换本次选中的引用。快照保证只覆盖本库元数据，不覆盖外部文件或粒子数据。

### 4.2 执行与策略

run 的 mode 优先级为：显式参数、同一数据库的适用 db.use 上下文、默认 auto。另一个数据库的上下文不能改变 Study 的归属。errors 默认 raise。已有单个 Calculator 的策略保留；新 Study 不把 collect_partial 当成整个单元格成功的策略。

预检只读；任何登记或 Run 创建前，验证请求轴、归属、选择器、签名完整性与权限。必要的物化元数据只在通过请求验证后的实际单元格执行阶段写入。auto 先判断缓存命中，再要求可执行 Calculator、loader 和输入位置。所选缺失格，或 manual/refresh 所选格需要的计算绑定缺失，是任何格开始前的配置错误；全命中研究不需要这些绑定。执行开始后，实际文件读取和科学计算异常才是单元格失败。如果预检后缓存状态变化，重新验证相关格；出错时提供部分进度，不隐式改变策略。

`cc.run(obj)` 与每个 Study 单元格走同一 Database 协调入口：精确查找 → 按需加载 → EvalEngine → ResultStore → Run 关联。ORM run_study 保留为底层兼容接口；其只接受根输入、先加载的预检不能原样承担新合同。第一版按行、列顺序串行执行。不自动融合 Pipeline、不自动共享活粒子快照缓存、不引入并行后端，也不承诺跨列 DAG 复用。有状态或不可逆变换不得污染后续格。

| 模式 | 读取结果缓存 | 计算 | 保存成功格 | 持久记录 Run |
| --- | --- | --- | --- | --- |
| manual | 否 | 所有所选格 | 否 | 否 |
| auto | 是 | 缺失格 | 是 | 是，包括缓存命中 |
| refresh | 否 | 所有所选格 | 是，产生新版本 | 是 |

只读数据库允许读取，以及对已登记输入做 manual 计算；auto/refresh 在任何所选格开始前拒绝，即使预计全部命中。bind_loader 和 bind_calculators 只改变进程内绑定，允许使用。manual 可能读取粒子，其表不能伪造 Run ID 或保存引用。

## 5. 状态、旧值与恢复

StudyTable 记录 origin="run" 或 "collect"、有效调用策略（run）、捕获时间、完整固定来源定义、显示的轴及单元格状态。过滤后的表保留来源定义 ID 和原坐标，只缩小显示行；不能悄悄变成另一次执行的正式报告。

| 状态 | 有结果值 | 成功结果引用 | 错误或原因 | 含义 |
| --- | --- | --- | --- | --- |
| succeeded | 有，可以是 None、NaN 或空值 | auto/refresh/collect 必须有；manual 无 | 无执行错误 | 已确认的结果 |
| missing | 无 | None | 缓存缺失 | 仅 collect |
| failed | 无公开成功值 | None | 必须有结构化诊断 | 本次尝试失败，旧缓存可能仍在 |
| not_run | 无 | None | 可记录未选择或停止原因 | 本次没有尝试该格 |
| incomplete | 无公开成功值 | None | 必须有中断或持久化原因 | 已开始，但完成尚未确认 |

序列化必须有显式的“有无值”标志，不能只用 value=None 表示缺失。collect 与复用成功格的 from_store 为 True，新算成功格为 False，其他格为 None；run_id 只能指向真实持久调用。失败计算的部分诊断可以另存，但不能成为成功数值。not_applicable 暂缓，等待明确科学结果协议；不能从空值或异常猜测。

设旧版本为 V1，一次 refresh 失败：run 表记录 failed/ref=None；V1 仍是当前缓存；之后 collect 可以得到 succeeded/V1。二者不冒充彼此。之后恢复确认 V2，也不会改变以前返回的 incomplete 表，必须另作观察。失败格的 ref 不指向 V1，也不指向未确认保存回执。

errors="collect" 只在可靠记录了普通单元格执行失败后继续；manual 则保留内存失败信息。持久化未完成、配置非法、甚至无法记录失败时，研究立即停止。raise 策略的 StudyExecutionError 携带 report、失败坐标和原异常 cause。StudyPersistenceError 携带 report、中断坐标，以及已有的精确 receipt/result_ref。格开始前元数据写入失败时，该格为 not_run；已开始但未确认完成时为 incomplete。不通过整批回滚删除以前成功的结果。

可捕获的用户中断仍作为中断传播，并尽力附上部分报告；不能当成普通失败后继续。强制结束进程无法承诺最后一份报告，既有 Run 可能停留在 running。重新开始是新的显式调用，复用已经确认的缓存；不承诺恰好执行一次或恢复原进程。未完成持久化的回执必须显式检查和恢复，重试不负责自动消除歧义。

`table.cells(status="failed")` 返回精确选择器，包含来源定义 ID、数据库命名空间、输入 key、列名与签名 hash。执行前验证整个选择器集合；不同定义或外库表的选择器拒绝。不把选择器展开为失败行集合 × 失败列集合。空选择不执行，返回 not_run 网格。auto 重试可能复用别处产生的有效缓存；要重试失败的刷新，显式使用 refresh。重试表中未选择的格保持 not_run，不隐式拼接旧报告的成功格。

## 6. 独立重开与归属

| 重启 Python 后的操作 | 所属数据库 | 粒子文件 | Calculator 实现 | loader 绑定 |
| --- | --- | --- | --- | --- |
| db.studies[name]、查看定义 | 需要 | 不需要 | 不需要 | 不需要 |
| study.plan() | 需要 | 不需要 | 不重建，可检查绑定可用性 | 不调用 |
| study.collect() | 需要，含所选载荷 | 不需要 | 不需要 | 不需要 |
| 全命中的 study.run(auto) | 需要且可写 | 不需要 | 不需要 | 不需要 |
| 需要计算的 study.run | 需要 | 按输入要求 | 相应格需要 | 相应格需要 |
| StudyTable.load(path)、查看、过滤、导出 | 不需要 | 不需要 | 不需要项目 Calculator 代码 | 不需要 |

Study 独立于旧 Python 会话和结果表文件，但仍依赖所属数据库。StudyTable 的内嵌可移植数值独立于 Study 登记和数据库本身。不需要 Calculator 的缓存读取针对受支持的可移植载荷；旧 opaque 结果仍可能需要其值类型的实现，本设计不赋予它新的可移植保证。两者都不是粒子归档。ExternalFileRef 仍然只是描述，读取外部文件需要显式操作。

保存的定义保留完整签名，不只存 hash 或 Python 对象。study.bind_calculators({label: cc}) 返回同一定义的一个新活绑定，校验完整签名，不修改已存定义。若可在需要时重建标准类，绑定可省略。稳定但不可重建的 Calculator 即使没有活绑定，也应支持 collect 和全命中 auto。仅打开定义或表，不能任意导入项目实现。

启用本功能的可写初始化在新增元数据中创建 Database UUID。只读打开不创建 UUID、不升级 schema，也不创建缺失文件。旧数据库可以保留原来的读取能力；缺少 Study 元数据时请求新功能，明确报告兼容性错误。已有缓存地址与载荷不重写。逐字节备份保留命名空间；复制后可能分叉，因此 UUID 相同也不能证明某个固定版本存在，访问时仍验证精确引用。自动合并或创建分叉数据库不在范围内。

表文件引用携带数据库命名空间和可选位置提示。加载不会沿提示打开数据库，也不会扫描寻找数据库。把 table.objects 传给 db.study 是显式重新绑定请求，命名空间、精确身份和定位信息必须一致。修改文件名不能把外库变成本库；精确版本缺失时明确报错，不退回当前缓存。

## 7. StudyTable 保存格式与数值访问

拟用 schema：定义为 pynbodyext.database.study/v1，独立表为 pynbodyext.study.table/v1。不是 SampleTable v3，也不自动升级 ORM StudyReport JSON。

定义字段包含 schema、数据库 UUID、有序对象描述、有序具名完整签名；登记元数据另存内容摘要、名称绑定和创建时间。表字段包含 schema、完整来源定义及摘要、origin/捕获信息、有效策略、显示轴和单元格。每格包含坐标、状态、有无值标志、有值时的可移植编码、已确认时的固定引用、真实 Run ID、from_store 和诊断。恢复信息与成功引用分开。表文件必须足以离线验证坐标和签名。显示轴的每个行列坐标必须恰有一个明确单元格，包括 missing/not_run/incomplete，不允许用省略表达状态。过滤表另行保留完整来源定义和显示子集。

接受文件前验证 schema、重复 JSON 字段、类型化身份、完整签名 hash 与载荷、定义摘要、重复或未知坐标、状态/值/引用关系、固定版本与归属。未知版本明确拒绝。新格式拒绝 opaque/pickle 标签，包括嵌套标签；使用已有可移植 ValueCodec 标签处理受支持的 NumPy、SimArray、标量、容器和 ExternalFileRef 描述。成功 None 可往返。不改变旧编码器策略。

save(path, overwrite=False) 先完成验证和编码，再通过临时文件在同一文件系统原子发布。已有文件默认拒绝，必须显式允许覆盖；验证或编码失败不能损坏原文件。不承诺抵御任意硬件故障。不可移植值导致明确导出错误，数据库里的原 Result 仍可使用；以后导出可移植投影属于另一项显式操作。

表持有独立数值快照；返回数组需要保护或复制，受支持的可变嵌套容器需防御性复制。查看、换单位、过滤、导出不能改动保存的值或引用。StudyTable 不提供 compute、fill_from_store、自动绑定数据库或通用原地编辑。

cell(obj_or_ref, label) 和 row.cell(label) 返回单元格视图；row[label] 只有在 succeeded 时返回受保护值，否则报单元格取值错误。where(predicate) 只过滤当前快照，保留顺序和来源，不加载粒子。column(label, units=...) 提供数值标量列，对非成功状态返回掩码，不把有效零或 NaN 当缺失；成功 None、非数值或非标量要求逐格访问，不能隐式强制成数值。量纲不兼容时拒绝；兼容但单位不同的列要求显式目标单位，不猜第一行的单位。

to_dataframe(units=None, include_status=True, include_refs=True) 是派生视图，明确单位转换选择并默认携带状态/引用列，不修改表。CSV 是交换投影，不是无损归档。第一版采用完整内嵌可移植数值的 JSON；仅引用或延迟读取大载荷的表另作设计。

## 8. 两条目标代码故事

### 8.1 单个星系、复用与历史

```python
import pynbody
import pynbodyext as pe
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

stellar_mass = ParamSum("mass").filter(FamilyFilter("star"))
sim = pynbody.load("/data/galaxy_099")
sim.physical_units()
result = stellar_mass.run(sim)
with pe.Database("analysis.sqlite") as db:
    first_ref = result.save(db, name="stellar_mass")
    obj = db.snapshots["galaxy_099"]
    key = obj.key
    assert stellar_mass.run(obj).store_ref == first_ref
    fresh = stellar_mass.run(obj, mode="refresh")
    assert fresh.store_ref != first_ref
    old = db.fetch(first_ref)

with pe.Database("analysis.sqlite", read_only=True) as db:
    obj = db.snapshots[key]
    saved = obj.results[stellar_mass]  # 不需要原粒子文件在线
    optional = obj.results.get(ParamSum("mass"))
```

上例变量跨越 Notebook 中的关闭与重开。新进程应保存精确对象 key，并使用已有 RecordRef JSON 连同数据库命名空间保存固定引用。同名短名称报歧义，不取第一行。逻辑身份沿用当前加载根对象的 db.register(sim, ...) 并在计算前登记；上一轮草案的仅路径登记暂缓。

### 8.2 固定样本、重试、重开与论文数据

假设已登记可重放的目录对象 7、12、23，目录创建沿用[现有指南](../database.zh-CN.md)。创建 Study 不扫描或加载目录。

```python
with pe.Database("sample.sqlite") as db:
    cat = db.snapshots["snapshot_099"].catalogues["groups"]
    objects = [cat.halos[i] for i in (7, 12, 23)]
    study = db.study(objects=objects, calculations={
        "mass": ParamSum("mass"), "stellar_mass": stellar_mass,
    })
    study.save(name="galaxies-v1")
    print(study.plan())
    before = study.collect(missing="keep")
    table = study.run(errors="collect")
    table.save("attempt-1.json")

# 新进程继续研究定义，不需要以前的结果表。
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    current_attempt = study.run(mode="auto", errors="collect")
    current_attempt.save("attempt-2.json")

# 另一个进程查看结果表，不需要数据库或 Calculator 实现。
frozen = pe.StudyTable.load("attempt-1.json")
for cell in frozen.cells(status="failed"):
    print(cell)  # 精确选择器，不是粒子对象

# 修复临时执行问题后，只重试所选精确单元格。
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    retried = study.run(cells=frozen.cells(status="failed"), errors="collect")
    retried.save("retry-1.json")
    publication = study.collect(missing="keep")
    print(publication)  # 确定论文样本前检查缺失状态
    publication.save("paper-sample-v1.json")
```

重试旧的刷新失败时使用 mode="refresh"。保存报告不表示未完成持久化已恢复。严格要求当前缓存完整时可用 collect(missing="raise")，但完整缓存不证明最近一次刷新成功。

```python
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    publication = pe.StudyTable.load("paper-sample-v1.json")
    # 扩展定义，不需要重建已有 Calculator。
    expanded = db.study(objects=study.objects, calculations={
        **study.calculations,
        "gas_mass": ParamSum("mass").filter(FamilyFilter("gas")),
    })
    expanded.save(name="galaxies-v2")
    expanded_table = expanded.run()

    # 明确只选择成功测量且质量达到阈值的对象。
    selected = publication.where(
        lambda row: row.cell("stellar_mass").status == "succeeded"
        and row["stellar_mass"].in_units("Msol") > 1e10
    )
    next_study = db.study(objects=selected.objects, calculations={"mass": ParamSum("mass")})
```

表列名不等于对象结果别名：用 objects[0].results[stellar_mass]，不能假定同名字符串已经登记。无并发写入时，其引用与相应成功 auto/collect 单元格相同。

## 9. 分层、备选方案与兼容性

推荐的 Module seam 是 ORM 上方的 Database 协调层，科学执行委托 EvalEngine，持久化委托 ResultStore。Study 定义与解析位于上层；独立 StudyTable/Cell/ObjectRef 数据及编码验证位于计算层，不导入 SQLAlchemy 或 Database。根 StudyTable 导出必须惰性，并纳入导入分层测试。ORM 报告兼容可使用显式内部转换，低层表不能自动寻找上层仓库。

不采用的路线：让 cc.run(list) 随输入数量改变返回类型；把可变 SampleTable 方法改成不可变操作；另建有独立缓存策略的批量执行器；隐式 Pipeline 融合；为所有时间演化和多输入场景先建一个通用框架。独立表虽然增加一个公开类型，却保留已有构建器语义，并使新的两个结果入口保持一致。

ADR-021 增加只读打开、results.get、Study 定义/绑定/协调、独立报告及数据库命名空间元数据。保留既有 ResultStore key、不可变结果版本、Result.save 语义、已接受的低层接口与测试。旧 SampleTable v1/v2、ORM study/v1 文件继续由原加载器处理，不静默转换，不伪造运行或引用历史。

## 10. 验收场景

场景编号对应 FEAT-0018 中记录的测试覆盖。

| ID | 场景 | 必须观察到的行为 |
| --- | --- | --- |
| A01 | 单对象保存后批量 auto | 相同固定引用，命中时零粒子加载 |
| A02 | 批量 auto 后单对象 auto | 相同 K/ref，命中不重复写载荷 |
| A03 | 改 Study/列名或增加列 | 原计算复用，仅缺失计算执行 |
| A04 | 改参数、签名版本、输入 revision 或选择 | K 不同，不错误命中 |
| A05 | 两个目录来源指向同一精确输入 | 写入前明确拒绝重复，不默默合并行 |
| A06 | 两列使用同一签名 | auto 顺序复用，refresh 顺序产生版本，各格引用准确 |
| A07 | 对未物化的受支持目录句柄只读 plan/collect | 不登记选择，plan 不解码载荷，不读粒子或创建 Run |
| A08 | 粒子离线或 loader 未绑定但缓存全命中 | collect 与可写 auto 成功，不加载 |
| A09 | 缺少可执行类，全命中与部分缺失分别测试 | 查看/collect/命中成功，必要的缺失绑定在预检失败 |
| A10 | 绑定错误签名或修改活 Calculator | 拒绝，不执行不匹配定义 |
| A11 | 刷新失败且旧缓存成功 | run 格 failed/ref=None，collect 读旧引用，旧表不变 |
| A12 | 成功 None、零、NaN、空数组 | 文件往返仍成功，与 missing 分清 |
| A13 | collect 错误策略下普通执行失败 | 后续格继续，失败格无旧引用 |
| A14 | raise 错误策略 | 部分报告保留失败及后续 not_run 格 |
| A15 | 格开始前、开始后发生持久化失败 | not_run/incomplete 分清，回执分离，停止后续格 |
| A16 | 部分报告保存后执行精确恢复 | 不重复写载荷，不修改旧报告 |
| A17 | 重试两个对角位置的失败格 | 只针对两个格，不展开为笛卡尔积 |
| A18 | 从变更定义或外库表取得重试选择器 | 整个请求在执行前拒绝 |
| A19 | 不带旧表，关闭并重开保存的 Study | 类型化有序轴和签名一致，打开时不重建 Calculator |
| A20 | 数据库和项目 Calculator 包均不可用时加载表 | 内嵌值、单位、状态、引用可独立使用，无后端访问 |
| A21 | 外库恰有同名对象和相同 SQL ID | 拒绝重新绑定，不按名称替代 |
| A22 | 搬迁备份 UUID 一致但固定版本缺失 | 明确报版本缺失，不回退最新 |
| A23 | collect 期间并发刷新 | 数值对应最初固定的引用 |
| A24 | 修改从表取得的值 | 原表和来源不受影响，或写入被拒绝 |
| A25 | 未知 schema、重复字段、损坏引用、嵌套 opaque | 在不安全解码前拒绝，不产生 IO/执行副作用 |
| A26 | 导出编码失败或目标已存在 | 原文件不变，覆盖必须显式 |
| A27 | 只读打开、文件缺失、旧 schema；全命中 auto | 不创建/升级，明确报告权限或能力错误 |
| A28 | manual、空轴、空选择 | 不写 Run/载荷，未持久成功/not_run 状态真实 |
| A29 | 带单位和失败状态的过滤/换单位/导出 | 顺序和引用保留，掩码明确，量纲或单位歧义拒绝 |
| A30 | 分层与导入兼容 | 独立表不依赖 SQLAlchemy/Database，旧测试保留 |
| A31 | 同定义同名重复保存，再保存冲突内容 | 相同绑定幂等，冲突不覆盖 |
| A32 | KeyboardInterrupt 与进程强制退出 | 不吞中断继续，不承诺最终报告或恰好一次执行 |

## 11. 明确暂缓的能力

仅路径登记、动态查询样本、重复行或 bootstrap 研究、一个 Study 混合多个数据库、分布式调度、自动共享粒子加载、审核表后批量提交缓存、多输入/时间树计算、科学 not-applicable 结果、任意 opaque 对象可移植化、仅引用的延迟结果表，都另行设计；它们不是本提案隐含但未完成的前置需求。

三个评审问题在合同层明确：单对象与批量共享 K 和执行协调；本次状态不借旧值伪装成功；Study 从所属数据库重开，StudyTable 从自己的文件重开。
