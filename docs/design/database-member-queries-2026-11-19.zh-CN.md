# 粒子成员导航与计算结果查询设计评审

设计草案，2026-11-19。源码基线为合并 ADR-021 的 `c56120f`。[English](database-member-queries-2026-11-19.md)

从一个星系出发，分析者需要找到其中的恒星和孔径选集。从论文中的一个数值出发，则需要找回当时使用的粒子、坐标系与计算定义。建议用 `obj.members[选择器]` 浏览已登记的成员，用 `obj.results.where(...)` 发现已存输出，用固定结果引用追溯一次具体测量。目录、单次计算和 Study 已有可用接口。本文中的 `members`、`used_by`、结果集合的 `where` 与选集坐标重载均为提议。

## 从 Subhalo 到薄盘内的恒星

假定目录已登记宿主关系，并能给出这个 Subhalo 的精确成员。示例在首次保存薄盘测量后查看其唯一来源。

```python
from pynbodyext.filters import Disc
from pynbodyext.properties import AngMomVec, KappaRot
from pynbodyext.transforms import AlignVec

sub = db.snapshots["snapshot_099"].catalogues["groups"].halos[42].subhalos[7]
stars = sub.members.register("star")
face_on = AlignVec(AngMomVec())
disc = Disc("10 kpc", "1 kpc")
krot = KappaRot().transform(face_on).filter(disc)

result = krot.run(stars)
disk_stars = stars.members[disc]
disk_stars.origin.context
disk_stars.used_by
```

`register` 是显式建立成员的入口，默认在来源对象的快照坐标中求值，同一份已确认记录可复用。普通数据库计算保存成功时，自动关联实际使用的筛选视图，上例只执行一次薄盘筛选。以后重开数据库，用 `sub.members["star"]` 即可继续导航。`"star"` 与 `FamilyFilter("star")` 共用一个选择定义，`"dm"` 同理。自定义 Filter 以持久的完整签名寻址。

`[]` 先在当前对象的直接来源关系中匹配，再按精确粒子输入归并。零个匹配给出缺失和已有的登记信息，多批不同粒子给出歧义及候选。**同一批粒子有多个来源时，仍然可以直接 `.load()`。** 例如多次转动星系仍选出相同恒星，用户只想取粒子时无须选坐标系。`.origins` 列出这些来源，`.origin` 在来源唯一时可用。已确认的空集也有自己的成员句柄。

`members.where(...)` 每个不同的来源返回一项，`.one()` 要求唯一。选定来源后再读取其上下文或重放坐标。未选定来源时，`.used_by` 汇集匹配来源的实际用途，每条都携带自己的来源。继续向下浏览时仅搜索这个句柄所代表的来源分支。显式登记默认基于精确输入的快照坐标，只有选定了上一级来源才写入该来源的嵌套路径。

```python
stars.members.where(filter=Disc)
chosen = stars.members.where(filter=disc, transform=face_on).one()
native_particles = chosen.load()
disk_particles = chosen.load(frame="selection")
```

`.load()` 从对应版本的快照按已保存的有序根索引取粒子，保留重复索引和单位约定。`frame="selection"` 在独立加载的视图上应用筛选当时已记录的变换。首批适合支持有明确目标、单位及参数的旋转、位置平移和速度平移。上下文不完整时报出无法重放的步骤，仍可取原坐标成员。两个入口都需要加载器能打开当时的输入版本。若文件内容被原地替换，仅凭索引和矩阵无法还原先前的粒子数组。

## 从结果反查成员

```python
hits = db.results.where(objects=sub.host.subhalos, transform=face_on)
all_discs = stars.results.where(filter=Disc)
same_aperture = stars.results.where(filter=disc)
same_frame = stars.results.where(context=chosen.origin.context)
```

- `objects` 限定直接计算输入。`sub.host.subhalos` 查询各子晕本身。上例的 `krot` 保存于 `stars`，可从 `stars.results` 找到。`chosen.results` 列以薄盘成员为直接输入的结果。`chosen.used_by` 列使用该来源的固定结果、输出路径和用途。
- `transform=` 查输出配方的整条有序变换链，单个实例代表一步。定心再定轴可传 `(center, face_on)`。`filter=` 查输出的直接筛选表达式，传类型匹配其根类型，传实例匹配完整表达式。不同条件必须命中同一输出。完整作用域可通过 `scope=` 匹配，保留父子作用域、步骤顺序和回退策略。
- `uses=` 查配方内任意用途，返回节点路径与角色。例如 `uses=Sphere` 可发现 `~Sphere(...)`、复合孔径中的球和定轴使用的球。可从命中行看见逻辑运算和用途。`where(filter=Sphere)` 只收直接以球形筛选表达式为根的输出。
- `context=` 查已见证的实际坐标上下文。成员查询比较筛选时的上下文，结果查询比较最终输出测量时的上下文。比较包括根输入版本、作用目标、单位和有序实际操作。相同定轴方法可以给另一颗星系算出不同矩阵。上下文不明的输出仍可按配方发现，并显示记录状态。

结果查询默认返回当前已确认版本的元数据，包含科学对象定位键、完整 Calculator 签名、输出路径、固定引用及上下文可用性。列举时只读元数据。独立结果占一行，Pipeline 的具名叶输出按键路径定位，例如 `("kinematics", "krot")`。同一根结果的不同输出共享版本引用。显式 `output=()` 选取根结果，包括 Pipeline 根。`calculator=` 接受完整 Calculator 或已存签名，可用来限定某一个精确配方。

重画旧图时，从当初的 Study 单元格进入。这里假设 `profiles` 是一次成功保存的 Pipeline，其 `krot` 输出实际用过 `disc`。

```python
from pynbodyext import Database, StudyTable

frozen = StudyTable.load("paper-sample.json")
cell = frozen.cell(frozen.objects[0], "profiles")
with Database("analysis.sqlite", read_only=True) as db:
    hit = db.results.where(ref=cell.ref, output=("krot",)).one()
    prior = hit.members.where(role="output_filter", filter=disc).one()
    value = db.fetch(hit.ref).value["krot"]
    particles = prior.load(frame="selection")
```

`ref=` 锚定所属数据库和确切历史版本。即使当前缓存已刷新，用上述引用也会找到当时的来源。旧结果若只有配方，数值和签名仍可读，成员显示“未记录”。同一输出确有多次满足条件的筛选时，命中行还能按出现的 DAG 节点路径区分。`StudyTable` 文件本身可独立读值，回查来源需要所属数据库。

## 批量比较子晕与宿主

若研究问题是“每个子晕的恒星质量与其宿主总质量有什么关系”，应以子晕为样本行。恒星数量是样本属性，不应决定某个子晕是否进入分母。下例中 `cat` 是已选择的目录，`targets` 是具有精确输入且互异的 Study 子样本。

```python
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

total_mass = ParamSum("mass")
stellar_mass = total_mass.filter(FamilyFilter("star"))
sample = db.study(objects=targets, calculations={"Mstar": stellar_mass})
coverage = sample.collect(missing="keep")

roster = cat.subhalos.to_dataframe()
satellites = db.results.where(objects=cat.subhalos, calculator=stellar_mass, output=()).to_dataframe()
hosts = db.results.where(objects=cat.halos, calculator=total_mass, output=()).to_dataframe()
sat_refs = satellites[["object_locator", "ref"]].rename(columns={"ref": "stellar_mass_ref"})
host_refs = hosts[["object_locator", "ref"]].rename(
    columns={"object_locator": "host_locator", "ref": "host_mass_ref"}
)
pairs = roster.merge(sat_refs, on="object_locator", how="left", validate="one_to_one")
pairs = pairs.merge(host_refs, on="host_locator", how="left", validate="many_to_one")
```

`coverage` 给出固定样本中的已有值与缺测格。两次左连接从完整目录清单出发，子晕即使缺少自身测量、宿主质量或粒子绑定，也保留一行。拟议的目录 `to_dataframe()` 给出科学定位键、宿主定位键和成员资料可用性。结果查询用同一动词提取元数据和固定引用，读取数值时使用对应的 `db.fetch(ref).value`。显式传入多个科学定位键时，每个请求的定位键、配方、输出路径各占一行，即使两行复用同一份数值。恒星质量求和与空样本的 κrot 各自遵从 Calculator 的成功或失败语义。

子晕和宿主的两次查询各自固定读到的版本，中间可能发生刷新。如果图表要求两个群体来自同一缓存时刻，且各对象具有互异的精确输入，可把两类对象、两项 Calculator 放进同一个 Study，一次 `collect(missing="keep")` 固定版本，然后投影所需行列。

## 分析场景与适用性

| 场景 | 合适的入口 | 影响结论的细节 |
|---|---|---|
| 01 单文件孤立星系，只有根快照 | `snap.members["star"]`，`snap.results` | 路径型输入可直接使用。成员沿根快照寻址。 |
| 02 两份目录都有 Halo 42，同一目录登记了宿主和子晕 | `snap.catalogues[key].halos[42].subhalos[7]`，`sub.host` | 宿主来自目录声明。跨表连接采用包含数据库、根、目录版本、种类和类型化编号的定位键。 |
| 03 目录只有对象元数据，或确知一个子晕没有恒星 | 浏览目录，或读取已确认空的成员 | 缺少绑定要显示“成员未知”。仅在家族资料和成员已确定时才能返回零。已有根级 `SelectionSpec.family` 对不存在的族会报错，高层已确认的空集可用空索引表示。 |
| 04 两个选集得到完全相同的粒子，或手选索引含重复 | `member.origin`，`member.load()` | 相同精确输入可共享索引与数值。各自的来源、次序和重复次数都有科学意义。只有裸索引时可保留精确输入，目录归属需有证据。 |
| 05 同一薄盘测 κrot 和 Re | `stars.results.where(transform=face_on)`，`disk_stars.used_by` | 两个配方分别寻址。相同选集可关联两个输出。Re 的投影、权重或波段由其 Calculator 定义。 |
| 06 在子晕上测恒星质量，或直接在恒星成员上测总质量 | `sub.results[stellar_mass]`，`stars.results[total_mass]` | 即使值相同，调用的输入与配方各有自己的存储键。由 `stars.used_by` 可追到真正用它筛选的计算。 |
| 07 用恒星定轴，以全部粒子测薄盘 | `AlignVec(AngMomVec()).measure_with(FamilyFilter("star"))` | 定轴恒星记录变换之前的上下文和 `transform_measure` 角色。薄盘记录变换之后的上下文和 `output_filter` 角色。 |
| 08 用两倍半径做自适应孔径 | `Sphere(2 * rhalf)`，或先进入恒星成员 | 同一层组合筛选的两个条件在同一输入上求值。先取恒星再量半径会改变输入。记录半径配方和实际解析值、单位。 |
| 09 复合、取反及排除掩码 | `members[a & b]`，`results.where(uses=Sphere)` | 登记实际形成的交集。只有在中间视图真正被消费时，才有可沿两级访问的成员。 |
| 10 居中、定轴、改孔径，或多个 Pipeline 分支各用不同坐标 | 整链 `transform=`，固定版本加 `output=` | 只给真正使用该筛选的输出建立关联。恢复后的父坐标、作用于整根还是子集、实际残留的变换都影响上下文。 |
| 11 同一孔径刷新后，成员或方向变化 | `db.results.where(ref=..., output=...)` | 历史图固定到原来的成员、实际矩阵和版本。按配方与变换方法查询仍可能有多个来源。 |
| 12 每个宿主下有多个子晕，统计 Mstar 并关联宿主质量 | `db.study(objects=halo.subhalos, calculations=...)` | 取每个子晕作一行，在列上用恒星筛选。这样多个无星子晕仍各占一行。宿主质量表先选成每个宿主一行，再按定位键做多对一左连接。 |
| 13 只想知道样本中哪些对象算过，随后再补测 | `db.results.where(objects=..., ...)`，`study.collect(missing="keep")` | 前者返回已存命中。后者对固定、可寻址的完整样本保留缺测格。仅有目录条目且精确成员未知的对象可先列入目录清单。 |
| 14 手动计算后再保存、自动命中、刷新失败或保存中断 | `res.save(db)`，`db.fetch(ref)`，`db.recover(ref)` | 延迟保存使用计算时冻结的成员事实。命中沿用生产该版本的记录。失败的尝试可查状态，只有确认保存的输出才有成功关联。 |
| 15 合作者只带数据库或导出的表，没有粒子文件或自定义 Calculator | 只读库看摘要与来源，离线用 `StudyTable.load` 看值 | 浏览与定位版本无需重建 Calculator 或读取粒子。重载粒子仍需加载器。 |
| 16 周期回卷、直接在自定义代码中切片或旋转 | 检查 `member.origin.context` 的完整性 | 精确根索引可独立重载。回卷可能按粒子有不同位移，自定义操作需要显式报告才能支持原筛选坐标重放。 |
| 17 大目录重复扫多种孔径、多个时间步有相同编号 | 限定快照及对象集合，按完整配方索引分页查询 | 各时间步分别寻址。只保留实际消费的视图并复用索引，避免每个结果复制一份大掩码。跨快照的粒子追踪需要另一类记录。 |

## 两个会改变 API 语义的实测

**一个星族条件和一个动态孔径。** 六个粒子沿 x 轴排列，暗物质的位置为 10、20、30，各质量 10，恒星的位置为 1、2、6，各质量 1。用 `rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")`，全体与恒星的投影半质量半径分别为 14 和 2，均为样例单位。以下两次测量得到的恒星质量分别为 **3 和 2**。

```python
from pynbodyext.calculate import Pipeline
from pynbodyext.filters import Sphere
from pynbodyext.properties import ParamContain

rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")
star = FamilyFilter("star")
ap = Sphere(2 * rhalf)
mass = ParamSum("mass")
joint = mass.filter(star & ap).run(sim)
sequential = Pipeline({"mass": mass.filter(ap)}).filter(star).run(sim)
```

在该基线上，`mass.filter(star).filter(ap)` 的签名等于 `mass.filter(star & ap)`。需要第二个条件在前一批粒子上求值时，应通过已得到的成员继续，或用真正嵌套的 scope。自动登记必须遵守运行时的求值对象。

**同一 Pipeline 的兄弟输出。** 另建一个六粒子快照，前两颗为暗物质，后四颗恒星依次位于 x、y、z 单位轴端点和原点。对 `sim.star` 用 `Disc(3.0, 0.1)` 筛选，在各自独立的新快照上，原坐标选中的根索引为 `[2, 3, 5]`，把 x 方向对齐到 z 轴后为 `[3, 4, 5]`，两个计数都为 3。在同一个活快照先计算旋转输出，再计算只声明薄盘的输出，两者都返回 `[3, 4, 5]`，根坐标发生变化且仍有活动的 `Rotation`。反过来排，原坐标输出返回 `[2, 3, 5]`。这是在该版本、该普通 Property/Pipeline 路径上复现的结果。

配方查询可以读取签名，实际坐标查询需要在筛选和输出消费时记录有效的变换。只要有步骤无法确认，就标明该上下文不完整。对声明自动恢复的分支，实施验收要求两种输出顺序分别与各自独立运行一致，并检查调用方的根坐标。显式要求保留变换的计算应记录其有效区间和作用目标。[作用域执行](../../pynbodyext/core/calculate/runtime/engine.py) · [Pipeline 输出](../../pynbodyext/core/calculate/nodes/pipeline.py) · [变换测量输入](../../pynbodyext/core/calculate/nodes/transforms.py)

## 持久化与验收

| 数据对象 | 身份和职责 |
|---|---|
| `CatalogueObject`、`DataObject`、`Selection` | 宿主关系、精确输入、带根版本的有序粒子选择。目录成员绑定继续连接科学对象与输入。 |
| `SelectionOrigin`（拟增） | 一份不可变的成员来源。包含直接来源对象的定位键（嵌套时还含上一级来源）、完整 Filter、实际筛选上下文、已解析参数和精确成员引用。相同定义产生不同观察时可并存。 |
| `SelectionUse`（拟增） | 一次使用连接固定结果版本、输出路径、DAG 节点路径、角色和来源。多次使用可以引用同一来源。缓存复用的调用仍可另有自己的 Run。 |
| 输出配方索引与输出观察 | 对每个 Recipe 的具名输出索引有序作用域与带角色的依赖。对每个已确认的版本、输出路径保存实际测量上下文及其完整性，即使该输出没有 Filter。 |
| `ResultStore`、`Study`、`StudyTable` | 缓存键仍为精确输入 × 完整 Calculator 签名。Study 固定有序输入与列。已登记成员作为 Study 输入时可绑定现有精确视图定位键，v1 表要求每个精确输入仅占一行。需要区分多个科学宿主时以父对象作行，或在清单保留各自的科学定位键。 |

运行时在筛选被消费时捕获根索引，沿用现有家族分块次序与重复索引约束。只保留冻结的描述和可共享的索引块即可延迟保存，活快照可以释放。成员视图读取原坐标与单位。实际变换记录保留顺序、作用目标、测量来源、旋转矩阵或带单位的位移及重放能力。遇到缺失的步骤可标记上下文不完整。大型选择可将连续区间、稀疏索引或共享索引块作为内部存储选择，重载必须逐项一致。

在基线实现中，`cat.halos[number]` 先枚举目录行，再在 Python 中匹配编号。大目录下这会让一次单对象访问读整批元数据。编号查询应直接按目录、种类和类型化编号定位，按宿主枚举应使用 `host_id`。对象与结果的批量浏览按集合查询，预览只读一页。[目录集合实现](../../pynbodyext/database/catalogues.py)

核心只产生无数据库依赖的观察数据，数据库层随原有保存确认和恢复操作关联这些数据。保存中断后的精确回执恢复须幂等，并发登记同一来源须复用同一记录。已登记来源可以跨结果版本共享。元数据查询在同一读快照固定命中版本，避免翻阅结果时被并发刷新替换。旧库可用签名重建配方索引，历史成员仅来自已存在的观察记录。

### 优先验收场景

1. 一次 `calc.run(obj)` 或 `run(sim).save(db)` 后，用新构造的同签名 Filter 找回成员。原生加载器与 Filter 都禁用时仍能只读浏览。精确粒子重载与原 `get_index_list(root)` 逐项相等。
2. 复现上述两项数值对照，核对取反及复合 Filter、父节点继承、定轴前后的角色，以及有意持续和应当恢复的变换。旋转/平移重放须在独立副本上同时核对位置与相关速度、单位和目标。
3. 两个无星子晕、一份缺少成员绑定的目录项、一次缺测和一次失败刷新能在样本中解释清楚。历史结果固定到原成员，缓存命中不产生新的历史观察。
4. 对多个子晕与两个目录版本做宿主连接，核对行数和键。在只读、旧数据和大样本上统计 SQL 调用、载荷解码、粒子加载及重复索引字节数。目录集合按集合查询，元数据发现的粒子加载与科学载荷解码次数都应为零。

上述新增接口属于设计合同。已有行为可对照[基线版本的 ADR-021](https://github.com/wx-ys/pynbody-extras/blob/c56120ffb31f0461450393eaa4039dfa25369da7/docs/adr/0021-database-studies-and-detached-tables.md)、[粒子选择合同](../adr/0016-replayable-particle-selections.md)和[目录对象测试](../../tests/test_database_catalogues.py)。

本次基线验证覆盖目录、Database Study、StudyTable、签名往返和 ORM 选择的快速测试，139 项通过。项目环境缺少可选 pandas，一项跳过，另有一项数据测试由标记排除。在有 pandas 的独立环境中，示例的两次连接保留了合成清单的全部五行，正确区分两个目录中的同号对象并拒绝重复的宿主键。
