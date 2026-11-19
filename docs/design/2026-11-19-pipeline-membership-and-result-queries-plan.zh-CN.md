# Pipeline 变换恢复、Filter 成员与结果反查实施计划

2026-11-19 制订 · 2026-11-20 已实施 · 源码基线 `c56120f`（含 ADR-021 的 Database / Study）。[设计与 17 个分析场景](database-member-queries-2026-11-19.zh-CN.md)

**目标**　让两个 Pipeline 输出互换顺序仍测得各自应得的粒子，并让已保存的测量能回答“用了哪批粒子、当时处于哪个坐标系、还有哪些输出用过它”。

**架构**　执行器在变换和筛选实际发生时记录事实。数据库在既有保存操作确认时把事实关联到固定结果版本。配方、精确粒子输入、目录对象、一次实际观察各有自己的身份。沿用 pynbody、NumPy、SQLAlchemy、pytest 和现有 ResultStore。

| 阶段 | 完成后用户可观察的行为 | 进入下一阶段的条件 |
|---|---|---|
| 1　Pipeline 分支隔离 | 同一活快照上，旋转分支和原坐标分支各得正确结果 | 两种顺序、嵌套、失败与轻量执行通过，旧缓存不能冒充新计算 |
| 2　捕获实际使用的 Filter 视图 | 普通计算和 Pipeline 能交出计算时冻结的成员、用途及实际坐标证据 | 精确根索引、动态参数、变换前后时点均能由内存测试验证 |
| 3　保存成员与原坐标导航 | 关闭并重开库后可用 `obj.members[filter].load()` 找回成员 | 延迟保存、缓存命中、故障恢复和并发登记通过 |
| 4　按 context 导航与重放 | 能找到指定来源，在独立视图重现筛选时的位置与速度 | 原坐标、选集坐标、部分可知的坐标均有可区分的结果 |
| 5　结果发现与双向反查 | 能按 transformer / Filter 找到输出，从论文里的固定数值追回当时的选集 | 同输出匹配、辅助用途、历史刷新与只读查询通过 |
| 6　目录、Study 与批量验收 | 能从子晕样本对接宿主、缺测和固定结果，用大目录验证查询成本 | 端到端科学样例、性能断言与项目全量检查通过 |

制订计划时主工作区的 HEAD 为 `d862742`，本文的源码位置与验证结果以含 ADR-021 的 `c56120f` 为准。实施从包含该版本的分支开始。各阶段按表中顺序交付。每阶段从所列用例写红灯测试，保留失败证据，接上真实入口后用同一测试验绿。工作项按仓库流程分配 ID，同步 `en.md`、`zh-CN.md` 和 `handoff.md`。涉及 core 的作用域和观察协议由相应 ADR 记录，随实现接受人工审阅。下文路径均相对仓库根，以 `tests/test_pipeline_branch_scopes.py` 等标出的新文件是计划交付物。

2026-11-20 完成六阶段验收。各阶段的红绿证据见 [分支恢复](../work/BUG-0013/handoff.md)、[执行时观察](../work/FEAT-0019/handoff.md)、[保存与恢复](../work/FEAT-0020/handoff.md)、[坐标重放](../work/FEAT-0021/handoff.md)、[结果反查](../work/FEAT-0022/handoff.md) 和 [目录与 Study](../work/FEAT-0023/handoff.md)。用户示例见[数据库指南](../database.zh-CN.md)。

## 1　先恢复 Pipeline 分支的变换

**改动位置**　`pynbodyext/core/calculate/nodes/pipeline.py` 的具名输出循环，`runtime/engine.py` 的完整及轻量执行路径，`runtime/context.py`、`runtime/input.py`、`nodes/transforms.py` 的作用目标和回退，以及 `nodes/postprocessors.py` 已有的异常清理。本段短路径均在 `pynbodyext/core/calculate/` 下。配方版本落在 `nodes/serde.py` 的签名编码。新增 `tests/test_pipeline_branch_scopes.py`。

- [x] **1A　先固定科学对照。** 将下面的测试放入新测试文件。在新的六粒子快照上也分别单跑两个 Calculator，确认独立对照。每个用例有自己的快照，非零速度能检出只恢复位置的错误。

```python
import numpy as np
import pynbody
import pytest

from pynbodyext.calculate import Pipeline, PropertyBase
from pynbodyext.filters import Disc
from pynbodyext.transforms import AlignVec


@PropertyBase.dataclass
class RootIndices(PropertyBase):
    def calculate(self, sim, params=None):
        return np.asarray(sim.get_index_list(sim.ancestor)).copy()


def branch_snapshot():
    sim = pynbody.new(dm=2, star=4)
    sim["pos"][:] = [[10, 10, 10], [11, 11, 11], [1, 0, 0],
                     [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    sim["vel"][:] = np.arange(18.0).reshape(6, 3)
    return sim


@pytest.mark.parametrize("order", [("rotated", "native"), ("native", "rotated")])
def test_branch_order_restores_entry_frame(order):
    sim = branch_snapshot()
    before = {key: np.asarray(sim[key]).copy() for key in ("pos", "vel")}
    entry = sim.current_transformation()
    disc = Disc(3.0, 0.1)
    face_on = AlignVec(np.array([1.0, 0.0, 0.0]))
    outputs = {"rotated": RootIndices().transform(face_on).filter(disc),
               "native": RootIndices().filter(disc)}
    expected = {"native": [2, 3, 5], "rotated": [3, 4, 5]}
    for key in order:
        fresh = branch_snapshot()
        np.testing.assert_array_equal(outputs[key].run(fresh.star).value, expected[key])
    result = Pipeline({key: outputs[key] for key in order}).run(sim.star)
    assert result.ok
    for key in order:
        np.testing.assert_array_equal(result.value[key], expected[key])
    for key, original in before.items():
        np.testing.assert_allclose(sim[key], original, atol=1e-12, rtol=0)
    assert sim.current_transformation() is entry
```

- [x] **1B　落实作用域所有权。** 每个具名分支以父输入的当前坐标入场。分支新建的默认可逆作用域在其结果完成物化后按逆序退出，再评估兄弟分支。嵌套 Pipeline 自身的变换持续到它的最后一个子输出结束。把已生效步骤及时交给作用域管理，覆盖 Filter、输入检查、属性计算、物化时抛错以及变换链部分成功。完整、轻量执行和已有异常回退共享同一份所有权，每步只清理一次。原生复合 handle 的 `revert()` 会连父变换一起撤销。跨作用域时在已变换的实际目标 SimSnap 上建立独立可回退的步骤，使子分支退出时父 handle 仍然有效。验证根目标及 `move_all=False` 的子集目标。
- [x] **1C　验证生命周期与失败。** 外部先施加一层平移，Pipeline 再施加父变换和子旋转。成功、`raise`、`collect`、`collect_partial` 四条路径均核对位置、速度、活动 handle 和调用次数。外部 handle 仍可由调用方正常撤销。`collect_partial` 中失败输出为 `None`，安全退出后的兄弟值等于独立对照，部分结果无成功保存。返回数组的分支也要在清理后保留测得的值和单位。对 `batch()`、观察器开关及运行时缓存开关重复分支测试。共用读坐标节点时，施加和回退后均使受影响的缓存失效。两个兄弟复用同一变换实例时，第二个也须实际处于其声明的坐标系，不能命中已撤销的 handle。
- [x] **1D　界定兼容边界。** 本期改变 Pipeline 所有的默认可逆作用域，保留单跑 Property 的已建立合同、直接运行 Transform 的 handle 所有权及显式 `revert=False` / `NEVER` 的持续效果。对显式保留的平移断言后续分支看见该平移。不可逆操作或清理错误使坐标无法确定时，保留原始异常和清理诊断，停止依赖未知坐标的兄弟输出。阶段 2 起将未知坐标标记为不完整。用一次 Bin 的分箱值与其惰性后续访问覆盖既有的回退警告合同。
- [x] **1E　更新旧约定并隔开历史缓存。** 先将 Pipeline 边界写入 ADR，再更新 `tests/test_postprocessors.py` 中三项已记录旧行为的断言。分别为 `test_inherited_ancestor_transform_is_not_cleaned_by_child_processor`（子节点保留父层，父层退出时恢复）、`test_native_cleanup_callback_retains_its_existing_cascading_semantics`（验证父层持续到兄弟结束、父子各退一次）与 `test_collect_partial_cleanup_leaves_successful_sibling_science_correct`（成功兄弟仍测得 15，最终根位置回到 0）。保留原有科学值、失败诊断和保存断言。`tests/test_calculate_runtime.py` 的独立调用合同继续通过。

**缓存验收。** 建议给执行内置分支循环的 Pipeline 配方加入持久的执行语义版本，覆盖内置类、沿用该循环的子类以及含这类节点的外层配方。版本进入规范签名，不能仅靠日志或进程内缓存。用旧版写入顺序污染的结果，升级后 `mode="auto"` 必须重算并返回 1A 的两个集合。原固定引用仍读出旧值，含旧列的 Study 仍可读历史，不能把新算法写入旧列。普通独立 Property 的既有地址继续命中。首版可保守地让使用该分支循环的无变换 Pipeline 也首次重算。

**红绿命令**　先运行 `.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py -k branch_order`。旧版应有两个失败，修复后两种排列均通过。再运行 `.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py tests/test_postprocessors.py tests/test_calculate_runtime.py tests/test_calculate_batch.py tests/test_bins_transform.py tests/test_signature_roundtrip.py tests/test_database_study.py`。

## 2　在计算时冻结真正用过的 Filter 成员

**改动位置**　`pynbodyext/core/calculate/runtime/input.py` 的 `FilterResult.filtered_sim` 和 `NodeInput.active_sim`，同目录的 `engine.py`、`persistence.py`，以及 `nodes/filters.py`、`nodes/transforms.py`、`nodes/pipeline.py`、`result/result.py`。新增 `runtime/selection_observations.py` 存放无数据库依赖的冻结记录，新增 `tests/test_selection_observations.py`。记录必须能随 `run(sim)` 的 Result 延迟保存，和用户是否打开调试观察器无关。

- [x] **2A　记录消费时点。** 让测试 Filter 统计掩码求值次数，让消费端取一次真实 SubSimSnap。冻结直接来源、根版本、有序根索引、完整 Filter 签名、当次已解析参数及单位。链接具名输出路径、DAG 出现路径与用途。共享视图可被两个输出引用。仅返回布尔掩码、`a & b` 的中间掩码、`~a` 的被取反项分别测试。只有真正送入下游的视图产生来源。索引与当时的 `get_index_list(root)` 逐项相同。同一观察不能为了留档再跑 Filter。
- [x] **2B　分开记录筛选和测量。** 用 `AlignVec(AngMomVec()).measure_with(FamilyFilter("star"))` 定轴，随后用 Disc 取薄盘，再加一个同父层且无 Filter 的对照输出。断言定轴前的恒星视图角色为 `transform_measure`，薄盘角色为 `output_filter`，后者带实际旋转矩阵。每个具名叶输出还在自身测量时记录有效 context。无 Filter 的叶也有其记录。若同一 Filter 先后出现在这两个角色或从 DAG 缓存取回，保留两次用途和各自时点。父层共用的筛选可关联到所有实际继承它的输出。
- [x] **2C　用动态孔径验证筛选的输入。** 下列数据放入 `test_dynamic_aperture_uses_actual_input`，给坐标与质量设置一致的样例单位。在下列科学断言外，检查冻结的两个实际孔径分别为 28、4，单位与坐标可换算。交集记录一个成员来源。嵌套的计算可从恒星来源继续找到球形成员。

```python
def test_dynamic_aperture_uses_actual_input():
    import numpy as np
    import pynbody
    from pynbody.array import SimArray
    from pynbodyext.calculate import Pipeline
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.properties import ParamContain, ParamSum

    sim = pynbody.new(dm=3, star=3)
    xyz = np.zeros((6, 3))
    xyz[:, 0] = [10, 20, 30, 1, 2, 6]
    sim["pos"] = SimArray(xyz, "kpc")
    sim["mass"] = SimArray([10., 10., 10., 1., 1., 1.], "Msol")
    rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")
    star, aperture = FamilyFilter("star"), Sphere(2 * rhalf)
    mass = ParamSum("mass")
    assert float(rhalf.run(sim).value) == 14.0
    assert float(rhalf.run(sim.star).value) == 2.0
    joint = mass.filter(star & aperture)
    chained = mass.filter(star).filter(aperture)
    assert chained.to_signature().identity() == joint.to_signature().identity()
    assert float(joint.run(sim).value) == 3.0
    nested = Pipeline({"mass": mass.filter(aperture)}).filter(star)
    assert float(nested.run(sim).value["mass"]) == 2.0
```

- [x] **2D　验证记录脱离活对象。** 以同一家族内含重复、非升序索引的视图为输入，测量后修改原数组、删除活快照并回收。用返回标量的 Calculator 验证冻结索引、参数和矩阵保持原值，记录不持有快照、FilterResult 或 handle。库原有的跨家族块重排限制要在登记前给出明确错误。普通不可持久化的纯计算仍能运行。未由运行时观察到的自定义切片不会生成推测出来的成员。

**红绿命令**　`.venv/bin/python -m pytest -q tests/test_selection_observations.py tests/test_pipeline_branch_scopes.py`。基线已能给出 2C 的科学值。本阶段的红灯应来自缺少实际观察。验绿同时检查用例的数值、出现路径、角色和筛选次数。

## 3　随保存确认写入成员，打通原坐标导航

**改动位置**　`pynbodyext/database/models.py`、`database/database.py` 的 `save_result`、`_finish_save`、`recover`、`_record_reuse`，`database/objects.py` 和 `database/catalogues.py` 的对象入口。复用 `pynbodyext/orm/selections.py` 与 `orm/repository.py` 的根版本、精确视图与家族顺序校验。新增 `pynbodyext/database/members.py`、`tests/test_database_members.py`、`tests/test_database_member_recovery.py`。

| 数据或关系 | 持久身份与读写条件 |
|---|---|
| 精确成员输入 | 同一根版本下逐项相同的索引序列与输入约定可共享 Selection / DataObject。顺序或重复次数不同必须可区分。 |
| `SelectionOrigin` | 直接来源的科学定位键或精确输入、可选上一级来源、完整 Filter、实际参数和筛选 context、精确成员引用共同标识一次可复用的观察。 |
| `SelectionUse` | 固定结果版本、输出路径、DAG 出现路径、角色、来源共同标识一条用途。同一成员被两个输出使用时保留两条关系。 |
| 输出观察与保存操作 | 每个已确认版本和输出路径保存测量 context 与完整性。待确认的冻结记录关联 SaveOperation 的操作 ID。大索引按内容共享暂存，只在操作上写引用。获得精确回执后，使用关系、输出观察与结果摘要及完成标记在同一最终事务内发布。 |

- [x] **3A　接通两条生产入口。** 同一快照在两个独立库分别走 `calc.run(obj)` 和 `calc.run(sim).save(db)`。延迟保存前改动活数据，再禁用 Filter。重开库，用新构造的同签名 Filter 找到原来那批粒子。在同一输入约定下两个库得到相同根索引。只读浏览时禁用粒子加载器也能看见来源，显式 `.load()` 才加载对应根版本。
- [x] **3B　把登记和浏览放在同一成员集合。** `obj.members.register("star")` 在原坐标登记，重复登记复用唯一已确认的对应来源。`obj.members["star"]` 和 `obj.members[FamilyFilter("star")]` 找到同一成员，`"dm"` 同理。读取缺失的 `[]` 给出缺失信息，且 SQL 写入数、Filter 执行数均为零。为已知无恒星的精确子晕登记空索引。元数据目录项无绑定时显示未知。两者均须与一次科学计算失败区分。
- [x] **3C　检验身份和嵌套。** 同批索引在两个不同来源得到时，直接 `.load()` 成功，`.origins` 有两项，`.origin` 报歧义。同一 Filter 在不同坐标选出两批粒子时，`[]` 报歧义并列出候选。相同大小、不同顺序或重复次数的成员也必须可区分。指定上一级来源后只能浏览该路径登记的后代。两个目录对象即使共享精确输入，也分别保留科学来源。显式登记本身没有结果用途。
- [x] **3D　验证重试、并发和旧库。** 在四处注入中断——收到回执前、已有 payload 和精确回执但未确认、写入首条用途后且事务未提交、确认完成后再次调用恢复。有精确回执时连续两次 `recover(ref)` 仍只有原版本和预期条数，半写入的关系对读者不可见。缺回执时不能凭当前版本猜测。两个独立数据库连接并发保存同来源时复用一份来源，两个不同结果版本各保留自己的用途。分别以来源及以版本和输出路径查用途，验证两个方向的外键和唯一约束。`auto` 命中增加复用 Run，生产版本的观察数保持原值。从共享同一精确输入的另一个目录对象命中时也不能替它补造筛选来源。`refresh` 失败或部分成功都不发布成功关系，先前固定引用仍可读取。旧版库首次可写打开后升级。只读打开旧库可读原有值和“未记录”状态，DDL 为零。

**红绿命令**　`.venv/bin/python -m pytest -q tests/test_database_members.py tests/test_database_member_recovery.py tests/test_database.py tests/test_orm_selections.py`。新库空查、写入后重开、第二次恢复以及两个独立连接均要经过公开 Database 入口。结果的基本寻址继续使用精确输入和完整 Calculator 签名。

## 4　按实际 context 导航并重放筛选坐标

**改动位置**　扩充阶段 2 的观察记录、阶段 3 的 `database/members.py`。在 `pynbodyext/transforms/rotate.py`、`shift.py` 接入实际变换参数，在 `wrap.py` 表明能力边界。新增 `pynbodyext/database/selection_frames.py` 和 `tests/test_member_frames.py`。

- [x] **4A　按来源筛选。** `stars.members.where(filter=Disc)` 匹配根类型，传实例匹配完整表达式。`transform=face_on` 匹配该来源声明的有序变换方法。`context=chosen.origin.context` 匹配相同根版本、目标、单位和实际有序操作。只给完整实际上下文做相等匹配。用相同定轴配方、不同实际方向造两个来源，再给同批索引造两个不同 context。`.one()` 对零个或多个来源报出数量与可选来源。指定唯一来源后可读取 context，并继续在该来源下导航。
- [x] **4B　验证可重放坐标。** 在真实筛选消费点用测试探针拷贝 `pos`、`vel`、单位和根索引作为金标准。对旋转、位置平移、速度平移，以及“先平移再旋转”的链，比较 `chosen.load(frame="selection")` 与金标准。验证交换两种操作次序得到不同结果。`chosen.load()` 与从新打开的原快照用同一批索引取得的值相同。数值统一单位后以 `assert_allclose` 检查，并独立断言返回单位和索引。
- [x] **4C　覆盖目标与无法重放的情况。** 分别对根和子集施加变换，在混合受影响与未受影响粒子的后续视图上，按各步实际目标逐粒子验证。含重复索引按粒子身份施加正确次数。先后加载两个来源，确认原快照和另一来源坐标保持各自金标准。遇到未提供证据的自定义变换、周期回卷或无法说明的外部变换时，显示具体步骤和不完整状态。选择坐标加载给出可解释错误，精确原坐标成员仍可加载。缺少对应根版本的粒子文件时，两种加载都应明确报告输入不可用，元数据仍可浏览。再测登记版本或家族布局不符的加载器。对仅有路径且无法检测原文件被等量替换的输入，重放依赖调用方保留原文件。

**红绿命令**　`.venv/bin/python -m pytest -q tests/test_member_frames.py tests/test_transforms_wrap.py tests/test_database_members.py`。重放测试在记录后禁用原 Filter 和用于定轴的 Calculator，防止重算恰好得到相似坐标而误过。

## 5　按配方或 context 发现结果，双向追溯成员

**改动位置**　新增 `pynbodyext/database/result_queries.py` 和 `tests/test_database_result_queries.py`。`database/models.py` 增加可从 Recipe 重建的输出与依赖索引。`database/database.py` 与 `database/objects.py`、`catalogues.py`、`members.py` 共享查询入口。保存时从规范签名建立输出路径、整条有序作用域、直接 Filter、带路径及角色的依赖。实际使用关系和实际坐标来自阶段 3 的已确认记录。读取沿用 `SavedResultRef` 的数据库命名空间校验。

- [x] **5A　按科学问题查输出。** 用同一个有足够恒星的星系，构造三个具名输出。`krot` 与 `Re` 均 face-on，只有 `krot` 用 Disc，`edge` 用另一种朝向和 Sphere。给 `Re` 测试单设光度列，明确它是此样例的投影半光半径。`stars.results.where(transform=face_on)` 恰得 `krot`、`Re`。加 `filter=Disc` 只得 `krot`。加 `filter=Sphere` 得零项，即使这两个条件各自在同一 Pipeline 里出现。再加两种不同参数的 Disc，类型查询均命中，实例查询只命中完整签名相同者。`db.results.where(objects=[stars], ...)` 给出相同命中。
- [x] **5B　区分声明、用途与实际坐标。** 用“定心→定轴”、反序、单独定轴三组。整链 `transform=` 只匹配正确顺序与长度，`scope=` 保留层级和回退策略。`uses=Sphere` 可以发现球形筛选、交集里的球、取反的球、定轴辅助用的球，并给出各自出现路径及角色。`filter=Sphere` 只收直接以 Sphere 为根的输出。`context=` 以测量当时的完整观察匹配，含无 Filter 的测量。单凭同名 transformer、未知历史或不同实际矩阵均不能命中。旧版本可从签名重建配方索引。无相应旧表的只读库从现有签名提供同样的配方发现，既不写库也不构造 Calculator。
- [x] **5C　检验两端的关系。** 在子晕上运行“恒星筛选后求质量”，并在其恒星成员上直接运行“求质量”。`sub.results` 与 `stars.results` 各得各自的直接结果。`stars.used_by` 找到子晕上真实消费该来源的输出。对具名输出从 `hit.members` 按角色和 Filter 返回原来源，再从该来源 `used_by` 回到同一固定引用、同一输出路径与出现路径。同一来源供两个结果使用时应有两条用途。同一输出反复使用同一 Filter 后仍可按路径区分。嵌套路径用键元组，如 `("kinematics", "krot")`。
- [x] **5D　锁定历史版本。** 单独保存的结果默认占一行，路径为 `()`。具名 Pipeline 默认每个叶输出占一行，共享所属根结果的固定引用。`output=()` 可显式选 Pipeline 根。`calculator=` 可用完整 Calculator 或已存签名，二者找到同一配方。导出一次 StudyTable，再刷新到新矩阵和新成员。默认查当前已确认结果。加载旧表后运行下面这段合同，须取回旧值、旧成员及旧 context。异库引用被拒绝，旧结果缺观察时显示“未记录”，有证据表明零次使用时显示已知为空。查询进行中由另一连接刷新时，同一次查询的引用与成员也必须属于同一版本。

```python
cell = frozen.cell(frozen.objects[0], "profiles")
hit = db.results.where(ref=cell.ref, output=("krot",)).one()
prior = hit.members.where(role="output_filter", filter=disc).one()
value = db.fetch(hit.ref).value["krot"]
np.testing.assert_allclose(value, old_krot)
particles = prior.load()
np.testing.assert_array_equal(particles.get_index_list(particles.ancestor), old_indices)
```

这里 `frozen` 是刷新前导出再加载的表，`old_krot` 和 `old_indices` 是第一次运行时直接测得的对照。另设查询测试禁用加载器、自定义 Calculator 类的重建及 ResultStore 数值解码。元数据查询和双向遍历仍通过，显式 `db.fetch` 才读数值。保存查询索引时验证这些叶子没有额外生成独立的 ResultStore 版本。

**红绿命令**　`.venv/bin/python -m pytest -q tests/test_database_result_queries.py tests/test_database_member_recovery.py tests/test_database_study.py tests/test_study_table.py`。分别检查查询行的输出路径、固定引用、角色及计数，防止只验证命中数量掩盖跨分支串联。

## 6　把关系用在真实样本与大目录

**改动位置**　`pynbodyext/database/catalogues.py`、`pynbodyext/orm/repository.py`、`pynbodyext/database/result_queries.py`，Study 复用 `pynbodyext/database/study.py` 与既有 ObjectRef / StudyTable。新增 `tests/test_database_member_catalogue_workflows.py`，扩充 `tests/test_database_catalogues.py`。使用示例进入 `docs/database.zh-CN.md` 和 `docs/database.md`。

- [x] **6A　打通目录定位。** 验证 `snapshot.catalogues[key].halos[42].subhalos[7]`、`sub.host`、`sub.members["star"]` / `["dm"]`。同一快照两份目录、同目录两个 revision、两个时间步、整数 `7` 与字符串 `"7"` 都不得串号。宿主取自目录已声明的关系。无已知宿主时清单保留空键。按库命名空间、根版本、目录标识与版本、种类、类型化编号生成可连接的科学定位键。复用现有 `(catalogue_fk, kind, typed_object_key)` 唯一约束和 `host_id` 索引做点查/按宿主枚举。两个科学对象共用相同索引和数值时，按显式对象集合查询仍各返回该对象自己的定位键。
- [x] **6B　用五行清单验收样本分母。** 以 A、B 两份目录的完整 `cat.subhalos.to_dataframe()` 合并清单作左表，结果 `.to_dataframe()` 提供定位键与 `ref`。按 `object_locator` 一对一、`host_locator` 多对一分别左连接。用下表逐行比对，每次连接后都恰好五行。A/8 和 A/9 的 DM 输入不同，恒星质量求和各有一个取值为零的成功结果。故意给宿主表加一条重复键时必须报错。把要求非空输入的 Calculator 用在相同两行时另验其失败语义。

| 目录 / 子晕 | 宿主 | 恒星成员 | 预期恒星质量引用 | 预期宿主质量引用 |
|---|---|---|---|---|
| A / 7 | A / 42 | 有星 | A/7 的结果 | A/42 的结果 |
| A / 8 | A / 42 | 已确认为空 | A/8 的结果 | A/42 的结果 |
| A / 9 | A / 43 | 已确认为空 | A/9 的结果 | 缺测 |
| A / 10 | A / 42 | 无精确绑定 | 缺测 | A/42 的结果 |
| B / 7 | B / 42 | 有星 | 尚未测量 | B/42 的结果 |

- [x] **6C　与 Study 共享结果。** 以精确输入互异的子晕本身作行，以恒星筛选后的质量作列。单对象先算，Study 再 `collect(missing="keep")`，同格固定引用相同。一次失败刷新不把旧数值记作本次成功。两个无星子晕保留两行，未知精确输入的对象仍留在目录清单。把同一精确空成员两次作为 Study 行应按 v1 合同拒绝。要求同一读时刻的宿主与子晕测量放入一次 Study.collect 固定版本。关库后 `StudyTable.load` 能读表中数值，历史成员仍由原数据库按固定引用解析。
- [x] **6D　用工作量验收性能。** 生成 100 与 1,000 个子晕，各取同尺寸首页并读一个指定编号。通过 SQLAlchemy 事件统计 SELECT，固定页大小下两种样本量的查询数相同，编号查询带类型化键，宿主查询带 `host_id`，无逐对象补查。用 SQLite 查询计划核对索引使用。页间顺序稳定，遍历后定位键无重复、无遗漏。清单、结果元数据和 `repr` 的粒子加载次数及科学载荷解码次数均为 0。另用同一大选集供 10 个不同输出消费，核对持久索引块及其占用字节只有一份、用途恰有 10 条。第一版的存储粒度为实际消费的粒子集合。

**红绿命令**　`.venv/bin/python -m pytest -q tests/test_database_member_catalogue_workflows.py tests/test_database_catalogues.py tests/test_database_study.py`。让含 pandas 的项目测试环境实际执行两次连接和重复键测试。跑现有数据驱动的精确成员重载，用一份有单位的真实快照验证阶段 4 的重放。阶段交付跑 `make check`。涉及公开行为、数据路径的实现还跑 `make test-all`。把命令、具体红灯与绿灯、可选后端/数据是否实际运行记入该阶段工作项。

## 交付核对

| 设计用例 | 主要验收步骤 |
|---|---|
| 01–04　根对象、同号目录、空与未知、相同或重复成员 | 2D、3A–3C、6A |
| 05–07　同朝向多个量、直接输入与使用、定轴前后 | 2B、4A–4B、5A–5C |
| 08–10　自适应孔径、布尔表达式、父子和兄弟坐标 | 1A–1D、2A–2C、4B–4C、5B |
| 11–14　刷新、宿主统计、覆盖率、延迟保存和恢复 | 1E、3D、5C–5D、6B–6C |
| 15–17　只读旧库、不可重放操作、大目录与多个快照 | 3D、4C、5B、5D、6A、6D |

**本轮已运行的基线证据。** 直接取出本文 1A 的测试放到临时目录，在 `c56120f` 运行，两个顺序均为预期红灯。旋转先跑时原坐标输出实得 `[3, 4, 5]`，独立对照为 `[2, 3, 5]`。原坐标先跑时两个输出通过，但根位置恢复断言失败。2C 给出的半径 14 和 2、质量 3 和 2 四项科学断言通过。另行运行阶段 1 涉及的三项旧 Postprocessor 测试及已有 runtime、batch、Bin 变换测试，16 项通过。另用两个 DM 子集实测空恒星质量求和都返回 `0 Msol`，并用 pandas 独立验证 6B 的五行期望引用和重复宿主报错。其余新增用例的通过条件供各阶段实施时验收。
