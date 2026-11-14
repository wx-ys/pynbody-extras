# 保存和复用计算结果

[English](database.md) · [可运行示例](database_quickstart.py)

用 Calculator 计算，用 Result 查看输出，用 Database 保存和查找。Database 在
一个本地 SQLite 文件中保存结果和输入描述；粒子文件仍在原位置。

完整的 `pynbodyext` 安装包含 SQLAlchemy。使用轻量 `pynbodyext-core` 时，需要启用
`store` extra 才能使用 Database。纯计算不需要 SQLAlchemy。仓库开发环境可通过
`make setup` 准备虚拟环境。

下列代码块按 Notebook 中的顺序复用同一个打开的 `db`，最后再关闭。请将示例路径
换成你的快照路径；完整的自建数据示例见文末脚本。

## 先计算，再保存

```python
import pynbody
import pynbodyext as pe
from pynbodyext.properties import ParamSum

sim = pynbody.load('/data/snapshot_099')
sim.physical_units()
cc = ParamSum('mass')
res = cc.run(sim)

db = pe.Database('analysis.sqlite')
ref = res.save(db, name='total_mass')
snap = db.snapshots['snapshot_099']
saved = snap.results['total_mass']
print(saved.value)
print(saved.calculator)
particles = snap.load()
```

`save()` 自动建表、登记输入与计算器 signature，并编码保存结果。返回值是固定到
该版本的 `RecordRef`，也可通过 `res.store_ref` 访问。`run()` 已经捕获输入身份和
计算器 signature，因此删除原来的粒子对象后仍可保存结果。

`name` 可以省略，直接用计算器精确查询：

```python
saved = snap.results[cc]
```

一个输入内，一个别名绑定一个计算器 signature。将同一别名用于另一个计算会报
冲突。展示名和短 hash 不决定缓存身份。能重建已登记计算器类型时，
`saved.calculator` 返回可运行的计算器；否则返回 `None`，数值和已存 signature
仍可查看。

## 让重复计算使用数据库

```python
with db.use(mode='auto'):
    first = cc.run(sim)
    again = cc.run(sim)
    assert again.from_store

    fresh = cc.run(sim, mode='refresh')
    assert fresh.store_ref != first.store_ref

old = db.fetch(first.store_ref)
versions = db.versions(first.store_ref)
```

| 模式 | 读取已存结果 | 保存新算结果 |
| --- | --- | --- |
| `manual` | 否 | 显式调用 `res.save()` 才保存 |
| `auto` | 输入和计算完全匹配时读取 | 未命中后计算并保存 |
| `refresh` | 否，重新计算 | 保存为新版本 |

没有数据库或上下文时，`cc.run(sim)` 默认 manual。`db.use()` 默认 auto。
显式 `database=db` 或数据库对象输入在没有适用上下文时默认 auto；有适用上下文则
使用其模式。单次显式 `mode` 优先。

```python
res = cc.run(sim, database=db, mode='auto')

with db.use(mode='manual'):
    res = cc.run(sim)
    ref = res.save(name='total_mass')
```

不带参数的 `save()` 使用当前上下文或 Result 已绑定的数据库；都没有时请传
`save(db)`。关闭数据库后，已解码的数值仍可使用，后续数据库操作需要打开的
Database。退出上下文会恢复外层设置，发生异常时也一样。

每次成功的显式 save 都产生新版本。auto 命中复用已有版本，refresh 后仍可读取
旧引用。`snap.results[cc]` 在结果不存在时抛出 `KeyError`，不会自动计算。

## 浏览对象、重载粒子

```python
db.snapshots                         # 已登记的快照
snap = db.snapshots['snapshot_099']
snap.results                         # 已存结果摘要
snap.load()                          # pynbody 粒子
```

集合展示只读取元数据，不加载粒子或解码结果数组。通过低层 store 写入的已登记
输入结果仍会列出；没有摘要的结果标为 `summary unavailable`，不会为补摘要读取载荷。
集合列出已登记对象，不扫描数据目录。不同路径可以有相同 basename；名称有歧义时，查看 `db.snapshots`，再用
`db.snapshots[snap.key]` 按完整 key 选择。

数据库对象也能直接作为计算输入：

```python
res = cc.run(snap)                    # 先查结果，未命中才加载
```

对象的精确输入登记后，缓存命中不会调用粒子加载器，原粒子文件离线时也能读取
已存结果。`cc.run(sim, database=db)` 无法省掉调用者此前已经执行的加载。

## 保留精确的粒子选择

```python
with db.use():
    selected_result = cc.run(sim[[2, 0, 2]])

snap = db.snapshots['snapshot_099']
selected = next(iter(snap.selections))  # 本例仅有一个已登记选择
particles = selected.load()
assert len(particles) == 3
saved = selected.results[cc]
```

已存选择保留粒子顺序和重复索引。Pynbody 要求选择遵循根快照的 family-block
顺序；跨 family 交错会报错。重载会检查根粒子数和选择身份，但这些检查无法发现
任意保持粒子数不变的数据修改。

## 浏览 halo

本版本已验证的适配器支持 `HaloNumberCatalogue`，其整数数组给每个已加载粒子
指定一个 halo 编号。下面假定你的 `sim` 提供 `grp` 数组，且目录中有 ID 为 7 的 halo：

```python
from pynbody.halo.number_array import HaloNumberCatalogue

catalogue = HaloNumberCatalogue(sim, array='grp', ignore=-1)
res = cc.run(catalogue[7])
res.save(db, name='halo_mass')

snap = db.snapshots['snapshot_099']
halo = snap.halos[7]
halo.results['halo_mass']
halo.load()
cc.run(halo)                          # 命中已存结果时不加载粒子
```

保存前保留 catalogue 对象，让库能够捕获目录元数据。对于此适配器，保存一个活的
Halo 会自动登记目录，并根据粒子归属描述生成 revision。目录已失效或不受支持时，
库会给出提示，已存对象仍可通过 `snap.selections` 访问。

也可以自己命名并登记目录：

```python
cat = db.register_catalogue(snap, catalogue, name='groups', revision='v1')
halo = cat.halos[7]
```

`7` 是原目录 ID，不是行号。登记会复制已经加载的归属数组，并列出 ID，不逐个
加载 halo。存在多个目录或 revision 时，使用完整目录 key，或显式设定默认目录：
`db.set_default_catalogue(snap, cat.key)`。不同目录命名空间中的精确粒子选择，若
输入身份一致，可以共享计算缓存。

`cat.subhalos` 和 `halo.subhalos` 展示已登记的 subhalo 元数据。重载 subhalo
需要显式保存的粒子选择绑定。整数数组适配器不会推断 host 或 subhalo 归属。
其他 halo finder 需要经过验证的适配器，库不会猜测其构造参数。

## 按需登记模拟名称、自定义加载器

默认快照身份使用绝对文件路径。需要明确的科学身份时，可以登记模拟、快照和
可选的数据 revision：

```python
def load_custom(path):
    loaded = pynbody.load(path)       # 在这里绑定格式专用参数
    loaded.physical_units()
    return loaded

snap = db.register(sim, simulation='simulation-A', snapshot='099',
                   revision='r1', loader='custom')
db.bind_loader('custom', load_custom)
res = cc.run(snap)
db.simulations['simulation-A'].snapshots['099']
```

之后使用返回的对象，或在 `db.use()` 内计算。逻辑登记不会改写先前按路径身份
保存的结果。同一路径有多个 revision 时，需要选择具体对象；原始 sim 输入会
报告歧义。重开数据库后重新绑定自定义 loader；读取缓存结果不要求 loader 绑定。
Database 不会恢复早先 `pynbody.load()` 的任意 kwargs，也不序列化加载器函数。

原地修改数组不会改变默认缓存 key。用 `refresh` 更新当前结果，或用新的数据
`revision` 区分输入版本。默认模式下移动路径会改变身份；显式逻辑身份支持位置
别名。Calculator 在同样参数下改变科学行为时，需要更新其声明的 signature
version。数据库不归档粒子文件。

持久保存要求稳定的文件输入和计算器身份。内存 `pynbody.new()`、部分加载的文件
根对象、有外部活动变换的对象不能静默作为完整文件写入缓存。部分加载默认文件时，
需要用独立的逻辑身份和显式自定义 loader 定义其重载方式；不能沿用完整文件的默认
身份。支持的变换应放进 Calculator DAG。SQLAlchemy 是可选存储依赖；已验证的后端为 SQLite，不提供旧
数据库迁移。

元数据最终关联失败且已有结果引用时，可查看 `DatabasePersistenceError.result_ref`，
修复原因后调用 `db.recover(ref)`。恢复只验证并关联该精确版本，不重算、不追加
载荷。原有低层 `store=`/`reuse=` 调用仍可使用，请不要与 Database/mode 参数混用。

完成当前 Notebook 的数据库操作后关闭连接：

```python
db.close()
```

脚本中可以使用 `with pe.Database(...) as db:` 管理生命周期，但需将依赖该数据库
的操作放在这个上下文内。

## 运行完整示例

在仓库根目录执行：

```bash
PYTHONPATH=. .venv/bin/python docs/database_quickstart.py
```

脚本创建并重载四粒子的 Tipsy 文件、检查单位、保存并刷新结果、重载精确选择与
halo，然后重开 SQLite 并绑定一个必定报错的加载器。最后的命中断言证明根快照、
选择和 halo 结果都能在不加载粒子的情况下复用。脚本打印临时输出目录，并保留
文件供检查。
