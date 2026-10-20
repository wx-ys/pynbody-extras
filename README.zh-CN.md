# pynbodyext：pynbody 的扩展与工具

[English](README.md) | [简体中文](README.zh-CN.md)

`pynbodyext` 通过可组合的 calculator、分箱 profile 和可直接发布的二维图来扩展
[pynbody](https://github.com/pynbody/pynbody)。

**注意：项目仍在积极开发中，欢迎反馈和贡献！**

## 安装

需要 Python 3.10–3.12。

使用包：

```bash
git clone https://github.com/wx-ys/pynbody-extras.git
cd pynbody-extras
python -m pip install -e ".[image,store]"
```

`image` extra 增加 adaptive binning（`powerbin`），`store` 增加 SQLAlchemy
结果存储。C++17 编译器用于构建可选的 native 扩展（`pynbodyext._native`），
gravity 求解器和 `bins["vcirc"]` 需要它；没有编译器时会跳过这个扩展，这些功能
不可用。编译器支持 OpenMP 时构建会启用多线程，否则为单线程。在 macOS 上，
Apple clang 需要 Homebrew 的 OpenMP 运行库才能多线程构建：

```bash
brew install libomp   # 会被自动找到；也可以设置 LIBOMP_PREFIX=/path/to/libomp
```

设置 `PYNBODYEXT_OPENMP=0` 可强制单线程构建。构建时会打印实际使用的 OpenMP 配置。

开发包：

```bash
make setup      # uv 管理的 .venv（由 .python-version 固定）、全部 extras、git hooks
make check      # lint + typecheck + 快速测试；不下载 testdata
make test-all   # 全量测试；首次运行会下载约 1.1 GB 的 pynbody testdata
```

如果还没有 [uv](https://docs.astral.sh/uv/)，请先安装。
如果你更愿意直接用 `uv`，等价的初始化命令是 `uv sync --all-extras`。
架构词汇和测试所约束的不变量在 [`CONTEXT.md`](CONTEXT.md)。
工作项和跨设备交接在 [`docs/README.md`](docs/README.md)；
当前队列在 [`docs/work/INDEX.md`](docs/work/INDEX.md)。

---

## 快速开始

以下示例假设 `sim` 是已经加载的 `pynbody` snapshot。

### Calculator

Properties、filters 和 transforms 都是惰性组合对象：构建一次图，可以在任意多个
snapshot 上求值。

```python
import numpy as np

from pynbodyext.filters import FamilyFilter, Sphere
from pynbodyext.properties import ParamContain, ParamSum
from pynbodyext.transforms import ShiftPosTo, WrapBox

# 30 kpc 内恒星的一半质量半径，使用 wrapped、centre-shifted 坐标
re = ParamContain(0.5).filter(Sphere("30 kpc") & FamilyFilter("star")).transform(
    WrapBox().then(ShiftPosTo("ssc"))
)

re(sim)                              # 求值：半质量半径
stellar_mass = ParamSum("mass").filter(FamilyFilter("star"))
stellar_mass / (4 * np.pi * re**2)   # 算术运算构建新的 calculator，调用时求值
```

调用 calculator 返回值；`run(...)` 返回完整执行结果——树、耗时、警告：

```python
res = re.run(sim, progress="node", perf_memory=True)   # 值 + 树、耗时、警告
print(res.value, res.report_summary())
```

`re.dependency_tree` 不运行计算，直接打印图本身：

```
ParamContain(0.5, "r", "mass")<prop>
├─ TransformChain<trans>
│  ├─ WrapBox(None, "minirange")<trans>
│  └─ ShiftPosTo("ssc")<trans>
│     └─ CenPos("ssc")<prop>
└─ AndFilter<filt>
   ├─ Sphere("30 kpc", (0, 0, 0))<filt>
   └─ FamilyFilter("star")<filt>
```

一个 calculator 可以复用为另一个 calculator 的输入：上面的 `re**2` 是
calculator 值输入，运行时自动解析，不需要手动安排计算顺序。

### Profile

`Bin1D`（以及用于第二轴的 `BinND`）对 snapshot 的粒子分箱；结果按名称回答每个
bin 的查询。查询写作 `"<field>.<transform>.<stat>@<weight>"`，除 field 外都可
省略——所以 `"vz.abs.mean@mass"` 是 *每个 bin 里 |vz| 的质量加权平均*。

```python
from pynbodyext.core.calculate import Bin1D
from pynbodyext.filters import FamilyFilter, Sphere

# 半径方向 20 个等数量 bin，到 30 kpc（mode 可选 "linear"、"log" 或
# "equaln"；也可以用 edges=、lows=/highs= 给显式 bin）
r = Bin1D("r", vmin="0 kpc", vmax="30 kpc", nbins=20, mode="equaln")
profile = r(sim)

profile["count"]              # 每个 bin 的粒子数
profile["vz.mean"]            # vz 数组的 mean
profile["vz.abs.mean@mass"]   # 先变换 |vz|，再做质量加权平均
profile["mass.sum"].plot()
(profile["mass.sum"] / profile["count"]).plot()  # 算术运算保持类型：仍然是 map

profile.star                  # 同样的 bin，只取恒星粒子
profile[Sphere("30 kpc") & FamilyFilter("star")]  # 同样的 bin，取一个子集

# 两个轴组成 map：同样的查询，每个 cell 一个答案
bins2d = (Bin1D("x", vmin="-30 kpc", vmax="30 kpc", nbins=128, alias="x")
          @ Bin1D("y", vmin="-30 kpc", vmax="30 kpc", nbins=128, alias="y"))(sim)
```

统计量有 `sum`、`mean`、`median`、`rms`、`disp` 和百分位（`pXX`）；变换有
`abs`、`log`、`log10`、`sqrt`、`square`；`count` 不需要 field；`@<field>` 可以
是 snapshot 携带的任意数组。

用 SPH kernel 平滑粒子，而不是把粒子数进 cell，用的是同一套查询；这样在 bin
稀疏的地方 map 仍然平滑：

```python
bins2d.sph_render["mass.sum"]         # 每个 cell 的质量，kernel 平滑，单位相同
bins2d.s.sph_render["count"]          # 只取恒星；是分数——这是 kernel count
bins2d.sph_render["vz.abs.mean@mass"] # 先变换 |vz|，再做质量加权平均
bins2d.sph_render["vz.median"]        # 分位数：kernel 触及的每个粒子
bins2d.sph_render["mass.sum.density"] # 派生密度，先平滑再求比值
bins2d.sph_render["gas_fraction"]     # 声明 allow_sph=True 的派生属性
```

它需要两个或三个空间轴，且 bin 等间距（`x`/`y`，可加 `z`）。统计量决定引擎：
kernel 求和（`count`、`sum`、`mean`、`rms`、`disp`）由 pynbody 的 C renderer
累加；分位数（`median`、`pXX`）把每个粒子 scatter 到 kernel 触及的 cell——
精确，但耗时约为 kernel sum 的十几倍；当 scatter 规模过大时会发出警告。派生
密度（`mass.sum.density`）和声明 `allow_sph=True` 的属性（`gas_fraction`、
`number_density`）用同样的方式回答：把它们自己的定义重新跑在平滑查询上。每个
粒子都需要 smoothing length；如果 snapshot 的 families 不是都带这个 block——
比如 gas 加 collisionless stars——就用 `bins2d.sph_render(smooth="kdtree")`，
它按 pynbody 自己的 `rho` 的方式推导。这个 view 可以调用：传入 `kernel`、
`smooth_floor`、`wrap`、`smooth` 中的任意参数即可得到配置好的 view，结果会像
默认 view 一样缓存。

### 图像后处理与可视化

两个轴组成 map；`bins2d["query"].image` 是其中一个查询的 `ImageData`——值加上
bin edges、单位和标签。`image.display.*` 负责显示，`image.process.*` 负责处理。

```python
from pynbodyext.plot import image

bins2d.imshow("mass.sum", log=True, colorbar=True)           # 一行代码：分箱数组 -> 画出的 map

mass = bins2d["mass.sum"].image
smoothed = mass.process.smooth.gaussian(fwhm=1.0)            # NaN 感知，fwhm 使用轴单位
observed = smoothed.process.psf.convolve(fwhm=3.0)           # 望远镜会看到的样子
detected = observed.process.noise.poisson(exposure=0.05, rng=1)       # 带种子，可 mask
restored = observed.process.psf.wiener(image.gaussian_psf(fwhm=3.0))  # 再还原

velocity = bins2d["vz.mean"].image                           # 速度图
binned = velocity.process.adaptive.bin(bins2d["mass.sum"], target_nbins=200)
binned.display.imshow(cmap="sauron", symmetric=True, colorbar="bottom")
velocity.display.contour(levels=[-100, 0, 100], colors="w")  # 画在同一组 axes 上

gas, dm = bins2d.gas["mass.sum"].image, bins2d.dm["mass.sum"].image   # 两张图，一个 figure
gas.process.compose.imshow(
    dm, style=image.MapStyle(cmap="inferno", stretch="log"), other_style=image.MapStyle(cmap="cividis")
)
```

每一步都返回新的 `ImageData`，操作记录在 `.ops` 里（所以 `repr` 会显示操作链）；
自由函数（`gaussian_smooth`、`compose_maps` 等）仍然可用于普通数组。
`cmap="sauron"` 是 Cappellari & Emsellem 的 SAURON/ATLAS³D 速度图（零值处为
绿色）；stitching 背后的 mask 也暴露出来
（`mask1, mask2 = gas.process.compose.masks()`）。Adaptive binning 需要可选的
`powerbin`（`pip install pynbodyext[image]`）；其余只需要 `numpy`、`scipy` 和
`matplotlib`。

---
