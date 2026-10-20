# BUG-0003: 未编译原生 gravity 扩展时仍提供 vcirc，调用即 ImportError

## 背景

在 macOS（Python 3.11.7）上跑 `make test`，`tests/test_bins_vcirc.py` 的三个测试
全部失败。该环境没有编译出 `pynbodyext._native`（源码树和 `.venv` 里都没有
`_native*.so`），因此 `GRAVITY_NATIVE_AVAILABLE` 为 `False`，
`pynbodyext.gravity` 只导出 `GRAVITY_NATIVE_AVAILABLE`。`tests/test_gravity.py`
在这种情况下会跳过（`pytest.importorskip("pynbodyext._native")`），
`tests/test_bins_vcirc.py` 不会；`vcirc` 这个派生查询本身也不检查扩展是否可用。

## 目标

没有原生扩展时，不提供 `vcirc`（或给出清楚的错误信息），测试改为跳过而不是失败。

## 实际行为

```text
FAILED tests/test_bins_vcirc.py::test_vcirc_available_for_radial_axes_only
FAILED tests/test_bins_vcirc.py::test_vcirc_matches_analytic_point_mass
FAILED tests/test_bins_vcirc.py::test_vcirc_is_finite_and_nonnegative

pynbodyext/core/calculate/bins/result.py:924: in _vcirc
    from pynbodyext.gravity import KernelKind, calculate_acceleration
E   ImportError: cannot import name 'KernelKind' from 'pynbodyext.gravity'
```

- 即使缺少扩展，一维 `rxy` profile 的 `"vcirc" in bins.queries.names()` 仍为
  `True`；`condition` 只检查了坐标轴。
- 随后 `bins["vcirc"]` 抛出关于 `KernelKind` 的 `ImportError`，没有告诉用户
  真正的原因是缺少 C++ 扩展。

## 期望行为

- 当 `GRAVITY_NATIVE_AVAILABLE` 为 `False` 时，要么 `bins.queries.names()` 中
  不列出 `vcirc`，要么 `bins["vcirc"]` 抛出的错误写明缺少 `pynbodyext._native`
  扩展以及如何编译。
- 没有扩展时，`tests/test_bins_vcirc.py` 中依赖 gravity 的测试跳过，做法与
  `tests/test_gravity.py` 一致。
- 有扩展时行为不变。

## 复现方式

```bash
.venv/bin/python -c "from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE as A; print(A)"  # False
.venv/bin/python -m pytest tests/test_bins_vcirc.py
```

```python
import numpy as np, pynbody
from pynbody.array import SimArray
from pynbodyext.core.calculate import Bin1D

sim = pynbody.new(dm=1)
sim["pos"] = SimArray(np.zeros((1, 3)), "kpc")
sim["mass"] = SimArray([1e10], "Msol")
sim["smooth"] = SimArray([0.0], "kpc")
bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)
print("vcirc" in bins.queries.names())  # True
bins["vcirc"]                            # ImportError: KernelKind
```

## 非目标

- 修改 `vcirc` 的计算方法。
- 修复原生扩展的编译问题本身（见待决问题）；那属于 `setup.py` / `cpp/` 的构建
  改动，需要单独立项并经人工审核。

## 验收标准

- [x] 没有扩展时，`vcirc` 不出现在 `bins.queries.names()` 中，或
      `bins["vcirc"]` 抛出的错误信息写明 `pynbodyext._native`（二选一，并在
      `en.md` / `zh-CN.md` 中记录选择）。已选：给出清楚的错误（见“决定”）。
- [ ] 缺少 `pynbodyext._native` 时，`tests/test_bins_vcirc.py` 中依赖 gravity
      的测试跳过；有扩展时照常运行。
- [ ] 有一个测试覆盖“缺少扩展”的路径（例如 monkeypatch
      `GRAVITY_NATIVE_AVAILABLE`），在任何机器上都会运行。
- [ ] 有扩展和没有扩展时 `make check` 都通过。

## 设计说明

- `pynbodyext/core/calculate/bins/result.py` 中
  `@BinNDResult.derived("vcirc", ...)` 的 `condition` 可以再要求
  `GRAVITY_NATIVE_AVAILABLE`。应在调用时读取，而不是在导入时读取，方便测试
  monkeypatch。
- 如果查询仍然列出，就把 import 包起来，抛出 `ImportError`（或项目自己的错误
  类型），信息类似 “vcirc needs the compiled pynbodyext._native extension”。
- `pynbodyext/core/` 下的改动需要人工审核（见 `AGENTS.md`）；这次改动应很小。

## 决定

一维 `r` / `rxy` profile 上继续列出 `vcirc`；没有扩展时，`bins["vcirc"]` 抛出
`ImportError(name="pynbodyext._native")`，信息写明缺少的扩展以及如何重新编译。
理由：

- 如果用 `condition` 把它隐藏，`bins["vcirc"]` 会落到粒子数组的查找路径，报一个
  不相关的错误，用户仍然不知道是缺少扩展。
- 查询仍可被发现；`names()` 不随编译结果变化。

检查在调用时读取 `pynbodyext.util.deps.GRAVITY_NATIVE_AVAILABLE`，测试可以
monkeypatch 它。

## 测试计划

- 红：在没有扩展的机器上，`tests/test_bins_vcirc.py` 因 `ImportError` 失败
  （已观察到）。
- 为依赖 gravity 的测试加上 `pytest.importorskip("pynbodyext._native", ...)`。
- 增加一个把 `GRAVITY_NATIVE_AVAILABLE` monkeypatch 为 `False` 的测试，验证
  所选行为。
- `make check`；在有扩展的机器上跑 `make test-all`。

## 结果

## 待决问题

- macOS 上为什么没有扩展？`setup.py` 用 `-fopenmp` 和 `optional=True` 编译
  `pynbodyext._native`。Apple clang 在没有 `libomp` 时不支持 `-fopenmp`，
  所以编译很可能悄悄失败了。可用
  `.venv/bin/python setup.py build_ext --inplace` 查看编译器报错来确认。若确认，
  为 macOS 构建另开一个条目（例如只在 OpenMP 可用时加 `-fopenmp`，或在文档中
  说明 `brew install libomp`）。
