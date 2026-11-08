# BUG-0002: SPH 三维渲染测试在锁定的 pynbody 2.3.3 下失败

## 背景

用 `make setup` 新建的环境（macOS、Python 3.11.7）跑 `make test`，
`tests/test_bins_sph_render.py` 里有三个测试失败。`uv.lock` 对 Python >= 3.11
把 `pynbody` 解析为 **2.3.3**（3.10 为 2.0.0），而 `pyproject.toml` 只要求
`pynbody>=2.0`。这些测试以及一处生产行为依赖更新的 pynbody：BUG-0001 的
handoff 记录了测试通过的那台机器上是 pynbody 2.8.0。

## 目标

`make setup` 创建的锁定环境能通过 `make test`；`nz != ny` 的三维
`sph_render` 永远不会悄悄给出错误结果。

## 实际行为

```text
FAILED tests/test_bins_sph_render.py::test_a_volume_render_places_the_kernel_on_the_cell_grid
  AttributeError: 'CubicSplineKernel' object has no attribute 'value'
FAILED tests/test_bins_sph_render.py::test_a_volume_render_matches_a_direct_kernel_sum
  AttributeError: 'CubicSplineKernel' object has no attribute 'value'
FAILED tests/test_bins_sph_render.py::test_an_unequal_z_resolution_renders_the_grid_it_was_asked_for
  assert (np.int64(0), ..., np.int64(0)) == (2, 3, 2)
```

有两个独立原因：

1. **kernel API（仅测试）。** pynbody 2.3.3 的 kernel 只有
   `get_value(d, h=1)`，其中 `d` 已经是 `r / h`，而且只接受标量。第 341、369、
   403 行调用的是 `kernel.value(distance, h)`，传入物理距离（第 369 行传的是
   数组）。同一文件第 260 行仍在用 `get_value(r / h)`，两种 API 混用。
2. **`to_3d_grid` 的 z 方向尺寸（生产代码）。** pynbody 2.3.3 的 `to_3d_grid`
   用 `ny` 来确定 z 方向像素。对 `(4, 6, 3)` 的网格，本应落在 `(2, 3, 2)` 的
   粒子落到了 `(0, 0, 0)`。测试注释说针对这个上游 bug 的警告已经删掉，因为
   “已修复”——但这只对新版 pynbody 成立。在 2.3.3 上，`nz != ny` 时
   `bins.sph_render[...]` 会**不带任何警告**地返回错误网格。

## 期望行为

- 用 `uv.lock` 中的版本，`make setup && make test` 能通过。
- 在 `pyproject.toml` 允许的任意 pynbody 版本下，`nz != ny` 的三维渲染要么
  正确，要么给出警告（或报错）。

## 复现方式

```bash
make setup
.venv/bin/python -c "import pynbody; print(pynbody.__version__)"   # 2.3.3
.venv/bin/python -m pytest tests/test_bins_sph_render.py -k "volume_render or unequal_z"
```

## 非目标

- 修改 SPH 视图的约定，或 `_pynbody_render` 适配层的位置参数签名
  （由 `tests/test_pynbody_render_contract.py` 固定）。
- 支持 pynbody 1.x。

## 验收标准

- [x] `pyproject.toml` 和 `uv.lock` 在一个测试确实能通过的 pynbody 下限上
      保持一致；或者测试在下限允许的所有版本上都能通过。
- [x] 上述三个测试在 `make setup` 创建的环境中通过。
- [x] `tests/test_bins_sph_render.py` 使用同一个 kernel 求值辅助函数，不再
      混用 `get_value` 和 `value`。
- [x] 对 `to_3d_grid` 存在 z 尺寸 bug 的 pynbody，`nz != ny` 的三维渲染会
      警告或报错，并有测试覆盖（或者下限排除这些版本，并有契约测试固定）。
- [x] `make check` 和 `make test-all` 通过。

## 设计说明

### 维护者决策 — 2026-11-08

已批准方案 A：要求 **Python >=3.11、pynbody >=2.8.0**，锁定并测试
pynbody 2.8.0，明确取消 Python 3.10 支持。
2.8.0 是选定的已知可用基线，不表示它是最早修复版本。
上游 [040ade1](https://github.com/pynbody/pynbody/commit/040ade1eaf58a5fa9ac22194d7b4be8b26c536aa)
修复了 z 像素尺寸（`ny` -> `nz`），该提交已包含在
[v2.8.0](https://github.com/pynbody/pynbody/releases/tag/v2.8.0) 中；该版本的
包元数据要求 Python >=3.11。保留适配层签名及数值容差。

检查安装后的 2.8.0 源码发现，`Kernel2D.value` 仍会抛出 `NotImplementedError`。
因此数值参考统一通过一个辅助函数，在单位 h 下调用 `get_value(r / h)`，再除以
`h**kernel.h_power`，保持二维投影和三维核的归一化。生产代码不增加兼容分支。
投影测试除原有 h=1 外，还覆盖 h=0.7 和 h=1.4。

未采用的方案是保留旧下限，并增加依赖版本的警告或绕过逻辑。
已批准的版本下限避免了额外的生产行为分支。

## 测试计划

- 在 2.3.3 下复现三个 SPH 失败及新增的版本下限契约失败。
- 重新锁定并同步全部 extras，在最低支持版本 2.8.0 下测试渲染器签名、数值参考和
  非立方网格的粒子位置。
- 运行 `make check`、`make test-all`，验证锁文件一致性。

## 结果

在 `codex/p0-image-sph-baseline` 分支实现，接在 BUG-0006 中值滤波修复之后。
分支从 main 创建，不包含 codec PR #33。

- 更新 Python/依赖元数据、Ruff/mypy 目标、锁文件及双语 README。
  删除过时的 Python 3.10 typing 兼容分支及其直接依赖。
- 保留清华镜像，锁定 pynbody 2.8.0。其余保留包的版本不变；锁文件移除
  Python 3.10 专用解析结果，并加入 pynbody 所需的 `osfclient`/HTTP 依赖。
- 红灯：2.3.3 下 renderer contract + SPH 专项 **4 failed、42 passed**，
  即新增的版本下限契约及原有三个失败。
- 绿灯：同一专项加两个新的投影尺度用例，**48 passed**。
- `make check`：lint、mypy 通过；**830 passed、1 skipped、3 deselected**；
  文档校验通过。
- `make test-all`：**833 passed、1 skipped**。保留原有断言及数值容差，
  没有跳过或删除失败测试。
- 在 Linux / Python 3.11.9 / pynbody 2.8.0 下，`uv sync --locked --all-extras`
  及离线锁文件校验通过。尚未运行其他 Python/平台组合。
- 本地 `.github/` 文件原本未被 Git 跟踪，不包含在此次提交中；其中草稿 CI 矩阵
  仍有 Python 3.10，单独发布该工作流前需移除这一项。

由维护者继续审查、push、创建 PR 和合并。见 [PR 正文](pr-body.md)。

PR #34 审查补强：版本契约要求明确的 `>=2.8.0` 或更严格下限，防止仅排除
2.3.3 却意外允许 2.4-2.7。变异验证：在已安装 2.8.0 的环境中临时将声明改为
>=2.4.0，契约测试失败；恢复已批准的声明后通过。

### 合并验证 — 2026-11-08

已通过 [PR #34](https://github.com/wx-ys/pynbody-extras/pull/34) 合并
（提交 `d1be459`）。两批改动的组合验证通过：make check 为
866 passed / 1 skipped / 3 deselected；make test-all 为 869 passed / 1 skipped。
本地 main 已同步至 88ae3b4，与验证过的集成版本内容一致。
工作项保留至下次发布或 30 天后再归档。

## 待决问题

本次修复无待决问题；维护者已批准上述版本下限变更。
