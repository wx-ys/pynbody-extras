# BUG-0006: median_filter 在非正方形 size 下补边错误，有遮罩时直接报错

## 背景

在修 TASK-0001 的 mypy 错误时发现（`plot/image/smooth.py:173`）。
`median_filter` 在取滑动窗口之前，先对填了 NaN 的图像补边：

```python
pad_y, pad_x = kernel_y // 2, kernel_x // 2
padded = np.pad(filled, ((pad_y, pad_x), (pad_y, pad_x)), mode=...)
```

第 0 轴补的是 `(pad_y, pad_x)`，第 1 轴也是 `(pad_y, pad_x)`；正确的应是
`((pad_y, pad_y), (pad_x, pad_x))`。正方形 size 时两者相同，所以只有在 size
不是正方形、且有遮罩（或非有限像素）时才会出问题——只有这条路径会补边（全部
有效的图像会直接交给 `scipy.ndimage.median_filter`）。

## 目标

对任意奇数 `ky`、`kx`，`median_filter(data, size=(ky, kx), mask=...)` 返回与
`data` 形状相同的图像，每个像素是其 `ky x kx` 邻域内有效像素的中位数。

## 实际行为

用同样的 NumPy 步骤复现（6 x 7 的图像，一个无效像素）：

```text
size (3, 3): medians (6, 7)            # 正确
size (3, 5): medians (7, 6)            # 形状错误
ValueError: operands could not be broadcast together with shapes (6,7) (7,6) ()
```

对某些形状（例如正方形图像），广播可能成功，返回的是错位窗口的中位数，而不会
报错。

## 期望行为

输出形状与输入相同；数值与逐像素在 `ky x kx` 窗口上直接求 `nanmedian`（同样的
边界模式）一致。

## 复现方式

```python
import numpy as np
from pynbodyext.plot.image.smooth import median_filter

data = np.random.default_rng(0).random((6, 7))
mask = np.ones_like(data, bool); mask[0, 0] = False
median_filter(data, size=(3, 5), mask=mask)  # ValueError
```

## 非目标

- 偶数 size、其他滤波器。

## 验收标准

- [x] 有一个非正方形 `size` 加遮罩的测试，在修复前失败。
- [x] 补边宽度为 `((pad_y, pad_y), (pad_x, pad_x))`。
- [x] 在小图像上，`size=(3, 5)` 和 `(5, 3)` 的结果与暴力 `nanmedian` 参考值
      一致，`size=3` 仍然一致。
- [x] `make check` 通过。

## 设计说明

`pynbodyext/plot/image/smooth.py` 中一行修改。检查 `ImageData` 的
`median_filter` 操作以及其他转发 `size` 的调用方。

## 测试计划

- 红：把上面的复现写成测试。
- 绿：修复；与参考值比较；`make check`。

## 结果

在 `codex/p0-median-filter-padding` 分支实施，与 codec PR #33 独立。
mask/NaN 路径按每个轴自己的半径对称补边。新增十个回归用例，覆盖
(3, 5)、(5, 3)、正方形窗口、mask、NaN/Inf、全遮罩和 ImageData 接口。
参考计算逐个枚举并截断邻域坐标，不使用补边或滑动窗口；同时检查图像几何、
单位、操作记录以及输入未被修改。

- 红灯：修复前 `pytest tests/test_plot_image_smooth.py -k median -q`：
  **8 failed、4 passed、28 deselected**。
- 绿灯：smooth/ops/methods 专项 **90 passed**。
- `make check`：ruff 和 mypy 通过；快速测试 **824 passed、3 failed、
  1 skipped、3 deselected**。
- `make test-all`：**827 passed、3 failed、1 skipped**。
- 三个失败仍为 pynbody 2.3.3 上已有 BUG-0002 的 SPH 失败，所以上方
  make check 验收项保持未勾选。
- make check 在 pytest 失败处停止，因此另外执行文档检查。
- 未改变依赖、架构、边界模式或偶数窗口契约。


### 最终基线验证 — 2026-11-08

分支已更名为 `codex/p0-image-sph-baseline`。维护者批准 BUG-0002 的依赖升级后，
`make check` 通过（830 passed、1 skipped、3 deselected），`make test-all` 通过
（833 passed、1 skipped）。上面的失败计数保留为仅完成中值滤波修复时的历史检查点；
剩余验收项现已满足。见[合并后的 PR 正文](../BUG-0002/pr-body.md)。

## 待决问题
