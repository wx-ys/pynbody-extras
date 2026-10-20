# TASK-0001: 修掉已有的 mypy 错误，让 pre-commit 钩子能通过

## 背景

pre-commit 的 `mypy` 钩子会检查每个暂存的 `pynbodyext/` 文件，并顺着它的 import
继续检查。`main` 上已经有 mypy 错误，所以只要提交改到相关模块，钩子就会失败。
BUG-0003（PR #25）只能用 `--no-verify` 提交；BUG-0001 也记录过 `make typecheck`
报出 22 个错误，都在它没改动的文件里。

BUG-0003 提交时（暂存了 `bins/result.py`）钩子报出的错误：

```text
pynbodyext/gravity/base.py:193: error: Argument 1 to "full" has incompatible type "tuple[Any]"; expected "tuple[()]"  [arg-type]
pynbodyext/gravity/base.py:193: error: Argument 1 to "float" has incompatible type "generic[Any] | complex | str | bytes | memoryview[int]"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
pynbodyext/core/calculate/store/sqlalchemy_store.py:68: error: Incompatible types in assignment (expression has type "str | Engine", variable has type "Engine")  [assignment]
pynbodyext/core/calculate/store/sqlalchemy_store.py:107: error: Value of type "Any | None" is not indexable  [index]
pynbodyext/core/calculate/bins/result.py:353: error: Incompatible return value type (got "ndarray[tuple[Any, ...], dtype[Any]] | None", expected "ndarray[tuple[Any, ...], dtype[Any]]")  [return-value]
```

### 完整输出

用户 Mac 上的 `make typecheck`（2026-10-20）：**10 个文件共 22 个错误**
（检查了 110 个文件）。mypy 1.19.0、numpy 2.3.5、matplotlib 3.10.7、
SQLAlchemy 2.0.52。省略了 `note:` 行。

```text
pynbodyext/gravity/base.py:193: error: Argument 1 to "full" has incompatible type "tuple[Any]"; expected "tuple[()]"  [arg-type]
pynbodyext/gravity/base.py:193: error: Argument 1 to "float" has incompatible type "generic[Any] | complex | str | bytes | memoryview[int]"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
pynbodyext/core/calculate/store/sqlalchemy_store.py:68: error: Incompatible types in assignment (expression has type "str | Engine", variable has type "Engine")  [assignment]
pynbodyext/core/calculate/store/sqlalchemy_store.py:107: error: Value of type "Any | None" is not indexable  [index]
pynbodyext/plot/image/smooth.py:173: error: No overload variant of "pad" matches argument types "ndarray[tuple[Any, ...], dtype[Any]]", "tuple[tuple[int, int], tuple[int, int]]", "str"  [call-overload]
pynbodyext/plot/image/smooth.py:338: error: Cannot call function of unknown type  [operator]
pynbodyext/plot/image/noise.py:93: error: Argument 1 to "_broadcast" has incompatible type "float | ... | None"; expected "ImageData | ..."  [arg-type]
pynbodyext/plot/image/cmaps.py:323: error: Argument "bad" to "with_extremes" of "Colormap" has incompatible type "str | tuple[float, ...]"; expected "tuple[float, float, float] | str | ... | None"  [arg-type]
pynbodyext/plot/image/cmaps.py:352: error: Argument 2 to "LinearSegmentedColormap" has incompatible type "dict[str, ndarray[...]]"; expected "dict[Literal['red', 'green', 'blue', 'alpha'], Sequence[tuple[float, ...]]]"  [arg-type]
pynbodyext/plot/image/cmaps.py:355: error: (same as 352)  [arg-type]
pynbodyext/plot/image/display.py:200: error: Incompatible types in assignment (expression has type "_ScalarMappable", variable has type "AxesImage | QuadMesh | ContourSet | PathCollection | Colorbar | None")  [assignment]
pynbodyext/plot/image/display.py:207: error: Argument 1 to "colorbar" of "FigureBase" has incompatible type "AxesImage | ... | None"; expected "_ScalarMappable | ColorizingArtist"  [arg-type]
pynbodyext/plot/image/display.py:242: error: Incompatible return value type (got "tuple[... | ImageData | AdaptiveMap | None, None]", expected "tuple[... | None, ImageData | None]")  [return-value]
pynbodyext/plot/image/display.py:272: error: Argument 1 to "set_aspect" of "_AxesBase" has incompatible type "str | float"; expected "Literal['auto', 'equal'] | float"  [arg-type]
pynbodyext/plot/image/display.py:456: error: Argument "shading" to "pcolormesh" of "Axes" has incompatible type "str"; expected "Literal['flat', 'nearest', 'gouraud', 'auto'] | None"  [arg-type]
pynbodyext/plot/image/display.py:526: error: Incompatible types in assignment (expression has type "ndarray[..., dtype[float64]]", variable has type "int | Sequence[float]")  [assignment]
pynbodyext/plot/image/display.py:1044: error: (same as 526)  [assignment]
pynbodyext/plot/image/compose.py:419: error: Argument 1 to "add_colorbar" has incompatible type "_ScalarMappable"; expected "AxesImage | ... | AdaptiveMap | None"  [arg-type]
pynbodyext/core/calculate/bins/_query_engines.py:216: error: Incompatible types in assignment (expression has type "ndarray[...]", variable has type "BinsArray")  [assignment]
pynbodyext/core/calculate/bins/result.py:353: error: Incompatible return value type (got "ndarray[...] | None", expected "ndarray[...]")  [return-value]
pynbodyext/transforms/wrap.py:212: error: Incompatible types in assignment (expression has type "ndarray[...]", variable has type "int")  [assignment]
pynbodyext/transforms/wrap.py:213: error: No overload variant of "__call__" of "_UFunc_Nin2_Nout1" matches argument types "int", "ndarray[..., dtype[float64]]", "int"  [call-overload]
```

按区域：`plot/image` 14 个（display 7、cmaps 3、smooth 2、noise 1、compose 1），
`core` 4 个（store 2、bins 2），`gravity` 2 个，`transforms` 2 个。大多是存根
精度问题（matplotlib 的 `Literal` 参数、numpy 的形状类型），而不是逻辑错误；
`display.py:200/207/242` 和 `compose.py:419` 共用一个类型写错的 “mappable”
别名，应一起修。

## 目标

`make typecheck`（`mypy pynbodyext`）不报错，提交时不再需要 `--no-verify`。

## 非目标

- 改变运行时行为。每处修改都只在类型层面（收窄、注解、`cast`），除非发现真正
  的 bug；真正的 bug 另开条目。
- 提高 mypy 的严格程度。

## 验收标准

- [x] 在本条目中记录完整的 `make typecheck` 输出（见“完整输出”）。
- [x] 在锁定的环境（`uv.lock`）下 `make typecheck` 通过。
- [x] 不新增没有错误代码和理由的 `# type: ignore`。（一个也没加。）
- [x] `make check` 通过；运行时行为不变。（lint 和 typecheck 通过；`make test`：
      814 通过、3 失败——即 BUG-0002 的三个 SPH 测试，与本条目无关。）

## 设计说明

已知 5 个错误的可能修法：

- `gravity/base.py:193` —— `np.isscalar` 不会收窄 `softening` 的类型。改用
  `isinstance(softening, (int, float, np.floating, np.integer))` 或
  `np.ndim(softening) == 0` 加 `float(np.asarray(softening))`，形状写成
  `(int(pos.shape[0]),)`。
- `store/sqlalchemy_store.py:68` —— `hasattr(..., "connect")` 不会把
  `str | Engine` 收窄；改用 `isinstance(url_or_engine, str)`（或 `cast`）。
- `store/sqlalchemy_store.py:107` —— 存根里 `inserted_primary_key` 是
  `Row | None`；取下标前先检查 `None`（缺失时报错）。
- `bins/result.py:353` —— `BinResultModel.valid_mask` 是 `np.ndarray | None`，
  而 `BinNDResult.valid_mask` 承诺返回 `np.ndarray`。先确定 `None` 会不会传到
  这里；不会就在赋值处（第 270 行）收窄，会就构造全为真的掩码。

有些错误可能取决于 numpy / matplotlib / sqlalchemy 存根的版本（已随输出记录）。
pre-commit 钩子用的是 `uv run mypy`；确认它与 `make typecheck`
（`.venv/bin/python -m mypy`）结果一致。

建议顺序（影响面小的先做，每个区域一个提交）：先修 `gravity` + `store` +
`bins` + `transforms`（8 个，正是会拦住 core 提交的那些），再修 `plot/image`（14 个）。

## 改动

每个错误一处类型层面的修改；除特别注明外，运行时行为不变。

| 错误 | 修改 |
|---|---|
| `gravity/base.py:193`（2 个） | 用 `isinstance(softening, (int, float, np.integer, np.floating))` 代替 `np.isscalar`；`np.full(int(n), ...)`。0 维数组或数字字符串不再被当作标量（两者都不符合 `float` 注解）。 |
| `store/sqlalchemy_store.py:68` | 在原有的鸭子类型检查 `hasattr(..., "connect")` 之后 `cast("Engine", ...)`。 |
| `store/sqlalchemy_store.py:107` | 检查 `inserted_primary_key` 是否为 `None`，是则抛 `RuntimeError`（原来是 `TypeError`）。 |
| `plot/image/smooth.py:173`、`:338` | `_PAD_MODES` 标注为 `np.pad` 的 mode `Literal`；`_REDUCERS` 标注为 `dict[str, Callable[..., Any]]`。 |
| `plot/image/noise.py:93` | 在原有“恰好给一个”检查之后加 `assert snr is not None`。 |
| `plot/image/cmaps.py:323` | `cast("ColorType", bad)`。 |
| `plot/image/cmaps.py:352`、`:355` | 新增 `_segmented()`，把数组形式的 cdict cast 成存根要求的类型。 |
| `plot/image/display.py:200`、`compose.py:419` | `Artist` 别名加入 `ScalarMappable`（`resolve_type_hints` 的命名空间也加上）。 |
| `plot/image/display.py:207`、`:242` | 在代码已保证的位置 `cast` 为 `ScalarMappable` / `Artist | None`。 |
| `plot/image/display.py:272`、`:456` | `aspect` / `shading` cast 为 matplotlib 的 `Literal`（matplotlib 运行时会校验）。 |
| `plot/image/display.py:526`、`:1044` | 由 `np.geomspace` / `np.linspace` 生成的等值线层级以 `.tolist()` 传入。 |
| `bins/_query_engines.py:216` | 拆成 `measure` / `denominator` 两个变量。 |
| `bins/result.py:353` | 模型没有 `valid_mask` 时抛 `RuntimeError`（两条构造路径都不会出现）。 |
| `transforms/wrap.py:212-213` | 循环变量 `axis` 改名为 `coords`（它与上面 `int` 类型的循环变量同名）。 |

顺带发现：`median_filter` 在非正方形 `size` 下补边宽度错误——这是真正的 bug，
已开为 BUG-0006，这里不修。

## 测试计划

- 修改前跑 `make typecheck`（记录错误数），修改后为零。
- `make check`；如果改到 `gravity` 或 `store` 的代码路径，再跑 `make test-all`。

## 结果

## 待决问题

- 用 `uv.lock` 里的版本和用户的 conda Python，错误数是否一样？钩子运行的是
  `uv run mypy`。
