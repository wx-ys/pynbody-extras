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

## 目标

`make typecheck`（`mypy pynbodyext`）不报错，提交时不再需要 `--no-verify`。

## 非目标

- 改变运行时行为。每处修改都只在类型层面（收窄、注解、`cast`），除非发现真正
  的 bug；真正的 bug 另开条目。
- 提高 mypy 的严格程度。

## 验收标准

- [ ] 在本条目中记录完整的 `make typecheck` 输出（上面的列表只是其中一部分）。
- [ ] 在锁定的环境（`uv.lock`）下 `make typecheck` 通过。
- [ ] 不新增没有错误代码和理由的 `# type: ignore`。
- [ ] `make check` 通过；运行时行为不变。

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

有些错误可能取决于 numpy / sqlalchemy 存根的版本；记录输出时一并记录版本。

## 测试计划

- 修改前跑 `make typecheck`（记录错误数），修改后为零。
- `make check`；如果改到 `gravity` 或 `store` 的代码路径，再跑 `make test-all`。

## 结果

## 待决问题

- 用 `uv.lock` 里的版本和用户的 conda Python，错误数是否一样？钩子运行的是
  `uv run mypy`。
