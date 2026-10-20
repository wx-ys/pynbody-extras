# BUG-0005: 原生扩展是否可用由 find_spec 判断，而不是实际加载 pynbodyext._native

## 背景

来自 PR #25（BUG-0003）上一条合并时未处理的审查评论。
`pynbodyext/util/deps.py` 中：

```python
GRAVITY_NATIVE_AVAILABLE = module_available("pynbodyext._native")  # find_spec(...) is not None
```

`find_spec` 只检查模块文件是否存在。如果 `_native*.so` 存在但加载不了（架构
不对、缺少 `libomp.dylib`、ABI 不匹配），结果仍然是 `True`。于是：

- `pynbodyext/gravity/__init__.py` 会导入 `.base`，其中的
  `from pynbodyext._native import ...` 报错，`import pynbodyext.gravity`
  直接失败，而不是按现有逻辑发出 `ImportWarning` 后降级；
- `bins["vcirc"]` 通过 BUG-0003 加的检查，然后在导入 gravity 时失败。

`_sph_scatter.native_pair_builder()` 的做法是对的：先尝试导入，遇到
`ImportError` 就回退。

## 目标

“可用”的意思是“能成功导入”。损坏的扩展和缺失的扩展表现一致，并保留失败原因
供错误信息使用。扩展不可用时，导入 `pynbodyext.gravity` 会发出默认可见的警告
（同时回答 BUG-0004 的第二个待决问题）。

## 实际行为

尚未在真实机器上观察到；由代码推断（见背景）。

## 期望行为

- `_native` 存在但损坏时，`GRAVITY_NATIVE_AVAILABLE` 为 `False`，
  `import pynbodyext.gravity` 成功并发出 `ImportWarning`，`bins["vcirc"]`
  抛出 BUG-0003 的错误，并附带原始的导入错误。
- 扩展正常或缺失时，行为不变。

## 复现方式

```python
# 模拟损坏的扩展：spec 存在但导入失败。
import sys, importlib.util, types
# 例如 monkeypatch importlib.util.find_spec，使其为 "pynbodyext._native"
# 返回一个 spec，并设置 sys.modules["pynbodyext._native"] = None，
# 然后 reload pynbodyext.util.deps 和 pynbodyext.gravity。
```

## 非目标

- 修复 macOS 构建（BUG-0004）。
- 改变哪些功能使用这个扩展。

## 验收标准

- [x] `deps` 通过尝试 `import pynbodyext._native`（只做一次并缓存）判断原生
      扩展是否可用，而不是用 `find_spec`。
- [x] 保留导入错误（例如 `deps.GRAVITY_NATIVE_ERROR`），并附到 `vcirc` 和
      `pynbodyext.gravity` 的错误信息里。（`deps.native_import_error()`、
      `deps.native_unavailable_reason()`。）
- [ ] 有一个测试模拟“存在但无法加载”的扩展，检查 `import pynbodyext.gravity`
      成功、`bins["vcirc"]` 抛出清楚的 `ImportError`；在任何机器上都会运行。
- [ ] `tests/test_package_layers_contract.py` 仍然通过（在 `util.deps` 中导入
      `_native` 不能破坏分层规则）。
- [ ] 有扩展和没有扩展时 `make check` 都通过。

## 设计说明

- `util/deps.py` 很早就被导入；在那里真正导入 `_native` 会改变导入开销和分层。
  备选：把判断做成惰性的（带缓存的函数 `gravity_native_available()`），
  `GRAVITY_NATIVE_AVAILABLE` 保留为兼容别名。
- BUG-0003 的测试 monkeypatch 了 `deps.GRAVITY_NATIVE_AVAILABLE`；要么保持它
  可用，要么随需求一起更新测试。
- `_native` 里还有 SPH 散布引擎；可考虑用一个共享的辅助函数，让 gravity 和
  散布对“是否可用”的判断一致。

## 决定

- `util/deps.py`：`native_import_error()` 导入一次 `pynbodyext._native`（结果
  缓存），返回 `ImportError` 或 `None`；
  `GRAVITY_NATIVE_AVAILABLE = native_import_error() is None`（仍是模块属性，
  BUG-0003 的 monkeypatch 继续有效）。在 `deps` 导入时就判断：只有 `gravity`、
  `plot.image.adaptive` 和 `_vcirc` 内部会导入 `deps`，加载扩展的开销也很小。
  `native_unavailable_reason()` 区分“未编译”（`pynbodyext._native` 的
  `ModuleNotFoundError`）和“存在但加载失败（<错误>）”，并说明如何重新编译。
- 新增 `deps.NativeExtensionWarning(UserWarning)`，默认可见。扩展不可用时，
  `pynbodyext.gravity` 在导入时发出这个警告，取代原来默认隐藏的
  `ImportWarning`。可用 `warnings.filterwarnings("ignore",
  category=NativeExtensionWarning)` 关闭。
- `bins["vcirc"]`：`ImportError` 的信息末尾附上原因，并用 `from` 链接到原始的
  导入错误。
- `import pynbodyext` 本身不发警告：从不使用 gravity 或 `vcirc` 的用户不应看到
  它。`_sph_scatter.native_pair_builder()` 保留它自己的静默回退。

## 测试计划

- 红：用伪造的 spec 加失败的导入写一个测试，显示
  `GRAVITY_NATIVE_AVAILABLE is True` 且 `import pynbodyext.gravity` 报错。
- 绿：改动后同一个测试通过；`make check`。

## 结果

## 待决问题

- 在 `deps` 导入时多做一次导入可以接受吗，还是应该做成惰性判断？
