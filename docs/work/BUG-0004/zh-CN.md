# BUG-0004: macOS 上没有编译出 pynbodyext._native；可选构建悄悄失败

## 背景

在用户的 Mac（Apple Silicon，Python 3.11.7）上，源码树和 `.venv` 里都没有
`_native*.so`，因此 `GRAVITY_NATIVE_AVAILABLE` 为 `False`：gravity 层不可用，
`bins["vcirc"]` 报错（BUG-0003），SPH 散布退回 NumPy 实现。整个过程没有任何
明显提示：`setup.py` 用 `optional=True` 声明这个扩展，编译失败只会打印一条
警告。

`setup.py` 在 `extra_compile_args` 和 `extra_link_args` 里都加了 `-fopenmp`。
Apple clang 在没有 `-Xpreprocessor` 和 `libomp` 运行库时不接受 `-fopenmp`，
所以那里的编译很可能失败。在看到编译器输出之前，这只是推测。

## 目标

在 macOS 上运行 `make setup`（或 `pip install -e .`）时，要么编译出
`pynbodyext._native`，要么以用户不会错过的方式失败或警告，并给出有文档的
修复方法。

## 实际行为

```text
$ .venv/bin/python -c "from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE as A; print(A)"
False
```

编译器的具体报错还没有记录（见复现方式）。

## 期望行为

- 在使用 Apple clang 的 macOS 上能编译出扩展：有 `libomp` 时启用 OpenMP，没有
  时串行编译；或者构建直接停止，并说明需要安装什么。
- README 写明 macOS 的构建要求。

## 复现方式

```bash
.venv/bin/python setup.py build_ext --inplace 2>&1 | tail -40
clang++ --version
brew list libomp 2>/dev/null && echo "libomp present"
```

## 非目标

- 修改 C++ 算法。
- Python 端的可用性判断（BUG-0005）。

## 验收标准

- [ ] 在本条目中记录 macOS 上的编译报错，并确认（或更正）原因。
- [ ] 在 macOS 上，`make setup` 能产出可导入的 `pynbodyext._native`（启用或
      不启用 OpenMP 均可），或打印明确、可操作的错误。
- [ ] Linux 构建保留 `-fopenmp`，性能不变。
- [ ] 修复后在 macOS 上 `make cpp-test` 和 `tests/test_gravity.py` 通过。
- [ ] README / README.zh-CN 写明 macOS 的构建要求。

## 设计说明

- 可选方案：在 `setup.py` 中检测 OpenMP 支持（试编译一个很小的 OpenMP 程序），
  只在可用时加上相关参数；macOS 上使用 `-Xpreprocessor -fopenmp -lomp`，并
  加上 Homebrew `libomp` 的 include/lib 路径；否则串行编译。
- 检查 C++ 代码在没有 OpenMP 时能否编译（`#pragma omp` 会被忽略，但
  `#include <omp.h>` 或 `omp_*` 调用需要用 `_OPENMP` 之类的宏保护）。
- `setup.py` 和 `cpp/` 的改动需要人工审核（见 `AGENTS.md`）。

## 测试计划

- 红：记录 macOS 上 `build_ext` 失败的输出。
- 绿：`build_ext --inplace` 成功；`import pynbodyext._native` 可用；
  `make cpp-test`；`pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`
  中 gravity 测试实际运行（而不是跳过）。

## 结果

## 待决问题

- macOS 上是否要求必须有 OpenMP（在文档中写 `brew install libomp`），还是可以
  接受串行回退？
- `optional=True` 的静默失败是否应在导入时变成可见的警告？（`pynbodyext.gravity`
  已经会发出 `ImportWarning`，但它默认不显示。）
