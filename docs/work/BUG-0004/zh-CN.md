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

编译器输出（2026-10-20，Apple clang，Python 3.11 来自 `/opt/anaconda3`）：

```text
$ uv run --with setuptools --with pybind11 python setup.py build_ext --inplace 2>&1 | tail -60
running build_ext
building 'pynbodyext._native' extension
clang++ ... -c cpp/bindings/module.cpp -o build/temp.macosx-11.1-arm64-cpython-311/cpp/bindings/module.o -fvisibility=hidden -g0 -std=c++17 -mmacosx-version-min=10.14 -O2 -fopenmp
clang++: error: unsupported option '-fopenmp'
warning: build_ext: building extension "pynbodyext._native" failed: Command '['clang++', ..., '-fopenmp']' returned non-zero exit status 1.
```

已确认：Apple clang 不接受 `-fopenmp`；由于扩展声明了 `optional=True`，
setuptools 把失败变成一条警告，安装出来的包里没有 `_native`。

## 期望行为

- 在使用 Apple clang 的 macOS 上能编译出扩展：有 `libomp` 时启用 OpenMP，没有
  时串行编译；或者构建直接停止，并说明需要安装什么。
- README 写明 macOS 的构建要求。

## 复现方式

项目的 `.venv`（由 uv 创建）里没有 `setuptools`，直接调用 `setup.py` 会报
`ModuleNotFoundError: No module named 'setuptools'`（2026-10-20 观察到）。用
`pyproject.toml` 中的构建依赖临时补上，不改动 `.venv`：

```bash
uv run --with setuptools --with pybind11 python setup.py build_ext --inplace 2>&1 | tail -60
clang++ --version
brew list libomp 2>/dev/null && echo "libomp present"
```

## 非目标

- 修改 C++ 算法。
- Python 端的可用性判断（BUG-0005）。

## 验收标准

- [x] 在本条目中记录 macOS 上的编译报错，并确认（或更正）原因。（已确认：
      `unsupported option '-fopenmp'`。）
- [x] 在 macOS 上，`make setup` 能产出可导入的 `pynbodyext._native`（启用或
      不启用 OpenMP 均可），或打印明确、可操作的错误。
      （已用 `setup.py build_ext --inplace` 验证：通过 libomp 启用 OpenMP。）
- [x] Linux 构建保留 `-fopenmp`，性能不变。
      （Linux/GCC 11：探测选中 `-fopenmp`，链接了 `libgomp`。）
- [x] 修复后在 macOS 上 `make cpp-test` 和 `tests/test_gravity.py` 通过。
- [x] README / README.zh-CN 写明 macOS 的构建要求。

## 设计说明

- 可选方案：在 `setup.py` 中检测 OpenMP 支持（试编译一个很小的 OpenMP 程序），
  只在可用时加上相关参数；macOS 上使用 `-Xpreprocessor -fopenmp -lomp`，并
  加上 Homebrew `libomp` 的 include/lib 路径；否则串行编译。
- 目前 C++ 代码在没有 OpenMP 时无法编译：`cpp/bindings/module.cpp` 和
  `cpp/image/scatter.cpp` 里的 `#include <omp.h>` 没有保护；`module.cpp` 调用了
  8 次 `omp_set_num_threads`，`scatter.cpp` 调用了 `omp_get_max_threads`。
  串行回退需要 `#ifdef _OPENMP` 保护（或在 `cpp/` 加一个小的兼容头文件）。
  `gravity/direct.cpp`（2 处）、`gravity/traversal.cpp`（4 处）和
  `image/scatter.cpp`（1 处）中的 `#pragma omp` 在没有 OpenMP 时会被忽略，
  无需改动。
- `setup.py` 和 `cpp/` 的改动需要人工审核（见 `AGENTS.md`）。

## 决定

构建所用的编译器能编译并链接 OpenMP 时就用 OpenMP，否则单线程构建，并给出
明显的警告（回答了第一个待决问题）。

- `setup.py`：新增 `BuildExt` 子类，构建前用一个很小的 OpenMP 程序探测编译器。
  macOS 上先试 Homebrew libomp（依次为 `LIBOMP_PREFIX`、`brew --prefix libomp`、
  `/opt/homebrew/opt/libomp`、`/usr/local/opt/libomp`），编译参数
  `-Xpreprocessor -fopenmp -I<prefix>/include`，链接参数
  `-L<prefix>/lib -lomp -Wl,-rpath,<prefix>/lib`；然后试普通的 `-fopenmp`。
  都不行就打印警告（macOS 上提示 `brew install libomp`），不带 OpenMP 构建。
  `PYNBODYEXT_OPENMP=0` 强制单线程构建。构建时总会打印实际使用的配置。
- `cpp/bindings/module.cpp`、`cpp/image/scatter.cpp`：`#include <omp.h>` 用
  `#ifdef _OPENMP` 保护；8 处 `omp_set_num_threads` 改为调用
  `set_omp_threads()`，没有 OpenMP 时什么也不做，`threads` 参数被忽略。
- `Makefile` 的 `cpp-test`：`OMP_FLAGS` 在 macOS 上用 Homebrew libomp，其他
  平台用 `-fopenmp`；`make cpp-test OMP_FLAGS=` 为单线程构建。
- README / README.zh-CN：写明构建要求、`brew install libomp`、
  `PYNBODYEXT_OPENMP`、`LIBOMP_PREFIX`。

## 测试计划

- 红：记录 macOS 上 `build_ext` 失败的输出。
- 绿：`build_ext --inplace` 成功；`import pynbodyext._native` 可用；
  `make cpp-test`；`pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`
  中 gravity 测试实际运行（而不是跳过）。

## 结果

## 待决问题

- macOS 上是否要求必须有 OpenMP（在文档中写 `brew install libomp`），还是可以
  接受串行回退？
  - 参考：用户的 Mac 上已经装了 Homebrew `libomp` 23.1.2
    （`/opt/homebrew/Cellar/libomp/23.1.2`，含 `lib/libomp.dylib`），所以
    “找到 libomp 就用 OpenMP，否则串行”的构建可以在这台机器上验证。Homebrew
    的 libomp 是 keg-only：需要显式传入它的 `include/` 和 `lib/` 路径
    （`brew --prefix libomp`），链接时用 `-lomp` 而不是 `-fopenmp`。
- `optional=True` 的静默失败是否应在导入时变成可见的警告？（`pynbodyext.gravity`
  已经会发出 `ImportWarning`，但它默认不显示。）
