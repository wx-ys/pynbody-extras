# 完整与轻量发行方案 / Full and lightweight distribution proposal

2026-11-09 · SPIKE-0001 · **用户已批准方案与候选名称，见 ADR-019；未发布**。
基线 `5df15dd`，Python 3.11.9，Linux x86_64；临时实验版本 `0.0.1`。
正式 `pyproject.toml`、`setup.py`、`uv.lock` 均未修改。

## 建议的具体选择

候选名 **`pynbodyext-core`（轻量代码包）+ `pynbodyext`（完整元包）**。
名称仅供审阅，未查询可用性、未保留、未上传 registry。
保留现有 Python import 路径 `pynbodyext.*`；core 独占全部代码、native 和资源。
core 的基础依赖保持 pynbody >=2.8.0 与 matplotlib，SQLAlchemy 为可选 store extra。
完整元包强制依赖同版本 core 和 SQLAlchemy >=2.0，不拥有 Python 包文件。
这里“完整”包含 ORM；image/powerbin 仍为可选能力。

正式实施时完整元包的 image/dev/bench/store extras 应转发到同版本 core 对应 extra，
避免旧安装命令失去含义；本次最小元数据实验只验证基础完整安装依赖，未验证 extra 转发。

备选是保持目前 optional-store 单包；其默认不带 ORM，不符合既有 ADR-005 的完整默认
安装目标，选择它须明确修改 ADR-005。extra 是追加依赖，不能把已经声明的必需 ORM
依赖移除；依据 [PyPA dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras)。
`--no-deps` 不作为轻量受支持安装方式。

## 文件归属

| 路径集合（覆盖全部 import 包路径） | 唯一发行归属 |
|---|---|
| `pynbodyext/*.py`、`core/**`、`properties/**`、`filters/**`、`transforms/**` | core |
| `pynbodyext/orm/**`（代码存在；使用 ORM 需 SQLAlchemy） | core |
| `pynbodyext/plot/**`、`gravity/**`、`util/**` | core |
| `pynbodyext/_native*.so` / 平台等价二进制 | core |
| `pynbodyext/core/calculate/display.css`、未来包内资源 | core |
| `pynbodyext_core-*.dist-info/**` | core |
| `pynbodyext-*.dist-info/**`，无 import 包目录 | full 元包 |

实际 core wheel 共 **119 个 `pynbodyext/` 文件**，含一个 native `.so` 和一个 CSS；
full wheel 为 **0 个 import 包文件**。ZIP 文件清单交集为空。
完整路径及 Requires-Dist 的实际清单在本地实验目录 `inventory.json`。
不采用两个发行包各携带一份 `pynbodyext/__init__.py` 或复制整个源码的方案。

## 安装与卸载矩阵

下表是由依赖元数据与唯一文件归属推导的预期行为，**并非已通过的干净环境认证**。
安装实验因离线缓存缺少依赖未完成，不能把本表标为实际安装通过。

| 操作 | 预期结果 | 本次证据 |
|---|---|---|
| 新环境安装 core | 计算、绘图基础依赖，无强制 SQLAlchemy | wheel Requires-Dist 已验证；实际安装未运行 |
| 新环境安装 full | 同版本 core + SQLAlchemy，ORM 可用 | 元数据已验证；实际离线解析失败 |
| 先 core 再 full | 同一 core 文件不被重复拥有 | wheel 归属交集为空；实际安装未验证 |
| core/full 同时安装 | core 是唯一 import 文件所有者 | wheel 归属交集为空；实际安装未验证 |
| 卸载 full | core 的源码/native/CSS 保留；无自动卸载依赖承诺 | 根据归属推导；实际卸载未运行 |
| 只卸载 core 而保留 full | full 的依赖被破坏，应通过 pip check 检出并重装 core | 预期错误状态，未实测 |
| 卸载二者（仅移除这两个项目） | 没有二者拥有的 import 文件遗留 | 待 TASK-0002 验证 |

由旧单包原地升级到元包时，旧 wheel RECORD 仍拥有 import 文件；安装器操作顺序可能
影响新 core 文件。正式迁移必须在隔离环境覆盖旧版本升级/卸载路径；不能只凭新 wheel
交集为空宣布升级安全。该发布行为与“无现存结果数据库迁移需求”无关。

## 实验命令与结果

临时目录：`/tmp/pynbody-packaging-spike/`；驱动脚本：
`/tmp/pynbody-packaging-spike.py`、`/tmp/pynbody-packaging-rebuild.py`。
原库 `.venv` 缺 setuptools；仅通过 PYTHONPATH 读取 uv 缓存中的
setuptools 84.0.0、wheel 0.48.0、pybind11 3.1.0；不全局安装、不联网。
缓存归档路径分别为：

```text
/home/yxi/.cache/uv/archive-v0/9vR3pzfvVJiXx02SMvsDd
/home/yxi/.cache/uv/archive-v0/UJ5GZvgQs9B5z5q_BTAa3
/home/yxi/.cache/uv/archive-v0/31ikUFBbeqMy--KZT75mB
```

实验准备：复制当前 `pynbodyext/`（排除现成 `.so`、pycache）、`cpp/`、
`pyproject.toml`、`setup.py`、README 和 LICENSE 到 original/core 临时目录。
只在 core 副本把 `[project].name` 改为 `pynbodyext-core`。
full 临时 pyproject 用 setuptools backend，`packages=[]`，固定版本 0.0.1，
依赖 `["pynbodyext-core==0.0.1", "sqlalchemy>=2.0"]`。没有修改 import 包内容。

通过 `.venv/bin/python -c 'import setuptools.build_meta as b; ...'` 运行：

| cwd / 动作 | 精确 backend 调用或命令 | 退出码 / 实际结果 |
|---|---|---|
| original 元数据 | `b.prepare_metadata_for_build_wheel('metadata')` | 0；SQLAlchemy 仅 store extra |
| core 构建 | `b.build_wheel('dist'); b.build_sdist('dist')` | 0；wheel 含实际新编译 Linux/OpenMP native 与 CSS |
| full 构建 | `b.build_wheel('dist'); b.build_sdist('dist')` | 0；纯 metadata wheel，无包文件 |
| 解压 core sdist 后重建 | `b.build_wheel('dist')` | 0，但 native 编译失败被 optional build 吸收，产物无 `.so` |
| 隔离环境创建 | `uv venv --python /home/yxi/Simulation/pynbody-extras/.venv/bin/python --no-project --offline --cache-dir /tmp/pynbody-packaging-spike/uv-cache /tmp/pynbody-packaging-spike/install-env` | 0 |
| 离线 full 安装 | `uv pip install --python /tmp/pynbody-packaging-spike/install-env/bin/python --offline --find-links /tmp/pynbody-packaging-spike/core/dist --find-links /tmp/pynbody-packaging-spike/full/dist pynbodyext` | 首次 2（缓存只读）；标准提权重试 1（缺 registry 依赖缓存） |

实际生成 `pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl` 与
`pynbodyext-0.0.1-py3-none-any.whl`，以及两份 sdist。
完整 stdout/退出码在 `commands.json`、`*-build.log`、`full-install.log`，
清单在 `inventory.json`，sdist 重建回执在 `sdist-rebuild.json`。
这些 `/tmp` 路径是本次原始日志，不作为永久构建依赖。

core wheel 的基础 Requires-Dist 为 pynbody >=2.8.0、matplotlib；SQLAlchemy 仅 store extra。
full wheel 的基础 Requires-Dist 为 pynbodyext-core ==0.0.1、SQLAlchemy >=2.0。
元数据已实证区分两条安装路径，不依靠绕过依赖解析。

## sdist 的实际缺陷与后续门槛

原有打包配置的临时副本 sdist 含 13 个 `cpp/` 源文件条目，**0 个 C++ 头文件**。
解压后编译报 `fatal error: gravity/octree.hpp: No such file or directory`；
由于扩展 `optional=True`，build_wheel 仍退出 0，却产出不含 native 的 wheel。
不能仅检查退出码。TASK-0002 必须显式纳入 `.hpp` 等必需头文件、断言二进制存在并测试
实际导入；头文件补充可采用正式 MANIFEST.in 或受支持的源码清单机制。
[setuptools 文件清单文档](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html)
解释了默认发现不能覆盖所有自定义头文件，需要显式清单的情形。

只做了 Linux/CPython 3.11 的临时源码目录构建。macOS、Windows、其他 Python、
无编译器/无 OpenMP、干净环境 ORM/轻量运行、卸载及从旧发行包升级均未验证。
未访问 registry 查询或发布名称；离线安装因依赖缓存不足失败。

## 批准后的实施要求

采用一个版本源；core/full 版本同步，full 对 core 精确等号依赖。更新正式 pyproject 与
uv.lock，锁文件应覆盖开发时的两个本地成员，而发布元数据保留可解析的发行依赖。
分别构建两份 sdist/wheel，禁止 core/full 重复拥有包路径，extra 转发要有元数据验证。
每次验证 sdist 重建 wheel 的 native/CSS 真实存在，不能把 optional build 静默失败算成功。
产物安装必须从仓库外运行 smoke checks，防止源码遮蔽已安装 wheel。

请评审：候选发行名称、全部代码归 core、full 强制 ORM、精确同版本绑定，以及保留
full extras 的转发合同。批准后才记录 focused ADR，转 TASK-0002 实施与安装矩阵。

English: Recommend a code-owning lightweight core plus a metadata-only full package
with mandatory ORM dependencies. Names are unreserved proposals. Temporary wheel
ownership and dependency metadata passed; isolated installation did not complete
offline. An sdist rebuild silently omitted native code because headers were absent.
Production packaging and TASK-0002 remain gated on human review and real artifact tests.

2026-11-09 approval: 用户明确批准该发行方案与候选名称，见 [ADR-019](../adr/0019-full-core-distribution-ownership.md)。文中待评审要求是实验时历史状态；TASK-0002 的决定前置条件现已满足，安装矩阵验证仍需实际执行。

## 后续临时安装实验补充（功能实施期间）

获得发行方案批准后，允许在 `/tmp` 隔离环境下载现有项目版本依赖。以
`current-constraints.txt` 固定 `.venv` 已用版本，显式使用
`--index-url https://pypi.org/simple`，临时 full 安装成功。随后 core 在另一个隔离环境
通过该来源缓存离线安装成功。原先默认来源不匹配时仍会报“缓存缺少依赖”，已通过
显式匹配来源解决。

从仓库外、移除 PYTHONPATH 后执行 `/tmp/pynbody-artifact-smoke.py full|light`：
full 成功进行 ORM/SQLite 计算、保存和 fetch，科学值为 6；light 确认不存在 SQLAlchemy，
纯计算值为 6。两者均读取到 core CSS 并导入实际 native。日志见
`full-install-network.log`、`light-install.log`、`full-smoke.log`、`light-smoke.log`。
这些仍是先前临时构建的原型 wheel，不包含本次后续全部功能，不是正式安装矩阵验收；
TASK-0002 必须在全部功能代码完成后重建当前源码产物并重新验证。
