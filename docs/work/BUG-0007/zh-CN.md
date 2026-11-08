# BUG-0007: ValueCodec 保留 NumPy 标量与字典键的类型和值

## 背景

2026-11-08 设计审计在 ResultStore 的 codec 接口复现了数据丢失。
用户批准在新分支开始 V1 规划，push、PR 和 merge 由用户完成。
本项是 P0 的第一个可独立审查阶段。

## 目标

解码已有 NumPy 标量和字面量 tuple/None 键时保留类型和值，保持编码格式不变。

## 实际行为

NumPy 数值标量因解码器直接转换编码后的字典而抛出 TypeError。
False NumPy 布尔值变成 true；None 键变成字符串；tuple 键按逗号拆分，
丢失元素类型、嵌套和引号。不同键可能合并并覆盖值。

## 期望行为

经过 JSON 和 SQLite 往返后，保留标量 dtype 和受支持字典键的身份。

## 复现方式

```python
import numpy as np
from pynbodyext.core.calculate.store import ValueCodec
codec = ValueCodec()
codec.decode(codec.encode(np.int64(3)))
codec.decode(codec.encode({None: 1, "None": 2, (1, 2): 3, ("1", "2"): 4}))
```

## 非目标

不实施身份键迁移、不可变结果版本、ORM 表、新编码格式、任意对象扩展、
SPH 兼容调整或远端 GitHub 操作。

## 验收标准

- [x] NumPy 整数/无符号整数/浮点/布尔/字符串/bytes 标量通过 JSON 往返后，
  保留类型、dtype 和值，包括 NaN/Inf。
- [x] None 和字面量 tuple 键保留身份，包括单元素、嵌套及含逗号/引号的字符串，
  不同键不会合并。
- [x] 已有 NoneType/none 和字面量 tuple 键记录可读取。
- [x] 无效或非字面量 tuple 文本明确报错，不执行其中代码。
- [x] 编码格式与内容 hash 不变，hash 不依赖字典插入顺序。
- [x] SQLite 关闭再打开后，混合标量和键身份不变。
- [x] 执行专项测试、make check 和 make test-all，记录已有基线失败。

## 设计说明

决定：仅修正解码器，不改变 import 分层、ResultStore schema、calculator signature、
simulation key 或 pickle 策略。人类审查交给用户后续 PR，不实施架构迁移。

递归解码 np_scalar.value 后用 [()] 取得保留 dtype 的 NumPy 标量，不用 .item()。
U/S 字符串标量按已保存的 dtype 宽度恢复 NUL 补位。
None 键同时接受 NoneType 和 none 标签。tuple 文本使用 ast.literal_eval 安全解析，
要求结果是可 hash 的 tuple；无效文本统一抛出包含说明的 ValueError。禁止 eval。

本轮支持 Python 字面量 tuple（可嵌套）。(nan,)、(np.int64(1),) 等非字面量文本
会被拒绝；任意旧对象表示需另行设计带类型的键格式。
旧记录若已经错误解码并重新保存，丢失的类型无法自动恢复。

## 测试计划

使用 writing-plans / TDD 流程，在 codex/p0-value-codec-roundtrip 分支顺序实施。

- [x] 增加 tests/test_value_codec.py，并在 tests/test_sql_store.py 增加关闭重开测试，
  保留已有断言。
- [x] 执行 `.venv/bin/python -m pytest tests/test_value_codec.py tests/test_sql_store.py -q`，
  确认因标量 TypeError 和键不一致而失败。
- [x] 只修改 store/codec.py 的相关分支：NumPy 标量递归解码后使用 [()]；
  None 接受两个标签；tuple 安全解析并验证可 hash，将 SyntaxError/ValueError/TypeError
  统一为包含说明的 ValueError。
- [x] 执行 codec/store/sample 专项测试，再执行 make check 和 make test-all。
- [x] 审查 diff、同步双语和 handoff、状态改为 review，准备 PR 描述和用户操作命令。

## 结果

红灯：新 codec 测试和 SQL store 测试共 29 failed、9 passed。
绿灯：codec/store/reuse/sample 专项测试 95 passed、1 skipped。
`make check`：ruff 和 mypy 通过；快速测试 850 passed、3 failed、1 skipped、
3 deselected。`make test-all`：853 passed、3 failed、1 skipped。
三个失败均为已有 BUG-0002 的 SPH/pynbody 2.3.3 兼容失败，修复前快速基线为
814 passed、3 failed、1 skipped。没有移除或跳过失败测试。
make check 在 pytest 失败处停止，因此另外执行文档检查。

### 组合基线验证 — 2026-11-08

将 PR #34 的图像/SPH 基线及版本契约补强合入本分支后进行组合验证。
只有自动生成的工作项索引发生冲突，重新生成后完整保留八个工作项。
集成过程没有修改 codec 实现或已有数值断言。

`make check` 现已通过：**866 passed、1 skipped、3 deselected**，lint、mypy
及文档校验均通过。`make test-all`：**869 passed、1 skipped**。
环境为 Linux / Python 3.11.9 / pynbody 2.8.0。上方失败计数保留为旧依赖基线
下的历史证据。维护者现已明确授权执行 push、更新 PR 和合并。

### 合并验证 — 2026-11-08

已通过 [PR #33](https://github.com/wx-ys/pynbody-extras/pull/33) 合并
（提交 `88ae3b4`）。两批改动的组合验证通过：make check 为
866 passed / 1 skipped / 3 deselected；make test-all 为 869 passed / 1 skipped。
本地 main 已同步至 88ae3b4，与验证过的集成版本内容一致。
工作项保留至下次发布或 30 天后再归档。

## 待决问题

本项修复无待决问题；带类型的键格式演进另行规划。

独立审查发现尾部 NUL 会导致字符串 dtype 宽度丢失。新增四个回归先失败
（4 failed / 31 passed），按已保存的 dtype 恢复补位后通过。复核验证 35 个
codec 测试和另外八个字符串边界样例，无剩余发现。
