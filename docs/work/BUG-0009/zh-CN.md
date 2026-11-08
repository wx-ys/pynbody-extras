# BUG-0009: 防止不同类型的模拟身份共用结果存储键

## 背景

主线 3a30cff 已合入 codec、SPH 和中值滤波修复，继续增量 V1 规划。
_sig_text 将身份元素转成 str；Python tuple 相等规则还会合并 SampleTable 的
bool/int/float 身份。

## 目标

读取、覆盖、删除、列表及 SampleTable 共用带类型的身份契约。
不改变 CalculatorSignature 数值等价语义。

## 实际行为

RecordRef(("sim", 1), "a" * 64).key 与 ("sim", "1") 的键相同。
同一 calculator 保存两个输入时，一份结果被覆盖。

## 期望行为

不同类型身份可并存；相同身份仅复用自己的结果。

## 复现方式

执行 `.venv/bin/python -m pytest tests/test_sim_identity_store.py -q`。

## 非目标

用户明确无旧数据库或缓存，不做兼容迁移。不增加逻辑登记、不可变历史结果、ORM
或依赖，不改变 pickle 策略及计算签名。

## 验收标准

- [x] str/int/bool/float/None/嵌套 tuple 的不同身份具有不同键。
- [x] 内存和 SQLite 的值、列表、读取、删除及刷新独立，重开后保留身份类型。
- [x] SampleTable 的 add/from_store/save/load/where 保留不同类型的行。
- [x] 不支持的值及非有限浮点明确报错。
- [x] 当前记录 schema 可读，未知或缺失版本拒绝。
- [x] 专项、make check、make test-all 通过并记录红绿证据。

## 设计说明

见 [ADR-006](../../adr/0006-typed-simulation-store-identity.md)。使用内建标量标签
和递归 tuple，浮点正负零归一；NumPy ID 由 provider 显式转换。ResultRecord 使用
v3，不做迁移。

## 测试计划

先增加公开行为回归，确认键碰撞及行合并失败，再实现共享叶子编码器，用于 store
及 table 身份边界，执行专项和完整测试。

## 结果

快速基线：pynbody 2.8.0 下 916 passed、1 skipped、3 deselected。
红灯：新增回归在实现前为 19 failed / 1 passed。
绿灯：身份/store/reuse/sample 专项为 80 passed / 1 skipped。
最终 make check：lint、mypy（111 文件）通过；快速测试 936 passed /
1 skipped / 3 deselected；文档校验通过（10 项）。make test-all：939 passed /
1 skipped。git diff --check 通过，没有弱化原有断言。

初次基线文档检查与本工作项创建同时发生，报告索引过期；重新生成后解决。
上述最终检查覆盖完成后的工作项，无剩余生产基线失败。

自审覆盖类型标签、嵌套身份、正负零、非法 ID、SQLite 重开、筛选/删除、
SampleTable 往返、记录版本拒绝及导入分层。草稿 PR 仍待人类架构及代码审查。
不可变历史结果版本留待下一阶段。

## 待决问题

本项无待决问题，用户已明确旧数据库兼容策略。
