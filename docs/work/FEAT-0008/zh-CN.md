# FEAT-0008: 在现有 DAG 上增加消费结果值的后处理器

## 背景与目标

提供 process(value) 编写接口，沿用同一执行器且不重复运行上游。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：BUG-0010, BUG-0011

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-014](../../adr/0014-dag-value-postprocessors.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

实现设计 §6.1：PostprocessorBase 复用 property/runtime 合同，以现有 Param 机制声明一个 source calculator 依赖。通用适配层解析 source 后调用 process(value)。签名含 source 与 dataclass 参数，遵守 signature_version 和可重建标志。首版只接受 Calculator source，明确拒绝裸 Result/RecordRef。经 calculate facade 导出。不加拟合依赖、全局注册或独立调度。

首版 processor 只有一个动态字段 source，其余配置为静态参数。适配层实现 calculate(sim, params) 并调用 process(params.source)，科学子类只覆盖 process。不得静默接受额外动态配置，除非本增量同时评审并测试这些求值参数的传递合同。

适配层异常路径只调用自身已成功应用的绑定变换的清理回调，遵守 scope/transform 的 `NEVER`
及不可回退 handle。清理失败只记录日志，不替换原始异常。祖先继承的 scope 和通用
property 成功路径保持现有归属；进入适配层前的 transform/filter 应用失败不属于此
狭义 process 异常清理路径。实施边界记录于 ADR-014。
执行器沿用最后记录 phase 的报告规则，清理可能使错误 phase 显示为 `revert`；
原始异常类型、消息和 traceback 保留。
清理委托现有 transform 回调：pynbody 原生复合 handle 可能连带回退更早的变换，
保留的原生 `NEVER` 步骤也可能阻止更早 handle 的清理。沿用这些 handle 语义；
清理失败记录日志并保留原始 process 异常。

## 文件范围

- `pynbodyext/core/calculate/nodes/postprocessors.py`
- `pynbodyext/core/calculate/nodes/__init__.py`
- `pynbodyext/core/calculate/__init__.py`
- `pynbodyext/calculate.py`（实际的短路径公共 facade）
- `tests/test_postprocessors.py`
- `CONTEXT.md`

## 验收标准

- [x] 模块级 ScaleValues 示例缩放带单位数组，签名可往返。
- [x] 一个 Pipeline 中两个下游共享的计数上游只求值一次。
- [x] 参数/source/版本变化改变签名；stable 与 constructible 仍相互独立。
- [x] process 异常沿用 collect/raise、清理变换输入且不保存失败输出；缓存成功返回固定 ref。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，按模板整理单独 PR，保留红绿证据。

```bash
.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_calculate_contract.py tests/test_signature_roundtrip.py tests/test_layering_contract.py -q
```

## 已实施 API 与验证证据

```python
from pynbodyext.calculate import PostprocessorBase

@PostprocessorBase.dataclass
class ScaleValues(PostprocessorBase[object]):
    factor: float = 2.0

    def process(self, value):
        return value * self.factor

scaled = ScaleValues(3.0, source=upstream)
result = scaled.run(sim)
```

`source` 继承自基类，是必填 keyword-only 参数，子类带默认值的静态字段仍可作为
普通位置参数。构造时拒绝裸值、Result、RecordRef、额外动态字段/spec、失去签名或
动态身份的 source，以及 source 单位转换元数据。覆盖 `calculate` 在签名类型登记前
即报错。复用通用 dataclass/property 签名路径；大静态数组可稳定但不可重建，与包含
不稳定 source callable 的情形区分。沿用现有 recipe 签名记忆化规则；版本变化使用
新的 recipe 实例验证。

红灯：缺少短 facade API 导致预期的一次导入收集错误（`/tmp/feat0008-red.log`）。
异常作用域验收随后暴露通用 property 缺少清理路径；扩展清理测试为六失败、三十通过
（`/tmp/feat0008-cleanup-red.log`）。角色内包装层使其通过，未改通用执行器行为。
探索性原生回调测试根据现有复合 handle 语义纠正，不承诺逐回调回退或祖先效果隔离。

专项覆盖全部验收及签名、Param、运行时、异常、存储、固定引用、分层回归：
**164 通过**，用时 2.77 秒（`/tmp/feat0008-focused.log`）。其中 39 项后处理测试包含
实际 `collect_partial` Pipeline：失败分支清理后，带独立作用域的成功分支观测正确位置值，
失败 Pipeline 不被持久化。

完整验证：`make check` 通过 lint、mypy（120 源文件）、**1355 通过、2 跳过、
4 排除**（40.00 秒）及文档校验（27 项）。`make test-all` 为 **1359 通过、
2 跳过**（41.00 秒）。两项跳过依赖可选 pandas。日志：`/tmp/feat0008-check.log`、
`/tmp/feat0008-test-all.log`。双语文档、元数据及 handoff 已更新；状态为 review，
工作索引已重新生成。前置项仅已本地验证，不声称合并。未执行 Git、提交、推送、
PR 或 release 发布。本次验证 Linux/SQLite，不据此认证其他平台/SQL 方言或通用
成功作用域清理。PR 发布留给父任务流程。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 集成结果 — 2026-11-09

已通过 [FEAT-0008 PR](https://github.com/wx-ys/pynbody-extras/pull/48) 合并，merge commit `83ae4b5b3c363ef151f179265157c1d03f799cf4`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
