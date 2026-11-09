# FEAT-0011: 从版本化 JSON 清单重放根输入任务

## 背景与目标

在明确的同一个元数据 repository 内，利用既有签名和 run_input 提供小型任务描述。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：FEAT-0006

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-017](../../adr/0017-root-input-job-manifests.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

按 §7 提供冻结 JobSpec：schema、calculator_signature_text、input_id、location_id、store_name、reuse。公共 API 为 from_calculator(...)、to_json()、from_json()、execute(repo)。严格内建字段类型，ID 为正整数且拒绝 bool，store_name 非空。只支持默认 RunOptions。拒绝不稳定/不可重建签名、未知字段/schema；仅从已可用注册类型重建计算器。execute 调 repo.run_input；绑定由执行时提供，不落盘。仅根输入，selection job 留后续独立 schema。不暴露 _EvaluationPlan、不建执行器。

## 文件范围

- `pynbodyext/orm/jobs.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_jobs.py`
- `CONTEXT.md`

## 验收标准

- [x] 新 Python 进程重开 SQLite、绑定 loader/store 后取得相同科学值与签名。
- [x] 第二次执行有新的缓存命中 Run 和同一 pinned ref；reuse=False 新增版本。
- [x] 缺 loader/store/type 或非法 schema 在定义的预检阶段失败且不计算。
- [x] 清单不含凭据、callable、live snapshot、可执行 pickle、backend URL。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，保留红绿证据。
- [x] 父任务集成流程：获得发布授权后按模板整理单独 PR。

```bash
.venv/bin/python -m pytest tests/test_orm_jobs.py tests/test_orm_loading.py tests/test_signature_roundtrip.py -q
```

## 本地实施与验证

已实现冻结 `JobSpec`，schema 为 `pynbodyext.orm.job/v1`，严格包含批准的六个字段。`from_calculator(calculator, input_id, *, store_name, location_id=None, reuse=True)` 复制签名文本；`from_json` 拒绝缺失/未知/重复字段、非有限 JSON 常量、未知 schema 和非严格内建字段类型。签名转换前检查整个 live graph 的默认 RunOptions，重建后再次检查，避免静默丢弃原选项。

任何签名构造器执行前，先遍历整个嵌套 payload，只接受明确注册的计算器名称/别名、与注册类相同的签名版本及已加载 Enum 类型。项目代码须在读取/执行前显式 import/注册所需类型；仅有 class path、即便可 import 的自定义类也明确拒绝。重建后递归规范化注册别名、保留数值等价并核对 recipe identity，拒绝构造器/重建器改变参数。这一严格 job recipe/version 校验不改变通用 registry/decoder 语义，不自动导入模块/安装插件，也不做全局 monkeypatch。注册构造器及重建器仍是可信应用代码。

`execute(repo)` 重建并重新检查签名，拒绝所有非根输入（包括有效的 FEAT-0010 selection），然后原样委托 `run_input`。store/loader/location 预检不建 Run、不执行科学计算；后续加载/计算错误沿用既有运行失败语义。不序列化绑定、backend URL、callable、live snapshot 或 pickle；科学参数字符串原样保留，不自动清除用户写入的秘密。ID 仅属于调用者明确指定的元数据 repository。

TDD 红灯证据：缺失 `orm.JobSpec` API 导致 **1 failed**，见 `/tmp/feat0011-red.log`。新模块首次绿灯 **55 passed**，见 `/tmp/feat0011-jobs.log`；之后补充嵌套签名 envelope flag 和已加载非 Enum callable 绕过两项安全回归。真实子进程重开 SQLite，显式 import ParamSum 并绑定 loader/store：原值 **6**，即便新加载质量和为 **30**，缓存命中仍返回 **6**，有新 Run、相同签名/pinned ref；`reuse=False` 得 **30** 和新版本，旧 Run 仍 fetch **6**。

审查发现 recipe drift：通用 decoder 只警告旧版本，重建参数也可能改变。根/嵌套版本和 dataclass/generic 构造器四项回归先 **4 red → green**，见 `/tmp/feat0011-drift-red.log`、`/tmp/feat0011-drift-green.log`；更新后新 jobs 模块 **62 passed**。修正后重新运行最终源码专项/check/all；此前成功计数明确保存在 `*-pre-drift.log`，不作为最终证据。源码/测试已重新冻结供独立复审。本项在已验证本地 FEAT-0006/0009/0010 之上顺序实施，不声称远程合并或发布。SQLite/Linux 验证不代表其他 SQL 方言/平台已验证。

别名修正前验证：专项 **226 passed**，22.94s（`/tmp/feat0011-focused-pre-alias.log`）；`make check` lint/mypy/docs 及 **1572 passed、2 skipped、5 deselected**（62.95s）全通过（`/tmp/feat0011-check-pre-alias.log`）；完整套件 **1577 passed、2 skipped**，64.29s（`/tmp/feat0011-all-pre-alias.log`）。这些成功日志早于下述 alias-only 修正，不作为最终源码证据。两项跳过均需可选 pandas。

父任务最终探针发现：类只声明明确注册的 alias、未声明 `signature_name` 时，重建签名使用 canonical class path，原检查错误拒绝这一有效别名。根/嵌套两项回归复现 **2 red**（`/tmp/feat0011-alias-red.log`）。现在只将已完整预检的注册类型及其 canonical name 保存在 job 本地映射中，供重建检查和 identity 规范化使用。初始 payload 仍须明确 registry 成员，不向全局注册 canonical name，不改变 class-path-only/from_calculator 限制。根/嵌套 alias-only job 均可 JSON 往返并真实执行得 **18**；未知返回类型和参数漂移仍拒绝。新模块绿灯 **65 passed**（`/tmp/feat0011-alias-green.log`）。源码/测试再次冻结供独立复审。状态仍为 review；无 Git/远程操作。

别名修正后最终验证：专项 **229 passed**，24.15s（`/tmp/feat0011-focused.log`）；`make check` lint/mypy（123 个源码文件）/docs 通过，**1575 passed、2 skipped、5 deselected**，67.25s（`/tmp/feat0011-check.log`）；`make test-all` **1580 passed、2 skipped**，67.09s（`/tmp/feat0011-all.log`）。两项跳过均需可选 pandas。最终文档检查通过（`/tmp/feat0011-docs.log`）；独立审查/发布由父任务继续。

父任务已确认别名修正后的规范与质量独立复审均通过；审查者专项 **10 passed、55 deselected**。父任务独立核对了上述最终专项/check/all 证据，继续顺序工作。状态仍为 review，不声称 commit/PR/远程合并。最终验证与复审期间源码/测试始终冻结。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；检查本地前项验证结果，不声称远程合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。

## 集成结果 — 2026-11-09

已通过 [FEAT-0011 PR](https://github.com/wx-ys/pynbody-extras/pull/51) 合并，merge commit `a7d6f6a053ed41a439a55ce3e88e3c6b2f2f46cb`。当前状态为 done，取代上文历史记录中的仅本地/review 状态。逐项集成验证见 handoff.md 与[集成记录](../../design/integration-2026-11-09.md)。未发布 release。
