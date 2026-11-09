# FEAT-0007: 编码显式的本地外部文件结果引用

## 背景与目标

让结果明确描述外部文件，并区分路径描述符版本与文件内容版本。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：BUG-0010, BUG-0011

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-013](../../adr/0013-local-external-file-references.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

实现设计 §5.3 的冻结 ExternalFileRef 和版本化 codec tag。要求本地绝对 URI/path、非空 media_type、可空非负整数 size_bytes、可空小写 64 位 sha256。拒绝未知字段/schema、bool 大小、远端和相对路径。显式 inspect(verify_checksum=False) 返回 available/missing/changed 及原因，权限错误传播。编解码不做 IO；文件均归用户管理，不复制/删除或自动转存。

## 文件范围

- `pynbodyext/core/calculate/store/external.py`
- `pynbodyext/core/calculate/store/codec.py`
- `pynbodyext/core/calculate/store/__init__.py`
- `tests/test_external_file_refs.py`
- `CONTEXT.md`

## 验收标准

- [x] 嵌套引用经 JSON、内存与 SQL 重开恢复完全一致描述符。
- [x] 文件不存在时仍能构造/解码；显式检查报告 missing。
- [x] 大小或校验值变化报告 changed；没有校验值不声称内容相同。
- [x] 刷新保留旧描述符但不能恢复被覆盖的外部文件，文档明确此限制。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [ ] 更新两种语言、meta.json 和 handoff，按模板整理单独 PR，保留红绿证据。

```bash
.venv/bin/python -m pytest tests/test_external_file_refs.py tests/test_value_codec.py tests/test_result_versions.py -q
```

## 已实现接口与文件所有权

从 `pynbodyext.core.calculate.store` 导入 `ExternalFileRef` 和
`ExternalFileInspection`。冻结描述符保留调用者原始文本；显式 codec tag 为
`external_file_ref_v1`，描述符 schema 为
`pynbodyext.result.external-file-ref/v1`。JSON 严格要求
schema/uri/media_type/size_bytes/sha256 五个字段，可选值也显式写 null；拒绝未知或重复字段。
仅接受本地绝对路径，以及 authority 为空或 localhost 的绝对 file URI；
拒绝 URI 查询/片段、非法转义和非本地 authority。

```python
from pynbodyext.core.calculate.store import ExternalFileRef, ValueCodec

ref = ExternalFileRef("/data/results/map-v1.png", "image/png")
restored = ValueCodec().decode(ValueCodec().encode(ref))  # 不访问文件
assert restored == ref
observation = restored.inspect(verify_checksum=True)
# 不存在时为 missing/not_found；无 sha256 时为 available/checksum_not_supplied。
# checksum_verified 仅在显式请求且摘要匹配时为 True。
```

检查结果包含 status/reason/checksum_verified。大小不符返回 changed；显式请求的
SHA256 不符也返回 changed，包括等长覆盖。默认 available 仅表示 stat 观察结果，
不证明文件可读或内容一致。PermissionError 等访问错误原样传播。文件不存在也能
构造、序列化和保存描述符，只有显式检查才观察文件。符号链接沿用普通 stat/open
语义，不锁定并发外部写入。

ResultStore 版本只固定描述符。刷新后旧固定结果仍保留原大小/摘要，但取回它无法
恢复被覆盖的文件字节。调用者拥有文件，并自行维护不同的版本路径以重现历史内容；
不复制、不删除、不自动外置载荷。

红灯证据：缺少公共接口（1 个断言失败）、缺少检查方法（13 个失败）、缺少显式
codec tag（5 个失败、12 个通过），以及 URI 解析器静默清理空白（4 个失败、21 个通过）。
专项绿灯：260 passed；日志为 `/tmp/feat0007-{red,inspect-red,codec-red,uri-red,focused}.log`。

完整验证：`make check` 通过 lint、mypy（119 个源码文件）、1316 个测试
（2 skipped、4 deselected）及文档校验（27 项）；`make test-all` 为
1320 passed、2 skipped。日志为 `/tmp/feat0007-check.log` 和
`/tmp/feat0007-test-all.log`。只验证 Linux/SQLite。状态为 review，
前置项经本地验证，不声称已合并。双语文档、metadata 和 handoff 均已更新，
PR 发布留给主代理流程。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先检查前项已合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。
