# FEAT-0009: 登记有目录命名空间的 halo 与 subhalo 元数据

## 背景与目标

区分目录科学身份、快照视图身份以及物理 host 层级。

依据：[修订设计](../../design/v1-2026-11-09.md)、[审查](../../design/review-2026-11-09.md)。
前置工作：无新增工作项依赖；从最新主分支开始。

用户已于 2026-11-09 明确批准本合同，决定记录于 [ADR-015](../../adr/0015-catalogue-object-metadata.md)。在已验证本地前置变更之上顺序实施，不假称前项已合并。

## 范围与合同

按 §6.2 只实现元数据。Catalogue(root_input_id,catalogue_id,finder,revision,loader_id) 用包含 null revision 的类型化规范身份键唯一。CatalogueObject 含 catalogue 外键、kind、typed_object_key、可空 host_id。repository 提供幂等 register/get/list，返回脱离 session 后可读的直接关系。编号接受内建 string/int/bool/有限 float，经 sim_identity_key 区分类型。host 必须是同 catalogue 的 halo；仅 subhalo 可有 host。根须是经校验的 snapshot DataObject。不改 provider 签名、DataObject.parent、run_input。

repository 方法：register_catalogue(root_input_id, catalogue_id, *, finder, revision=None, loader_id="default") -> Catalogue；get_catalogue(catalogue_pk)；catalogues(root_input_id)；register_catalogue_object(catalogue_pk, object_key, *, kind="halo", host_id=None) -> CatalogueObject；get_catalogue_object(object_pk)；catalogue_objects(catalogue_pk, *, kind=None, host_id=None)。校验非空字符串标签；目录身份含根类型化身份及四个标签，finder/revision/loader 改变产生另一目录。列表按主键排序以保证确定性。

## 文件范围

- `pynbodyext/orm/models.py`
- `pynbodyext/orm/catalogues.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_catalogues.py`
- `tests/test_orm_associations.py`（仅已批准的元数据表数量预期）
- `CONTEXT.md`

已批准的两个新模型会将元数据表数量从 7 增至 9。修改 test_orm_associations.py 时保留“不包含载荷列”的断言，仅更新已批准元数据表的预期集合/数量。这是显式 schema 扩展，不删除、跳过或压制失败。

## 验收标准

- [x] 两个目录内相同 halo 编号共存；revision/loader 区分身份。
- [x] 1、"1"、True、1.0 不合并；null revision 重复幂等或被约束拒绝。
- [x] 未知或跨目录 host、halo 带 host、目录父对象非根时原子失败。
- [x] SQLite 重开/join 恢复层级；无快照加载和载荷写入。

## 执行步骤与验证

- [x] 阅读 AGENTS.md、CONTEXT.md、本项双语 spec/metadata/handoff 和上述测试。
- [x] 每条验收转为行为测试；先运行，记录预期缺少 API/行为的红灯，不接受导入环境损坏作为红灯。
- [x] 只在文件范围内实施最小改动；若需变更核心合同，先记录设计决定。
- [x] 重跑专项，再运行 make check、make test-all；报告准确计数及未验证范围。
- [x] 更新两种语言、meta.json 和 handoff，保留红绿证据。
- [ ] 父任务按模板整理单独 PR。

```bash
.venv/bin/python -m pytest tests/test_orm_catalogues.py tests/test_orm_associations.py tests/test_package_layers_contract.py -q
```

## 本地实施证据

Catalogue 通过 sim_identity_key 把根逻辑签名及全部目录标签组成规范身份，包含
None revision。对象按 catalogue、kind、类型化 key 唯一；只接受精确内建
string/int/bool/有限 float，浮点正负零等价。目录 loader/revision 标签仍是元数据，
不改变或绑定快照 provider。

相同元数据登记幂等；不同 host（包括 None 与有 host）报 ValueError，不自动改挂。
未知 catalogue/host/get ID 报 KeyError，根/host/kind/标签冲突报 ValueError，
不支持的 key 类型报 TypeError，非有限浮点报 ValueError。允许无已知 host 的
subhalo。列表按主键排序，None 表示省略筛选，无匹配时返回空列表。

脱离 session 的 catalogue 预载根、根 parent 和 timestep/simulation；对象还预载
catalogue/根层级及标量 host。集合和更深 host 关系需调用方 session。数据库约束
保障唯一键、外键及基础 kind/host 规则；同目录 halo host 和逻辑根层级由 repository
方法校验。直接 SQL 写入由调用方负责。

红灯：**69 个新增用例均因缺少 register_catalogue 失败**
（/tmp/feat0009-red.log）；绿灯：**69 passed**，4.43s
（/tmp/feat0009-green.log）。目录、关联、加载、类型身份和包分层专项共
**163 passed**，14.10s（/tmp/feat0009-focused.log）。元数据边界测试绑定绝不能
调用的 loader，并确认结果引用、recipe 和 run 为空；SQLite 重开及映射 root/host
join 能恢复层级。


FEAT-0009 源码冻结检查点的完整验证：`make check` 的 lint、mypy（**121 个源码文件**）、
**1424 passed、2 skipped、4 deselected**（44.40s）及文档检查（27 个工作项）通过。
`make test-all`：**1428 passed、2 skipped**，45.16s；两个跳过均需要可选 pandas。
日志：`/tmp/feat0009-check.log`、`/tmp/feat0009-all.log`。独立规格审查和质量审查
均无发现。本地状态为 review，不声称已合并、提交、推送、开 PR 或发布。
本轮验证 SQLite/Linux；其他 SQL 方言/平台及 catalogue 加载不在本增量内。
未加迁移或依赖；保留“不包含载荷列”断言，仅将已批准的元数据表数量由 7 改为 9。

## 交接边界

不顺带重写 DAG、不新增数据库迁移/自动 GC、不发布 release。一个工作项一个可审阅改动；后续任务先验证本地前项，不声称远端合并。具体提交/推送权限沿用届时用户授权，不从此模板推断。
