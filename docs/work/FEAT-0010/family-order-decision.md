# FEAT-0010 family 顺序边界：已批准的具体修订

2026-11-09。原合同允许根相对索引任意有序、保留重复；真实多 family 回归发现
当前 pynbody 2.8.0 的 IndexedSubSnap 无法表示任意跨 family 重排。

## 已验证事实

根为 `pynbody.new(dm=2, star=2)`、family 顺序 dm 后 star 时：

- `[1, 0, 1, 3, 2, 3]`：同一 family 块内允许任意顺序及重复，保持输入顺序。
- `[3, 1, 3, 0]`：star/dm 交错；上游拒绝构造 view。
- `[3, 1]`：family 块相对根顺序颠倒，上游同样拒绝。
- `allow_family_sort=True` 会改变请求索引顺序，不能用于实现原合同。

已直接读取当前安装的 `pynbody.snapshot.subsnap.IndexingViewMixin.__init__`：
它要求所选粒子的 family 编号单调不减；每个 family 通过连续 slice 表达。
这也与[上游公开源码](https://pynbody.readthedocs.io/v2.3.2/_modules/pynbody/snapshot/subsnap.html)
记录的结构约束一致；实际裁定依据为本环境 2.8.0 源码与真实多 family 回归。
现有 Slice/Indexed/Family view 均不提供符合原顺序的任意跨 family 适配入口。

## 已批准修订 A：首版遵循现有 family 块边界

继续保存并逐项重放原始索引顺序及重复；要求所选 family 块顺序与根一致。
跨块重排/交错在登记时明确 ValueError，不排序、不改变 family 身份、不保存记录。
既有已登记定义若重放时不再满足约束，在已创建 Run 的加载/选择阶段记录 failed，
不执行科学分析计算器。family/filter、严格描述符、计数/digest/revision 验证、
Run.selection_id、根位置归属及准确回执恢复合同保持原定验收。

这项选择限制原合同。用户已于 2026-11-09 明确批准：“首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”。先更新双语 spec、ADR-016、计划与交接，再更新对应测试和实现。
任意跨 family 顺序保留为未来有独立设计/科学语义验证的视图适配工作。

## 备选 B：保持原合同，先设计新的视图适配器

另行设计能保留真实 family 身份、任意索引顺序与重复的 pynbody 兼容视图，
明确 family 专属数组、派生量、根索引、写回/变换与缓存行为，再进入实施。
不能把全部粒子伪装成同一 family，也不能用复制后改身份的方法冒充原生 view。
这会扩大 FEAT-0010 的架构范围，需审阅新方案；其余不依赖选择的功能仍可继续。

当前状态：方案 A 已获用户明确批准并进入实施；对应验收改为显式拒绝不受支持的跨块重排，不排序、不伪造 family。通用视图适配器留待独立架构工作。
