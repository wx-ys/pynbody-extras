# FEAT-0004: 通过显式加载器绑定加载已登记快照

状态：设计草案；加载配置的持久化边界待用户审阅。

## 目标与范围

接续 FEAT-0003，打通“重开数据库 → 找到已登记根快照 → 选择 DataLocation 加载 →
执行现有 Calculator DAG → 固定结果引用”。当前 ORM 只登记位置，尚不能加载。
halo/subset 目前只有粒子索引摘要，无法重建选择，本阶段仅加载根快照。

保留 revision 可省略/手动刷新、身份与位置分离、元数据/载荷分开提交和显式恢复。
不兼容旧数据库/缓存；不增加依赖、执行器或导入层。

## 推荐的加载边界

AnalysisRepository 接受 loaders={loader_id: callable}，像命名 store 一样复制并
校验进程内映射。加载函数接受一个位置字符串，返回 pynbody 根 SimSnap。
参数在项目代码中预绑定（partial 或适配器）。数据库保存现有 loader_id；不存
Python 代码、导入路径、凭据或序列化函数。重开数据库需重新绑定等价加载器。
不隐式提供默认加载器；标准用法显式绑定 {"default": pynbody.load}。
加载含义改变时更换 loader_id；重新绑定不会检测语义变化或自动使缓存失效。

另一种方案是持久化带版本的声明式加载规格及参数，可提高配置可移植性，但需额外
定义允许的参数、编解码和重建协议。首版推荐沿用现有 loader_id 由用户管理的约定。

V1 加载本地绝对路径或快照前缀，不使用 isfile 判断：多文件快照前缀可能不是文件。
远端 URI/相对位置仍可登记为元数据，但加载预检拒绝。位置必须属于该输入；只有一条
时直接使用，没有则报错，多条时必须明确 location_id。不逐条探测或失败后自动换位置。
数据等价性仍是用户声明，不以此证明文件内容相同。

## 公共接口与数据流

- get_input(input_id)：预加载时间步/模拟、父对象和位置，可在 session 关闭后查看。
- load_input(input_id, *, location_id=None)：返回 LoadedInput，包含 SimSnap、选定
  DataObject/DataLocation 及可传入 sim_identity 的 ConfiguredSnapshotIdentity；不建 Run。
- run_input(calculator, input_id, *, store_name, location_id=None, reuse=True,
  options=None)：预检元数据和绑定，登记流程并提交 running，再加载并调用原计算入口；
  返回现有 RecordedRun。加载/计算期间不保持 ORM 事务。即使缓存命中也加载；跳过 IO
  的优化留待后续。
- Run 增加可空 input_location_id 外键，查询时预加载位置。run_input 记录所选位置；
  现有 run(calculator, sim, ...) 保持空值，因为快照不是 repository 选择位置加载的。
  loader_id 已在输入身份中。这里记录选用位置的关联，不复制可变元数据的历史内容。

加载前校验完整身份、kind、层级，必须是根快照；视图在调用加载器前拒绝。
返回值必须是根 SimSnap、有可用本地 filename，且无外部活动变换。加载器可能把前缀
规范为另一 filename；以返回的 filename 绑定原逻辑身份，并验证所得签名与登记完全
一致。不改写 SimSnap.filename，不为无效结果臆造稳定身份。
LoadedInput.sim_identity 沿用现有 subset/halo 与外部变换语义。

加载器是受信任的项目代码，可能执行 IO；不根据函数名称推断科学等价性。普通对象、
无文件身份的内存快照和返回的子视图均拒绝。

## 失败与恢复

未知输入/loader/store、非法层级、不支持的位置、多位置未选择及非法请求在预检
失败，不调用加载器、不建 Run。建 Run 后加载失败或返回值校验失败，记录 failed，
抛出带 run_id 和原异常链的 RunExecutionError。状态写入失败抛 RunPersistenceError，
保留可检查的 running，与 FEAT-0003 一致。单独 load_input 原样传播错误。
计算/缓存/载荷/最后提交沿用原语义，finish_run 继续只显式关联用户选定版本。

## 验收标准

- [ ] SQLite 重开后可查看/加载登记的根快照，provider 可直接用于现有 run。
- [ ] run_input 记录输入、所选位置、流程、状态和固定版本结果。
- [ ] 路径搬迁/别名不改变缓存键，revision/loader 改变时隔离。
- [ ] 多位置必须选定；位置归属错误、非法本地路径在 IO 前拒绝；多文件前缀不误拒。
- [ ] 缺失绑定、损坏层级、已登记视图在预检拒绝。
- [ ] 拒绝非 SimSnap、子视图、无文件身份或有外部活动变换的加载返回值。
- [ ] 加载失败有 failed Run；状态写入失败可显式恢复。
- [ ] 缓存命中/刷新/历史结果/最后提交失败保持 FEAT-0003 语义。
- [ ] 加载期间不持有 ORM 事务；纯计算不引入 SQLAlchemy；保留现有测试。
- [ ] 专项红绿、make check、make test-all 通过，并有真实本地 pynbody 加载/重开测试。

## 后续阶段

Catalogue 命名空间与 Halo/Subhalo 身份、可重建粒子选择及容器加载、MergerTree、
声明式加载配置、免加载缓存读取、完整/轻量安装分别推进。本阶段不扫描目录、下载、
做指纹、自动回退、全局注册或缓存存活快照。

见 [ADR-010 草案](../../adr/0010-explicit-snapshot-loading.md)。
