# TASK-0003 — 核查已合并 PR 中的 Sourcery 评审意见

## 范围

以 main 的 `60fc13f` 为基线，审核 PR #1–59 的全部 Sourcery 对话评论、评审正文和行内线程。已解决和已过时的线程也逐项与当前实现核对，并记录判断依据。

## 验收标准

- 样本对象与指定名称数量不符时，在计算和写入第一行前报错。导出保留重复显示名称，按名称查找时能识别歧义。数值列返回 NumPy 数组并保留复数。
- 结构化数组编解码对读写使用固定的 1 MiB NPY 文件头上限，受支持的宽记录在禁用 pickle 时仍能往返。
- 根据公开约定核对签名、存储、展示和 Pipeline 的评审意见。通过实际 API 验证仍需修复的问题。
- 保留 ADR-011 定义的缺失值和带类型的行身份。合并前验证计算器分层、既有测试、数据测试和双语索引。

## 审核记录

GitHub 返回了 79 条对话评论，其中 53 条为评审导读，26 条为额度或提交体积提示，另有 33 份评审正文。51 条行内意见分布在 18 个 PR，审核时其中 40 个线程未标记解决，11 个已标记解决。

| 已合并 PR | 在当前主分支上的核查及本次处理结果 |
| --- | --- |
| [#3](https://github.com/wx-ys/pynbody-extras/pull/3) | 渲染测试已导入仓库中的辅助模块，并断言完整的 `RBelow` 标签。两条意见均已有回归覆盖。 |
| [#4](https://github.com/wx-ys/pynbody-extras/pull/4) | 恢复旧查询使用的 `statistics.parse_stat_key` 导入，并给出弃用提示。Card 已转义标签和值，两个渲染器均覆盖含标记字符的文本。原始 HTML 辅助函数及其过时文档已被替换。 |
| [#7](https://github.com/wx-ys/pynbody-extras/pull/7) | 将测试辅助函数 `RBelow.calculate` 的返回类型改为 NumPy 布尔掩码。 |
| [#14](https://github.com/wx-ys/pynbody-extras/pull/14) | 现有位置参数文档和测试一致，也覆盖其他参数显式指定的情况。 |
| [#16](https://github.com/wx-ys/pynbody-extras/pull/16) | 图像注解按文档使用延迟导入的 `resolve_type_hints`，避免启动时载入绘图模块。颜色条接受的 artist 类型已包含散点图的 `PathCollection`。 |
| [#17](https://github.com/wx-ys/pynbody-extras/pull/17) | 展示已初始化的工厂值及非 dataclass 配方参数；展开详情时才生成内容。补充 asyncio 的样式继承规则和 `InfoView` 子类从 `_display_body()` 迁移到 `_display_blocks()` 的写法。 |
| [#18](https://github.com/wx-ys/pynbody-extras/pull/18) | 重建 Pipeline 时验证类，并注明重建使用可信输入。区分对象数组内容和不透明节点实例，拒绝缺少签名的存储记录，按规范化身份比较 NaN，阻止旧会话键写入持久存储。修复 Pipeline 输出可变及子类身份问题。普通 Pipeline 的变换作用域已计入签名。并发写入的版本语义遵循 ADR-007，短键迁移已由 ADR-006 排除。 |
| [#22](https://github.com/wx-ys/pynbody-extras/pull/22) | 执行任何计算前核对名称数量。保留重名及带类型的行身份，按重名查找时报告歧义。数值列支持复数，并明确断言 NumPy 返回类型。按 ADR-011 的现有格式，成功返回的 `None` 仍与缺失值共享哨兵。 |
| [#23](https://github.com/wx-ys/pynbody-extras/pull/23) | 结果树改为迭代遍历。深度超过 Python 递归上限的树可以在富文本卡片中显示最后一个节点。 |
| [#25](https://github.com/wx-ys/pynbody-extras/pull/25) | `vcirc` 公开异常已标明原生扩展名称并保留加载失败原因。原生测试现在会报告安装损坏，仅在扩展缺失时跳过。 |
| [#33](https://github.com/wx-ys/pynbody-extras/pull/33) | 工作项已使用含提交号的仓库链接，可从 PR 和 Markdown 文件访问。 |
| [#34](https://github.com/wx-ys/pynbody-extras/pull/34), [#35](https://github.com/wx-ys/pynbody-extras/pull/35) | 常量边界窗口已排除图像外样本。现有测试覆盖 pynbody 最低版本、中值计算的分批上限、极端有限值及文档约定的偶数窗口行为。 |
| [#37](https://github.com/wx-ys/pynbody-extras/pull/37) | 修正 `EvalEngine.run` 的刷新说明。刷新会追加不可变版本并更新当前缓存指针。 |
| [#43](https://github.com/wx-ys/pynbody-extras/pull/43) | 读入 NPY 文件头正文前检查 1 MiB 上限，写入时使用相同上限。小文件头配大数据区仍可正常往返。 |
| [#44](https://github.com/wx-ys/pynbody-extras/pull/44) | 恢复测试及模块说明使用公开术语“单元格错误”。 |
| [#57](https://github.com/wx-ys/pynbody-extras/pull/57) | 自定义 TransformChain 混用直接和委托施加的步骤时，按回执顺序恢复。分离 tuple、list、dict 子类属性里的粒子数组并保留别名。ADR-022 要求调用者同步共享粒子根对象，区分可恢复的参数准备失败和修改情况不明的应用钩子异常。测试及可运行示例已统一命名输出术语。 |
| [#58](https://github.com/wx-ys/pynbody-extras/pull/58) | 已有回归覆盖工作线程的变换及原始 Filter 的按需消费。Database 确认保存或恢复后，可按固定版本查询成员来源及使用位置。仅含数值的存储命中显式返回 `selection_observations is None`。按 ADR-023，传给 `check_input` 的作用域输入属于已经消费的输入。 |

Pipeline 子类修复也覆盖旧版 pickle 和数据库目录。重建配方后，输出名称、顺序和缓存键均一致。共用计算器的多个输出仍可在 Database 中分别按名称查到。在独立复核的七层配方中，每个子计算器只保存一次后，编码大小从 415,441 字节降至 1,739 字节。
