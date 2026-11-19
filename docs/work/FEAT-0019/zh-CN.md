# FEAT-0019 · 在计算时记录 Filter 成员与测量坐标

计算消费筛选视图时，冻结其直接来源、有序根索引、Filter 实参、输出与 DAG 路径、用途及坐标。记录每个具名输出实际测量时的坐标。 [实施计划](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

阶段一基线上十项测试得到了预期科学值，随即因缺少成员观察而失败。同层交集和嵌套孔径分别为 28 与 4 kpc，质量分别为 3 与 2 Msol。

完整执行现已记录独立于快照的来源、输出用途和测量坐标。运行时也能识别科学计算钩子内短暂施加的原生变换，十个输出共用的一组成员只保留一份索引。

[tests/test_selection_observations.py](../../../tests/test_selection_observations.py) · [Handoff](handoff.md)
