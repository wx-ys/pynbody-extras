# FEAT-0021 · 查询并重放筛选时的坐标

用 Filter、声明的变换或实际坐标找到来源，在独立粒子视图上按筛选时的单位与顺序重放已支持的操作。 [实施计划](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

四项端到端测试在调用坐标重放时复现入口缺失。真实 Gadget HDF 快照另行复现了 pynbody 的按需复制与 pathlib 文件名不兼容。测试将重放结果逐项与现场的位置、速度和根索引对照。

已将原坐标和选集坐标与现场测量逐项对照。真实 Gadget 用例保留有序索引 [3, 1, 3, 0]。即使自定义加载器返回共享快照，重放也在独立数组上转换单位。

[tests/test_member_frames.py](../../../tests/test_member_frames.py) · [Handoff](handoff.md)
