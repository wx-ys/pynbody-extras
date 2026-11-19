# FEAT-0021

## 2026-11-20

Four end-to-end cases reached the missing replay implementation. A real Gadget HDF snapshot also exposed a pynbody copy-on-access incompatibility with pathlib filenames. Verification compares to position, velocity and root indices copied at measurement time.

四项端到端测试在调用坐标重放时复现入口缺失。真实 Gadget HDF 快照另行复现了 pynbody 的按需复制与 pathlib 文件名不兼容。测试将重放结果逐项与现场的位置、速度和根索引对照。

Verified rotations, ordered position and velocity shifts, partial targets and repeated particle indices against values copied during the calculation. Replaying a shared loader in different units leaves its live arrays intact. With the snapshot moved, both load modes report the missing file and the fixed result still exposes the origin.

旋转、位置与速度平移的次序、局部目标及重复索引均已和计算现场保存的数值对照。以其他单位重放时，共享加载器的原数组保持原单位。移走快照后，两种加载模式都报告文件缺失，固定结果仍能查回来源。

Integrated verification passed `make check` (1,920 fast tests) and `make test-all` (1,928 tests) with pandas and real data. The complementary empty selection produced its expected warning.

集成验证已通过 `make check`（1,920 项快速测试）和 `make test-all`（1,928 项完整测试），含 pandas 和真实快照。取补集后为空的选集触发了预期提示。
