# FEAT-0019

## 2026-11-20

On the Phase 1 baseline ten tests reached the scientific values but failed because the Result had no selection observations. The joint and nested aperture controls were 28 and 4 kpc, with masses 3 and 2 Msol.

阶段一基线上十项测试得到了预期科学值，随即因缺少成员观察而失败。同层交集和嵌套孔径分别为 28 与 4 kpc，质量分别为 3 与 2 Msol。

The final isolated run passed all 27 observation tests, 1,869 fast tests and 1,876 full tests with the repository data and pandas available. Ruff, mypy, the workspace lock and the paired work-item check passed. The native observer restores the original pynbody hook after overlapping executions and exceptions. ADR-023 covers external handles and incomplete evidence.

最终隔离验证通过全部 27 项观察回归、1,869 项快速测试和 1,876 项完整测试，运行环境有仓库实测数据和 pandas。Ruff、mypy、工作区锁文件和双语工作项检查均通过。并行运行和异常结束后，原生观察器会恢复 pynbody 钩子。ADR-023 记录了外部变换和证据不完整时的边界。

PR #58 review reproduced a translated worker-thread measurement with an incorrectly complete frame and an untouched custom Filter mask with a spurious use. Native notifications now reach the in-flight runs on their actual snapshot root. Frame observation reuses the particle view checked by the executor. Five additional tests cover workers, nested runs, raw masks and the explicitly unavailable evidence on generic ResultStore cache hits.

PR #58 的评审复现了工作线程短暂平移后仍被标为完整的坐标，以及未取出粒子视图的自定义掩码被误记为使用。原生变换现在通知对应根快照上正在执行的计算，坐标观察复用执行器已经检查的视图。新增五项测试涵盖工作线程、嵌套执行、原始掩码及普通 ResultStore 缓存命中的未记录状态。

After review, `make check` passed 1,874 fast tests and `make test-all` passed all 1,881 tests, including all 32 observation cases.

修复评审问题后，`make check` 通过 1,874 项快速测试，`make test-all` 通过全部 1,881 项，含 32 项执行时观察回归。
