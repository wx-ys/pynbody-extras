# FEAT-0020

## 2026-11-20

Thirteen initial tests failed because snapshot and catalogue handles lacked members. The contracts cover deferred save, concurrent writers, cache reuse, incomplete publication, nested origins and ten outputs sharing one block.

最初十三项测试因快照和目录对象缺少 members 入口而失败。用例涵盖延迟保存、并发写入、命中复用、未确认保存、嵌套来源和十个输出共享索引。

Failure injection covered receipt loss, interrupted confirmation, concurrent first use and repeated recovery. Pending selections stay private until confirmation. A cross-root coordinate target, changed root count or damaged selection descriptor is rejected before particle I/O. The broader membership, recovery and replay run passed 151 tests in the base environment, with its optional pandas test exercised in the full environment.

故障注入覆盖回执丢失、确认中断、并发首次使用和重复恢复。待确认选集在确认前不可见。跨根坐标目标、根粒子数变化及选集定义损坏均在粒子读取前报告。基础环境的成员、恢复和重放联测通过 151 项，可选 pandas 用例由完整环境实跑。

Integrated verification passed `make check` (1,920 fast tests) and `make test-all` (1,928 tests) with pandas and real data. The complementary empty selection produced its expected warning.

集成验证已通过 `make check`（1,920 项快速测试）和 `make test-all`（1,928 项完整测试），含 pandas 和真实快照。取补集后为空的选集触发了预期提示。
