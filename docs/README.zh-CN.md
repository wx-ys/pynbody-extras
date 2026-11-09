# 工作项与交接

[English](README.md) | [简体中文](README.zh-CN.md)

这个目录用于记录跨设备、跨会话、跨 agent 的工作。新的开发者或 agent 应该能
只读仓库就接手任务，而不需要翻聊天记录。

当前实现审查、V1 修订设计与逐项 agent 任务从 [设计入口](design/README.md) 开始。

## 每个工作项的三类记录

| 记录 | 文件 | 作用 |
| --- | --- | --- |
| 状态 | `meta.json` | 机器可读的身份和状态；`id`、`type`、`status`、`owner`、`issue` 的唯一来源。 |
| 规格 | `en.md` / `zh-CN.md` | 稳定的需求描述，中英文使用相同章节。 |
| 交接 | `handoff.md` | 追加式操作日志：做了什么、下一步、命令、阻塞。 |

## 目录结构

```text
docs/work/
  INDEX.md                 # 自动生成的双语总表
  _template/
    meta.json
    en.md
    zh-CN.md
    handoff.md
  BUG-0001/
    meta.json
    en.md
    zh-CN.md
    handoff.md
```

ID 全局递增、永不复用：`REQ-*`、`FEAT-*`、`BUG-*`、`TASK-*`、`SPIKE-*`。
目录名就是 ID，标题放在 `meta.json`，这样改标题不会改变路径。

## 创建工作项

```bash
make work-item TYPE=bug \
  TITLE_EN="Short English title" \
  TITLE_ZH="简短中文标题"
```

然后填写中英文正文和验收标准。新工作项默认是 `draft`。

## 状态机

```text
draft -> ready -> in-progress -> review -> done
                    |  ^
                    v  |
                  blocked
```

| 状态 | 含义 |
| --- | --- |
| `draft` | 想法，待评估或补充。 |
| `ready` | 验收标准和测试计划已完整，agent 可以开始。 |
| `in-progress` | 正在处理。 |
| `blocked` | 等待外部决定或依赖。 |
| `review` | PR 已开。 |
| `done` | 已合并并验证。 |
| `archived` | 不再相关。 |

状态只写在 `meta.json`，不要在中英文正文里重复。

## 工作流

1. 创建或关联 GitHub issue，把编号写进 `meta.json.issue`。
2. 创建工作项，填写中英文两份文件。
3. bug 要先复现，记录实际行为和期望行为。
4. 先写失败测试，再改实现。
5. 做最小修复；需要时运行 `make check` 和 `make test-all`。
6. 把 `status` 改成 `review`，追加 `handoff.md`，开 PR。
7. 同一个 PR 里同时更新中英文。

## 完成与归档

1. 把 `meta.json.status` 改成 `done`，填写 `completed`、`resolution` 和 `pr`。
2. 在中英文正文里增加 `结果` / `Outcome` 一节，并勾选全部验收标准。
3. 在 `handoff.md` 追加最终记录，写清验证证据。
4. 提交完成后的工作项（**Commit A**）。归档前它必须已经存在于该 commit。
5. 让它继续留在 `docs/work/`，直到下一个 release 或 30 天，方便复查或重新打开。
6. 执行归档：

   ```bash
   make archive-item ID=BUG-0001
   ```

   命令会向 `docs/history/YYYY-MM.md` 和
   `docs/history/YYYY-MM.zh-CN.md` 各追加一行，指向 Commit A，然后删除工作项
   目录并重新生成 `docs/work/INDEX.md`。
7. 审核并提交 history 行和删除操作（**Commit B**）。

History 必须指向 Commit A，因为 Commit B 里工作项文件已经不存在。ID 永不复用：
重新打开的事项使用新 ID，并在 `related` 里引用旧 ID。

## 交接协议

每次追加新的一节，不要改写历史：

```markdown
## 2026-10-20 14:30 +08:00 — agent-name @ device

- State: in-progress
- Done: ...
- Next: ...
- Commands: ...
- Failing test: ...
- Blockers: ...
- Files: ...
- Env: Python ... / uv ... / pynbody ...
```

字段名保持英文，方便工具解析；正文可以用中文或英文。不要在 handoff 里写
token、绝对家目录路径或机器专属凭据。

## 双语约定

- 代码标识符、API 名称、命令在两种语言里都保持英文。
- `en.md` 和 `zh-CN.md` 必须在同一次修改中一起更新。
- `docs-check` 检查配对、ID、状态和索引是否最新；它无法判断翻译质量。
- 需求变化时，先同时更新中英文验收标准，再改实现。

## 常用命令

```bash
make work-item TYPE=bug TITLE_EN="..." TITLE_ZH="..."
make docs-index     # 重新生成 docs/work/INDEX.md
make docs-check     # 检查配对、元数据和索引
make archive-item ID=BUG-0001
```
