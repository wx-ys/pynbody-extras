# Design entry point / 设计入口

Current review baseline: `5df15dd`, 2026-11-09.

- [Updated V1 design / 更新后的 V1 设计](v1-2026-11-09.md)
- [Implementation review / 实现审查](review-2026-11-09.md)
- [Sequential task handoff / 逐项实施入口](implementation-roadmap.md)
- [Established decision register / 既有决策索引](decisions.md)
- [Implementation contract review / 后续接口确认](implementation-approval.md)
- [Full/lightweight packaging proposal / 完整与轻量发行提案](packaging-decision.md)
- [Observed installation matrix / 实际安装矩阵](installation-matrix.md)
- [Local delivery review / 本地交付审查](local-delivery-review-2026-11-09.md)

2026-11-09 local implementation update: BUG-0010–0012 and FEAT-0005–0012 are
implemented, verified and independently reviewed; no commit or merge is claimed.
The user explicitly approved all eight feature contracts and packaging
names/ownership; accepted ADR-011–019 record the decisions. FEAT-0010 additionally
records the approved root-family-block ordering restriction. TASK-0002 implements
the core/full split; its actual installation results and final source checks are
in the matrix and work-item handoff. Each increment retains red/green evidence.
The dated review retains its original baseline findings.

`CONTEXT.md` describes implemented contracts. The design distinguishes those
contracts from proposed additions; a proposal is not an accepted architecture
decision. Work-item status lives in `docs/work/*/meta.json`.

本目录是此次审阅整理出的、可随仓库交接的设计版本。来源包括本地 V1 文档
《pynbody-ext 设计与开发文档》§21–22.7、`.develop_notes/requirements_vision.md`、
现有 ADR、源码和测试。外部 V1 原稿保留；其早期实现差距应以此次源码核验为准。
2026-10-22 的 Workspace/Artifact 整体重建方案是历史方案，不是本轮实施指令。
