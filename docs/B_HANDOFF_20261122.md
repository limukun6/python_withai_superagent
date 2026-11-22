# B 线交接 · 2026-11-22

请从 `codex/b-a31-joint` 取主交付。它基于双方共同的 A31 提交 `d910dc4`，包含 B 的可执行候选目录、调度与选型逻辑，以及先前的匿名正式入口和新进程预测验收。A 当前公开分支的 `67ba67f` 已做 Git 合并预检。两份代码可自动合并，合并后的运行验收由双方的整合仓库完成。

B 的生产改动集中在 `ab_catalog.py`、`ab_methods.py`、`ab_scheduler.py` 和表征模块，另有配套的文本对输入修正。正式入口、动作执行、反馈和账本以 A 的后续更新为准。优先审阅这些文件及对应测试，再决定是否引入以下独立实验。

| GitHub 分支 | 已验证内容 |
|---|---|
| `codex/b-a31-joint` | 共用入口与 B 的稳定方法。22 条真实开发路径和 12 条 A29 路径的停止规则回放，发布任务输入覆盖与错题分析 |
| `codex/b-route-representative` | 代表参数和低置信度时采用第二模型的开发对照 |
| `codex/b-nonlinear-frozen-text` | 新增非线性 MiniLM 表征，留出评论来源和独立电影语料的复核 |
| `codex/b-evidence-text-recipes` | 由真实训练证据触发的词字符组合，及反馈、任务文件名、主指标的回放 |
| `codex/b-canonical-preference-output` | 显式二选一偏好交付。两份开发留出 3,358 行的正反概率互补，原方向预测逐字节一致 |

主分支的匿名模型已通过两次独立进程重载，31 行输出一致。冻结模型的接收验证在仅有 A31 基线的仓库导入五条分支，核对八份原生预测文件与两组标题概率。该接收验证之后的新提交为离线回放、错题分析及其文档，当前补充分析使用零次新训练和零次模型 API。

整合验收沿用 `AGENTS.md` 和 A/B 接口约定。运行结果用各自的提交、分区和费用记录追溯。保留两份已登记的迁移任务，等双方共同冻结再定。

[A31 接入验收](batches/BA31_01_02_integration_results.md) · [真实动作回放](batches/B_STOPPING_PREFIX_01_results.md) · [A29 交叉检查](batches/BA31_04_confirmation_conflicts_results.md) · [发布任务错题分析](batches/B_RELEASED_INPUT_DIFFICULTY_02_results.md)
