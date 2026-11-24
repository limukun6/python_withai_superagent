# A92：当前状态与历史快照分离，0fit/API

AGENTS.md连续累积多份“最新/正在跑”旧快照，靠新前缀覆盖容易让下一位Codex/DeepSeek误重启。保存完整旧文件作历史资料，然后保留全部稳定开发约定与既有校园网络诊断边界，把单一当前阶段放docs/CURRENT_AGENT_STATE.md，历史继续EXPERIMENT_PROGRESS。不得删改密钥、未知费用、独立案例、H/test、source冻结和失败不重试约束。

只改交接文档，不动superagent或A89 immutable脚本，不触发训练/模型请求。当前状态卡注明A89实际completion/ledger/fit日志、A91已排队、默认restricted及临时防待机释放条件。以后只更新状态卡，不再在AGENTS叠加历史阶段。完整旧文本归档research/a92_continuation_state，供审计；它不是新有效指令文件。
