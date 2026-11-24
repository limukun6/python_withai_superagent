# A1-04：正式 CLI、累计账本与请求失败停止

运行前计划。新增 planner=raw-history/checked-history，复用现有 ExperimentLLM 与 TaskBudgetAccount；两组同一端点/Flash/流式/输出契约，差异仅检查和记忆。限制不兼容的多审阅、动态研究和 allocator 组合，避免未经实验的功能叠加。

实验：通过 build_parser 的正式 handler 运行（与 python -m superagent run 共用），真实数值任务训练、三划分确认、全量 refit；只替换 HTTP Session 的传输响应。客户端解析、token 记账和 SQLite 累计不模拟。两组同任务账本跨运行累计；分别注入 transport timeout、缺少 usage、无效 JSON，要求首个错误即停，不随机恢复、不发下一请求，未知费用不写零。

资源：两次成功各 7 拟合=14；三次失败各保留 anchor 的 1 拟合=3，共预计 17 次真实拟合。成功各1个模拟 HTTP，失败各1个，共5；真实 API=0。单元配置检查不训练。宽回归夹具另记。预算里的 token 与金额只是模拟账本数，不是真实收费。

先写测试与 runner，baseline 在旧 CLI 上应显示缺少 planner 支持；再实现，fixed 跑全入口验收。最后回归旧 CLI/账本/前批模块。所有运行输出保留，源码未冻结时不采作最终证据。

从主仓库：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_cli_batch.py --phase baseline --output .\runs\a1_04_baseline
# 实现后：
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_cli_batch.py --phase fixed --output .\runs\a1_04_fixed
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_cli_batch.py --phase regression --output .\runs\a1_04_regression
```

续跑先读 EXPERIMENT_PROGRESS.md，已完成目录不覆盖。故障例使用独立的模拟账本，不改真实任务账本。当前不调用真实 API，A2 必须先写任务/预算/历史状态协议。此批只接 raw/checked，动态预算调度属于 B 的 Scheduler 接口，不在这里强接。
