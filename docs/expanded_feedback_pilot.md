# 扩展候选库上的逐轮反馈对照

以 `b29b558` 的扩展候选库和训练流程为底座。`fixed` 复用 `local_interleaved_routes`，Flash 组每轮看到剩余候选的配置和已完成试验的搜索分数，由它选择下一项。两组在各任务上使用同一份训练内留出划分、目录首项、相同的 4 次搜索拟合、前三名复核和一次完整重训。每组完成 11 次拟合，单次拟合最多 90 秒，两组各有 600 秒的总时间预算。

两个任务都固定外层种子 `42011`、预留分区种子 `41101`、内层搜索种子 `707`。先用 `prepare` 冻结两个固定组，同时保存任务包、代码、依赖和划分。`compare` 校验这些条件及固定组模型哈希，复用两个固定组。每个任务等两臂选型冻结后，才在同一训练内留出集上计算准确率、净改正数和新增错误数。本轮的样本量适合初筛。课程验证集和正式测试集均不参与。

## 运行

从整合分支仓库根目录运行，`--output` 指向新目录。两条命令使用相同的任务包、输出目录和 Python 环境。

```powershell
python -m tools.expanded_feedback_pilot prepare --task1 <Task1.zip> --task2 <Task2.zip> --output <PilotDir>
python -m tools.expanded_feedback_pilot compare --task1 <Task1.zip> --task2 <Task2.zip> --output <PilotDir> --api-key-file <KeyFile> --budget-ledger <ExistingCourseLedger>
```

`prepare` 全程本地。`compare` 最多调用 6 次校园 Flash，每个任务的课程预估额度为 ¥0.10。课程账本使用 `task1` 和 `task2`，开跑前用 `budget-status` 核对现有账本和既往记录。校园网直连使用 SSE 与最终 usage，读超时最多 60 秒。请求失败立即终止，费用状态写入原有持久账本。`paired_results.json` 保存两臂分数、训练次数、用量及逐行配对统计。

在普通任务入口也可显式选择 `python -m superagent run <Task.zip> --planner scalar-feedback`，并提供 `--api-key-file`、`--budget-ledger` 和 `--task-budget-id`。该模式默认使用校园 API 和 `DeepSeek-V4.1-Flash`，可用 `--trials 4 --trial-timeout 90 --time-limit 600 --feature-cache run` 对齐试点的搜索参数。正式配对使用上面的 `prepare` 和 `compare`，以固定相同的外层留出集。
