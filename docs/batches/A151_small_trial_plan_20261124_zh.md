# A151 条件小试计划：真实闭环，两条路径各 3 fit / 3 次 Flash

2026-11-24。紧接 A151 零成本验收（`docs/batches/A151_issue_layering_results_20261124_zh.md`）。本计划先落盘，再执行；执行前先 `--validate-only` 预检，不通过就不花钱。

## 1 目的

只回答一件事：**同一个正式入口，能不能把"事实 → 提案 → 执行 → 真实反馈 → 问题状态 → 重训 → 独立进程预测"串起来**，并且开关 `--issue-layering by-class` 不改变候选菜单与调度动作集合。

不回答"分层是否提分"。两条路径、单任务、单划分，没有外层评价。

## 2 任务与固定条件

| 项 | 值 |
|---|---|
| 任务包 | `D:\codex\python_withai\data\released_task2.zip`（成对文本，课程已发布任务） |
| 任务包 SHA-256 | 运行前写入 `protocol.json`，运行后再核一次 |
| 任务性质 | **已见开发任务**，不是未见任务；不产生通用性结论 |
| seed | 707（分区与训练种子；不因结果换 seed） |
| 调度 | `--scheduler ab-v1`、`--portfolio ab-v1`、`--catalog-scope full`、`--preference-catalog standard`（试跑发现 `ab-fixed` 硬性要求 fit-limit 11，与 3-fit 上限冲突，故改用 `ab-v1`） |
| 预算 | `--fit-limit 3`（含最终 refit）、每条最多 3 次 Flash |
| 时限 | `--time-limit 600`、`--trial-timeout 120` |
| 交付 | `--partial-delivery checkpoint` |
| 反馈视图 | `--feedback-view compact` |
| 账本 | `runs/a129_B_ledger_return/private/ledger.sqlite`（唯一真实账本） |
| 历史策略 | `--history-policy incremental-research` + 已接受历史快照（原 12 条 unknown 保持 null） |

## 3 两条路径

| 路径 | 提案契约 | 分层 | task_budget_id | 每路径上限 |
|---|---|---|---|---|
| L0 | `calibrated-v2` | `off` | `a151_loop_off` | 3 fit / 3 调用 / ¥0.15 |
| L1 | `grounded-v1` | `by-class` | `a151_loop_by_class` | 3 fit / 3 调用 / ¥0.15 |

合计上沿 **6 fit / 6 调用 / ¥0.30**。两条路径共用同一任务包、同一 seed、同一菜单；差异只有提案契约与分层开关。

## 4 验收（全满足才算通过）

1. 两条路径都 `requested_planner_completed=true`，无自动重试。
2. 实际 fit 在 2–3 之间，且 `attempted == completed`，无未知失败。
3. `new_api_requests` ≤3，且等于账本请求增量（逐条对账）。
4. 账本新增 unknown/pending 为 0；原 12 条 unknown 不变。
5. 两条路径选出的候选记录在案；若不同，如实记录，不解释成收益。
6. 每条路径的模型在**独立进程**用 `python -m superagent predict` 产出预测行；行数与任务声明一致或记下失败原因。
7. 源码 SHA 在运行前后一致（`ab_run.sources()`）。

## 5 停止规则

- 任一验收项失败：立即停止本批，不重试，保留原 run 目录与 outcome。
- 出现新的 unknown/pending 用量：停止，不继续第二条路径。
- 实际调用数达到 3 仍无合法提案：停止，记为语义修复上限触发。
- 网络失败：停止并保留错误，不改代理/防火墙、不加长超时。

## 6 允许与不允许的表述

- 允许："正式入口能完成一次真实闭环"、"开关不改变菜单"、"这两条路径的实际 fit/调用/费用是 X"。
- 不允许："分层提高/降低了分数"（无外层评价）、"证明了通用性"（单任务已见）、"节省了训练"（未做同动作配对）。

## 7 交付物

- `scripts/run_a151_small_trial.py`（默认只校验，`--execute` 才花钱）
- 输出目录 `research/a151_small_trial/`：`protocol.json`、`command_*.json`、`<arm>/` 运行目录、`predictions_<arm>.csv`、`completion.json`、`current_ledger_owner.json`
- 报告 `docs/batches/A151_small_trial_results_20261124_zh.md`
