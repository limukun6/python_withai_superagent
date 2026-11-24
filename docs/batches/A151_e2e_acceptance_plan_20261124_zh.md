# A151 端到端验收计划：压缩菜单下两条路径各 ≤3 fit / ≤3 调用

2026-11-24，前置：单调用机制探测已成功（`batches/A151_menu_probe_results_20261124_zh.md`，1 次调用、0 fit、无新增 unknown）。本计划按复核第 4 节第 4 条：只有机制成功后才立端到端验收。

## 1 目的

同一任务、同一划分、同一预算下，用压缩菜单表示跑通两条路径，验证：

1. 正式入口能自主完成"提案 → 执行 → 真实反馈 → 复核 → 重训 → 独立进程预测"；
2. `menu-view compact` 不改变候选集合、参数与选型规则；
3. L1（`grounded-v1` + `--issue-layering by-class`）在压缩表示下是否可用。

不回答提分、不回答通用性、不回答压缩是否降低总耗时。

## 2 固定条件

| 项 | 值 |
|---|---|
| 任务 | `D:\codex\python_withai\data\released_task2.zip`（已见开发任务） |
| seed | 707 |
| 调度 | `--scheduler ab-v1 --portfolio ab-v1 --catalog-scope full --preference-catalog standard` |
| 预算 | `--fit-limit 3`（含最终 refit）、每路径最多 3 次 Flash、`--llm-budget-rmb .15` |
| 时限 | `--time-limit 600 --trial-timeout 120` |
| 表示 | **`--menu-view compact`（两条路径都开）** |
| 反馈视图 | `--feedback-view compact` |
| 交付 | `--partial-delivery checkpoint` |
| 账本 | 唯一真实账本；`--authorization research/a130_accepted_history_v3` |

## 3 两条路径

| 路径 | 契约 | 分层 | task_budget_id |
|---|---|---|---|
| L0 | `calibrated-v2` | off | `a151_loop_off` |
| L1 | `grounded-v1` | `by-class` | `a151_loop_by_class` |

合计上沿 **6 fit / 6 调用 / ¥0.30**。

## 4 验收（全满足才算通过）

1. 两条路径 `requested_planner_completed=true`，无自动重试；
2. 实际 fit 在 2–3 之间且 `attempted == completed`，无未知失败；
3. 实际调用 ≤3，且等于账本请求增量；
4. 新增 unknown / pending 为 0；
5. 两条路径各自选出候选并记录；不同则如实记录，不解释为收益；
6. 每条路径用 `python -m superagent predict` 在独立进程产出预测行；
7. 运行前后 `ab_run.sources()` 一致。

## 5 停止规则

- 任一验收项失败：停止本批，不重试，保留原件。
- 出现新 unknown/pending：立即停止，不再跑第二条路径。
- L1 在 3 次调用内仍无合法提案：停止，记 `addressed`/`requires_semantic_repair` 次数，不换模型、不加 token。
- 不因结果不理想而换 seed 或加调用。

## 6 允许与不允许的表述

- 允许："两条路径都完成了端到端运行"、"压缩表示下 L1 可用/不可用"、"实际 fit/调用/费用是 X"。
- 不允许："提分"（无外层评价对照）、"通用性"（单任务、已见）、"压缩省时"（未做同条件配对）。

## 7 交付物

- 运行器：`scripts/run_a151_small_trial.py --menu-view compact`
- 输出：`research/a151_e2e_acceptance/`（protocol、command_*、L0/L1 运行目录、predictions、completion）
- 报告：`docs/batches/A151_e2e_acceptance_results_20261124_zh.md`
