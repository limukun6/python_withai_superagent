# A151 迁移与复核验收计划：五种任务/契约组合，预登记后执行

2026-11-24。依据 Codex 复核提出的两项"尚未启动"：① 把压缩表示与语义修复迁移到其它真实任务类型；② 另立一个**确实包含 confirm 动作**的工程验收。

## 1 目的

| 工作流 | 问题 | 判据 |
|---|---|---|
| W1 迁移 | 压缩菜单与 grounded 修复路径在别的任务类型上是否同样可用 | 每条路径完成闭环、提案合法、预测可交付；不比较分数 |
| W2 复核 | 调度真正走到 `confirm` 时，独立复核训练能否被观测 | `independent_confirmation_observed=true`，且两个 finalist 在两个划分上都有测量 |

不回答提分、不回答通用性。

## 2 固定条件

| 项 | 值 |
|---|---|
| seed | 707 |
| 表示 | `--menu-view compact`（全部路径） |
| 反馈视图 | `--feedback-view compact` |
| 时限 | `--time-limit 600`、`--trial-timeout 120` |
| 交付 | `--partial-delivery checkpoint` |
| 每路径调用上限 | 3（`ab_run` 硬上限），每条路径单独 `--llm-budget-rmb .15` |
| 账本 | 唯一真实账本，`--authorization research/a130_accepted_history_v3` |
| 停止 | 任一验收项失败或出现新 unknown/pending 立即停止，不重试 |

## 3 路径矩阵（预登记）

| 名称 | 任务包 | 调度 | fit 上限 | 契约 | 分层 | task_budget_id |
|---|---|---|---:|---|---|---|
| w1_categorical | `released_task1.zip`（类别表格，15 候选） | ab-v1 | 3 | calibrated-v2 | off | `a151_w1_cat` |
| w1_numeric | `dry_bean_numeric_tabular.zip`（数值表格，5 候选） | ab-v1 | 3 | calibrated-v2 | off | `a151_w1_num` |
| w1_single_text | `sms_spam_single_text.zip`（单文本，14 候选） | ab-v1 | 3 | calibrated-v2 | off | `a151_w1_txt` |
| w1_single_text_grounded | 同上 | ab-v1 | 3 | grounded-v1 | by-class | `a151_w1_txtg` |
| w2_confirm | `released_task2.zip`（成对文本） | **ab-finalists** | **5** | calibrated-v2 | off | `a151_w2_confirm` |

`ab-finalists` 需要 5–11 fit 且预留 1 次 refit；`fit_limit=5` 时 `target=2`：2 次 primary 探索 + 2 次确认 + 1 次 refit，调用 2 次（≤3）。确认划分会进入 `basis`，因此可以观测独立复核。

预算上沿：**5 条路径 × 最多 3 次调用 = 15 次调用；fit 合计约 3×4 + 5 = 17**。预计费用约 ¥0.1–0.2。

## 4 每条路径的验收项

1. `requested_planner_completed=true`，无自动重试；
2. `attempted_classifier_fits == completed_classifier_fits`，且不超过该路径 fit 上限；
3. 实际调用 ≤3，且等于账本请求增量；
4. 新增 unknown / pending 为 0；
5. 独立进程 `superagent predict` 产出预测行；
6. 运行前后 `ab_run.sources()` 一致；
7. **仅 w2_confirm 额外要求** `independent_confirmation_observed=true`。

## 5 停止规则

- 任一路径验收失败：立即停止整个矩阵，保留该路径原件与 outcome，不重试、不改参数重跑。
- 出现新 unknown/pending：立即停止，不执行后续路径。
- w2_confirm 若 `independent_confirmation_observed=false`：判 W2 失败，保留证据并报告，不换调度重试。

## 6 交付物

- 计划与矩阵：本文 + `research/a151_matrix/plan.json`
- 运行器：`scripts/run_a151_matrix.py`（默认只解析校验，`--execute` 才发请求）
- 结果：`research/a151_matrix/{protocol,completion}.json` 与每条路径自己的运行目录
- 报告：`docs/batches/A151_matrix_results_20261124_zh.md`
