# A151 多 seed 配对计划：grounded 是否普遍更差

2026-11-24。前置：A151 的两条 grounded 路径（成对文本、单文本）都可执行但不比普通路径好，各只有 1 个样本。本计划用**同任务、同候选、同划分、同 fit 预算**的配对，在三个 seed 上看这个方向是否稳定。

## 1 问题与判定

**问题**：在相同预算下，`grounded-v1` + `--issue-layering by-class` 的选型质量是否低于 `calibrated-v2`？

**预登记判定**（在看到结果前写下）：

- 主指标：每对 `grounded − plain` 的官方验证分数差（同任务同一份验证集）。
- 报告：每任务的 3 个配对差、均值、胜/平/负计数。
- 结论"grounded 不优于普通"：两个任务各自均值 ≤ 0，或任一任务没有 2/3 个 seed 为正。
- 结论"有正向信号"：某任务 ≥2/3 个 seed 为正且均值 > 0.2 个百分点。
- 不做显著性声明：n=3，且三个 seed 共用同一份验证集，不是三个独立任务。

## 2 固定条件

| 项 | 值 |
|---|---|
| 任务 | `released_task2.zip`（成对文本）、`sms_spam_single_text.zip`（单文本） |
| seed | **707、811、823**（707 已用过，811/823 为本批新登记） |
| 调度 | `ab-v1`、`ab-v1` portfolio、catalog full、preference standard |
| fit 上限 | 3（含 refit） |
| 调用上限 | 3（`ab_run` 硬上限） |
| 表示 | `--menu-view compact` 两臂相同 |
| 反馈 | `--feedback-view compact` 两臂相同 |
| 臂 | plain=`calibrated-v2`；grounded=`grounded-v1` + `--issue-layering by-class` |
| 时限 | 600 秒 / 单次 120 秒 |
| 账本 | 唯一真实账本 + `research/a130_accepted_history_v3` |

上沿：12 条路径 × 最多 3 次调用 = 36 次调用；fit 约 36。预计费用约 ¥0.25。

## 3 路径顺序（交替先后，避免顺序偏置）

| # | 任务 | seed | 臂 |
|---:|---|---:|---|
| 1 | pair_text | 707 | plain |
| 2 | pair_text | 707 | grounded |
| 3 | pair_text | 811 | grounded |
| 4 | pair_text | 811 | plain |
| 5 | pair_text | 823 | plain |
| 6 | pair_text | 823 | grounded |
| 7 | single_text | 707 | plain |
| 8 | single_text | 707 | grounded |
| 9 | single_text | 811 | grounded |
| 10 | single_text | 811 | plain |
| 11 | single_text | 823 | plain |
| 12 | single_text | 823 | grounded |

## 4 每条路径的验收

沿用 A151 矩阵的口径：`requested_planner_completed=true`、fit 计数完整且 ≤3、调用 = 账本增量且 ≤3、无新增 unknown/pending、交付校验一致（`released_task2` 用 CLI predict；`sms_spam` 无 test split，用新进程重载 + 验证集评分）。

## 5 停止规则

- 任一路径验收失败：停止整批，保留原件，不重试。
- 出现新 unknown/pending：立即停止。
- 不因结果方向换 seed 或加预算。

## 6 交付物

- 计划：本文 + `research/a151_multiseed/plan.json`
- 运行器：复用 `scripts/run_a151_matrix.py --plan ... --output ...`
- 报告：`docs/batches/A151_multiseed_results_20261124_zh.md`
