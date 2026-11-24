# A151 探索广度计划：2 个候选 vs 4 个候选

2026-11-24。免费分析显示：选型只损失 0.15pp，而"探索过的最优候选"比单次实际均值高 **1.81pp（成对文本）/ 2.12pp（单文本）**。本计划直接检验这个天花板的真实性。

## 1 问题与判定

**问题**：在其它条件全同的前提下，把可探索候选从 2 个加到 4 个，验证分数是否提高？

**预登记判定**（看到结果前写定）：

- 主指标：每个（任务, seed）配对的 `breadth4 − breadth2` 验证分数差。
- "加宽有收益"：4 个配对里 **≥3 个为正** 且均值 **≥ +0.5pp**。
- "加宽无收益"：其余情况。
- 记录但不判定：加宽后是否选到 breadth2 池外的候选；是否接近池最优上界。
- **漂移约束**：本轮已量到同配置重跑可差 7.9pp，因此单对差值 < 1pp 一律视为不可区分。

## 2 固定条件

| 项 | 值 |
|---|---|
| 任务 | `released_task2`（成对文本）、`sms_spam`（单文本） |
| seed | 707、811 |
| 视图 | `compact`（两臂相同；选它是因为成对文本在 full 下有 417 秒断流史） |
| 契约 | `calibrated-v2`（两臂相同） |
| 臂 A（breadth2） | `--fit-limit 3 --llm-request-limit 3` → anchor + 2 次探索 + refit = 3 fit |
| 臂 B（breadth4） | `--fit-limit 6 --llm-request-limit 4` → anchor + 4 次探索 + refit = 6 fit |
| 调度 | `ab-v1` |
| 反馈 | `--feedback-view compact` |
| 时限 | 600 / 120 秒 |
| 账本 | 唯一真实账本 + `accepted_history_v3` |

上沿：4 × 1 + 4 × 4 = **20 次调用**；fit 合计 4×3 + 4×6 = **36**；预计 ¥0.25 左右。

## 3 路径

| # | 任务 | seed | 臂 |
|---:|---|---:|---|
| 1 | pair_text | 707 | breadth2 |
| 2 | pair_text | 707 | breadth4 |
| 3 | pair_text | 811 | breadth4 |
| 4 | pair_text | 811 | breadth2 |
| 5 | single_text | 707 | breadth2 |
| 6 | single_text | 707 | breadth4 |
| 7 | single_text | 811 | breadth4 |
| 8 | single_text | 811 | breadth2 |

## 4 验收与停止

沿用既有口径（规划完成、fit 计数完整且 ≤ 该臂上限、调用 = 账本增量且 ≤ 该臂上限、无新 unknown、交付校验一致）。任一路径失败即停止整批；出现新 unknown 立即停止；不因结果换 seed 或加预算。

## 5 代码变更（先做、默认不变）

`ab_run` 原先把规划请求上限写死为 3。本轮把它参数化为 `--llm-request-limit`（默认 3，允许 1..12），并同步放宽两处校验上限（`ab_scheduler` 的预算自洽检查、`scheduler_bridge` 的上限检查）到 12。**不带该参数时行为与之前逐字节一致**，由 `tests/test_request_limit.py` 锁定。

## 6 交付物

- 计划：本文 + `research/a151_breadth/plan.json`
- 运行器：`scripts/run_a151_matrix.py`（新增 `request_limit` 字段）
- 报告：`docs/batches/A151_breadth_results_20261124_zh.md`
