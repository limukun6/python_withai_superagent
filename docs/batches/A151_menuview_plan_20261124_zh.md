# A151 表示配对计划：full vs compact 只换视图

2026-11-24。压缩菜单已接入并通过迁移与端到端验收，但**从未做过同任务的配对**：现有证据只有字节数（21,473→13,949，−35.039%），没有"换表示后选型或代价有没有变"的对照。本计划补这一条。

## 1 问题与判定

**问题**：只把 `--menu-view` 从 `full` 换成 `compact`（其余全同：任务、seed、划分、契约、fit 预算、反馈视图），选型与代价是否改变？

**预登记判定**（看到结果前写定）：

- 主指标：每对的 **候选是否相同** 与 **官方验证分数差**（compact − full）。
- 结论"表示中性"：≥5/6 配对候选相同，或全部配对 |差| < 0.5pp。
- 结论"表示有害"：均值 < −0.5pp，或 ≥2 个配对差 < −1pp。
- 次指标：每对的输入 token 与单次耗时（compact − full），只描述不判定。
- 不做显著性声明：每格 n=1，seed 之间共用同一验证集。

## 2 固定条件

| 项 | 值 |
|---|---|
| 任务 | `released_task2`（成对文本 29 候选）、`released_task1`（类别表格 15）、`sms_spam`（单文本 14） |
| seed | 707、911 |
| 视图 | `full` 与 `compact` 各一条 |
| 契约 | `calibrated-v2`（两臂相同） |
| 调度 | `ab-v1`，fit 上限 3，调用上限 3 |
| 反馈 | `--feedback-view compact`（两臂相同） |
| 时限 | 600 / 120 秒 |
| 账本 | 唯一真实账本 + `accepted_history_v3` |

上沿：12 条路径 × 最多 3 次调用；预计 12 次左右调用、约 ¥0.12、36 fit。

## 3 顺序（小菜单先跑，29 候选的 full 放最后）

| # | 任务 | seed | 视图 |
|---:|---|---:|---|
| 1 | single_text | 707 | full |
| 2 | single_text | 707 | compact |
| 3 | single_text | 911 | compact |
| 4 | single_text | 911 | full |
| 5 | categorical | 707 | full |
| 6 | categorical | 707 | compact |
| 7 | categorical | 911 | compact |
| 8 | categorical | 911 | full |
| 9 | pair_text | 707 | compact |
| 10 | pair_text | 707 | full |
| 11 | pair_text | 911 | full |
| 12 | pair_text | 911 | compact |

把 29 候选的 `full` 放在最后，是因为它正是 A151 最初那次 417 秒断流的形态；若它再次失败，前面 8 条已提供小菜单结论，且失败会如实记成新的 unknown 并停止本批。

## 4 验收与停止

沿用既有口径（规划完成、fit 计数完整、调用=账本增量且 ≤3、无新 unknown、交付校验一致）。任一路径失败即停止整批；出现新 unknown 立即停止；不因结果换 seed。

## 5 交付物

- 计划：本文 + `research/a151_menuview/plan.json`
- 报告：`docs/batches/A151_menuview_results_20261124_zh.md`
- 运行器：复用 `scripts/run_a151_matrix.py --plan ... --output ...`

## 6 追加：分歧格重复（执行后补记）

首轮 6 对里 4 对同选、2 对改变且方向相反（`text/707` +7.89pp、`pair/707` −2.55pp）。单次无法区分"表示效应"与"模型漂移"，因此对这两个格各再跑一次**同视图**重复（不改任何参数）：

| # | 任务 | seed | 视图 | 用途 |
|---:|---|---:|---|---|
| 1 | single_text | 707 | full | 重放，看是否仍选 C0.3 |
| 2 | single_text | 707 | compact | 重放，看是否仍选 C3.0 |
| 3 | pair_text | 707 | full | 重放 |
| 4 | pair_text | 707 | compact | 重放 |

判定：若同一视图两次给出不同候选，则该格的分歧落在漂移范围内，不能归因于表示；若同视图稳定复现，则表示效应成立。追加 4 条路径，上沿 4 次调用。